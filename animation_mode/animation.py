import os
import pathlib
import sys
from types import SimpleNamespace
import cv2
import pandas as pd
import numpy as np
from pytorch_lightning import seed_everything
import torch
from torch import autocast
from torchvision import transforms
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image

sys.path.extend([
    './animation_mode/src/taming-transformers',
    './animation_mode/src/clip',
    './animation_mode/stable-diffusion/',
    './animation_mode/k-diffusion',
    './animation_mode/AdaBins',
    './animation_mode/MiDaS',
    './animation_mode',
])

import config
from .utility.utils import *
from helpers import DepthModel


def generate(pipe,
            prompt,
            height,
            width,
            strength,
            seed,
            use_init,
            init_image,
            return_sample=False):
    """ Image generator """
    seed_everything(seed)
    device = "cuda"
    convert_tensor = transforms.ToTensor()

    results = []
    if use_init:
        with autocast(device):
            with torch.no_grad():
                image = pipe(prompt=prompt,
                            init_image=init_image,
                            strength=strength,
                            guidance_scale=config.guidance_scale,
                            num_inference_steps=config.num_inference_steps).images[0]
                torch.cuda.empty_cache()
    else:
        with autocast(device):
            with torch.no_grad():
                image = pipe(prompt=prompt, height=height, width=width).images[0]
                torch.cuda.empty_cache()

    if return_sample:
        samples = convert_tensor(image)
        results.append(samples)
    results.append(image)

    return results


def render_input_video(pipe_txt2img, pipe_img2img):
    """ Function for animate video """
    # create a folder for the video input frames to live in
    video_in_frame_path = os.path.join(os.getcwd(), 'inputframes')
    os.makedirs(video_in_frame_path, exist_ok=True)

    # save the video frames from input video
    print(f"Exporting Video Frames from (1 every {config.extract_nth_frame}) \
            frames to {video_in_frame_path}...")
    try:
        for f in pathlib.Path(video_in_frame_path).glob('*.png'):
            f.unlink()
    except:
        pass
    cap = cv2.VideoCapture(config.video_init_path)
    success, image = cap.read()
    count = 0
    while success:
        file_name = "inputframes/frame_" + str(count) + ".png"
        cv2.imwrite(file_name, image)
        success,image = cap.read()
        count = count + 1 + (config.extract_nth_frame - 1)
        if config.max_frames is not None and count > config.max_frames:
            break
    cap.release()

    # determine max frames from length of input frames
    num_frames = len([f for f in pathlib.Path(video_in_frame_path).glob('*.png')])

    print(f"Loading {num_frames} input frames from {video_in_frame_path} \
                and saving video frames to {video_in_frame_path}")
    render_animation(pipe_txt2img, pipe_img2img)


def render_animation(pipe_txt2img, pipe_img2img):
    """ Function for animate 2D, animate 3D """
    device = "cuda"
    W, H = (config.width, config.height)
    depth_model = None
    models_path = "./models"
    init_image = None
    video_width, video_height = None, None

    angle_series = get_inbetweens(parse_key_frames(config.angle))
    zoom_series = get_inbetweens(parse_key_frames(config.zoom))
    translation_x_series = get_inbetweens(parse_key_frames(config.translation_x))
    translation_y_series = get_inbetweens(parse_key_frames(config.translation_y))
    translation_z_series = get_inbetweens(parse_key_frames(config.translation_z))
    rotation_3d_x_series = get_inbetweens(parse_key_frames(config.rotation_3d_x))
    rotation_3d_y_series = get_inbetweens(parse_key_frames(config.rotation_3d_y))
    rotation_3d_z_series = get_inbetweens(parse_key_frames(config.rotation_3d_z))
    strength_schedule_series = get_inbetweens(parse_key_frames(config.strength_schedule))
    midas_weight_dict = {"midas_weight":config.midas_weight}
    anim_args = SimpleNamespace(**midas_weight_dict)

    start_frame = 0
    outdir = os.path.join(os.getcwd(), r'images')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(f"\nSaving animation frames to {outdir}")

    # check for video inits
    using_vid_init = config.animation_mode == 'Video Input'
    use_init = using_vid_init

    max_frames = config.max_frames
    if using_vid_init:
        max_frames = len([f for f in pathlib.Path(os.path.join(os.getcwd(), 'inputframes')).glob('*.png')])
        cap = cv2.VideoCapture(config.video_init_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    # expand prompts out to per-frame
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in config.animation_prompts.items():
        prompt_series[int(i)] = prompt
    prompt_series = prompt_series.ffill().bfill()

    # load depth model for 3D
    predict_depths = (config.animation_mode == '3D' and config.use_depth_warping) or config.save_depth_maps
    if predict_depths:
        depth_model = DepthModel("cpu")
        depth_model.load_midas(models_path)
        if config.midas_weight < 1.0:
            depth_model.load_adabins()
    else:
        depth_model = None
        config.save_depth_maps = False

    turbo_steps = 1 if using_vid_init else int(config.diffusion_cadence)
    turbo_prev_image, turbo_prev_frame_idx = None, 0
    turbo_next_image, turbo_next_frame_idx = None, 0

    # resume animation
    prev_sample = None
    color_match_sample = None
    frame_idx = start_frame

    seed = config.seed
    while frame_idx < max_frames:
        print(f"\nRendering animation frame {frame_idx} of {max_frames}")
        strength = strength_schedule_series[frame_idx]
        strength = max(0.0, min(1.0, strength))
        depth = None

        # emit in-between frames
        if turbo_steps > 1:
            tween_frame_start_idx = max(0, frame_idx-turbo_steps)
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                print(f"creating in between frame {tween_frame_idx} tween:{tween:0.2f}")

                advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                advance_next = tween_frame_idx > turbo_next_frame_idx

                if depth_model is not None:
                    assert turbo_next_image is not None
                    depth_model.midas_model = depth_model.midas_model.to(device)
                    depth_model.device = device
                    with torch.no_grad():
                        depth = depth_model.predict(turbo_next_image, anim_args).cpu()
                        torch.cuda.empty_cache()
                    depth_model.midas_model = depth_model.midas_model.to("cpu")
                    depth_model.device = "cpu"

                if config.animation_mode == '2D':
                    if advance_prev:
                        turbo_prev_image = anim_frame_warp_2d(turbo_prev_image, W, H, angle_series, zoom_series, \
                                            translation_x_series, translation_y_series, tween_frame_idx)
                    if advance_next:
                        turbo_next_image = anim_frame_warp_2d(turbo_next_image, W, H, angle_series, zoom_series, \
                                            translation_x_series, translation_y_series, tween_frame_idx)
                else: # '3D'
                    if advance_prev:
                        turbo_prev_image = anim_frame_warp_3d(turbo_prev_image,
                                                                depth,
                                                                translation_x_series,
                                                                translation_y_series,
                                                                translation_z_series,
                                                                rotation_3d_x_series,
                                                                rotation_3d_y_series,
                                                                rotation_3d_z_series,
                                                                config.near_plane,
                                                                config.far_plane,
                                                                config.fov,
                                                                config.sampling_mode,
                                                                config.padding_mode,
                                                                tween_frame_idx)
                    if advance_next:
                        turbo_next_image = anim_frame_warp_3d(turbo_next_image,
                                                                depth,
                                                                translation_x_series,
                                                                translation_y_series,
                                                                translation_z_series,
                                                                rotation_3d_x_series,
                                                                rotation_3d_y_series,
                                                                rotation_3d_z_series,
                                                                config.near_plane,
                                                                config.far_plane,
                                                                config.fov,
                                                                config.sampling_mode,
                                                                config.padding_mode,
                                                                tween_frame_idx)

                turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                if turbo_prev_image is not None and tween < 1.0:
                    img = turbo_prev_image*(1.0-tween) + turbo_next_image*tween
                else:
                    img = turbo_next_image

                # apply color matching
                if config.color_coherence != 'None':
                    if color_match_sample is not None:
                        img = maintain_colors(img, color_match_sample, config.color_coherence)

                # smoothen image
                if config.smooth != 'None':
                    img = smoothen_image(Image.fromarray(img.astype(np.uint8)), config.smooth)
                    img = np.array(img)

                init_image = Image.fromarray(img.astype(np.uint8))
                filename = f"frame_{tween_frame_idx}.png"
                cv2.imwrite(os.path.join(outdir, filename), cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                if config.save_depth_maps:
                    depth_model.save(os.path.join(outdir, f"depth_{tween_frame_idx:05}.png"), depth)
            if turbo_next_image is not None:
                prev_sample = turbo_next_image

        # apply transforms to previous frame
        if prev_sample is not None:
            if config.animation_mode == '2D':
                prev_img = anim_frame_warp_2d(prev_sample, W, H, angle_series, zoom_series, \
                                translation_x_series, translation_y_series, frame_idx)
            else: # '3D'
                prev_img_cv2 = prev_sample
                depth_model.midas_model = depth_model.midas_model.to(device)
                depth_model.device = device
                with torch.no_grad():
                    depth = depth_model.predict(prev_img_cv2, anim_args).cpu() if depth_model else None
                    torch.cuda.empty_cache()
                depth_model.midas_model = depth_model.midas_model.to("cpu")
                depth_model.device = "cpu"
                prev_img = anim_frame_warp_3d(prev_img_cv2,
                                                depth,
                                                translation_x_series,
                                                translation_y_series,
                                                translation_z_series,
                                                rotation_3d_x_series,
                                                rotation_3d_y_series,
                                                rotation_3d_z_series,
                                                config.near_plane,
                                                config.far_plane,
                                                config.fov,
                                                config.sampling_mode,
                                                config.padding_mode,
                                                frame_idx)

            if config.color_coherence != 'None':
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()

            use_init = True

        # grab prompt for current frame
        prompt = prompt_series[frame_idx]
        print(f"\nSeed: {seed}\nPrompt: {prompt} \n")

        # grab init image for current frame
        if using_vid_init:
            init_frame = "./inputframes/" + "frame_" + str(frame_idx) + ".png"
            print(f"\nUsing video init frame {init_frame}")
            try:
                init_image = load_img(init_frame, (config.width, config.height))
            except:
                frame_idx += 1
                continue

        # sample the diffusion model
        torch.cuda.empty_cache()
        if use_init:
            pipe_img2img = pipe_img2img.to(device)
            sample, image = generate(pipe_img2img, prompt, H, W, \
                        strength, seed, use_init, init_image, return_sample=True)
            pipe_img2img.to("cpu")
        else:
            pipe_txt2img = pipe_txt2img.to(device)
            sample, image = generate(pipe_txt2img, prompt, H, W, \
                        strength, seed, use_init, init_image, return_sample=True)
            pipe_txt2img.to("cpu")

        torch.cuda.empty_cache()
        if not using_vid_init:
            prev_sample = sample

        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            turbo_next_image, turbo_next_frame_idx = sample_to_cv2(sample, type=np.float32), frame_idx
            frame_idx += turbo_steps
        else:
            filename = f"frame_{frame_idx}.png"
            if using_vid_init and config.video_same_size:
                image = image.resize((video_width, video_height), resample=Image.LANCZOS)
            if not image.convert("L").getextrema() == (0, 0): # check for black image
                image.save(os.path.join(outdir, filename))
            if config.save_depth_maps:
                if depth is None:
                    depth = depth_model.predict(sample_to_cv2(sample), anim_args)
                depth_model.save(os.path.join(outdir, f"depth_{frame_idx:05}.png"), depth)
            frame_idx += 1

        seed = next_seed(seed, config.seed_behavior)


def animate(use_local_model, save):
    """ Top level function for animate 2D, animate 3D, and animate video """
    path = "./stable-diffusion-v1-4" if use_local_model else "CompVis/stable-diffusion-v1-4"
    local_files_only = use_local_model
    use_auth_token = not local_files_only

    # Get access token
    access_token = False
    if use_auth_token:
        access_token = input("\nEnter Hugging face user access token: ")

    print(f"\nMax cuda memory reserved before running the app: \
            {torch.cuda.max_memory_reserved(torch.device('cuda'))} bytes\n")
    print("\nLoading Diffusion model...")
    pipe_txt2img = StableDiffusionPipeline.from_pretrained(path,
                                                            use_auth_token=access_token,
                                                            local_files_only=local_files_only,
                                                            torch_dtype=torch.float16,
                                                            revision='fp16')
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(path,
                                                            use_auth_token=access_token,
                                                            local_files_only=local_files_only,
                                                            torch_dtype=torch.float16,
                                                            revision='fp16')
    print("\nModel loaded successfully")

    if config.animation_mode == '2D' or config.animation_mode == '3D':
        render_animation(pipe_txt2img, pipe_img2img)
    elif config.animation_mode == 'Video Input':
        render_input_video(pipe_txt2img, pipe_img2img)
    else:
        print(f"\nInvalid animation mode {config.animation_mode}. \
                Supported modes = [2D, 3D, Video Input].")

    if save:
        save_video(config.width, config.height)
