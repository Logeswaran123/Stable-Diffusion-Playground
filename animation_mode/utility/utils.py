import math
import os
import random
import requests

import sys
import cv2
from einops import rearrange
import numpy as np
import pandas as pd
from skimage.exposure import match_histograms
import torch
from PIL import Image, ImageFilter

sys.path.extend([
    './animation_mode/pytorch3d-lite',
])

from ..config import *
import py3d_tools as p3d


def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def anim_frame_warp_2d(prev_img_cv2,
                        W,
                        H,
                        angle_series,
                        zoom_series,
                        translation_x_series,
                        translation_y_series,
                        frame_idx):
    angle = angle_series[frame_idx]
    zoom = zoom_series[frame_idx]
    translation_x = translation_x_series[frame_idx]
    translation_y = translation_y_series[frame_idx]

    center = (W // 2, H // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    xform = np.matmul(rot_mat, trans_mat)

    return cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP if border == 'wrap' else cv2.BORDER_REPLICATE
    )

def anim_frame_warp_3d(prev_img_cv2,
                        depth,
                        translation_x_series,
                        translation_y_series,
                        translation_z_series,
                        rotation_3d_x_series,
                        rotation_3d_y_series,
                        rotation_3d_z_series,
                        near_plane,
                        far_plane,
                        fov,
                        sampling_mode,
                        padding_mode,
                        frame_idx):
    device = "cuda"
    TRANSLATION_SCALE = 1.0/200.0 # matches Disco
    translate_xyz = [
        -translation_x_series[frame_idx] * TRANSLATION_SCALE,
        translation_y_series[frame_idx] * TRANSLATION_SCALE,
        -translation_z_series[frame_idx] * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(rotation_3d_x_series[frame_idx]),
        math.radians(rotation_3d_y_series[frame_idx]),
        math.radians(rotation_3d_z_series[frame_idx])
    ]
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    result = transform_image_3d(prev_img_cv2, depth, rot_mat, translate_xyz, \
                    near_plane, far_plane, fov, sampling_mode, padding_mode)
    torch.cuda.empty_cache()
    return result

def get_inbetweens(key_frames, integer=False, interp_method='Linear'):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def load_img(path, shape):
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)

    image = image.resize(shape, resample=Image.LANCZOS)

    return image

def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else: # 'Match Frame 0 LAB'
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

def next_seed(seed, seed_behavior):
    if seed_behavior == 'iter':
        seed += 1
    elif seed_behavior == 'fixed':
        pass # always keep seed the same
    else:
        seed = random.randint(0, 2**32)
    return seed

def parse_key_frames(string, prompt_parser=None):
    import re
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)

def save_video(width, height):
    """ Helper function to create and save video """
    frames_dir = os.path.join(os.getcwd(), r'images')
    if not os.path.exists(frames_dir):
        print(f"\nNo generated {frames_dir} dir found.")
        return

    save_dir = os.path.join(os.getcwd(), r'out_video')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\nCreating video from generated frames...")
    out = cv2.VideoWriter("out_video/output.avi", # video file name
                            cv2.VideoWriter_fourcc(*'MJPG'), # fourcc format
                            FPS, # video fps
                            (width, height) # (frame width, frame height)
                        )
    for count in range(0, max_frames):
        filename = "frame_" + str(count) + ".png"
        try:
            out.write(cv2.imread(os.path.join(frames_dir, filename)))
        except:
            pass
    out.release()
    print(f"\nVideo saved in {os.path.join(save_dir, 'out_video.avi')}")

def smoothen_image(image, mode):
    if mode == 'Smooth':
        return image.filter(ImageFilter.SMOOTH)
    else: # 'SMOOTH_MORE'
        return image.filter(ImageFilter.SMOOTH_MORE)

def transform_image_3d(prev_img_cv2,
                        depth_tensor,
                        rot_mat,
                        translate,
                        near_plane,
                        far_plane,
                        fov,
                        sampling_mode,
                        padding_mode):
    # adapted and optimized version of transform_image_3d
    # from Disco Diffusion https://github.com/alembics/disco-diffusion
    device = "cuda"
    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    aspect_ratio = float(w) / float(h)
    near, far, fov_deg = near_plane, far_plane, fov
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, \
                        R=rot_mat, T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y, x = torch.meshgrid(torch.linspace(-1., 1. , h, dtype=torch.float32, device=device), \
                 torch.linspace(-1., 1., w, dtype=torch.float32, device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.], [0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1, 1, h, w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1/512 - 0.0001).unsqueeze(0),
        offset_coords_2d,
        mode=sampling_mode,
        padding_mode=padding_mode,
        align_corners=False
    )

    # convert back to cv2 style numpy array
    result = rearrange(
        new_image.squeeze().clamp(0,255),
        'c h w -> h w c'
    ).cpu().numpy().astype(prev_img_cv2.dtype)
    return result
