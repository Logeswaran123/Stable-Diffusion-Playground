import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import autocast
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

FPS = 24

class StableDiffusionPipe():
    """ Pipline for Stable Diffusion model applications """
    def __init__(self, use_local_model: bool = True, device: str = "cpu") -> None:
        self.use_local_model = use_local_model
        self.device = device if device == "cpu" else "cuda"

    def slerp(self, t, v0, v1, DOT_THRESHOLD=0.9995):
        """ helper function to spherically interpolate two arrays v1 v2 """

        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            input_device = v0.device
            v0 = v0.cpu().numpy()
            v1 = v1.cpu().numpy()

        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(input_device)

        return v2

    def save_images(self, images):
        save_dir = os.path.join(os.getcwd(), r'images')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, image in enumerate(images):
            image.save("images/image_" + str(i) + ".png")

    def save_video(self, images, width, height):
        out = cv2.VideoWriter("images/output.avi", # video file name
                                cv2.VideoWriter_fourcc(*'MJPG'), # fourcc format
                                FPS, # video fps
                                (width, height) # (frame width, frame height)
                            )
        for _, pil_image in enumerate(images):
            out.write(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        out.release()

    def TexttoImage(self, num_images: int = 1, save_images: bool = True, use_limited_mem: bool = True):
        """ Text to Image function """        

        path = "./stable-diffusion-v1-4" if self.use_local_model else "CompVis/stable-diffusion-v1-4"
        local_files_only = self.use_local_model
        use_auth_token = not local_files_only

        # Get access token
        access_token = False
        if use_auth_token:
            access_token = input("\nEnter Hugging face user access token: ")

        # Load the model
        print("\nLoading model...")
        pipe = StableDiffusionPipeline.from_pretrained(path, use_auth_token=access_token, 
                                    local_files_only=local_files_only,
                                    torch_dtype=torch.float16, revision='fp16')
        pipe = pipe.to(self.device)
        print("\nModel loaded successfully")

        # Get prompt
        prompt = input("\nEnter prompt: ")
        height, width = input("\nEnter height and width of image: ").split()
        height = int(height)
        width = int(width)

        # Convert height and width to multiple of 64 for model.
        height = height - height % 64
        width = width - width % 64

        # Generate images
        images = []
        if use_limited_mem:
            prompts = [prompt]
            for _ in range (1, num_images + 1):
                print("\nRunning Text to Image generation...")
                with autocast(self.device):
                    images.append(pipe(prompt=prompts, height=height, width=width).images[0])
        else:
            print("\nRunning Text to Image generation...")
            prompts = [prompt] * num_images
            images = pipe(prompt=prompts, height=height, width=width).images

        # Save images
        if save_images:
            print("Saving images...")
            self.save_images(images)

    def ImagetoImage(self, num_images: int = 1, save_images: bool = True, use_limited_mem: bool = True):
        """ Image to Image function """        

        path = "./stable-diffusion-v1-4" if self.use_local_model else "CompVis/stable-diffusion-v1-4"
        local_files_only = self.use_local_model
        use_auth_token = not local_files_only

        # Get access token
        access_token = False
        if use_auth_token:
            access_token = input("\nEnter Hugging face user access token: ")

        # Load the model
        print("\nLoading model...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(path, use_auth_token=access_token, 
                                    local_files_only=local_files_only,
                                    torch_dtype=torch.float16, revision='fp16')
        pipe = pipe.to(self.device)
        print("\nModel loaded successfully")

        # Get prompt
        image_path = input("\nEnter initial image path: ")
        prompt = input("\nEnter prompt: ")
        strength = float(input("\nEnter strength in [0, 1] range: "))
        if not 0 <= strength <= 1:
            print("{} is an invalid strength value. Enter strength in [0, 1] range.".format(strength))

        init_image = Image.open(image_path).convert("RGB")

        # Generate images
        images = []
        if use_limited_mem:
            prompts = [prompt]
            for _ in range (1, num_images + 1):
                print("\nRunning Image to Image generation...")
                with autocast(self.device):
                    images.append(pipe(prompt=prompts, init_image=init_image, strength=strength).images[0])
        else:
            print("\nRunning Image to Image generation...")
            prompts = [prompt] * num_images
            images = pipe(prompt=prompts, init_image=init_image, strength=strength).images

        # Save images
        if save_images:
            print("Saving images...")
            self.save_images(images)

    def Dream(self, num_images: int = 1, save_images: bool = True):
        """ Dream function """        

        path = "./stable-diffusion-v1-4" if self.use_local_model else "CompVis/stable-diffusion-v1-4"
        local_files_only = self.use_local_model
        use_auth_token = not local_files_only

        # Get access token
        access_token = False
        if use_auth_token:
            access_token = input("\nEnter Hugging face user access token: ")

        # Load the model
        print("\nLoading model...")
        lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        pipe = StableDiffusionPipeline.from_pretrained(path, use_auth_token=access_token, 
                                    local_files_only=local_files_only, scheduler=lms,
                                    torch_dtype=torch.float16, revision='fp16')
        pipe = pipe.to(self.device)
        print("\nModel loaded successfully")

        # Get prompt
        prompt = input("\nEnter prompt: ")
        height, width = input("\nEnter height and width of image: ").split()
        height = int(height)
        width = int(width)

        # Convert height and width to multiple of 64 for model.
        height = height - height % 64
        width = width - width % 64

        source_latent = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=self.device)
        target_latent = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=self.device)

        images = []
        print("\nDreaming...")
        for _, t in enumerate(np.linspace(0, 1, num_images)):
            init_latent = self.slerp(float(t), source_latent, target_latent)

            with autocast("cuda"):
                image = pipe(prompt, latents=init_latent).images[0]
                if not image.convert("L").getextrema() == (0, 0): # check for black image
                    images.append(image)

         # Save images and video
        if save_images:
            print("Saving images...")
            self.save_images(images)
            print("Saving video...")
            self.save_video(images, width, height)