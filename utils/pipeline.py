import imp
from operator import imod
import numpy as np
from PIL import Image
import torch
from torch import autocast
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

from .utility import save_images, save_video, slerp


class StableDiffusionPipe():
    """ Pipline for Stable Diffusion model applications """
    def __init__(self, use_local_model: bool = True, device: str = "cpu") -> None:
        self.use_local_model = use_local_model
        self.device = device if device == "cpu" else "cuda"

    def TexttoImage(self, num_images: int = 1, save: bool = True, use_limited_mem: bool = True):
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
        if save:
            print("Saving images...")
            save_images(images)

    def ImagetoImage(self, num_images: int = 1, save: bool = True, use_limited_mem: bool = True):
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
            raise ValueError("{} is an invalid strength value. Enter strength in [0, 1] range.".format(strength))

        init_image = Image.open(image_path).convert("RGB")
        width, height = init_image.size

        # Convert height and width to multiple of 64 for model.
        width = width - width % 64
        height = height - height % 64
        init_image = init_image.resize((width, height))

        # Generate images
        images = []
        if use_limited_mem:
            prompts = [prompt]
            for _ in range (1, num_images + 1):
                print("\nRunning Image to Image generation...")
                with autocast(self.device):
                    images.append(pipe(prompt=prompts,
                                        init_image=init_image,
                                        strength=strength).images[0])
        else:
            print("\nRunning Image to Image generation...")
            prompts = [prompt] * num_images
            images = pipe(prompt=prompts,
                            init_image=init_image,
                            strength=strength).images

        # Save images
        if save:
            print("Saving images...")
            save_images(images)

    def Inpaint(self, num_images: int = 1, save: bool = True, use_limited_mem: bool = True):
        """ Inpaint function """        

        path = "./stable-diffusion-v1-4" if self.use_local_model else "CompVis/stable-diffusion-v1-4"
        local_files_only = self.use_local_model
        use_auth_token = not local_files_only

        # Get access token
        access_token = False
        if use_auth_token:
            access_token = input("\nEnter Hugging face user access token: ")

        # Load the model
        print("\nLoading model...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(path, use_auth_token=access_token, 
                                    local_files_only=local_files_only,
                                    torch_dtype=torch.float16, revision='fp16')
        pipe = pipe.to(self.device)
        print("\nModel loaded successfully")

        # Get prompt
        image_path = input("\nEnter initial image path: ")
        mask_path = input("\nEnter mask image path: ")
        prompt = input("\nEnter prompt: ")
        strength = float(input("\nEnter strength in [0, 1] range: "))
        if not 0 <= strength <= 1:
            raise ValueError("{} is an invalid strength value. Enter strength in [0, 1] range.".format(strength))

        init_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")
        image_width, image_height = init_image.size
        mask_width, mask_height = mask_image.size

        if (not image_width == mask_width) or (not image_height == mask_height):
            raise ValueError("Init image size must match mask image size.")

        # Convert height and width to multiple of 64 for model.
        image_width = image_width - image_width % 64
        image_height = image_height - image_height % 64
        init_image = init_image.resize((image_width, image_height))
        mask_image = mask_image.resize((image_width, image_height))

        # Generate images
        images = []
        if use_limited_mem:
            prompts = [prompt]
            for _ in range (1, num_images + 1):
                print("\nRunning Inpaint...")
                with autocast(self.device):
                    images.append(pipe(prompt=prompts,
                                        init_image=init_image,
                                        mask_image=mask_image,
                                        strength=strength).images[0])
        else:
            print("\nRunning Inpaint...")
            prompts = [prompt] * num_images
            images = pipe(prompt=prompts,
                            init_image=init_image,
                            mask_image=mask_image,
                            strength=strength).images

        # Save images
        if save:
            print("Saving images...")
            save_images(images)

    def Dream(self, num_images: int = 1, save: bool = True):
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
            init_latent = slerp(float(t), source_latent, target_latent)

            with autocast("cuda"):
                image = pipe(prompt, latents=init_latent).images[0]
                if not image.convert("L").getextrema() == (0, 0): # check for black image
                    images.append(image)

         # Save images and video
        if save:
            print("Saving images...")
            save_images(images)
            print("Saving video...")
            save_video(images, width, height)