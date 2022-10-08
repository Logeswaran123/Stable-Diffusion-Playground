# ⛹️‍♀️:basketball: Stable-Diffusion-Playground :soccer:⛹️
An application that generates images or videos using Stable Diffusion models.

## Description :scroll:
What is the term "diffusion"? <br />

From Wikipedia, "Diffusion is the net movement of anything (for example, atoms, ions, molecules, energy) generally from a region of higher concentration to a region of lower concentration." <br />

Similar to the definition, diffusion models essentially apply noise to an image sequentially across multiple steps in forward pass. This essentially diffuses the pixels. In the backward pass, the noisy image is denoised across same steps. Since it is a sequential process, there is less chance of mode collapse (a problem with GANs) to occur. <br />

Most diffusion models use UNet architecture to preserve the dimensionality of the image. Usually, diffusion models apply diffusion in pixel space, but stable diffusion models apply diffusion in latent space. Hence, the term "Latent diffusion model (LDM)". The conversion between pixel space to latent space is done using Encoder and Decoder. This method is memory efficient compared to previous methods, and also produces highly detailed image. <br />

Read through the [paper](https://arxiv.org/abs/2112.10752) for more details. Big-ups to the researchers/creators for the work and for open-sourcing it. <br />

## General Requirements :mage_man:
* Atleast 6GB of VRAM is required to generate a single 512x512 image.
* For better image generation, use descriptive and detailed prompt.

## Code Requirements :mage_woman:
```python
pip install -r requirements.txt
```

## How to run :running_man:

<b> Command line arguments: </b>
| Argument         | Requirement   | Default | Choices                       | Description  |
| ---------------- |:-------------:|:-------:|:-----------------------------:| :------------|
| --mode / -m      | True          | -       | "txt2img", "img2img", "dream" | Mode of application. |
| --local / -l     | False         | False   | True / False                  | If argument is provided, use local model files. Else download from hugging face. |
| --device / -d    | False         | "cpu"   | "cpu", "gpu"                  | Run on target device. |
| --num / -n       | False         | 1       | integer number                | Number of images to generate. |
| --save / -s      | False         | False   | True / False                  | If argument is provided, save generated images. |
| --limit / -limit | False         | True    | True / False                  | If argument is provided, limit memory usage. |

There are five different modes of running the application, <br />
* Text to Image (txt2img)
* Image to Image (img2img)
* Inpaint (inpaint)
* Dream (dream)
* Animate (animate)

<b> Mode: Text to Image </b> <br />
```python
python run.py --mode txt2img --device gpu --save
```

<b> Mode: Image to Image </b> <br />
```python
python run.py --mode img2img --device gpu --save
```

<b> Mode: Inpaint </b> <br />
```python
python run.py --mode inpaint --device gpu --save
```

<b> Mode: Dream </b> <br />
```python
python run.py --mode dream --device gpu --save --num <number of frames>
```

<b> Mode: Dream </b> <br />
```python
python run.py --mode animate --device gpu --save
```
Note: <br />
* For each of the modes, run the command and follow the cli to provide hugging face user token, prompt and size (Height, Width) of image. <br />
* Images/Video will be saved to $PWD/images dir.
* Single 512x512 image generation takes ~12 seconds on NVIDIA GeForce RTX 3060 with 6GB VRAM.
* Dream mode will generate --num image frames based on input prompt, and create a video. <br />
* Image to Image mode will generate new image from initial image and input prompt. Inpaint mode will generate the masked part of image from initial image, mask image and input prompt. The strength input in CLI will indicate the amount of change from initial image. In range [0, 1]; with 0 indicating no change and 1 indicating complete change from original image.

<b> Hugging face Access Token: </b><br />
* Create an account in [huggingface.co](https://huggingface.co/). Go to Settings -> Access Tokens. Creata a access token with read permission. <br />

## Results :bar_chart:
<p align="center"> :star: <b> Text to Image </b> :star: </p>

```python
python run.py --mode txt2img --device gpu --num 1 --limit --save
```

|||
|:-------------------------:|:-------------------------:|
|![](https://github.com/Logeswaran123/Stable-Diffusion-Playground/blob/main/images/49.png)|![](https://github.com/Logeswaran123/Stable-Diffusion-Playground/blob/main/images/3.png)|
|![](https://github.com/Logeswaran123/Stable-Diffusion-Playground/blob/main/images/78.png)|![](https://github.com/Logeswaran123/Stable-Diffusion-Playground/blob/main/images/93.png)|
---
<p align="center"> :star: <b> Image to Image </b> :star: </p>

```python
python run.py --mode img2img --device gpu --num 1 --limit --save
```
CLI inputs: <br />
```python
Enter Hugging face user access token: <user access token>

Loading model...

Model loaded successfully

Enter initial image path: flower.png

Enter prompt: beautiful red flower, vibrant, realistic, smooth, bokeh, highly detailed, 4k

Enter strength in [0, 1] range: 0.8

Running Image to Image generation...
```
|||
|:-------------------------:|:-------------------------:|
|![](https://github.com/Logeswaran123/Stable-Diffusion-Playground/blob/main/images/flower.png)|![](https://github.com/Logeswaran123/Stable-Diffusion-Playground/blob/main/images/img2img_1.png)|

---
<p align="center"> :star: <b> Inpaint </b> :star: </p>

```python
python run.py --mode inpaint --device gpu --num 1 --limit --save
```
CLI inputs: <br />
```python
Enter Hugging face user access token: <user access token>

Loading model...

Model loaded successfully

Enter initial image path: rose.png

Enter mask image path: mask_rose.png

Enter prompt: beautiful blue butterfly on a rose, glossy, detailed, sharp, 4k

Enter strength in [0, 1] range: 0.8

Running Inpaint...
```

| Initial image | Mask | Inpainted image |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![](https://github.com/Logeswaran123/Stable-Diffusion-Playground/blob/main/images/rose.png)|![](https://github.com/Logeswaran123/Stable-Diffusion-Playground/blob/main/images/mask_rose.png)|![](https://github.com/Logeswaran123/Stable-Diffusion-Playground/blob/main/images/inpaint_rose.png)|

---
<p align="center"> :star: <b> Dream </b> :star: </p>

```python
python run.py --mode dream --device gpu --num 780 --limit --save
```
CLI inputs: <br />
```python
Enter Hugging face user access token: <user access token>

Loading model...

Model loaded successfully

Enter prompt: highly detailed bowl of lucrative ramen, stephen bliss, unreal engine, fantasy art by greg rutkowski, loish, rhads and lois van baarle, ilya kuvshinov, rossdraws, tom bagshaw, alphonse mucha, global illumination, detailed and intricate environment

Enter height and width of image: 512 512

Dreaming...
```

https://user-images.githubusercontent.com/36563521/192521369-32673804-009f-44c6-918c-a7746cc94dba.mp4

---
Few of the Text to Image generations compiled into a video, <br />
** TODO **

## References :page_facing_up:
* [stability.ai](https://stability.ai/blog/stable-diffusion-public-release) blog.
* LDM [paper](https://arxiv.org/abs/2112.10752).
* LDM [repo](https://github.com/CompVis/latent-diffusion).
* [Hugging face diffuser](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion) for API usage.
* [Gist](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355) by Andrej Karpathy.
* [lexica.art](https://lexica.art/) for cool prompts.

Happy Learning! 😄
