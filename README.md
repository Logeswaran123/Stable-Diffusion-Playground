# Stable-Diffusion-Playground
An application that generates images or videos using Stable Diffusion models.

## Description :scroll:
** TODO **

## General Requirements :mage_man:
* Atleast 4GB of VRAM is required to generate a single 512x512 image.
* For better image generation, use detailed prompt.

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

There are three different modes of running the application, <br />
* Text to Image (txt2img)
* Image to Image (img2img)
* Inpaint (inpaint)
* Dream (dream)

<b> Mode: Text to Image </b> <br />
```python
python run.py --mode txt2img --device gpu --save
```

<b> Mode: Image to Image </b> <br />
```python
python run.py --mode img2img --device gpu --save
```

<b> Mode: Dream </b> <br />
```python
python run.py --mode dream --device gpu --save --num 100
```

Note: <br />
For each of the modes, run the command and follow the cli to provide hugging face user token, prompt and size (Height, Width) of image. <br />
Dream mode will generate --num images, and create a video. <br />
Images/Video will be saved to $PWD/images dir.

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

## References :page_facing_up:
* [Hugging face diffuser](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion) for API usage.
* [Gist](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355) by Andrej Karpathy.
* [lexica.art](https://lexica.art/) for cool prompts.

Happy Learning! 😄
