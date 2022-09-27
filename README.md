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
There are three different modes of running the application, <br />
* Text to Image (txt2img)
* Image to Image (img2img)
* Inpaint (inpaint)
* Dream (dream)

<b> Command line arguments: </b>
| Argument         | Requirement   | Default | Choices                       | Description  |
| ---------------- |:-------------:|:-------:|:-----------------------------:| :------------|
| --mode / -m      | True          | -       | "txt2img", "img2img", "dream" | Mode of application. |
| --local / -l     | False         | False   | True / False                  | If argument is provided, use local model files. Else download from hugging face. |
| --device / -d    | False         | "cpu"   | "cpu", "gpu"                  | Run on target device. |
| --num / -n       | False         | 1       | integer number                | Number of images to generate. |
| --save / -s      | False         | False   | True / False                  | If argument is provided, save generated images. |
| --limit / -limit | False         | True    | True / False                  | If argument is provided, limit memory usage. |

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

## Results :bar_chart:
** TODO **

## References :page_facing_up:
* [Hugging face diffuser](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion)
* [Gist by Andrej Karpathy](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355)

Happy Learning! 😄
