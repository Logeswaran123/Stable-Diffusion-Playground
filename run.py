import argparse

from utils.pipeline import StableDiffusionPipe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', "--num", required=False, default=1,
                                        help="Number of images to generate", type=int)
    parser.add_argument('-l', "--local", required=False, action='store_true', default=False,
                                        help="local model or download from huggingface")
    parser.add_argument('-s', "--save", required=False, action='store_true', default=False,
                                        help="Save generated image")
    parser.add_argument('-d', "--device", required=False, default="gpu", choices=["cpu", "gpu"],
                                        help="cpu or gpu device", type=str)
    parser.add_argument('-m', "--mode", required=True, default="txt2img",
                                        choices=["txt2img", "img2img", "inpaint", "dream", "animate"],
                                        help="Select the mode", type=str)
    parser.add_argument('-limit', "--limit", required=False, action='store_true', default=True,
                                        help="Limited memory usage")

    args = parser.parse_args()
    num_images = args.num
    is_local_model = args.local
    save = args.save
    device = args.device
    mode = args.mode
    limit = args.limit

    pipe = StableDiffusionPipe(is_local_model, device)

    if mode.lower() == "txt2img":
        pipe.TexttoImage(num_images, save, limit)
    elif mode.lower() == "img2img":
        pipe.ImagetoImage(num_images, save, limit)
    elif mode.lower() == "inpaint":
        pipe.Inpaint(num_images, save, limit)
    elif mode.lower() == "dream":
        pipe.Dream(num_images, save)
    elif mode.lower() == "animate":
        pipe.Animate(save)
    else:
        print(f"\n {mode} is an invalide mode. Select a valid mode.")
