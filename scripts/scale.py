import argparse
import os

import diffusers.utils
import PIL.Image


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--image", type=str, help="source image")
    parser.add_argument("-W", "--width", type=int, help="width scale to")
    parser.add_argument("-H", "--height", type=int, help="height scale to")

    return parser.parse_args()


if __name__ == "__main__":
    option = parse_option()

    image = diffusers.utils.load_image(option.image).resize(
        (option.width, option.height), resample=PIL.Image.LANCZOS
    )

    outpath = os.path.join("outputs/scale", os.path.basename(option.image))
    os.makedirs(outpath, exist_ok=True)
    outpath = os.path.join(outpath, f"{option.width}x{option.height}.png")
    image.save(outpath)
    print(os.path.abspath(outpath))
