import argparse
import os

import cv2
import diffusers.utils
import numpy
import PIL.Image


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--image", type=str, help="source image")
    parser.add_argument("-L", "--low", type=int, default=100, help="low threshold")
    parser.add_argument("-H", "--high", type=int, default=200, help="high threshold")
    parser.add_argument(
        "-S", "--scale", type=float, default=1, help="get resized canny"
    )

    return parser.parse_args()


if __name__ == "__main__":
    option = parse_option()

    image = diffusers.utils.load_image(option.image)
    if option.scale is None:
        scale_modifier = ""
    else:
        image = image.resize(
            (round(image.width * option.scale), round(image.height * option.scale))
        )
        scale_modifier = f"x{option.scale}"
    image = numpy.array(image)
    image = cv2.Canny(image, option.low, option.high)
    image = image[:, :, None]
    image = numpy.concatenate([image, image, image], axis=2)
    image = PIL.Image.fromarray(image)

    outpath = os.path.join("outputs/canny", os.path.basename(option.image))
    os.makedirs(outpath, exist_ok=True)
    outpath = os.path.join(outpath, f"{option.low}-{option.high}{scale_modifier}.png")
    image.save(outpath)
    print(os.path.abspath(outpath))
