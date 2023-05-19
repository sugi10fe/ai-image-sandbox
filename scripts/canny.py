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

    return parser.parse_args()


if __name__ == "__main__":
    option = parse_option()

    image = diffusers.utils.load_image(option.image)
    image = numpy.array(image)
    image = cv2.Canny(image, option.low, option.high)
    image = image[:, :, None]
    image = numpy.concatenate([image, image, image], axis=2)
    image = PIL.Image.fromarray(image)

    outpath = os.path.join("outputs/canny", os.path.basename(option.image))
    os.makedirs(outpath, exist_ok=True)
    image.save(os.path.join(outpath, f"{option.low}-{option.high}.png"))
