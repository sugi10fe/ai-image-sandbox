import argparse
import os
from pathlib import Path

from diffusers import StableDiffusionPipeline


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I", "--image", type=str, nargs="?", help="image converted from"
    )
    parser.add_argument("-P", "--prompt", type=str, help="positive prompt")
    parser.add_argument("-N", "--negative", type=str, nargs="?", help="negative prompt")
    parser.add_argument(
        "-W", "--width", type=int, nargs="?", default=320, help="width of output image"
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        nargs="?",
        default=320,
        help="height of output image",
    )
    parser.add_argument(
        "-M",
        "--model",
        type=str,
        help="model id in huggingface / path to checkpoint directory / path to checkpoint file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    option = parse_option()

    model_path = Path(option.model)
    if model_path.is_file():
        pipe = StableDiffusionPipeline.from_ckpt(
            pretrained_model_link_or_path=option.model
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=option.model
        )

    pipe = pipe.to(torch_device="cuda")
    pipe.enable_xformers_memory_efficient_attention()

    outpath = os.path.join("outputs/fromtxt", os.path.basename(option.model))
    os.makedirs(name=outpath, exist_ok=True)
    img_count = len(os.listdir(outpath))

    while True:
        out = pipe(
            prompt=option.prompt,
            negative_prompt=option.negative,
            width=option.width,
            height=option.height,
        )

        if out.nsfw_content_detected[0] == False:
            out.images[0].save(os.path.join(outpath, f"{img_count:06}.png"))
            img_count += 1
