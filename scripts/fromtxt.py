import argparse
import os

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I", "--image", type=str, nargs="?", help="local image path converted from"
    )
    parser.add_argument(
        "-S",
        "--strength",
        type=float,
        nargs="?",
        default=0.8,
        help="how much to transform the reference image",
    )
    parser.add_argument(
        "-G",
        "--guidance",
        type=float,
        nargs="?",
        default=7.5,
        help="higher guidance scale encourages to generate images that are closely linked to the text prompt",
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
    parser.add_argument("--nsfw", action="store_true", help="allow NSFW image")
    return parser.parse_args()


if __name__ == "__main__":
    option = parse_option()

    generate_params = {
        "prompt": option.prompt,
        "negative_prompt": option.negative,
    }

    if option.image is None:
        pipeline_class = StableDiffusionPipeline
        image = None
        generate_params["width"] = option.width
        generate_params["height"] = option.height
    else:
        pipeline_class = StableDiffusionImg2ImgPipeline
        image = Image.open(option.image).convert("RGB")
        image.thumbnail((option.width, option.height))
        generate_params["image"] = image
        generate_params["strength"] = option.strength
        generate_params["guidance_scale"] = option.guidance

    if os.path.isfile(option.model):
        pipe = pipeline_class.from_ckpt(pretrained_model_link_or_path=option.model)
    else:
        pipe = pipeline_class.from_pretrained(
            pretrained_model_name_or_path=option.model
        )

    if option.nsfw and pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, False)

    pipe = pipe.to(torch_device="cuda")
    pipe.enable_xformers_memory_efficient_attention()

    outpath = os.path.join(
        "outputs/fromtxt",
        os.path.basename(option.model),
        "." if option.image is None else f"{os.path.basename(option.image)}→",
    )
    os.makedirs(name=outpath, exist_ok=True)
    img_count = len(os.listdir(outpath))

    while True:
        out = pipe(**generate_params)

        if (
            option.nsfw is True
            or out.nsfw_content_detected is None
            or out.nsfw_content_detected[0] == False
        ):
            out.images[0].save(
                os.path.join(
                    outpath, f"{os.path.basename(option.model)} - {img_count:06}.png"
                )
            )
            img_count += 1
