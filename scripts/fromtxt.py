import argparse
import os

import diffusers.utils
import torch
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)


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
    parser.add_argument(
        "--cnet", action="extend", type=str, nargs="*", help="ControlNet model id"
    )
    parser.add_argument(
        "--cnimage",
        action="extend",
        type=str,
        nargs="*",
        help="path to ControlNet image",
    )
    parser.add_argument(
        "--cnscale",
        action="extend",
        type=float,
        nargs="*",
        help="conditioning scale of ControlNet",
    )

    option = parser.parse_args()
    if option.cnet is not None and (
        option.cnimage is None
        or option.cnscale is None
        or len(option.cnet) != len(option.cnimage)
        or len(option.cnet) != len(option.cnscale)
    ):
        assert False

    return option


if __name__ == "__main__":
    option = parse_option()

    generate_params = {
        "prompt": option.prompt,
        "negative_prompt": option.negative,
    }

    if option.image is None:
        pipeline_class = StableDiffusionPipeline
        generate_params["width"] = option.width
        generate_params["height"] = option.height
    else:
        pipeline_class = StableDiffusionImg2ImgPipeline
        image = diffusers.utils.load_image(option.image)
        image.thumbnail((option.width, option.height))
        generate_params["image"] = image
        generate_params["strength"] = option.strength
        generate_params["guidance_scale"] = option.guidance

    if option.cnet is None:
        controlnet = None
    else:
        controlnet = [
            ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            for model_id in option.cnet
        ]
        generate_params["controlnet_conditioning_scale"] = option.cnscale
        controlnet_conditioning_image = [
            diffusers.utils.load_image(image) for image in option.cnimage
        ]

        if option.image is None:
            generate_params["image"] = controlnet_conditioning_image
        else:
            generate_params[
                "controlnet_conditioning_image"
            ] = controlnet_conditioning_image
            generate_params["width"] = option.width
            generate_params["height"] = option.height

    if os.path.isfile(option.model):
        pipe = pipeline_class.from_ckpt(pretrained_model_link_or_path=option.model)
    else:
        pipe = pipeline_class.from_pretrained(
            pretrained_model_name_or_path=option.model
        )

    if controlnet:
        if option.image is None:
            pipe = StableDiffusionControlNetPipeline(
                **pipe.components, controlnet=controlnet
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
                custom_pipeline="stable_diffusion_controlnet_img2img",
                **pipe.components,
                controlnet=controlnet,
            )

    if option.nsfw and pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, False)

    pipe = pipe.to(torch_device="cuda", torch_dtype=torch.float16)
    pipe.enable_xformers_memory_efficient_attention()

    outpath = os.path.join(
        "outputs/fromtxt",
        os.path.basename(option.model),
        "." if option.image is None else f"{os.path.basename(option.image)}→",
    )
    os.makedirs(name=outpath, exist_ok=True)
    img_count = len(os.listdir(outpath))

    while True:
        with torch.inference_mode():
            out = pipe(**generate_params)

        if (
            option.nsfw is True
            or out.nsfw_content_detected is None
            or not out.nsfw_content_detected[0]
        ):
            out.images[0].save(
                os.path.join(
                    outpath, f"{os.path.basename(option.model)} - {img_count:06}.png"
                )
            )
            img_count += 1
