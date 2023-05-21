import argparse
import io
import os

import diffusers.utils
import omegaconf
import requests
import safetensors.torch
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_vae_checkpoint,
    create_vae_diffusers_config,
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
    parser.add_argument(
        "--cnguess", action="store_true", help="Enable ControlNet Guess Mode"
    )
    parser.add_argument(
        "--ti", action="extend", type=str, nargs="*", help="textual inversion to apply"
    )
    parser.add_argument("--vae", type=str, nargs="?", help="vae checkpoint to apply")

    option = parser.parse_args()
    if option.cnet is not None and (
        option.cnimage is None
        or option.cnscale is None
        or len(option.cnet) != len(option.cnimage)
        or len(option.cnet) != len(option.cnscale)
    ):
        assert False

    return option


def load_original_config():
    # model_type = "v1"
    config_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    original_config_file = io.BytesIO(requests.get(config_url).content)
    return omegaconf.OmegaConf.load(original_config_file)


if __name__ == "__main__":
    option = parse_option()

    generate_params = {
        "prompt": option.prompt,
        "negative_prompt": option.negative,
        "guidance_scale": option.guidance,
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
            generate_params["guess_mode"] = option.cnguess
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

    if option.vae is not None:
        if os.path.isfile(option.vae):
            if option.vae.endswith(".safetensors"):
                checkpoint = safetensors.torch.load_file(option.vae)
            else:
                checkpoint = torch.load(option.vae)

            while "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            checkpoint = {
                f"first_stage_model.{key}": value for key, value in checkpoint.items()
            }
            # 512 is default
            vae_config = create_vae_diffusers_config(
                load_original_config(), image_size=512
            )
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                checkpoint, vae_config
            )
            vae = AutoencoderKL(**vae_config)
            vae.load_state_dict(converted_vae_checkpoint)

        else:
            vae = AutoencoderKL.from_pretrained(option.vae)

        pipe = pipeline_class(**dict(pipe.components, vae=vae))

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

    if option.ti is not None:
        for ti in option.ti:
            if os.path.isfile(ti):
                pipe.load_textual_inversion(
                    ti, token=os.path.splitext(os.path.basename(ti))[0]
                )
            else:
                pipe.load_textual_inversion(ti)

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
