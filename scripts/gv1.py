import argparse
import distutils.util
import inspect
import io
import os
import time
import typing
from types import NoneType, UnionType
from typing import Callable, Literal, TypeVar, Union

import diffusers.pipelines.stable_diffusion.convert_from_ckpt
import diffusers.utils
import docstring_parser
import PIL.Image
import PIL.ImageOps
import requests
import safetensors.torch
import torch
import torch.utils
import torchvision.transforms.functional
from compel import Compel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from omegaconf import OmegaConf
from PIL.Image import Image
from PIL.PngImagePlugin import PngInfo

VAE_CONFIG = diffusers.pipelines.stable_diffusion.convert_from_ckpt.create_vae_diffusers_config(
    OmegaConf.load(
        io.BytesIO(
            requests.get(
                "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
            ).content
        )
    ),
    # 512 is default
    image_size=512,
)

OUTDIR = os.path.join("outputs", "gv1")
os.makedirs(OUTDIR, exist_ok=True)

T = TypeVar("T")


def infinite_randoms():
    while True:
        yield round(time.time())


def or_else(v: T | None, default: T):
    return default if v is None else v


_image_cache = {}


class InferedImage:
    def __init__(
        self,
        prompt: str | None,
        negative: str | None,
        step: int,
        seed: int,
        guidance: float,
        width: int | None,
        height: int | None,
        image: Union[Image, "InferedImage", None],
        prestep: int,
        poststep: int,
        i2i: bool,
        strength: float,
        model: str | None,
        nsfw: bool,
        cnet: list[str] | None,
        cnimage: list[Union[Image, "InferedImage"]] | None,
        cnscale: list[float] | None,
        cnguess: bool,
        ti: list[str] | None,
        vae: str | None,
        scheduler: str | None,
        float32: bool,
        vae_tiling: bool,
        pil: Image | None = None,
        latent: torch.FloatTensor | None = None,
    ):
        self.prompt = prompt
        self.negative = negative
        self.step = step
        self.seed = seed
        self.guidance = guidance
        self.width = width
        self.height = height
        self.image = image
        self.prestep = prestep
        self.poststep = poststep
        self.i2i = i2i
        self.strength = strength
        self.model = model
        self.nsfw = nsfw
        self.cnet = cnet
        self.cnimage = cnimage
        self.cnscale = cnscale
        self.cnguess = cnguess
        self.ti = ti
        self.vae = vae
        self.scheduler = scheduler
        self.float32 = float32
        self.vae_tiling = vae_tiling
        self.pil = pil
        self.latent = latent

    @property
    def sum_of_step(self):
        return self.step + (
            self.image.sum_of_step if isinstance(self.image, InferedImage) else 0
        )

    @classmethod
    def apply_to_pnginfo(
        cls, pnginfo: PngInfo, prefix: str, image: Union[Image, "InferedImage", None]
    ):
        if image is None:
            return

        def add_itxt(field: str, value):
            if value is not None:
                pnginfo.add_itxt(f"{prefix}.{field}", str(value))

        if isinstance(image, Image):
            pnginfo.add_text(f"{prefix}.content", image.tobytes())
            add_itxt("mode", image.mode)
            add_itxt("width", image.width)
            add_itxt("height", image.height)
            return

        def add_itxt_list(field: str, value: list | None):
            if value is not None:
                for i, v in enumerate(value):
                    add_itxt(f"{field}.{i}", v)

        add_itxt("prompt", image.prompt)
        add_itxt("negative", image.negative)
        add_itxt("step", image.step)
        add_itxt("seed", image.seed)
        add_itxt("guidance", image.guidance)
        add_itxt("width", image.width)
        add_itxt("height", image.height)
        cls.apply_to_pnginfo(pnginfo, f"{prefix}.image", image.image)
        add_itxt("poststep", image.poststep)
        add_itxt("i2i", image.i2i)
        add_itxt("strength", image.strength)
        add_itxt("model", image.model)
        add_itxt("nsfw", image.nsfw)
        add_itxt_list("cnet", image.cnet)
        if image.cnimage is not None:
            for i, v in enumerate(image.cnimage):
                cls.apply_to_pnginfo(pnginfo, f"{prefix}.cnimage.{i}", v)
        add_itxt_list("cnscale", image.cnscale)
        add_itxt("cnguess", image.cnguess)
        add_itxt_list("ti", image.ti)
        add_itxt("vae", image.vae)
        add_itxt("scheduler", image.scheduler)
        add_itxt("float32", image.float32)
        add_itxt("vae_tiling", image.vae_tiling)

    def save(self, path: str):
        pil = self.pil
        if pil is None:
            pil = self.result("pil")
        if pil is None:
            pil = PIL.Image.new("L", (1, 1))

        pnginfo = PngInfo()
        type(self).apply_to_pnginfo(pnginfo, "gv1", self)

        pil.save(path, pnginfo=pnginfo)

    @classmethod
    def load_from_pngtext(
        cls,
        text: dict[str,],
        prefix: str,
        prestep: int | None,
        poststep: int | None,
    ):
        if not any([k.startswith(f"{prefix}.") for k in text.keys()]):
            return None

        def fetch(dtype: Callable[[str], T]):
            def fetcher(field: str):
                data = text.get(f"{prefix}.{field}")
                return None if data is None else dtype(data)

            return fetcher

        fetch_str = fetch(str)
        fetch_int = fetch(int)
        fetch_float = fetch(float)
        fetch_bool = fetch(distutils.util.strtobool)

        def fetch_image(
            field: str, prestep: int | None = None, poststep: int | None = None
        ):
            content = fetch_str(f"{field}.content")
            if content is None:
                return cls.load_from_pngtext(
                    text, f"{prefix}.{field}", prestep, poststep
                )
            else:
                return PIL.Image.frombytes(
                    fetch_str(f"{field}.mode"),
                    (
                        fetch_int(f"{field}.width"),
                        fetch_int(f"{field}.height"),
                    ),
                    bytes(content, "latin-1"),
                )

        def fetch_list(field: str, fetcher: Callable[[str], T]):
            if not any([k.startswith(f"{prefix}.{field}.") for k in text.keys()]):
                return None

            result: list[T] = []
            while True:
                data = fetcher(f"{field}.{len(result)}")
                if data is None:
                    break
                else:
                    result.append(data)

            return result

        prompt = fetch_str("prompt")
        negative = fetch_str("negative")
        seed = fetch_int("seed")
        guidance = fetch_float("guidance")
        width = fetch_int("width")
        height = fetch_int("height")
        image = fetch_image("image", prestep, poststep)
        i2i = fetch_bool("i2i")
        strength = fetch_float("strength")
        model = fetch_str("model")
        nsfw = fetch_bool("nsfw")
        cnet = fetch_list("cnet", fetch_str)
        cnimage = fetch_list("cnimage", fetch_image)
        cnscale = fetch_list("cnscale", fetch_float)
        cnguess = fetch_bool("cnguess")
        ti = fetch_list("ti", fetch_str)
        vae = fetch_str("vae")
        scheduler = fetch_str("scheduler")
        float32 = fetch_bool("float32")
        vae_tiling = fetch_bool("vae_tiling")

        prestep_of_image = image.sum_of_step if isinstance(image, InferedImage) else 0

        if prestep is None:
            step = fetch_int("step")
        else:
            step = prestep - prestep_of_image
            if step <= 0:
                return image

        if poststep is None:
            poststep_of_image = or_else(fetch_int("poststep"), 0)
        else:
            poststep_of_image = poststep
            subimage = image
            while isinstance(subimage, InferedImage):
                subimage.poststep += step
                subimage = subimage.image

        return InferedImage(
            prompt=prompt,
            negative=negative,
            step=step,
            seed=seed,
            guidance=guidance,
            width=width,
            height=height,
            image=image,
            prestep=prestep_of_image,
            poststep=poststep_of_image,
            i2i=i2i,
            strength=strength,
            model=model,
            nsfw=nsfw,
            cnet=cnet,
            cnimage=cnimage,
            cnscale=cnscale,
            cnguess=cnguess,
            ti=ti,
            vae=vae,
            scheduler=scheduler,
            float32=float32,
            vae_tiling=vae_tiling,
        )

    @classmethod
    def load(
        cls,
        path_or_img: Union[str, Image],
        prestep: int | None = None,
        poststep: int | None = None,
    ):
        if isinstance(path_or_img, str):
            cache_key = (path_or_img, prestep, poststep)
            if cache_key in _image_cache.keys():
                return _image_cache[cache_key]

            image = PIL.Image.open(path_or_img)
        else:
            image = path_or_img

        infered = cls.load_from_pngtext(
            getattr(image, "text", None), "gv1", prestep, poststep
        )

        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        if infered is None:
            if isinstance(path_or_img, str):
                _image_cache[cache_key] = image
            return image

        infered.pil = image
        if isinstance(path_or_img, str):
            _image_cache[cache_key] = infered

        return infered

    def result(self, output_type: Literal["pil", "latent"]):
        output = getattr(self, output_type)
        if output is not None:
            return output

        args = {
            "prompt": self.prompt,
            "negative": self.negative,
            "step": self.step,
            "seed": self.seed,
            "guidance": self.guidance,
            "width": self.width,
            "height": self.height,
            "image": self.image,
            "prestep": self.prestep,
            "poststep": self.poststep,
            "i2i": self.i2i,
            "strength": self.strength,
            "model": self.model,
            "nsfw": self.nsfw,
            "cnet": self.cnet,
            "cnimage": self.cnimage,
            "cnscale": self.cnscale,
            "cnguess": self.cnguess,
            "ti": self.ti,
            "vae": self.vae,
            "scheduler": self.scheduler,
            "float32": self.float32,
            "vae_tiling": self.vae_tiling,
        }

        output = gv1(
            **{k: v for k, v in args.items() if v is not None},
            output_type=output_type,
        )

        setattr(self, output_type, output)
        return output

    @property
    def parameters(self):
        parameters = inspect.signature(gv1).parameters
        return {
            k: v
            for k, v in vars(self).items()
            if k not in ("pil", "latent") and v != parameters[k].default
        }


def init_dpmsolver(
    algorithm_type: Literal["dpmsolver", "dpmsolver++", "deis"], use_karras: bool
):
    def initializer(config):
        solver = DPMSolverMultistepScheduler.from_config(
            config | {"algorithm_type": algorithm_type}
        )
        solver.use_karras_sigmas = use_karras
        return solver

    return initializer


SchedulersLiteral = Literal["DPM++Karras"]
SCHEDULERS_INITS = {"DPM++Karras": init_dpmsolver("dpmsolver++", True)}


class ClippingScheduler(SchedulerMixin):
    def __init__(self, original, prestep: int, poststep: int):
        self.original = original
        self.prestep = prestep
        self.poststep = poststep

    @property
    def init_noise_sigma(self):
        # apply only "first" inference
        return self.original.init_noise_sigma if self.prestep == 0 else 1.0

    @property
    def order(self):
        return self.original.order

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: None = None,
        *args,
        **kwargs,
    ):
        assert num_inference_steps is not None
        assert device is not None
        assert timesteps is None

        self.original.set_timesteps(
            num_inference_steps=self.prestep + num_inference_steps + self.poststep,
            device=device,
            *args,
            **kwargs,
        )

        # bacause len(timesteps) == num_inference_steps * order "+ 1",
        # so capture last one in "last step"
        if self.poststep == 0:
            self.timesteps = self.original.timesteps[self.prestep * self.order :].to(
                device
            )
        else:
            self.timesteps = self.original.timesteps[
                self.prestep
                * self.order : (self.prestep + num_inference_steps)
                * self.order
            ].to(device)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        *args,
        **kwargs,
    ):
        return self.original.step(model_output, timestep, sample, *args, **kwargs)

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: int | None = None, *args, **kwargs
    ):
        return self.original.scale_model_input(
            sample,
            timestep,
            *args,
            **kwargs,
        )

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ):
        return self.original.add_noise(original_samples, noise, timesteps)


_pipeline_modules_cache = {}


def load_pipeline_modules(model: str, ti: list[str] | None):
    if ti is not None:
        ti.sort()
    key = (model, *or_else(ti, []))
    if key in _pipeline_modules_cache.keys():
        return _pipeline_modules_cache[key]

    if (
        model.endswith(".pt")
        or model.endswith(".ckpt")
        or model.endswith(".safetensors")
    ):
        pipe = StableDiffusionPipeline.from_ckpt(model)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model)

    for ti_model in or_else(ti, []):
        if os.path.isfile(ti_model):
            pipe.load_textual_inversion(
                ti_model, token=os.path.splitext(os.path.basename(ti_model))[0]
            )
        else:
            pipe.load_textual_inversion(ti_model)

    modules = pipe.components
    _pipeline_modules_cache[key] = modules
    return modules


_controlnet_model_cache = {}


def load_controlnet_model(model: str, torch_dtype: torch.dtype):
    key = (model, torch_dtype)
    if key in _controlnet_model_cache.keys():
        return _controlnet_model_cache[key]

    instance = ControlNetModel.from_pretrained(model, torch_dtype=torch_dtype)
    _controlnet_model_cache[key] = instance
    return instance


_vae_cache = {}


def load_vae(vae: str):
    key = vae
    if key in _vae_cache.keys():
        return _vae_cache[key]

    if os.path.isfile(vae):
        if vae.endswith(".safetensors"):
            checkpoint = safetensors.torch.load_file(vae)
        else:
            checkpoint = torch.load(vae)

        checkpoint = {
            f"first_stage_model.{key}": value for key, value in checkpoint.items()
        }

        checkpoint = diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_vae_checkpoint(
            checkpoint, VAE_CONFIG
        )
        vae_module = AutoencoderKL(**VAE_CONFIG)
        vae_module.load_state_dict(checkpoint)

    else:
        vae_module = AutoencoderKL.from_pretrained(vae)

    _vae_cache[key] = vae_module
    return vae_module


def gv1(
    prompt: str | None = None,
    prompt2: str | None = None,
    negative: str | None = None,
    negative2: str | None = None,
    step: int = 50,
    seed: int | None = None,
    guidance: float = 7.5,
    width: int | None = None,
    height: int | None = None,
    image: str | Image | InferedImage | None = None,
    prestep: int | None = None,
    poststep: int | None = None,
    i2i: bool = False,
    strength: float = 0.8,
    model: str | None = None,
    nsfw: bool = False,
    cnet: list[str] | None = None,
    cnimage: list[str] | None = None,
    cnscale: list[float] | None = None,
    cninit: Literal["add", "replace"] = "add",
    cnguess: bool = False,
    ti: list[str] | None = None,
    tiinit: Literal["add", "replace"] = "add",
    vae: str | None = None,
    scheduler: SchedulersLiteral | None = None,
    float32: bool = False,
    vae_tiling: bool = False,
    output_type: Literal["png", "pil", "latent"] = "png",
    info: bool = False,
    args: bool = False,
    inherit: bool = False,
    inherit_only: bool = False,
):
    r"""
    execute inference steps

    Args:
        prompt: positive prompt
        prompt2: additional positive prompt on `inherit`
        negative: negative prompt
        negative2: additional negative prompt on `inherit`
        step: num of inference steps
        seed: fix seed to it
        guidance: higher guidance scale encourages to generate images that are closely linked to the text prompt
        width: width of output image
        height: height of output image
        image: source image
        prestep: num of inference step for `image` input
        poststep: should be 0
        i2i: use `image` as img2img source
        strength: how much to transform the reference image
        model: model id in huggingface / path to checkpoint directory / path to checkpoint file
        nsfw: allow NSFW image
        cnet: ControlNet model id to apply
        cnimage: path to image passed to ControlNet
        cnscale: conditioning scale of ControlNet
        cninit: `replace` to discard `image`'s ControlNet
        cnguess: enable ControlNet guess mode
        ti: textual inversion checkpoint to apply
        tiinit: `replace` to discard `image`'s textual inversion
        vae: vae checkpoint to apply
        scheduler: scheduler name to use
        float32: use float32 to dtype
        vae_tiling: enable vae tiling
        output_type: `png` to output png
        info: output gv1 parameters of `image`
        args: output gv1 parameters of `image` by argument style and argument images
        inherit: with `image` generated by gv1, inherit `prompt`, `negative`, `width`, `height`, `model`, `vae`, `scheduler` if not decribed
        inherit: inherit parameters same with `--inherit` but discard `image`
    """

    # validation
    if cnet is not None:
        assert cnimage is not None
        assert cnscale is not None
        assert len(cnet) == len(cnimage)
        assert len(cnet) == len(cnscale)
    if width is not None:
        assert height is not None
    if info or args:
        assert not (info and args)
        assert image is not None

    # select num of poststep
    post_step = 0 if poststep is None else poststep

    # select pipeline by --image
    if image is None:
        pipeline_class = StableDiffusionPipeline
        image_parameters = {}
        prev_image = None
        prev_step = or_else(prestep, 0)
    elif info:
        text = getattr(PIL.Image.open(image), "text", {})
        print(
            {
                k: "<bytes>" if k.endswith(".content") else v
                for k, v in text.items()
                if k.startswith("gv1.")
            }
        )

        return
    elif args:
        # if poststep is omitted, output saved poststep
        prev_image = InferedImage.load(image, prestep, poststep)

        dirname = os.path.join(OUTDIR, f"{len(os.listdir(OUTDIR)):09}")

        def quote(v):
            result = str(v)
            return f'"{result}"' if " " in result else result

        arg_list = [
            f"--{k} {quote(v)}"
            for k, v in prev_image.parameters.items()
            if not k in ("image", "cnimage")
        ]
        images = {}

        if prev_image.image is not None:
            filename = os.path.join(dirname, "image.png")
            arg_list.append(f"--image {quote(filename)}")
            images[filename] = prev_image.image

        if prev_image.cnimage is not None:
            for i, v in enumerate(prev_image.cnimage):
                filename = os.path.join(dirname, f"cn{i}.png")
                arg_list.append(f"--cnimage {quote(filename)}")
                images[filename] = v

        if len(images) > 0:
            os.makedirs(dirname)
            for k, v in images.items():
                v.save(k)

        print(" ".join(arg_list))

        return
    else:
        prev_image = InferedImage.load(image, prestep, step + post_step)

        if (inherit or inherit_only) and isinstance(prev_image, InferedImage):
            prompt = or_else(prompt, prev_image.prompt)
            negative = or_else(negative, prev_image.negative)
            width = or_else(width, prev_image.width)
            height = or_else(height, prev_image.height)
            model = or_else(model, prev_image.model)
            vae = or_else(vae, prev_image.vae)
            scheduler = or_else(scheduler, prev_image.scheduler)

        if inherit_only:
            pipeline_class = StableDiffusionPipeline
            image_parameters = {}
            prev_image = None
            prev_step = or_else(prestep, 0)
        elif i2i:
            pipeline_class = StableDiffusionImg2ImgPipeline
            image_parameters = {
                "image": (
                    prev_image.result("pil")
                    if isinstance(prev_image, InferedImage)
                    else prev_image
                ).resize((width, height)),
                "strength": strength,
            }
            prev_step = 0
        else:
            assert isinstance(prev_image, InferedImage)
            pipeline_class = StableDiffusionPipeline
            image_parameters = {
                "latents": torchvision.transforms.functional.resize(
                    prev_image.result("latent"),
                    (width // 8, height // 8),
                    antialias=True,
                )
            }
            prev_step = prev_image.sum_of_step

    # merge prompts
    def merge_prompt(*prompts: str):
        result = ", ".join([p for p in prompts if p is not None])
        return None if result == "" else result

    prompt = merge_prompt(prompt, prompt2)
    negative = merge_prompt(negative, negative2)

    # merge list parameters
    current_cnimage = [InferedImage.load(p) for p in or_else(cnimage, [])]
    merged_cnet = cnet
    merged_cnimage = current_cnimage
    merged_cnscale = cnscale
    if cninit == "add" and isinstance(prev_image, InferedImage):
        merged_cnet = or_else(cnet, []) + or_else(prev_image.cnet, [])
        merged_cnimage = or_else(current_cnimage, []) + or_else(prev_image.cnimage, [])
        merged_cnscale = or_else(cnscale, []) + or_else(prev_image.cnscale, [])
        if len(merged_cnet) == 0:
            merged_cnet = None
            merged_cnimage = None
            merged_cnscale = None

    merged_ti = ti
    if tiinit == "add" and isinstance(prev_image, InferedImage):
        merged_ti = or_else(ti, []) + or_else(prev_image.ti, [])
    if merged_ti is not None and len(merged_ti) == 0:
        merged_ti = None

    # prepare ControlNet
    if merged_cnet is None:
        controlnet = None
        controlnet_inference_parameters = {}
    else:
        controlnet = [
            load_controlnet_model(
                model_id, torch_dtype=torch.float32 if float32 else torch.float16
            )
            for model_id in merged_cnet
        ]
        controlnet_inference_parameters = {
            "controlnet_conditioning_image"
            if "image" in image_parameters.keys()
            else "image": [
                i.result("pil") if isinstance(i, InferedImage) else i
                for i in merged_cnimage
            ],
            "controlnet_conditioning_scale": merged_cnscale,
        } | ({} if "image" in image_parameters.keys() else {"guess_mode": cnguess})

    # select image size
    size = (
        {}
        if width is None or (i2i and merged_cnet is None)
        else {"width": width, "height": height}
    )

    # init pipe and load model
    pipe = pipeline_class(**load_pipeline_modules(model, merged_ti))

    # replace scheduler
    if scheduler is not None:
        pipe.scheduler = SCHEDULERS_INITS[scheduler](pipe.scheduler.config)

    # clip scheduler
    pipe.scheduler = ClippingScheduler(pipe.scheduler, prev_step, post_step)

    # replace vae
    if vae is not None:
        pipe.vae = load_vae(vae)

    # apply ControlNet to pipe
    if controlnet is not None:
        if "image" in image_parameters.keys():
            pipe = DiffusionPipeline.from_pretrained(
                # model path is dummy
                pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
                custom_pipeline="stable_diffusion_controlnet_img2img",
                **pipe.components,
                controlnet=controlnet,
            )
        else:
            pipe = StableDiffusionControlNetPipeline(
                **pipe.components, controlnet=controlnet
            )

    # prepare weighted prompts
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    prompt_embeds = None if prompt is None else compel(prompt)
    negative_embeds = None if negative is None else compel(negative)

    # disable nsfw filter
    if nsfw and pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, False)

    # memory optimization
    pipe = pipe.to("cuda", torch_dtype=torch.float32 if float32 else torch.float16)
    if vae_tiling:
        pipe.enable_vae_tiling()

    # make path to save image
    img_count = len(os.listdir(OUTDIR))

    # inference loop
    for s in infinite_randoms() if seed is None else [seed]:
        with torch.inference_mode():
            out = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                num_inference_steps=step,
                guidance_scale=guidance,
                generator=torch.manual_seed(s),
                output_type="latent" if output_type == "latent" else "pil",
                **size,
                **controlnet_inference_parameters,
                **image_parameters,
            )

        is_safe = (
            nsfw
            or out.nsfw_content_detected is None
            or not out.nsfw_content_detected[0]
        )

        if output_type == "latent":
            return out.images if is_safe else None

        if output_type == "pil":
            return out.images[0] if is_safe else None

        if is_safe:
            InferedImage(
                prompt=prompt,
                negative=negative,
                step=step,
                seed=s,
                guidance=guidance,
                width=width,
                height=height,
                image=prev_image,
                prestep=prev_image.sum_of_step
                if isinstance(prev_image, InferedImage)
                else 0,
                poststep=0,
                i2i=i2i,
                strength=strength,
                model=model,
                nsfw=nsfw,
                cnet=merged_cnet,
                cnimage=merged_cnimage,
                cnscale=merged_cnscale,
                cnguess=cnguess,
                ti=merged_ti,
                vae=vae,
                scheduler=scheduler,
                float32=float32,
                vae_tiling=vae_tiling,
                pil=out.images[0],
            ).save(os.path.join(OUTDIR, f"{img_count:09}.png"))
            img_count += 1


def call_by_args(fn: Callable):
    parser = argparse.ArgumentParser()

    docs = docstring_parser.parse_from_object(fn).params
    shorthands = set()
    for parameter in inspect.signature(fn).parameters.values():
        names = [f"--{parameter.name}"]

        shorthand = parameter.name[0].upper()
        if not shorthand in shorthands:
            shorthands.add(shorthand)
            names.append(f"-{shorthand}")

        doc = [d.description for d in docs if d.arg_name == parameter.name]

        annotation = parameter.annotation
        origin = typing.get_origin(annotation)
        type_args = typing.get_args(annotation)
        is_optional = parameter.default != inspect.Signature.empty

        if origin in [Union, UnionType]:
            annotation = next((t for t in type_args if t != NoneType), None)
            origin = typing.get_origin(annotation)
            type_args = typing.get_args(annotation)

        if origin is None:
            origin = annotation
            type_args = ()

        if origin == bool and parameter.default == False:
            extra = {"action": "store_true"}
        elif origin == list:
            extra = {
                "type": type_args[0],
                "action": "extend",
                "nargs": "*" if is_optional else "+",
            }
        elif origin == Literal:
            extra = {
                "type": type(type_args[0]),
                "choices": type_args,
                "default": parameter.default,
            }
        elif is_optional:
            extra = {"type": origin, "nargs": "?", "default": parameter.default}
        else:
            extra = {"type": origin}

        parser.add_argument(
            *names,
            **extra,
            help=doc[0] if len(doc) > 0 else None,
        )

    fn(**vars(parser.parse_args()))


if __name__ == "__main__":
    call_by_args(gv1)
