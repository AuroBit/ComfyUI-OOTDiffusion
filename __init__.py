import os

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from .humanparsing.aigc_run_parsing import Parsing
from .inference_ootd import OOTDiffusion
from .ootd_utils import get_mask_location
from .openpose.run_openpose import OpenPose

_category_get_mask_input = {
    "upperbody": "upper_body",
    "lowerbody": "lower_body",
    "dress": "dresses",
}


class LoadOOTDPipeline:
    display_name = "Load OOTDiffusion Local"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": (["Half body", "Full body"],),
                "path": ("STRING", {"default": "models/OOTDiffusion"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"

    CATEGORY = "OOTD"

    @staticmethod
    def load_impl(type, path):
        if type == "Half body":
            type = "hd"
        elif type == "Full body":
            type = "dc"
            raise RuntimeError("full body is not supported yet")
        else:
            raise ValueError(
                f"unknown input type {type} must be 'Half body' or 'Full body'"
            )
        if not os.path.isdir(path):
            raise ValueError(f"input path {path} is not a directory")
        return OOTDiffusion(path, model_type=type)

    def load(self, type, path):
        return (self.load_impl(type, path),)


class LoadOOTDPipelineHub(LoadOOTDPipeline):
    display_name = "Load OOTDiffusion from Hub🤗"

    repo_id = "levihsu/OOTDiffusion"
    repo_revision = "c63b33843e01c8c2c8e591a1d6b88a2feba478b8"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": (["Half body", "Full body"],),
            }
        }

    def load(self, type):  # type: ignore
        # DiffusionPipeline.from_pretrained doesn't support subfolder
        # So we use snapshot_download to get local path first
        path = snapshot_download(
            self.repo_id,
            revision=self.repo_revision,
            resume_download=True,
        )
        return (LoadOOTDPipeline.load_impl(type, path),)


class OOTDGenerate:
    display_name = "OOTDiffusion Generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL",),
                "cloth_image": ("IMAGE",),
                "model_image": ("IMAGE",),
                # Openpose from comfyui-controlnet-aux not work
                # "keypoints": ("POSE_KEYPOINT",),
                # TODO: add category when dc model release
                # "category": ("STRING", ["upperbody", "lowerbody", "dress"]),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 14.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "image_masked")
    FUNCTION = "generate"

    CATEGORY = "OOTD"

    def generate(self, pipe, cloth_image, model_image, seed, steps, cfg):
        category = "upperbody"
        # if model_image.shape != (1, 1024, 768, 3) or (
        #     cloth_image.shape != (1, 1024, 768, 3)
        # ):
        #     raise ValueError(
        #         f"Input image must be size (1, 1024, 768, 3). "
        #         f"Got model_image {model_image.shape} cloth_image {cloth_image.shape}"
        #     )

        # (1,H,W,3) -> (3,H,W)
        model_image = model_image.squeeze(0)
        model_image = model_image.permute((2, 0, 1))
        model_image = to_pil_image(model_image)
        if model_image.size != (768, 1024):
            print(f"Inconsistent model_image size {model_image.size} != (768, 1024)")
        model_image = model_image.resize((768, 1024))
        cloth_image = cloth_image.squeeze(0)
        cloth_image = cloth_image.permute((2, 0, 1))
        cloth_image = to_pil_image(cloth_image)
        if cloth_image.size != (768, 1024):
            print(f"Inconsistent cloth_image size {cloth_image.size} != (768, 1024)")
        cloth_image = cloth_image.resize((768, 1024))

        model_parse, _ = Parsing(pipe.device)(model_image.resize((384, 512)))
        keypoints = OpenPose()(model_image.resize((384, 512)))
        mask, mask_gray = get_mask_location(
            pipe.model_type,
            _category_get_mask_input[category],
            model_parse,
            keypoints,
            width=384,
            height=512,
        )
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, model_image, mask)
        images = pipe(
            category=category,
            image_garm=cloth_image,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_image,
            num_samples=1,
            num_steps=steps,
            image_scale=cfg,
            seed=seed,
        )

        # pil(H,W,3) -> tensor(H,W,3)
        output_image = to_tensor(images[0])
        output_image = output_image.permute((1, 2, 0))
        masked_vton_img = masked_vton_img.convert("RGB")
        masked_vton_img = to_tensor(masked_vton_img)
        masked_vton_img = masked_vton_img.permute((1, 2, 0))

        return ([output_image], [masked_vton_img])


_export_classes = [
    LoadOOTDPipeline,
    LoadOOTDPipelineHub,
    OOTDGenerate,
]

NODE_CLASS_MAPPINGS = {c.__name__: c for c in _export_classes}

NODE_DISPLAY_NAME_MAPPINGS = {
    c.__name__: getattr(c, "display_name", c.__name__) for c in _export_classes
}
