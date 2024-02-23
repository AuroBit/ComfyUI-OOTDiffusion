import numpy as np
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "model_type": ("STRING", ["hd", "dc"]),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"

    CATEGORY = "OOTD"

    def load(self):
        return (OOTDiffusion(hg_root="models/OOTDiffusion"),)


class OOTDGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL",),
                "cloth_image": ("IMAGE",),
                "model_image": ("IMAGE",),
                # Openpose from comfyui-controlnet-aux not work
                # "keypoints": ("POSE_KEYPOINT",),
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

        output_image = to_tensor(images[0])
        output_image = output_image.permute((1, 2, 0))
        masked_vton_img = masked_vton_img.convert("RGB")
        masked_vton_img = to_tensor(masked_vton_img)
        masked_vton_img = masked_vton_img.permute((1, 2, 0))

        return ([output_image], [masked_vton_img])


NODE_CLASS_MAPPINGS = {
    "LoadOOTDPipeline": LoadOOTDPipeline,
    "OOTDGenerate": OOTDGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOOTDPipeline": "Load OOTDiffusion",
    "OOTDGenerate": "OOTDiffusion Generate",
}
