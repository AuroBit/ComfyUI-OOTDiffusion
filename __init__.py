import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from .humanparsing.aigc_run_parsing import Parsing
from .inference_ootd import OOTDiffusion
from .ootd_utils import get_mask_location
from .openpose.run_openpose import OpenPose


class LoadOOTDPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

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
                # "seed": ("INT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "image_masked")
    FUNCTION = "generate"

    CATEGORY = "OOTD"

    def generate(self, pipe, cloth_image, model_image):
        # (1,H,W,3) -> (3,H,W)
        model_image = model_image.squeeze(0)
        model_image = model_image.permute((2, 0, 1))
        model_image = to_pil_image(model_image)
        cloth_image = cloth_image.squeeze(0)
        cloth_image = cloth_image.permute((2, 0, 1))
        cloth_image = to_pil_image(cloth_image)
        model_parse, _ = Parsing(pipe.device)(model_image)
        keypoints = OpenPose()(model_image)
        mask, mask_gray = get_mask_location(
            "hd",
            "upper_body",
            model_parse,
            keypoints,
            width=model_image.size[0],
            height=model_image.size[1],
        )
        masked_vton_img = Image.composite(mask_gray, model_image, mask)
        images = pipe(
            model_type="hd",
            category="upper_body",
            image_garm=cloth_image,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_image,
            num_samples=1,
            num_steps=20,
            image_scale=1,
            seed=-1,
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
