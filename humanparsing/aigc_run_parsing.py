from .parsing_api import load_atr_model, load_lip_model, inference
import torch


class Parsing:
    def __init__(self, device):
        self.device = device
        self.atr_model = load_atr_model()
        self.lip_model = load_lip_model()

    def __call__(self, input_image):
        parsed_image, face_mask = inference(self.atr_model, self.lip_model, input_image)
        return parsed_image, face_mask
