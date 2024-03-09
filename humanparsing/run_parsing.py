from pathlib import Path
import os
import onnxruntime as ort
from .parsing_api import onnx_inference
import torch


class Parsing:
    def __init__(self, atr_model_path, lip_model_path):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # session_options.add_session_config_entry('gpu_id', str(gpu_id))
        self.session = ort.InferenceSession(atr_model_path,
                                            sess_options=session_options, providers=['CPUExecutionProvider'])
        self.lip_session = ort.InferenceSession(lip_model_path,
                                                sess_options=session_options, providers=['CPUExecutionProvider'])


    def __call__(self, input_image):
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask