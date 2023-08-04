"""
Yolov8 Pose detection
"""

from ..base import PoseModel
from .utils import download_yolo_pose
from ultralytics import YOLO

import torch
import numpy as np
import os


def get_torch_device():
    # if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     return torch.device("mps")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class YoloPoseModel(PoseModel):
    def __init__(self, model_path: str = "./models/yolov8n-pose.pt"):
        self._device = get_torch_device()
        self._model = self._load_model(model_path)

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            download_yolo_pose(model_path)
        model = YOLO(model_path)
        return model

    @torch.no_grad()
    def _run_model(self, image):
        return self._model(image, device=self._device)

    def predict(self, image):
        return self._run_model(image)

    def draw_landmarks(self, image, results):
        return results[0].plot(img=image)

    def pose_landmarks_to_nparray(self, pose_landmarks, height, width):
        return np.squeeze(pose_landmarks)

    def results_to_pose_landmarks(self, results, height=None, width=None):
        return np.squeeze(results[0].keypoints[0].xy.cpu().numpy())
