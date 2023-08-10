"""
Yolov8 Pose detection
"""
from ..base import PoseModel, COCO_POSE_KEYPOINTS
from ultralytics import YOLO
import numpy as np
import torch
import os


# TODO: remove and use from common utils
def get_torch_device():
    # if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     return torch.device("mps")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def download_yolo_pose_model(output_path):
    try:
        os.system(
            "wget "
            + "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt"
            + " -O "
            + output_path
        )
    except Exception as e:
        raise Exception(f"could not download yolo pose: {e}")


class YoloPoseModel(PoseModel):
    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        self._device = get_torch_device()
        self._model = self._load_model(model_path)

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            download_yolo_pose_model(model_path)
        model = YOLO(model_path)
        return model

    @torch.no_grad()
    def predict(self, image):
        results = self._model(image, device=self._device, verbose=False)
        if (
            results is None
            or results[0].keypoints.shape[1] == 0
            or results[0].boxes.conf[0] < 0.6
        ):
            return None
        return results

    def draw_landmarks(self, image, results):
        return results[0].plot(img=image)

    def results_to_pose_landmarks(self, results, height=None, width=None):
        return np.squeeze(results[0].keypoints[0].data.numpy())

    @property
    def landmarks_names(self):
        return COCO_POSE_KEYPOINTS
