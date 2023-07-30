"""
Yolov7 Pose detection
"""

import math
from ..pose import PoseModel
from .utils import (
    letterbox,
    non_max_suppression_kpt,
    output_to_keypoint,
    draw_prediction_on_image,
    download_yolo_pose,
)
import torch
import torchvision
import numpy as np
import os


# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def get_torch_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("mps backend available")
        return torch.device("mps")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class YoloPoseModel(PoseModel):
    def __init__(self, model_path: str = "yolov7-w6-pose.pt"):
        self._device = get_torch_device()
        self._model = self._load_model(model_path)

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            download_yolo_pose(model_path)
        weigths = torch.load(model_path, map_location=self._device)
        model = weigths["model"]
        _ = model.float().eval()
        return model

    def _process_image(self, image):
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image = torchvision.transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        return image

    @torch.no_grad()
    def _run_model(self, image):
        return self._model(image)

    def _run_inference(self, image):
        if torch.cuda.is_available():
            image = image.half().to(self._device)
        else:
            image = image.to(self._device)
        output, _ = self._run_model(image)
        return output

    def _process_output(self, output):
        output = non_max_suppression_kpt(
            output,
            0.25,
            0.65,
            nc=self._model.yaml["nc"],
            nkpt=self._model.yaml["nkpt"],
            kpt_label=True,
        )
        with torch.no_grad():
            output = output_to_keypoint(output)
        if len(output.shape) == 2:
            output = (output[0, 7:].T).reshape((1, 1, 17, 3))
            output[:, :, :, [1, 0]] = output[:, :, :, [0, 1]]
            output[:, :, :, 1] /= 960
            output[:, :, :, 0] /= 640
            return output
        return None

    def predict(self, image):
        image = self._process_image(image)
        output = self._run_inference(image)
        return self._process_output(output)

    def draw_landmarks(self, image, pose_landmarks):
        # print(image.shape)
        # print(pose_landmarks)
        return draw_prediction_on_image(image, pose_landmarks)

    def pose_landmarks_to_nparray(self, pose_landmarks, height, width):
        return pose_landmarks
