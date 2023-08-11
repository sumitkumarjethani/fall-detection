"""Yolov8 object detection module"""
import os
import numpy as np

from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ultralytics import YOLO
from ..utils import get_torch_device


@dataclass
class ObjectDetectionSample:
    class_name: str
    xyxy: np.ndarray
    # xywh: np.ndarray
    # xyxyn: np.ndarray  # box with xyxy format but normalized
    # xywhn: np.ndarray  # box with xywh format but normalized
    # embeddings: np.ndarray


class ObjectDetector(ABC):
    @abstractmethod
    def predict(self, image):
        pass

    @abstractmethod
    def draw_results(self, image, results):
        pass

    @abstractmethod
    def results_to_object_detection_samples(self, results):
        pass


def download_yolo_object_model(output_path):
    try:
        os.system(
            "wget "
            + "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
            + " -O "
            + output_path
        )
    except Exception as e:
        raise Exception(f"could not download yolo pose: {e}")


class YoloObjectDetector(ObjectDetector):
    def __init__(self, model_path: str = "yolov8n.pt"):
        self._device = get_torch_device()
        self._model_name = model_path
        self._model = self._load_yolo_model(model_path)
        self._conf = 0.6
        self._classes = [0, 59, 57, 56]

    def _load_yolo_model(self, model_path: str):
        if not os.path.exists(model_path):
            download_yolo_object_model(model_path)
        model = YOLO(model_path)
        return model

    def predict(self, image):
        results = self._model(
            image,
            device=self._device,
            verbose=False,
            conf=self._conf,
            classes=self._classes,
        )
        return results[0]

    def draw_results(self, image, results):
        return results.plot(img=image)

    def results_to_object_detection_samples(
        self, results
    ) -> List[ObjectDetectionSample]:
        cls_names = results.names
        out = []
        for bb in results.boxes:
            cls = bb.cls.cpu().numpy()[0]
            cls_name = cls_names[cls]
            od = ObjectDetectionSample(
                class_name=cls_name,
                xyxy=np.squeeze(bb.xyxy.cpu().numpy()),
                # xywh=np.squeeze(bb.xywh.cpu().numpy()),
                # xyxyn=np.squeeze(bb.xyxyn.cpu().numpy()),
                # xywhn=np.squeeze(bb.xywhn.cpu().numpy()),
            )
            out.append(od)
        return out
