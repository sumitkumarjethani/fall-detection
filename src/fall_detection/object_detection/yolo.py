"""Yolov8 object detection module"""

from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2


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


class YoloObjectDetector(ObjectDetector):
    def __init__(self, model_name: str = "yolov8n.pt"):
        self._model_name = model_name
        self._model = self._load_yolo_model(model_name)

    def _load_yolo_model(self, model_name: str):
        return YOLO(model_name)

    def predict(self, image):
        results = self._model(image)
        return results[0]

    def draw_results(self, image, results):
        return results.plot(img=image)

    def results_to_object_detection_samples(self, results):
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
