"""Fall detection pipeline"""

from typing import Optional
from .classification import EMADictSmoothing, PoseClassifier
from .detection import StateDetector
from .embedding import PoseEmbedder
from .rules import PersonIsAlone, PersonNotOnFurniture, RulesChecker
from ..object_detection.yolo import ObjectDetector
from ..pose.base import PoseModel
from ..fall.plot import plot_fall_text
import numpy as np


class Pipeline:
    def __init__(
        self,
        pose_model: PoseModel,
        object_model: ObjectDetector,
        classification_model: PoseClassifier,
        detector: Optional[StateDetector] = None,
        detect_class: str = "fall",
        enter_threshold: int = 6,
        exit_threshold: int = 4,
        window_size: int = 10,
        alpha: float = 0.3,
        rules_checker: Optional[RulesChecker] = None,
    ):
        self._pose_model = pose_model

        self._object_model = object_model

        self._classification_model = classification_model

        self._pose_embedder = self._create_pose_embedder()

        self._detector = (
            detector
            if detector is not None
            else StateDetector(
                class_name=detect_class,
                enter_threshold=enter_threshold,
                exit_threshold=exit_threshold,
            )
        )

        self._classification_smoother = EMADictSmoothing(
            window_size=window_size, alpha=alpha
        )

        self._rules_checker = (
            rules_checker if rules_checker is not None else self._create_rules_checker()
        )

    def _create_rules_checker(self):
        return RulesChecker(
            # rules=[PersonIsAlone(), PersonNotHorizontal(), PersonNotOnFurniture()]
            rules=[PersonIsAlone(), PersonNotOnFurniture()],
        )

    def _create_pose_embedder(self):
        return PoseEmbedder(
            landmark_names=self._pose_model.landmarks_names,
            dims=self._classification_model._n_dimensions,
        )

    def _create_result_dict(
        self, classification, smooth_classification, detection, message
    ):
        return {
            "classification": classification,
            "smooth_classification": smooth_classification,
            "detection": detection,
            "message": message,
        }

    def _run(self, image):
        # detect and draw objects
        objs_results = self._object_model.predict(image)
        objs = self._object_model.results_to_object_detection_samples(
            results=objs_results
        )

        # check manual rules
        if not self._rules_checker.check(objs):
            if objs_results is not None:
                image = self._object_model.draw_results(image, objs_results)

            return (
                plot_fall_text(image, False),
                self._create_result_dict(
                    classification=None,
                    smooth_classification=None,
                    detection=0,
                    message="Manual rules failed",
                ),
            )

        # detect and draw pose
        pose_results = self._pose_model.predict(image)

        if pose_results is None:
            if objs_results is not None:
                image = self._object_model.draw_results(image, objs_results)

            return (
                plot_fall_text(image, False),
                self._create_result_dict(
                    classification=None,
                    smooth_classification=None,
                    detection=0,
                    message="Pose failed",
                ),
            )

        # pose landmarks
        pose_landmarks = self._pose_model.results_to_pose_landmarks(
            pose_results, image.shape[0], image.shape[1]
        )

        print(pose_landmarks)

        # check landmarks
        ok_landmarks = pose_landmarks[:, 2] > 0.2
        
        if not np.all(ok_landmarks):
            image = self._pose_model.draw_landmarks(image, pose_results)
            if objs_results is not None:
                image = self._object_model.draw_results(image, objs_results)

            return (
                plot_fall_text(image, False),
                self._create_result_dict(
                    classification=None,
                    smooth_classification=None,
                    detection=0,
                    message="Landmarks scores failed",
                ),
            )

        # classify pose
        classification_result = self._classification_model(pose_landmarks)

        # smooth classificaion (for videos)
        smooth_classification_result = self._classification_smoother(
            classification_result
        )

        # detect state
        detection = self._detector(smooth_classification_result)

        if objs_results is not None:
            image = self._object_model.draw_results(image, objs_results)
        if pose_results is not None:
            image = self._pose_model.draw_landmarks(image, pose_results)

        if detection == 0:
            return (
                plot_fall_text(image, False),
                self._create_result_dict(
                    classification=classification_result,
                    smooth_classification=smooth_classification_result,
                    detection=detection,
                    message="No Fall",
                ),
            )
        else:
            return (
                plot_fall_text(image, True),
                self._create_result_dict(
                    classification=classification_result,
                    smooth_classification=smooth_classification_result,
                    detection=detection,
                    message="Fall",
                ),
            )

    def __call__(self, image):
        return self._run(image)
