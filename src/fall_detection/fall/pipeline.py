"""Fall detection pipeline"""

from typing import List, Optional
from sklearn.ensemble import RandomForestClassifier
from fall_detection.fall.classification import (
    EMADictSmoothing,
    EstimatorClassifier,
    PoseClassifier,
)
from fall_detection.fall.data import load_pose_samples_from_dir
from fall_detection.fall.detection import StateDetector
from fall_detection.fall.embedding import PoseEmbedder
from fall_detection.fall.rules import (
    PersonIsAlone,
    PersonNotHorizontal,
    PersonNotOnFurniture,
    RulesChecker,
)
from fall_detection.object_detection.yolo import ObjectDetector, YoloObjectDetector
from fall_detection.pose.base import PoseModel
from fall_detection.pose.data import PoseLandmarksGenerator
from fall_detection.pose.yolo.yolo import YoloPoseModel


class Pipeline:
    def __init__(
        self,
        pose_model: PoseModel,
        object_model: ObjectDetector,
        classification_model: PoseClassifier,
        detector: Optional[StateDetector] = None,
        detect_class: str = "fall",
        enter_threshold: int = 9,
        exit_threshold: int = 5,
        window_size: int = 10,
        alpha: float = 0.2,
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
        )

    def _empty_result(self):
        return {
            "classification": None,
            "smooth_classification": None,
            "detection": 0,
        }

    def _run(self, image):
        # detect objects
        objs_results = self._object_model(image)

        objs = self._object_model.results_to_object_detection_samples(
            results=objs_results
        )

        # check manual rules
        if not self._rules_checker.check(objs):
            return self._empty_result()

        # detect pose
        pose_results = self._pose_model(image)

        if pose_results is None:
            return self._empty_result()

        pose_landmarks = self._pose_model.results_to_pose_landmarks(pose_results)

        # classify pose
        classification_result = self._classification_model(pose_landmarks)

        # smooth classificaion (for videos)
        smooth_classification_result = self._classification_smoother(
            classification_result
        )

        # detect state
        detection = self._detector(smooth_classification_result)

        return {
            "classification": classification_result,
            "smooth_classification": smooth_classification_result,
            "detection": detection,
        }

    def __call__(self, image):
        self._run(image)
