import os
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import _is_fitted
from fall_detection.fall.data import PoseSample

from fall_detection.fall.detection import StateDetector
from fall_detection.fall.pipeline import Pipeline
from fall_detection.utils import load_image
from fall_detection.fall.rules import (
    PersonIsAlone,
    PersonNotHorizontal,
    PersonNotOnFurniture,
)
from fall_detection.fall import RulesChecker
from fall_detection.object_detection import ObjectDetectionSample
from fall_detection.pose import (
    YoloPoseModel,
    PoseLandmarksGenerator,
)
from fall_detection.object_detection import YoloObjectDetector
from fall_detection.fall import PoseEmbedder, EstimatorClassifier
from fall_detection.fall import load_pose_samples_from_dir
from fall_detection.fall.embedding import COCO_POSE_KEYPOINTS


@pytest.fixture
def landmarks():
    return np.array(
        [
            [0.5, 0.5, 0.5],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
            [15, 15, 15],
            [16, 16, 16],
        ],
        dtype=float,
    )


def test_pose_embedder(landmarks):
    embedder = PoseEmbedder(landmark_names=COCO_POSE_KEYPOINTS, dims=2)

    embeddings = embedder(landmarks)
    assert embeddings.shape == (25, 2)

    distances = embedder.distances(landmarks)
    assert distances.shape == (25, 2)

    angles = embedder.angles(landmarks)
    assert angles.shape == (7,)


def test_pose_classification(landmarks):
    embedder = PoseEmbedder(landmark_names=COCO_POSE_KEYPOINTS, dims=2)
    classifier = EstimatorClassifier(
        estimator=make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
        pose_embedder=embedder,
    )

    pose_samples = []
    for i in range(10):
        if i % 2 == 0:
            pose_samples.append(
                PoseSample(
                    f"pose_sample_{i}",
                    landmarks=landmarks,
                    class_name="fall",
                    embedding=embedder(landmarks),
                )
            )
        else:
            pose_samples.append(
                PoseSample(
                    f"pose_sample_{i}",
                    landmarks=landmarks + 1,
                    class_name="nofall",
                    embedding=embedder(landmarks + 1),
                )
            )
    classifier.fit(pose_samples=pose_samples)
    assert _is_fitted(classifier._model) == True

    prediction = classifier.predict(landmarks)
    assert isinstance(prediction, dict)
    assert "fall" in prediction.keys()
    assert "nofall" in prediction.keys()


def test_classification_rules():
    rules_checker = RulesChecker(
        rules=[PersonIsAlone()],
    )
    objs = [
        ObjectDetectionSample(class_name="person", xyxy=np.ones(shape=(4,))),
    ]
    result = rules_checker(objs)

    assert 1.0 == result

    rules_checker = RulesChecker(
        rules=[PersonIsAlone()],
    )
    objs = [
        ObjectDetectionSample(class_name="person", xyxy=np.ones(shape=(4,))),
        ObjectDetectionSample(class_name="person", xyxy=np.ones(shape=(4,))),
    ]
    result = rules_checker(objs)

    assert 0.0 == result

    rules_checker = RulesChecker(
        rules=[PersonNotOnFurniture()],
    )
    objs = [
        ObjectDetectionSample(class_name="person", xyxy=np.array([2, 2, 5, 5])),
        ObjectDetectionSample(class_name="bed", xyxy=np.array([1, 1, 10, 10])),
    ]
    result = rules_checker(objs)

    assert 0.0 == result

    rules_checker = RulesChecker(
        rules=[PersonIsAlone(), PersonNotHorizontal(), PersonNotOnFurniture()],
    )
    objs = [
        ObjectDetectionSample(class_name="person", xyxy=np.ones(shape=(4,))),
        ObjectDetectionSample(class_name="couch", xyxy=np.ones(shape=(4,))),
        ObjectDetectionSample(class_name="bed", xyxy=np.ones(shape=(4,))),
    ]
    result = rules_checker(objs)

    assert 0.67 == np.round(result, 2)

    rules_checker = RulesChecker(
        rules=[PersonIsAlone(), PersonNotHorizontal(), PersonNotOnFurniture()],
        weights=[1 / 4, 2 / 4, 1 / 4],
    )
    result = rules_checker(objs)

    assert result == 0.75


def test_smooth_pose_classification():
    # TODO
    pass


def test_state_detection():
    # TODO
    pass


def test_landmarks_generator():
    pose_model = YoloPoseModel(model_path="./models/yolov8n-pose.pt")

    pose_sample_generator = PoseLandmarksGenerator(
        images_in_folder="./tests/test_data/test_dataset",
        images_out_folder="./tests/test_data/test_dataset_out",
        csvs_out_folder="./tests/test_data/test_dataset_csv",
        per_pose_class_limit=4,
    )

    pose_sample_generator(pose_model=pose_model)

    assert os.path.exists("./tests/test_data/test_dataset_out")
    assert os.path.exists("./tests/test_data/test_dataset_csv")
    assert "fall" in os.listdir("./tests/test_data/test_dataset_out")
    assert "no-fall" in os.listdir("./tests/test_data/test_dataset_out")
    assert "fall.csv" in os.listdir("./tests/test_data/test_dataset_csv")
    assert "no-fall.csv" in os.listdir("./tests/test_data/test_dataset_csv")


# @pytest.mark.skip(reason="too expensive to test all the time")
def test_fall_pipeline():
    pose_model = YoloPoseModel(model_path="./models/yolov8n-pose.pt")

    pose_sample_generator = PoseLandmarksGenerator(
        images_in_folder="./tests/test_data/test_dataset",
        images_out_folder="./tests/test_data/test_dataset_out",
        csvs_out_folder="./tests/test_data/test_dataset_csv",
        per_pose_class_limit=4,
    )

    pose_sample_generator(pose_model=pose_model)

    embedder = PoseEmbedder(landmark_names=COCO_POSE_KEYPOINTS, dims=2)

    classifier = EstimatorClassifier(
        estimator=make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
        pose_embedder=embedder,
    )

    pose_samples = load_pose_samples_from_dir(
        pose_embedder=embedder,
        n_landmarks=17,
        landmarks_dir="./tests/test_data/test_dataset_csv",
        file_extension="csv",
        file_separator=",",
    )

    classifier.fit(pose_samples)

    object_model = YoloObjectDetector(model_path="./models/yolov8n.pt")

    detector = StateDetector(class_name="fall", enter_threshold=6, exit_threshold=4)

    pipeline = Pipeline(
        pose_model=pose_model,
        object_model=object_model,
        classification_model=classifier,
        detector=detector,
    )

    image_names = [
        "./tests/test_data/fall-sample.png",
        "./tests/test_data/fall-sample-2.jpeg",
        "./tests/test_data/fall-sample-3.jpeg",
    ]

    for image_name in image_names:
        image = load_image(image_name)
        results = pipeline(image)
        assert results.shape == image.shape
