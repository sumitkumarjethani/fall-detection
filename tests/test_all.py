import sys
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import _is_fitted
import os

from fall_detection.utils import load_image, save_image


def check_module(modulename):
    if modulename not in sys.modules:
        return False
    return True


def test_pgk_version():
    import fall_detection

    assert fall_detection.__version__ == "0.1"


def test_import_modules():
    from fall_detection import pose
    from fall_detection import object_detection
    from fall_detection import fall
    from fall_detection import datasets

    assert check_module(datasets.__name__) == True
    assert check_module(fall.__name__) == True
    assert check_module(object_detection.__name__) == True
    assert check_module(pose.__name__) == True


def test_import_pose_models():
    from fall_detection.pose import MovenetModel
    from fall_detection.pose import MediapipePoseModel
    from fall_detection.pose import YoloPoseModel
    from fall_detection.pose import PoseModel

    assert issubclass(MovenetModel, PoseModel)
    assert issubclass(MediapipePoseModel, PoseModel)
    assert issubclass(YoloPoseModel, PoseModel)


@pytest.mark.skip(reason="too expensive to test all the time")
def test_load_pose_models():
    from fall_detection.pose import MovenetModel
    from fall_detection.pose import MediapipePoseModel
    from fall_detection.pose import YoloPoseModel

    model = MovenetModel(model_name="movenet_thunder")
    assert model._module != None
    model = MovenetModel(model_name="movenet_lightning")
    assert model._module != None
    model = MediapipePoseModel()
    assert isinstance(model, MediapipePoseModel)
    model = YoloPoseModel()
    assert model._model != None


# @pytest.mark.skip(reason="too expensive to test all the time")
def test_pose_inference_yolo():
    from fall_detection.pose import YoloPoseModel

    model = YoloPoseModel(model_path="./models/yolov8n-pose.pt")
    image_names = [
        "./data/fall-sample.png",
        "./data/fall-sample-2.jpeg",
        "./data/fall-sample-3.jpeg",
    ]
    for image_name in image_names:
        image = load_image(image_name)
        results = model.predict(image)
        output_image = model.draw_landmarks(image, results)
        save_image(
            output_image,
            os.path.join(
                os.path.dirname(image_name), "yolo-" + os.path.basename(image_name)
            ),
        )
        pose_landmarks = model.results_to_pose_landmarks(results)
        assert pose_landmarks.shape == (17, 2)


@pytest.mark.skip(reason="too expensive to test all the time")
def test_pose_inference_movenet():
    from fall_detection.pose import MovenetModel

    image = load_image("./data/fall-sample-2.jpeg")
    model = MovenetModel(model_name="movenet_thunder")
    results = model.predict(image)
    output_image = model.draw_landmarks(image, results)
    save_image(output_image, "./data/fall_sample-2-movenet-thunder-pose-inference.jpg")
    pose_landmarks = model.results_to_pose_landmarks(
        results, image.shape[0], image.shape[1]
    )
    assert pose_landmarks.shape == (17, 3)


@pytest.mark.skip(reason="too expensive to test all the time")
def test_pose_inference_mediapipe():
    from fall_detection.pose import MediapipePoseModel

    image = load_image("./data/fall-sample.png")
    model = MediapipePoseModel()
    results = model.predict(image)
    pose_landmarks = model.results_to_pose_landmarks(
        results, image.shape[0], image.shape[1]
    )
    assert pose_landmarks.shape == (33, 3)


def test_pose_embedder():
    from fall_detection.fall import PoseEmbedder
    from fall_detection.fall.embedding import COCO_POSE_KEYPOINTS

    embedder = PoseEmbedder(landmark_names=COCO_POSE_KEYPOINTS)
    landmarks = np.array(
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
    embeddings = embedder(landmarks)
    assert embeddings.shape == (25, 3)

    distances = embedder.distances(landmarks)
    assert distances.shape == (25, 3)

    angles = embedder.angles(landmarks)
    assert angles.shape == (6,)


def test_pose_classification():
    # TODO
    pass


def test_classification_rules():
    # TODO
    pass


def test_smooth_pose_classification():
    # TODO
    pass


def test_state_detection():
    # TODO
    pass


@pytest.mark.skip(reason="too expensive to test all the time")
def test_train_and_predict_pipeline():
    print("test train and predict running")
    from fall_detection.pose import MovenetModel, PoseLandmarksGenerator
    from fall_detection.fall import PoseEmbedder, EstimatorClassifier
    from fall_detection.fall import load_pose_samples_from_dir
    from fall_detection.fall.embedding import COCO_POSE_KEYPOINTS

    pose_model = MovenetModel(model_name="movenet_thunder")

    pose_sample_generator = PoseLandmarksGenerator(
        images_in_folder="./data/test_dataset",
        images_out_folder="./data/test_dataset_out",
        csvs_out_folder="./data/test_dataset_csv",
        per_pose_class_limit=4,
    )
    pose_sample_generator(pose_model=pose_model)

    embedder = PoseEmbedder(landmark_names=COCO_POSE_KEYPOINTS)

    classifier = EstimatorClassifier(
        estimator=make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
        pose_embedder=embedder,
    )

    pose_samples = load_pose_samples_from_dir(
        pose_embedder=embedder,
        n_dimensions=3,
        n_landmarks=17,
        landmarks_dir="./data/test_dataset_csv",
        file_extension="csv",
        file_separator=",",
    )
    classifier.fit(pose_samples)

    assert _is_fitted(classifier._model) == True

    image = load_image("./data/fall-sample.png")

    pose_landmarks = pose_model.predict(image)

    # TODO: revisit this api. Probably can be simplify
    pose_landmarks = pose_model.pose_landmarks_to_nparray(
        pose_landmarks, image.shape[0], image.shape[1]
    )
    prediction = classifier.predict(pose_landmarks)

    assert isinstance(prediction, dict)
    assert "fall" in prediction.keys()
    assert "no-fall" in prediction.keys()
