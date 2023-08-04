import sys
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import _is_fitted

from fall_detection.utils import load_image


def check_module(modulename):
    if modulename not in sys.modules:
        print("You have not imported the {} module".format(modulename))
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


def test_full_import_modules():
    from fall_detection.logger.logger import configure_logging
    from fall_detection.pose.mediapipe import MediapipePoseModel
    from fall_detection.pose.movenet import MovenetModel
    from fall_detection.pose.yolo import YoloPoseModel
    from fall_detection.pose.data import PoseLandmarksGenerator

    assert True


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

    # from fall_detection.pose import YoloPoseModel

    model = MovenetModel(model_name="movenet_thunder")
    assert model._module != None
    model = MovenetModel(model_name="movenet_lightning")
    assert model._module != None
    model = MediapipePoseModel()
    assert isinstance(model, MediapipePoseModel)

    # TODO: test when new yolov8 added
    # model = YoloPoseModel(model_path="yolov7-w6-pose.pt")
    # assert model._model != None


def test_load_fall_module_components():
    from fall_detection.fall import (
        KnnPoseClassifier,
        EMADictSmoothing,
        EstimatorClassifier,
        PoseSample,
        load_pose_samples_from_dir,
        StateDetector,
        PoseEmbedder,
        PoseClassificationVisualizer,
    )

    assert True


def test_train_and_predict_pipeline():
    print("test train and predict running")
    from fall_detection.pose import MovenetModel, PoseLandmarksGenerator
    from fall_detection.fall import PoseEmbedder, EstimatorClassifier, PoseSample
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
