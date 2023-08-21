import pytest
import os

from fall_detection.utils import load_image, save_image


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
        "./tests/test_data/fall-sample.png",
        "./tests/test_data/fall-sample-2.jpeg",
        "./tests/test_data/fall-sample-3.jpeg",
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
        assert pose_landmarks.shape == (17, 3)


@pytest.mark.skip(reason="too expensive to test all the time")
def test_pose_inference_movenet_thunder():
    from fall_detection.pose import MovenetModel

    model = MovenetModel(model_name="movenet_thunder")
    image_names = [
        "./tests/test_data/fall-sample.png",
        "./tests/test_data/fall-sample-2.jpeg",
        "./tests/test_data/fall-sample-3.jpeg",
    ]
    for image_name in image_names:
        image = load_image(image_name)
        results = model.predict(image)
        output_image = model.draw_landmarks(image, results)
        save_image(
            output_image,
            os.path.join(
                os.path.dirname(image_name),
                "movenet-thunder-" + os.path.basename(image_name),
            ),
        )
        pose_landmarks = model.results_to_pose_landmarks(
            results, image.shape[0], image.shape[1]
        )
        assert pose_landmarks.shape == (17, 3)


@pytest.mark.skip(reason="too expensive to test all the time")
def test_pose_inference_mediapipe():
    from fall_detection.pose import MediapipePoseModel

    model = MediapipePoseModel()

    image_names = [
        "./tests/test_data/fall-sample.png",
        "./tests/test_data/fall-sample-2.jpeg",
        "./tests/test_data/fall-sample-3.jpeg",
    ]
    for image_name in image_names:
        image = load_image(image_name)
        results = model.predict(image)
        if results is None:
            continue
        output_image = model.draw_landmarks(image, results)
        save_image(
            output_image,
            os.path.join(
                os.path.dirname(image_name),
                "mediapipe-" + os.path.basename(image_name),
            ),
        )
        pose_landmarks = model.results_to_pose_landmarks(
            results, image.shape[0], image.shape[1]
        )

        assert pose_landmarks.shape == (33, 3)
