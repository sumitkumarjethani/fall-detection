import pytest
from fall_detection.utils import load_image, save_image
from fall_detection.object_detection import (
    YoloObjectDetector,
    ObjectDetectionSample,
)


@pytest.mark.skip(reason="too expensive to test all the time")
def test_object_detection_yolo():
    model = YoloObjectDetector(model_path="./models/yolov8n.pt")
    image = load_image("./tests/test_data/fall-sample.png")
    results = model.predict(image)
    assert results is not None
    output_image = model.draw_results(image, results)
    save_image(output_image, "./tests/test_data/fall-sample-object-detection-yolo.png")
    odsamples = model.results_to_object_detection_samples(results)
    assert isinstance(odsamples[0], ObjectDetectionSample)
