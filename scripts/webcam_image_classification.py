import pickle
import cv2
import numpy as np
import tensorflow as tf

from fall_detection.fall.classification import EMADictSmoothing
from fall_detection.object_detection import YoloObjectDetector
from fall_detection.pose import YoloPoseModel


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ImageClassificationModel:
    def __init__(self, model_path: str, input_size: int = 224):
        self._model = self._load_model(model_path)
        self._input_size = input_size

    def _load_model(self, path: str):
        model = tf.keras.models.load_model(path)
        return model

    def _preprocess_image(self, image):
        image = cv2.resize(image, (self.input_size, self.input_size))
        return image

    def predict(self, image):
        image = self._preprocess_image(image)
        logits = self._model.predict(np.expand_dims(image, axis=0))
        preds = sigmoid(logits[0][0])
        return {
            "Fall": round(10 * preds, 2),
            "NoFall": round(10 * (1 - preds), 2),
        }

    def draw_prediction(self, image, results):
        text = f"Prediction: {results}"
        coordinates = (20, 60)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        color = (0, 0, 0)
        thickness = 2
        image = cv2.putText(
            image, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA
        )
        return image


def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    # image = image.astype(np.float32) / 255.0
    return image


def draw_prediction(image, prediction):
    text = f"Prediction: {prediction}"
    coordinates = (20, 60)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color = (0, 0, 0)
    thickness = 2
    image = cv2.putText(
        image, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA
    )
    return image


def run_inference(model, image, threshold=0.5):
    image_processed = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(np.expand_dims(image_processed, axis=0))
    scaled_preds = sigmoid(prediction[0][0])
    result = {
        "Fall": round(10 * scaled_preds, 2),
        "NoFall": round(10 * (1 - scaled_preds), 2),
    }
    print(result)

    return result


def load_pickled_model(path):
    with open(path, "rb") as f:
        pose_classifier = pickle.load(f)
    return pose_classifier


if __name__ == "__main__":
    # image_model = load_model("../models/fall-image-classification.keras")
    image_model = load_model("../models/fall-image-classification-2.keras")

    # object_model = YoloObjectDetector("../models/yolov8n.pt")
    # pose_model = YoloPoseModel("../models/yolov8n-pose.pt")
    smoother = EMADictSmoothing(window_size=10, alpha=0.2)

    # pose_classifier = load_pickled_model("../models/yolo_estimator_model.pkl")

    # cam = cv2.VideoCapture(0)
    cam = cv2.VideoCapture(
        "rtsp://falldetection:falldetection@192.168.1.133:554/stream1"
    )
    while True:
        check, frame = cam.read()
        if not check:
            continue

        # obj_results = object_model.predict(frame)

        # objs = object_model.results_to_object_detection_samples(obj_results)

        # if "person" in [obj.class_name for obj in objs]:
        #     img_preds = run_inference(image_model, frame)
        # else:
        #     img_preds = {
        #         "Fall": 0,
        #         "NoFall": 10,
        #     }

        img_preds = run_inference(image_model, frame)
        smooth_preds = smoother(img_preds)

        # pose_results = pose_model.predict(frame)

        # if pose_results is not None:
        #     pose_landmarks = pose_model.results_to_pose_landmarks(
        #         pose_results, (frame.shape[0], frame.shape[1])
        #     )

        # pose_preds = pose_classifier(pose_landmarks)

        # smooth_preds = smoother(pose_preds)

        # if pose_results is not None:
        # frame = pose_model.draw_landmarks(frame, pose_results)

        # if obj_results is not None:
        # frame = object_model.draw_results(frame, obj_results)

        frame = draw_prediction(frame, smooth_preds)

        key = cv2.waitKey(1)
        if key == 27:
            break

        cv2.imshow("video", frame)

    cam.release()
    cv2.destroyAllWindows()
