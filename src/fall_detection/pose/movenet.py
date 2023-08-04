"""
Script to download and load movenet models from tenworflow hub.
"""
import os
from typing import List, Literal, get_args, Tuple
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from .base import PoseModel

_movenet_models = {
    "movenet_lightning_f16.tflite": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
    "movenet_thunder_f16.tflite": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
    "movenet_lightning_int8.tflite": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
    "movenet_thunder_int8.tflite": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite",
    "movenet_lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
    "movenet_thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
}

_movenet_input_size = {
    "movenet_lightning_f16.tflite": 192,
    "movenet_thunder_f16.tflite": 256,
    "movenet_lightning_int8.tflite": 192,
    "movenet_thunder_int8.tflite": 256,
    "movenet_lightning": 192,
    "movenet_thunder": 256,
}

_model_names = [
    "movenet_lightning",
    "movenet_thunder",
    "movenet_lightning_f16.tflite",
    "movenet_thunder_f16.tflite",
    "movenet_lightning_int8.tflite",
    "movenet_thunder_int8.tflite",
]


# VALID_ARGUMENTS: Tuple[_model_names, ...] = get_args(_model_names)

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

cyan = (255, 255, 0)
magenta = (255, 0, 255)
EDGE_COLORS = {
    (0, 1): magenta,
    (0, 2): cyan,
    (1, 3): magenta,
    (2, 4): cyan,
    (0, 5): magenta,
    (0, 6): cyan,
    (5, 7): magenta,
    (7, 9): cyan,
    (6, 8): magenta,
    (8, 10): cyan,
    (5, 6): magenta,
    (5, 11): cyan,
    (6, 12): magenta,
    (11, 12): cyan,
    (11, 13): magenta,
    (13, 15): cyan,
    (12, 14): magenta,
    (14, 16): cyan,
}


def _draw_edges(denormalized_coordinates, image, threshold=0.11):
    """
    Draws the edges on a image frame
    """

    # Iterate through the edges
    for edge, color in EDGE_COLORS.items():
        # Get the dict value associated to the actual edge
        p1, p2 = edge
        # Get the points
        y1, x1, confidence_1 = denormalized_coordinates[p1]
        y2, x2, confidence_2 = denormalized_coordinates[p2]
        # Draw the line from point 1 to point 2, the confidence > threshold
        if (confidence_1 > threshold) & (confidence_2 > threshold):
            cv2.line(
                img=image,
                pt1=(int(x1), int(y1)),
                pt2=(int(x2), int(y2)),
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,  # Gives anti-aliased (smoothed) line which looks great for curves
            )
    return image


def get_affine_transform_to_fixed_sizes_with_padding(size, new_sizes):
    width, height = new_sizes
    scale = min(height / float(size[1]), width / float(size[0]))
    M = np.float32([[scale, 0, 0], [0, scale, 0]])
    M[0][2] = (width - scale * size[0]) / 2
    M[1][2] = (height - scale * size[1]) / 2
    return M


def denormalize_keypoints(keypoints, height, width, orig_size=256):
    w, h = width, height
    keypoints_with_scores = np.squeeze(keypoints)
    orig_w, orig_h = w, h
    M = get_affine_transform_to_fixed_sizes_with_padding(
        (orig_w, orig_h), (orig_size, orig_size)
    )
    M = np.vstack((M, [0, 0, 1]))
    M_inv = np.linalg.inv(M)[:2]
    xy_keypoints = keypoints_with_scores[:, :2] * orig_size
    xy_keypoints = cv2.transform(np.array([xy_keypoints]), M_inv)[0]
    keypoints_with_scores = np.hstack((xy_keypoints, keypoints_with_scores[:, 2:]))
    denormalized_coordinates = keypoints_with_scores
    return denormalized_coordinates


def draw_prediction_on_image(image, keypoints, threshold=0.11, normalized=True):
    """Draws the keypoints on a image frame"""
    # Denormalize the coordinates : multiply the normalized coordinates by the input_size(width,height)
    if normalized:
        keypoints_with_scores = np.squeeze(keypoints)
        w, h = image.shape[0], image.shape[1]
        orig_w, orig_h = w, h
        M = get_affine_transform_to_fixed_sizes_with_padding(
            (orig_w, orig_h), (256, 256)
        )
        M = np.vstack((M, [0, 0, 1]))
        M_inv = np.linalg.inv(M)[:2]
        xy_keypoints = keypoints_with_scores[:, :2] * 256
        xy_keypoints = cv2.transform(np.array([xy_keypoints]), M_inv)[0]
        keypoints_with_scores = np.hstack((xy_keypoints, keypoints_with_scores[:, 2:]))
        denormalized_coordinates = keypoints_with_scores
    else:
        denormalized_coordinates = keypoints

    # Iterate through the points
    for keypoint in denormalized_coordinates:
        # Unpack the keypoint values : y, x, confidence score
        keypoint_y, keypoint_x, keypoint_confidence = keypoint
        if keypoint_confidence > threshold:
            """ "
            Draw the circle
            Note : A thickness of -1 px will fill the circle shape by the specified color.
            """
            cv2.circle(
                img=image,
                center=(int(keypoint_x), int(keypoint_y)),
                radius=4,
                color=(255, 0, 0),
                thickness=-1,
            )
    return _draw_edges(
        denormalized_coordinates,
        image,
    )


def _preprocess_image_for_movenet(image, input_size):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize_with_pad(image, input_size, input_size)
    return image


def get_model_url(model_name) -> str:
    return _movenet_models.get(model_name, None)


def get_model_input_size(model_name) -> int:
    return _movenet_input_size.get(model_name, None)


def get_model_names() -> List[str]:
    return _model_names


class MovenetModel(PoseModel):
    """
    Output is a [1, 1, 17, 3] tensor.
    """

    def __init__(self, model_name: Literal["movenet_lightning", "movenet_thunder"]):
        self.model_name = model_name
        self._module = self._load_standard_model(model_name)
        self._input_size = get_model_input_size(model_name)

    def _load_standard_model(self, model_name: str):
        url = get_model_url(model_name)
        module = hub.load(url)
        return module

    def __call__(self, image):
        image = _preprocess_image_for_movenet(image, self._input_size)
        model = self._module.signatures["serving_default"]
        image = tf.cast(image, dtype=tf.int32)
        outputs = model(image)
        keypoints_with_scores = outputs["output_0"].numpy()
        return keypoints_with_scores

    def predict(self, image):
        return self.__call__(image)

    def draw_landmarks(self, image, pose_landmarks):
        # image = tf.image.resize_with_pad(image, 1280, 1280)
        return draw_prediction_on_image(image, pose_landmarks)

    def pose_landmarks_to_nparray(self, pose_landmarks, height, width):
        pose_landmarks = np.squeeze(np.multiply(pose_landmarks, [width, height, 1]))
        return pose_landmarks

    def results_to_pose_landmarks(self, results, height, width):
        if "movenet_thunder" == self.model_name:
            return denormalize_keypoints(results, height, width, 256)
        else:
            return denormalize_keypoints(results, height, width, 192)


class TFLiteMovenetModel(PoseModel):
    def __init__(self, model_name: str, output_dir: str = "."):
        self.model_name = model_name
        self._interpreter = self._load_tflite_model(model_name, output_dir)
        self._input_size = get_model_input_size(model_name)

    def _load_tflite_model(self, model_name: str, output_dir: str):
        url = get_model_url(model_name)

        output_path = os.path.join(output_dir, model_name)
        if not os.path.exists(output_path):
            os.system(f"wget -q -O model.tflite {url}")

        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter

    def __call__(self, image):
        image = _preprocess_image_for_movenet(image, self._input_size)
        image = tf.cast(image, dtype=tf.uint8)
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        self._interpreter.set_tensor(input_details[0]["index"], image.numpy())
        # Invoke inference.
        self._interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = self._interpreter.get_tensor(output_details[0]["index"])
        return keypoints_with_scores

    def predict(self, image):
        return self.__call__(image)

    def draw_landmarks(self, image, pose_landmarks):
        return draw_prediction_on_image(
            image, pose_landmarks, input_size=self._input_size
        )


def load_tflite_model(model_name: str, output_dir: str):
    url = get_model_url(model_name)
    if not url:
        raise ValueError("Unsupported model name: %s" % model_name)

    output_path = os.path.join(output_dir, model_name)
    if not os.path.exists(output_path):
        os.system(f"wget -q -O model.tflite {url}")

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter


def load_standard_model(model_name: str):
    url = get_model_url(model_name)
    module = hub.load(url)
    return module
