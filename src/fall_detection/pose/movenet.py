"""
Script to download and load movenet models from tenworflow hub.
"""
from abc import ABC
from dataclasses import dataclass
import os
from typing import List

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches


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

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
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


def _keypoints_and_edges_for_display(
    keypoints_with_scores, height, width, keypoint_threshold=0.11
):
    """Returns high confidence keypoints and edges for visualization.

    Args:
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      height: height of the image in pixels.
      width: width of the image in pixels.
      keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
      A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1
        )
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :
        ]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (
                kpts_scores[edge_pair[0]] > keypoint_threshold
                and kpts_scores[edge_pair[1]] > keypoint_threshold
            ):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


# def draw_prediction_on_image(
#     image,
#     keypoints_with_scores,
#     crop_region=None,
#     output_image_height=None,
# ):
#     """Draws the keypoint predictions on image.

#     Args:
#       image: A numpy array with shape [height, width, channel] representing the
#         pixel values of the input image.
#       keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
#         the keypoint coordinates and scores returned from the MoveNet model.
#       crop_region: A dictionary that defines the coordinates of the bounding box
#         of the crop region in normalized coordinates (see the init_crop_region
#         function below for more detail). If provided, this function will also
#         draw the bounding box on the image.
#       output_image_height: An integer indicating the height of the output image.
#         Note that the image aspect ratio will be the same as the input image.

#     Returns:
#       A numpy array with shape [out_height, out_width, channel] representing the
#       image overlaid with keypoint predictions.
#     """
#     height, width, channel = image.shape
#     print(height, width)
#     aspect_ratio = float(width) / height
#     fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
#     # To remove the huge white borders
#     fig.tight_layout(pad=0)
#     ax.margins(0)
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     plt.axis("off")

#     im = ax.imshow(image)
#     line_segments = LineCollection([], linewidths=(4), linestyle="solid")
#     ax.add_collection(line_segments)
#     # Turn off tick labels
#     scat = ax.scatter([], [], s=60, color="#FF1493", zorder=3)

#     (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(
#         keypoints_with_scores, height, width
#     )

#     line_segments.set_segments(keypoint_edges)
#     line_segments.set_color(edge_colors)
#     if keypoint_edges.shape[0]:
#         line_segments.set_segments(keypoint_edges)
#         line_segments.set_color(edge_colors)
#     if keypoint_locs.shape[0]:
#         scat.set_offsets(keypoint_locs)

#     if crop_region is not None:
#         xmin = max(crop_region["x_min"] * width, 0.0)
#         ymin = max(crop_region["y_min"] * height, 0.0)
#         rec_width = min(crop_region["x_max"], 0.99) * width - xmin
#         rec_height = min(crop_region["y_max"], 0.99) * height - ymin
#         rect = patches.Rectangle(
#             (xmin, ymin),
#             rec_width,
#             rec_height,
#             linewidth=1,
#             edgecolor="b",
#             facecolor="none",
#         )
#         ax.add_patch(rect)

#     fig.canvas.draw()
#     print(fig.canvas.get_width_height())
#     image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     image_from_plot = image_from_plot.reshape(
#         fig.canvas.get_width_height()[::-1] + (3,)
#     )
#     plt.close(fig)
#     if output_image_height is not None:
#         output_image_width = int(output_image_height / height * width)
#         image_from_plot = cv2.resize(
#             image_from_plot,
#             dsize=(output_image_width, output_image_height),
#             interpolation=cv2.INTER_CUBIC,
#         )
#     return image_from_plot


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


def draw_prediction_on_image(image, keypoints, threshold=0.11, input_size=1280):
    """Draws the keypoints on a image frame"""
    # Denormalize the coordinates : multiply the normalized coordinates by the input_size(width,height)
    denormalized_coordinates = np.squeeze(
        np.multiply(keypoints, [input_size, input_size, 1])
    )
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
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    return input_image


def get_model_url(model_name) -> str:
    return _movenet_models.get(model_name, None)


def get_model_input_size(model_name) -> int:
    return _movenet_input_size.get(model_name, None)


def get_model_names() -> List[str]:
    return _model_names


class MovenetModel:
    """
    Output is a [1, 1, 17, 3] tensor.
    """

    def __init__(self, model_name: str):
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
        image = tf.image.resize_with_pad(image, 1280, 1280)
        return draw_prediction_on_image(image.numpy(), pose_landmarks, input_size=1280)

    # def draw_landmarks(self, image, pose_landmarks):
    #     # Visualize the predictions with image.
    #     display_image = tf.expand_dims(image, axis=0)
    #     display_image = tf.cast(
    #         tf.image.resize_with_pad(display_image, 1200, 1200), dtype=tf.int32
    #     )
    #     print(display_image.shape)
    #     output_overlay = draw_prediction_on_image(
    #         np.squeeze(display_image.numpy(), axis=0), pose_landmarks
    #     )

    #     return output_overlay


class TFLiteMovenetModel:
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

    # def draw_landmarks(self, image, pose_landmarks):
    #     # Visualize the predictions with image.
    #     display_image = tf.expand_dims(image, axis=0)
    #     display_image = tf.cast(
    #         tf.image.resize_with_pad(display_image, 1200, 1200), dtype=tf.int32
    #     )
    #     print(display_image.shape)
    #     output_overlay = draw_prediction_on_image(
    #         np.squeeze(display_image.numpy(), axis=0), pose_landmarks
    #     )
    #     return output_overlay


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


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)
    return image


def save_image(image, output_path):
    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
