import tensorflow_hub as hub
import tensorflow as tf

movenet_models = {
    "movenet_lightning": {
        "url": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        "input_size": 192
    },
    "movenet_thunder": {
        "url": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
        "input_size": 256
    }
}

# Dictionary that maps from joint names to keypoint indices.
key_point_dict = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}


def get_model_url(model_name):
    return movenet_models.get(model_name, {}).get("url", None)


def get_model_input_size(model_name):
    return movenet_models.get(model_name, {}).get("input_size", None)


def get_model_names():
    return list(movenet_models.keys())


class MovenetModel:
    """
    Output is a [1, 1, 17, 3] tensor.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.input_size = get_model_input_size(model_name)
        self.model = self.load_standard_model()

    def load_standard_model(self):
        url = get_model_url(self.model_name)
        module = hub.load(url)
        return module

    def get_key_points(self, image):
        """Runs detection on an input image.
            Args:
                image: A [1, height, width, 3] tensor represents the input image pixels.
                Note that the height/width should already be resized and match the expected input resolution of the
                model before passing into this function.
            Returns:
                A [1, 1, 17, 3] float numpy array representing the predicted keypoint coordinates and scores.
        """
        model = self.model.signatures["serving_default"]
        input_size = self.input_size

        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)

        # Run model inference.
        outputs = model(input_image)

        # Output is a [1, 1, 17, 3] tensor.
        key_points_with_scores = outputs["output_0"].numpy()
        return key_points_with_scores
