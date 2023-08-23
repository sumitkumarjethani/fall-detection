try:
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import pose as mp_pose
except:
    raise Exception(
        "mediapipe dependency was not found. Make sure you pip installed mediapipe-requirements.txt"
    )

import numpy as np
import cv2
from .base import PoseModel, BLAZE_POSE_KEYPOINTS

def _preprocess_image_for_mediapipe(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class MediapipePoseModel(PoseModel):
    def predict(self, image):
        # Preprocess image for mediapipe
        image = _preprocess_image_for_mediapipe(image)

        # Initialize fresh pose tracker and run it.
        with mp_pose.Pose(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            model_complexity=2,
        ) as pose_tracker:
            result = pose_tracker.process(image=image)
            pose_landmarks = result.pose_landmarks
        return pose_landmarks

    def draw_landmarks(self, image, results):
        if results is not None:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results,
                connections=mp_pose.POSE_CONNECTIONS,
            )
        return image

    def pose_landmarks_to_nparray(self, pose_landmarks, height, width):
        pose_landmarks = np.array(
            [
                [
                    lmk.x * width,
                    lmk.y * height,
                    lmk.z * width,
                ]
                for lmk in pose_landmarks.landmark
            ],
            dtype=np.float32,
        )
        return pose_landmarks

    def results_to_pose_landmarks(self, results, height, width):
        pose_landmarks = np.array(
            [
                [
                    lmk.x * width,
                    lmk.y * height,
                    lmk.z * width,
                ]
                for lmk in results.landmark
            ],
            dtype=np.float32,
        )
        return pose_landmarks

    @property
    def landmarks_names(self):
        return BLAZE_POSE_KEYPOINTS
