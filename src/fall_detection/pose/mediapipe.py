try:
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import pose as mp_pose
except:
    raise Exception(
        "mediapipe dependency was not found. Make sure you pip installed mediapipe-requirements.txt"
    )

import numpy as np
from .base import PoseModel


class MediapipePoseModel(PoseModel):
    def predict(self, image):
        # Initialize fresh pose tracker and run it.
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
        ) as pose_tracker:
            result = pose_tracker.process(image=image)
            pose_landmarks = result.pose_landmarks
        return pose_landmarks

    def draw_landmarks(self, image, pose_landmarks):
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=pose_landmarks,
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
