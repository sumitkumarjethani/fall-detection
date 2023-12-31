from abc import ABC, abstractmethod


class PoseModel(ABC):
    @abstractmethod
    def predict(self, image):
        pass

    @abstractmethod
    def draw_landmarks(self, image, results):
        pass

    @abstractmethod
    def results_to_pose_landmarks(self, results, height=None, width=None):
        pass

    @property
    def landmarks_names(self):
        raise NotImplementedError()


class PoseAugmentation(ABC):
    @abstractmethod
    def apply(self, image):
        pass

    @abstractmethod
    def get_pose_augmentaion_name(self):
        pass

    def __call__(self, image):
        return self.apply(image)


# depending on the landmarks shape, use one or other landmark names:
BLAZE_POSE_KEYPOINTS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky_1",
    "right_pinky_1",
    "left_index_1",
    "right_index_1",
    "left_thumb_2",
    "right_thumb_2",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

COCO_POSE_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
