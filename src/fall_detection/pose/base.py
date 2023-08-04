from abc import ABC, abstractmethod


class PoseModel(ABC):
    @abstractmethod
    def predict(self, image):
        pass

    @abstractmethod
    def draw_landmarks(self, image, pose_landmarks):
        pass

    @abstractmethod
    def pose_landmarks_to_nparray(self, pose_landmarks, height, width):
        pass

    @abstractmethod
    def results_to_pose_landmarks(self, results):
        pass
