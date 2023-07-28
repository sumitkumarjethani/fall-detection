from abc import ABC, abstractmethod


class PoseModel(ABC):
    @abstractmethod
    def predict(self, image):
        pass

    @abstractmethod
    def draw_landmarks(self, image, pose_landmarks):
        pass
