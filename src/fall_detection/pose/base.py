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
