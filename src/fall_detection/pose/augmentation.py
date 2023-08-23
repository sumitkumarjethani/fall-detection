from .base import PoseAugmentation
import cv2


class HorizontalFlip(PoseAugmentation):
    def apply(self, image):
        return cv2.flip(image, 1)
    
    def get_pose_augmentaion_name(self):
        return "horizontal_flip"


class Rotate(PoseAugmentation):
    def __init__(self, degrees=10):
        self.degrees = degrees

    def apply(self, image):
        image_height, image_width, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), self.degrees, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height))
        return rotated_image
    
    def get_pose_augmentaion_name(self):
        return f"rotate_{self.degrees}"


class Zoom(PoseAugmentation):
    def __init__(self, zoom_factor=1.1):
        self.zoom_factor = zoom_factor
    
    def apply(self, image):
        image_height, image_width, _ = image.shape
        zoom_matrix = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), 0, self.zoom_factor)
        zoomed_image = cv2.warpAffine(image, zoom_matrix, (image_width, image_height))
        return zoomed_image
    
    def get_pose_augmentaion_name(self):
        return f"zoom_{self.zoom_factor}"