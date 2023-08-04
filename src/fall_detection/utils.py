import os
import cv2


def load_image(image_path):
    return cv2.imread(image_path)


def save_image(image, image_path):
    return cv2.imwrite(image_path, image)


def is_directory(path: str) -> bool:
    if os.path.exists(path) and os.path.isdir(path):
        return True
    return False


def file_exists(path: str) -> bool:
    if os.path.exists(path) and os.path.isfile(path):
        return True
    return False


def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()
