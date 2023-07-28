import os


def is_directory(path: str) -> bool:
    if os.path.exists(path) and os.path.isdir(path):
        return True
    return False


def file_exists(path: str) -> bool:
    if os.path.exists(path) and os.path.isfile(path):
        return True
    return False
