import os


def is_directory(path):
    if os.path.exists(path) and os.path.isdir(path):
        return True
    return False


def file_exists(path):
    if os.path.exists(path) and os.path.isfile(path):
        return True
    return False
