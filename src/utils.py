import os


def is_directory(path):
    if os.path.isdir(path):
        return True
    return False


def path_exist(path):
    if os.path.exists(path):
        return True
    return False

