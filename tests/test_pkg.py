import sys
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import _is_fitted
import os
from fall_detection.fall.detection import StateDetector
from fall_detection.fall.pipeline import Pipeline

from fall_detection.utils import load_image, save_image


def check_module(modulename):
    if modulename not in sys.modules:
        return False
    return True


def test_pgk_version():
    import fall_detection

    assert fall_detection.__version__ == "0.1"


def test_import_modules():
    from fall_detection import pose
    from fall_detection import object_detection
    from fall_detection import fall
    from fall_detection import datasets

    assert check_module(datasets.__name__) == True
    assert check_module(fall.__name__) == True
    assert check_module(object_detection.__name__) == True
    assert check_module(pose.__name__) == True
