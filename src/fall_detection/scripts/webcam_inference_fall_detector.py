import argparse
import logging
import os
import sys
import cv2
import numpy as np
import tqdm

# setting path
sys.path.append("./")

from logger.logger import configure_logging

from fall.classification import EMADictSmoothing
from fall.detection import StateDetector
from fall.plot import PoseClassificationVisualizer
from pose.mediapipe import MediapipePoseModel
from pose.movenet import MovenetModel
import pickle

logger = logging.getLogger("app")


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--pose-model",
        help="model name to save.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--classification-model",
        help="model name to save.",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def run_inference(
    pose_model,
    pose_classifier,
    pose_classification_smoother,
    fall_detector,
    frame,
):
    # Run pose tracker.
    # input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

    pose_landmarks = pose_model.predict(image=frame)

    if pose_landmarks is not None:
        pose_model.draw_landmarks(
            image=frame,
            pose_landmarks=pose_landmarks,
        )

    if pose_landmarks is not None:
        # Get landmarks.
        frame_height, frame_width = (
            frame.shape[0],
            frame.shape[1],
        )
        pose_landmarks = pose_model.pose_landmarks_to_nparray(
            pose_landmarks, frame_height, frame_width
        )

        # Classify the pose on the current frame.
        pose_classification = pose_classifier(pose_landmarks)

        # Smooth classification using EMA.
        pose_classification_filtered = pose_classification_smoother(pose_classification)

        fall_detection = fall_detector(pose_classification_filtered)
    else:
        pose_classification = None
        pose_classification_filtered = pose_classification_smoother(dict())
        pose_classification_filtered = None
        fall_detection = fall_detector.state

    print(pose_classification_filtered)

    cv2.putText(
        frame,
        str(pose_classification_filtered),
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_4,
    )

    return frame


import cv2

if __name__ == "__main__":
    args = cli()

    cam = cv2.VideoCapture(0)
    if args.pose_model == "mediapipe":
        pose_model = MediapipePoseModel()
    else:
        pose_model = MovenetModel()
    with open(f"{args.classification_model}", "rb") as f:
        pose_classifier = pickle.load(f)

    # Initialize EMA smoothing.
    pose_classification_smoother = EMADictSmoothing(window_size=10, alpha=0.2)

    # Initialize counter.
    fall_detector = StateDetector(
        class_name="Fall", enter_threshold=8, exit_threshold=4
    )

    while True:
        check, frame = cam.read()
        frame = run_inference(
            pose_model,
            pose_classifier,
            pose_classification_smoother,
            fall_detector,
            frame,
        )
        cv2.imshow("video", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
