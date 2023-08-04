import argparse
import logging
import sys
import cv2

# setting path
sys.path.append("./")
sys.path.append("../../yolov7")

from fall_detection.logger.logger import configure_logging
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel
from fall_detection.pose.yolo import YoloPoseModel

logger = logging.getLogger("app")


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        help="model name to save.",
        type=str,
        choices=["mediapipe", "movenet", "yolo"],
        default="movenet",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    configure_logging()
    args = cli()

    logger.info(f"loading model")

    if args.model == "mediapipe":
        pose_model = MediapipePoseModel()
    elif args.model == "movenet":
        pose_model = MovenetModel()
    elif args.model == "yolo":
        pose_model = YoloPoseModel()
    else:
        raise ValueError("model input not valid")

    cam = cv2.VideoCapture(0)
    while True:
        check, frame = cam.read()
        if check:
            pose_landmarks = pose_model.predict(frame.copy())
        else:
            continue
        if pose_landmarks is not None:
            pose_model.draw_landmarks(frame, pose_landmarks)

        cv2.imshow("video", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
