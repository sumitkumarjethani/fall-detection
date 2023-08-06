import argparse
import sys
import cv2

from fall_detection.logger.logger import LoggerSingleton
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel
from fall_detection.pose.yolo import YoloPoseModel


logger = LoggerSingleton("app").get_logger()


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        help="model name to save.",
        type=str,
        choices=["mediapipe", "movenet", "yolo"],
        default="yolo",
    )

    parser.add_argument(
        "-p",
        "--yolo_model_path",
        help="yolo model path to use for the inference.",
        required=False,
        default="yolov8n-pose.pt",
    )

    return parser.parse_args()


def main():
    try:
        args = cli()
        model = args.model

        logger.info(f"loading model: {model}")

        if model == "mediapipe":
            pose_model = MediapipePoseModel()
        elif model == "movenet":
            pose_model = MovenetModel()
        elif model == "yolo":
            yolo_model_path = args.yolo_model_path
            pose_model = YoloPoseModel(model_path=yolo_model_path)
        else:
            raise ValueError("model input not valid")

        cam = cv2.VideoCapture(0)
        while True:
            check, frame = cam.read()

            if check:
                results = pose_model.predict(frame)
            else:
                continue

            if results is not None:
                frame = pose_model.draw_landmarks(frame, results)

            cv2.imshow("video", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
