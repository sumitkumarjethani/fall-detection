import argparse
import sys
import cv2

from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel, TFLiteMovenetModel
from fall_detection.pose.yolo import YoloPoseModel


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pose-model-name",
        "--pose-model-name",
        help="pose model to use.",
        type=str,
        choices=["mediapipe", "movenet", "yolo"],
        default="yolo",
    )
    parser.add_argument(
        "--movenet-version",
        "--movenet_version",
        help="specific movenet model to use for pose inference.",
        required=False,
        default="movenet_thunder",
        choices=[
            "movenet_lightning",
            "movenet_thunder",
            "movenet_lightning_f16.tflite",
            "movenet_thunder_f16.tflite",
            "movenet_lightning_int8.tflite",
            "movenet_thunder_int8.tflite",
        ]
    )
    parser.add_argument(
        "--yolo-pose-model-path",
        "--yolo-pose-model-path",
        help="yolo pose model path to use for the inference.",
        required=False,
        default="yolov8n-pose.pt",
    )
    return parser.parse_args()


def main():
    try:
        args = cli()
        pose_model_name = args.pose_model_name

        print(f"Loading pose model: {pose_model_name}")

        if pose_model_name == "mediapipe":
            pose_model = MediapipePoseModel()
        elif pose_model_name == "movenet":
            movenet_version = args.movenet_version
            pose_model = TFLiteMovenetModel(movenet_version) \
                if movenet_version.endswith("tflite") else MovenetModel(movenet_version)
        elif pose_model_name == "yolo":
            yolo_pose_model_path = args.yolo_pose_model_path
            pose_model = YoloPoseModel(model_path=yolo_pose_model_path)
        else:
            raise ValueError("Model input not valid")

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
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
