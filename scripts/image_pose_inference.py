import argparse
import os
import sys

from fall_detection.logger.logger import LoggerSingleton
from fall_detection.pose.yolo import YoloPoseModel
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel, TFLiteMovenetModel
from fall_detection.utils import load_image, save_image

logger = LoggerSingleton("app").get_logger()


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="input image to read the image in a jpg/png format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output path to save the image with draw inference.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="pose model to use.",
        type=str,
        choices=["mediapipe", "movenet", "yolo"],
        default="yolo",
    )
    parser.add_argument(
        "-mv",
        "--movenet_version",
        help="specific movenet model name to use for pose inference.",
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
            movenet_version = args.movenet_version
            pose_model = TFLiteMovenetModel(movenet_version)\
                if movenet_version.endswith("tflite") else MovenetModel(movenet_version)
        elif model == "yolo":
            yolo_model_path = args.yolo_model_path
            pose_model = YoloPoseModel(model_path=yolo_model_path)
        else:
            raise ValueError("model input not valid")

        if not os.path.exists(args.input):
            raise Exception(f"input image {args.input} not found.")

        logger.info(f"loading input image {args.input}")
        input_image = load_image(args.input)

        logger.info(f"running inference")
        results = pose_model.predict(input_image)

        if results is not None:
            logger.info(f"drawing inference")
            output_image = pose_model.draw_landmarks(input_image, results)

            logger.info(f"saving output image {args.output}")
            save_image(output_image, args.output)

        sys.exit(0)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
