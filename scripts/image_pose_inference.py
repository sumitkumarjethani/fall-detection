import argparse
import os
import sys

from fall_detection.pose.yolo import YoloPoseModel
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel, TFLiteMovenetModel
from fall_detection.utils import load_image, save_image


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="input image to read in a jpg/png format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output path to save the image with drawn pose inference.",
        type=str,
        required=True,
    )
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
            pose_model = TFLiteMovenetModel(movenet_version)\
                if movenet_version.endswith("tflite") else MovenetModel(movenet_version)
        elif pose_model_name == "yolo":
            yolo_pose_model_path = args.yolo_pose_model_path
            pose_model = YoloPoseModel(model_path=yolo_pose_model_path)
        else:
            raise ValueError("Model input not valid")

        if not os.path.exists(args.input):
            raise Exception(f"Input image {args.input} not found.")

        print(f"Loading input image: {args.input}")
        input_image = load_image(args.input)

        print(f"Running inference")
        results = pose_model.predict(input_image)

        if results is not None:
            print(f"Drawing inference")
            output_image = pose_model.draw_landmarks(input_image, results)

            print(f"Saving output image: {args.output}")
            save_image(output_image, args.output)

        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
