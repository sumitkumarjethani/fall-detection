import argparse
import logging
import os
import sys

# setting path
sys.path.append("./")
sys.path.append("../../yolov7")
from logger.logger import configure_logging
from pose.mediapipe import MediapipePoseModel
from pose.movenet import MovenetModel
from pose.yolo import YoloPoseModel
from pose.data import PoseLandmarksGenerator

logger = logging.getLogger("app")


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-images",
        help="input path to read the images from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-images",
        help="output path to save images with landmarks",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--output-file",
        help="output path to save csv file with landmarks.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model name to use.",
        type=str,
        required=True,
        choices=["mediapipe", "movenet", "yolo"],
        default="mediapipe",
    )

    parser.add_argument(
        "--max-samples",
        help="max number of samples to use from each class",
        type=int,
        required=False,
        default=None,
    )

    args = parser.parse_args()

    return args


def main():
    try:
        configure_logging()
        args = cli()

        logger.info(f"loading model")

        if args.model == "mediapipe":
            model = MediapipePoseModel()
        elif args.model == "movenet":
            model = MovenetModel()
        elif args.model == "yolo":
            model = YoloPoseModel()
        else:
            raise ValueError("model name not valid")

        generator = PoseLandmarksGenerator(
            images_in_folder=args.input_images,
            images_out_folder=args.output_images,
            csvs_out_folder=args.output_file,
            per_pose_class_limit=args.max_samples,
        )

        generator(model)

        sys.exit(0)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
