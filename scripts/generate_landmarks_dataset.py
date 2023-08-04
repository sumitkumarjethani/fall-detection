import argparse
import sys

from fall_detection.logger.logger import LoggerSingleton
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel
from fall_detection.pose.yolo import YoloPoseModel
from fall_detection.pose.data import PoseLandmarksGenerator

logger = LoggerSingleton("app").get_logger()


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
        default="yolo",
    )
    parser.add_argument(
        "-p",
        "--yolo_model_path",
        help="yolo model path to use for the inference.",
        required=False,
        default="yolov8n-pose.pt",
    )
    parser.add_argument(
        "--max-samples",
        help="max number of samples to use from each class",
        type=int,
        required=False,
        default=None,
    )
    return parser.parse_args()


def main():
    try:
        args = cli()

        logger.info(f"Loading model: {args.model}")

        if args.model == "mediapipe":
            model = MediapipePoseModel()
        elif args.model == "movenet":
            model = MovenetModel()
        elif args.model == "yolo":
            yolo_model_path = args.yolo_model_path
            model = YoloPoseModel(model_path=yolo_model_path)
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
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
