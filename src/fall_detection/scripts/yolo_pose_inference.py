import argparse
import logging
import os
import sys

# setting path
sys.path.append("./")
sys.path.append("../yolov7")
from logger.logger import configure_logging
from pose.yolo import YoloPoseModel
from pose.utils import load_image, save_image

logger = logging.getLogger("app")

import sys


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
        help="model path to use for the inference.",
        required=False,
        default="yolov7-w6-pose.pt",
    )
    args = parser.parse_args()

    return args


def main():
    try:
        configure_logging()
        args = cli()
        logger.info(f"loading model {args.model}")

        model = YoloPoseModel(model_path=args.model)

        if not os.path.exists(args.input):
            raise Exception(f"input image {args.input} not found.")

        logger.info(f"loading input image {args.input}")
        input_image = load_image(args.input)

        logger.info(f"running inference")
        pose_landmarks = model.predict(input_image)

        logger.info(f"drawing inference")
        output_image = model.draw_landmarks(input_image, pose_landmarks)

        logger.info(f"saving output image {args.output}")
        save_image(output_image, args.output)
        sys.exit(0)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
