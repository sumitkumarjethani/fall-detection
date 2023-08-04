import argparse
import logging
import os
import sys

# setting path
sys.path.append("./")

from fall_detection.logger.logger import configure_logging
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.utils import load_image, save_image

logger = logging.getLogger("app")


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

    args = parser.parse_args()

    return args


def main():
    try:
        configure_logging()
        args = cli()

        logger.info(f"loading model")
        model = MediapipePoseModel()

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
