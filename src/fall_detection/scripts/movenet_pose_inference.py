import argparse
import logging
import os
import sys

# setting path
sys.path.append("./")

from logger.logger import configure_logging
from pose.movenet import MovenetModel, TFLiteMovenetModel
from pose.utils import save_image, load_image


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
    parser.add_argument(
        "-m",
        "--model",
        help="model name to use for the inference.",
        required=False,
        default="movenet_thunder",
    )
    args = parser.parse_args()

    return args


def main():
    try:
        configure_logging()
        args = cli()
        logger.info(f"loading model {args.model}")
        if "tflite" in args.model:
            model = TFLiteMovenetModel(args.model, ".")
        else:
            model = MovenetModel(args.model)
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
