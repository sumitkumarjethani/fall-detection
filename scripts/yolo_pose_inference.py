import argparse
import os
import sys

# setting path
sys.path.append("./")
from fall_detection.logger.logger import LoggerSingleton
from fall_detection.pose.yolo import YoloPoseModel
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
        help="model path to use for the inference.",
        required=False,
        default="yolov8n-pose.pt",
    )
    args = parser.parse_args()

    return args


def main():
    try:
        args = cli()
        logger.info(f"loading model {args.model}")

        model = YoloPoseModel(model_path=args.model)

        if not os.path.exists(args.input):
            raise Exception(f"input image {args.input} not found.")

        logger.info(f"loading input image {args.input}")
        input_image = load_image(args.input)

        logger.info(f"running inference")
        results = model.predict(input_image)

        if results is not None:
            logger.info(f"drawing inference")
            output_image = model.draw_landmarks(input_image, results)

            logger.info(f"saving output image {args.output}")
            save_image(output_image, args.output)
        sys.exit(0)

    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
