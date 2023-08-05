"""Script to covert a yolo formated dataset into a folder like structured dataset for classification"""
import argparse
import os
import shutil
import yaml
from typing import Dict
from fall_detection.logger.logger import LoggerSingleton

logger = LoggerSingleton("app").get_logger()


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="input directory to read data from.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output directory to save processed data.",
        type=str,
        required=True,
    )
    return parser.parse_args()


def get_class_names(path):
    with open(os.path.join(path, "data.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    return {i: c for i, c in enumerate(cfg["names"])}


def move_images(input_dir: str, output_dir: str, class_names: Dict[int, str]):
    for file in os.listdir(os.path.join(input_dir, "labels")):
        with open(os.path.join(input_dir, "labels", file), "r") as f:
            for line in f.readlines():
                image_file = file[:-3] + "jpg"
                src = os.path.join(input_dir, "images", image_file)
                class_name = class_names[int(line[0])]
                if not os.path.exists(os.path.join(output_dir, class_name)):
                    os.mkdir(os.path.join(output_dir, class_name))
                dst = os.path.join(output_dir, class_name, image_file)
                shutil.copy(src=src, dst=dst)


def main(input_dir: str, output_dir: str):
    # create output dirs
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    class_names = get_class_names(input_dir)

    for dir in os.listdir(os.path.join(input_dir)):
        # for dir in os.walk(os.path.join(input_dir)):
        input_subdir = os.path.join(input_dir, dir)
        if os.path.isdir(input_subdir) and dir in ["train", "valid", "test"]:
            output_subdir = os.path.join(output_dir, dir)
            if not os.path.exists(output_subdir):
                os.mkdir(output_subdir)
            move_images(input_subdir, output_subdir, class_names)


if __name__ == "__main__":
    args = cli()

    logger.debug(f"converting dataset from {args.input}")
    main(args.input, args.output)
    logger.debug(f"saving dataset to {args.output}")
