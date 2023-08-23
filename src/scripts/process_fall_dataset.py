import argparse
import logging
import sys
# setting path
sys.path.append("./")

from src.logger.logger import configure_logging
from src.dataset.loaders.fall_dataset import process_dataset


logger = logging.getLogger("app")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I",
        "--input",
        help="input directory path to read raw downloaded fall dataset data. Defaults to current dir",
        required=False,
        default=".",
    )
    parser.add_argument(
        "-O",
        "--output",
        help="output directory path to save downloaded data. If not specified, input dir is used",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    return args


def main():
    try:
        configure_logging()
        args = cli()
        process_dataset(input_dir=args.input, output_dir=args.output)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
