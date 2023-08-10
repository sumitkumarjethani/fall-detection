import argparse
import logging
import os
import sys

from fall_detection.datasets.falldataset import (
    download_dataset_from_url,
    get_falldataset_urls,
    get_falldataset_train_urls,
    get_falldataset_valid_urls,
    get_falldataset_test_urls,
)

from fall_detection.logger.logger import LoggerSingleton

logger = LoggerSingleton("app").get_logger()


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output",
        help="output dirpath to save downloaded data. Defaults to current dir",
        required=False,
        default=".",
    )
    parser.add_argument(
        "--train",
        help="in train split sholuld be included",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--valid",
        help="in valid split sholuld be included",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--test",
        help="in test split sholuld be included",
        default=True,
        action="store_true",
    )
    args = parser.parse_args()

    return args


def main():
    try:
        args = cli()
        urls = get_falldataset_urls()
        if args.train:
            output_dir = os.path.join(args.output, "train")
            urls = get_falldataset_train_urls()
            for url in urls:
                download_dataset_from_url(url, output_dir)
        if args.valid:
            output_dir = os.path.join(args.output, "valid")
            urls = get_falldataset_valid_urls()
            for url in urls:
                download_dataset_from_url(url, output_dir)
        if args.test:
            output_dir = os.path.join(args.output, "test")
            urls = get_falldataset_test_urls()
            for url in urls:
                download_dataset_from_url(url, output_dir)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
