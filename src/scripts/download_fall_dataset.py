import argparse
import logging
import sys
# setting path
sys.path.append("./")

from src.logger.logger import configure_logging
from src.dataset.loaders.fall_dataset import download_dataset_from_url, get_fall_dataset_urls


logger = logging.getLogger("app")


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-O",
        "--output",
        help="output directory path to save downloaded data. Defaults to current dir",
        required=False,
        default=".",
    )
    parser.add_argument(
        "-N",
        "--nmax",
        help="Max number of links to download. If not specified. it downloads all available",
        required=False,
        default=None,
        type=int
    )
    args = parser.parse_args()

    return args


def main():
    try:
        configure_logging()
        args = cli()
        urls = get_fall_dataset_urls()
        if args.nmax:
            urls = urls[: args.nmax]
        for url in urls:
            download_dataset_from_url(url, args.output)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
