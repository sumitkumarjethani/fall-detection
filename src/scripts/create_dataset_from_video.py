import argparse
import logging
import sys
# setting path
sys.path.append('./')

from src.logger.logger import configure_logging
from src.utils import is_directory, file_exists

models = ['default_model']

logger = logging.getLogger('app')


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='key-point model to use for the dataset',
                        required=False, default='default_model')
    parser.add_argument('--source-path', help='source directory with video frames and labels.txt',
                        required=True, default=None)
    parser.add_argument('--video', help='video directory for which the dataset will be created',
                        required=False, default=None)

    args = parser.parse_args()

    # Model validations
    if args.model not in models:
        raise ValueError("Key-point model " + args.model + " not supported")

    # Source directory structure validations
    if not is_directory(args.source_path):
        raise ValueError(args.source_path + " is not a directory")

    if not file_exists(args.source_path + "/labels.txt"):
        raise ValueError("Label.txt does not exist in " + args.source_path)

    # Video directory validations
    if args.video and not is_directory(args.source_path + "/" + args.video):
        raise ValueError("Video directory does not exist in " + args.source_path)

    return args


def main():
    try:
        configure_logging()
        args = cli()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
