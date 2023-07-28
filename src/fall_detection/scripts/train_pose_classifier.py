import argparse
import logging
import os
import sys

# setting path
sys.path.append("./")

from logger.logger import configure_logging

from fall.data import load_pose_samples_from_dir
from fall.classification import KnnPoseClassifier
from fall.embedding import PoseEmbedder
import pickle

logger = logging.getLogger("app")


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-file",
        help="input path to read the images from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model name to save.",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def main():
    try:
        configure_logging()
        args = cli()

        # Initialize embedder.
        pose_embedder = PoseEmbedder()

        # load csv file with pose samples
        pose_samples = load_pose_samples_from_dir(
            pose_embedder=pose_embedder,
            landmarks_dir=args.input_file,
            n_landmarks=33,
            n_dimensions=3,
        )

        # Initialize classifier.
        pose_classifier = KnnPoseClassifier(
            pose_embedder=pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10,
            n_landmarks=33,
            n_dimensions=3,
        )

        pose_classifier.fit(pose_samples)

        with open(f"{args.model}", "wb") as f:
            pickle.dump(pose_classifier, f)

        sys.exit(0)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
