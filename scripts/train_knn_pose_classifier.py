import argparse
import sys
import pickle

from fall_detection.logger.logger import LoggerSingleton
from fall_detection.fall.data import load_pose_samples_from_dir
from fall_detection.fall.classification import KnnPoseClassifier
from fall_detection.fall.embedding import (PoseEmbedder, COCO_POSE_KEYPOINTS, BLAZE_POSE_KEYPOINTS)

logger = LoggerSingleton("app").get_logger()


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
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
    parser.add_argument(
        "--n-kps",
        help="number of keypoints generaly 33 or 17 depending on pose model used",
        type=int,
        required=True,
        default=33,
    )
    parser.add_argument(
        "--n-dim",
        help="number of dimensions of the inputs. Generarly 3 (x,y,z) or (x,y,score)",
        type=int,
        required=True,
        default=3,
    )
    parser.add_argument(
        "--n-neighbours",
        help="number of neighbours used to predict in knn algorithm",
        type=int,
        required=True,
        default=10,
    )
    args = parser.parse_args()
    return args


def main():
    try:
        args = cli()

        # Initialize embedder.
        if args.n_kps == 17:
            landmark_names = COCO_POSE_KEYPOINTS
        elif args.n_kps == 33:
            landmark_names = BLAZE_POSE_KEYPOINTS
        else:
            raise ValueError("number of keypoints supported are 17 or 33")

        pose_embedder = PoseEmbedder(landmark_names=landmark_names)

        # load csv file with pose samples
        pose_samples = load_pose_samples_from_dir(
            pose_embedder=pose_embedder,
            landmarks_dir=args.input,
            n_landmarks=args.n_kps,
            n_dimensions=args.n_dim,
        )

        # Initialize classifier.
        pose_classifier = KnnPoseClassifier(
            pose_embedder=pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=args.n_neighbours,
            n_landmarks=args.n_kps,
            n_dimensions=args.n_dim,
        )

        pose_classifier.fit(pose_samples)

        with open(f"{args.model}", "wb") as f:
            pickle.dump(pose_classifier, f)

        sys.exit(0)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
