import argparse
import logging
import os
import sys
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline

# setting path
sys.path.append("./")
sys.path.append("../../yolov7")
from logger.logger import configure_logging

from fall.data import load_pose_samples_from_dir
from fall.classification import EstimatorClassifier
from fall.embedding import PoseEmbedder, BLAZE_POSE_KEYPOINTS, COCO_POSE_KEYPOINTS
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

    parser.add_argument(
        "--n-kps", help="number of keypoints", type=int, required=True, default=33
    )
    parser.add_argument(
        "--n-dim",
        help="number of dimensions (x, y, z),(x,y,confidence),(x,y)",
        type=int,
        required=True,
        default=3,
    )

    args = parser.parse_args()

    return args


def main():
    try:
        configure_logging()
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
            landmarks_dir=args.input_file,
            n_landmarks=args.n_kps,
            n_dimensions=args.n_dim,
        )

        # Initialize estimator
        # model = StackingClassifier(
        #     estimators=[
        #         make_pipeline(
        #             StandardScaler(), LogisticRegression(max_iter=9000, random_state=42)
        #         ),
        #         RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        #     ],
        #     final_estimator=KNeighborsClassifier(n_neighbors=5),
        # )
        # model = make_pipeline(
        #     StandardScaler(), LogisticRegression(max_iter=9000, random_state=42)
        # )
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

        # Initialize classifier.
        pose_classifier = EstimatorClassifier(model, pose_embedder)

        pose_classifier.fit(pose_samples)

        with open(f"{args.model}", "wb") as f:
            pickle.dump(pose_classifier, f)

        sys.exit(0)
    except Exception as e:
        raise e
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
