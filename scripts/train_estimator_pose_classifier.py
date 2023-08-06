import argparse
import sys
from sklearn.ensemble import RandomForestClassifier
from fall_detection.logger.logger import LoggerSingleton
from fall_detection.fall.data import load_pose_samples_from_dir
from fall_detection.fall.classification import EstimatorClassifier, KnnPoseClassifier
from fall_detection.fall.embedding import (PoseEmbedder, BLAZE_POSE_KEYPOINTS, COCO_POSE_KEYPOINTS)
import pickle

logger = LoggerSingleton("app").get_logger()


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_file",
        help="input path to read the images from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model to train.",
        type=str,
        choices=["knn", "rf"],
        default="rf",
    )
    parser.add_argument(
        "-name",
        "--output_model_name",
        help="trained model name.",
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
        required=False,
        default=10,
    )
    return parser.parse_args()


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
            landmarks_dir=args.input_file,
            n_landmarks=args.n_kps,
            n_dimensions=args.n_dim,
        )

        # Initialize pose classifier.
        if args.model == "knn":
            pose_classifier = KnnPoseClassifier(pose_embedder=pose_embedder, top_n_by_max_distance=30,
                                                top_n_by_mean_distance=args.n_neighbours, n_landmarks=args.n_kps,
                                                n_dimensions=args.n_dim)
        elif args.model == "rf":
            pose_classifier = EstimatorClassifier(
                RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), pose_embedder)
        else:
            raise ValueError("Supported trainable models are KNN or RF")

        pose_classifier.fit(pose_samples)

        with open(f"{args.output_model_name}", "wb") as f:
            pickle.dump(pose_classifier, f)

        sys.exit(0)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
