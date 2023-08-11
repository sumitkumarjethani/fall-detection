import argparse
import sys
import pickle
import numpy as np

from fall_detection.logger.logger import LoggerSingleton
from fall_detection.fall.data import load_pose_samples_from_dir
from sklearn.metrics import classification_report, roc_auc_score


logger = LoggerSingleton("app").get_logger()


def get_metrics(y_test, y_pred):
    cls_report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    return cls_report + f"\n auc:{auc}"


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_file",
        help="input path to read the landmarks",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="output txt file to save metrics",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--classification-model",
        help="pose classification model to use.",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main():
    try:
        args = cli()

        with open(f"{args.classification_model}", "rb") as f:
            logger.info("Loading pose classifier...")
            pose_classifier = pickle.load(f)

        # load csv file with pose samples
        logger.info("Loading pose samples...")
        pose_samples = load_pose_samples_from_dir(
            pose_embedder=pose_classifier._pose_embedder,
            landmarks_dir=args.input_file,
            n_landmarks=pose_classifier._n_landmarks,
            n_dimensions=pose_classifier._n_dimensions,
        )

        logger.info("Predicting pose samples...")
        y = np.where(np.array([ps.class_name for ps in pose_samples]) == "Fall", 1, 0)

        y_pred = np.where(
            pose_classifier.predict_pose_samples(pose_samples) == "Fall", 1, 0
        )

        logger.info(f"Writing metrics in {args.output_file} ...")
        with open(args.output_file, "w") as f:
            f.write(get_metrics(y, y_pred))

        sys.exit(0)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
