import argparse
import sys
import os
import pickle
import numpy as np
from fall_detection.fall.data import load_pose_samples_from_dir
from sklearn.metrics import classification_report, roc_auc_score


def get_metrics(y_test, y_pred):
    cls_report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    return cls_report + f"\n auc:{auc}"


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="input path to read the landmarks",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output path to save metrics",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--file-name",
        help="metrics file name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pose-classifier",
        "--pose-classifier",
        help="pose classification model to use.",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main():
    try:
        args = cli()

        with open(f"{args.pose_classifier}", "rb") as f:
            print(f"Loading pose classifier from: {args.pose_classifier}")
            pose_classifier = pickle.load(f)

        # load csv file with pose samples
        print(f"Loading pose samples from: {args.input}")
        pose_samples = load_pose_samples_from_dir(
            pose_embedder=pose_classifier._pose_embedder,
            landmarks_dir=args.input,
            n_landmarks=pose_classifier._n_landmarks,
        )

        print("Predicting on loaded pose samples...")
        y = np.where(np.array([ps.class_name for ps in pose_samples]) == "fall", 1, 0)
        y_pred = np.where(
            pose_classifier.predict_pose_samples(pose_samples) == "fall", 1, 0
        )

        # Create output folder if not exists.
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        print(f"Writing metrics in {args.output} ...")
        with open(os.path.join(args.output, f"{args.file_name}.txt"), "w") as f:
            f.write(get_metrics(y, y_pred))

        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
