import argparse
import sys

from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel, TFLiteMovenetModel
from fall_detection.pose.yolo import YoloPoseModel
from fall_detection.pose.augmentation import HorizontalFlip, Rotate, Zoom
from fall_detection.pose.data import PoseLandmarksGenerator


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-images",
        help="input path to read the images from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-images",
        help="output path to save images with landmarks drawn",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--output-file",
        help="output path to save csv files with landmarks and image names.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pose-model-name",
        "--pose-model-name",
        help="pose model to use.",
        type=str,
        choices=["mediapipe", "movenet", "yolo"],
        default="yolo",
    )
    parser.add_argument(
        "--movenet-version",
        "--movenet_version",
        help="specific movenet model to use for pose inference.",
        required=False,
        default="movenet_thunder",
        choices=[
            "movenet_lightning",
            "movenet_thunder",
            "movenet_lightning_f16.tflite",
            "movenet_thunder_f16.tflite",
            "movenet_lightning_int8.tflite",
            "movenet_thunder_int8.tflite",
        ]
    )
    parser.add_argument(
        "--yolo-pose-model-path",
        "--yolo-pose-model-path",
        help="yolo pose model path to use for the inference.",
        required=False,
        default="yolov8n-pose.pt",
    )
    parser.add_argument(
        "--max-samples",
        help="max number of samples to use from each class",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--horizontal-flip",
        "--horizontal-flip",
        help="apply horizontal flip data augmentation to image while generating landmarks",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--rotate",
        "--rotate",
        help="apply rotate data augmentation to image while generating landmarks",
        type=float,
        required=False
    )
    parser.add_argument(
        "--zoom",
        "--zoom",
        help="apply zoom data augmentation to image while generating landmarks",
        type=float,
        required=False
    )
    return parser.parse_args()


def main():
    try:
        args = cli()
        pose_model_name = args.pose_model_name

        print(f"Loading pose model: {pose_model_name}")

        if pose_model_name == "mediapipe":
            model = MediapipePoseModel()
        elif pose_model_name == "movenet":
            movenet_version = args.movenet_version
            model = TFLiteMovenetModel(movenet_version) \
                if movenet_version.endswith("tflite") else MovenetModel(movenet_version)
        elif pose_model_name == "yolo":
            yolo_pose_model_path = args.yolo_pose_model_path
            model = YoloPoseModel(model_path=yolo_pose_model_path)
        else:
            raise ValueError("Model name not valid")
        
        pose_augmentators = []

        if args.horizontal_flip:
            pose_augmentators.append(HorizontalFlip())
        if args.rotate:
            pose_augmentators.append(Rotate(args.rotate))
            pose_augmentators.append(Rotate(-args.rotate))
        if args.zoom:
            pose_augmentators.append(Zoom(args.zoom))
        
        print(f"Pose augmentions to apply per image: {len(pose_augmentators)}")

        generator = PoseLandmarksGenerator(
            images_in_folder=args.input_images,
            images_out_folder=args.output_images,
            csvs_out_folder=args.output_file,
            pose_augmentators=pose_augmentators,
            per_pose_class_limit=args.max_samples,
        )
        generator(model)

        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
