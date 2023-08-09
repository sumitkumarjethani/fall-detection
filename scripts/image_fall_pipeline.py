import argparse
import sys
import pickle

from fall_detection.fall.pipeline import Pipeline
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel, TFLiteMovenetModel
from fall_detection.pose.yolo import YoloPoseModel
from fall_detection.object_detection.yolo import YoloObjectDetector
from fall_detection.logger.logger import LoggerSingleton
from fall_detection.utils import load_image, save_image

logger = LoggerSingleton("app").get_logger()


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="input image to read the image in a jpg/png format.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="output path to save the image with fall inference.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--pose-model-name",
        "--pose_model_name",
        help="pose model to use.",
        type=str,
        choices=["mediapipe", "movenet", "yolo"],
        default="yolo",
    )

    parser.add_argument(
        "--movenet-version",
        "--movenet_version",
        help="specific movenet model name to use for pose inference.",
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
        "--yolo_pose_model_path",
        help="yolo pose model path to use for the inference.",
        required=False,
        default="yolov8n-pose.pt",
    )

    parser.add_argument(
        "--yolo-object-model-path",
        "--yolo_object_model_path",
        help="yolo object model path to use for the inference.",
        required=True,
        default="yolov8n.pt",
    )

    parser.add_argument(
        "--pose-classifier",
        "--pose_classifier",
        help="pose classification model to use.",
        type=str,
        required=True,
    )
    
    return parser.parse_args()


def main():
    try:
        args = cli()

        pose_model_name = args.pose_model_name
        logger.info(f"Loading pose model: {pose_model_name}")

        if pose_model_name == "mediapipe":
            pose_model = MediapipePoseModel()
        elif pose_model_name == "movenet":
            movenet_version = args.movenet_version
            pose_model = TFLiteMovenetModel(movenet_version) \
                if movenet_version.endswith("tflite") else MovenetModel(movenet_version)
        elif pose_model_name == "yolo":
            pose_model = YoloPoseModel(model_path=args.yolo_pose_model_path)
        else:
            raise ValueError("pose model name not valid")
        
        with open(f"{args.pose_classifier}", "rb") as f:
            logger.info("Loading pose classifier...")
            pose_classifier = pickle.load(f)
        
        logger.info("Loading yolo object detector model")
        object_model = YoloObjectDetector(model_path=args.yolo_object_model_path)

        logger.info(f"Loading input image from: {args.input}")
        input_image = load_image(args.input)

        logger.info("Creating pipeline...")
        pipeline = Pipeline(
            pose_model=pose_model,
            classification_model=pose_classifier,
            object_model=object_model
        )

        logger.info("Running pipeline...")
        output_image = pipeline._run(input_image)

        logger.info(f"Saving output image in: {args.output}")
        save_image(output_image, args.output)

        sys.exit(0)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()    
 