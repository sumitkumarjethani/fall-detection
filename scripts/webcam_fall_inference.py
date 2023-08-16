import argparse
import sys
import cv2

from fall_detection.fall.classification import EMADictSmoothing
from fall_detection.fall.detection import StateDetector
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel, TFLiteMovenetModel
from fall_detection.pose.yolo import YoloPoseModel
import pickle


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pose-model-name",
        "--pose-model-name",
        help="pose model name",
        type=str,
        required=True,
        choices=["movenet", "mediapipe", "yolo"],
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
        pose_model_name = args.pose_model_name

        # Open the webcam
        cam = cv2.VideoCapture(0)

        print(f"Loading pose model: {pose_model_name}")

        if pose_model_name == "mediapipe":
            pose_model = MediapipePoseModel()
        elif pose_model_name == "movenet":
            movenet_version = args.movenet_version
            pose_model = TFLiteMovenetModel(movenet_version) \
                if movenet_version.endswith("tflite") else MovenetModel(movenet_version)
        elif pose_model_name == "yolo":
            yolo_pose_model_path = args.yolo_pose_model_path
            pose_model = YoloPoseModel(model_path=yolo_pose_model_path)
        else:
            raise ValueError("Model input not valid")

        with open(f"{args.pose_classifier}", "rb") as f:
            pose_classifier = pickle.load(f)

        # Initialize EMA smoothing.
        pose_classification_smoother = EMADictSmoothing(window_size=10, alpha=0.3)

        # Initialize counter.
        fall_detector = StateDetector(class_name="fall", enter_threshold=6, exit_threshold=4)

        while True:
            success, input_frame = cam.read()
            if not success:
                break

            # Run pose tracker.
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            results = pose_model.predict(image=input_frame)

            # Draw pose prediction.
            output_frame = input_frame.copy()
            if results is not None:
                output_frame = pose_model.draw_landmarks(image=output_frame, results=results)

            if results is not None:
                # Get landmarks.
                frame_height, frame_width = (output_frame.shape[0], output_frame.shape[1])
                pose_landmarks = pose_model.results_to_pose_landmarks(results, frame_height, frame_width)

                # Classify the pose on the current frame.
                pose_classification = pose_classifier(pose_landmarks)

                # Smooth classification using EMA.
                pose_classification_filtered = pose_classification_smoother(pose_classification)
                fall_detection = fall_detector(pose_classification_filtered)
            else:
                pose_classification = None
                pose_classification_filtered = pose_classification_smoother(dict())
                pose_classification_filtered = None
                fall_detection = fall_detector.state

            print(pose_classification_filtered)

            cv2.putText(
                output_frame, str(pose_classification_filtered), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2, cv2.LINE_4
            )

            cv2.imshow("video", output_frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
    except ValueError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
