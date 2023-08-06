import argparse
import sys
import cv2

from fall_detection.logger.logger import LoggerSingleton
from fall_detection.fall.classification import EMADictSmoothing
from fall_detection.fall.detection import StateDetector
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel
from fall_detection.pose.yolo import YoloPoseModel
import pickle

logger = LoggerSingleton("app").get_logger()


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--pose-model",
        help="pose model name",
        type=str,
        required=True,
        choices=["movenet", "mediapipe", "yolo"],
        default="yolo",
    )
    parser.add_argument(
        "-p",
        "--yolo_model_path",
        help="yolo model path to use for the inference.",
        required=False,
        default="yolov8n-pose.pt",
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

        # Open the webcam
        cam = cv2.VideoCapture(0)

        logger.info(f"loading pose model")

        if args.pose_model == "mediapipe":
            pose_model = MediapipePoseModel()
        elif args.pose_model == "movenet":
            pose_model = MovenetModel()
        elif args.pose_model == "yolo":
            yolo_model_path = args.yolo_model_path
            pose_model = YoloPoseModel(model_path=yolo_model_path)
        else:
            raise ValueError("model input not valid")

        with open(f"{args.classification_model}", "rb") as f:
            pose_classifier = pickle.load(f)

        # Initialize EMA smoothing.
        pose_classification_smoother = EMADictSmoothing(window_size=10, alpha=0.3)

        # Initialize counter.
        fall_detector = StateDetector(class_name="Fall", enter_threshold=6, exit_threshold=4)

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

            logger.info(pose_classification_filtered)

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
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
