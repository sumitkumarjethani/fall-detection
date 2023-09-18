import argparse
import sys
import cv2
import numpy as np
import tqdm
import pickle
from fall_detection.fall.classification import EMADictSmoothing
from fall_detection.fall.detection import StateDetector
from fall_detection.fall.plot import PoseClassificationVisualizer
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel, TFLiteMovenetModel
from fall_detection.pose.yolo import YoloPoseModel


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="input path to read the video from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output path to save the video",
        type=str,
        required=True,
    )
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
        ],
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
    args = parser.parse_args()
    return args


def main():
    try:
        args = cli()
        pose_model_name = args.pose_model_name

        # Open the video.
        video_cap = cv2.VideoCapture(args.input)

        # Get some video parameters to generate output video with classificaiton.
        video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Loading pose model: {pose_model_name}")

        if pose_model_name == "mediapipe":
            pose_model = MediapipePoseModel()
        elif pose_model_name == "movenet":
            movenet_version = args.movenet_version
            pose_model = (
                TFLiteMovenetModel(movenet_version)
                if movenet_version.endswith("tflite")
                else MovenetModel(movenet_version)
            )
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
        fall_detector = StateDetector(
            class_name="fall", enter_threshold=6, exit_threshold=4
        )

        # Initialize renderer.
        pose_classification_visualizer = PoseClassificationVisualizer(
            class_name="fall",
            plot_x_max=video_n_frames,
            plot_y_max=10,
            plot_location_x=0.5,
            plot_location_y=0.05,
            detector_location_x=0.01,
            detector_location_y=0.05,
            detector_font_color="red",
            detector_font_size=0.05,
            plot_max_height=0.4,
            plot_max_width=0.4,
            plot_figsize=(9, 4),
        )

        out_video = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (video_width, video_height),
        )

        frame_idx = 0
        output_frame = None
        with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
            while True:
                # Get next frame of the video.
                success, input_frame = video_cap.read()
                if not success:
                    break

                # Run pose tracker.
                results = pose_model.predict(image=input_frame)

                # Draw pose prediction.
                output_frame = input_frame.copy()

                if results is not None:
                    output_frame = pose_model.draw_landmarks(
                        image=output_frame, results=results
                    )

                if results is not None:
                    # Get landmarks.
                    frame_height, frame_width = (
                        output_frame.shape[0],
                        output_frame.shape[1],
                    )
                    pose_landmarks = pose_model.results_to_pose_landmarks(
                        results, frame_height, frame_width
                    )

                    # Classify the pose on the current frame.
                    pose_classification = pose_classifier(pose_landmarks)

                    # Smooth classification using EMA.
                    pose_classification_filtered = pose_classification_smoother(
                        pose_classification
                    )

                    fall_detection = fall_detector(pose_classification_filtered)

                else:
                    pose_classification = None
                    pose_classification_filtered = pose_classification_smoother(dict())
                    pose_classification_filtered = None
                    fall_detection = fall_detector.state

                print(pose_classification_filtered)

                # Draw classification plot and repetition counter.
                output_frame = pose_classification_visualizer(
                    frame=output_frame,
                    pose_classification=pose_classification,
                    pose_classification_filtered=pose_classification_filtered,
                    detector_state=fall_detection,
                )

                # Save the output frame.
                out_video.write(np.array(output_frame))

                frame_idx += 1
                pbar.update()

        # Close output video.
        out_video.release()
        sys.exit(0)
    except ValueError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
