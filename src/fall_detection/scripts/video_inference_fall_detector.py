import argparse
import logging
import os
import sys
import cv2
import numpy as np
import tqdm

# setting path
sys.path.append("./")

from logger.logger import configure_logging

from fall.classification import EMADictSmoothing
from fall.embedding import PoseEmbedder
from fall.detection import StateDetector
from fall.plot import PoseClassificationVisualizer
from pose.mediapipe import MediapipePoseModel
from pose.movenet import MovenetModel
import pickle

logger = logging.getLogger("app")


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
        "-o",
        "--output",
        help="input path to read the images from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--pose-model",
        help="model name to save.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--classification-model",
        help="model name to save.",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def main():
    try:
        configure_logging()
        args = cli()

        # Open the video.

        video_cap = cv2.VideoCapture(args.input)

        # Get some video parameters to generate output video with classificaiton.
        video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if args.pose_model == "mediapipe":
            pose_model = MediapipePoseModel()
        else:
            pose_model = MovenetModel()

        with open(f"{args.classification_model}", "rb") as f:
            pose_classifier = pickle.load(f)

        # Initialize EMA smoothing.
        pose_classification_smoother = EMADictSmoothing(window_size=10, alpha=0.2)

        # Initialize counter.
        fall_detector = StateDetector(
            class_name="Fall", enter_threshold=8, exit_threshold=4
        )

        # Initialize renderer.
        pose_classification_visualizer = PoseClassificationVisualizer(
            class_name="Fall",
            plot_x_max=video_n_frames,
            plot_y_max=10,
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
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                pose_landmarks = pose_model.predict(image=input_frame)

                # Draw pose prediction.
                output_frame = input_frame.copy()

                if pose_landmarks is not None:
                    pose_model.draw_landmarks(
                        image=output_frame,
                        pose_landmarks=pose_landmarks,
                    )

                if pose_landmarks is not None:
                    # Get landmarks.
                    frame_height, frame_width = (
                        output_frame.shape[0],
                        output_frame.shape[1],
                    )
                    pose_landmarks = np.array(
                        [
                            [
                                lmk.x * frame_width,
                                lmk.y * frame_height,
                                lmk.z * frame_width,
                            ]
                            for lmk in pose_landmarks.landmark
                        ],
                        dtype=np.float32,
                    )
                    assert pose_landmarks.shape == (
                        33,
                        3,
                    ), "Unexpected landmarks shape: {}".format(pose_landmarks.shape)

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
                out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

                frame_idx += 1
                pbar.update()

        # Close output video.
        out_video.release()

        sys.exit(0)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
