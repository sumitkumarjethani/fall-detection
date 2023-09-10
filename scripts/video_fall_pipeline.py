import argparse
import sys
import cv2
import pickle
import tqdm

from fall_detection.fall.pipeline import Pipeline
from fall_detection.pose.mediapipe import MediapipePoseModel
from fall_detection.pose.movenet import MovenetModel, TFLiteMovenetModel
from fall_detection.pose.yolo import YoloPoseModel
from fall_detection.object_detection.yolo import YoloObjectDetector


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
        help="pose model to use.",
        type=str,
        choices=["mediapipe", "movenet", "yolo"],
        default="yolo",
    )
    parser.add_argument(
        "--movenet-version",
        "--movenet-version",
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
        ],
    )
    parser.add_argument(
        "--yolo-pose-model-path",
        "--yolo-pose-model_path",
        help="yolo pose model path to use for the inference.",
        required=False,
        default="yolov8n-pose.pt",
    )
    parser.add_argument(
        "--yolo-object-model-path",
        "--yolo-object-model-path",
        help="yolo object model path to use for the inference.",
        required=True,
        default="yolov8n.pt",
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
            print(f"Loading pose classifier from: {args.pose_classifier}")
            pose_classifier = pickle.load(f)

        print(f"Loading yolo object detector model from: {args.yolo_object_model_path}")
        object_model = YoloObjectDetector(model_path=args.yolo_object_model_path)

        print("Creating pipeline...")
        pipeline = Pipeline(
            pose_model=pose_model,
            classification_model=pose_classifier,
            object_model=object_model,
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

                output_frame, result_dict = pipeline._run(image=input_frame)
                print(result_dict)

                # Save the output frame.
                out_video.write(output_frame)

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
