"""Converts a set of videos of certain classes into a image dataset"""

import argparse
import os
import cv2
from fall_detection.utils import is_directory, save_image


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="input directory to read data from.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output directory to save processed data.",
        type=str,
        required=True,
    )
    return parser.parse_args()


def video_into_images(input, output, frames_freq=1, max_frames=10):
    video_cap = cv2.VideoCapture(input)

    video_total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)

    max_frames = video_total_frames if max_frames is None else max_frames

    n_frames = 0
    while True:
        ok, frame = video_cap.read()
        if not ok:
            break

        if n_frames == max_frames:
            break

        if n_frames % frames_freq == 0:
            output_name = output + f"_frame_{n_frames}.jpg"
            save_image(output_name, frame)
        n_frames += 1


def main(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for dir in os.listdir(input_dir):
        if is_directory(os.path.join(input_dir, dir)):
            output_subdir = os.path.join(output_dir, dir)
            if not os.path.exists(output_subdir):
                os.mkdir(output_subdir)
            input_subdir = os.path.join(input_dir, dir)
            for file in os.listdir(input_subdir):
                input_file_path = os.path.join(input_dir, dir, file)
                output_file_path = os.path.join(
                    output_subdir, os.path.basename(file)[:-4]
                )
                video_into_images(
                    input_file_path, output_file_path, frames_freq=3, max_frames=None
                )


if __name__ == "__main__":
    args = cli()
    main(args.input, args.output)
