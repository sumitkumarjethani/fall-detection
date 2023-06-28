import argparse
import logging
import os
import pandas as pd
import sys
import tensorflow_hub as hub
import tensorflow as tf

# setting path
sys.path.append('./')

from src.logger.logger import configure_logging
from src.utils import is_directory, path_exist

models = ['default_model']

logger = logging.getLogger('app')

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

column_names = []
for key in KEYPOINT_DICT.keys():
    column_names.append(key+"_x")
    column_names.append(key+"_y")
    column_names.append(key+"_score")
column_names.append('target')
column_names.append('image_name')

module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
input_size = 192

def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='key-point model to use for the dataset',
                        required=False, default='default_model')
    parser.add_argument('--source-path', help='source directory with video frames and labels.txt',
                        required=True, default=None)
    parser.add_argument('--video', help='video directory for which the dataset will be created',
                        required=False, default=None)

    args = parser.parse_args()

    # Model validations
    if args.model not in models:
        raise ValueError("Key-point model " + args.model + " not supported")

    # Source directory structure validations
    if not path_exist(args.source_path):
        raise ValueError(args.source_path + " does not exist")

    if not is_directory(args.source_path):
        raise ValueError(args.source_path + " is not a directory")

    # Video directory validations
    if args.video and not path_exist(args.source_path + "/" + args.video):
        raise ValueError("Video name directory does not exist in " + args.source_path)

    if args.video and not is_directory(args.source_path + "/" + args.video):
        raise ValueError("Video name is not a directory " + args.source_path)

    return args


def get_video_folder_names(source_path, video):
    if video is not None:
        return [video]
    return [directory for directory in os.listdir(source_path) if is_directory(os.path.join(source_path, directory))]


def create_keypoint_dataset(model, source_path, sources_list):
    if not sources_list:
        raise ValueError("No directories found for creating keypoint dataset.")

    # Check dateset directory
    kp_dataset_dir = "./src/dataset"
    if not path_exist(kp_dataset_dir):
        logger.debug("Keypoint dataset directory not found. Creating it on path: " + kp_dataset_dir)
        os.makedirs(kp_dataset_dir)

    # Check if video folders already exists
    for source_name in sources_list:
        if path_exist(kp_dataset_dir + "/" + source_name):
            raise ValueError(source_name + " folder already exists in " +
                             kp_dataset_dir + ". Please delete it.")

    target_classes = ['fall', 'no fall']
    splits = ['train', 'valid', 'test']

    for source_name in sources_list:
        logger.debug("Creating dataset for: " + source_name)
        source_name_dataset_dir = kp_dataset_dir + "/" + source_name
        os.makedirs(source_name_dataset_dir)
        for source_name_dir in os.listdir(source_path + "/" + source_name):
            if source_name_dir in splits:
                logger.debug(source_name_dir + " split...")

                # TODO: change by final keypoint structure
                source_name_dir_df = pd.DataFrame(columns=column_names)

                for class_folder in os.listdir(source_path + "/" + source_name + "/" + source_name_dir):
                    if class_folder in target_classes:
                        class_folder_path = source_path + "/" + source_name + "/" + source_name_dir + "/" + class_folder
                        kps = []
                        for img_name in os.listdir(class_folder_path):
                            image = tf.io.read_file(class_folder_path + "/" + img_name)
                            image = tf.image.decode_jpeg(image)
                            # Resize and pad the image to keep the aspect ratio and fit the expected size.
                            input_image = tf.expand_dims(image, axis=0)
                            input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

                            # Run model inference.
                            keypoints_with_scores = movenet(input_image).flatten().tolist()
                            keypoints_with_scores.append(class_folder)
                            keypoints_with_scores.append(img_name)
                            kps.append(keypoints_with_scores)
                        source_name_dir_df = pd.concat([
                            source_name_dir_df,
                            pd.DataFrame(kps, columns=column_names)
                        ], axis=0)
                source_name_dir_df.to_csv(source_name_dataset_dir + "/"
                                          + source_name_dir + ".csv", index=False)

def main():
    try:
        configure_logging()
        args = cli()

        sources_list = get_video_folder_names(args.source_path, args.video)
        logger.info("Found " + str(len(sources_list)) + " directories in source_path")

        logger.info("Creating keypoint dataset...")
        create_keypoint_dataset(args.model, args.source_path, sources_list)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
