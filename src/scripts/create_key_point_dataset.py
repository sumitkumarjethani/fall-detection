import argparse
import logging
import os
import pandas as pd
import sys
import tensorflow as tf
from tqdm import tqdm


# setting path
sys.path.append('./')

from src.logger.logger import configure_logging
from src.utils import is_directory, path_exist
from src.models.pose.movenet import key_point_dict, MovenetModel

logger = logging.getLogger('app')

model_names = ['movenet_thunder', 'movenet_lightning']
target_classes = ["no fall", "fall"]


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-M',
        '--model',
        help='key-point model to use for the dataset creation',
        required=False,
        default=model_names[0]
    )
    parser.add_argument(
        '-I',
        '--input',
        help='input directory with images separated in fall & no fall sub-directories',
        required=True,
        default=None
    )
    parser.add_argument(
        '-O',
        '--output',
        help='output directory where key point data will be saved',
        required=True,
        default=None
    )
    args = parser.parse_args()

    # Model validations
    if args.model not in model_names:
        raise ValueError("Key-point model " + args.model + " not supported")

    # Input directory structure validations
    if not path_exist(args.input):
        raise ValueError(args.input + " does not exist")

    if not is_directory(args.input):
        raise ValueError(args.input + " is not a directory")

    input_dirs = [directory for directory in os.listdir(args.input) if is_directory(os.path.join(args.input, directory))]
    if target_classes[0] not in input_dirs:
        raise ValueError(target_classes[0] + " directory not in " + args.input)

    if target_classes[1] not in input_dirs:
        raise ValueError(target_classes[1] + " directory not in " + args.input)

    # Output directory structure validations
    if not path_exist(args.output):
        raise ValueError(args.output + " does not exist")

    if not is_directory(args.output):
        raise ValueError(args.output + " is not a directory")

    return args


def decode_image(image_path):
    image = tf.io.read_file(image_path)

    if image_path.lower().endswith('.jpeg') or image_path.lower().endswith('.jpg'):
        image = tf.image.decode_jpeg(image)
    elif image_path.lower().endswith('.png'):
        image = tf.image.decode_png(image)
    elif image_path.lower().endswith('.gif'):
        image = tf.image.decode_gif(image)
    elif image_path.lower().endswith('.bmp'):
        image = tf.image.decode_bmp(image)
    else:
        raise ValueError('Image format not supported')

    return image


def get_model(model_name):
    if model_name == "movenet_thunder" or model_name == 'movenet_lightning':
        model = MovenetModel(model_name)
    else:
        raise ValueError('Model not supported')

    return model


def create_keypoint_dataset(model_name, input, output):
    model = get_model(model_name)
    logger.info("Model " + model_name + " loaded successfully")

    # Create final df columns
    column_names = [f"{key}_{suffix}" for key in key_point_dict.keys() for suffix in ["x", "y", "score"]]
    column_names.extend(['image_path', 'target'])

    for target_class in target_classes:
        logger.info("Processing " + target_class + " images...")
        target_class_dir = os.path.join(input, target_class)
        image_names = [file for file in os.listdir(target_class_dir)
                       if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

        key_points = []
        for image_name in tqdm(image_names):
            image_path = os.path.join(target_class_dir, image_name)
            image = decode_image(image_path)
            image_key_points = model.get_key_points(image).flatten().tolist()
            image_key_points.append(image_path)
            image_key_points.append(target_class)
            key_points.append(image_key_points)

        pd.DataFrame(key_points, columns=column_names).to_csv(os.path.join(output, target_class + ".csv"), index=False)


def main():
    try:
        configure_logging()
        args = cli()

        model_name = args.model
        input = args.input
        output = args.output

        logger.info("Creating keypoint dataset from: " + input + " to: " + output + " using model: " + model_name)
        create_keypoint_dataset(model_name, input, output)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
