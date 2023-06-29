"""
Utility module to load and prepare de dataset from 
falldataset.com
"""
import os
import csv
import shutil
import logging
from typing import Dict, List, Optional
import sys
# setting path
sys.path.append("./")

from src.utils import is_directory, path_exist

logger = logging.getLogger('app')

# dataset urls
_dataset_urls = [
    "https://falldataset.com/data/1301/1301.tar.gz",
    "https://falldataset.com/data/1790/1790.tar.gz",
    "https://falldataset.com/data/722/722.tar.gz",
    "https://falldataset.com/data/1378/1378.tar.gz",
    "https://falldataset.com/data/1392/1392.tar.gz",
    "https://falldataset.com/data/807/807.tar.gz",
    "https://falldataset.com/data/758/758.tar.gz",
    "https://falldataset.com/data/1843/1843.tar.gz",
    "https://falldataset.com/data/569/569.tar.gz",
    "https://falldataset.com/data/1260/1260.tar.gz",
    "https://falldataset.com/data/489/489.tar.gz",
    "https://falldataset.com/data/731/731.tar.gz",
    "https://falldataset.com/data/1219/1219.tar.gz",
    "https://falldataset.com/data/1954/1954.tar.gz",
    "https://falldataset.com/data/581/581.tar.gz",
    "https://falldataset.com/data/1176/1176.tar.gz",
    "https://falldataset.com/data/2123/2123.tar.gz",
    "https://falldataset.com/data/832/832.tar.gz",
    "https://falldataset.com/data/786/786.tar.gz",
    "https://falldataset.com/data/925/925.tar.gz",
]

_dataset_classes = {
    0: "Empty",
    1: "Standing",
    2: "Sitting",
    3: "Lying", # this is the one that matters for us
    4: "Bending",
    5: "Crawling",
}

_output_classes = {0: "no fall", 1: "fall"}

_classes_mapping = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 0,
    5: 0,
}


def get_classes_mapping() -> Dict[int, int]:
    return _classes_mapping


def get_output_classes() -> Dict[int, str]:
    return _output_classes


def get_fall_dataset_classes() -> Dict[int, str]:
    return _dataset_classes


def get_fall_dataset_urls() -> List[str]:
    return _dataset_urls


def get_fir_name_from_url(url: str) -> str:
    return os.path.basename(url).split(".")[0]


def download_dataset_from_url(url: str, output_dir: str) -> None:
    temp_tar_file = os.path.join(output_dir, "temp.tar.gz")
    try:
        os.system("wget " + url + " -O " + temp_tar_file)
    except Exception as e:
        raise Exception(f"could not download data from {url} : {e}")

    try:
        os.system(f"tar -xf {temp_tar_file} -C {output_dir} ")
    except Exception as e:
        raise Exception(f"could not untar downloaded temp.tar.gz file: {e}")

    try:
        #os.system(f"rm {temp_tar_file}")
        os.system(f"del {temp_tar_file}")   # windows
    except Exception as e:
        raise Exception(f"could not delete temp.tar.gz file: {e}")

    try:
        # remove depth directory because we don't need it.
        depth_dir = os.path.join(output_dir, get_fir_name_from_url(url), "depth")
        #os.system(f"rm -r {depth_dir}")
        os.system(f"rmdir /s /q {depth_dir}")   # windows
    except Exception as e:
        raise Exception(f"could not delete depth directory: {e}")


def process_dataset(input_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Process fall dataset to the form of:

    fall/
        image_00001.png
        image_00002.png
        ...
    no fall/
        image_00001.png
        image_00002.png
        ...

    Args:
        input_dir (str): the directory where all data of fall dataset was downloaded.
        output_dir (str): the directory where processed data will be placed. Default same as input.
    """
    if not output_dir:
        output_dir = input_dir

    if not path_exist(input_dir):
        raise ValueError(f"input directory does not exists {input_dir}")

    if not path_exist(output_dir):
        os.mkdir(output_dir)

    output_classes = get_output_classes()
    classes_mapping = get_classes_mapping()

    for class_index in output_classes:
        output_class_dir = os.path.join(output_dir, output_classes[class_index])

        if not path_exist(output_class_dir):
            os.mkdir(output_class_dir)

    # re arrange images
    n_images = {0: 0, 1: 0}
    for directory in os.listdir(input_dir):
        current_dir = os.path.join(input_dir, directory)

        if not is_directory(current_dir):
            continue
        if directory in output_classes.values():
            continue
        if "rgb" not in os.listdir(current_dir):
            continue
        if "labels.csv" not in os.listdir(current_dir):
            continue

        labels_path = os.path.join(current_dir, "labels.csv")
        images_dir = os.path.join(current_dir, "rgb")

        with open(labels_path, newline="", mode="r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                src_image_number = str(row["index"]).zfill(4)
                src_image_filename = f"rgb_{src_image_number}.png"
                src_image_path = os.path.join(images_dir, src_image_filename)
                if not path_exist(src_image_path):
                    continue
                class_index = classes_mapping.get(int(row["class"]), -1)
                if class_index == -1:
                    continue

                class_name = output_classes.get(class_index)
                n_images[class_index] += 1
                dst_image_number = str(n_images[class_index]).zfill(9)
                dst_image_filename = f"image_{dst_image_number}.png"

                dst_image_path = os.path.join(
                    output_dir, class_name, dst_image_filename
                )
                logger.info("copying:    ", src_image_path, "     to:     ", dst_image_path)
                shutil.copy(src=src_image_path, dst=dst_image_path)
