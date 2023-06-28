"""
Utility module to load and prepare de dataset from 
falldataset.com
"""

import os
from typing import Dict, List, Optional, AnyStr
import csv
import shutil
import uuid

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
    "https://falldataset.com/data/1176/1176.tar.gz",
    "https://falldataset.com/data/2123/2123.tar.gz",
    "https://falldataset.com/data/832/832.tar.gz",
    "https://falldataset.com/data/786/786.tar.gz",
    "https://falldataset.com/data/925/925.tar.gz",
]

_dataset_classes = {
    0: "Empty",
    1: "Standing",
    2: "Sitting",  # this is the one that matters for us
    3: "Lying",
    4: "Bending",
    5: "Crawling",
}

_output_classes = {0: "NotFall", 1: "Fall"}

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


def get_falldataset_classes() -> Dict[int, str]:
    return _dataset_classes


def get_falldataset_urls() -> List[str]:
    return _dataset_urls


def get_firname_from_url(url: str) -> str:
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
        os.system(f"rm {temp_tar_file}")
    except Exception as e:
        raise Exception(f"could not delete temp.tar.gz file: {e}")

    try:
        # remove depth directory because we dont need it.
        depth_dir = os.path.join(output_dir, get_firname_from_url(url), "depth")
        os.system(f"rm -r {depth_dir}")
    except Exception as e:
        raise Exception(f"could not delete depth directory: {e}")


def process_dataset(
    input_dir: str,
    output_dir: Optional[str] = None,
) -> None:
    """
    Process falldataset to the form of:

    Fall/
        image_00001.png
        image_00002.png
        ...
    NotFall/
        image_00001.png
        image_00002.png
        ...

    Args:
        input_dir (str): the directory where all data of falldataset was downloaded.
        output_dir (str): the directory where processed data will be placed. Default same as input.

    ...
    """
    if not output_dir:
        output_dir = input_dir

    if not os.path.exists(input_dir):
        raise ValueError(f"input directory does not exists {input_dir}")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_classes = get_output_classes()

    classes_mapping = get_classes_mapping()

    for class_index in output_classes:
        output_class_dir = os.path.join(output_dir, output_classes[class_index])

        if not os.path.exists(output_class_dir):
            os.mkdir(output_class_dir)

    # rearrangin images
    n_images = {0: 0, 1: 0}
    for dir in os.listdir(input_dir):
        curdir = os.path.join(input_dir, dir)

        if dir in output_classes.values():
            continue
        if "rgb" not in os.listdir(curdir):
            continue
        if "labels.csv" not in os.listdir(curdir):
            continue

        labels_path = os.path.join(curdir, "labels.csv")

        images_dir = os.path.join(curdir, "rgb")

        with open(labels_path, newline="", mode="r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:

                src_image_number = str(row["index"]).zfill(4)
                src_image_filename = f"rgb_{src_image_number}.png"
                src_image_path = os.path.join(images_dir, src_image_filename)

                class_index = classes_mapping.get(int(row["class"]), -1)
                if class_index == -1:
                    continue

                class_name = output_classes.get(class_index)
                n_images[class_index] += 1
                dest_image_number = str(n_images[class_index]).zfill(9)
                dst_image_filename = f"image_{dest_image_number}.png"

                dst_image_path = os.path.join(
                    output_dir, class_name, dst_image_filename
                )
                print("copying:    ", src_image_path, "     to:     ", dst_image_path)
                shutil.copy(src=src_image_path, dst=dst_image_path)
