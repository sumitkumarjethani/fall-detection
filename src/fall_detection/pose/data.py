import csv
import cv2
import os
import tqdm

from .base import PoseModel
from ..logger.logger import Logger

logger = Logger("fall-detection").get_logger()


class PoseLandmarksGenerator(object):
    """Helps to bootstrap images and filter pose samples for classification."""

    def __init__(
        self,
        images_in_folder,
        images_out_folder,
        csvs_out_folder,
        per_pose_class_limit=None,
    ):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_folder = csvs_out_folder
        self._per_pose_class_limit = per_pose_class_limit
        # Get list of pose classes and print image statistics.
        self._pose_class_names = sorted(
            [n for n in os.listdir(self._images_in_folder) if not n.startswith(".")]
        )

    def __call__(self, pose_model: PoseModel):
        """Bootstraps images in a given folder.

        Required image in folder (same use for image out folder):
          class_name_1/
            image_001.jpg
            image_002.jpg
            ...
          class_name_2/
            image_001.jpg
            image_002.jpg
            ...
          ...

        Produced CSVs out folder:
          pushups_up.csv
          pushups_down.csv

        Produced CSV structure with pose 3D landmarks:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
        """
        # Create output folder for CVSs.
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)

        for pose_class_name in self._pose_class_names:
            logger.info(f"Processing Images {pose_class_name}")

            # Paths for the pose class.
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + ".csv")
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            with open(csv_out_path, "w") as csv_out_file:
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=",", quoting=csv.QUOTE_MINIMAL
                )
                # Get list of images.
                image_names = sorted(
                    [n for n in os.listdir(images_in_folder) if not n.startswith(".")]
                )
                if self._per_pose_class_limit is not None:
                    image_names = image_names[: self._per_pose_class_limit]

                # Bootstrap every image.
                for image_name in tqdm.tqdm(image_names):
                    # Load image.
                    input_frame = cv2.imread(os.path.join(images_in_folder, image_name))

                    # Initialize fresh pose tracker and run it.
                    results = pose_model.predict(input_frame)

                    # Save image with pose prediction (if pose was detected).
                    output_frame = input_frame.copy()

                    if results is not None:
                        output_frame = pose_model.draw_landmarks(
                            image=output_frame,
                            results=results,
                        )

                    cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

                    # Save landmarks if pose was detected.
                    if results is not None:
                        # Get landmarks.
                        frame_height, frame_width = (
                            output_frame.shape[0],
                            output_frame.shape[1],
                        )
                        pose_landmarks = pose_model.results_to_pose_landmarks(
                            results, frame_height, frame_width
                        )
                        csv_out_writer.writerow(
                            [image_name] + pose_landmarks.flatten().astype(str).tolist()
                        )
        self.align_images_and_csvs()

    def align_images_and_csvs(self, print_removed_items=False):
        """Makes sure that image folders and CSVs have the same sample.

        Leaves only intersetion of samples in both image folders and CSVs.
        """
        for pose_class_name in self._pose_class_names:
            # Paths for the pose class.
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + ".csv")

            # Read CSV into memory.
            rows = []
            with open(csv_out_path) as csv_out_file:
                csv_out_reader = csv.reader(csv_out_file, delimiter=",")
                for row in csv_out_reader:
                    if len(row) == 0: continue
                    rows.append(row)

            # Image names left in CSV.
            image_names_in_csv = []

            # Re-write the CSV removing lines without corresponding images.
            with open(csv_out_path, "w") as csv_out_file:
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=",", quoting=csv.QUOTE_MINIMAL
                )
                for row in rows:
                    image_name = row[0]
                    image_path = os.path.join(images_out_folder, image_name)
                    if os.path.exists(image_path):
                        image_names_in_csv.append(image_name)
                        csv_out_writer.writerow(row)
                    elif print_removed_items:
                        logger.info(f"Removed image from CSV: {image_path}")

            # Remove images without corresponding line in CSV.
            for image_name in os.listdir(images_out_folder):
                if image_name not in image_names_in_csv:
                    image_path = os.path.join(images_out_folder, image_name)
                    os.remove(image_path)
                    if print_removed_items:
                        logger.info(f"Removed image from folder: {image_path}")
