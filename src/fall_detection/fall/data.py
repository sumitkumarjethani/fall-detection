import csv
import numpy as np
import os


class PoseSample(object):
    def __init__(
        self,
        name,
        landmarks,
        class_name,
        embedding,
        distances_embedding=None,
        angles_embedding=None,
    ):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name
        self.embedding = embedding
        self.distances_embedding = distances_embedding
        self.angles_embedding = angles_embedding


def load_pose_samples_from_dir(
    pose_embedder,
    n_landmarks=33,
    landmarks_dir="./data",
    file_extension="csv",
    file_separator=",",
):
    """Loads pose samples from a given folder.

    Required folder structure:
      neutral_standing.csv
      pushups_down.csv
      pushups_up.csv
      squats_down.csv
      ...

    Required CSV structure:
      sample_00001,x1,y1,z1,x2,y2,z2,....
      sample_00002,x1,y1,z1,x2,y2,z2,....
      ...
    """
    # Each file in the folder represents one pose class.
    file_names = [
        name for name in os.listdir(landmarks_dir) if name.endswith(file_extension)
    ]

    pose_samples = []
    for file_name in file_names:
        # Use file name as pose class name.
        class_name = file_name[: -(len(file_extension) + 1)]

        # Parse CSV.
        with open(os.path.join(landmarks_dir, file_name)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=file_separator)
            for row in csv_reader:
                if len(row) == 0:
                    continue
                assert (
                    len(row) == n_landmarks * 3 + 1
                ), "Wrong number of values: {}".format(len(row))
                landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, 3])
                pose_samples.append(
                    PoseSample(
                        name=row[0],
                        landmarks=landmarks,
                        class_name=class_name,
                        embedding=pose_embedder(landmarks),
                        distances_embedding=pose_embedder(landmarks),
                        angles_embedding=pose_embedder(landmarks),
                    )
                )

    return pose_samples
