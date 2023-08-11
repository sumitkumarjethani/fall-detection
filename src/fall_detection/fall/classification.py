import numpy as np
from typing import List
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from fall_detection.fall.embedding import PoseEmbedder
from .data import PoseSample


class PoseClassifier(ABC):
    @abstractmethod
    def fit(self, pose_samples):
        raise NotImplemented()

    @abstractmethod
    def predict_pose_samples(self, pose_samples):
        raise NotImplemented()

    @abstractmethod
    def predict(self, pose_landmarks):
        raise NotImplemented()

    def __call__(self, pose_landmarks):
        return self.predict(pose_landmarks)


class EstimatorClassifier(PoseClassifier):
    def __init__(
        self,
        estimator: BaseEstimator,
        pose_embedder: PoseEmbedder,
        n_output_scaler=10,
        n_landmarks=33,
        n_dimensions=3,
    ):
        self._pose_embedder = pose_embedder
        self._n_output_scaler = n_output_scaler
        self._model = estimator
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions

    def fit(self, pose_samples: List[PoseSample]):
        print(f"fitting on {len(pose_samples)} pose samples")
        X = np.array([ps.embedding for ps in pose_samples]).reshape(
            len(pose_samples), -1
        )
        y = np.array([ps.class_name for ps in pose_samples])
        self._model.fit(X, y)
        return self

    def predict_pose_samples(self, pose_samples: List[PoseSample]):
        print(f"predict on {len(pose_samples)} pose samples")
        X = np.array([ps.embedding for ps in pose_samples]).reshape(
            len(pose_samples), -1
        )
        y_pred = self._model.predict(X)
        return y_pred

    def predict(self, pose_landmarks: np.ndarray):
        pose_embedding = self._pose_embedder(pose_landmarks).reshape(1, -1)
        pred_prob = self._model.predict_proba(pose_embedding)
        result = {
            c: self._n_output_scaler * pred_prob[0][i]
            for i, c in enumerate(self._model.classes_)
        }
        return result


class KnnPoseClassifier(PoseClassifier):
    """Classifies pose landmarks."""

    def __init__(
        self,
        pose_embedder,
        n_landmarks=33,
        n_dimensions=3,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10,
        axes_weights=(1.0, 1.0, 0.1),
    ):
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights

        self._pose_samples = []

    def fit(self, pose_samples):
        print(f"fitting on {len(pose_samples)}  pose samples")
        self._pose_samples = pose_samples

    def predict_pose_samples(self, pose_samples: List[PoseSample]):
        y_preds = [self.predict(pose_sample.landmarks) for pose_sample in pose_samples]
        return np.array([max(y_pred, key=y_pred.get) for y_pred in y_preds])

    def predict(self, pose_landmarks):
        """Classifies given pose.

        Classification is done in two stages:
          * First we pick top-N samples by MAX distance. It allows to remove samples
            that are almost the same as given pose, but has few joints bent in the
            other direction.
          * Then we pick top-N samples by MEAN distance. After outliers are removed
            on a previous step, we can pick samples that are closes on average.

        Args:
          pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

        Returns:
          Dictionary with count of nearest pose samples from the database. Sample:
            {
              'pushups_down': 8,
              'pushups_up': 2,
            }
        """
        # Check that provided and target poses have the same shape.
        assert pose_landmarks.shape == (
            self._n_landmarks,
            self._n_dimensions,
        ), "Unexpected shape: {}".format(pose_landmarks.shape)

        # Get given pose embedding.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(
            pose_landmarks * np.array([-1, 1, 1])
        )

        # Filter by max distance.
        #
        # That helps to remove outliers - poses that are almost the same as the
        # given one, but has one joint bent into another direction and actually
        # represents a different pose class.
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(
                    np.abs(sample.embedding - flipped_pose_embedding)
                    * self._axes_weights
                ),
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[: self._top_n_by_max_distance]
        # Filter by mean distance.
        # After removing outliers we can find the nearest pose by mean distance.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(
                    np.abs(sample.embedding - flipped_pose_embedding)
                    * self._axes_weights
                ),
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[: self._top_n_by_mean_distance]
        # Collect results into map: (class_name -> n_samples)
        class_names = [
            self._pose_samples[sample_idx].class_name
            for _, sample_idx in mean_dist_heap
        ]
        result = {
            class_name: class_names.count(class_name) for class_name in set(class_names)
        }

        return result


class EMADictSmoothing(object):
    """Smoothes pose classification."""

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        """Smoothes given pose classification.

        Smoothing is done by computing Exponential Moving Average for every pose
        class observed in the given time window. Missed pose classes arre replaced
        with 0.

        Args:
          data: Dictionary with pose classification. Sample:
              {
                'class_name_1': 8,
                'class_name_2': 2,
              }

        Result:
          Dictionary in the same format but with smoothed and float instead of
          integer values. Sample:
            {
              'class_name_1': 8.3,
              'class_name_2': 1.7,
            }
        """
        # Add new data to the beginning of the window for simpler code.

        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[: self._window_size]

        # Get all keys.
        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # Get smoothed values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update factor.
                factor *= 1.0 - self._alpha

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data
