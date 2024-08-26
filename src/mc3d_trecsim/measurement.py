"""A module for the measurement of different aspects of the skeleton."""
from __future__ import annotations
from cProfile import label
from collections import deque
from typing import Annotated, Any, Sequence, TypeVar, Generic, TypeAlias, cast, overload
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import numpy.typing as npt
from mc3d_trecsim_gmm import Camera

from .mc3d_types import Skeleton, SkeletonValidity
from .enums import KPT_IDXS


def plot_measurement_history(measurement: Measurement[np.double | float], ax: None = None, mean_window: int = 0):
    """Plot the measurement history."""
    if ax is None:
        _, ax = plt.subplots()
    ax.set_title(measurement.label)
    ax.set_xlabel('Frame Nr.')
    ax.set_ylabel(f'{measurement.label} $[{measurement.unit}]$')

    segments, frames = measurement.segmented_history

    p = 1
    for i, segment in enumerate(segments):
        for history in segment:
            mean_history = np.convolve(history, np.ones(mean_window)/mean_window, mode='valid') if mean_window > 0 else history
            x = frames[i][mean_window-1:] if mean_window > 0 else frames[i]

            ax.plot(x, mean_history, label='Person %d' % p)
            p += 1


def to_csv(measurements: Sequence[Measurement[np.double | float]], csv_path: Path):
    """Plot the measurement history."""
    history_table = [[], []]
    labels = ['Frame', 'Person']

    for i, measurement in enumerate(measurements):
        if isinstance(measurement, ScalarMeasurement):
            history_table.append([])
            labels.append(measurement.label)
        elif isinstance(measurement, BaseVectorMeasurement):
            labels.append(measurement.label + ' x')
            labels.append(measurement.label + ' y')
            labels.append(measurement.label + ' z')
            for _ in range(3):
                history_table.append([])

        segments, frames = measurement.segmented_history
        p = 1
        for segment, frame in zip(segments, frames):
            for history in segment:
                if isinstance(measurement, ScalarMeasurement):
                    history_table[-1] += list(history)
                elif isinstance(measurement, BaseVectorMeasurement):
                    np_history = np.array(history)
                    history_table[-3] += list(np_history[:, 0])
                    history_table[-2] += list(np_history[:, 1])
                    history_table[-1] += list(np_history[:, 2])
                if i == 0:
                    history_table[0] += frame
                    history_table[1] += [p]*len(history)
                    p += 1

    np.savetxt(csv_path, np.array(history_table).T, delimiter=';',
               header=';'.join(labels), comments='')


MT = TypeVar('MT')
class Measurement(ABC, Generic[MT]):
    """A class for the measurement of different aspects of the skeleton."""
    @property
    def values(self) -> list[MT]:
        """Return the values of the measurement.

        Returns:
            list[MT]: The values of the measurement.
        """
        return self._values

    @property
    def history(self) -> deque[list[MT]]:
        """Return the values of the measurement.

        Returns:
            list[MT]: The values of the measurement.
        """
        return self._history
    
    @property
    def segmented_history(self) -> tuple[list[list[MT]], list[list[int]]]:
        """Return the values of the measurement.

        Returns:
            list[MT]: The values of the measurement.
        """
        segments: list[list[MT]] = []
        frames: list[list[int]] = []
        last_length = -1

        for i, history_values in enumerate(self.history):
            if len(history_values) != last_length:
                segments.append([[] for _ in range(len(history_values))])
                frames.append([])

            last_length = len(history_values)
            
            for j, history_value in enumerate(history_values):
                segments[-1][j].append(history_value)

            frames[-1].append(i)

        return segments, frames

    def __init__(self, history_size: int = 60*25):
        """Initialize the measurement.

        Args:
            default (MT): The default value of the measurement.
        """
        self._history: deque[list[MT]] = deque(maxlen=history_size)
        self._values: list[MT] = []

    def update(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        """Update the measurement.

        Args:
            skeletons (list[Skeleton]): The skeletons.
            cameras (list[Camera]): The cameras.
        """
        self.calculate(skeletons, skeleton_keypoint_validitites, cameras, fit_results)
        self.history.append(self._values)

    @abstractmethod
    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        """Update the measurement.

        Args:
            skeletons (list[Skeleton]): The skeletons.
            cameras (list[Camera]): The cameras.
        """

    def get_composite_measurements(self) -> list[Measurement]:
        """Return the composite measurements."""
        return []


class ScalarMeasurement(Measurement[np.double | float]):
    """A base class for vectorial measurements."""


class BaseVectorMeasurement(Measurement[npt.NDArray[np.double]]):
    """A base class for vectorial measurements."""


class ScalarVectorMeasurement(BaseVectorMeasurement):
    """A class for the measurement of the vector between two points."""
    def __init__(self, x: float, y: float, z: float):
        super().__init__()

        self.x = x
        self.y = y
        self.z = z

    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        self._values = [
            np.array([self.x, self.y, self.z]) for _ in skeletons
        ]


class VectorMeasurement(BaseVectorMeasurement):
    """A class for the measurement of the vector between two points."""
    def __init__(self, point_from: Measurement, point_to: Measurement):
        super().__init__()

        self.point_from: Measurement = point_from
        self.point_to: Measurement = point_to

    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        zipped = zip(self.point_from.values, self.point_to.values)
        self._values = [
            value_to - value_from for value_from, value_to in zipped
        ]

    def get_composite_measurements(self) -> list[Measurement]:
        return [self.point_from, self.point_to]


class SkeletonPointMeasurement(BaseVectorMeasurement):
    """A class for the measurement of a single keypoint of the skeleton."""
    def __init__(self,
                 keypoint_index: int,
                 label: str = '',
                 unit: str = '',
                 color: Annotated[Sequence[int], 4] | None = None,
                 font_size: int = 12):
        super().__init__()

        self.keypoint_index: int = keypoint_index
        self.label: str = label
        self.unit: str = unit
        self.color: Annotated[Sequence[int],
                              4] = color if color is not None else [255]*4
        self.font_size: int = font_size

    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        self._values = []
        
        for i, skeleton in enumerate(skeletons):
            if self.keypoint_index in fit_results and fit_results[self.keypoint_index]['supports'][i]:
                self._values.append(skeleton[self.keypoint_index])
            else:
                self._values.append(np.array([0, 0, 0]))


class MeanSkeletonPointMeasurement(BaseVectorMeasurement):
    """A class for the measurement of the mean of multiple keypoints of the skeleton."""
    def __init__(self, keypoint_indizes: list[int]):
        super().__init__()

        self.keypoint_indizes: list[int] = keypoint_indizes

    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        self._values = [
            np.array([
                skeleton[keypoint_index] for keypoint_index in self.keypoint_indizes
            ]).mean(axis=0)
            for skeleton in skeletons
        ]


class DistanceMeasurement(ScalarMeasurement):
    """A class for the measurement of the distance between two points."""

    def __init__(self,
                 measurement_point_a: Measurement,
                 measurement_point_b: Measurement,
                 label: str = '',
                 unit: str = '',
                 color: Annotated[Sequence[int], 4] | None = None,
                 font_size: int = 12):
        super().__init__()

        self.measurement_point_a: Measurement = measurement_point_a
        self.measurement_point_b: Measurement = measurement_point_b
        self.label: str = label
        self.unit: str = unit
        self.color: Annotated[Sequence[int],
                              4] = color if color is not None else [255]*4
        self.font_size: int = font_size

    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        zipped = zip(self.measurement_point_a.values,
                     self.measurement_point_b.values)
        self._values = [
            np.double(np.linalg.norm(value_a - value_b) if np.linalg.norm(value_a) > 0 and np.linalg.norm(value_b) > 0 else 0) for value_a, value_b in zipped
        ]

    def get_composite_measurements(self) -> list[Measurement]:
        return [self.measurement_point_a, self.measurement_point_b]


class AngleMeasurement(ScalarMeasurement):
    """A class for the measurement of the angle between two vectors."""
    def __init__(self,
                 vector_measurement_a: Measurement[npt.NDArray[np.double]],
                 vector_measurement_b: Measurement[npt.NDArray[np.double]],
                 direction_measurement: Measurement[npt.NDArray[np.double]],
                 label: str = '',
                 unit: str = '°',
                 color: Annotated[Sequence[int], 4] | None = None,
                 font_size: int = 12):
        super().__init__()

        self.vector_measurement_a: Measurement[npt.NDArray[np.double]] = vector_measurement_a
        self.vector_measurement_b: Measurement[npt.NDArray[np.double]] = vector_measurement_b
        self.direction_vector_measurement: Measurement[npt.NDArray[np.double]] = direction_measurement
        self.label: str = label
        self.unit: str = unit
        self.color: Annotated[Sequence[int], 4] = color if color is not None else [255]*4
        self.font_size: int = font_size

    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        self._values = []
        zipped = zip(
            self.vector_measurement_a.values,
            self.vector_measurement_b.values,
            self.direction_vector_measurement.values
        )

        for A, B, U in zipped:
            right_way = np.linalg.norm(A+B) < np.linalg.norm(A+B+U)
            num = np.dot(A, B)
            den = np.linalg.norm(A) * np.linalg.norm(B)

            value = np.double(np.rad2deg(np.arccos(
                np.clip(
                    num / den,
                    -1.0,
                    1.0
                )
            ))) if den != 0 else 0

            self._values.append(value if right_way else 360 - value)

    def get_composite_measurements(self) -> list[Measurement]:
        return [self.vector_measurement_a, self.vector_measurement_b, self.direction_vector_measurement]


class DistanceSumMeasurement(ScalarMeasurement):
    """A class for the measurement of the sum of the distances between
    multiple keypoints of the skeleton."""
    def __init__(self,
                 distance_measurements: list[ScalarMeasurement],
                 label: str = '',
                 unit: str = '°',
                 color: Annotated[Sequence[int], 4] | None = None,
                 font_size: int = 12):
        super().__init__()

        self.distance_measurements: list[ScalarMeasurement] = distance_measurements
        self.label: str = label
        self.unit: str = unit
        self.color: Annotated[Sequence[int], 4] = color if color is not None else [255]*4
        self.font_size: int = font_size

    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        self._values = np.array([
            distance_measurement.values for distance_measurement in self.distance_measurements
        ]).sum(axis=0)

    def get_composite_measurements(self) -> list[Measurement]:
        return self.distance_measurements


class FloorAngleMeasurment(BaseVectorMeasurement):
    """A class for the measurement of the angle between the floor and the skeleton."""

    def __init__(self,
                 label: str = 'Floor Angles',
                 unit: str = '°',
                 color: Annotated[Sequence[int], 4] | None = None,
                 font_size: int = 12):
        super().__init__()

        self.label: str = label
        self.unit: str = unit
        self.color: Annotated[Sequence[int],
                              4] = color if color is not None else [255]*4
        self.font_size: int = font_size

    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        self._values = []

        for skeleton in skeletons:
            shoulder_center: npt.NDArray[np.double] = (
                skeleton[KPT_IDXS.RIGHT_SHOULDER] + skeleton[KPT_IDXS.LEFT_SHOULDER]
            ) / 2
            hip_center: npt.NDArray[np.double] = (
                skeleton[KPT_IDXS.RIGHT_HIP] + skeleton[KPT_IDXS.LEFT_HIP]
            ) / 2

            dist: npt.NDArray[np.double] = shoulder_center - hip_center
            alpha = np.rad2deg(np.arctan(dist[1] / dist[2]) if dist[2] != 0 else 0)
            beta = np.rad2deg(np.arctan(dist[0] / dist[1]) if dist[1] != 0 else 0)
            gamma = np.rad2deg(np.arctan(dist[0] / dist[2]) if dist[2] != 0 else 0)

            self._values.append(np.array([alpha, beta, gamma]))



class CrossProductMeasurement(BaseVectorMeasurement):
    """A class for the measurement of the sum of the distances between
    multiple keypoints of the skeleton."""

    def __init__(self,
                 vector_a: BaseVectorMeasurement,
                 vector_b: BaseVectorMeasurement,
                 label: str = '',
                 unit: str = '°',
                 color: Annotated[Sequence[int], 4] | None = None,
                 font_size: int = 12):
        super().__init__()

        self.vector_a: BaseVectorMeasurement = vector_a
        self.vector_b: BaseVectorMeasurement = vector_b
        self.label: str = label
        self.unit: str = unit
        self.color: Annotated[Sequence[int], 4] = color if color is not None else [255]*4
        self.font_size: int = font_size

    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        self._values = []

        for a, b in zip(self.vector_a.values, self.vector_b.values):
            self._values.append(np.cross(a, b))

    def get_composite_measurements(self) -> list[Measurement]:
        return [self.vector_a, self.vector_b]


class FloorVectorApproximateMeasurement(BaseVectorMeasurement):
    """A class for the measurement of the angle between the floor and the skeleton."""
    def __init__(self,
                 label: str = '',
                 unit: str = '°',
                 color: Annotated[Sequence[int], 4] | None = None,
                 font_size: int = 12):
        super().__init__()

        self.label: str = label
        self.unit: str = unit
        self.color: Annotated[Sequence[int], 4] = color if color is not None else [255]*4
        self.font_size: int = font_size
        self.mean_floor_vector: npt.NDArray[np.double] = np.zeros(3)
        self.mean_floor_vector_hat: npt.NDArray[np.double] = np.array([1.0, 0.0, 0.0])
        self.mean_count: int = 0

    def calculate(self, skeletons: list[Skeleton], skeleton_keypoint_validitites: list[SkeletonValidity], cameras: list[Camera], fit_results: Any) -> None:
        """Calculates the angles of the skeleton to make it upright.

        Args:
            skeleton (Skeleton): The skeleton of the skeleton.
        """
        self._values: list[npt.NDArray[np.double]] = []

        for skeleton in skeletons:
            shoulder_center: npt.NDArray[np.double] = (
                skeleton[KPT_IDXS.RIGHT_SHOULDER] + skeleton[KPT_IDXS.LEFT_SHOULDER]
            ) / 2
            hip_center: npt.NDArray[np.double] = (
                skeleton[KPT_IDXS.RIGHT_HIP] + skeleton[KPT_IDXS.LEFT_HIP]
            ) / 2

            floor_vector: npt.NDArray[np.double] = hip_center - shoulder_center
            self.mean_floor_vector += ((self.mean_floor_vector * self.mean_count) + floor_vector) / (self.mean_count + 1)
            self.mean_count += 1

        self.mean_floor_vector_hat = self.mean_floor_vector / np.linalg.norm(self.mean_floor_vector)

        self._values = [ self.mean_floor_vector_hat for _ in range(len(skeletons)) ]


class MeasurementManager:
    """A class for the management of measurements."""
    def __init__(self, measurements: Sequence[Measurement]):
        """Initialize the measurement manager.

        Args:
            measurements (Sequence[Measurement]): The measurements.
        """

        self.measurements: Sequence[Measurement] = measurements
        self.all_measurements: list[Measurement] = []

        for measurement in self.measurements:
            self.__check_measurement(measurement)

    def update(self,
               skeletons: list[Skeleton],
               skeleton_validities: list[bool],
               skeleton_keypoint_validitites: list[SkeletonValidity],
               cameras: list[Camera],
               fit_results: Any
            ) -> tuple[list[list[str]], list[list[np.double]]]:
        """Update all the measurements managed by the MeasurementManager.

        Args:
            skeltons (list[Skeleton]): The skeletons.
            cameras (list[Camera]): The cameras.
        """
        for measurement in self.all_measurements:
            measurement.update(skeletons, skeleton_keypoint_validitites, cameras, fit_results)

        values: list[list[str]] = []
        orig_values: list[list[np.double]] = []

        if len(self.measurements) > 0:
            for _ in range(len(self.measurements[0].values)):
                values.append([])
                orig_values.append([])

        for measurement in self.measurements:
            for i, value in enumerate(measurement.values):
                if isinstance(measurement, BaseVectorMeasurement):
                    values[i].append('%s: [%.0f, %.0f, %.0f]%s' % (measurement.label, value[0], value[1], value[2], measurement.unit))
                else:
                    values[i].append('%s: %.1f%s' % (measurement.label, value, measurement.unit))
                orig_values[i].append(value)

        return values, orig_values

    def __check_measurement(self, measurement: Measurement) -> None:
        for composite_measurement in measurement.get_composite_measurements():
            self.__check_measurement(composite_measurement)

        if measurement not in self.all_measurements:
            self.all_measurements.append(measurement)

class DefaultMeasurements:
    """A class for the default measurements."""
    right_shoulder = SkeletonPointMeasurement(KPT_IDXS.RIGHT_SHOULDER)
    right_elbow = SkeletonPointMeasurement(KPT_IDXS.RIGHT_ELBOW)
    left_elbow = SkeletonPointMeasurement(KPT_IDXS.LEFT_ELBOW)
    right_hand = SkeletonPointMeasurement(KPT_IDXS.RIGHT_HAND)
    left_hand = SkeletonPointMeasurement(KPT_IDXS.LEFT_HAND)
    left_foot = SkeletonPointMeasurement(KPT_IDXS.LEFT_FOOT)
    left_knee = SkeletonPointMeasurement(KPT_IDXS.LEFT_KNEE)
    left_hip = SkeletonPointMeasurement(KPT_IDXS.LEFT_HIP)
    left_shoulder = SkeletonPointMeasurement(KPT_IDXS.LEFT_SHOULDER)
    right_foot = SkeletonPointMeasurement(KPT_IDXS.RIGHT_FOOT)
    right_knee = SkeletonPointMeasurement(KPT_IDXS.RIGHT_KNEE)
    right_hip = SkeletonPointMeasurement(KPT_IDXS.RIGHT_HIP)
    right_shoulder = SkeletonPointMeasurement(KPT_IDXS.RIGHT_SHOULDER)
    neck = MeanSkeletonPointMeasurement([KPT_IDXS.RIGHT_SHOULDER, KPT_IDXS.LEFT_SHOULDER])
    middle_feet = MeanSkeletonPointMeasurement([KPT_IDXS.RIGHT_FOOT, KPT_IDXS.LEFT_FOOT])
    tailbone = MeanSkeletonPointMeasurement([KPT_IDXS.RIGHT_HIP, KPT_IDXS.LEFT_HIP])
    knee_center = MeanSkeletonPointMeasurement([KPT_IDXS.RIGHT_KNEE, KPT_IDXS.LEFT_KNEE])
    nose = SkeletonPointMeasurement(KPT_IDXS.NOSE)
