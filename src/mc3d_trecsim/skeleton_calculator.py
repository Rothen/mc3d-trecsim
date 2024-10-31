"""A module for calculating the skeleton of a 3D pose."""

from typing import Callable, TypeAlias
import numpy as np
import numpy.typing as npt

from .gmm import GMM
from .mc3d_types import FitResults, Skeleton, SkeletonPath, SkeletonValidity

def default_time_calc(gmm: GMM, fit_results: FitResults) -> tuple[float, float, float]:
    """Calculates the time for the skeleton calculation."""
    t_0 = gmm.frames[0].time
    t_n = gmm.frames[-1].time
    t_start = (gmm.frames[0].time + gmm.frames[-1].time) / 2

    return t_0, t_n, t_start

TimeFunction: TypeAlias = Callable[[GMM, FitResults], tuple[float, float, float]]
CalulationResult: TypeAlias = tuple[list[Skeleton],
                                    list[SkeletonPath],
                                    list[bool],
                                    list[SkeletonValidity]]

class SkeletonCalculator:
    """A class for calculating the skeleton of a 3D pose."""
    def __init__(self, keypoints: list[int] | None = None, head: int = 40, tail: int = 40, rotation_matrix: npt.NDArray[np.double] = np.eye(3, 3), translation_vector: npt.NDArray[np.double] = np.zeros((3, 1))) -> None:
        """Initializes the skeleton calculator."""
        self.keypoints = keypoints if keypoints is not None else []
        self.head = head
        self.tail = tail
        self.path_desing_matrix = np.array([])
        self.point_desing_matrix = np.array([])
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector

    def calculate(self,
                  gmm: GMM,
                  fit_results: FitResults,
                  time_fn: TimeFunction = default_time_calc
                  ) -> tuple[list[Skeleton], list[SkeletonPath], list[bool], list[SkeletonValidity]]:
        """Calculates the 3D skeletons using a fit result.

        Args:
            gmm: GMM
            fit_results: FitResults

        Returns:
            tuple[list[Skeleton], list[SkeletonPath], list[bool], list[SkeletonValidity]]:
                The skeletons, the skeleton paths, the skeletons' validity
                and the skeletons' keypoints validity.
        """
        skeletons: list[Skeleton] = []
        skeleton_paths: list[SkeletonPath] = []
        skeleton_validities: list[bool] = []
        skeleton_keypoint_validitites: list[SkeletonValidity] = []

        t_0, t_n, t_start = time_fn(gmm, fit_results)
        self.path_desing_matrix = gmm.spline.designMatrix(
            np.linspace(t_0, t_n, self.head + self.tail + 1)
        )
        self.point_desing_matrix = gmm.spline.designMatrix(np.array([t_start]))

        for p, person in enumerate(self._separate_people(gmm, fit_results)):
            thetas: npt.NDArray[np.double] = np.array([
                person[KEYPOINT] for KEYPOINT in self.keypoints
            ])

            skeleton, paths = self._calculate_skeleton(thetas)
            skeleton_valid, skeleton_keypoint_validity = self._calculate_skeleton_validity(skeleton, fit_results, p)

            skeletons.append(skeleton)
            skeleton_paths.append(paths)
            skeleton_validities.append(skeleton_valid)
            skeleton_keypoint_validitites.append(skeleton_keypoint_validity)

        return skeletons, skeleton_paths, skeleton_validities, skeleton_keypoint_validitites

    def _calculate_skeleton_validity(self, skeleton: Skeleton, fit_results: FitResults, person_index: int) -> tuple[bool, SkeletonValidity]:
        """Calculates the skeleton validity."""
        skeleton_valid: bool = False
        skeleton_keypoint_validity: SkeletonValidity = {}

        for keypoint in self.keypoints:
            valid: bool = bool(1500 >= np.linalg.norm(skeleton[keypoint]) >= 1) and fit_results[keypoint]['supports'][person_index]
            skeleton_keypoint_validity[keypoint] = valid
            if valid:
                skeleton_valid = True

        return skeleton_valid, skeleton_keypoint_validity

    def _separate_people(self, gmm: GMM, fit_result: FitResults) -> list[Skeleton]:
        """Separates the people from the fit result."""
        people: list[Skeleton] = [{
            kpt_idx: fit_result[kpt_idx]['theta'][:, j*3:(j+1)*3].T.flatten()
                for kpt_idx in self.keypoints
            }
            for j in range(gmm.J)
        ]

        return people

    def _calculate_skeleton(self,
                            thetas: npt.NDArray[np.double]
                            ) -> tuple[Skeleton, dict[int, npt.NDArray[np.double]]]:
        """Draws a skeleton in 3D.

        Args:
            person (Skeleton): The person.

        Returns:
            Skeleton: The skeleton.
        """

        skeleton: Skeleton = {}
        paths: dict[int, npt.NDArray[np.double]] = {}

        for theta, keypoint in zip(thetas, self.keypoints):
            w = theta.reshape((3, self.point_desing_matrix.shape[1])).T
            p_ws: npt.NDArray[np.double] = self.point_desing_matrix @ w
            path_ws: npt.NDArray[np.double] = self.path_desing_matrix @ w

            p_ws = (self.rotation_matrix @ p_ws.T + self.translation_vector).T
            path_ws = (self.rotation_matrix @ path_ws.T + self.translation_vector).T

            skeleton[keypoint] = np.array([p_ws[0, 0], p_ws[0, 1], p_ws[0, 2]])
            paths[keypoint] = path_ws

        return skeleton, paths
