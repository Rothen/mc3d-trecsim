"""A module for visualizing 3D data in real-time."""
import sys
from threading import Event
from typing import Annotated, Literal, Sequence, TypeVar, Generic, TypeAlias
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from PyQt6 import QtCore
import PyQt6.QtWidgets as QtWidgets

from .gmm import Camera
from .enums import KPT_COLORS, LIMB_COLORS, LIMB_KPTS, Color, COLORS
from .interval import Interval
from .mc3d_types import ColorLike, ColorsLike, Skeleton, SkeletonPath, SkeletonValidity

QtWidgets.QApplication.setAttribute(
    QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)


Limb: TypeAlias = tuple[Annotated[Sequence[int], 2], Color]
IT = TypeVar('IT')


class Visualizer(ABC, Generic[IT]):
    """A class to represent a visualizer for 3D data."""

    def __init__(self,
                 keypoints: Sequence[int] | None = None,
                 rotation_matrix: npt.NDArray[np.double] = np.eye(3),
                 translation_vector: npt.NDArray[np.double] = np.zeros((3, 1)),
                 correct_axis: bool = False,
                 scale_factor: float = 1.0,
                 project_feet_to_ground: bool = False,
                 additional_limbs: list[Limb] | None = None,
                 ) -> None:
        """Initializes the visualizer.

        Args:
            keypoints (list[int], optional): The keypoints of the skeleton. Defaults to [].
            rotation_matrix (npt.NDArray[np.double], optional): A 3x3 rotation matrix.
                Defaults to np.eye(3).
            translation_vector (npt.NDArray[np.double], optional): A 3x1 translation vector.
                Defaults to np.zeros((3, 1)).
            correct_axis (bool, optional): Whether to correct the axis. Defaults to False.
            scale_factor (float, optional): A scale factor. Defaults to 1.0.
            project_feet_to_ground (bool, optional): Whether to project the feet to the ground.
                Defaults to False.
            additional_limbs (list[Limb], optional): Additional
                limbs to draw. Defaults to [].
        """
        self.rotation_matrix: npt.NDArray[np.double] = rotation_matrix
        self.translation_vector: npt.NDArray[np.double] = translation_vector
        self.keypoints: Sequence[int] = keypoints if keypoints is not None else []
        self.additional_limbs: list[Limb] = \
            additional_limbs if additional_limbs is not None else []
        self.path_t_start: float = -1
        self.skeletons: list[Skeleton] = []
        self.correct_axis: bool = correct_axis
        self.scale_factor: float = scale_factor
        self.project_feet_to_ground: bool = project_feet_to_ground
        self.all_items: list[IT] = []
        self.skeleton_items: list[IT] = []
        self.has_changes = Event()
        self.people: list[Skeleton] = []
        self.path_t: float = 0
        self.path_t_end: float = 0
        self.path_design_matrix: npt.NDArray[np.double] = np.array([])
        self.point_desing_matrix: npt.NDArray[np.double] = np.array([])

    @abstractmethod
    def remove_item(self, item: IT) -> None:
        """Removes an item from the visualizer.

        Args:
            item (IT): The item to remove.
        """

    @abstractmethod
    def line(self,
             pos: npt.NDArray[np.double],
             color: ColorsLike = (1.0, 0.0, 0.0, 1.0),
             linewidth: int = 2,
             markersize: int = 3,
             draw_point: bool = False
             ) -> list[IT]:
        """Draws a line.

        Args:
            pos (npt.NDArray[np.double]): The points of the line.
            color (ColorsLike, optional):
                The color of the line. Defaults to (1.0, 0.0, 0.0, 1.0).
            linewidth (int, optional): The width of the line. Defaults to 2.
            markersize (int, optional): The size of the markers. Defaults to 3.
            draw_point (bool, optional): The draw the point. Defaults to False.

        Returns:
            list[IT]: The drawn items.
        """

    @abstractmethod
    def scatter(self,
                pos: npt.NDArray[np.double],
                color: ColorsLike = (1.0, 0.0, 0.0, 1.0),
                markersize: int = 3,
                px_mode: bool = True) -> list[IT]:
        """Draws a scatter plot.

        Args:
            pos (npt.NDArray[np.double]): The points of the scatter plot.
            color (ColorsLike, optional):
                The color of the scatter plot. Defaults to (1.0, 0.0, 0.0, 1.0).
            markersize (int, optional): The size of the markers. Defaults to 3.
            px_mode (bool, optional): Whether to use pixel mode. Defaults to True.

        Returns:
            list[IT]: The drawn items.
        """

    @abstractmethod
    def draw_measurements(self, measurements: list[list[str]]) -> None:
        """Draws the measurements.

        Args:
            measurements (list[list[str]]): The measurements to draw.
        """

    @abstractmethod
    def run_app(self) -> None:
        """Runs the visualizer."""

    def show(self) -> None:
        """Displays the visualizer and updates itself 25 times per second."""
        def update():
            if self.has_changes.is_set():
                self.has_changes.clear()
                for item in self.all_items:
                    self.remove_item(item)

                self.all_items = []

                for person in self.people:
                    self.all_items += self.draw_person(
                        np.array([person[KEYPOINT]
                                 for KEYPOINT in self.keypoints])
                    )

        interval = Interval(5, 1/25, update)
        interval.start()

        sys.exit(self.run_app())

    def draw_camera(self,
                    extrinsic_matrix: npt.NDArray[np.double],
                    cam_size: float,
                    cam_edge: int,
                    cam_color: ColorLike = (1.0, 1.0, 1.0, 1.0),
                    rotation_matrix: npt.NDArray[np.double] = np.eye(3),
                    translation_vector: npt.NDArray[np.double] = np.zeros((3, 1))
                    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        """Draws a fake camera on a figure for visual inspection.

        Args:
            extrinsic_matrix (npt.NDArray[np.double]): The 4x4 extrinsic camera matrix.
            cam_size (float): The size of the camera.
            cam_edge (int): The edge of the camera.
            cam_color (ColorLike, optional): The color of the camera.
                Defaults to (1.0, 1.0, 1.0, 1.0).

        Returns:
            tuple[NDArray[double], NDArray[double]]: A tuple consisting of the center point
                of the camera and the bounding box of the camera.
        """

        points = np.array([
            [cam_size,       -cam_size,      -2*cam_size],
            [cam_size,       cam_size,       -2*cam_size],
            [-cam_size,      -cam_size,      -2*cam_size],
            [-cam_size,      cam_size,       -2*cam_size],
            [cam_size,       -cam_size,      2*cam_size],
            [cam_size,       cam_size,       2*cam_size],
            [-cam_size,      -cam_size,      2*cam_size],
            [-cam_size,      cam_size,       2*cam_size],
            [1.5*cam_size,   -1.5*cam_size,  3*cam_size],
            [1.5*cam_size,   1.5*cam_size,   3*cam_size],
            [-1.5*cam_size,  -1.5*cam_size,  3*cam_size],
            [-1.5*cam_size,  1.5*cam_size,   3*cam_size],
            [0,              0,              0],
            [0,              0,              2*cam_size],
            [0,              2*cam_size,     0],
            [2*cam_size,     0,              0]
        ])
        homo_points = np.hstack([points, np.ones((points.shape[0], 1))])
        rot_pnts = (extrinsic_matrix @ homo_points.T).T[:, :3]
        rot_pnts = (rotation_matrix @ rot_pnts.T + translation_vector).T

        self.draw_3d_line(rot_pnts[0], rot_pnts[1],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[0], rot_pnts[2],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[3], rot_pnts[2],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[1], rot_pnts[3],
                          color=cam_color, linewidth=cam_edge)

        self.draw_3d_line(rot_pnts[4], rot_pnts[5],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[4], rot_pnts[6],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[7], rot_pnts[6],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[5], rot_pnts[7],
                          color=cam_color, linewidth=cam_edge)

        self.draw_3d_line(rot_pnts[0], rot_pnts[4],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[1], rot_pnts[5],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[2], rot_pnts[6],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[3], rot_pnts[7],
                          color=cam_color, linewidth=cam_edge)

        self.draw_3d_line(rot_pnts[8], rot_pnts[9],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[8], rot_pnts[10],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[11], rot_pnts[10],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[9], rot_pnts[11],
                          color=cam_color, linewidth=cam_edge)

        self.draw_3d_line(rot_pnts[4], rot_pnts[8],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[5], rot_pnts[9],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[6], rot_pnts[10],
                          color=cam_color, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[7], rot_pnts[11],
                          color=cam_color, linewidth=cam_edge)

        self.draw_3d_line(rot_pnts[12], rot_pnts[13],
                          color=COLORS.RED, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[12], rot_pnts[14],
                          color=COLORS.BLUE, linewidth=cam_edge)
        self.draw_3d_line(rot_pnts[12], rot_pnts[15],
                          color=COLORS.GREEN, linewidth=cam_edge)

        return rot_pnts[12], rot_pnts

    def visualise_cameras(self,
                          cameras: list[Camera],
                          size: int = 10,
                          edge: int = 2,
                          color: Color = (1.0, 1.0, 1.0, 1.0)
                          ) -> None:
        """Visualizes the cameras in 3D.

        Args:
            cameras (list[Camera]): The cameras to visualize.
            size (int, optional): The size of the camera. Defaults to 10.
            edge (int, optional): The edge size of the camera. Defaults to 2.
            color (Color, optional): The color of the camera. Defaults to (1.0, 1.0, 1.0, 1.0).
        """
        for camera in cameras:
            _ = self.draw_camera(camera.P, size, edge, color)

    def visualise_skeletons(self,
                    skeletons: list[Skeleton],
                    skeleton_paths: list[SkeletonPath],
                    skeleton_validities: list[bool],
                    skeleton_keypoint_validitites: list[SkeletonValidity],
                    measurements: list[list[str]],
                    head: int = 40,
                    tail: int = 40
                    ) -> None:
        """Draws a person in 3D.

        Args:
            thetas (npt.NDArray[np.double]): The thetas of the person.
            head (int, optional): The head of the persons path. Defaults to 40.
            tail (int, optional): The tail of the persons path. Defaults to 40.
        """

        for skeleton_item in self.skeleton_items:
            self.remove_item(skeleton_item)

        self.skeleton_items = []
        self.skeletons = []

        for skeleton, skeleton_path, skeleton_validity, skeleton_keypoint_validity in zip(skeletons, skeleton_paths, skeleton_validities, skeleton_keypoint_validitites):
            if skeleton_validity:
                self.skeleton_items += self.draw_skeleton(skeleton, skeleton_keypoint_validity)
                self.skeleton_items += self.draw_skeleton_paths(skeleton_path, skeleton_keypoint_validity, head, tail)
                self.draw_measurements(measurements)

    def draw_skeleton(self, skeleton: Skeleton, skeleton_validity: SkeletonValidity) -> list[IT]:
        """Draws a skeleton in 3D.

        Args:
            skeleton (Skeleton): The skeleton to draw.

        Returns:
            list[IT]: The drawn items.
        """

        items: list[IT] = []

        for limb_index, limb_joints in LIMB_KPTS.items():
            are_joints_valid = True

            for joint in limb_joints:
                if joint not in skeleton or not skeleton_validity[joint]:
                    are_joints_valid = False
                    break

            if are_joints_valid:
                items += self.draw_3d_line(
                    skeleton[limb_joints[0]],
                    skeleton[limb_joints[1]],
                    color=LIMB_COLORS[limb_index],
                    draw_point=True,
                    linewidth=6
                )

        for joints, color in self.additional_limbs:
            are_joints_valid = True

            for joint in joints:
                if joint not in skeleton or not skeleton_validity[joint]:
                    are_joints_valid = False
                    break

            if are_joints_valid:
                items += self.draw_3d_line(
                    skeleton[joints[0]],
                    skeleton[joints[1]],
                    color=color,
                    draw_point=True,
                    linewidth=6
                )
        self.skeletons.append(skeleton)
        return items

    def draw_skeleton_paths(self,
                            skeleton_paths: SkeletonPath,
                            skeleton_validity: SkeletonValidity,
                            head: int = 40,
                            tail: int = 40
                            ) -> list[IT]:
        """Draws the paths of the skeleton.

        Args:
            skeleton_paths (dict[int, npt.NDArray[np.double]]): The paths of the skeleton.
            head (int): The head of the path.
            tail (int): The tail of the path.

        Returns:
            list[IT]: The drawn items.
        """
        resolution = head + tail + 1

        alpha_start = 0.3
        alpha_center = 1.0
        alpha_end = 0.3

        alpha = np.hstack([
            np.linspace(alpha_start, alpha_center,
                        head + 1)[:-1], [alpha_center],
            np.linspace(alpha_center, alpha_end, tail + 1)[1:]
        ])

        color = np.zeros((resolution, 4))
        color[:, 3] = alpha

        items = []

        for keypoint, skeleton_path in skeleton_paths.items():
            if skeleton_validity[keypoint]:
                color[:, :3] = KPT_COLORS[keypoint][:3]

                items += self.draw_3d_path(
                    skeleton_path,
                    color=[
                        tuple(c) for c in color
                    ],
                    linewidth=4
                )

        return items

    def preprocess_points(self,
                          pos: npt.NDArray[np.double],
                          reposition: bool = True
                          ) -> npt.NDArray[np.double]:
        """Preprocesses the points.

        Args:
            pos (npt.NDArray[np.double]): The points to preprocess.
            reposition (bool, optional): Whether to reposition the points. Defaults to True.

        Returns:
            npt.NDArray[np.double]: The preprocessed points.
        """
        if reposition:
            pos = self._inherent_reposition(pos)

        if self.correct_axis:
            temp = np.copy(pos[:, 1])
            pos[:, 1] = pos[:, 2]
            pos[:, 2] = -temp

        return pos * self.scale_factor

    def draw_3d_line(self,
                     start_point: npt.NDArray[np.double],
                     end_point: npt.NDArray[np.double],
                     color: ColorsLike = (
                         1.0, 0.0, 0.0, 1.0),
                     linewidth: int = 2,
                     markersize: int = 5,
                     draw_point: bool = False
                     ) -> list[IT]:
        """Draws a 3D line.

        Args:
            start_point (npt.NDArray[np.double]): The start point.
            end_point (npt.NDArray[np.double]): The end point.
            color (ColorsLike, optional):
                The color of the line. Defaults to (1.0, 0.0, 0.0, 1.0).
            linewidth (int, optional): The width of the line. Defaults to 2.
            markersize (int, optional): The size of the markers. Defaults to 5.
            draw_point (bool, optional): The draw the point. Defaults to False.

        Returns:
            list[IT]: The drawn items.
        """
        pos = self.preprocess_points(np.array([start_point, end_point]))
        return self.line(pos, color, linewidth, markersize, draw_point)

    def draw_3d_path(self,
                     pos: npt.NDArray[np.double],
                     color: ColorsLike = (
                         1.0, 0.0, 0.0, 1.0),
                     linewidth: int = 2,
                     reposition: bool = True
                     ) -> list[IT]:
        """Draws a 3D path.

        Args:
            pos (npt.NDArray[np.double]): The points of the path.
            color (ColorsLike, optional):
                The color of the path. Defaults to (1.0, 0.0, 0.0, 1.0).
            linewidth (int, optional): The width of the line. Defaults to 2.
            reposition (bool, optional): Whether to reposition the points. Defaults to True.

        Returns:
            list[IT]: The drawn items.
        """
        pos = self.preprocess_points(pos, reposition=reposition)
        return self.line(pos, color, linewidth, draw_point=False)

    def _inherent_reposition(self, pos: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        """Repositions the points to the origin.

        Args:
            pos (npt.NDArray[np.double]): The points to reposition.

        Returns:
            npt.NDArray[np.double]: The repositioned points.
        """
        return (self.rotation_matrix @ pos.T + self.translation_vector).T

    def clear(self) -> None:
        """Clears the visualizer."""
        for item in self.all_items:
            self.remove_item(item)
