"""A module for the parallel PyQt visualizer."""
import ctypes
import logging
import multiprocessing as mp
from multiprocessing import Process
from threading import Thread
from multiprocessing.managers import ListProxy, ValueProxy
from threading import Event
from typing import Annotated, Any, cast, Callable
import time

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QTimer
import PyQt6.QtCore as QtCore
import PyQt6.QtWidgets as QtWidgets
import pyqtgraph.opengl as gl

from .gmm import Camera
from .enums import LIMB_KPTS, LIMBS, Color
from .measurement import Measurement
from .mc3d_types import Limb, Skeleton, SkeletonValidity, SkeletonPath
from .qt_visualizer import QtVisualizer


class ParallelQtVisualizer(Process):
    """A class to represent a parallel visualizer for 3D data using PyQt."""
    @property
    def people(self):
        """Return the people.

        Returns:
            list[Any]: The people.
        """
        return self._people

    @people.setter
    def people(self, values: list):
        self._people[:] = values

    @property
    def path_design_matrix(self):
        """Return the path design matrix.

        Returns:
            npt.NDArray[np.double]: The path design matrix.
        """
        return self._path_design_matrix

    @path_design_matrix.setter
    def path_design_matrix(self, values: list):
        self._path_design_matrix[:] = values

    @property
    def point_desing_matrix(self):
        """Return the point design matrix.

        Returns:
            npt.NDArray[np.double]: The point design matrix.
        """
        return self._point_desing_matrix

    @point_desing_matrix.setter
    def point_desing_matrix(self, values: list):
        self._point_desing_matrix[:] = values

    @property
    def visualise_cameras_data(self):
        """Return the visualise cameras data.

        Returns:
            list[Any]: The visualise cameras data.
        """
        return self._visualise_cameras_data

    @visualise_cameras_data.setter
    def visualise_cameras_data(self, values: list):
        self._visualise_cameras_data[:] = values

    @property
    def visualise_skeletons_data(self):
        """Return the visualise skeletons.

        Returns:
            list[Any]: The visualise skeletons.
        """
        return self._visualise_skeletons_data

    @visualise_skeletons_data.setter
    def visualise_skeletons_data(self, values: list):
        self._visualise_skeletons_data[:] = values

    @property
    def line_data(self):
        """Return the line data.

        Returns:
            tuple[Any]: The line data.
        """
        return self._line_data

    @line_data.setter
    def line_data(self, values: list):
        self._line_data[:] = values

    @property
    def scatter_data(self):
        """Return the scatter data.

        Returns:
            tuple[Any]: The scatter data.
        """
        return self._scatter_data

    @scatter_data.setter
    def scatter_data(self, values: list):
        self._scatter_data[:] = values

    @property
    def mesh_data(self):
        """Return the mesh data.

        Returns:
            tuple[Any]: The mesh data.
        """
        return self._mesh_data
    
    @mesh_data.setter
    def mesh_data(self, values: list):
        self._mesh_data[:] = values

    @property
    def camera_position_data(self):
        """Return the camera position data.

        Returns:
            tuple[Any]: The camera position data.
        """
        return self._camera_position_data

    @camera_position_data.setter
    def camera_position_data(self, values: list):
        self._camera_position_data[:] = values

    @property
    def distance(self) -> float:
        """Return the camera distance.

        Returns:
            float: The camera distance.
        """
        return cast(float, self._distance.value)

    @distance.setter
    def distance(self, value: float):
        self._distance.value = cast(ctypes.c_double, value)

    @property
    def azimuth(self) -> float:
        """Return the camera azimuth.

        Returns:
            float: The camera azimuth.
        """
        return cast(float, self._azimuth.value)

    @azimuth.setter
    def azimuth(self, value: float):
        self._azimuth.value = cast(ctypes.c_double, value)

    @property
    def elevation(self) -> float:
        """Return the camera elevation.

        Returns:
            float: The camera elevation.
        """
        return cast(float, self._elevation.value)

    @elevation.setter
    def elevation(self, value: float):
        self._elevation.value = cast(ctypes.c_double, value)

    @property
    def center_x(self) -> float:
        """Return the line data.

        Returns:
            float: The line data.
        """
        return cast(float, self._center_x.value)

    @center_x.setter
    def center_x(self, value: float):
        self._center_x.value = cast(ctypes.c_double, value)

    @property
    def center_y(self) -> float:
        """Return the line data.

        Returns:
            float: The line data.
        """
        return cast(float, self._center_y.value)

    @center_y.setter
    def center_y(self, value: float):
        self._center_y.value = cast(ctypes.c_double, value)

    @property
    def center_z(self) -> float:
        """Return the line data.

        Returns:
            float: The line data.
        """
        return cast(float, self._center_z.value)

    @center_z.setter
    def center_z(self, value: float):
        self._center_z.value = cast(ctypes.c_double, value)

    @property
    def t_start(self) -> float:
        """Return the start time.

        Returns:
            float: The start time.
        """
        return cast(float, self._t_start.value)

    @t_start.setter
    def t_start(self, value: float):
        self._t_start.value = cast(ctypes.c_double, value)

    @property
    def t(self) -> float:
        """Return the time.

        Returns:
            float: The time.
        """
        return cast(float, self._t.value)

    @t.setter
    def t(self, value: float):
        self._t.value = cast(ctypes.c_double, value)

    @property
    def t_end(self) -> float:
        """Return the end time.

        Returns:
            float: The end time.
        """
        return cast(float, self._t_end.value)

    @t_end.setter
    def t_end(self, value: float):
        self._t_end.value = cast(ctypes.c_double, value)

    def __init__(self,
                 keypoints: list[int] | None = None,
                 rotation_matrix: npt.NDArray[np.double] = np.eye(3),
                 translation_vector: npt.NDArray[np.double] = np.zeros((3, 1)),
                 project_feet_to_ground: bool = False,
                 limbs: list[Limb] = LIMBS,
                 measurements: list[Measurement[Any]] | None = None,
                 max_fps: float = 25,
                 on_pause: Callable[[Any], None] = lambda _: None,
                 on_print: Callable[[Any], None] = lambda _: None,
                 show_floor: bool = True,
                 position: list[int] = None
                 ) -> None:
        """Initializes the visualizer.

        Args:
            keypoints (Sequence[int] | None, optional): The keypoints of the skeleton.
                Defaults to [].
            rotation_matrix (npt.NDArray[np.double], optional): A 3x3 rotation matrix.
                Defaults to np.eye(3).
            translation_vector (npt.NDArray[np.double], optional): A 3x1 translation
                vector. Defaults to np.zeros((3, 1)).
            project_feet_to_ground (bool, optional): Whether to project the feet to the
                ground. Defaults to False.
            limbs (list[tuple[Annotated[list[int], 2], Color]] | None, optional):
                Limbs to draw. Defaults to [].
            measurements (list[Measurement[Any]] | None, optional): The measurements to show.
            max_fps (float): The maximum frames per second. Defaults to 25.
            show_floor (bool, optional): Whether to show the floor. Defaults to True.
        """

        super().__init__()

        self.keypoints: list[int] = keypoints if keypoints is not None else []
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.project_feet_to_ground = project_feet_to_ground
        self.limbs = limbs if limbs is not None else []
        self.measurements = measurements if measurements is not None else []
        self.max_fps: float = max_fps
        self.visualizer: QtVisualizer = QtVisualizer(auto_init=False)
        self.on_pause: Callable[[Any], None] = on_pause
        self.on_print: Callable[[Any], None] = on_print
        self.show_floor: bool = show_floor
        self.position: list[int] = position

        if len(self.keypoints) == 0:
            logging.warning('Visualizer has no keypoints set.')

        manager = mp.Manager()
        self.started: Event = manager.Event()
        self.changed: Event = manager.Event()
        self.is_drawing: Event = manager.Event()
        self.camera_updated: Event = manager.Event()
        self.skeletons_updated: Event = manager.Event()
        self.line_updated: Event = manager.Event()
        self.scatter_updated: Event = manager.Event()
        self.mesh_updated: Event = manager.Event()
        self.camera_position_updated: Event = manager.Event()
        self.activate_window_updated: Event = manager.Event()
        self.on_pause_event: Event = manager.Event()
        self.on_print_event: Event = manager.Event()

        self._people: ListProxy[Any] = manager.list()
        self._path_design_matrix: ListProxy[Any] = manager.list()
        self._point_desing_matrix: ListProxy[Any] = manager.list()
        self._visualise_cameras_data: ListProxy[Any] = manager.list()
        self._visualise_skeletons_data: ListProxy[Any] = manager.list()
        self._line_data: ListProxy[Any] = manager.list()
        self._scatter_data: ListProxy[Any] = manager.list()
        self._mesh_data: ListProxy[Any] = manager.list()
        self._camera_position_data: ListProxy[Any] = manager.list()

        self._distance = manager.Value(
            ctypes.c_double,
            cast(ctypes.c_double, 0.0)
        )
        self._azimuth = manager.Value(
            ctypes.c_double,
            cast(ctypes.c_double, 0.0)
        )
        self._elevation = manager.Value(
            ctypes.c_double,
            cast(ctypes.c_double, 0.0)
        )
        self._center_x = manager.Value(
            ctypes.c_double,
            cast(ctypes.c_double, 0.0)
        )
        self._center_y = manager.Value(
            ctypes.c_double,
            cast(ctypes.c_double, 0.0)
        )
        self._center_z = manager.Value(
            ctypes.c_double,
            cast(ctypes.c_double, 0.0)
        )
        self._t_start = manager.Value(
            ctypes.c_double,
            cast(ctypes.c_double, 0.0)
        )
        self._t = manager.Value(
            ctypes.c_double,
            cast(ctypes.c_double, 0.0)
        )
        self._t_end = manager.Value(
            ctypes.c_double,
            cast(ctypes.c_double, 0.0)
        )

        self.event_checker = Thread(target=self.check_events)

    def check_events(self) -> None:
        """Checks the events."""
        while self.is_alive():
            if self.on_pause_event.is_set():
                self.on_pause_event.clear()
                self.on_pause(self)

            if self.on_print_event.is_set():
                self.on_print_event.clear()
                self.on_print(self)
            time.sleep(1/100)

    def start(self) -> None:
        self.show()

    def show(self) -> None:
        """Start the process and show the visualizer."""
        super().start()

        while not self.started.wait(0):
            pass

        self.event_checker.start()

    def visualise_cameras(self,
                          cameras: list[Camera],
                          size: int = 10,
                          edge: int = 2,
                          color: Color = (1.0, 1.0, 1.0, 1.0),
                          rotation_matrix: npt.NDArray[np.double] = np.eye(3),
                          translation_vector: npt.NDArray[np.double] = np.zeros((3, 1))
                          ) -> None:
        """Visualizes the cameras in 3D.

        Args:
            cameras (list[Camera]): The cameras to visualize.
            size (int, optional): The size of the camera. Defaults to 10.
            edge (int, optional): The edge size of the camera. Defaults to 2.
            color (Color, optional): The color of the camera. Defaults to (1.0, 1.0, 1.0, 1.0).
        """
        self.visualise_cameras_data = [
            (camera.P, size, edge, color, rotation_matrix, translation_vector) for camera in cameras]
        self.camera_updated.set()

    def visualise_skeletons(self,
                    skeletons: list[Skeleton],
                    paths: list[SkeletonPath],
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

        Returns:
            list[IT]: The drawn items.
        """
        self.visualise_skeletons_data = [(skeletons, paths, skeleton_validities, skeleton_keypoint_validitites, measurements, head, tail)]
        self.skeletons_updated.set()

    def line(self,
             pos: npt.NDArray[np.double],
             color: Color | npt.NDArray[np.double] | list[tuple[float, ...]] = (
                 1.0, 0.0, 0.0, 1.0),
             linewidth: int = 2,
             markersize: int = 10,
             draw_point: bool = True
             ) -> None:
        """Draws a line.

        Args:
            pos (npt.NDArray[np.double]): The points of the line.
            color (Color | npt.NDArray[np.double] | list[tuple[float, ...]], optional):
                The color of the line. Defaults to (1.0, 0.0, 0.0, 1.0).
            linewidth (int, optional): The width of the line. Defaults to 2.
            markersize (int, optional): The size of the markers. Defaults to 3.
            draw_point (bool, optional): The draw the point. Defaults to False.

        Returns:
            list[IT]: The drawn items.
        """
        self.line_data = [(pos, color, linewidth, markersize, draw_point)]
        self.line_updated.set()

    def scatter(self,
                pos: npt.NDArray[np.double],
                color: Color | npt.NDArray[np.double] | list[tuple[float, ...]] = (
                    1.0, 0.0, 0.0, 1.0),
                size: int = 3,
                px_mode: bool = True
                ) -> None:
        """Draws a scatter plot.

        Args:
            pos (npt.NDArray[np.double]): The points of the scatter plot.
            color (Color | npt.NDArray[np.double] | list[tuple[float, ...]], optional):
                The color of the scatter plot. Defaults to (1.0, 0.0, 0.0, 1.0).
            markersize (int, optional): The size of the markers. Defaults to 3.
            px_mode (bool, optional): Whether to use pixel mode. Defaults to True.

        Returns:
            list[IT]: The drawn items.
        """
        self.scatter_data = self.scatter_data + [(pos, color, size, px_mode)]
        self.scatter_updated.set()
        
    def mesh(self, vertices: npt.NDArray[np.double], faces: npt.NDArray[np.double],
             color: Color | npt.NDArray[np.double] | list[tuple[float, ...]] = (1.0, 0.0, 0.0, 1.0),
             edge_color: Color | npt.NDArray[np.double] | list[tuple[float, ...]] = (1.0, 0.0, 0.0, 1.0),
             draw_edges: bool = False, draw_faces: bool = True, only_projection: bool = False):
        self.mesh_data = self.mesh_data + \
            [(vertices, faces, color, edge_color, draw_edges, draw_faces, only_projection)]
        self.mesh_updated.set()

    def get_camera_position(self) -> tuple[float, float, float, list[float]]:
        """Gets the camera position."""
        return self.distance, self.azimuth, self.elevation, [self.center_x, self.center_y, self.center_z]

    def set_camera_position(self, distance: float, azimuth: float, elevation: float, center: list[float]) -> None:
        """Sets the camera position."""
        self.camera_position_data = self.camera_position_data + [(distance, azimuth, elevation, center[0], center[1], center[2])]
        self.camera_position_updated.set()
    
    def activate_window(self) -> None:
        """Activates the window."""
        self.activate_window_updated.set()

    def run(self) -> None:
        """Runs the visualizer."""

        # pylint: disable=c-extension-no-member
        QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        visualizer = QtVisualizer(
            self.keypoints,
            self.rotation_matrix,
            self.translation_vector,
            self.project_feet_to_ground,
            self.limbs,
            self.measurements,
            auto_init=False,
            on_pause=lambda _: self.on_pause_event.set(),
            on_print=lambda _: self.on_print_event.set(),
            show_floor=self.show_floor,
            position=self.position
        )
        visualizer.init_app()
        self.started.set()

        def update():
            try:
                self.is_drawing.set()

                self.distance, self.azimuth, self.elevation, [self.center_x, self.center_y, self.center_z] = visualizer.get_camera_position()

                if self.camera_updated.is_set():
                    self.camera_updated.clear()
                    for (camera_p, cam_size, cam_edge, cam_color, rotation_matrix, translation_vector) in self.visualise_cameras_data:
                        _ = visualizer.draw_camera(
                            camera_p, cam_size, cam_edge, cam_color, rotation_matrix, translation_vector)
                    self.visualise_cameras_data = []

                if self.skeletons_updated.is_set():
                    self.skeletons_updated.clear()
                    for (skeletons, paths, skeleton_validities, skeleton_keypoint_validitites, measurements, head, tail) in self.visualise_skeletons_data:
                        visualizer.visualise_skeletons(
                            skeletons, paths, skeleton_validities, skeleton_keypoint_validitites, measurements, head, tail)
                    self.visualise_skeletons_data = []

                if self.line_updated.is_set():
                    self.line_updated.clear()
                    for (pos, color, linewidth, markersize, draw_point) in self.line_data:
                        _ = visualizer.line(
                            pos, color, linewidth, markersize, draw_point)
                    self.line_data = []

                if self.scatter_updated.is_set():
                    self.scatter_updated.clear()
                    for (pos, color, size, px_mode) in self.scatter_data:
                        _ = visualizer.scatter(pos, color, size, px_mode)
                    self.scatter_data = []
                    
                if self.mesh_updated.is_set():
                    self.mesh_updated.clear()
                    for (vertices, faces, color, edge_color, draw_edges, draw_faces, only_projection) in self.mesh_data:
                        _ = visualizer.mesh(vertices, faces,
                                            color, edge_color, draw_edges, draw_faces, only_projection)
                    self.mesh_data = []

                if self.camera_position_updated.is_set():
                    self.camera_position_updated.clear()
                    for (distance, azimuth, elevation, center_x, center_y, center_z) in self.camera_position_data:
                        visualizer.set_camera_position(distance, azimuth, elevation, [center_x, center_y, center_z])
                    self.camera_position_data = []

                if self.activate_window_updated.is_set():
                    self.activate_window_updated.clear()
                    visualizer.activate_window()

                self.is_drawing.clear()
            except Exception as e:
                print(e)
                pass

        q_timer = QTimer(visualizer.view)
        q_timer.timeout.connect(update)
        q_timer.start(int(1000/self.max_fps))
        visualizer.show()


if __name__ == '__main__':
    v = ParallelQtVisualizer([])
    v.show()
    while v.is_alive():
        pass
