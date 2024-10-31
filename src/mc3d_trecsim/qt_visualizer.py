"""A module for the PyQt visualizer."""
from typing import Annotated, Any, Callable, Sequence

import numpy as np
import numpy.typing as npt
from PyQt6 import QtCore
from PyQt6 import QtGui
import PyQt6.QtWidgets as QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsTextItem

from .enums import LIMB_KPTS, LIMBS
from .mc3d_types import ColorsLike
from .visualizer import Limb, Visualizer
from .measurement import Measurement, Skeleton

class EventFilter(QtCore.QObject):
    space_pressed = QtCore.pyqtSignal()
    print_pressed = QtCore.pyqtSignal()

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self.widget.installEventFilter(self)

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self.widget and event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_Space:
                self.space_pressed.emit()
            if event.key() == QtCore.Qt.Key.Key_P:
                self.print_pressed.emit()
        return super().eventFilter(obj, event)


class CustomGraphicsView(QGraphicsView):
    """A custom graphics view."""

    def __init__(self, gl_view_widget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gl_view_widget = gl_view_widget

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """Handles the mouse press event.
        
        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        self.gl_view_widget.mousePressEvent(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        self.gl_view_widget.mouseMoveEvent(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self.gl_view_widget.mouseReleaseEvent(event)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self.gl_view_widget.wheelEvent(event)
        super().wheelEvent(event)

class QtVisualizer(Visualizer[GLGraphicsItem]):
    """A class to represent a visualizer for 3D data using PyQt."""

    def __init__(self,
                 keypoints: Sequence[int] | None = None,
                 project_feet_to_ground: bool = False,
                 limbs: list[Limb] = LIMBS,
                 measurements: list[Measurement[float]] | None = None,
                 auto_init: bool = True,
                 on_pause: Callable[[Any], None] = lambda _: None,
                 on_print: Callable[[Any], None] = lambda _: None,
                 show_floor: bool = True,
                 position: list[int] = None
            ):
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
            measurements (list[Measurement[Any]] | None, optional): The measurements to show. Defaults to [].
            auto_init (bool, optional): Whether to automatically initialize the visualizer.
                Defaults to True.
            show_floor (bool, optional): Whether to show the floor. Defaults to True.
        """

        super().__init__(
            keypoints,
            correct_axis=True,
            project_feet_to_ground=project_feet_to_ground,
            limbs=limbs
        )

        # pylint: disable=c-extension-no-member
        self.app: QtWidgets.QApplication | None = None
        self.all_items: list[GLGraphicsItem] = []
        self.skeletons: list[Skeleton] = []
        self.measurements: list[Measurement[float]] = measurements if measurements is not None else []
        self.on_pause: Callable[[Any], None] = on_pause
        self.on_print: Callable[[Any], None] = on_print
        self.show_floor: bool = show_floor
        self.position: list[int] = position

        if auto_init:
            self.init_app()

    def update(self):
        """Updates the visualizer."""
        if self.has_changes.is_set():
            self.has_changes.clear()
            for item in self.all_items:
                self.remove_item(item)

                # self.all_items += self.draw_measurement(measurement)

    def show(self) -> None:
        """Shows the visualizer and starts the event loop."""

        self.run_app()

    def init_app(self) -> None:
        """Initializes the visualizer."""
        self.traces = {}

        self.app = pg.mkQApp("MC3D-TRECSIM")
        self.app.setAttribute(
            # pylint: disable=c-extension-no-member
            QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

        self.main = QtWidgets.QMainWindow()
        if self.position is not None:
            self.main.move(self.position[0], self.position[1])
        else:
            self.main.move(int(1920/4), int(1080/4))
        self.main.setFixedSize(int(1920/2), int(1080/2))
        # self.main.setWindowFlags(
        #     QtCore.Qt.WindowType.CustomizeWindowHint | QtCore.Qt.WindowType.FramelessWindowHint)

        self.view = gl.GLViewWidget()
        transform = np.array([
            [1, 0, 0, 0],  # X-axis remains the same
            [0, 0, 1, 0],  # Z-axis becomes the Y-axis
            [0, 1, 0, 0],  # Y-axis becomes the Z-axis
            [0, 0, 0, 1]   # Homogeneous coordinate
        ])
        # self.view.(transform
        self.view.setMouseTracking(True)
        self.view.setCameraParams(distance=20)
        self.view.setWindowTitle('MC3D-TRECSIM: 3D Visualizer')
        self.view.setGeometry(0, 110, int(1920/2), int(1080/2))

        eventFilter = EventFilter(self.view)
        eventFilter.space_pressed.connect(lambda: self.on_pause(self))
        eventFilter.print_pressed.connect(lambda: self.on_print(self))


        self.main.setCentralWidget(self.view)

        ##################################################

        self.graphics_view = CustomGraphicsView(self.view)
        self.front_plane = QGraphicsScene()
        self.graphics_view.setScene(self.front_plane)

        # Make the graphics view transparent
        self.graphics_view.setStyleSheet("background: transparent; width: 500px;")
        # self.graphics_view.setFrameShape(.NoFrame)
        self.graphics_view.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        
        self.front_plane_text = QGraphicsTextItem('')
        self.front_plane_text.setFont(QtGui.QFont('Helvetica', 12))
        self.front_plane_text.setDefaultTextColor(QtCore.Qt.GlobalColor.white)
        self.front_plane.addItem(self.front_plane_text)

        ####################################################3

        self.layout_box = pg.QtWidgets.QHBoxLayout(self.view)
        self.layout_box.setContentsMargins(0, 0, 0, 0)
        self.layout_box.addWidget(self.graphics_view)

        self.main.show()

        if self.show_floor:
            floor = gl.GLGridItem()
            # floor.rotate(90, 1, 0, 0)
            floor.setSize(20, 20, 1)
            self.view.addItem(floor)

    def get_camera_position(self) -> tuple[float, float, float, list[float]]:
        """Gets the camera position."""
        distance = self.view.cameraParams()['distance']
        azimuth = self.view.cameraParams()['azimuth']
        elevation = self.view.cameraParams()['elevation']
        center = [self.view.cameraParams()['center'][0], self.view.cameraParams()['center'][1], self.view.cameraParams()['center'][2]]
        return distance, azimuth, elevation, center

    def set_camera_position(self, distance: float, azimuth: float, elevation: float, center: list[float]) -> None:
        """Sets the camera position."""
        self.view.setCameraPosition(
            pos=QtGui.QVector3D(center[0], center[1], center[2]),
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
        )

    def remove_item(self, item: GLGraphicsItem) -> None:
        """Removes an item from the visualizer.

        Args:
            item (GLGraphicsItem): The item to remove.
        """
        if item in self.view.items:
            self.view.removeItem(item)

    def line(self,
             pos: npt.NDArray[np.double],
             color: ColorsLike = (1.0, 0.0, 0.0, 1.0),
             linewidth: int = 2,
             markersize: int = 10,
             draw_point: bool = False
             ) -> list[GLGraphicsItem]:
        """Draws a line.

        Args:
            pos (npt.NDArray[np.double]): The positions of the line.
            color (ColorsLike, optional):
                The color of the line. Defaults to (1.0, 0.0, 0.0, 1.0).
            linewidth (int, optional): The width of the line. Defaults to 2.
            markersize (int, optional): The size of the markers. Defaults to 10.
            draw_point (bool, optional): True if the points should be drawn. Defaults to False.

        Returns:
            list[GLGraphicsItem]: The drawn items.
        """
        line_mode = 'line_strip'
        line = gl.GLLinePlotItem(pos=pos, color=np.array(color) if isinstance(
            color, list) else color, width=linewidth, mode=line_mode, glOptions='translucent')
        line.rotate(90, 1, 0, 0)
        self.view.addItem(line)
        items: list[GLGraphicsItem] = [line]

        if draw_point:
            # scatter = self.scatter(pos=pos, color=color, size=markersize, pxMode=True)
            # items += scatter
            pass

        return items

    def scatter(self,
                pos: npt.NDArray[np.double],
                color: ColorsLike = (1.0, 0.0, 0.0, 1.0),
                size: int = 3,
                px_mode: bool = True
                ) -> list[GLGraphicsItem]:
        """Draws a scatter plot.

        Args:
            pos (npt.NDArray[np.double]): The positions of the scatter plot.
            color (ColorsLike, optional):
                The color of the scatter plot. Defaults to (1.0, 0.0, 0.0, 1.0).
            size (int, optional): The size of the scatter plot. Defaults to 3.
            px_mode (bool, optional): True if the plot should be in pixel mode. Defaults to True.

        Returns:
            list[GLGraphicsItem]: The drawn items.
        """
        pos = self.preprocess_points(pos)
        scatter = gl.GLScatterPlotItem(pos=np.array(pos), color=np.array(
            color) if isinstance(color, list) else color, size=size, pxMode=px_mode)
        scatter.rotate(90, 1, 0, 0)
        self.view.addItem(scatter)
        return [scatter]
    
    def mesh(self,
             vertices: npt.NDArray[np.double],
             faces: npt.NDArray[np.double],
             color=None,
             edge_color=None,
             vertex_colors=None,
             face_colors=None,
             draw_edges: bool = False,
             draw_faces: bool = True,
             only_projection: bool = False
        ) -> None:
        """Draws a mesh."""
        vertices = self.preprocess_points(vertices)
        if only_projection:
            vertices[1, :] = 0

        mesh_data: gl.MeshData = gl.MeshData(
            vertexes=vertices,
            faces=faces,
            vertexColors=vertex_colors,
            faceColors=face_colors
        )
        mesh_item: gl.GLMeshItem = gl.GLMeshItem(
            meshdata=mesh_data, color=color, edgeColor=edge_color, drawEdges=draw_edges, drawFaces=draw_faces)
        mesh_item.setGLOptions('translucent')
        mesh_item.rotate(90, 1, 0, 0)
        self.view.addItem(mesh_item)

    def draw_measurements(self, measurements_list: list[list[str]]) -> None:
        """Draws the measurements.

        Args:
            measurements_list (list[list[str]]): The measurements to draw.
        """
        values: list[list[str]] = [['Person %i:'%(i, )] for i in range(len(measurements_list))]
        for i, measurements in enumerate(measurements_list):
            for measurement in measurements:
                values[i].append(measurement)

        self.front_plane_text.setPlainText(
            '\n\n'.join(['\n'.join(value) for value in values]))
    
    def activate_window(self) -> None:
        """Activates the window."""
        self.main.activateWindow()

    def run_app(self) -> None:
        """Starts the event loop."""
        pg.exec()
