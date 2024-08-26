"""The configuration classes for the live fitting algorithm."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time

from dataclass_wizard import YAMLWizard

from .enums import KPT_IDXS

@dataclass
class LiveConfig(YAMLWizard):
    """The arguments for the live fitting algorithm.

    Args:
        sources (list[str]): The video sources. Can be a file, a URL or a number.
        calibration_file (str): The patho of the calibration file.
        keypoints (list[int]): The indizes of the keypoints which should be fitted.
        disable_visualiser (bool): Disable the visualiser.
        show_video_feeds (bool): Show the video feeds.
        max_fps (float): The maximum frames per second. Ignored if the source is not a file.
        tol (float): The tolerance for the fitting algorithm.
        max_iter (int): The maximum number of iterations for the fitting algorithm.
        keypoint_confidence_threshold (float): The confidence threshold for the keypoints
            returned by the 2d pose estimator.
        spline_knot_delta (float): The delta for the spline knots in milliseconds.
        max_frame_buffer (int): The maximum number of frames to keep in the buffer.
        spline_degree (int): The degree of the splines used by the fitting algorithm.
        nu (float): The precision parameter for the fitting algorithm.
        auto_manage_theta (bool): Whether the theta parameter should be automatically
            managed by the fitting algorithm.
        copy_last_thetas (bool): Whether the last thetas should be copied to the next
            frame if a new knot is added to the spline.
        spline_smoothing_factor (float): The smoothing factor for the splines used by
            the fitting algorithm.
        max_iterations (int): The maximum number of iterations for the fitting algorithm.
        max_linesearch (int): The maximum number of linesearches for the fitting algorithm.
        skip_frames (int): The number of frames to skip before starting the fit.
        camera_ids (Optional[List[str]]): The camera IDs in the calibration configuration
            to use. If not provided, the cameras will be named camera_0, camera_1, etc.
        weights_file (str): The path to the weights file.

        calibration_path (Path): The path object of the calibration file.
        weights_path (Path): The path object of the weights file.

    Raises:
        ValueError: The number of sources and camera IDs must be the same
        ValueError: Calibration file does not exist
        ValueError: Calibration file is not a file
    """
    sources: list[str]
    calibration_file: str
    keypoints: list[int] = field(default_factory=lambda: [
        KPT_IDXS.NOSE,
        KPT_IDXS.RIGHT_SHOULDER,
        KPT_IDXS.LEFT_SHOULDER,
        KPT_IDXS.RIGHT_ELBOW,
        KPT_IDXS.LEFT_ELBOW,
        KPT_IDXS.RIGHT_HAND,
        KPT_IDXS.LEFT_HAND,
        KPT_IDXS.RIGHT_HIP,
        KPT_IDXS.LEFT_HIP,
        KPT_IDXS.RIGHT_KNEE,
        KPT_IDXS.LEFT_KNEE,
        KPT_IDXS.RIGHT_FOOT,
        KPT_IDXS.LEFT_FOOT]
    )
    disable_visualiser: bool = False
    show_video_feeds: bool = True
    max_fps: float = 25.0
    drag_along_unsupported_key_points: bool = True
    num_support_cameras: int = 2
    tol: float = 1e-6
    max_iter: int = 5
    keypoint_confidence_threshold: float = 0.5
    spline_knot_delta: float = 500.0
    max_frame_buffer: int = 20
    spline_degree: int = 3
    nu: float = 500.0
    auto_manage_theta: bool = True
    auto_manage_hypothesis: bool = True
    copy_last_thetas: bool = True
    spline_smoothing_factor: float = 100.0
    lbfgs_max_iterations: int = 10
    lbfgs_max_linesearch: int = 5
    skip_frames: int = 0
    camera_ids: Optional[list[str]] = None
    weights_file: str = '../data/models/yolov7/yolov7-w6-pose.pt'
    not_supported_since_threshold: int = 5
    responsibility_look_back: int = 5
    responsibility_support_threshold: float = 0.3
    rotation_degrees: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0])
    translation_vector: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    show_floor: bool = True
    camera_distances: Optional[list[float]] = None
    min_valid_key_points: int = 6
    pause_at: int = -1
    seed: int = int(time.time())
    undistort_images: bool = False
    distance: float = 0.0
    azimuth: float = 0.0
    elevation: float = 0.0
    center: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    csv_file: Optional[str] = None
    times_csv_file: Optional[str] = None

    def check(self) -> None:
        """Check the arguments.

        Raises:
            ValueError: The number of sources and camera IDs must be the same
            ValueError: Calibration file does not exist
            ValueError: Calibration file is not a file
        """
        if self.camera_ids is not None and len(self.sources) != len(self.camera_ids):
            raise ValueError('The number of sources and camera IDs must be the same.')

        calibration_path = Path(self.calibration_file)
        if not calibration_path.exists():
            raise ValueError(f'Calibration file {calibration_path} does not exist.')

        if not calibration_path.is_file():
            raise ValueError(f'Calibration file {calibration_path} is not a file.')

        weights_path = Path(self.weights_file)
        if not weights_path.exists():
            raise ValueError(f'Weights file {weights_path} does not exist.')

        if not weights_path.is_file():
            raise ValueError(f'Weights file {weights_path} is not a file.')
