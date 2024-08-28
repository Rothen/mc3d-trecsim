"""Live pose estimation and tracking using GMMs."""
import sys
import os
import signal
import logging
import time
from pathlib import Path
from typing import Any, TypeVar, cast
import torch

import typed_argparse as tap
import cv2
from scipy.spatial.transform import Rotation
import numpy as np
import numpy.typing as npt

from mc3d_trecsim.gmm import GMM, Camera, Frame, GMMParam, LBFGSParam
from mc3d_trecsim.args import LiveArgs
from mc3d_trecsim.calibration import Calibration
from mc3d_trecsim.model import PoseEstimator, YOLOv7
from mc3d_trecsim.plotting import paint_skeleton_on_image
from mc3d_trecsim.video_streamer import VideoStreamer
from mc3d_trecsim.skeleton_calculator import SkeletonCalculator
from mc3d_trecsim.parallel_qt_visualizer import ParallelQtVisualizer
from mc3d_trecsim.enums import KPT_IDXS, COLORS
from mc3d_trecsim.config import LiveConfig


def create_cameras(config: LiveConfig) -> list[Camera]:
    """Create the cameras.

    Args:
        config (LiveConfig): The arguments.

    Returns:
        list[Camera]: The cameras.
    """
    if config.camera_ids is None:
        cameras = [Camera(f'camera_{i}') for i in range(len(config.sources))]
    else:
        cameras = [Camera(camera_id) for camera_id in config.camera_ids]

    calibration = Calibration(Path(config.calibration_file))
    calibration.load_calibration(cameras)

    if config.camera_distances is None:
        for camera in cameras:
            camera.distance = 300.0
    else:
        for camera, camera_distance in zip(cameras, config.camera_distances):
            camera.distance = camera_distance

    logging.info('Cameras %s loaded.', ', '.join(
        [camera.id for camera in cameras]))

    return cameras


def create_visualizer(
    config: LiveConfig,
    cameras: list[Camera],
    rotation_matrix: npt.NDArray[np.double] = np.eye(3, 3),
    translation_vector: npt.NDArray[np.double] = np.zeros((3, 1))
) -> ParallelQtVisualizer:
    """Create the visualizer.

    Args:
        config (LiveConfig): The arguments.
        cameras (list[Camera]): The cameras.

    Returns:
        ParallelQtVisualizer: The visualizer.
    """
    visualizer = ParallelQtVisualizer(config.keypoints,
                                      project_feet_to_ground=False,
                                      additional_limbs=[
                                          ([KPT_IDXS.NOSE, KPT_IDXS.RIGHT_SHOULDER],
                                           COLORS.GREEN),
                                          ([KPT_IDXS.NOSE, KPT_IDXS.LEFT_SHOULDER],
                                           COLORS.GREEN)
                                      ],
                                      show_floor=config.show_floor
                                      )
    visualizer.visualise_cameras(
        cameras, rotation_matrix=rotation_matrix, translation_vector=translation_vector)

    logging.info('Visualizer created.')

    return visualizer


def create_pose_estimator(config: LiveConfig) -> PoseEstimator:
    """Create the pose estimator.

    Args:
        config (LiveConfig): The arguments.

    Returns:
        PoseEstimator: The pose estimator.
    """
    pose_estimator = YOLOv7(Path(config.weights_file),
                            new_shape=448, stride=64)
    logging.info('Pose estimator created.')
    return pose_estimator


def create_video_streamers(config: LiveConfig,
                           overwrite_max_fps: float = 0) -> tuple[list[VideoStreamer], bool]:
    """Create video streamers for the sources.

    Args:
        config (LiveConfig): The arguments.

    Returns:
        list[VideoStreamer]: The video streamers.
    """
    caps: list[VideoStreamer] = []
    is_live: bool = False

    for source in config.sources:
        fixed_source: str | int = str(source)
        source_path: Path = Path(source)
        api_preference: int = cv2.CAP_ANY
        read_blocked: bool = False
        max_fps: float = np.inf

        if source.isnumeric():
            fixed_source = int(source)
            logging.info('Creating local video streamer (%s)', fixed_source)
        elif source.startswith('rtspsrc'):
            api_preference = cv2.CAP_GSTREAMER
            logging.info('Creating RTSP video streamer (%s)', fixed_source)
            is_live = True
        elif source_path.exists() and source_path.is_file():
            read_blocked = True
            max_fps = max(config.max_fps, overwrite_max_fps)
            logging.info('Creating file video streamer (%s)', fixed_source)

        try:
            caps.append(VideoStreamer(
                fixed_source, api_preference, read_blocked, max_fps))
        except Exception as e:
            logging.error(
                'Error creating video streamer for source %s: %s', fixed_source, e)
            sys.exit(1)

    logging.info('Video streamers created with stream FPSs: %s',
                 ', '.join([str(cap.get(cv2.CAP_PROP_FPS)) for cap in caps]))

    return caps, is_live


def cleanup(config: LiveConfig,
            caps: list[VideoStreamer], visualizer: ParallelQtVisualizer) -> None:
    """Cleanup the resources.

    Args:
        config (LiveConfig): The arguments.
        caps (list[VideoStreamer]): The video streamers.
        visualizer (ParallelQtVisualizer): The visualizer.
    """
    for cap in caps:
        cap.release()

    if not config.disable_visualiser and visualizer.is_alive():
        visualizer.kill()
        visualizer.join()

    if config.show_video_feeds:
        cv2.destroyAllWindows()


def create_gmm_param(config: LiveConfig) -> GMMParam:
    """Create GMM parameters from the arguments.

    Args:
        config (LiveConfig): The arguments.

    Returns:
        GMMParam: The GMM parameters.
    """
    gmm_param: GMMParam = GMMParam()
    gmm_param.tol = config.tol
    gmm_param.KEYPOINTS = config.keypoints
    gmm_param.maxIter = config.max_iter
    gmm_param.keypointConfidenceThreshold = config.keypoint_confidence_threshold
    gmm_param.splineKnotDelta = config.spline_knot_delta
    gmm_param.maxFrameBuffer = config.max_frame_buffer
    gmm_param.splineDegree = config.spline_degree
    gmm_param.nu = config.nu
    gmm_param.autoManageTheta = config.auto_manage_theta
    gmm_param.autoManageHypothesis = config.auto_manage_hypothesis
    gmm_param.splineSmoothingFactor = config.spline_smoothing_factor
    gmm_param.splineSmoothingFactor = config.spline_smoothing_factor
    gmm_param.notSupportedSinceThreshold = config.not_supported_since_threshold
    gmm_param.responsibilityLookback = config.responsibility_look_back
    gmm_param.responsibilitySupportThreshold = config.responsibility_support_threshold
    gmm_param.dragAlongUnsupportedKeyPoints = config.drag_along_unsupported_key_points
    gmm_param.numSupportCameras = config.num_support_cameras
    gmm_param.minValidKeyPoints = config.min_valid_key_points

    return gmm_param


def signal_handler(config: LiveConfig, caps: list[VideoStreamer], visualizer: ParallelQtVisualizer):
    """Handle the signal.

    Args:
        sig (int): The signal.
        frame (FrameType | None): The frame.
        config (LiveConfig): The arguments.
        caps (list[VideoStreamer]): The video streamers.
        visualizer (ParallelQtVisualizer): The visualizer.
    """
    cleanup(config, caps, visualizer)


T = TypeVar('T', bound=LiveConfig)


def create_or_load_config(args: LiveArgs, config_class: type[T]) -> T:
    """Create or load the configuration.

    Args:
        args (LiveArgs): The arguments.
        config_class (type[T]): The configuration class.

    Returns:
        T: The configuration.
    """
    if args.create:
        config: T = config_class([], '')
        config.to_yaml_file(str(args.configuration_file))
        print(f'Configuration \'{args.configuration_file}\' created.')
        sys.exit(0)
    else:
        configs = config_class.from_yaml_file(str(args.configuration_file))
        config: T = cast(T, configs)
        config.check()

    return config
