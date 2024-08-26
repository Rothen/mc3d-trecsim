"""Live pose estimation and tracking using GMMs."""
import sys
import os
import signal
import logging
import time
from pathlib import Path
from typing import Any, TypeVar, cast
import csv

import typed_argparse as tap
import cv2
from scipy.spatial.transform import Rotation
import numpy as np
import numpy.typing as npt

from mc3d_trecsim_gmm import GMM, Camera, Frame, GMMParam, LBFGSParam
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
    gmm_param.setSeed(config.seed)

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


def runner(args: LiveArgs):
    """Run the live pose estimation and tracking.

    Args:
        args (LiveArgs): The arguments.
    """
    logging.basicConfig(level=logging.INFO)
    args.check()

    config = create_or_load_config(args, LiveConfig)

    cameras: list[Camera] = create_cameras(config)

    pose_estimator: PoseEstimator = create_pose_estimator(config)
    pose_estimator.predict(
        np.zeros((1, cameras[0].height, cameras[0].width, 3)))

    gmm_param: GMMParam = create_gmm_param(config)

    lbfgs_param: LBFGSParam = LBFGSParam()
    lbfgs_param.max_iterations = config.lbfgs_max_iterations
    lbfgs_param.max_linesearch = config.lbfgs_max_linesearch

    gmm = GMM(J=0, cameras=cameras, gmmParam=gmm_param, lbfgsParam=lbfgs_param)

    rotation_matrix = Rotation.from_rotvec(
        np.array(config.rotation_degrees), degrees=True).as_matrix()
    translation_vector = np.array(config.translation_vector).reshape((3, 1))

    skeleton_calculator: SkeletonCalculator = SkeletonCalculator(
        config.keypoints, rotation_matrix=rotation_matrix, translation_vector=translation_vector)
    visualizer: ParallelQtVisualizer = create_visualizer(
        config, cameras, rotation_matrix, translation_vector)

    caps, is_live = create_video_streamers(
        config, overwrite_max_fps=np.inf)

    signal.signal(signal.SIGINT, lambda sig,
                  frame: signal_handler(config, caps, visualizer))

    if not config.disable_visualiser:
        visualizer.show()

    max_fps: float = min(config.max_fps, np.min(
        [cap.get(cv2.CAP_PROP_FPS) for cap in caps]))

    start: float = time.perf_counter()
    if config.skip_frames > 0:
        for j, cap in enumerate(caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, config.skip_frames)
            cap.read()

    empty_images = [np.zeros((camera.shape[0], camera.shape[1], 3))
                    for camera in cameras]

    for camera, empty_image in zip(cameras, empty_images):
        cv2.imshow(camera.id, empty_image[::2, ::2])
    cv2.startWindowThread()

    i: int = 0
    initial_loop: bool = False
    fps_list: list[float] = []
    start: float = time.perf_counter()
    fps_start: float = time.perf_counter()
    start_timestamp: float = 0.0

    pause_start: float = 0.0
    pause_delay: float = 0.0
    current_loop: int = 0
    paused: bool = False

    def on_pause(visualizer: ParallelQtVisualizer):
        nonlocal paused, pause_start, pause_delay
        paused = not paused
        if paused:
            pause_start = time.perf_counter()
        else:
            pause_delay += time.perf_counter() - pause_start

    def on_print(visualizer: ParallelQtVisualizer):
        nonlocal args, config
        distance, azimuth, elevation, center = visualizer.get_camera_position()

        config.distance = distance
        config.azimuth = azimuth
        config.elevation = elevation
        config.center = center
        config.pause_at = current_loop

        config.to_yaml_file(str(args.configuration_file))

    if not is_live:
        visualizer.on_pause = on_pause

    visualizer.on_print = on_print

    if config.distance != 0.0:
        visualizer.set_camera_position(
            config.distance,
            config.azimuth,
            config.elevation,
            config.center
        )

    fit_times: list[float] = []
    loop_times: list[float] = []
    inference_times: list[float] = []
    post_inference_times: list[float] = []

    while np.all([cap.isOpened() for cap in caps]) and (not config.disable_visualiser or visualizer.is_alive()):
        loop_start = time.perf_counter()

        if config.pause_at == current_loop and not is_live:
            config.pause_at = -1
            paused = True
            pause_start = time.perf_counter()

        if paused:
            time.sleep(1/1000)
            continue

        fps_start = time.perf_counter()
        frames: list[np.ndarray] = []
        timestamps = []

        for cap, camera, empty_image in zip(caps, cameras, empty_images):
            grabbed, frame = cap.read()

            if grabbed:
                if config.undistort_images:
                    frame = cv2.undistort(frame, camera.A, camera.d, None)
                frames.append(frame)
            else:
                frames.append(empty_image)

            timestamps.append((time.perf_counter() - pause_delay)*1000)

        inference_start = time.perf_counter()
        all_kpts = pose_estimator.predict(frames)
        inference_end = time.perf_counter()

        if initial_loop:
            initial_loop = False
            continue

        if start_timestamp == 0.0:
            start_timestamp = timestamps[0]

        if config.show_video_feeds:
            for frame, camera, kpts in zip(frames, cameras, all_kpts):
                for person in kpts:
                    paint_skeleton_on_image(frame, person, plot_side=False)
                cv2.imshow(camera.id, frame[::2, ::2])
            cv2.startWindowThread()

        post_inference_start = time.perf_counter()
        ziped = zip(frames, all_kpts, timestamps, list(enumerate(cameras)))

        for frame, kpts, timestamp, (camera_index, camera) in ziped:
            if not config.undistort_images:
                for person in kpts:
                    person[:, :2] = cv2.undistortImagePoints(
                        person[:, :2].reshape((len(person[:, :2]), 1, 2)), camera.A, camera.d, None)\
                        .reshape(person[:, :2].shape)

            cpp_frame: Frame = Frame(
                camera_index, kpts, timestamp - start_timestamp, timestamp)
            gmm.addFrame(cpp_frame)
        post_inference_end = time.perf_counter()

        current_loop += 1

        inference_times.append(inference_end - inference_start)
        post_inference_times.append(post_inference_end - post_inference_start)

        fit_start = time.perf_counter()
        fit_result = gmm.fit()
        fit_times.append(time.perf_counter() - fit_start)

        skeletons, paths, validities, keypoint_validitites = skeleton_calculator.calculate(
            gmm, fit_result)

        visualizer.visualise_skeletons(
            skeletons, paths, validities, keypoint_validitites, [])

        now = time.perf_counter()
        if now - start >= 1:
            start = now
            fps_list.append(i)
            i = 0
        i += 1
        used: float = time.perf_counter() - fps_start

        loop_times.append(time.perf_counter() - loop_start)
        time.sleep(max(1/max_fps - used, 0))

    logging.info('Mean FPS: %.2f FPS.', np.array(
        fps_list).mean() if len(fps_list) > 0 else 0.0)

    cleanup(config, caps, visualizer)


def main() -> int:
    """Main function.

    Returns:
        int: The exit code.
    """
    tap.Parser(LiveArgs).bind(runner).run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
