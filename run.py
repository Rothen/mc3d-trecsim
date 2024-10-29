"""Live pose estimation and tracking using GMMs."""
import sys
import signal
import logging
import time
import torch

import typed_argparse as tap
import cv2
from scipy.spatial.transform import Rotation
import numpy as np

from mc3d_trecsim.gmm import GMM, Camera, Frame, GMMParam, LBFGSParam
from mc3d_trecsim.args import LiveArgs
from mc3d_trecsim.model import PoseEstimator
from mc3d_trecsim.plotting import paint_skeleton_on_image
from mc3d_trecsim.skeleton_calculator import SkeletonCalculator
from mc3d_trecsim.parallel_qt_visualizer import ParallelQtVisualizer
from mc3d_trecsim.config import LiveConfig
from mc3d_trecsim.run_helpers import create_cameras, \
    create_visualizer, \
    create_or_load_config, \
    create_pose_estimator, \
    create_gmm_param, \
    create_video_streamers, \
    signal_handler, \
    cleanup


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

    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    gmm_param.setSeed(config.seed)

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
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, config.skip_frames)
            cap.read()

    empty_images = [np.zeros((camera.shape[0], camera.shape[1], 3))
                    for camera in cameras]

    for camera, empty_image in zip(cameras, empty_images):
        cv2.imshow(camera.id, empty_image[::2, ::2])

    if config.video_feed_positions is not None:
        for camera, video_feed_position in zip(cameras, config.video_feed_positions):
            cv2.moveWindow(camera.id, *video_feed_position)

    cv2.startWindowThread()
    visualizer.activate_window()

    i: int = 0
    fps_list: list[float] = []
    start: float = time.perf_counter()
    fps_start: float = time.perf_counter()
    start_timestamp: float = 0.0

    pause_start: float = 0.0
    pause_delay: float = 0.0
    current_loop: int = 0
    paused: bool = not is_live and not config.auto_start

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

        if start_timestamp == 0.0:
            start_timestamp = timestamps[0]

        if config.show_video_feeds:
            for frame, camera, kpts in zip(frames, cameras, all_kpts):
                for person in kpts:
                    paint_skeleton_on_image(frame, person, plot_sides=config.plot_sides)
                cv2.imshow(camera.id, frame[::2, ::2])
            # cv2.startWindowThread()

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
