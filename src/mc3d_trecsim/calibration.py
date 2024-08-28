import cv2
import numpy as np
from .gmm import Camera
from pathlib import Path

class Calibration:
    _calibration_file_path: Path
    cameras: list[Camera]
    
    def __init__(self, calibration_file_path: Path):
        self._calibration_file_path = calibration_file_path
    
    def load_calibration(self, cameras: list[Camera]):
        fs = cv2.FileStorage(str(self._calibration_file_path.resolve()), cv2.FILE_STORAGE_READ)
        self.nb_camera = int(fs.getNode("nb_camera").real())

        for camera in cameras:
            camera_node = fs.getNode(camera.id)
            if camera_node.empty():
                print('Camera %s not calibrated'%(camera.id, ))
                continue
            A = np.array(camera_node.getNode("camera_matrix").mat())
            d = np.array(camera_node.getNode("distortion_vector").mat())
            P = np.array(camera_node.getNode("camera_pose_matrix").mat())
            height = int(camera_node.getNode("img_height").real())
            width = int(camera_node.getNode("img_width").real())
            camera.setCalibration(A, d[0], P, height, width)
    