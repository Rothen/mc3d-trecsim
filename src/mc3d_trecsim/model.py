import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal, TypeVar
from abc import ABC, abstractmethod
from pathlib import Path
import cv2

import torch
from torchvision import transforms
import sys
from .enums import KPT_IDXS
# sys.path.insert(1, '../../2d_pose_estimators/yolov7')
from models.yolo import Model
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# from mmpose.apis import inference_topdown, init_model
import numpy as np

DType = TypeVar('DType', bound=np.generic)
KeyPoints = Annotated[npt.NDArray[np.float32], Literal[17, 3]]

class PoseEstimator(ABC):
    name: str
    available_keypoints: dict

    def __init__(self, name: str) -> None:
        self.name = name
        self._create_model()
    
    @abstractmethod
    def _create_model(self) -> None:
        pass
    
    @abstractmethod
    def predict(self, images: list[np.ndarray] | np.ndarray) -> KeyPoints:
        pass

class YOLOv7(PoseEstimator):
    device: torch.device
    model: Model
    available_keypoints: dict = {
        KPT_IDXS.NOSE: 0,
        KPT_IDXS.RIGHT_EYE: 1,
        KPT_IDXS.LEFT_EYE: 2,
        KPT_IDXS.RIGHT_EAR: 3,
        KPT_IDXS.LEFT_EAR: 4,
        KPT_IDXS.RIGHT_SHOULDER: 5,
        KPT_IDXS.LEFT_SHOULDER: 6,
        KPT_IDXS.RIGHT_ELBOW: 7,
        KPT_IDXS.LEFT_ELBOW: 8,
        KPT_IDXS.RIGHT_HAND: 9,
        KPT_IDXS.LEFT_HAND: 10,
        KPT_IDXS.RIGHT_HIP: 11,
        KPT_IDXS.LEFT_HIP: 12,
        KPT_IDXS.RIGHT_KNEE: 13,
        KPT_IDXS.LEFT_KNEE: 14,
        KPT_IDXS.RIGHT_FOOT: 15,
        KPT_IDXS.LEFT_FOOT: 16
    }
    
    lb_images: torch.Tensor
    
    def __init__(self, weights_file_path: Path, new_shape=(960, 960), stride=64) -> None:
        self.weights_file_path: Path = weights_file_path
        self.new_shape = new_shape
        self.stride = stride
        super().__init__('YOLOv7')

    def _create_model(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weigths = torch.load(self.weights_file_path, map_location=self.device)
        self.model = weigths['model']
        _ = self.model.float().eval()
        
        if torch.cuda.is_available():
            self.model.half().to(self.device)
            
    def predict(self, images: list[np.ndarray] | np.ndarray, new_shape=None, stride=None):
        new_shape = self.new_shape if new_shape is None else new_shape
        stride = self.stride if stride is None else stride
        
        # Resize and pad image while meeting stride-multiple constraints
        lb_images, self.ratios, self.dwhs = self.letterbox(images, new_shape, stride=stride, auto=True)
        lb_images = np.array([transforms.ToTensor()(lb_image).numpy() for lb_image in lb_images])
        
        self.lb_images = torch.from_numpy(lb_images)

        if torch.cuda.is_available():
            self.lb_images = self.lb_images.half().to(self.device)

        with torch.no_grad():
            self.output, _ = self.model(self.lb_images)
            self.output_tensor = non_max_suppression_kpt(self.output, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True) # [batch_id, class_id, x, y, w, h, conf]
            self.lb_keypoints = output_to_keypoint(self.output_tensor)

        self.recalculated_keypoints = [[] for _ in range(len(images))]
        
        for keypoints in self.lb_keypoints:
            batch_id = int(keypoints[0])
            recalculated_keypoints = keypoints[7:].reshape((17, 3))
            recalculated_keypoints[:,0] -= self.dwhs[batch_id][0]
            recalculated_keypoints[:,1] -= self.dwhs[batch_id][1]
            recalculated_keypoints[:,0] /= self.ratios[batch_id][0]
            recalculated_keypoints[:,1] /= self.ratios[batch_id][1]
            self.recalculated_keypoints[batch_id].append(recalculated_keypoints)
        
        return self.recalculated_keypoints
    
    def letterbox(self, images: list[np.ndarray] | np.ndarray, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Resize and pad image while meeting stride-multiple constraints
        new_imgs = []
        ratios = []
        dwhs = []
        
        for image in images:
            shape = image.shape[:2]  # current shape [height, width]
            if isinstance(new_shape, int):
                new_shape = (new_shape, new_shape)

            # Scale ratio (new / old)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            if not scaleup:  # only scale down, do not scale up (for better test mAP)
                r = min(r, 1.0)

            # Compute padding
            ratio = [r, r]  # width, height ratios
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
            if auto:  # minimum rectangle
                dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
            elif scaleFill:  # stretch
                dw, dh = 0.0, 0.0
                new_unpad = (new_shape[1], new_shape[0])
                ratio = [new_shape[1] / shape[1], new_shape[0] / shape[0]]  # width, height ratios

            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if shape[::-1] != new_unpad:  # resize
                image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            
            new_imgs.append(new_img)
            ratios.append(ratio)
            dwhs.append([dw, dh])

        return np.array(new_imgs), np.array(ratios), np.array(dwhs)
