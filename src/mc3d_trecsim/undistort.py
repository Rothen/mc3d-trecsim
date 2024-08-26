from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import numpy as np
import cv2

I = TypeVar('I')
O = TypeVar('O')

class Pipe(ABC, Generic[I, O]):
    def __init__(self):
        pass
        
    @abstractmethod
    def pipe(input: I) -> O:
        pass
    

class Undistort(Pipe[np.ndarray, np.ndarray]):
    A: np.ndarray
    d: np.ndarray
    P: np.ndarray

    def __init__(self, A: np.ndarray, d: np.ndarray, P: np.ndarray, width, height, optimize: bool = False) -> None:
        super().__init__()
        self.A = A
        self.d = d
        self.P = P
        self.optimize = optimize
        self.newcameramatrix = cv2.getOptimalNewCameraMatrix(
            self.A, self.d, (width, height), 1, (width, height)
        )[0] if self.optimize else None
        
    def pipe(self, input: np.ndarray) -> np.ndarray:
        return cv2.undistort(
            input, self.A, self.d, None, self.newcameramatrix if self.optimize else None
        )
        
    def undistort(self, xy, k, distortion, iter_num=3):
        k1, k2, p1, p2, k3 = distortion[0]
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[:2, 2]
        x, y = xy.astype(float)
        x = (x - cx) / fx
        x0 = x
        y = (y - cy) / fy
        y0 = y
        for _ in range(iter_num):
            r2 = x ** 2 + y ** 2
            k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
            delta_x = 2 * p1 * x*y + p2 * (r2 + 2 * x**2)
            delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x*y
            x = (x0 - delta_x) * k_inv
            y = (y0 - delta_y) * k_inv
        return np.array((x * fx + cx, y * fy + cy))