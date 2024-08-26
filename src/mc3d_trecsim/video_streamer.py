from __future__ import annotations
from threading import Thread, Event
import cv2
import numpy as np
import time
import cv2.typing as cv2t
from typing import Callable

def none_callback(vs: VideoStreamer) -> None:
    return

class VideoStreamer:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """
    read_blocked: bool
    _frame_read: Event
    _stopped: Event
    _grabbed: bool
    _frame: cv2t.MatLike
    _frame_time_stamp: float
    _src: str|int
    _video_capture: cv2.VideoCapture
    _api_preference: int
    read_callback: Callable[[VideoStreamer], None]
    
    @property
    def src(self) -> str | int:
        return self._src

    def __init__(self,
            src: str | int = 0,
            api_preference: int = cv2.CAP_ANY,
            read_blocked: bool = False,
            max_fps: float = np.inf,
            debug: bool = False,
            read_callback: Callable[[VideoStreamer], None] = none_callback):
        super().__init__()

        self._src = src
        self._api_preference = api_preference
        self.read_blocked = read_blocked
        self.max_fps = max_fps
        self.debug = debug

        self._stopped = Event()
        self._video_capture = cv2.VideoCapture(self._src, apiPreference=api_preference)
        self._grabbed, self._frame = False, np.zeros((0, 0))
        self._frame_time_stamp = 0
        self._frame_read = Event()
        self.read_callback = read_callback
        self.__read_frame()
        self._thread = Thread(target=self.run, args=())
        self._thread.start()
    
    @property
    def frame_time_stamp(self) -> float:
        return self._frame_time_stamp
    
    @property
    def grabbed(self) -> bool:
        return self._grabbed
    
    @property
    def frame(self) -> cv2t.MatLike:
        return self._frame
    
    @property
    def video_capture(self) -> cv2.VideoCapture:
        return self._video_capture
    
    def get_time(self):
        self._video_capture.get(cv2.CAP_PROP_POS_MSEC)
    
    def isOpened(self) -> bool:
        return self._video_capture.isOpened()

    def run(self) -> None:
        start: float = time.perf_counter()

        while not self._stopped.wait(max(0, 1/self.max_fps - (time.perf_counter() - start))):
            start = time.perf_counter()

            if not self.read_blocked or self._frame_read.wait():
                if not self._stopped.is_set():
                    self.__read_frame()

            if not self._grabbed:
                break

    def __read_frame(self) -> None:            
        self._grabbed, self._frame = self._video_capture.read()
        if self._grabbed:
            self.read_callback(self)
        self._frame_time_stamp = time.time() * 1000
        self._frame_read.clear()

        if self.debug:
            print("CV_CAP_PROP_POS_MSEC:", self.get(cv2.CAP_PROP_POS_MSEC))
            print("CV_CAP_PROP_POS_FRAMES:", self.get(cv2.CAP_PROP_POS_FRAMES))
            print("CV_CAP_PROP_FPS:", self.get(cv2.CAP_PROP_FPS))
                
    def release(self):
        self._stopped.set()
        self._frame_read.set()
        self._thread.join()
        self._video_capture.release()
                
    def get(self, field: int):
        return self._video_capture.get(field)
                
    def set(self, field: int, value: float):
        return self._video_capture.set(field, value)

    def imwrite(self, file_name: str) -> bool:
        if self._frame is not None:
            return cv2.imwrite(file_name, self._frame)
        else:
            return False
    
    def read(self) -> tuple[bool, cv2t.MatLike]:
        self._frame_read.set()
        return self._grabbed, self._frame