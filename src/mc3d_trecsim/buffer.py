from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, MutableSequence, TypeVar
import numpy as np
from collections import deque
from threading import Lock, Condition

from camera import Camera
    
F = TypeVar("F", bound=MutableSequence[np.ndarray])
T = TypeVar("T", bound=MutableSequence[float])
C = TypeVar("C", bound=MutableSequence[Camera])

class Buffer(ABC, Generic[F, T, C]):
    __added: bool
    __added_counter: int
    __iter_index: int
    _frames: F
    _timestamps: T
    _camera_refs: C
    _lock: Lock
    
    @property
    def added(self) -> bool:
        return self.__added
    
    @property
    def added_counter(self) -> int:
        return self.__added_counter

    def __init__(self) -> None:
        self.__added = False
        self.__added_counter = 0
        self._lock = Lock()
        self._nonempty = Condition()

    @abstractmethod
    def _append(self, frame: np.ndarray, timestamp: float, camera_ref: Camera) -> None:
        pass

    def append(self, frame: np.ndarray, timestamp: float, camera_ref: Camera) -> None:
        """
        Add a frame with the corresponding timestamp to the buffer. Sets `added` to `True` and increases the `added_counter` by 1.

        Args:
            frame (np.ndarray): The frame that should be added to the buffer.
            timestamp (float): The timestamp in milliseconds of when the frame has been read.
        """
        with self._nonempty:
            self.__added = True
            self.__added_counter += 1
            self._append(frame, timestamp, camera_ref)
            self._nonempty.notify_all()
        
    def get_added(self) -> tuple[F, T, C]:
        with self._nonempty:

            while self.__added_counter == 0:
                self._nonempty.wait()
                
            res = self._frames[-self.__added_counter:], self._timestamps[-self.__added_counter:], self._camera_refs[-self.__added_counter:]
            self._reset()
            return res
    
    def _reset(self) -> None:
        self.__added = False
        self.__added_counter = 0
    
    def __len__(self) -> int:
        return len(self._frames)
    
    '''def __iter__(self) -> Buffer:
        self._reset()
        self.__iter_index = -1
        return self

    def __next__(self) -> tuple[np.ndarray, float, Camera]:
        self.__iter_index += 1
        return self._frames[self.__iter_index], self._timestamps[self.__iter_index], self._camera_refs[self.__iter_index]'''

class TimedRingBuffer(Buffer[list, list, list]):
    __dt: float
    
    @property
    def dt(self) -> float:
        return self.__dt
    
    def __init__(self, dt: float) -> None:
        super().__init__()
        self.__dt = dt
        self._frames = []
        self._timestamps = []
        self._camera_refs = []
        
    def _append(self, frame: np.ndarray, timestamp: float, camera_ref: Camera) -> None:
            insert_index = self.__get_insert_index(timestamp)
            
            self._frames.insert(insert_index, frame)
            self._timestamps.insert(insert_index, timestamp)
            self._camera_refs.insert(insert_index, camera_ref)
            
            delete_count = self.__get_delete_count()
            del self._frames[:delete_count]
            del self._timestamps[:delete_count]
            del self._camera_refs[:delete_count]
            
    def __get_insert_index(self, timestamp: float) -> int:
        insert_index: int = 0

        for i in range(len(self._timestamps)-1, -1, -1):
            if self._timestamps[i] < timestamp:
                insert_index = i + 1
                break
            
        return insert_index

    def __get_delete_count(self) -> int:
        delete_count: int = 0
        max_ts = self._timestamps[-1] if len(self._timestamps) > 0 else np.inf
        oldest_ts = max_ts - self.__dt
        
        for timestamp in self._timestamps:
            if timestamp < oldest_ts:
                delete_count += 1
            else:
                break
        
        return delete_count

class RingBuffer(Buffer[deque, deque, deque]):
    maxlen: int

    def __init__(self, maxlen: int) -> None:
        super().__init__()
        self.maxlen = maxlen
        self._frames = deque(maxlen=self.maxlen)
        self._timestamps = deque(maxlen=self.maxlen)
        self._camera_refs = deque(maxlen=self.maxlen)

    def _append(self, frame: np.ndarray, timestamp: float, camera_ref: Camera) -> None:
        self._frames.append(frame)
        self._timestamps.append(timestamp)
        self._camera_refs.append(camera_ref)
        self._nonempty.notify_all()
