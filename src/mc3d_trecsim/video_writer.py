from threading import Thread, Event
from .video_streamer import VideoStreamer
import cv2
import time

class VideoWriter(Thread):
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, video_streamer: VideoStreamer, video_path: str, fps: int = 25):
        Thread.__init__(self)
        self.stopped = Event()
        self.video_streamer = video_streamer
        self.video_path = video_path
        self.fps = fps

        self.width = int(self.video_streamer.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_streamer.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height))
        self.timestamps = []
        self.cv_timestamps = []

    def run(self):
        start = time.perf_counter()
        while not self.stopped.wait(max(0, 1/self.fps - (time.perf_counter() - start))):
            start = time.perf_counter()
            if not self.video_streamer.grabbed:
                pass
            else:
                self.writer.write(self.video_streamer.frame)
                self.timestamps.append(self.video_streamer.frame_time_stamp)
                self.cv_timestamps.append(self.video_streamer.get(cv2.CAP_PROP_POS_MSEC))

        self.release()
                
    def release(self):
        self.writer.release()

    def stop(self):
        self.stopped.set()