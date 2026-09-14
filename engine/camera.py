import multiprocessing
import os
import time
import logging
import threading
import numpy as np
import cv2 as cv

logger = logging.getLogger(__name__)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ['OPENCV_FFMPEG_FFMPEG_DEBUG'] = "1"
os.environ['OPENCV_FFMPEG_FFMPEG_LOGLEVEL'] = "48"
cv.setNumThreads(multiprocessing.cpu_count())

# also acts (partly) like a cv.VideoCapture
class FreshestFrame(threading.Thread):
    RECONNECT_INITIAL_DELAY = 2.0
    RECONNECT_MAX_DELAY = 30.0
    RELEASE_TIMEOUT = 5.0

    def __init__(self, rtsp_url, name='FreshestFrame'):
        self.rtsp_url = rtsp_url
        self._create_capture()

        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()

        # this allows us to stop the thread gracefully
        self.running = False

        # keeping the newest frame around
        self.frame = None

        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0

        # this is just for demo purposes
        self.callback = None

        super().__init__(name=name, daemon=True)
        self.start()

    def start(self):
        self.running = True
        super().start()
    
    def get(self,proberty):
        self.cap.get(proberty)
    def _create_capture(self):
        try:
            self.cap = cv.VideoCapture(self.rtsp_url, cv.CAP_FFMPEG)
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                logger.warning(f"Camera {self.rtsp_url}: VideoCapture failed to open")
        except Exception as e:
            logger.warning(f"Camera {self.rtsp_url}: _create_capture failed: {e}")

    def release(self, timeout=None):
        self.running = False
        join_timeout = timeout if timeout is not None else self.RELEASE_TIMEOUT
        self.join(timeout=join_timeout)
        if self.is_alive():
            logger.warning(f"FreshestFrame thread did not stop within {join_timeout}s")
        try:
            self.cap.release()
        except Exception as e:
            logger.warning(f"Error releasing VideoCapture: {e}")

    def run(self):
        counter = 0
        consecutive_failures = 0
        while self.running:
            try:
                rv, img = self.cap.read()
            except Exception as e:
                logger.warning(f"Camera {self.rtsp_url}: cap.read() raised: {e}")
                rv, img = False, None
            if not rv or img is None:
                consecutive_failures += 1
                delay = min(
                    self.RECONNECT_INITIAL_DELAY * (2 ** (consecutive_failures - 1)),
                    self.RECONNECT_MAX_DELAY,
                )
                logger.warning(
                    f"Camera {self.rtsp_url}: lost frame "
                    f"(consecutive={consecutive_failures}), reconnecting in {delay:.1f}s..."
                )
                try:
                    self.cap.release()
                except Exception:
                    pass
                # Interruptible sleep so release()/stop() stays responsive
                # even during the max 30s backoff.
                slept = 0.0
                while slept < delay and self.running:
                    time.sleep(min(0.5, delay - slept))
                    slept += 0.5
                if not self.running:
                    break
                self._create_capture()
                continue

            consecutive_failures = 0
            with self.cond:
                self.frame = img
                self.latestnum = counter
                self.cond.notify_all()
            counter += 1
    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum+1
                if seqnumber < 1:
                    seqnumber = 1
                
                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)