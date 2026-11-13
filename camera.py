import os
import threading
import time
import cv2


class ThreadedCamera:
    """Reads the webcam on a background thread so the main loop never waits on the camera.
    read() returns the newest frame (and waits only if no new frame has arrived yet)."""

    def __init__(self, index=0, width=640, height=480, fps=30):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY   # faster start on Windows
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # lets webcams hold 30 fps
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._id = 0
        self._last = 0
        self._stop = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = f
                    self._id += 1
            else:
                time.sleep(0.01)

    def read(self, timeout=2.0):
        t0 = time.time()
        while self._id == self._last:
            if time.time() - t0 > timeout:
                return False, None
            time.sleep(0.001)
        with self._lock:
            f, self._last = self._frame, self._id
        return True, f

    def release(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self.cap.release()
