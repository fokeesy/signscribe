"""Low-latency frame sources: a threaded webcam reader and a video-file reader.

The reader thread always keeps only the *newest* frame, so a slow consumer never builds up a
backlog of stale frames (which is what makes naive OpenCV loops feel laggy).
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .hand import HandObservation


class CameraError(RuntimeError):
    pass


@dataclass
class FramePacket:
    """One frame delivered to the engine. ``hands`` is pre-filled by simulated sources."""

    frame: np.ndarray
    t: float
    hands: list[HandObservation] | None = None
    info: dict = field(default_factory=dict)


class CameraStream:
    """Threaded webcam capture that always exposes the most recent frame."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.name = f"Camera {index}"
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._seq = 0
        self._last_seq = 0
        self._running = False
        self.error: str | None = None

    def start(self) -> None:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                break
            cap.release()
            cap = None
        if cap is None:
            raise CameraError(
                f"Could not open camera {self.index}. Is it in use by another app, or is camera "
                "access blocked in your OS privacy settings?"
            )
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # much faster than raw YUY2
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera-reader", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        fails = 0
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 60:
                    self.error = "The camera stopped delivering frames."
                    with self._cond:
                        self._running = False
                        self._cond.notify_all()
                    return
                time.sleep(0.01)
                continue
            fails = 0
            with self._cond:
                self._frame = frame
                self._stamp = time.perf_counter()
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 0.5) -> FramePacket | None:
        """Block until a frame newer than the last one returned is available."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._last_seq or not self._running, timeout):
                return None
            if self._seq == self._last_seq:
                return None
            self._last_seq = self._seq
            return FramePacket(self._frame, self._stamp)

    @property
    def alive(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileSource:
    """Plays a video file in real time (for testing without a webcam)."""

    def __init__(self, path: str, loop: bool = True) -> None:
        self.path, self.loop = path, loop
        self.name = f"Video {path}"
        self._cap: cv2.VideoCapture | None = None
        self._next = 0.0
        self.error: str | None = None
        self._period = 1 / 30

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open video file: {self.path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._period = 1.0 / max(fps, 1.0)
        self._next = time.perf_counter()

    def read(self, timeout: float = 0.5) -> FramePacket | None:
        assert self._cap is not None
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(min(delay, timeout))
        self._next = max(self._next + self._period, time.perf_counter() - self._period)
        ok, frame = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                self.error = "End of video."
                return None
        return FramePacket(frame, time.perf_counter())

    @property
    def alive(self) -> bool:
        return self.error is None

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
