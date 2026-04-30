from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from picamera2 import Picamera2


class PiCamera:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._cam = Picamera2()
        config = self._cam.create_preview_configuration(
            main={"size": (width, height)},
            sensor={"output_size": (1640, 1232)},
        )
        self._cam.configure(config)

    def start(self) -> None:
        self._cam.start()

    def stop(self) -> None:
        self._cam.stop()

    def capture_bgr(self) -> Optional[np.ndarray]:
        frame = self._cam.capture_array()
        if frame is None:
            return None
        # libcamera picks XBGR8888 (4-ch) or RGB888 (3-ch) depending on driver
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
