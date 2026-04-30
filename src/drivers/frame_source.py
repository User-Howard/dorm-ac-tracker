from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from hal.camera import PiCamera


class FrameSource:
    def __init__(self, camera: PiCamera) -> None:
        self._camera = camera

    def read(self) -> Optional[np.ndarray]:
        return self._camera.capture_bgr()

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is not None:
                yield frame
