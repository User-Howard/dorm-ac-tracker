from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np


class Detection(NamedTuple):
    box: tuple[int, int, int, int]  # xmin, ymin, xmax, ymax
    confidence: float


class PersonDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...
