from __future__ import annotations

from loguru import logger

from inference.base import Detection


class ZoneDetectionService:
    def __init__(self, zone: tuple[int, int, int, int]) -> None:
        self._zone = zone
        self._in_zone = False

    def process(self, detections: list[Detection]) -> bool:
        x1, y1, x2, y2 = self._zone

        in_zone = any(
            x1 <= (d.box[0] + d.box[2]) // 2 <= x2
            and y1 <= (d.box[1] + d.box[3]) // 2 <= y2
            for d in detections
        )

        if in_zone and not self._in_zone:
            logger.info("Person entered zone")
        elif not in_zone and self._in_zone:
            logger.info("Person left zone")

        self._in_zone = in_zone
        return in_zone
