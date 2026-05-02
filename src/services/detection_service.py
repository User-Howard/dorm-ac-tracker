from __future__ import annotations

from typing import TypedDict

from inference.base import Detection


class OccupancyEvent(TypedDict):
    seat_id: str
    user_name: str
    event: str  # "enter" | "leave"


class SeatDetectionService:
    """Tracks person presence for each seat ROI independently."""

    def __init__(self, seats: list[dict]) -> None:
        # seats: list of {seat_id, user_name, x1, y1, x2, y2}
        self._seats = seats
        self._occupied: dict[str, bool] = {s["seat_id"]: False for s in seats}

    @property
    def occupied(self) -> dict[str, bool]:
        return dict(self._occupied)

    def reload(self, seats: list[dict]) -> None:
        self._seats = seats
        self._occupied = {s["seat_id"]: False for s in seats}

    def process(self, detections: list[Detection]) -> list[OccupancyEvent]:
        events: list[OccupancyEvent] = []
        for seat in self._seats:
            x1, y1, x2, y2 = seat["x1"], seat["y1"], seat["x2"], seat["y2"]
            now_occupied = any(
                x1 <= (d.box[0] + d.box[2]) // 2 <= x2
                and y1 <= (d.box[1] + d.box[3]) // 2 <= y2
                for d in detections
            )
            prev = self._occupied[seat["seat_id"]]
            if now_occupied != prev:
                self._occupied[seat["seat_id"]] = now_occupied
                events.append({
                    "seat_id": seat["seat_id"],
                    "user_name": seat["user_name"],
                    "event": "enter" if now_occupied else "leave",
                })
        return events
