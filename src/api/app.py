import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from loguru import logger
from pydantic import BaseModel

import config as cfg
from api import db
from drivers.frame_source import FrameSource
from hal.camera import PiCamera
from inference.yolo_detector import YoloDetector
from services.detection_service import SeatDetectionService

WIN_TITLE = "dorm-ac-tracker"
_INDEX_HTML = (Path(__file__).parent / "index.html").read_text()


class SeatIn(BaseModel):
    seat_id: str
    user_name: str
    x1: int
    y1: int
    x2: int
    y2: int


class AppState:
    def __init__(
        self,
        camera: PiCamera,
        detector: YoloDetector,
        seat_service: SeatDetectionService,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.seat_service = seat_service
        self.latest_frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()
        self.stop = False

    def reload_seats(self) -> None:
        seats = db.get_seats()
        self.seat_service.reload(seats)
        logger.info("Seats reloaded ({} seat(s))", len(seats))


def _draw_overlay(frame, detections, seats, occupied: dict[str, bool]) -> None:
    for d in detections:
        xmin, ymin, xmax, ymax = d.box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

    for seat in seats:
        color = (0, 255, 0) if occupied.get(seat["seat_id"]) else (0, 165, 255)
        cv2.rectangle(frame, (seat["x1"], seat["y1"]), (seat["x2"], seat["y2"]), color, 2)
        cv2.putText(
            frame, f"{seat['seat_id']} ({seat['user_name']})",
            (seat["x1"] + 4, seat["y1"] + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )


def _detection_loop(source: FrameSource, state: AppState, no_gui: bool) -> None:
    freq = cv2.getTickFrequency()
    t_prev = cv2.getTickCount()

    for frame in source:
        if state.stop:
            break

        detections = state.detector.detect(frame)
        events = state.seat_service.process(detections)

        for e in events:
            verb = "entered" if e["event"] == "enter" else "left"
            logger.info("Person {} {} ({})", verb, e["seat_id"], e["user_name"])
            db.log_occupancy(e["seat_id"], e["user_name"], e["event"])

        t_now = cv2.getTickCount()
        fps = freq / max(1, t_now - t_prev)
        t_prev = t_now

        _draw_overlay(frame, detections, state.seat_service._seats, state.seat_service.occupied)
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (8, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        with state.frame_lock:
            state.latest_frame = frame.copy()

        if not no_gui:
            cv2.imshow(WIN_TITLE, frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()

    no_gui = "--no-gui" in sys.argv
    conf = cfg.AppConfig()

    camera = PiCamera()
    camera.start()
    source = FrameSource(camera)
    detector = YoloDetector(conf.model.path, conf.model.conf_threshold, conf.model.nms_threshold)
    seat_service = SeatDetectionService(db.get_seats())

    state = AppState(camera, detector, seat_service)
    app.state.ctx = state

    thread = threading.Thread(
        target=_detection_loop,
        args=(source, state, no_gui),
        daemon=True,
    )
    thread.start()
    logger.info("Started with {} seat(s)", len(seat_service._seats))

    yield

    state.stop = True
    camera.stop()
    logger.info("Stopped.")


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


@app.get("/snapshot")
def snapshot():
    state: AppState = app.state.ctx
    with state.frame_lock:
        frame = state.latest_frame
    if frame is None:
        raise HTTPException(503, "No frame yet")
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(500, "Encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/seats")
def get_seats():
    return db.get_seats()


@app.post("/seats", status_code=204)
def save_seats(seats: List[SeatIn]):
    db.replace_seats([s.model_dump() for s in seats])


@app.post("/reload", status_code=204)
def reload_seats():
    app.state.ctx.reload_seats()
