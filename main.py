import sys

import cv2
from loguru import logger

from config import CONF_THRESHOLD, MODEL_PATH, NMS_THRESHOLD, ZONE
from drivers.frame_source import FrameSource
from hal.camera import PiCamera
from inference.yolo_detector import YoloDetector
from services.detection_service import ZoneDetectionService

WIN_TITLE = "dorm-ac-tracker"


def draw_overlay(frame, detections, zone, in_zone: bool) -> None:
    zx1, zy1, zx2, zy2 = zone
    zone_color = (0, 255, 0) if in_zone else (0, 165, 255)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), zone_color, 2)

    for d in detections:
        xmin, ymin, xmax, ymax = d.box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
        cv2.putText(
            frame, f"{d.confidence:.2f}",
            (xmin, max(16, ymin - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
        )

    label = "IN ZONE" if in_zone else "no person"
    color = (0, 255, 0) if in_zone else (120, 120, 120)
    cv2.putText(frame, label, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def main() -> None:
    no_gui = "--no-gui" in sys.argv

    cam = PiCamera()
    cam.start()

    detector = YoloDetector(MODEL_PATH, CONF_THRESHOLD, NMS_THRESHOLD)
    service = ZoneDetectionService(ZONE)
    source = FrameSource(cam)

    logger.info("Started. Zone={}", ZONE)
    if not no_gui:
        logger.info("Press q to quit.")

    freq = cv2.getTickFrequency()
    t_prev = cv2.getTickCount()

    try:
        for frame in source:
            detections = detector.detect(frame)
            in_zone = service.process(detections)

            t_now = cv2.getTickCount()
            fps = freq / max(1, t_now - t_prev)
            t_prev = t_now

            draw_overlay(frame, detections, ZONE, in_zone)
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (8, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

            if not no_gui:
                cv2.imshow(WIN_TITLE, frame)
                if cv2.waitKey(1) == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
