"""
Coding 2 - camera only: YOLO TFLite live; print/show when a person appears.

Camera: same source as 3.yolo_camera.py - Picamera2 only, identical create_preview_configuration.

Requires the same Python environment as 3.yolo_camera.py (Picamera2 + libcamera).

Run:  python3 coding2.py
 SSH:  ... --no-gui
Press q to quit (GUI mode).
"""
import sys
import time

# Load OpenCV the same way as 3.yolo_camera.py (no Qt env vars before import).
# Tweaking QT_QPA_* before cv2 can break HighGUI on some VNC / desktop setups.
import cv2
import numpy as np

WIN_TITLE = "Coding 2 - YOLO person (camera)"

try:
    from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
    from ai_edge_litert.interpreter import Interpreter

from picamera2 import Picamera2

# ====== configuration ======
MODEL_PATH = "models/yolo26n_float32.tflite"
IMW, IMH = 640, 480

CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.5

PERSON_CLASS_ID = 0
MIN_PERSON_HEIGHT_OVER_WIDTH = 0.25


def letterbox(im, new_h, new_w, color=(114, 114, 114)):
    h0, w0 = im.shape[:2]
    r = min(new_w / w0, new_h / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (new_w - nw) / 2.0, (new_h - nh) / 2.0
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, left, top


def lb_xyxy_to_original(x1, y1, x2, y2, input_w, input_h, r, pad_left, pad_top, imw, imh):
    xmin_lb = float(x1) * input_w
    ymin_lb = float(y1) * input_h
    xmax_lb = float(x2) * input_w
    ymax_lb = float(y2) * input_h

    xmin = int(round((xmin_lb - pad_left) / r))
    ymin = int(round((ymin_lb - pad_top) / r))
    xmax = int(round((xmax_lb - pad_left) / r))
    ymax = int(round((ymax_lb - pad_top) / r))

    xmin = max(0, min(xmin, imw - 1))
    xmax = max(0, min(xmax, imw - 1))
    ymin = max(0, min(ymin, imh - 1))
    ymax = max(0, min(ymax, imh - 1))
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax


def decode_yolo26_e2e(
    predictions,
    input_w,
    input_h,
    r,
    pad_left,
    pad_top,
    imw,
    imh,
    conf_th,
    person_class_id,
    min_h_over_w,
):
    boxes_xyxy = []
    conf_list = []
    class_list = []

    for i in range(predictions.shape[0]):
        x1, y1, x2, y2, score, cls = predictions[i]
        if float(score) < conf_th:
            continue
        cid = int(round(float(cls)))
        xmin, ymin, xmax, ymax = lb_xyxy_to_original(
            x1, y1, x2, y2, input_w, input_h, r, pad_left, pad_top, imw, imh
        )
        bw = max(1, xmax - xmin)
        bh = max(1, ymax - ymin)
        if cid == person_class_id and (bh / float(bw)) < min_h_over_w:
            continue

        boxes_xyxy.append([xmin, ymin, xmax, ymax])
        conf_list.append(float(score))
        class_list.append(cid)

    return boxes_xyxy, conf_list, class_list


def decode_yolov8_raw(
    output_tensor,
    input_w,
    input_h,
    r,
    pad_left,
    pad_top,
    imw,
    imh,
    conf_th,
    person_class_id,
    min_h_over_w,
):
    pred = np.squeeze(output_tensor)
    if pred.shape[0] == 84:
        pred = pred.T
    boxes = pred[:, :4]
    class_probs = pred[:, 4:]
    class_ids = np.argmax(class_probs, axis=1)
    confidences = np.max(class_probs, axis=1)

    boxes_xyxy = []
    conf_list = []
    class_list = []

    for i in range(len(confidences)):
        if confidences[i] < conf_th:
            continue
        x, y, w, h = boxes[i]
        xmin_lb = (x - w / 2.0) * input_w
        ymin_lb = (y - h / 2.0) * input_h
        xmax_lb = (x + w / 2.0) * input_w
        ymax_lb = (y + h / 2.0) * input_h

        xmin = int(round((xmin_lb - pad_left) / r))
        ymin = int(round((ymin_lb - pad_top) / r))
        xmax = int(round((xmax_lb - pad_left) / r))
        ymax = int(round((ymax_lb - pad_top) / r))

        xmin = max(0, min(xmin, imw - 1))
        xmax = max(0, min(xmax, imw - 1))
        ymin = max(0, min(ymin, imh - 1))
        ymax = max(0, min(ymax, imh - 1))
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin

        bw = max(1, xmax - xmin)
        bh = max(1, ymax - ymin)
        cid = int(class_ids[i])
        if cid == person_class_id and (bh / float(bw)) < min_h_over_w:
            continue

        boxes_xyxy.append([xmin, ymin, xmax, ymax])
        conf_list.append(float(confidences[i]))
        class_list.append(cid)

    return boxes_xyxy, conf_list, class_list


def decode_detections(
    raw_out,
    input_w,
    input_h,
    r,
    pad_left,
    pad_top,
    imw,
    imh,
    conf_th,
    person_class_id=PERSON_CLASS_ID,
    min_h_over_w=MIN_PERSON_HEIGHT_OVER_WIDTH,
):
    shp = raw_out.shape
    if shp[-1] == 6 or (len(shp) == 2 and shp[1] == 6):
        predictions = np.squeeze(raw_out)
        return decode_yolo26_e2e(
            predictions,
            input_w,
            input_h,
            r,
            pad_left,
            pad_top,
            imw,
            imh,
            conf_th,
            person_class_id,
            min_h_over_w,
        )
    if len(shp) == 3 and shp[1] == 84:
        return decode_yolov8_raw(
            raw_out,
            input_w,
            input_h,
            r,
            pad_left,
            pad_top,
            imw,
            imh,
            conf_th,
            person_class_id,
            min_h_over_w,
        )
    raise ValueError("Unsupported YOLO output shape: {}".format(shp))


def person_indices_after_nms(
    boxes_xyxy, conf_list, class_list, conf_th, nms_th, person_class_id=PERSON_CLASS_ID
):
    person_indices = []
    if len(boxes_xyxy) == 0:
        return person_indices
    indices = cv2.dnn.NMSBoxes(boxes_xyxy, conf_list, conf_th, nms_th)
    if indices is None or len(indices) == 0:
        return person_indices
    for j in indices.flatten():
        if class_list[j] == person_class_id:
            person_indices.append(int(j))
    return person_indices


def infer_detections(
    frame_bgr,
    interpreter,
    input_details,
    output_details,
    conf_threshold,
    nms_threshold,
    person_class_id=PERSON_CLASS_ID,
    min_person_aspect=MIN_PERSON_HEIGHT_OVER_WIDTH,
):
    imh, imw = frame_bgr.shape[:2]
    input_height = int(input_details[0]["shape"][1])
    input_width = int(input_details[0]["shape"][2])

    lb, r, pad_left, pad_top = letterbox(frame_bgr, input_height, input_width)
    input_data = np.expand_dims(lb, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    raw_out = interpreter.get_tensor(output_details[0]["index"])
    boxes_xyxy, conf_list, class_list = decode_detections(
        raw_out,
        input_width,
        input_height,
        r,
        pad_left,
        pad_top,
        imw,
        imh,
        conf_threshold,
        person_class_id,
        min_person_aspect,
    )
    person_indices = person_indices_after_nms(
        boxes_xyxy, conf_list, class_list, conf_threshold, nms_threshold, person_class_id
    )
    return boxes_xyxy, conf_list, class_list, person_indices


def init_picamera2():
    """Identical camera setup to ``3.yolo_camera.py`` (initial camera section)."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (IMW, IMH)},
        sensor={"output_size": (1640, 1232)},
    )
    picam2.configure(config)
    picam2.start()
    return picam2


def read_frame_bgr(picam2):
    frame = picam2.capture_array()
    if frame is None:
        return None
    # libcamera often picks XBGR8888 (4 channels) for this preview config; RGB888 is 3 channels.
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def main():
    no_gui = "--no-gui" in sys.argv

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    picam2 = init_picamera2()
    cam_status = "Picamera2 main={}x{} sensor=1640x1232 (same as 3.yolo_camera.py)".format(
        IMW, IMH
    )
    had_person = False
    print("Coding 2 camera - {}. Person event on enter/leave.".format(cam_status))
    if no_gui:
        print("GUI off (--no-gui). Events print to terminal only; Ctrl+C to stop.")
    else:
        print("Press 'q' in the OpenCV window to quit.")

    time.sleep(1.0)

    freq = cv2.getTickFrequency()
    t_prev = cv2.getTickCount()

    try:
        while True:
            frame = read_frame_bgr(picam2)
            if frame is None:
                continue

            boxes_xyxy, conf_list, class_list, person_indices = infer_detections(
                frame,
                interpreter,
                input_details,
                output_details,
                CONF_THRESHOLD,
                NMS_THRESHOLD,
                PERSON_CLASS_ID,
            )

            has_person = len(person_indices) > 0
            if has_person and not had_person:
                print("[EVENT] Person detected (entered frame).")
            if not has_person and had_person:
                print("[EVENT] Person left frame.")
            had_person = has_person

            if has_person:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    "EVENT: PERSON DETECTED",
                    (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2,
                )
                for j in person_indices:
                    xmin, ymin, xmax, ymax = boxes_xyxy[j]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        "person {:.2f}".format(min(1.0, conf_list[j])),
                        (xmin, max(18, ymin - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "No person",
                    (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (120, 120, 120),
                    2,
                )

            t_now = cv2.getTickCount()
            fps = freq / max(1e-6, (t_now - t_prev))
            t_prev = t_now
            cv2.putText(
                frame,
                "FPS: {:.1f}".format(fps),
                (8, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # Camera + clock so you can tell which device is live and that frames advance.
            info_y = frame.shape[0] - 42
            cv2.putText(
                frame,
                cam_status[: min(70, len(cam_status))],
                (8, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 220, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                time.strftime("%H:%M:%S"),
                (frame.shape[1] - 96, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 220, 200),
                1,
                cv2.LINE_AA,
            )

            if not no_gui:
                # Same HighGUI pattern as 3.yolo_camera.py (imshow then one waitKey).
                cv2.imshow(WIN_TITLE, frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                time.sleep(0.02)
    finally:
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
