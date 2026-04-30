from __future__ import annotations

import cv2
import numpy as np

PERSON_CLASS_ID = 0
MIN_PERSON_HEIGHT_OVER_WIDTH = 0.25


def letterbox(im: np.ndarray, new_h: int, new_w: int, color: tuple = (114, 114, 114)):
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


def lb_xyxy_to_original(
    x1, y1, x2, y2, input_w, input_h, r, pad_left, pad_top, imw, imh
):
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
    predictions, input_w, input_h, r, pad_left, pad_top, imw, imh, conf_th,
    person_class_id=PERSON_CLASS_ID, min_h_over_w=MIN_PERSON_HEIGHT_OVER_WIDTH,
):
    boxes_xyxy, conf_list, class_list = [], [], []
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
    output_tensor, input_w, input_h, r, pad_left, pad_top, imw, imh, conf_th,
    person_class_id=PERSON_CLASS_ID, min_h_over_w=MIN_PERSON_HEIGHT_OVER_WIDTH,
):
    pred = np.squeeze(output_tensor)
    if pred.shape[0] == 84:
        pred = pred.T
    boxes = pred[:, :4]
    class_probs = pred[:, 4:]
    class_ids = np.argmax(class_probs, axis=1)
    confidences = np.max(class_probs, axis=1)

    boxes_xyxy, conf_list, class_list = [], [], []
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
    raw_out, input_w, input_h, r, pad_left, pad_top, imw, imh, conf_th,
    person_class_id=PERSON_CLASS_ID, min_h_over_w=MIN_PERSON_HEIGHT_OVER_WIDTH,
):
    shp = raw_out.shape
    if shp[-1] == 6 or (len(shp) == 2 and shp[1] == 6):
        return decode_yolo26_e2e(
            np.squeeze(raw_out), input_w, input_h, r, pad_left, pad_top,
            imw, imh, conf_th, person_class_id, min_h_over_w,
        )
    if len(shp) == 3 and shp[1] == 84:
        return decode_yolov8_raw(
            raw_out, input_w, input_h, r, pad_left, pad_top,
            imw, imh, conf_th, person_class_id, min_h_over_w,
        )
    raise ValueError(f"Unsupported YOLO output shape: {shp}")
