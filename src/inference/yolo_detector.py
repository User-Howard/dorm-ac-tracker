from __future__ import annotations

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
    from ai_edge_litert.interpreter import Interpreter

from inference.base import Detection
from inference.preprocessor import decode_detections, letterbox

PERSON_CLASS_ID = 0


class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float, nms_threshold: float) -> None:
        self._conf = conf_threshold
        self._nms = nms_threshold
        self._interpreter = Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()
        self._output = self._interpreter.get_output_details()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        imh, imw = frame_bgr.shape[:2]
        input_h = int(self._input[0]["shape"][1])
        input_w = int(self._input[0]["shape"][2])

        lb, r, pad_left, pad_top = letterbox(frame_bgr, input_h, input_w)
        input_data = np.expand_dims(lb, axis=0).astype(np.float32) / 255.0

        self._interpreter.set_tensor(self._input[0]["index"], input_data)
        self._interpreter.invoke()
        raw_out = self._interpreter.get_tensor(self._output[0]["index"])

        boxes_xyxy, conf_list, class_list = decode_detections(
            raw_out, input_w, input_h, r, pad_left, pad_top, imw, imh, self._conf
        )

        if not boxes_xyxy:
            return []

        indices = cv2.dnn.NMSBoxes(boxes_xyxy, conf_list, self._conf, self._nms)
        if indices is None or len(indices) == 0:
            return []

        return [
            Detection(box=tuple(boxes_xyxy[j]), confidence=conf_list[j])
            for j in indices.flatten()
            if class_list[j] == PERSON_CLASS_ID
        ]
