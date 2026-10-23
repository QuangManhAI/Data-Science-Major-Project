"""Bộ nhận diện người mở rộng, hỗ trợ Ultralytics YOLO (v8n, 11n, 26n) và Intel OpenVINO.

`Detector` là interface chung cho cả production server và benchmark đánh giá.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import cv2
import numpy as np

from protocol import YOLO_CONF

PERSON_CLASS_ID: int = 0  # COCO class 0: person


class Detector(Protocol):
    def detect(self, img: np.ndarray) -> tuple[list[dict[str, float]], float]:
        """Trả `(detections, infer_ms)`; detections theo tọa độ chuẩn hóa [0, 1], sắp conf giảm dần."""
        ...


def normalize_boxes(
    xyxy: np.ndarray, confs: np.ndarray, width: int, height: int
) -> list[dict[str, float]]:
    """Box pixel ➔ tọa độ chuẩn hóa [0, 1], làm tròn 4 chữ số, sắp conf giảm dần.

    Box suy biến sau khi kẹp/làm tròn (x1 >= x2 hoặc y1 >= y2) bị bỏ.
    """
    dets: list[dict[str, float]] = []
    if len(xyxy) == 0:
        return dets
    for (x1, y1, x2, y2), conf in zip(np.asarray(xyxy, dtype=float), np.asarray(confs, dtype=float)):
        d = {
            "x1": round(min(max(x1 / width, 0.0), 1.0), 4),
            "y1": round(min(max(y1 / height, 0.0), 1.0), 4),
            "x2": round(min(max(x2 / width, 0.0), 1.0), 4),
            "y2": round(min(max(y2 / height, 0.0), 1.0), 4),
            "conf": round(float(conf), 4),
        }
        if d["x1"] < d["x2"] and d["y1"] < d["y2"]:
            dets.append(d)
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


class YoloDetector:
    """YOLO (v8n, 11n, 26n) qua ultralytics. Load model một lần lúc khởi tạo."""

    def __init__(self, model_path: str, conf: float = YOLO_CONF) -> None:
        from ultralytics import YOLO  # import trễ: test không cần ultralytics

        self._model: Any = YOLO(model_path)
        self._conf: float = conf
        # Chạy nháp một lần để warm-up
        self._model(
            np.zeros((480, 640, 3), dtype=np.uint8),
            classes=[PERSON_CLASS_ID],
            conf=self._conf,
            verbose=False,
        )

    def detect(self, img: np.ndarray) -> tuple[list[dict[str, float]], float]:
        t0 = time.perf_counter()
        result = self._model(
            img,
            classes=[PERSON_CLASS_ID],
            conf=self._conf,
            verbose=False,
        )[0]
        boxes = result.boxes
        h, w = img.shape[:2]
        dets = normalize_boxes(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), w, h)
        return dets, (time.perf_counter() - t0) * 1000.0


class IntelOpenVINODetector:
    """Detector cho dòng Intel OpenVINO Model Zoo (person-detection-retail-0013 / 0202)."""

    def __init__(self, xml_path: str, conf: float = YOLO_CONF, device: str = "CPU") -> None:
        import openvino as ov

        self._conf: float = conf
        core = ov.Core()
        model = core.read_model(xml_path)
        self._compiled = core.compile_model(model, device)
        self._input_layer = self._compiled.input(0)
        self._output_layer = self._compiled.output(0)
        # Shape đầu vào chuẩn NCHW: [1, 3, H, W]
        _, _, self._in_h, self._in_w = self._input_layer.shape

        # Warm-up
        dummy = np.zeros((1, 3, self._in_h, self._in_w), dtype=np.float32)
        self._compiled([dummy])

    def detect(self, img: np.ndarray) -> tuple[list[dict[str, float]], float]:
        orig_h, orig_w = img.shape[:2]
        t0 = time.perf_counter()

        # Resize về kích thước đầu vào của model OpenVINO
        resized = cv2.resize(img, (self._in_w, self._in_h))
        # Chuẩn bị tensor NCHW
        input_tensor = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)

        preds = self._compiled([input_tensor])[self._output_layer]
        # Output shape: [1, 1, N, 7] với mỗi dòng: [image_id, label, conf, x1, y1, x2, y2]
        detections = preds.reshape(-1, 7)

        xyxy_list: list[list[float]] = []
        conf_list: list[float] = []

        for det in detections:
            image_id, _label, conf, x1, y1, x2, y2 = det
            if image_id < 0:
                break
            if conf >= self._conf:
                xyxy_list.append([
                    float(np.clip(x1 * orig_w, 0.0, orig_w)),
                    float(np.clip(y1 * orig_h, 0.0, orig_h)),
                    float(np.clip(x2 * orig_w, 0.0, orig_w)),
                    float(np.clip(y2 * orig_h, 0.0, orig_h)),
                ])
                conf_list.append(float(conf))

        xyxy_arr = np.array(xyxy_list, dtype=float) if xyxy_list else np.zeros((0, 4))
        conf_arr = np.array(conf_list, dtype=float) if conf_list else np.zeros((0,))
        dets = normalize_boxes(xyxy_arr, conf_arr, orig_w, orig_h)
        return dets, (time.perf_counter() - t0) * 1000.0
