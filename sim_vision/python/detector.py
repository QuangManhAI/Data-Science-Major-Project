"""Bộ nhận diện đối tượng bệnh viện qua YOLOv6 / YOLOv8.

Hỗ trợ 17 nhãn đối tượng bệnh viện:
  0: patient, 1: doctor, 2..16: vật cản và trang thiết bị y tế.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from protocol import YOLO_CONF

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent


def resolve_model_path(path: str | Path | None = None) -> str:
    candidates: list[Path] = []
    if path:
        p = Path(path)
        candidates.extend([p, _REPO_ROOT / p, _THIS_DIR / p])
    candidates.extend([
        _THIS_DIR / "yolo26n_hospital_robot_best_map92.6.pt",
        _REPO_ROOT / "models" / "yolo26n_hospital_robot_v6_best" / "yolo26n_hospital_robot_best_map92.6.pt",
        _REPO_ROOT / "sim_vision" / "python" / "yolo26n_hospital_robot_best_map92.6.pt",
        _THIS_DIR / "yolov8n.pt",
        _REPO_ROOT / "models" / "yolov8n" / "yolov8n.pt",
    ])
    for c in candidates:
        if c.is_file():
            return str(c)
    return str(candidates[0] if candidates else "")


DEFAULT_YOLO_PATH = resolve_model_path("models/yolo26n_hospital_robot_v6_best/yolo26n_hospital_robot_best_map92.6.pt")
FALLBACK_YOLO_PATH = resolve_model_path("sim_vision/python/yolo26n_hospital_robot_best_map92.6.pt")

HOSPITAL_CLASSES: dict[int, str] = {
    0: "patient",
    1: "doctor",
    2: "wheelchair",
    3: "queue_kiosk",
    4: "reception_counter",
    5: "sanitizer_stand",
    6: "pccc_station",
    7: "medical_waste_bin",
    8: "water_dispenser",
    9: "wc_station",
    10: "elevator_doors",
    11: "hospital_bed",
    12: "stretcher_cart",
    13: "iv_stand",
    14: "medical_trolley",
    15: "vital_monitor",
    16: "hospital_chair",
}


def load_class_mapping(model_path: str | Path | None = None) -> dict[int, str]:
    """Tải từ điển 17 nhãn từ classes_17.json hoặc data.yaml nếu có."""
    import json
    candidates: list[Path] = []
    if model_path:
        p = Path(model_path).resolve()
        candidates.extend([p.parent / "classes_17.json", p.parent / "classes.txt"])
    candidates.extend([
        _REPO_ROOT / "models" / "yolo26n_hospital_robot_v6_best" / "classes_17.json",
        _REPO_ROOT / "models" / "yolo26n_hospital_robot_v6_best" / "classes.txt",
        _THIS_DIR / "classes_17.json",
    ])
    for c in candidates:
        if c.is_file():
            try:
                if c.suffix == ".json":
                    with open(c, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    return {int(k): str(v) for k, v in raw.items()}
                elif c.suffix == ".txt":
                    with open(c, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    return {i: name for i, name in enumerate(lines)}
            except Exception:
                continue
    return dict(HOSPITAL_CLASSES)


class Detector(Protocol):
    def detect(self, img: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        """Trả `(detections, infer_ms)`; detections chứa x1, y1, x2, y2, conf, class_id, class_name."""
        ...


def normalize_boxes(
    xyxy: np.ndarray,
    confs: np.ndarray,
    classes: np.ndarray,
    width: int,
    height: int,
    class_map: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Box pixel ➔ tọa độ chuẩn hóa [0, 1], làm tròn 4 chữ số, sắp conf giảm dần."""
    dets: list[dict[str, Any]] = []
    cmap = class_map if class_map is not None else HOSPITAL_CLASSES
    for (x1, y1, x2, y2), conf, cls_id in zip(
        np.asarray(xyxy, dtype=float),
        np.asarray(confs, dtype=float),
        np.asarray(classes, dtype=int),
    ):
        cid = int(cls_id)
        d = {
            "x1": round(min(max(x1 / width, 0.0), 1.0), 4),
            "y1": round(min(max(y1 / height, 0.0), 1.0), 4),
            "x2": round(min(max(x2 / width, 0.0), 1.0), 4),
            "y2": round(min(max(y2 / height, 0.0), 1.0), 4),
            "conf": round(float(conf), 4),
            "class_id": cid,
            "class_name": cmap.get(cid, f"class_{cid}"),
        }
        if d["x1"] < d["x2"] and d["y1"] < d["y2"]:
            dets.append(d)
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


class YoloDetector:
    """YOLO model loader qua ultralytics."""

    def __init__(self, model_path: str = DEFAULT_YOLO_PATH, conf: float = YOLO_CONF) -> None:
        from ultralytics import YOLO

        resolved_path = resolve_model_path(model_path)

        self._model: Any = YOLO(resolved_path)
        self._conf: float = conf
        self._names: dict[int, str] = load_class_mapping(resolved_path)
        if hasattr(self._model, "names") and isinstance(self._model.names, dict) and self._model.names:
            for k, v in self._model.names.items():
                if not str(v).lower().startswith("class_"):
                    self._names[int(k)] = str(v)
        # Chạy nháp một lần để khởi động model
        self._model(
            np.zeros((480, 640, 3), dtype=np.uint8),
            conf=self._conf,
            verbose=False,
        )

    def detect(self, img: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        result = self._model(img, conf=self._conf, verbose=False)[0]
        boxes = result.boxes
        h, w = img.shape[:2]
        if len(boxes) == 0:
            return [], (time.perf_counter() - t0) * 1000.0

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        dets = normalize_boxes(xyxy, confs, clss, w, h, self._names)
        return dets, (time.perf_counter() - t0) * 1000.0
