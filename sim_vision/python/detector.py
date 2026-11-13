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
    if path:
        p = Path(path)
        for cand in [
            p,
            _REPO_ROOT / p,
            _THIS_DIR / p,
            _REPO_ROOT / "release_models" / p.name,
            _REPO_ROOT / "release_models" / "yolo26n_hospital_robot_v6_best" / p.name,
        ]:
            if cand.is_file():
                return str(cand)
        if p.is_file():
            return str(p)
    candidates: list[Path] = [
        _REPO_ROOT / "release_models" / "yolo26n_hospital_robot_v6_best" / "yolo26n_hospital_robot_best_map92.6.pt",
        _THIS_DIR / "yolo26n_hospital_robot_best_map92.6.pt",
        _REPO_ROOT / "models" / "yolo26n_hospital_robot_v6_best" / "yolo26n_hospital_robot_best_map92.6.pt",
        _REPO_ROOT / "sim_vision" / "python" / "yolo26n_hospital_robot_best_map92.6.pt",
        _REPO_ROOT / "models" / "yolo26n_hospital_robot_v6_best" / "yolo26n.pt",
        _THIS_DIR / "yolov8n.pt",
        _REPO_ROOT / "models" / "yolov8n" / "yolov8n.pt",
    ]
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
    classes: np.ndarray | int | None = None,
    width: int | None = None,
    height: int | None = None,
    class_map: dict[int, str] | None = None,
    include_class: bool = True,
) -> list[dict[str, Any]]:
    """Box pixel ➔ tọa độ chuẩn hóa [0, 1], làm tròn 4 chữ số, sắp conf giảm dần.
    Tương thích cả gọi cũ (xyxy, confs, width, height) lẫn đa nhãn (xyxy, confs, classes, w, h)."""
    legacy_call = False
    if isinstance(classes, (int, float, np.integer)):
        h = int(width) if width is not None else 480
        w = int(classes)
        classes = np.zeros(len(confs), dtype=int)
        width = w
        height = h
        legacy_call = True
    elif classes is None:
        classes = np.zeros(len(confs), dtype=int)
    if width is None:
        width = 640
    if height is None:
        height = 480

    dets: list[dict[str, Any]] = []
    cmap = class_map if class_map is not None else HOSPITAL_CLASSES
    for (x1, y1, x2, y2), conf, cls_id in zip(
        np.asarray(xyxy, dtype=float),
        np.asarray(confs, dtype=float),
        np.asarray(classes, dtype=int),
    ):
        cid = int(cls_id)
        d: dict[str, Any] = {
            "x1": round(min(max(x1 / width, 0.0), 1.0), 4),
            "y1": round(min(max(y1 / height, 0.0), 1.0), 4),
            "x2": round(min(max(x2 / width, 0.0), 1.0), 4),
            "y2": round(min(max(y2 / height, 0.0), 1.0), 4),
            "conf": round(float(conf), 4),
        }
        if not legacy_call and include_class:
            d["class_id"] = cid
            d["class_name"] = cmap.get(cid, f"class_{cid}")
        if d["x1"] < d["x2"] and d["y1"] < d["y2"]:
            dets.append(d)
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


# YOLO26n / YOLOv8n base (COCO 80 lớp) dùng tạm khi chưa có trọng số hospital 17 lớp:
# map vài lớp COCO sang ID bệnh viện; lớp còn lại đẩy ra ngoài dải 0..16 (+100) để
# policy chỉ coi là vật cản (tránh COCO 1 "bicycle" ➔ doctor, 5 "bus" ➔ door).
COCO_TO_HOSPITAL: dict[str, int] = {
    "person": 0,        # patient (COCO không phân biệt bác sĩ / bệnh nhân)
    "bed": 11,          # hospital_bed
    "chair": 16,        # hospital_chair
    "bench": 16,
    "couch": 16,
}
COCO_OFFSET: int = 100


def is_coco_names(names: dict[int, str]) -> bool:
    return len(names) == 80 and str(names.get(0, "")).lower() == "person"


class YoloDetector:
    """YOLO model loader qua ultralytics."""

    def __init__(self, model_path: str = DEFAULT_YOLO_PATH, conf: float = YOLO_CONF, device: str = "auto") -> None:
        from ultralytics import YOLO

        resolved_path = resolve_model_path(model_path)

        self._model: Any = YOLO(resolved_path)
        self._conf: float = conf
        # GPU Apple (MPS): ~8 ms / frame thay vì ~19 ms trên CPU, box giống hệt (lệch < 0.001 px).
        if device == "auto":
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device: str = device
        self._names: dict[int, str] = load_class_mapping(resolved_path)
        if hasattr(self._model, "names") and isinstance(self._model.names, dict) and self._model.names:
            for k, v in self._model.names.items():
                if not str(v).lower().startswith("class_"):
                    self._names[int(k)] = str(v)
        self._coco_remap: np.ndarray | None = None
        model_names = getattr(self._model, "names", None)
        if isinstance(model_names, dict) and is_coco_names(model_names):
            lut = np.arange(len(model_names), dtype=int) + COCO_OFFSET
            names: dict[int, str] = {}
            for k, v in model_names.items():
                hid = COCO_TO_HOSPITAL.get(str(v).lower())
                if hid is not None:
                    lut[int(k)] = hid
                    names[hid] = HOSPITAL_CLASSES[hid]
                else:
                    names[int(k) + COCO_OFFSET] = f"coco_{v}"
            self._coco_remap = lut
            self._names = names
        # Chạy nháp một lần để khởi động model
        self._model(
            np.zeros((480, 640, 3), dtype=np.uint8),
            conf=self._conf,
            verbose=False,
            device=self.device,
        )

    def detect(self, img: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        result = self._model(img, conf=self._conf, verbose=False, device=self.device)[0]
        boxes = result.boxes
        h, w = img.shape[:2]
        if len(boxes) == 0:
            return [], (time.perf_counter() - t0) * 1000.0

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        if self._coco_remap is not None:
            clss = self._coco_remap[clss]
        dets = normalize_boxes(xyxy, confs, clss, w, h, self._names)
        return dets, (time.perf_counter() - t0) * 1000.0
