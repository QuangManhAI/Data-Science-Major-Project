"""Gắn embedding nhận diện vào box bệnh nhân trong yolo_server (docs/UPGRADE_HRI_PLAN.md mục D3–D4).

Server không nhớ ai: chỉ tính đặc trưng cho frame Godot yêu cầu. Việc ghi nhớ / so khớp nằm ở Godot
(scripts/patient_identity.gd), là nơi biết robot đang nói chuyện với ai và đang dẫn ai.
  - "track":  box patient ➔ "emb_b64" (OSNet x0.25, 512 float32 little-endian, đã chuẩn hóa L2, base64).
  - "enroll": như track + "face" (YuNet thấy mặt trong nửa trên box) để xác nhận đúng người đang nhìn robot.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import numpy as np

log = logging.getLogger("reid_service")

REID_MODES: tuple[str, ...] = ("enroll", "track")
PATIENT_CLASS_ID = 0
MIN_BOX_H = 0.08          # box thấp hơn 8% chiều cao ảnh (~ xa > 15 m) ➔ quá nhỏ, bỏ


def encode_emb(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode("ascii")


def decode_emb(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def is_patient(d: dict[str, Any]) -> bool:
    return int(d.get("class_id", -1)) == PATIENT_CLASS_ID or d.get("class_name") == "patient"


def crop_px(img: np.ndarray, d: dict[str, Any]) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    return (int(d["x1"] * w), int(d["y1"] * h), int(round(d["x2"] * w)), int(round(d["y2"] * h)))


class ReidAugmenter:
    """Nạp model lần đầu dùng (server khởi động nhanh); thiếu file model ➔ tắt, log một lần."""

    def __init__(self, reid=None, face=None) -> None:
        self._reid = reid
        self._face = face
        self._failed = False

    def preload(self) -> bool:
        """Nạp model ngay khi server khởi động: lần đầu robot cần (lúc nói chuyện) đã sẵn sàng."""
        return self._ensure()

    def _ensure(self) -> bool:
        if self._reid is not None:
            return True
        if self._failed:
            return False
        try:
            from identity.models import FaceModel, ReidModel

            self._reid = ReidModel("x0_25")
            self._face = FaceModel()
            log.info("ReID models loaded (OSNet x0.25 on %s + YuNet)", self._reid.device)
            return True
        except Exception as e:  # thiếu file / thiếu torch ➔ server vẫn chạy, không có emb
            self._failed = True
            log.warning("ReID disabled: %s (chạy python -m identity.download_models)", e)
            return False

    def detect_faces(self, img: np.ndarray) -> list[dict[str, float]]:
        """Box mặt YuNet, toạ độ chuẩn hoá [0, 1] như box YOLO (để hiển thị / kiểm tra mặt khi ghi nhớ)."""
        if not self._ensure() or self._face is None:
            return []
        h, w = img.shape[:2]
        out = []
        for f in self._face.detect(img):
            x, y, bw, bh = f.box
            out.append({"x1": round(max(x / w, 0.0), 4), "y1": round(max(y / h, 0.0), 4),
                        "x2": round(min((x + bw) / w, 1.0), 4), "y2": round(min((y + bh) / h, 1.0), 4),
                        "conf": round(float(f.score), 4)})
        return [f for f in out if f["x1"] < f["x2"] and f["y1"] < f["y2"]]

    def augment(self, img: np.ndarray, dets: list[dict[str, Any]], mode: str,
                faces: list[dict[str, float]] | None = None) -> None:
        """faces: box mặt đã tính (chuẩn hoá) ➔ không chạy YuNet lần nữa khi enroll."""
        if mode not in REID_MODES:
            return
        targets = [d for d in dets if is_patient(d) and d["y2"] - d["y1"] >= MIN_BOX_H]
        if not targets or not self._ensure():
            return
        crops = []
        for d in targets:
            x1, y1, x2, y2 = crop_px(img, d)
            crops.append(img[y1:y2, x1:x2])
        for d, e in zip(targets, self._reid.embed(crops)):
            d["emb_b64"] = encode_emb(e)
        if mode == "enroll" and self._face is not None:
            if faces is None:
                faces = self.detect_faces(img)
            for d in targets:
                d["face"] = any(d["x1"] <= (f["x1"] + f["x2"]) / 2 <= d["x2"]
                                and d["y1"] <= (f["y1"] + f["y2"]) / 2 <= d["y1"] + 0.5 * (d["y2"] - d["y1"])
                                for f in faces)
