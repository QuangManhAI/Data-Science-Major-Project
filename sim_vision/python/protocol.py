"""Hợp đồng message WebSocket (Mục 2.2): parse frame, dựng message trả về.

Hàm thuần, không phụ thuộc ultralytics.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

import cv2
import numpy as np

# Mục 2.1 (bản Python của các hằng số phía server).
WS_HOST: str = "127.0.0.1"
WS_PORT: int = 8765
YOLO_CONF: float = 0.4


class ProtocolError(Exception):
    """Message không hợp lệ. `frame_id` là id đọc được (hoặc None) để gửi kèm `error`."""

    def __init__(self, message: str, frame_id: int | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.frame_id: int | None = frame_id


def _is_int(value: Any) -> bool:
    # bool là lớp con của int trong Python; không chấp nhận true/false làm số.
    return isinstance(value, int) and not isinstance(value, bool)


def parse_frame(text: str | bytes) -> tuple[int, np.ndarray]:
    """Parse message `frame`, trả `(frame_id, ảnh BGR)`. Lỗi ➔ ProtocolError."""
    frame_id, img, _ = parse_frame_opts(text)
    return frame_id, img


def wants_faces(text: str | bytes) -> bool:
    """Frame có `"faces": true` ➔ server trả thêm box mặt (YuNet) để hiển thị. Lỗi parse ➔ False."""
    try:
        msg = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return False
    return isinstance(msg, dict) and msg.get("faces") is True


def parse_frame_opts(text: str | bytes) -> tuple[int, np.ndarray, str]:
    """Như parse_frame, thêm trường tùy chọn `reid` ("enroll" / "track" / "" — docs/UPGRADE_HRI_PLAN.md D3–D4).
    Giá trị lạ ➔ "" (bỏ qua, không lỗi)."""
    try:
        msg = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        raise ProtocolError("invalid json") from None
    if not isinstance(msg, dict):
        raise ProtocolError("invalid json")

    frame_id: int | None = msg.get("frame_id") if _is_int(msg.get("frame_id")) else None
    if frame_id is not None and frame_id < 0:
        frame_id = None

    if "type" not in msg:
        raise ProtocolError("missing field: type", frame_id)
    if msg["type"] != "frame":
        raise ProtocolError("invalid field: type", frame_id)
    if "frame_id" not in msg:
        raise ProtocolError("missing field: frame_id")
    if frame_id is None:
        raise ProtocolError("invalid field: frame_id")
    for name in ("jpeg_b64", "width", "height"):
        if name not in msg:
            raise ProtocolError(f"missing field: {name}", frame_id)
    if not isinstance(msg["jpeg_b64"], str):
        raise ProtocolError("invalid field: jpeg_b64", frame_id)
    for name in ("width", "height"):
        if not _is_int(msg[name]) or msg[name] <= 0:
            raise ProtocolError(f"invalid field: {name}", frame_id)

    try:
        raw = base64.b64decode(msg["jpeg_b64"], validate=True)
    except (binascii.Error, ValueError):
        raise ProtocolError("invalid jpeg", frame_id) from None
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR) if raw else None
    if img is None:
        raise ProtocolError("invalid jpeg", frame_id)
    if img.shape[1] != msg["width"] or img.shape[0] != msg["height"]:
        raise ProtocolError("size mismatch", frame_id)
    reid = msg.get("reid", "")
    return frame_id, img, reid if reid in ("enroll", "track") else ""


def build_detections(frame_id: int, detections: list[dict[str, float]], infer_ms: float,
                     faces: list[dict[str, float]] | None = None) -> str:
    msg = {
        "type": "detections",
        "frame_id": frame_id,
        "infer_ms": round(float(infer_ms), 1),
        "detections": detections,
    }
    if faces is not None:
        msg["faces"] = faces
    return json.dumps(msg)


def build_error(frame_id: int | None, message: str) -> str:
    return json.dumps({"type": "error", "frame_id": frame_id, "message": message})
