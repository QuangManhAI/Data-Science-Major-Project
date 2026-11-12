"""Nạp model nhận diện người (pretrained, không train lại).

- Mặt: YuNet (phát hiện, OpenCV Zoo, MIT), chạy bằng cv2 — chỉ dùng lúc ghi nhớ để xác nhận đúng người đang nói.
- Toàn thân: OSNet x0.25 MSMT17 (embedding 512-D, MIT), chạy bằng torch.
(Đã thử và bỏ: SFace nhận diện mặt, OSNet x1.0 — không chính xác hơn, xem sim_vision/reports/identity_eval.md.)
File model ở models/identity/ (tải bằng `python -m identity.download_models`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
MODEL_DIR = PROJECT_ROOT / "models" / "identity"
YUNET = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
OSNET = {"x0_25": MODEL_DIR / "osnet_x0_25_msmt17.pth"}
# Chỉ để đo thí nghiệm mặt dán (không dùng khi robot chạy): python -m identity.download_models --sface
SFACE = MODEL_DIR / "face_recognition_sface_2021dec.onnx"


@dataclass
class Face:
    box: tuple[float, float, float, float]   # x, y, w, h (pixel, trong ảnh đưa vào)
    score: float
    raw: np.ndarray                          # hàng 15 số của YuNet (box + 5 điểm mốc + score)


def l2n(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-8)


class FaceModel:
    def __init__(self, score_threshold: float = 0.6) -> None:
        self.det = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), score_threshold, 0.3, 50)

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None:
            return []
        return [Face(tuple(map(float, f[:4])), float(f[14]), f) for f in faces]


class FaceRecognizer:
    """SFace (OpenCV Zoo, Apache-2.0): mặt đã căn theo 5 điểm mốc YuNet ➔ embedding 128-D."""

    def __init__(self) -> None:
        self.rec = cv2.FaceRecognizerSF.create(str(SFACE), "")

    def embed(self, bgr: np.ndarray, face: Face) -> np.ndarray:
        return l2n(self.rec.feature(self.rec.alignCrop(bgr, face.raw)))


class ReidModel:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, variant: str = "x0_25", device: str = "auto") -> None:
        import torch

        from identity import osnet

        self.torch = torch
        # GPU Apple (MPS): ~9 ms cho 1–3 người, CPU ~30–70 ms ➔ đủ chạy ReID mỗi frame 30 Hz.
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        net = getattr(osnet, f"osnet_{variant}")(num_classes=1, pretrained=False, loss="softmax")
        state = torch.load(OSNET[variant], map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items() if not k.removeprefix("module.").startswith("classifier.")}
        missing, unexpected = net.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith("classifier.")]
        if bad or unexpected:
            raise RuntimeError(f"OSNet weights không khớp: missing={bad[:5]} unexpected={unexpected[:5]}")
        self.net = net.eval().to(device)

    def embed(self, bgr_crops: list[np.ndarray]) -> np.ndarray:
        """Ảnh cắt người (BGR, bất kỳ cỡ) ➔ (N, 512) đã chuẩn hóa L2."""
        if not bgr_crops:
            return np.zeros((0, 512), dtype=np.float32)
        batch = []
        for c in bgr_crops:
            rgb = cv2.cvtColor(cv2.resize(c, (128, 256), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            batch.append(((rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD).transpose(2, 0, 1))
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.stack(batch)).to(self.device)
            f = self.net(x).cpu().numpy()          # eval mode ➔ trả feature (không qua classifier)
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
