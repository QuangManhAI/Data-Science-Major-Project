"""Tải model nhận diện người vào models/identity/ (đã tải thì bỏ qua). Chạy: python -m identity.download_models"""

from __future__ import annotations

import sys
import urllib.request

from identity.models import MODEL_DIR, OSNET, SFACE, YUNET

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
_HF = "https://huggingface.co/kaiyangzhou/osnet/resolve/main"
_MSMT = "msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
FILES = {
    YUNET: f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    OSNET["x0_25"]: f"{_HF}/osnet_x0_25_{_MSMT}",
}


OPTIONAL = {SFACE: f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx"}


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    files = dict(FILES)
    if "--sface" in sys.argv[1:]:
        files.update(OPTIONAL)
    for path, url in files.items():
        if path.exists() and path.stat().st_size > 100_000:
            print(f"có sẵn  {path.name}")
            continue
        print(f"tải     {path.name}")
        urllib.request.urlretrieve(url, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
