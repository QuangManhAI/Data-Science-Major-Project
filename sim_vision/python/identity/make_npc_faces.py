"""Cắt mặt từ ảnh người thật (fptudsc/face-celeb-vietnamese, Apache-2.0) thành texture dán lên đầu NPC Godot.

Thử nghiệm của docs/UPGRADE_HRI_PLAN.md: "pixel thật" trong sim ➔ model nhận diện mặt học từ người thật dùng
nguyên vẹn được trong sim và ngoài đời. Chỉ dùng nội bộ: ảnh và texture đều nằm trong .gitignore, mapping lưu theo
id ảnh, không lưu tên người.

Chạy: .venv/bin/python -m identity.make_npc_faces
Ra:   sim_vision/godot/assets/textures/npc_faces/<npc>.png (RGB 256x256 vuông, mặt thẳng theo hai mắt, không cắt viền:
      dán nguyên lên mặt trước của đầu hình hộp, npc_face.gd)
"""

from __future__ import annotations

import csv
import json
import sys

import cv2
import numpy as np

from identity.models import PROJECT_ROOT, FaceModel

SRC = PROJECT_ROOT / "data" / "faces_vn"
OUT = PROJECT_ROOT / "sim_vision" / "godot" / "assets" / "textures" / "npc_faces"
W, H = 256, 256
# Vị trí chuẩn của hai mắt và miệng trong texture (tỉ lệ) ➔ khớp chỗ mắt / miệng trên đầu NPC.
EYE_L, EYE_R, MOUTH = (0.34, 0.42), (0.66, 0.42), (0.50, 0.72)
# NPC ➔ person_id trong labels.csv (chọn tay: mặt thẳng, không kính, đúng giới, người lớn tuổi cho NPC cao tuổi).
MAPPING: dict[str, int] = {
    "patient_elder_m03": 173, "patient_elder_m14": 102, "patient_elder_f09": 1,
    "patient_m02": 123, "patient_m05": 186, "human_m06": 43,
    "patient_f02": 222, "patient_f08": 23, "medical_doctor": 149, "medical_nurse": 117,
}


def best_face(fm: FaceModel, files: list[str]):
    best = None
    for fn in files:
        img = cv2.imread(str(SRC / fn))
        if img is None:
            continue
        faces = fm.detect(img)
        if len(faces) != 1:
            continue
        f = faces[0]
        lm = f.raw[4:14].reshape(5, 2)
        eye_d = float(np.linalg.norm(lm[1] - lm[0]))
        yaw = abs((lm[2][0] - (lm[0][0] + lm[1][0]) / 2) / max(eye_d, 1.0))
        q = f.score - 1.5 * yaw + 0.001 * min(f.box[3], 300)
        if best is None or q > best[0]:
            best = (q, fn, img, lm)
    return best


def align(img: np.ndarray, lm: np.ndarray) -> np.ndarray:
    src = np.float32([lm[0], lm[1], (lm[3] + lm[4]) / 2])          # mắt trái, mắt phải, giữa miệng
    dst = np.float32([[EYE_L[0] * W, EYE_L[1] * H], [EYE_R[0] * W, EYE_R[1] * H], [MOUTH[0] * W, MOUTH[1] * H]])
    face = cv2.warpAffine(img, cv2.getAffineTransform(src, dst), (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    return face                       # ảnh vuông nguyên vẹn (tóc, tai, nền quanh mặt giữ nguyên)


def main() -> int:
    rows = list(csv.DictReader(open(SRC / "labels.csv", encoding="utf-8")))
    by_pid: dict[int, list[str]] = {}
    for r in rows:
        by_pid.setdefault(int(r["person_id"]), []).append(r["file"])
    fm = FaceModel(0.8)
    OUT.mkdir(parents=True, exist_ok=True)
    used = {}
    for npc, pid in MAPPING.items():
        b = best_face(fm, by_pid[pid])
        if b is None:
            print(f"{npc}: không tìm được mặt cho person {pid}", file=sys.stderr)
            return 1
        cv2.imwrite(str(OUT / f"{npc}.png"), align(b[2], b[3]))
        used[npc] = {"person_id": pid, "source_file": b[1]}
        print(f"{npc:20s} ⇐ person {pid:3d} ({b[1]})")
    (OUT / "mapping.json").write_text(json.dumps(used, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
