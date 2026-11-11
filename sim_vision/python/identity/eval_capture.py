"""Đo nhận diện đúng người trên ảnh camera robot trong Godot (docs/UPGRADE_HRI_PLAN.md mục D2).

Mô phỏng đúng cách robot dùng lúc chạy, không có cơ sở dữ liệu người nào:
  1. Ghi nhớ (enroll): MỘT frame lúc bệnh nhân quay lại nói chuyện (đứng, nhìn thẳng, ENROLL_DIST_M) ➔
     box người từ YOLO của robot (yolo26n_hospital_robot_v6, lớp patient/doctor) ➔ embedding mặt + toàn thân.
  2. Bám theo: mọi frame khác (quay lưng, nghiêng, đang đi, xa 1–6 m) ➔ box ➔ embedding ➔ so với người đã ghi nhớ.
     Đúng = giống người đã ghi nhớ hơn mọi người khác đứng cạnh (9 NPC còn lại làm "người lạ").
Model đều pretrained trên người thật (YuNet, OSNet x0.25 MSMT17), không fine-tune trên ảnh Godot.
(Bản đầu còn so SFace và OSNet x1.0: không chính xác hơn ➔ đã bỏ.)

Ảnh chụp bằng: godot --path sim_vision/godot --resolution 640x480 -s tools/capture_identity.gd -- <dir>
Chạy: .venv/bin/python -m identity.eval_capture <dir> [--out reports/identity_eval.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from identity.models import PROJECT_ROOT, FaceModel, ReidModel

YOLO_PATH = PROJECT_ROOT / "models" / "yolo26n_hospital_robot_v6_best" / "yolo26n_hospital_robot_best_map92.6.pt"
PERSON_CLASSES = (0, 1)          # patient, doctor (classes_17.json)
ENROLL_DIST_M = 2.0              # ~ PATIENT_ENGAGE_DIST_M 1.85 m: lúc robot tới nói chuyện
YOLO_CONF = 0.25


def view_group(v: float) -> str:
    a = abs(v)
    return "trước (0–45°)" if a <= 45 else ("ngang (90°)" if a <= 90 else "sau (135–180°)")


def pick_box(boxes: np.ndarray, center: tuple[float, float]) -> np.ndarray | None:
    """Box người chứa tâm người (chiếu từ Godot); nhiều box ➔ box có tâm gần nhất."""
    cx, cy = center
    inside = [b for b in boxes if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
    if not inside:
        return None
    return min(inside, key=lambda b: (0.5 * (b[0] + b[2]) - cx) ** 2 + (0.5 * (b[1] + b[3]) - cy) ** 2)


def extract(frames: list[dict], img_dir: Path, reid: dict[str, ReidModel], face: FaceModel) -> pd.DataFrame:
    from ultralytics import YOLO

    yolo = YOLO(str(YOLO_PATH))
    rows = []
    for i, fr in enumerate(frames, 1):
        img = cv2.imread(str(img_dir / fr["file"]))
        res = yolo.predict(img, conf=YOLO_CONF, classes=list(PERSON_CLASSES), verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        b = pick_box(boxes, tuple(fr["center"]))
        row = {**fr, "detected": b is not None, "box_h": float(b[3] - b[1]) if b is not None else 0.0,
               "face": False, "face_px": 0.0}
        if b is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            crop = img[max(y1, 0):y2, max(x1, 0):x2]
            for name, m in reid.items():
                row[f"body_{name}"] = m.embed([crop])[0]
            # Mặt: tìm trong ảnh đầy đủ, chỉ nhận mặt nằm trong nửa trên box người này
            faces = [f for f in face.detect(img)
                     if x1 <= f.box[0] + f.box[2] / 2 <= x2 and y1 <= f.box[1] + f.box[3] / 2 <= y1 + 0.5 * (y2 - y1)]
            if faces:
                f = max(faces, key=lambda f: f.score)
                row.update(face=True, face_px=f.box[3])
        rows.append(row)
        if i % 100 == 0:
            print(f"  {i}/{len(frames)}", flush=True)
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, reid_names: list[str], enroll_views: tuple[float, ...] = (0.0,)) -> pd.DataFrame:
    """Mỗi frame thử (đã có box) ➔ đúng người không, theo từng cách so khớp.
    enroll_views: góc của các frame ghi nhớ ở ENROLL_DIST_M (0 = lúc nói chuyện; 180 = lúc vừa quay đi theo robot).
    Điểm giống một người = max qua các embedding đã ghi nhớ của người đó."""
    enroll = df[(df.pose == "idle") & df.view_deg.isin(enroll_views) & (df.distance_m == ENROLL_DIST_M) & df.detected]
    ids = sorted(set(enroll.identity))
    probes = df[df.detected & ~df.file.isin(enroll.file)].copy()
    owner = np.array([ids.index(i) for i in enroll.identity])
    for n in reid_names:
        all_sims = np.stack(probes[f"body_{n}"].to_numpy()) @ np.stack(enroll[f"body_{n}"].to_numpy()).T
        sims = np.stack([all_sims[:, owner == k].max(1) for k in range(len(ids))], axis=1)
        probes[f"ok_{n}"] = [ids[j] == t for j, t in zip(sims.argmax(1), probes.identity)]
        true_idx = [ids.index(t) for t in probes.identity]
        own = sims[np.arange(len(probes)), true_idx]
        other = np.where(np.eye(len(ids), dtype=bool)[true_idx], -np.inf, sims).max(1)
        probes[f"margin_{n}"] = own - other
        probes[f"own_{n}"] = own
        probes[f"other_{n}"] = other
        probes[f"pred_{n}"] = [ids[j] for j in sims.argmax(1)]
    return probes


def render(df: pd.DataFrame, probes: pd.DataFrame, reid_names: list[str], enroll_note: str = "1 frame / người (đứng, nhìn thẳng") -> str:
    cols = [f"ok_{n}" for n in reid_names]
    names = {f"ok_{n}": f"Toàn thân OSNet {n}" for n in reid_names}
    lines = ["# Đo nhận diện đúng người trên camera robot (Godot)", "",
             f"Ghi nhớ từ {enroll_note}, {ENROLL_DIST_M} m). 10 người, "
             f"{len(df)} frame, {len(probes)} frame thử. Đúng = giống người đã ghi nhớ hơn 9 người còn lại.", "",
             "## Phát hiện (YOLO của robot, lớp patient/doctor) và mặt (YuNet) theo khoảng cách", "",
             "| Khoảng cách (m) | YOLO thấy người | Thấy mặt (khi nhìn trước) | Cỡ mặt (px) |", "|---|---|---|---|"]
    for d, g in df.groupby("distance_m"):
        front = g[g.view_deg.abs() <= 45]
        fp = front[front.face].face_px
        lines.append(f"| {d:.1f} | {g.detected.mean():.0%} | {front.face.mean():.0%} | {fp.median():.0f} |" if len(fp)
                     else f"| {d:.1f} | {g.detected.mean():.0%} | {front.face.mean():.0%} | – |")
    for title, key in [("theo khoảng cách", "distance_m"), ("theo hướng nhìn", "view")]:
        p = probes.assign(view=probes.view_deg.map(view_group))
        lines += ["", f"## Tỉ lệ bám đúng người {title}", "",
                  "| " + {"distance_m": "Khoảng cách (m)", "view": "Hướng"}[key] + " | " + " | ".join(names[c] for c in cols) + " | n |",
                  "|---" * (len(cols) + 2) + "|"]
        for k, g in p.groupby(key):
            lab = f"{k:.1f}" if key == "distance_m" else k
            lines.append(f"| {lab} | " + " | ".join(f"{g[c].mean():.0%}" for c in cols) + f" | {len(g)} |")
    esc = probes[(probes.pose == "walk") & (probes.view_deg.abs() >= 135) & probes.distance_m.between(1.0, 3.0)]
    lines += ["", "## Tình huống dẫn đường (bệnh nhân đi phía trước, quay lưng, 1–3 m)", "",
              "| Cách | Bám đúng |", "|---|---|"] + [f"| {names[c]} | {esc[c].mean():.0%} (n={len(esc)}) |" for c in cols]
    lines += ["", "## Ngưỡng chấp nhận (lúc chạy: không box nào vượt ngưỡng ➔ coi như mất dấu, không bám người lạ)", "",
              "Ngưỡng = phân vị 99% điểm giống của người lạ giống nhất ➔ ≤ 1% frame nhận nhầm người lạ.", "",
              "| Model | Ngưỡng cosine | Nhận đúng người (≥ ngưỡng) | Nhận đúng, lúc quay lưng 1–3 m |", "|---|---|---|---|"]
    for n in reid_names:
        thr = float(np.quantile(probes[f"other_{n}"], 0.99))
        lines.append(f"| OSNet {n} | {thr:.2f} | {(probes[f'own_{n}'] >= thr).mean():.0%} | {(esc[f'own_{n}'] >= thr).mean():.0%} |")
    lines += ["", "## Theo từng người (toàn thân)", "", "| Người | " + " | ".join(f"OSNet {n}" for n in reid_names)
              + " | Hay nhầm với (x0_25) |", "|---" * (len(reid_names) + 2) + "|"]
    for k, g in probes.groupby("identity"):
        wrong = g[~g[f"ok_{reid_names[0]}"]][f"pred_{reid_names[0]}"]
        conf = wrong.value_counts().index[0] if len(wrong) else "–"
        lines.append(f"| {k} | " + " | ".join(f"{g[f'ok_{n}'].mean():.0%}" for n in reid_names) + f" | {conf} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture_dir", type=Path)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "sim_vision" / "reports" / "identity_eval.md")
    a = p.parse_args(sys.argv[1:] if argv is None else argv)
    frames = [json.loads(l) for l in (a.capture_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines() if l]
    reid_names = ["x0_25"]
    cache = a.capture_dir / "features.pkl"
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        df = extract(frames, a.capture_dir, {n: ReidModel(n) for n in reid_names}, FaceModel())
        df.to_pickle(cache)
    md = render(df, evaluate(df, reid_names, (0.0,)), reid_names, "1 frame / người (lúc nói chuyện: đứng, nhìn thẳng")
    md += "\n---\n\n" + render(df, evaluate(df, reid_names, (0.0, 180.0)), reid_names,
                                 "2 frame / người (lúc nói chuyện nhìn thẳng + lúc vừa quay lưng đi theo robot")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
