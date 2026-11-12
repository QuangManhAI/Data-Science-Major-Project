"""So nhận diện bằng MẶT (YuNet + SFace) với ReID TOÀN THÂN (OSNet) trên ảnh camera robot trong Godot.

Thí nghiệm "dán mặt người thật lên NPC": hai bộ ảnh cùng cảnh / góc / khoảng cách (tools/capture_identity.gd),
một bộ mặt Rocketbox gốc, một bộ dán ảnh mặt người thật (--faces). Ghi nhớ từ 1 frame (đứng, nhìn thẳng, 2 m);
đúng = giống người đã ghi nhớ hơn 9 người còn lại. SFace chỉ để đo, robot không dùng khi chạy.

Chạy: .venv/bin/python -m identity.eval_faces <bộ_gốc> <bộ_mặt_dán> [--out ../reports/face_paste_eval.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from identity.eval_capture import ENROLL_DIST_M, PERSON_CLASSES, YOLO_CONF, YOLO_PATH, pick_box, view_group
from identity.models import PROJECT_ROOT, FaceModel, FaceRecognizer, ReidModel

SFACE_THR = 0.363     # ngưỡng cosine chuẩn của SFace (OpenCV Zoo)


def extract(cap: Path) -> pd.DataFrame:
    from ultralytics import YOLO

    cache = cap / "faces_features.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frames = [json.loads(l) for l in (cap / "frames.jsonl").read_text(encoding="utf-8").splitlines() if l]
    yolo, fm, fr, reid = YOLO(str(YOLO_PATH)), FaceModel(0.6), FaceRecognizer(), ReidModel("x0_25")
    rows = []
    for fr_ in frames:
        img = cv2.imread(str(cap / fr_["file"]))
        res = yolo.predict(img, conf=YOLO_CONF, classes=list(PERSON_CLASSES), verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        b = pick_box(boxes, tuple(fr_["center"]))
        row = {**fr_, "detected": b is not None, "face": False, "face_px": 0.0}
        if b is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            row["body"] = reid.embed([img[max(y1, 0):y2, max(x1, 0):x2]])[0]
            faces = [f for f in fm.detect(img)
                     if x1 <= f.box[0] + f.box[2] / 2 <= x2 and y1 <= f.box[1] + f.box[3] / 2 <= y1 + 0.5 * (y2 - y1)]
            if faces:
                f = max(faces, key=lambda f: f.score)
                row.update(face=True, face_px=f.box[3], face_emb=fr.embed(img, f))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_pickle(cache)
    return df


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    enr = df[(df.pose == "idle") & (df.view_deg == 0) & (df.distance_m == ENROLL_DIST_M) & df.detected]
    ids = sorted(enr.identity)
    g_body = np.stack([enr.set_index("identity").loc[i, "body"] for i in ids])
    has_face = {i for i, f in zip(enr.identity, enr.face) if f}
    g_face = {i: enr.set_index("identity").loc[i, "face_emb"] for i in has_face}
    p = df[df.detected & ~df.file.isin(enr.file)].copy()
    sims = np.stack(p.body.to_numpy()) @ g_body.T
    p["ok_body"] = [ids[j] == t for j, t in zip(sims.argmax(1), p.identity)]
    ok_face, face_used = [], []
    for _, r in p.iterrows():
        if r["face"] and g_face and r["identity"] in g_face:
            s = {i: float(r["face_emb"] @ e) for i, e in g_face.items()}
            best = max(s, key=s.get)
            face_used.append(True)
            ok_face.append(best == r["identity"] and s[best] >= SFACE_THR)
        else:
            face_used.append(False)
            ok_face.append(False)
    p["face_used"] = face_used
    p["ok_face"] = ok_face
    p["view"] = p.view_deg.map(view_group)
    return p, len(has_face)


def section(name: str, df: pd.DataFrame, p: pd.DataFrame, n_enrolled: int) -> list[str]:
    out = [f"## {name}", "", f"Ghi nhớ được mặt: {n_enrolled}/10 người (frame nhìn thẳng 2 m).", "",
           "| Khoảng cách (m) | YuNet thấy mặt (nhìn trước) | Nhận đúng bằng mặt (nhìn trước) | Nhận đúng bằng toàn thân (mọi hướng) |",
           "|---|---|---|---|"]
    for d, g in p.groupby("distance_m"):
        fr = g[g.view_deg.abs() <= 45]
        out.append(f"| {d:.1f} | {fr.face.mean():.0%} | {fr.ok_face.mean():.0%} | {g.ok_body.mean():.0%} |")
    out += ["", "| Hướng | Thấy mặt | Nhận đúng bằng mặt | Nhận đúng bằng toàn thân |", "|---|---|---|---|"]
    for v, g in p.groupby("view"):
        out.append(f"| {v} | {g.face.mean():.0%} | {g.ok_face.mean():.0%} | {g.ok_body.mean():.0%} |")
    esc = p[(p.pose == "walk") & (p.view_deg.abs() >= 135) & p.distance_m.between(1.0, 3.0)]
    talk = p[(p.view_deg.abs() <= 45) & p.distance_m.between(1.0, 3.0)]
    out += ["", "| Tình huống | Bằng mặt | Bằng toàn thân |", "|---|---|---|",
            f"| Nói chuyện (nhìn trước, 1–3 m) | {talk.ok_face.mean():.0%} | {talk.ok_body.mean():.0%} |",
            f"| Dẫn đường (quay lưng, đang đi, 1–3 m) | {esc.ok_face.mean():.0%} | {esc.ok_body.mean():.0%} |", ""]
    return out


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser()
    a.add_argument("base", type=Path)
    a.add_argument("pasted", type=Path)
    a.add_argument("--out", type=Path, default=PROJECT_ROOT / "sim_vision" / "reports" / "face_paste_eval.md")
    args = a.parse_args(sys.argv[1:] if argv is None else argv)
    lines = ["# Dán mặt người thật lên NPC: nhận diện bằng mặt vs toàn thân", "",
             "Camera robot (1.2 m, 640×480), hành lang tầng 1, 10 NPC, 1–6 m, 8 hướng, đứng / đang đi.",
             "Mặt: YuNet + SFace (ngưỡng 0.363). Toàn thân: OSNet x0.25. Box người: YOLO của robot.", ""]
    for name, cap in [("Mặt Rocketbox gốc", args.base), ("Dán ảnh mặt người thật", args.pasted)]:
        df = extract(cap)
        p, n = evaluate(df)
        lines += section(name, df, p, n)
    md = "\n".join(lines)
    args.out.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
