#!/usr/bin/env python3
"""
scripts/cvat/generate_model_predictions_cvat_pkg.py
===================================================
Dùng model YOLO26n v6 (best.pt) để suy luận trên 660 ảnh Active Inspection
và đóng gói thành `model_v6_predictions.zip` chuẩn CVAT YOLO 1.1 format.
"""

import shutil
import zipfile
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = REPO_ROOT / "experiments/runs/train_hospital/yolo26n_golden_17class_v6/weights/best.pt"
IMAGES_DIR = REPO_ROOT / "data/vision/dataset/hospital_robot_inspection_testset/images"
OUT_PKG_DIR = REPO_ROOT / "data/vision/cvat_robot_inspection_package"

CLASS_NAMES = [
    "patient", "doctor", "wheelchair", "queue_kiosk", "reception_counter",
    "sanitizer_stand", "pccc_station", "medical_waste_bin", "water_dispenser",
    "wc_station", "elevator_doors", "hospital_bed", "stretcher_cart",
    "iv_stand", "medical_trolley", "vital_monitor", "hospital_chair"
]

def main():
    print("==================================================================")
    print("🤖 AI PRE-ANNOTATION CHO BỘ ACTIVE INSPECTION (660 FRAMES) BẰNG YOLO26n v6")
    print("==================================================================")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(IMAGES_DIR),
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device="0",
        verbose=False
    )

    temp_dir = OUT_PKG_DIR / "_temp_model_v6"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    obj_data_dir = temp_dir / "obj_train_data"
    obj_data_dir.mkdir(parents=True, exist_ok=True)

    with open(temp_dir / "obj.names", "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")

    with open(temp_dir / "obj.data", "w", encoding="utf-8") as f:
        f.write(f"classes = {len(CLASS_NAMES)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")

    images = sorted(list(IMAGES_DIR.glob("*.jpg")))
    train_lines = []

    pred_counts = {c: 0 for c in CLASS_NAMES}
    total_boxes = 0

    for res in results:
        img_p = Path(res.path)
        lbl_file = obj_data_dir / f"{img_p.stem}.txt"
        train_lines.append(f"data/obj_train_data/{img_p.name}")
        lines = []

        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cid = int(b.cls[0].item())
                cname = CLASS_NAMES[cid]
                xywhn = b.xywhn[0].tolist()
                lines.append(f"{cid} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
                pred_counts[cname] += 1
                total_boxes += 1

        with open(lbl_file, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

    with open(temp_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    zip_out = OUT_PKG_DIR / "model_v6_predictions.zip"
    if zip_out.exists():
        zip_out.unlink()

    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(temp_dir / "obj.names", arcname="obj.names")
        z.write(temp_dir / "obj.data", arcname="obj.data")
        z.write(temp_dir / "train.txt", arcname="train.txt")
        for f in obj_data_dir.glob("*.txt"):
            z.write(f, arcname=f"obj_train_data/{f.name}")

    shutil.rmtree(temp_dir)
    print(f"✅ Đã tạo gói suy luận mô hình: {zip_out} ({zip_out.stat().st_size / 1024:.1f} KB)")
    print(f"📊 Tổng số boxes do YOLO26n v6 dự đoán: {total_boxes} boxes")

if __name__ == "__main__":
    main()
