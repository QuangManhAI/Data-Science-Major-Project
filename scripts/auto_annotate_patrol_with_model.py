#!/usr/bin/env python3
"""
scripts/auto_annotate_patrol_with_model.py
==========================================
Dùng mô hình YOLO26n v5 (best.pt) để suy luận (AI auto-annotation / pseudo-labeling)
toàn bộ 531 khung hình tuần tra của Robot, sau đó đóng gói chuẩn CVAT YOLO 1.1
để người dùng mở CVAT lên check trực quan xem model bắt thế nào và sửa nhãn.
"""

import os
import shutil
import zipfile
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "experiments/runs/train_hospital/yolo26n_targeted_17class_v5/weights/best.pt"
IMAGES_DIR = REPO_ROOT / "data/vision/dataset/hospital_robot_patrol_testset/images"
OUT_LABELS_DIR = REPO_ROOT / "data/vision/dataset/hospital_robot_patrol_testset/model_v5_labels"
CVAT_PKG_DIR = REPO_ROOT / "data/vision/cvat_robot_patrol_task_package"

CLASS_NAMES = [
    "patient", "doctor", "wheelchair", "queue_kiosk", "reception_counter",
    "sanitizer_stand", "pccc_station", "medical_waste_bin", "water_dispenser",
    "wc_station", "elevator_doors", "hospital_bed", "stretcher_cart",
    "iv_stand", "medical_trolley", "vital_monitor", "hospital_chair"
]

def main():
    print("==================================================================")
    print("🤖 CHẠY MODEL YOLO26n v5 ĐỂ ĐÁNH NHÃN TỰ ĐỘNG BỘ TEST TUẦN TRA ROBOT")
    print(f"📦 Trọng số: {MODEL_PATH}")
    print(f"📸 Thư mục ảnh: {IMAGES_DIR}")
    print("==================================================================")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

    OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(list(IMAGES_DIR.glob("*.jpg")))
    print(f"🔍 Tìm thấy {len(images)} ảnh tuần tra.")

    # 1. Chạy model inference
    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(IMAGES_DIR),
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device="0",
        save=False,
        save_txt=False,
        verbose=False
    )

    total_pred_boxes = 0
    class_pred_counts = {c: 0 for c in CLASS_NAMES}

    for res in results:
        img_name = Path(res.path).stem
        lbl_file = OUT_LABELS_DIR / f"{img_name}.txt"
        lines = []

        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cid = int(b.cls[0].item())
                cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
                # xywh normalized
                xywhn = b.xywhn[0].tolist()
                cx, cy, w, h = xywhn[0], xywhn[1], xywhn[2], xywhn[3]
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                total_pred_boxes += 1
                class_pred_counts[cname] = class_pred_counts.get(cname, 0) + 1

        with open(lbl_file, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

    print(f"\n✅ Đã suy luận xong toàn bộ {len(images)} frames!")
    print(f"📊 Tổng số hộp nhãn model YOLO26n v5 phát hiện: {total_pred_boxes} boxes")
    print("📋 Thống kê chi tiết từng lớp được model đánh nhãn:")
    for c, cnt in class_pred_counts.items():
        if cnt > 0:
            print(f"   • {c:<20}: {cnt} boxes")

    # 2. Đóng gói chuẩn định dạng CVAT YOLO 1.1
    print("\n📦 Đang đóng gói lại file annotations.zip chuẩn CVAT YOLO 1.1...")
    temp_anno_dir = CVAT_PKG_DIR / "_temp_yolo11"
    if temp_anno_dir.exists():
        shutil.rmtree(temp_anno_dir)
    obj_data_dir = temp_anno_dir / "obj_train_data"
    obj_data_dir.mkdir(parents=True, exist_ok=True)

    # obj.names
    with open(temp_anno_dir / "obj.names", "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")

    # obj.data
    with open(temp_anno_dir / "obj.data", "w", encoding="utf-8") as f:
        f.write(f"classes = {len(CLASS_NAMES)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")

    # train.txt & copy labels
    train_lines = []
    for img in images:
        lbl_name = f"{img.stem}.txt"
        lbl_src = OUT_LABELS_DIR / lbl_name
        dest_lbl = obj_data_dir / lbl_name
        train_lines.append(f"data/obj_train_data/{img.name}")
        if lbl_src.exists():
            shutil.copy2(lbl_src, dest_lbl)
        else:
            dest_lbl.touch()

    with open(temp_anno_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    # Nén annotations.zip
    annotations_zip_path = CVAT_PKG_DIR / "annotations.zip"
    if annotations_zip_path.exists():
        annotations_zip_path.unlink()

    with zipfile.ZipFile(annotations_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_ann:
        zip_ann.write(temp_anno_dir / "obj.names", arcname="obj.names")
        zip_ann.write(temp_anno_dir / "obj.data", arcname="obj.data")
        zip_ann.write(temp_anno_dir / "train.txt", arcname="train.txt")
        for lbl in obj_data_dir.glob("*.txt"):
            zip_ann.write(lbl, arcname=f"obj_train_data/{lbl.name}")

    shutil.rmtree(temp_anno_dir)
    print(f"🎉 Hoàn tất đóng gói nhãn do model v5 sinh ra: {annotations_zip_path} ({annotations_zip_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
