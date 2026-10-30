#!/usr/bin/env python3
"""
pre_annotate_expanded_dataset.py
Tự động đánh Ground Truth / Pre-Annotate bộ dữ liệu mô phỏng mở rộng (600 frames):
- Kết hợp tọa độ hình học 3D từ Godot (Raycast/Mesh Projection) và mô hình best.pt (YOLO26n-Verified, Precision 93.63%).
- Tinh chỉnh và hợp nhất nhãn nhằm mang lại chất lượng nhãn cao nhất cho 17 classes trước khi đưa lên CVAT.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = REPO_ROOT / "data/vision/dataset/hospital_expanded_staging"
IMAGES_DIR = STAGING_DIR / "images"
LABELS_DIR = STAGING_DIR / "labels"
MODEL_WEIGHTS = REPO_ROOT / "experiments/runs/train_hospital/yolo26n_verified_17class_v1/weights/best.pt"

UNIFIED_CLASSES = {
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
    16: "hospital_chair"
}


def compute_iou(box1, box2):
    """Tính IoU giữa 2 box định dạng [cx, cy, w, h] (normalized)."""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def main():
    if not IMAGES_DIR.exists():
        print(f"❌ Không tìm thấy thư mục ảnh: {IMAGES_DIR}")
        sys.exit(1)

    if not MODEL_WEIGHTS.exists():
        print(f"❌ Không tìm thấy file weights: {MODEL_WEIGHTS}")
        sys.exit(1)

    print("==================================================================")
    print("🤖 BẮT ĐẦU QUY TRÌNH MODEL PRE-ANNOTATION & ENSEMBLE CHO 600 FRAMES")
    print(f"• Thư mục ảnh:   {IMAGES_DIR}")
    print(f"• Thư mục nhãn:  {LABELS_DIR}")
    print(f"• Model weights: {MODEL_WEIGHTS}")
    print("==================================================================")

    model = YOLO(str(MODEL_WEIGHTS))
    device = "0" if torch.cuda.is_available() else "cpu"

    image_files = sorted(list(IMAGES_DIR.glob("*.jpg")))
    print(f"📸 Tìm thấy {len(image_files)} ảnh cần đánh nhãn...")

    total_godot_boxes = 0
    total_model_boxes = 0
    total_merged_boxes = 0
    class_counter = {cid: 0 for cid in UNIFIED_CLASSES.keys()}

    for img_path in image_files:
        stem = img_path.stem
        lbl_path = LABELS_DIR / f"{stem}.txt"

        # 1. Đọc nhãn 3D sẵn có từ Godot
        godot_boxes = []  # list of (cid, [cx, cy, w, h])
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        coords = [float(p) for p in parts[1:5]]
                        godot_boxes.append((cid, coords))
                        total_godot_boxes += 1

        # 2. Suy luận bằng YOLO model
        results = model.predict(
            source=str(img_path),
            imgsz=640,
            conf=0.25,
            iou=0.45,
            device=device,
            verbose=False
        )

        model_boxes = []
        if results and len(results) > 0:
            res = results[0]
            boxes_obj = res.boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                xywhn = boxes_obj.xywhn.cpu().numpy()
                cls_arr = boxes_obj.cls.cpu().numpy().astype(int)
                conf_arr = boxes_obj.conf.cpu().numpy()
                for i in range(len(cls_arr)):
                    cid = int(cls_arr[i])
                    conf = float(conf_arr[i])
                    coords = xywhn[i].tolist()
                    model_boxes.append((cid, coords, conf))
                    total_model_boxes += 1

        # 3. Hợp nhất thông minh giữa Godot 3D Ground Truth và Model Predictions
        merged = []
        matched_model_indices = set()

        for g_cid, g_coords in godot_boxes:
            best_iou = 0.0
            best_idx = -1
            for m_idx, (m_cid, m_coords, m_conf) in enumerate(model_boxes):
                if g_cid == m_cid:
                    iou = compute_iou(g_coords, m_coords)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = m_idx

            if best_idx != -1 and best_iou >= 0.35:
                # Trùng khớp: Lấy box của model nếu conf cao vì biên dạng model fit sát người và vật hơn
                matched_model_indices.add(best_idx)
                m_cid, m_coords, m_conf = model_boxes[best_idx]
                if m_conf >= 0.50:
                    merged.append((g_cid, m_coords))
                else:
                    merged.append((g_cid, g_coords))
            else:
                # Không model nào bắt được nhưng Godot 3D đã tính toán: GIỮ LẠI (Ground Truth quan trọng cho cọc truyền/cáng)
                merged.append((g_cid, g_coords))

        # Thêm các vật thể model phát hiện được nhưng Godot chưa có (ví dụ NPC hoặc góc khuất)
        for m_idx, (m_cid, m_coords, m_conf) in enumerate(model_boxes):
            if m_idx not in matched_model_indices:
                if m_conf >= 0.45:
                    merged.append((m_cid, m_coords))

        # 4. Ghi lại file nhãn đã chuẩn hóa
        with open(lbl_path, "w", encoding="utf-8") as f:
            for cid, coords in merged:
                cx, cy, w, h = coords
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                class_counter[cid] += 1
                total_merged_boxes += 1

    print("\n==================================================================")
    print("✅ HOÀN TẤT PRE-ANNOTATION THÀNH CÔNG!")
    print(f"📊 Số hộp gốc từ Godot 3D:    {total_godot_boxes}")
    print(f"📊 Số hộp phát hiện từ Model:  {total_model_boxes}")
    print(f"📊 Tổng số hộp sau Hợp Nhất:   {total_merged_boxes}")
    print("------------------------------------------------------------------")
    print("📈 Thống kê số lượng hộp chi tiết 17 classes sau khi làm giàu:")
    for cid in sorted(class_counter.keys()):
        cname = UNIFIED_CLASSES[cid]
        cnt = class_counter[cid]
        print(f"  • Class {cid:02d} ({cname:<18}): {cnt:>4} boxes")
    print("==================================================================")


if __name__ == "__main__":
    main()
