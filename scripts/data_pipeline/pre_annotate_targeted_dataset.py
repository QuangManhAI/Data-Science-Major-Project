#!/usr/bin/env python3
"""
pre_annotate_targeted_dataset.py
================================
Tự động đánh Ground Truth / Pre-Annotate bộ dữ liệu mô phỏng mục tiêu:
- Hợp nhất tọa độ hình học 3D từ Godot (Raycast/Mesh Projection) và mô hình YOLO26n mới nhất.
- Hỗ trợ cả đợt 1 (400 frames) và đợt 2 (250 frames) trước khi xuất package lên CVAT cho người dùng duyệt.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]

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
    parser = argparse.ArgumentParser(description="Pre-annotate staging dataset with YOLO & 3D raycast ensemble")
    parser.add_argument(
        "--staging-dir",
        type=str,
        default=str(REPO_ROOT / "data/vision/dataset/hospital_targeted_staging_r2"),
        help="Path to staging directory containing images/ and labels/"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(REPO_ROOT / "experiments/runs/train_hospital/yolo26n_targeted_17class_v4/weights/best.pt"),
        help="Path to YOLO best.pt weights"
    )
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    args = parser.parse_args()

    staging_dir = Path(args.staging_dir)
    images_dir = staging_dir / "images"
    labels_dir = staging_dir / "labels"
    weights_path = Path(args.weights)

    if not images_dir.exists():
        print(f"❌ Không tìm thấy thư mục ảnh: {images_dir}")
        sys.exit(1)

    if not weights_path.exists():
        print(f"❌ Không tìm thấy file weights: {weights_path}")
        sys.exit(1)

    print("==================================================================")
    print("🤖 BẮT ĐẦU MODEL PRE-ANNOTATION (ENSEMBLE 3D RAYCAST + YOLO)")
    print(f"• Thư mục ảnh:   {images_dir}")
    print(f"• Thư mục nhãn:  {labels_dir}")
    print(f"• Model weights: {weights_path.name}")
    print("==================================================================")

    model = YOLO(str(weights_path))
    device = "0" if torch.cuda.is_available() else "cpu"

    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"📸 Tìm thấy {len(image_files)} ảnh cần đánh nhãn...")

    total_godot_boxes = 0
    total_model_boxes = 0
    total_merged_boxes = 0
    class_counter = {cid: 0 for cid in UNIFIED_CLASSES.keys()}

    for img_path in image_files:
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"

        # 1. Đọc nhãn 3D sẵn có từ Godot
        godot_boxes = []
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
            imgsz=args.imgsz,
            conf=args.conf,
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

        # 3. Hợp nhất thông minh
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

            if best_idx != -1 and best_iou >= 0.30:
                matched_model_indices.add(best_idx)
                m_cid, m_coords, m_conf = model_boxes[best_idx]
                if m_conf >= 0.45:
                    merged.append((g_cid, m_coords))
                else:
                    merged.append((g_cid, g_coords))
            else:
                merged.append((g_cid, g_coords))

        for m_idx, (m_cid, m_coords, m_conf) in enumerate(model_boxes):
            if m_idx not in matched_model_indices:
                if m_conf >= 0.40:
                    merged.append((m_cid, m_coords))

        # 4. Ghi lại file nhãn
        with open(lbl_path, "w", encoding="utf-8") as f:
            for cid, coords in merged:
                cx, cy, w, h = coords
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                class_counter[cid] += 1
                total_merged_boxes += 1

    print("\n==================================================================")
    print("✅ PRE-ANNOTATION HOÀN TẤT THÀNH CÔNG!")
    print(f"• Số hộp ban đầu từ Godot 3D:   {total_godot_boxes:,}")
    print(f"• Số hộp phát hiện bởi Model:   {total_model_boxes:,}")
    print(f"• Tổng số hộp sau hợp nhất:     {total_merged_boxes:,}")
    print("------------------------------------------------------------------")
    print("📊 Phân bổ nhãn 17 classes đã gán:")
    for cid in sorted(UNIFIED_CLASSES.keys()):
        cname = UNIFIED_CLASSES[cid]
        cnt = class_counter[cid]
        star = "⭐" if cname in ["wheelchair", "iv_stand"] else "  "
        print(f"  {star} Class {cid:02d} ({cname:<18}): {cnt:>4} boxes")
    print("==================================================================")


if __name__ == "__main__":
    main()
