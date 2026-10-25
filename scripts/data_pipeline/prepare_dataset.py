#!/usr/bin/env python3
"""
prepare_dataset.py
Phân chia dữ liệu từ data/vision/raw/ (hoặc in_review/) sang dataset/ theo chuẩn 70% Train - 15% Val - 15% Test.
Đảm bảo:
1. Nhóm theo Case ID (Tránh data leakage giữa các frame cùng 1 góc chụp).
2. Chuẩn hoá nhãn chỉ giữ 3 classes cốt lõi: 0 (person), 1 (wheelchair), 2 (hospital_bed).
3. Đóng băng Test set để phục vụ đánh giá khách quan.
"""

import os
import shutil
import random
import json
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "vision" / "raw"
DATASET_DIR = ROOT_DIR / "data" / "vision" / "dataset"

def split_and_prepare(seed=42):
    random.seed(seed)
    
    raw_images_dir = RAW_DIR / "images"
    raw_labels_dir = RAW_DIR / "labels"
    
    if not raw_images_dir.exists():
        print(f"❌ Không tìm thấy thư mục: {raw_images_dir}")
        return

    # Gom nhóm theo Case ID (ví dụ: U1_1, U2_8, U6_8)
    cases = defaultdict(list)
    image_files = sorted(list(raw_images_dir.glob("*.jpg")))
    
    for img_path in image_files:
        stem = img_path.stem  # e.g., U2_8_0001
        parts = stem.split("_")
        if len(parts) >= 2:
            case_id = f"{parts[0]}_{parts[1]}"
        else:
            case_id = "general"
        cases[case_id].append(img_path)
        
    print(f"📊 Tìm thấy tổng cộng: {len(image_files)} ảnh thuộc {len(cases)} ca kiểm thử lâm sàng.")

    # Phân chia Case-level Split: 70% Train, 15% Val, 15% Test
    case_ids = list(cases.keys())
    random.shuffle(case_ids)
    
    n_total = len(case_ids)
    n_test = max(1, int(n_total * 0.15))
    n_val = max(1, int(n_total * 0.15))
    
    test_cases = set(case_ids[:n_test])
    val_cases = set(case_ids[n_test:n_test + n_val])
    train_cases = set(case_ids[n_test + n_val:])
    
    print(f"🔹 Phân bổ: Train={len(train_cases)} cases, Val={len(val_cases)} cases, Test={len(test_cases)} cases")

    # Dọn dẹp dataset dir trước khi copy
    for split in ["train", "val", "test"]:
        split_img_dir = DATASET_DIR / split / "images"
        split_lbl_dir = DATASET_DIR / split / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        # Clean existing files
        for f in split_img_dir.glob("*.jpg"): f.unlink()
        for f in split_lbl_dir.glob("*.txt"): f.unlink()

    counts = {"train": 0, "val": 0, "test": 0}
    class_counts = defaultdict(int)

    for case_id, img_list in cases.items():
        if case_id in test_cases:
            target_split = "test"
        elif case_id in val_cases:
            target_split = "val"
        else:
            target_split = "train"
            
        for img_path in img_list:
            lbl_path = raw_labels_dir / f"{img_path.stem}.txt"
            
            # Đọc và lọc nhãn: Chỉ giữ class 0, 1, 2
            valid_lines = []
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            if cls_id in [0, 1, 2]:
                                valid_lines.append(line.strip())
                                class_counts[cls_id] += 1
                                
            # Copy ảnh
            dest_img = DATASET_DIR / target_split / "images" / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Ghi nhãn
            dest_lbl = DATASET_DIR / target_split / "labels" / f"{img_path.stem}.txt"
            with open(dest_lbl, "w") as f:
                if valid_lines:
                    f.write("\n".join(valid_lines) + "\n")
                    
            counts[target_split] += 1

    print("\n✅ HOÀN TẤT PHÂN CHIA DATASET:")
    print(f"   🏋️ Train: {counts['train']} frames ({counts['train']/len(image_files)*100:.1f}%)")
    print(f"   🔍 Val:   {counts['val']} frames ({counts['val']/len(image_files)*100:.1f}%)")
    print(f"   🧊 Test:  {counts['test']} frames ({counts['test']/len(image_files)*100:.1f}%)")
    print("\n📦 Phân bổ số lượng nhãn theo 3 Classes:")
    class_names = {0: "person", 1: "wheelchair", 2: "hospital_bed"}
    for cid, cname in class_names.items():
        print(f"   • Class {cid} ({cname}): {class_counts[cid]} instances")

if __name__ == "__main__":
    split_and_prepare()
