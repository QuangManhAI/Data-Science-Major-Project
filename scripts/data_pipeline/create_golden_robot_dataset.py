#!/usr/bin/env python3
"""
scripts/data_pipeline/create_golden_robot_dataset.py
====================================================
Tái cấu trúc Dataset chuẩn vàng (Golden Architecture) cho Hệ Thống Thị Giác Robot:
1. Gộp toàn bộ 4,370 frames tĩnh (3,277 train + 625 val + 468 test cũ) vào pool dữ liệu học.
2. Phân chia chuẩn mực: 85% Train (3,714 frames) và 15% Val (656 frames).
3. Đặt bộ dữ liệu hành trình Robot tuần tra (531 frames, 2,223 nhãn đã duyệt tay từ CVAT)
   làm BỘ TEST VÀNG ĐỘC LẬP (Golden Test Set).
4. KIỂM DUYỆT RÒ RỈ DỮ LIỆU TUYỆT ĐỐI (Zero Data Leakage Audit qua SHA-256):
   - Đảm bảo 0% trùng lặp ảnh giữa Train/Val và Test Robot.
5. Tạo cấu hình: configs/hospital_unified_golden_17class.yaml
"""

import hashlib
import json
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_UNIFIED_DIR = REPO_ROOT / "data/vision/dataset/hospital_unified_targeted_4370_17class"
SRC_PATROL_DIR = REPO_ROOT / "data/vision/dataset/hospital_robot_patrol_testset"
DEST_GOLDEN_DIR = REPO_ROOT / "data/vision/dataset/hospital_unified_golden_4901_17class"
CONFIG_FILE = REPO_ROOT / "configs/hospital_unified_golden_17class.yaml"

CLASS_NAMES = [
    "patient", "doctor", "wheelchair", "queue_kiosk", "reception_counter",
    "sanitizer_stand", "pccc_station", "medical_waste_bin", "water_dispenser",
    "wc_station", "elevator_doors", "hospital_bed", "stretcher_cart",
    "iv_stand", "medical_trolley", "vital_monitor", "hospital_chair"
]


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("==================================================================")
    print("🚀 BẮT ĐẦU TÁI CẤU TRÚC DATASET CHUẨN VÀNG (GOLDEN ROBOT ARCHITECTURE)")
    print(f"• Nguồn dữ liệu tĩnh:   {SRC_UNIFIED_DIR.name} (4,370 frames)")
    print(f"• Nguồn Testset Robot:  {SRC_PATROL_DIR.name} (531 frames CVAT-verified)")
    print(f"• Thư mục đích:         {DEST_GOLDEN_DIR.name}")
    print("==================================================================")

    random.seed(42)

    # 1. Thu thập toàn bộ 4,370 frames tĩnh
    all_static_pairs = []
    for split in ["train", "val", "test"]:
        img_dir = SRC_UNIFIED_DIR / split / "images"
        lbl_dir = SRC_UNIFIED_DIR / split / "labels"
        for img in img_dir.glob("*.*"):
            lbl = lbl_dir / f"{img.stem}.txt"
            all_static_pairs.append((img, lbl))

    print(f"\n📦 1. Đã thu thập {len(all_static_pairs)} cặp (ảnh, nhãn) tĩnh từ dataset cũ.")
    assert len(all_static_pairs) == 4370, f"Lỗi: Kỳ vọng 4370 frames, nhưng tìm thấy {len(all_static_pairs)}"

    # 2. Xáo trộn ngẫu nhiên và phân chia 85% Train - 15% Val
    random.shuffle(all_static_pairs)
    val_count = int(len(all_static_pairs) * 0.15)  # 655 frames
    train_count = len(all_static_pairs) - val_count  # 3,715 frames

    val_pairs = all_static_pairs[:val_count]
    train_pairs = all_static_pairs[val_count:]

    print(f"  • Phân chia vào Train: {len(train_pairs)} frames (85.0%)")
    print(f"  • Phân chia vào Val:   {len(val_pairs)} frames (15.0%)")

    # 3. Tạo thư mục đích
    if DEST_GOLDEN_DIR.exists():
        print(f"🧹 Dọn dẹp thư mục cũ {DEST_GOLDEN_DIR.name}...")
        shutil.rmtree(DEST_GOLDEN_DIR)

    for split in ["train", "val", "test"]:
        (DEST_GOLDEN_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DEST_GOLDEN_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Sao chép Train
    print("\n📂 2. Đang tạo tập Train (3,715 frames)...")
    for img, lbl in train_pairs:
        shutil.copy2(img, DEST_GOLDEN_DIR / "train/images" / img.name)
        if lbl.exists():
            shutil.copy2(lbl, DEST_GOLDEN_DIR / "train/labels" / lbl.name)
        else:
            (DEST_GOLDEN_DIR / "train/labels" / f"{img.stem}.txt").touch()

    # Sao chép Val
    print("📂 3. Đang tạo tập Val (655 frames)...")
    for img, lbl in val_pairs:
        shutil.copy2(img, DEST_GOLDEN_DIR / "val/images" / img.name)
        if lbl.exists():
            shutil.copy2(lbl, DEST_GOLDEN_DIR / "val/labels" / lbl.name)
        else:
            (DEST_GOLDEN_DIR / "val/labels" / f"{img.stem}.txt").touch()

    # Sao chép Test (Trỏ 100% sang Robot Patrol đã duyệt tay)
    print("📂 4. Đang nạp Bộ Test Vàng Robot Patrol (531 frames CVAT verified)...")
    patrol_images = sorted(list((SRC_PATROL_DIR / "images").glob("*.jpg")))
    assert len(patrol_images) == 531, f"Lỗi: Kỳ vọng 531 ảnh robot, tìm thấy {len(patrol_images)}"

    for img in patrol_images:
        shutil.copy2(img, DEST_GOLDEN_DIR / "test/images" / img.name)
        lbl = SRC_PATROL_DIR / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, DEST_GOLDEN_DIR / "test/labels" / lbl.name)
        else:
            (DEST_GOLDEN_DIR / "test/labels" / f"{img.stem}.txt").touch()

    # 4. KIỂM DUYỆT RÒ RỈ DỮ LIỆU TUYỆT ĐỐI (ANTI-DATA-LEAKAGE AUDIT)
    print("\n🔒 5. ĐANG TIẾN HÀNH KIỂM DUYỆT RÒ RỈ DỮ LIỆU (ZERO-DATA-LEAKAGE AUDIT)...")
    train_hashes = {compute_sha256(f): f.name for f in (DEST_GOLDEN_DIR / "train/images").glob("*.*")}
    val_hashes = {compute_sha256(f): f.name for f in (DEST_GOLDEN_DIR / "val/images").glob("*.*")}
    test_hashes = {compute_sha256(f): f.name for f in (DEST_GOLDEN_DIR / "test/images").glob("*.*")}

    overlap_train_test = set(train_hashes.keys()) & set(test_hashes.keys())
    overlap_val_test = set(val_hashes.keys()) & set(test_hashes.keys())
    overlap_train_val = set(train_hashes.keys()) & set(val_hashes.keys())

    if overlap_train_test:
        raise AssertionError(f"❌ NGUY HIỂM: Phát hiện {len(overlap_train_test)} ảnh rò rỉ giữa Train và Test!")
    if overlap_val_test:
        raise AssertionError(f"❌ NGUY HIỂM: Phát hiện {len(overlap_val_test)} ảnh rò rỉ giữa Val và Test!")
    if overlap_train_val:
        raise AssertionError(f"❌ NGUY HIỂM: Phát hiện {len(overlap_train_val)} ảnh rò rỉ giữa Train và Val!")

    print("   ✅ Train hashes count: ", len(train_hashes))
    print("   ✅ Val hashes count:   ", len(val_hashes))
    print("   ✅ Test hashes count:  ", len(test_hashes))
    print("   🔒 [AUDIT PASS] ZERO DATA LEAKAGE: 0 ảnh trùng lặp giữa Train, Val và Robot Testset!")

    # 5. Thống kê hộp nhãn chi tiết
    print("\n📊 6. Thống kê hộp nhãn theo từng split:")
    split_stats = {}
    for split in ["train", "val", "test"]:
        imgs = list((DEST_GOLDEN_DIR / split / "images").glob("*.*"))
        lbls = list((DEST_GOLDEN_DIR / split / "labels").glob("*.txt"))
        boxes_per_class = {cid: 0 for cid in range(17)}
        total_boxes = 0

        for lf in lbls:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cid = int(parts[0])
                        boxes_per_class[cid] += 1
                        total_boxes += 1

        split_stats[split] = {
            "frames": len(imgs),
            "boxes": total_boxes,
            "classes": boxes_per_class
        }
        print(f"  • Split '{split:<5}': {len(imgs):4d} frames | {total_boxes:5d} boxes")

    # 6. Ghi file cấu hình YAML
    yaml_content = f"""# Cấu hình Huấn luyện Siêu Mô Hình YOLO26n v6 (Golden Robot Architecture)
path: {DEST_GOLDEN_DIR.resolve()}
train: train/images
val: val/images
test: test/images

nc: 17
names:
  0: patient
  1: doctor
  2: wheelchair
  3: queue_kiosk
  4: reception_counter
  5: sanitizer_stand
  6: pccc_station
  7: medical_waste_bin
  8: water_dispenser
  9: wc_station
  10: elevator_doors
  11: hospital_bed
  12: stretcher_cart
  13: iv_stand
  14: medical_trolley
  15: vital_monitor
  16: hospital_chair
"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n📝 Đã lưu cấu hình huấn luyện chuẩn vàng: {CONFIG_FILE}")

    # Ghi manifest
    manifest = {
        "dataset_name": "hospital_unified_golden_4901_17class",
        "total_frames": 4901,
        "splits": {
            "train": split_stats["train"]["frames"],
            "val": split_stats["val"]["frames"],
            "test_golden_robot": split_stats["test"]["frames"]
        },
        "boxes": {
            "train": split_stats["train"]["boxes"],
            "val": split_stats["val"]["boxes"],
            "test_golden_robot": split_stats["test"]["boxes"],
            "total": sum(s["boxes"] for s in split_stats.values())
        },
        "anti_leakage_status": "CERTIFIED_ZERO_LEAKAGE_SHA256"
    }
    with open(DEST_GOLDEN_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("==================================================================")
    print("🎉 HOÀN TẤT TÁI CẤU TRÚC DATASET CHUẨN VÀNG THÀNH CÔNG!")
    print(f"📊 Tổng quy mô: 4,901 frames (Train: 3,715 | Val: 655 | Golden Test: 531)")
    print(f"📦 Tổng hộp nhãn: {manifest['boxes']['total']} boxes")
    print("==================================================================")


if __name__ == "__main__":
    main()
