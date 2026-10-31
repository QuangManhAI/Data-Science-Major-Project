#!/usr/bin/env python3
"""
merge_targeted_dataset.py
Hợp nhất bộ dữ liệu mô phỏng mục tiêu 400 frames (sau khi duyệt CVAT hoặc dùng bản Pre-Annotated)
vào tập Train và Val của bộ dữ liệu 17 lớp:
- Tạo bộ dataset mới: data/vision/dataset/hospital_unified_targeted_4120_17class
- CAM KẾT KHÔNG RÒ RỈ DỮ LIỆU (Zero Data Leakage 100%):
  * Giữ nguyên 100% Tập Test Đóng Băng (468 frames, 1,167 Ground Truth boxes).
  * Tuyệt đối 0% ảnh mới được phép lọt vào tập Test.
- Tạo cấu hình YAML: configs/hospital_unified_targeted_17class.yaml.
"""

import os
import sys
import shutil
import zipfile
import json
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_UNIFIED_DIR = REPO_ROOT / "data/vision/dataset/hospital_unified_expanded_3720_17class"
TARGETED_STAGING_DIR = REPO_ROOT / "data/vision/dataset/hospital_targeted_staging"
TARGET_DATASET_DIR = REPO_ROOT / "data/vision/dataset/hospital_unified_targeted_4120_17class"
CONFIG_FILE = REPO_ROOT / "configs/hospital_unified_targeted_17class.yaml"

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


def parse_args():
    parser = argparse.ArgumentParser(description="Hợp nhất dữ liệu mục tiêu 400 frames vào Train/Val")
    parser.add_argument("--cvat-zip", type=str, default=None, help="Đường dẫn file zip xuất từ CVAT sau khi duyệt (nếu có)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỷ lệ chia vào Val cho dữ liệu mới (mặc định 15%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def extract_cvat_annotations(zip_path: Path, output_labels_dir: Path):
    """Trích xuất file annotations từ zip CVAT."""
    print(f"📦 Đang giải nén nhãn CVAT đã duyệt từ: {zip_path}")
    label_to_id = {v: k for k, v in UNIFIED_CLASSES.items()}

    # Sao lưu nhãn staging cũ
    backup_dir = output_labels_dir.parent / "labels_backup_pre_cvat_review"
    if not backup_dir.exists() and output_labels_dir.exists():
        print(f"💾 Sao lưu nhãn staging cũ vào {backup_dir.name}...")
        shutil.copytree(output_labels_dir, backup_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        file_list = z.namelist()
        if "annotations.xml" in file_list:
            xml_content = z.read("annotations.xml")
            root = ET.fromstring(xml_content)
            for img in root.findall("image"):
                stem = Path(img.get("name")).stem
                w_img = float(img.get("width"))
                h_img = float(img.get("height"))

                yolo_lines = []
                for box in img.findall("box"):
                    label_name = box.get("label")
                    if label_name not in label_to_id: continue
                    cid = label_to_id[label_name]
                    xtl = max(0.0, min(w_img, float(box.get("xtl"))))
                    ytl = max(0.0, min(h_img, float(box.get("ytl"))))
                    xbr = max(0.0, min(w_img, float(box.get("xbr"))))
                    ybr = max(0.0, min(h_img, float(box.get("ybr"))))
                    bw = xbr - xtl
                    bh = ybr - ytl
                    if bw <= 0 or bh <= 0: continue
                    xc = (xtl + bw / 2.0) / w_img
                    yc = (ytl + bh / 2.0) / h_img
                    yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw/w_img:.6f} {bh/h_img:.6f}")

                txt_p = output_labels_dir / f"{stem}.txt"
                txt_p.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        else:
            # YOLO 1.1 format
            for f in file_list:
                if f.startswith("obj_train_data/") and f.endswith(".txt"):
                    stem = Path(f).stem
                    txt_p = output_labels_dir / f"{stem}.txt"
                    txt_p.write_bytes(z.read(f))


def main():
    args = parse_args()
    random.seed(args.seed)

    print("==================================================================")
    print("🔄 BẮT ĐẦU HỢP NHẤT DỮ LIỆU TARGETED (400 FRAMES) VÀO DATASET 17 CLASSES")
    print(f"• Dataset gốc:     {BASE_UNIFIED_DIR}")
    print(f"• Dữ liệu mới:     {TARGETED_STAGING_DIR}")
    print(f"• Dataset đích:    {TARGET_DATASET_DIR}")
    print("==================================================================")

    if not BASE_UNIFIED_DIR.exists():
        print(f"❌ Không tìm thấy dataset gốc: {BASE_UNIFIED_DIR}")
        sys.exit(1)

    staging_labels = TARGETED_STAGING_DIR / "labels"
    if args.cvat_zip:
        cvat_zip_path = Path(args.cvat_zip).resolve()
        if cvat_zip_path.exists():
            extract_cvat_annotations(cvat_zip_path, staging_labels)

    staging_images = TARGETED_STAGING_DIR / "images"
    new_images = sorted(list(staging_images.glob("*.jpg")))
    print(f"📸 Tìm thấy {len(new_images)} ảnh mới sẵn sàng hợp nhất.")

    # 1. Tạo thư mục đích
    if TARGET_DATASET_DIR.exists():
        shutil.rmtree(TARGET_DATASET_DIR)

    for split in ["train", "val", "test"]:
        (TARGET_DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (TARGET_DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # 2. Sao chép 100% dữ liệu cũ (Train, Val, Test)
    for split in ["train", "val", "test"]:
        src_img_dir = BASE_UNIFIED_DIR / split / "images"
        src_lbl_dir = BASE_UNIFIED_DIR / split / "labels"
        dst_img_dir = TARGET_DATASET_DIR / split / "images"
        dst_lbl_dir = TARGET_DATASET_DIR / split / "labels"

        for img_file in src_img_dir.glob("*.jpg"):
            shutil.copy2(img_file, dst_img_dir / img_file.name)
            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_lbl_dir / lbl_file.name)
            else:
                (dst_lbl_dir / f"{img_file.stem}.txt").touch()

    # 3. Phân chia ảnh mới vào Train và Val (TUYỆT ĐỐI KHÔNG VÀO TEST)
    shuffled_images = list(new_images)
    random.shuffle(shuffled_images)

    val_count = int(len(shuffled_images) * args.val_ratio)
    new_val_images = shuffled_images[:val_count]
    new_train_images = shuffled_images[val_count:]

    print(f"📊 Phân bổ {len(shuffled_images)} ảnh mới:")
    print(f"  • Thêm vào Train: {len(new_train_images)} ảnh (0% rò rỉ vào Test)")
    print(f"  • Thêm vào Val:   {len(new_val_images)} ảnh")
    print(f"  • Thêm vào Test:  0 ảnh (BẢO TOÀN NGUYÊN VẸN 468 FRAMES TEST ĐÓNG BĂNG)")

    for img in new_train_images:
        shutil.copy2(img, TARGET_DATASET_DIR / "train/images" / img.name)
        lbl_file = staging_labels / f"{img.stem}.txt"
        dest_lbl = TARGET_DATASET_DIR / "train/labels" / f"{img.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, dest_lbl)
        else:
            dest_lbl.touch()

    for img in new_val_images:
        shutil.copy2(img, TARGET_DATASET_DIR / "val/images" / img.name)
        lbl_file = staging_labels / f"{img.stem}.txt"
        dest_lbl = TARGET_DATASET_DIR / "val/labels" / f"{img.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, dest_lbl)
        else:
            dest_lbl.touch()

    # 4. Kiểm kê tổng số lượng
    counts = {}
    box_counts = {cid: 0 for cid in UNIFIED_CLASSES.keys()}
    for split in ["train", "val", "test"]:
        imgs = list((TARGET_DATASET_DIR / split / "images").glob("*.jpg"))
        counts[split] = len(imgs)
        for lbl in (TARGET_DATASET_DIR / split / "labels").glob("*.txt"):
            with open(lbl, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cid = int(parts[0])
                        if cid in box_counts:
                            box_counts[cid] += 1

    total_all = sum(counts.values())
    total_boxes = sum(box_counts.values())

    print("\n==================================================================")
    print(f"📈 TỔNG KẾT BỘ DỮ LIỆU TARGETED 17 CLASSES ({total_all} FRAMES):")
    print(f"• Tổng số frames: {total_all}")
    print(f"  - Train: {counts['train']} frames ({counts['train']/total_all*100:.1f}%)")
    print(f"  - Val:   {counts['val']} frames ({counts['val']/total_all*100:.1f}%)")
    print(f"  - Test:  {counts['test']} frames ({counts['test']/total_all*100:.1f}%) [FROZEN TEST BENCHMARK]")
    print(f"• Tổng số Ground Truth Boxes: {total_boxes:,}")
    print("------------------------------------------------------------------")
    print("🔍 Số lượng boxes sau khi tăng cường:")
    for cid in sorted(UNIFIED_CLASSES.keys()):
        cname = UNIFIED_CLASSES[cid]
        star = "⭐" if cname in ["iv_stand", "water_dispenser", "patient"] else "  "
        print(f"  {star} Class {cid:02d} ({cname:<18}): {box_counts[cid]:>5} boxes")
    print("==================================================================")

    # 5. Ghi file cấu hình YAML
    yaml_content = f"""# Cấu hình Huấn luyện YOLO Siêu Mô Hình 17 Lớp (Targeted Boost Dataset)
path: {TARGET_DATASET_DIR}
train: train/images
val: val/images
test: test/images

nc: 17
names:
"""
    for cid in sorted(UNIFIED_CLASSES.keys()):
        yaml_content += f"  {cid}: {UNIFIED_CLASSES[cid]}\n"

    CONFIG_FILE.write_text(yaml_content, encoding="utf-8")
    print(f"✅ Đã tạo file cấu hình huấn luyện: {CONFIG_FILE}")


if __name__ == "__main__":
    main()
