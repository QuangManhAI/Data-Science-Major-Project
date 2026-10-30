#!/usr/bin/env python3
"""
import_verified_expanded_data.py
Script nhập dữ liệu mô phỏng mở rộng (sau khi duyệt CVAT hoặc dùng bản Pre-Annotated):
- Hợp nhất dữ liệu mới vào tập Train và Val của bộ dữ liệu thống nhất 17 classes.
- CAM KẾT KHÔNG RÒ RỈ DỮ LIỆU (Zero Data Leakage):
  * Giữ nguyên 100% Tập Test Đóng Băng (468 frames, 1,162 Ground Truth boxes).
  * 0% ảnh mới được phép lọt vào tập Test.
- Tạo cấu hình YAML mới: configs/hospital_unified_expanded_17class.yaml.
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
BASE_UNIFIED_DIR = REPO_ROOT / "data/vision/dataset/hospital_unified_3120_17class"
EXPANDED_STAGING_DIR = REPO_ROOT / "data/vision/dataset/hospital_expanded_staging"
TARGET_DATASET_DIR = REPO_ROOT / "data/vision/dataset/hospital_unified_expanded_3720_17class"
CONFIG_FILE = REPO_ROOT / "configs/hospital_unified_expanded_17class.yaml"

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
    parser = argparse.ArgumentParser(description="Nhập dữ liệu mô phỏng mở rộng và hợp nhất vào Train/Val")
    parser.add_argument("--cvat-zip", type=str, default=None, help="Đường dẫn file annotations.zip xuất từ CVAT sau khi duyệt (nếu có)")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Tỷ lệ chia vào tập Val cho dữ liệu mới (mặc định 10%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def extract_cvat_annotations(zip_path: Path, output_labels_dir: Path):
    """Trích xuất file annotations từ zip CVAT (hỗ trợ cả CVAT XML 1.1 và YOLO 1.1 format)."""
    print(f"📦 Đang giải nén nhãn CVAT đã duyệt từ: {zip_path}")
    label_to_id = {v: k for k, v in UNIFIED_CLASSES.items()}

    # Sao lưu nhãn cũ trong staging nếu chưa có backup
    backup_dir = output_labels_dir.parent / "labels_backup_pre_cvat_review"
    if not backup_dir.exists() and output_labels_dir.exists():
        print(f"💾 Sao lưu nhãn staging cũ vào {backup_dir.name}...")
        shutil.copytree(output_labels_dir, backup_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        file_list = z.namelist()
        if "annotations.xml" in file_list:
            print("📄 Phát hiện định dạng 'CVAT for images 1.1' (annotations.xml)...")
            xml_content = z.read("annotations.xml")
            root = ET.fromstring(xml_content)
            images_elem = root.findall("image")
            print(f"📸 Tìm thấy {len(images_elem)} khung hình trong XML...")

            total_boxes = 0
            empty_count = 0

            for img in images_elem:
                img_name = img.get("name")
                stem = Path(img_name).stem
                w_img = float(img.get("width"))
                h_img = float(img.get("height"))

                yolo_lines = []
                for box in img.findall("box"):
                    label_name = box.get("label")
                    if label_name not in label_to_id:
                        continue
                    cid = label_to_id[label_name]
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))

                    xtl = max(0.0, min(w_img, xtl))
                    ytl = max(0.0, min(h_img, ytl))
                    xbr = max(0.0, min(w_img, xbr))
                    ybr = max(0.0, min(h_img, ybr))

                    w_box = xbr - xtl
                    h_box = ybr - ytl
                    if w_box <= 0 or h_box <= 0:
                        continue

                    x_center = (xtl + w_box / 2.0) / w_img
                    y_center = (ytl + h_box / 2.0) / h_img
                    norm_w = w_box / w_img
                    norm_h = h_box / h_img

                    x_center = min(max(x_center, 0.0), 1.0)
                    y_center = min(max(y_center, 0.0), 1.0)
                    norm_w = min(max(norm_w, 0.0), 1.0)
                    norm_h = min(max(norm_h, 0.0), 1.0)

                    yolo_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    total_boxes += 1

                dest_txt = output_labels_dir / f"{stem}.txt"
                with open(dest_txt, "w", encoding="utf-8") as f:
                    if yolo_lines:
                        f.write("\n".join(yolo_lines) + "\n")
                    else:
                        empty_count += 1

            print(f"✅ Đã trích xuất & chuẩn hóa thành công {len(images_elem)} file nhãn ({total_boxes} boxes, {empty_count} frames trống) từ CVAT XML!")
        else:
            temp_dir = output_labels_dir.parent / "_temp_cvat_extract"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            z.extractall(temp_dir)

            candidate_dirs = list(temp_dir.rglob("obj_train_data"))
            obj_dir = candidate_dirs[0] if candidate_dirs else temp_dir

            count = 0
            for txt_file in obj_dir.glob("*.txt"):
                if txt_file.name in ["train.txt", "obj.names", "obj.data"]:
                    continue
                dest_txt = output_labels_dir / txt_file.name
                shutil.copy2(txt_file, dest_txt)
                count += 1

            shutil.rmtree(temp_dir)
            print(f"✅ Đã trích xuất {count} file nhãn từ CVAT vào {output_labels_dir}")


def main():
    args = parse_args()
    random.seed(args.seed)

    print("==================================================================")
    print("🔄 BẮT ĐẦU HỢP NHẤT DỮ LIỆU EXPANDED VÀO DATASET 17 CLASSES")
    print(f"• Dataset gốc:     {BASE_UNIFIED_DIR}")
    print(f"• Dữ liệu mới:     {EXPANDED_STAGING_DIR}")
    print(f"• Dataset đích:    {TARGET_DATASET_DIR}")
    print("==================================================================")

    if not BASE_UNIFIED_DIR.exists():
        print(f"❌ Không tìm thấy dataset gốc: {BASE_UNIFIED_DIR}")
        sys.exit(1)

    # 1. Chuẩn bị thư mục nhãn mới
    staging_labels = EXPANDED_STAGING_DIR / "labels"
    if args.cvat_zip:
        cvat_zip_path = Path(args.cvat_zip).resolve()
        if cvat_zip_path.exists():
            extract_cvat_annotations(cvat_zip_path, staging_labels)
        else:
            print(f"⚠️ Không tìm thấy file CVAT zip {cvat_zip_path}, sử dụng nhãn staging hiện tại.")

    staging_images = EXPANDED_STAGING_DIR / "images"
    new_images = sorted(list(staging_images.glob("*.jpg")))
    print(f"📸 Tìm thấy {len(new_images)} ảnh mới sẵn sàng hợp nhất.")

    # 2. Tạo bản sao cấu trúc dataset đích
    if TARGET_DATASET_DIR.exists():
        shutil.rmtree(TARGET_DATASET_DIR)

    for split in ["train", "val", "test"]:
        (TARGET_DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (TARGET_DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # 3. Sao chép 100% dữ liệu gốc
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

    # 4. Phân chia ảnh mới vào Train và Val (TUYỆT ĐỐI KHÔNG VÀO TEST)
    shuffled_images = list(new_images)
    random.shuffle(shuffled_images)

    val_count = int(len(shuffled_images) * args.val_ratio)
    new_val_images = shuffled_images[:val_count]
    new_train_images = shuffled_images[val_count:]

    print(f"📊 Phân chia {len(shuffled_images)} ảnh mới:")
    print(f"  • Thêm vào Train: {len(new_train_images)} ảnh (0% rò rỉ vào Test)")
    print(f"  • Thêm vào Val:   {len(new_val_images)} ảnh")
    print(f"  • Thêm vào Test:  0 ảnh (BẢO TOÀN NGUYÊN VẸN TEST ĐÓNG BĂNG)")

    for img in new_train_images:
        dest_img = TARGET_DATASET_DIR / "train/images" / img.name
        shutil.copy2(img, dest_img)
        lbl_file = staging_labels / f"{img.stem}.txt"
        dest_lbl = TARGET_DATASET_DIR / "train/labels" / f"{img.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, dest_lbl)
        else:
            dest_lbl.touch()

    for img in new_val_images:
        dest_img = TARGET_DATASET_DIR / "val/images" / img.name
        shutil.copy2(img, dest_img)
        lbl_file = staging_labels / f"{img.stem}.txt"
        dest_lbl = TARGET_DATASET_DIR / "val/labels" / f"{img.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, dest_lbl)
        else:
            dest_lbl.touch()

    # 5. Kiểm kê thống kê số lượng
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
    print("📈 TỔNG KẾT BỘ DỮ LIỆU EXPANDED 17 CLASSES HOÀN TẤT:")
    print(f"• Tổng số frames: {total_all}")
    print(f"  - Train: {counts['train']} frames ({counts['train']/total_all*100:.1f}%)")
    print(f"  - Val:   {counts['val']} frames ({counts['val']/total_all*100:.1f}%)")
    print(f"  - Test:  {counts['test']} frames ({counts['test']/total_all*100:.1f}%) [FROZEN TEST BENCHMARK]")
    print(f"• Tổng số Ground Truth Boxes: {total_boxes}")
    print("------------------------------------------------------------------")
    print("🔍 Số lượng boxes theo từng lớp trong toàn bộ dataset:")
    for cid in sorted(box_counts.keys()):
        print(f"  • Class {cid:02d} ({UNIFIED_CLASSES[cid]:<18}): {box_counts[cid]:>5} boxes")
    print("==================================================================")

    # 6. Tạo file YAML cấu hình huấn luyện
    yaml_content = f"""# Cấu hình Huấn luyện YOLO Siêu Mô Hình 17 Lớp (Expanded Dataset)
path: {TARGET_DATASET_DIR}
train: train/images
val: val/images
test: test/images

nc: 17
names:
"""
    for cid in sorted(UNIFIED_CLASSES.keys()):
        yaml_content += f"  {cid}: {UNIFIED_CLASSES[cid]}\n"

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"✅ Đã tạo file cấu hình huấn luyện: {CONFIG_FILE}")


if __name__ == "__main__":
    main()
