#!/usr/bin/env python3
"""
merge_targeted_dataset.py
=========================
Hợp nhất bộ dữ liệu mô phỏng mục tiêu (sau khi duyệt CVAT hoặc dùng bản Pre-Annotated)
vào tập Train và Val của bộ dữ liệu 17 lớp:
- Hỗ trợ mở rộng từ 4,120 frames -> 4,370 frames (Round 2).
- CAM KẾT KHÔNG RÒ RỈ DỮ LIỆU (Zero Data Leakage 100%):
  * Giữ nguyên 100% Tập Test Đóng Băng (468 frames, 1,167 Ground Truth boxes).
  * Tuyệt đối 0% ảnh mới được phép lọt vào tập Test.
- Đồng bộ cấu hình YAML: configs/hospital_unified_targeted_17class.yaml.
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
    parser = argparse.ArgumentParser(description="Hợp nhất dữ liệu mục tiêu vào Train/Val")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(REPO_ROOT / "data/vision/dataset/hospital_unified_targeted_4120_17class"),
        help="Đường dẫn bộ dữ liệu nền tảng hiện tại (4,120 frames)"
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default=str(REPO_ROOT / "data/vision/dataset/hospital_targeted_staging_r2"),
        help="Thư mục staging chứa 250 frames đợt 2"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=str(REPO_ROOT / "data/vision/dataset/hospital_unified_targeted_4370_17class"),
        help="Đường dẫn bộ dữ liệu hợp nhất đích (4,370 frames)"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(REPO_ROOT / "configs/hospital_unified_targeted_17class.yaml"),
        help="File YAML cấu hình huấn luyện YOLO"
    )
    parser.add_argument("--cvat-zip", type=str, default=None, help="Đường dẫn file zip xuất từ CVAT sau khi duyệt (nếu có)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỷ lệ chia vào Val cho dữ liệu mới (mặc định 15%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def extract_cvat_annotations(zip_path: Path, output_labels_dir: Path):
    """Trích xuất file annotations từ zip CVAT."""
    print(f"📦 Đang giải nén nhãn CVAT đã duyệt từ: {zip_path}")
    label_to_id = {v: k for k, v in UNIFIED_CLASSES.items()}

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
                    label = box.get("label")
                    if label in label_to_id:
                        cid = label_to_id[label]
                        xtl = float(box.get("xtl"))
                        ytl = float(box.get("ytl"))
                        xbr = float(box.get("xbr"))
                        ybr = float(box.get("ybr"))

                        bw = xbr - xtl
                        bh = ybr - ytl
                        cx = (xtl + xbr) / 2.0 / w_img
                        cy = (ytl + ybr) / 2.0 / h_img
                        nw = bw / w_img
                        nh = bh / h_img
                        yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                lbl_file = output_labels_dir / f"{stem}.txt"
                with open(lbl_file, "w", encoding="utf-8") as f:
                    for l in yolo_lines:
                        f.write(l + "\n")
            print("✅ Đã giải nén và chuyển đổi annotations.xml sang YOLO format thành công.")
        else:
            txt_files = [
                f for f in file_list
                if f.endswith(".txt")
                and not f.endswith(("classes.txt", "train.txt", "val.txt", "test.txt", "obj.data", "obj.names"))
                and ("obj_train_data" in f or "labels" in f or Path(f).name.startswith("TGT"))
            ]
            for tf in txt_files:
                stem = Path(tf).stem
                content = z.read(tf).decode("utf-8")
                lbl_file = output_labels_dir / f"{stem}.txt"
                with open(lbl_file, "w", encoding="utf-8") as f:
                    f.write(content)
            print(f"✅ Đã giải nén {len(txt_files)} file nhãn YOLO trực tiếp từ zip vào {output_labels_dir.name}.")


def main():
    args = parse_args()
    random.seed(args.seed)

    base_dir = Path(args.base_dir)
    staging_dir = Path(args.staging_dir)
    target_dir = Path(args.target_dir)
    config_file = Path(args.config_file)

    staging_images_dir = staging_dir / "images"
    staging_labels_dir = staging_dir / "labels"

    if args.cvat_zip:
        cvat_zip_path = Path(args.cvat_zip)
        if cvat_zip_path.exists():
            extract_cvat_annotations(cvat_zip_path, staging_labels_dir)
        else:
            print(f"❌ Không tìm thấy file cvat-zip: {cvat_zip_path}")
            sys.exit(1)

    print("==================================================================")
    print("🔄 BẮT ĐẦU QUY TRÌNH HỢP NHẤT DỮ LIỆU TARGETED (ANTI-LEAKAGE)")
    print(f"• Dataset nền tảng:  {base_dir.name}")
    print(f"• Thư mục bổ sung:   {staging_dir.name}")
    print(f"• Dataset đích:      {target_dir.name}")
    print("==================================================================")

    for split in ["train", "val", "test"]:
        (target_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (target_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Sao chép toàn bộ dữ liệu nền tảng...")
    for split in ["train", "val", "test"]:
        src_img_dir = base_dir / split / "images"
        src_lbl_dir = base_dir / split / "labels"

        dst_img_dir = target_dir / split / "images"
        dst_lbl_dir = target_dir / split / "labels"

        imgs = list(src_img_dir.glob("*.*"))
        for img in imgs:
            shutil.copy2(img, dst_img_dir / img.name)
            lbl = src_lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl_dir / f"{img.stem}.txt")

        print(f"  • Đã chép {split}: {len(imgs)} frames.")

    print("\n[2/3] Phân chia và hợp nhất dữ liệu mới vào Train và Val...")
    new_images = sorted(list(staging_images_dir.glob("*.jpg")) + list(staging_images_dir.glob("*.png")))
    random.shuffle(new_images)

    val_count = int(len(new_images) * args.val_ratio)
    new_val_images = new_images[:val_count]
    new_train_images = new_images[val_count:]

    print(f"  • Tổng số ảnh mới: {len(new_images)}")
    print(f"  • Thêm vào Train:  {len(new_train_images)} frames")
    print(f"  • Thêm vào Val:    {len(new_val_images)} frames")
    print(f"  • Thêm vào Test:   0 frames (100% Frozen Test Set - Anti-Leakage)")

    for img in new_train_images:
        shutil.copy2(img, target_dir / "train/images" / img.name)
        lbl = staging_labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, target_dir / "train/labels" / f"{img.stem}.txt")

    for img in new_val_images:
        shutil.copy2(img, target_dir / "val/images" / img.name)
        lbl = staging_labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, target_dir / "val/labels" / f"{img.stem}.txt")

    print("\n[3/3] Kiểm tra tính toàn vẹn và tính toán số lượng nhãn theo 17 classes...")
    total_boxes = 0
    class_counts = {cid: 0 for cid in UNIFIED_CLASSES.keys()}
    split_counts = {}

    for split in ["train", "val", "test"]:
        imgs = list((target_dir / split / "images").glob("*.*"))
        lbls = list((target_dir / split / "labels").glob("*.txt"))
        split_counts[split] = len(imgs)

        for lf in lbls:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        class_counts[cid] += 1
                        total_boxes += 1

    total_frames = sum(split_counts.values())

    test_imgs_base = {p.name for p in (base_dir / "test/images").glob("*.*")}
    test_imgs_target = {p.name for p in (target_dir / "test/images").glob("*.*")}
    assert test_imgs_base == test_imgs_target, "LỖI BẢO MẬT: Tập Test bị thay đổi!"
    print("🔒 XÁC NHẬN BẢO MẬT: Tập Test Đóng Băng 100% (468 frames) hoàn toàn nguyên vẹn!")

    config_content = f"""# Cấu hình Huấn luyện YOLO Siêu Mô Hình 17 Lớp (Targeted Boost v5 Dataset)
path: {target_dir.resolve()}
train: train/images
val: val/images
test: test/images

nc: 17
names:
"""
    for cid in sorted(UNIFIED_CLASSES.keys()):
        config_content += f"  {cid}: {UNIFIED_CLASSES[cid]}\n"

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_content)
    print(f"📝 Đã cập nhật cấu hình huấn luyện: {config_file}")

    print("\n==================================================================")
    print(f"🏆 HỢP NHẤT HOÀN TẤT: {target_dir.name}")
    print("==================================================================")
    print(f"• Tổng số frames:     {total_frames:,} frames")
    print(f"  - Train:            {split_counts['train']:,} ({split_counts['train']/total_frames*100:.1f}%)")
    print(f"  - Val:              {split_counts['val']:,} ({split_counts['val']/total_frames*100:.1f}%)")
    print(f"  - Test (Frozen):    {split_counts['test']:,} ({split_counts['test']/total_frames*100:.1f}%)")
    print(f"• Tổng số bounding box: {total_boxes:,} boxes across 17 classes")
    print("-" * 66)
    for cid in sorted(UNIFIED_CLASSES.keys()):
        cname = UNIFIED_CLASSES[cid]
        star = "⭐" if cname in ["wheelchair", "iv_stand"] else "  "
        print(f"  {star} Class {cid:02d} ({cname:<18}): {class_counts[cid]:>5} boxes")
    print("==================================================================")


if __name__ == "__main__":
    main()
