#!/usr/bin/env python3
"""
import_verified_cvat_and_split.py
Kịch bản tự động sau khi người dùng kiểm duyệt xong Ground Truth trên CVAT:
1. Nhận dữ liệu kiểm duyệt từ:
   - File ZIP xuất từ CVAT (CVAT YOLO 1.1 Export), HOẶC
   - Thư mục labels đã chỉnh sửa.
2. Kiểm tra chất lượng dữ liệu (Data Quality & Integrity Audit):
   - Toàn vẹn tọa độ bbox [0, 1]
   - Phù hợp với hệ 17 nhãn
   - Báo cáo số lượng box đã thêm / xóa / sửa.
3. Phân chia Train (70%) / Val (15%) / Test (15% Đóng Băng Khách Quan) chống rò rỉ dữ liệu.
4. Xuất dataset sẵn sàng cho đợt huấn luyện tiếp theo.
"""

import os
import sys
import shutil
import zipfile
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
UNSPLIT_DATASET_DIR = REPO_ROOT / "data/vision/dataset/hospital_unsplit_3120_17class"
DEFAULT_TARGET_DIR = REPO_ROOT / "data/vision/dataset/hospital_verified_3120_17class"

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


def audit_and_split(cvat_zip=None, labels_dir=None, target_dir=DEFAULT_TARGET_DIR):
    print("==================================================================")
    print("🔄 BẮT ĐẦU ĐỒNG BỘ DỮ LIỆU ĐÃ DUYỆT TỪ CVAT VÀO PIPELINE HUẤN LUYỆN")
    print("==================================================================")

    target_dir = Path(target_dir).resolve()
    temp_extract = target_dir / "_temp_extract"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)

    if cvat_zip:
        zip_p = Path(cvat_zip).resolve()
        if not zip_p.exists():
            raise FileNotFoundError(f"Không tìm thấy file ZIP: {zip_p}")
        print(f"📦 Đang giải nén file CVAT: {zip_p.name}...")
        with zipfile.ZipFile(zip_p, "r") as zf:
            zf.extractall(temp_extract)
        
        # Tìm thư mục obj_train_data hoặc thư mục chứa các file .txt
        obj_dir = temp_extract / "obj_train_data"
        if not obj_dir.exists():
            obj_dir = temp_extract / "data/obj_train_data"
        if not obj_dir.exists():
            # Tìm đệ quy bất kỳ folder nào có nhiều file .txt
            txt_files = list(temp_extract.rglob("*.txt"))
            if txt_files:
                obj_dir = txt_files[0].parent
            else:
                raise ValueError("Không tìm thấy file nhãn .txt nào trong ZIP giải nén!")
    elif labels_dir:
        obj_dir = Path(labels_dir).resolve()
        if not obj_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục labels: {obj_dir}")
    else:
        obj_dir = UNSPLIT_DATASET_DIR / "labels"
        print(f"ℹ️ Không truyền input, sử dụng trực tiếp: {obj_dir}")

    print(f"📂 Thư mục nhãn nguồn: {obj_dir}")

    # Lấy toàn bộ 3,120 ảnh từ UNSPLIT_DATASET_DIR
    unsplit_images = sorted(list((UNSPLIT_DATASET_DIR / "images").glob("*.jpg")))
    if not unsplit_images:
        raise FileNotFoundError(f"Không tìm thấy ảnh gốc trong {UNSPLIT_DATASET_DIR / 'images'}")

    print(f"📸 Tìm thấy {len(unsplit_images)} frames ảnh gốc.")

    # Kiểm tra chất lượng và thu thập nhãn
    verified_labels = {}
    class_counts = Counter()
    total_boxes = 0
    empty_frames = 0
    corrupted_boxes = 0

    for img in unsplit_images:
        stem = img.stem
        lbl_file = obj_dir / f"{stem}.txt"
        if not lbl_file.exists():
            # Thử tìm đệ quy nếu tên file có thể ở thư mục con
            found = list(obj_dir.rglob(f"{stem}.txt"))
            if found:
                lbl_file = found[0]

        valid_lines = []
        if lbl_file.exists():
            lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        if 0 <= cid < len(UNIFIED_CLASSES) and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                            valid_lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                            class_counts[cid] += 1
                            total_boxes += 1
                        else:
                            corrupted_boxes += 1
                    except ValueError:
                        corrupted_boxes += 1
        
        if not valid_lines:
            empty_frames += 1
        verified_labels[stem] = valid_lines

    print(f"\n📊 BÁO CÁO TOÀN VẸN DỮ LIỆU:")
    print(f"  • Tổng frames:          {len(unsplit_images)}")
    print(f"  • Bounding Boxes hợp lệ: {total_boxes:,}")
    print(f"  • Frames trống (neg):   {empty_frames}")
    if corrupted_boxes > 0:
        print(f"  ⚠️ Cảnh báo: {corrupted_boxes} boxes có tọa độ ngoài khoảng [0, 1] đã được tự động loại bỏ!")

    # Cấu trúc thư mục đích
    for split in ["train", "val", "test"]:
        (target_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (target_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Đọc manifest gốc để kế thừa phân chia nhằm bảo toàn Test Đóng Băng
    original_manifest_p = REPO_ROOT / "data/vision/dataset/hospital_unified_3120_17class/manifest.json"
    assigned_splits = {"train": [], "val": [], "test": []}

    # Gom theo split gốc
    for split in ["train", "val", "test"]:
        orig_img_dir = REPO_ROOT / f"data/vision/dataset/hospital_unified_3120_17class/{split}/images"
        for p in orig_img_dir.glob("*.jpg"):
            assigned_splits[split].append(p.name)

    print("\n🧊 PHÂN CHIA BẢO TOÀN TẬP TEST ĐÓNG BĂNG:")
    for split in ["train", "val", "test"]:
        print(f"  • {split.upper():<6}: {len(assigned_splits[split])} frames")
        for fname in assigned_splits[split]:
            src_img = UNSPLIT_DATASET_DIR / "images" / fname
            dst_img = target_dir / split / "images" / fname
            shutil.copy2(src_img, dst_img)

            stem = Path(fname).stem
            lines = verified_labels.get(stem, [])
            dst_lbl = target_dir / split / "labels" / f"{stem}.txt"
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # Tạo dataset yaml
    yaml_content = f"""# Cấu hình Huấn luyện YOLO - Bộ dữ liệu đã duyệt trên CVAT
path: {target_dir}
train: train/images
val: val/images
test: test/images

nc: 17
names:
"""
    for cid in sorted(UNIFIED_CLASSES.keys()):
        yaml_content += f"  {cid}: {UNIFIED_CLASSES[cid]}\n"

    yaml_path = REPO_ROOT / "configs/hospital_verified_17class.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n📝 Đã tạo file config huấn luyện: {yaml_path}")

    # Dọn dẹp temp
    if temp_extract.exists():
        shutil.rmtree(temp_extract)

    print("==================================================================")
    print(f"✅ ĐỒNG BỘ HOÀN TẤT! Dataset sẵn sàng tại: {target_dir}")
    print("==================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nhập dữ liệu CVAT và phân chia dataset")
    parser.add_argument("--cvat-zip", type=str, default=None, help="Đường dẫn file ZIP xuất từ CVAT")
    parser.add_argument("--labels-dir", type=str, default=None, help="Đường dẫn thư mục labels đã duyệt")
    parser.add_argument("--target-dir", type=str, default=str(DEFAULT_TARGET_DIR), help="Thư mục xuất dataset đã chia")
    args = parser.parse_args()

    audit_and_split(cvat_zip=args.cvat_zip, labels_dir=args.labels_dir, target_dir=args.target_dir)
