#!/usr/bin/env python3
"""
export_cvat_to_dataset.py
Script nhập ngược (Import) kết quả gán nhãn đã kiểm duyệt từ CVAT về lại Unified Dataset.
Hỗ trợ cả 2 định dạng export phổ biến của CVAT:
1. CVAT for images 1.1 (file `annotations.xml` bên trong file ZIP)
2. YOLO 1.1 (thư mục `obj_train_data/*.txt` bên trong file ZIP)
Chuyển đổi nhãn chuẩn sang YOLO (.txt) tại `data/vision/dataset/labels/`
và di chuyển (MOVE) toàn bộ ảnh tương ứng từ `data/vision/raw/images/` sang `data/vision/dataset/images/`
(đảm bảo nguyên tắc Zero Duplication - không nhân bản dữ liệu).
"""

import os
import sys
import zipfile
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[2]

    if len(sys.argv) < 2:
        print("Sử dụng: python export_cvat_to_dataset.py <đường_dẫn_file_cvat_export.zip>")
        return

    cvat_zip_path = Path(sys.argv[1]).resolve()
    if not cvat_zip_path.exists():
        print(f"❌ Không tìm thấy file: {cvat_zip_path}")
        return

    dataset_dir = repo_root / "data" / "vision" / "dataset"
    dataset_img_dir = dataset_dir / "images"
    dataset_lbl_dir = dataset_dir / "labels"
    dataset_img_dir.mkdir(parents=True, exist_ok=True)
    dataset_lbl_dir.mkdir(parents=True, exist_ok=True)

    raw_img_dir = repo_root / "data" / "vision" / "raw" / "images"
    raw_lbl_dir = repo_root / "data" / "vision" / "raw" / "labels"

    temp_extract = repo_root / "data" / "vision" / "_temp_cvat_import"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    temp_extract.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("📥 BẮT ĐẦU NHẬP DỮ LIỆU ĐÃ KIỂM DUYỆT TỪ CVAT VỀ DATASET CHÍNH THỨC")
    print(f"• File CVAT Export: {cvat_zip_path}")
    print(f"• Đích đến:         {dataset_dir} (images/ & labels/)")
    print("==================================================================")

    with zipfile.ZipFile(cvat_zip_path, "r") as z:
        z.extractall(temp_extract)

    class_map = {
        "person": 0,
        "wheelchair": 1
    }

    imported_frames = 0
    total_boxes = 0

    # 1. Kiểm tra xem file ZIP chứa annotations.xml (CVAT 1.1 format) hay file .txt (YOLO format)
    xml_file = temp_extract / "annotations.xml"
    if xml_file.exists():
        print("🔍 Phát hiện định dạng: CVAT for images 1.1 (annotations.xml)")
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for img_elem in root.findall("image"):
            img_name = img_elem.get("name", "")
            img_w = float(img_elem.get("width", 640))
            img_h = float(img_elem.get("height", 480))
            stem = Path(img_name).stem

            yolo_lines = []
            for box in img_elem.findall("box"):
                lbl = box.get("label", "")
                if lbl in class_map:
                    cid = class_map[lbl]
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))

                    # Chuẩn hóa tọa độ YOLO [xc, yc, w, h] trong khoảng [0, 1]
                    xc = max(0.0, min(1.0, (xtl + xbr) / (2.0 * img_w)))
                    yc = max(0.0, min(1.0, (ytl + ybr) / (2.0 * img_h)))
                    bw = max(0.0, min(1.0, (xbr - xtl) / img_w))
                    bh = max(0.0, min(1.0, (ybr - ytl) / img_h))

                    yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    total_boxes += 1

            # Lưu file nhãn YOLO .txt vào dataset/labels/
            out_lbl = dataset_lbl_dir / f"{stem}.txt"
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            # Di chuyển (MOVE) file ảnh từ raw/ sang dataset/
            src_img = raw_img_dir / f"{stem}.jpg"
            if src_img.exists():
                shutil.move(str(src_img), str(dataset_img_dir / f"{stem}.jpg"))

            # Xóa nhãn nháp cũ trong raw/
            old_raw_lbl = raw_lbl_dir / f"{stem}.txt"
            if old_raw_lbl.exists():
                old_raw_lbl.unlink()

            imported_frames += 1

    else:
        # Định dạng YOLO 1.1 dạng file .txt
        print("🔍 Phát hiện định dạng: Native YOLO 1.1 (.txt)")
        txt_files = list(temp_extract.rglob("*.txt"))
        for txt_file in txt_files:
            if txt_file.name in ["obj.data", "obj.names", "train.txt"]:
                continue
            stem = txt_file.stem

            shutil.copy2(txt_file, dataset_lbl_dir / txt_file.name)

            src_img = raw_img_dir / f"{stem}.jpg"
            if src_img.exists():
                shutil.move(str(src_img), str(dataset_img_dir / f"{stem}.jpg"))

            old_raw_lbl = raw_lbl_dir / f"{stem}.txt"
            if old_raw_lbl.exists():
                old_raw_lbl.unlink()

            imported_frames += 1

    shutil.rmtree(temp_extract)

    # Nếu file zip ban đầu nằm ngay trong dataset/images/ thì dọn dẹp chuyển về cvat_task_package
    if cvat_zip_path.parent == dataset_img_dir:
        archive_dest = repo_root / "data" / "vision" / "cvat_task_package" / cvat_zip_path.name
        shutil.move(str(cvat_zip_path), str(archive_dest))
        print(f"📦 Đã di dời file ZIP kiểm duyệt sang: {archive_dest}")

    print("\n==================================================================")
    print(f"✅ ĐÃ NHẬP THÀNH CÔNG {imported_frames} FRAMES ({total_boxes} BOXES ĐÃ DUYỆT) VÀO DATASET!")
    print(f"• Số ảnh trong dataset/images:  {len(list(dataset_img_dir.glob('*.jpg')))}")
    print(f"• Số nhãn trong dataset/labels:  {len(list(dataset_lbl_dir.glob('*.txt')))}")
    print(f"• Số ảnh còn lại trong raw:     {len(list(raw_img_dir.glob('*.jpg')))}")
    print("==================================================================")

if __name__ == "__main__":
    main()
