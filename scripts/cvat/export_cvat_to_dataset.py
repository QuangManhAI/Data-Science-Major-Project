#!/usr/bin/env python3
"""
export_cvat_to_dataset.py
Script nhập ngược (Import) kết quả gán nhãn đã kiểm duyệt từ CVAT về lại Dataset chính thức.
Hỗ trợ cả 2 định dạng export phổ biến của CVAT:
1. CVAT for images 1.1 (file `annotations.xml` bên trong file ZIP)
2. YOLO 1.1 (thư mục `obj_train_data/*.txt` bên trong file ZIP)
"""

import os
import sys
import zipfile
import shutil
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse

def main():
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Nhập nhãn đã kiểm duyệt từ CVAT vào Dataset chính thức")
    parser.add_argument("zip_path", type=str, help="Đường dẫn tới file ZIP export từ CVAT")
    parser.add_argument("--dataset-dir", type=str, default=str(repo_root / "data/vision/dataset/hospital_props_1600_13class"),
                        help="Thư mục dataset đích cần cập nhật nhãn")
    args = parser.parse_args()

    cvat_zip_path = Path(args.zip_path).resolve()
    if not cvat_zip_path.exists():
        print(f"❌ Không tìm thấy file: {cvat_zip_path}")
        return

    dataset_dir = Path(args.dataset_dir).resolve()
    dataset_img_dir = dataset_dir / "images"
    dataset_lbl_dir = dataset_dir / "labels"
    dataset_img_dir.mkdir(parents=True, exist_ok=True)
    dataset_lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_dir / "manifest.json"
    class_map = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                m_data = json.load(mf)
                if "classes" in m_data:
                    for k, v in m_data["classes"].items():
                        try:
                            class_map[str(v)] = int(k)
                        except ValueError:
                            class_map[str(k)] = int(v)
        except Exception as e:
            print(f"⚠️ Không đọc được manifest: {e}")

    # Fallback mặc định nếu không có manifest
    if not class_map:
        class_map = {"patient": 0, "doctor": 1, "wheelchair": 2}

    temp_extract = repo_root / "data" / "vision" / "_temp_cvat_import"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    temp_extract.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("📥 BẮT ĐẦU NHẬP DỮ LIỆU ĐÃ KIỂM DUYỆT TỪ CVAT VỀ DATASET CHÍNH THỨC")
    print(f"• File CVAT Export: {cvat_zip_path}")
    print(f"• Đích đến:         {dataset_dir}")
    print(f"• Classes Map ({len(class_map)} lớp): {class_map}")
    print("==================================================================")

    with zipfile.ZipFile(cvat_zip_path, "r") as z:
        z.extractall(temp_extract)

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

            imported_frames += 1

    else:
        # Định dạng YOLO 1.1 dạng file .txt
        print("🔍 Phát hiện định dạng: Native YOLO 1.1 (.txt)")
        txt_files = list(temp_extract.rglob("*.txt"))
        for txt_file in txt_files:
            if txt_file.name in ["obj.data", "obj.names", "train.txt"]:
                continue
            shutil.copy2(txt_file, dataset_lbl_dir / txt_file.name)
            imported_frames += 1

            with open(txt_file, "r", encoding="utf-8") as tf:
                for line in tf:
                    if line.strip():
                        total_boxes += 1

    shutil.rmtree(temp_extract)

    print("\n==================================================================")
    print(f"✅ ĐÃ CẬP NHẬT THÀNH CÔNG {imported_frames} FRAMES ({total_boxes} BOXES ĐÃ DUYỆT) VÀO DATASET!")
    print(f"• Số ảnh trong dataset: {len(list(dataset_img_dir.glob('*.jpg')))}")
    print(f"• Số nhãn trong dataset: {len(list(dataset_lbl_dir.glob('*.txt')))}")
    print("==================================================================")

if __name__ == "__main__":
    main()
