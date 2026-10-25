#!/usr/bin/env python3
"""
prepare_cvat_task.py
Script tự động đóng gói toàn bộ ảnh và nhãn nháp thành gói CVAT Task hoàn chỉnh (CVAT YOLO 1.1 format).
Người dùng hoặc thành viên trong nhóm chỉ cần:
1. Mở CVAT (trên web hoặc localhost:8080) -> Bấm "Create new task".
2. Tải lên `images.zip` (hoặc chọn thư mục images).
3. Bấm "Upload Annotations" -> Chọn "YOLO 1.1" -> Tải lên `annotations.zip`.
Toàn bộ bounding box nháp sẽ hiện ngay lập tức trên màn hình CVAT để team vào duyệt!
"""

import os
import sys
import zipfile
import shutil
import json
from pathlib import Path

import argparse

def main():
    repo_root = Path(__file__).resolve().parents[2]
    
    parser = argparse.ArgumentParser(description="Đóng gói CVAT Task theo chuẩn YOLO 1.1")
    parser.add_argument("--images-dir", type=str, default=str(repo_root / "data/vision/raw/images"), help="Thư mục ảnh")
    parser.add_argument("--labels-dir", type=str, default=str(repo_root / "data/vision/raw/labels"), help="Thư mục nhãn (.txt)")
    parser.add_argument("--output-dir", type=str, default=str(repo_root / "data/vision/cvat_task_package"), help="Thư mục xuất gói CVAT")
    args = parser.parse_args()

    src_images_dir = Path(args.images_dir)
    src_labels_dir = Path(args.labels_dir)
    out_pkg_dir = Path(args.output_dir)
    out_pkg_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("📦 BẮT ĐẦU ĐÓNG GÓI TASK CVAT (NATIVE YOLO 1.1 CVAT FORMAT)")
    print(f"• Thư mục ảnh nguồn: {src_images_dir}")
    print(f"• Thư mục nhãn nguồn: {src_labels_dir}")
    print(f"• Thư mục xuất gói:   {out_pkg_dir}")
    print("==================================================================")

    image_files = sorted(list(src_images_dir.glob("*.jpg")))
    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào tại {src_images_dir}")
        return

    # 1. Đóng gói images.zip
    images_zip_path = out_pkg_dir / "images.zip"
    print(f"📸 1. Đang nén {len(image_files)} ảnh vào {images_zip_path.name}...")
    with zipfile.ZipFile(images_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_img:
        for img in image_files:
            zip_img.write(img, arcname=img.name)

    # 2. Tạo cấu trúc annotations chuẩn YOLO 1.1 cho CVAT
    # Cấu trúc:
    # annotations.zip/
    # ├── obj.data
    # ├── obj.names
    # ├── train.txt
    # └── obj_train_data/
    #     └── *.txt
    temp_anno_dir = out_pkg_dir / "_temp_yolo11"
    if temp_anno_dir.exists():
        shutil.rmtree(temp_anno_dir)
    obj_data_dir = temp_anno_dir / "obj_train_data"
    obj_data_dir.mkdir(parents=True, exist_ok=True)

    # obj.names (3 classes)
    with open(temp_anno_dir / "obj.names", "w", encoding="utf-8") as f:
        f.write("patient\ndoctor\nwheelchair\n")

    # obj.data
    with open(temp_anno_dir / "obj.data", "w", encoding="utf-8") as f:
        f.write(f"classes = 3\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")

    # train.txt & copy labels
    train_lines = []
    labels_copied = 0
    for img in image_files:
        lbl_name = f"{img.stem}.txt"
        lbl_src = src_labels_dir / lbl_name
        dest_lbl = obj_data_dir / lbl_name
        
        train_lines.append(f"data/obj_train_data/{img.name}")
        if lbl_src.exists():
            shutil.copy2(lbl_src, dest_lbl)
            labels_copied += 1
        else:
            # Tạo file rỗng nếu là negative sample
            dest_lbl.touch()

    with open(temp_anno_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    # Nén annotations.zip
    anno_zip_path = out_pkg_dir / "annotations.zip"
    print(f"🏷️ 2. Đang nén {labels_copied} nhãn vào {anno_zip_path.name} theo chuẩn CVAT YOLO 1.1...")
    with zipfile.ZipFile(anno_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_anno:
        for root, dirs, files in os.walk(temp_anno_dir):
            for file in files:
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(temp_anno_dir)
                zip_anno.write(abs_path, arcname=str(rel_path))

    # Dọn dẹp temp
    shutil.rmtree(temp_anno_dir)

    # 3. Copy file schema nhãn json
    schema_src = repo_root / "configs" / "cvat_labels_schema.json"
    if schema_src.exists():
        shutil.copy2(schema_src, out_pkg_dir / "cvat_labels_schema.json")

    # 4. Tạo file hướng dẫn sử dụng nhanh README_CVAT_SETUP.md
    readme_content = f"""# 🚀 HƯỚNG DẪN 3 BƯỚC NẠP DỮ LIỆU VÀO CVAT
Thư mục gói dữ liệu: `{out_pkg_dir}`

---

### Bước 1: Tạo Task trên CVAT
1. Mở CVAT (truy cập `http://localhost:8080` hoặc `https://app.cvat.ai`).
2. Bấm **"+ Create new task"**.
3. **Name**: `Hospital_Vision_Gold_Dataset`
4. **Labels**: Bấm tab **"Raw"**, mở file `cvat_labels_schema.json` copy toàn bộ nội dung dán vào (hoặc tự tạo 2 class: `person` màu xanh lá, `wheelchair` màu xanh cyan).
5. **Select files**: Kéo thả file `images.zip` vào mục tải ảnh lên.
6. Bấm **"Submit & Open"** (Chờ CVAT upload xong).

---

### Bước 2: Nạp sẵn Bounding Box nháp (AI Pre-annotation)
1. Trong màn hình Task vừa tạo, bấm vào dấu 3 chấm góc phải (`...`) hoặc menu **Actions**.
2. Chọn **"Upload Annotations"** $\\rightarrow$ Chọn format **"YOLO 1.1"**.
3. Chọn file `annotations.zip` trong thư mục này tải lên.
4. Xong! Mở Job ra bạn sẽ thấy **toàn bộ 696 frames đã có sẵn Bounding Box**, không ai phải vẽ tay từ đầu.

---

### Bước 3: Duyệt và sửa lỗi (Chỉ 5-10s/frame)
- Bấm `Space` để lướt qua các ảnh đã chuẩn.
- Bấm `N` để vẽ thêm người bị sót (ví dụ em bé bế trên tay).
- Click vào box chọn và bấm `Delete` nếu AI vẽ nhầm lên Standee poster.
- Bấm `Ctrl + S` lưu định kỳ.

Sau khi review xong, bấm **"Export Task Dataset"** $\\rightarrow$ chọn **"YOLO 1.1"** để tải nhãn hoàn chỉnh về!
"""
    with open(out_pkg_dir / "README_CVAT_SETUP.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\n==================================================================")
    print("✅ ĐÃ ĐÓNG GÓI HOÀN TẤT BỘ CVAT TASK PACKAGE:")
    print(f"• File ảnh:       {images_zip_path} ({images_zip_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"• File nhãn nháp: {anno_zip_path} ({anno_zip_path.stat().st_size / 1024:.1f} KB)")
    print(f"• File nhãn mẫu:  {out_pkg_dir / 'cvat_labels_schema.json'}")
    print(f"• Hướng dẫn chi tiết: {out_pkg_dir / 'README_CVAT_SETUP.md'}")
    print("==================================================================")

if __name__ == "__main__":
    main()
