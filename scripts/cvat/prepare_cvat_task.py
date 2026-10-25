#!/usr/bin/env python3
"""
prepare_cvat_task.py
Script tự động đóng gói toàn bộ ảnh và nhãn thành gói CVAT Task hoàn chỉnh (CVAT YOLO 1.1 format).
Hỗ trợ cả bộ 3 classes lâm sàng (Human) và bộ 13 classes trang thiết bị y tế (Hospital Props).

Người dùng hoặc thành viên trong nhóm chỉ cần:
1. Mở CVAT (trên web app.cvat.ai hoặc localhost:8080) -> Bấm "+ Create new task".
2. Dán schema từ `cvat_labels_schema.json` vào tab "Raw".
3. Tải lên `images.zip` (hoặc kéo thả cả folder ảnh).
4. Bấm "Upload Annotations" -> Chọn "YOLO 1.1" -> Tải lên `annotations.zip`.
Toàn bộ bounding box nháp sẽ hiện ngay lập tức trên màn hình CVAT để duyệt và sửa tay!
"""

import os
import sys
import zipfile
import shutil
import json
from pathlib import Path
import argparse

COLOR_PALETTE = [
    "#FF5722", "#4CAF50", "#00BCD4", "#F44336", "#FFEB3B",
    "#2196F3", "#9C27B0", "#795548", "#3F51B5", "#E91E63",
    "#607D8B", "#FF9800", "#8BC34A", "#009688", "#673AB7",
    "#CDDC39", "#FFC107", "#03A9F4", "#E040FB", "#FF5252"
]

def load_classes_from_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "classes" in data and isinstance(data["classes"], dict):
                # Class mapping: key -> name
                classes = {}
                for k, v in data["classes"].items():
                    try:
                        classes[int(k)] = str(v)
                    except ValueError:
                        classes[int(v)] = str(k)
                # Sort by class id
                return [classes[i] for i in sorted(classes.keys())]
    except Exception as e:
        print(f"⚠️ Không thể đọc manifest {manifest_path}: {e}")
    return None

def main():
    repo_root = Path(__file__).resolve().parents[2]
    
    parser = argparse.ArgumentParser(description="Đóng gói CVAT Task theo chuẩn YOLO 1.1")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Đường dẫn thư mục dataset (tự động lấy images, labels, manifest)")
    parser.add_argument("--images-dir", type=str, default=None, help="Thư mục ảnh (nếu không dùng --dataset-dir)")
    parser.add_argument("--labels-dir", type=str, default=None, help="Thư mục nhãn (.txt) (nếu không dùng --dataset-dir)")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục xuất gói CVAT")
    parser.add_argument("--task-name", type=str, default=None, help="Tên gợi ý của CVAT Task")
    args = parser.parse_args()

    # Xác định đường dẫn nguồn
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).resolve()
        src_images_dir = dataset_dir / "images"
        src_labels_dir = dataset_dir / "labels"
        manifest_path = dataset_dir / "manifest.json"
        dataset_name = dataset_dir.name
    else:
        src_images_dir = Path(args.images_dir) if args.images_dir else (repo_root / "data/vision/raw/images")
        src_labels_dir = Path(args.labels_dir) if args.labels_dir else (repo_root / "data/vision/raw/labels")
        manifest_path = src_images_dir.parent / "manifest.json"
        dataset_name = src_images_dir.parent.name

    # Xác định danh sách classes
    class_names = load_classes_from_manifest(manifest_path)
    if not class_names:
        # Fallback mặc định 3 lớp người
        class_names = ["patient", "doctor", "wheelchair"]

    # Xác định tên Task và thư mục xuất
    if not args.task_name:
        task_name = dataset_name.replace("_", " ").title().replace(" ", "_")
    else:
        task_name = args.task_name

    if args.output_dir:
        out_pkg_dir = Path(args.output_dir).resolve()
    else:
        if "props" in dataset_name.lower():
            out_pkg_dir = repo_root / "data" / "vision" / "cvat_props_task_package"
        else:
            out_pkg_dir = repo_root / "data" / "vision" / "cvat_task_package"

    out_pkg_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("📦 BẮT ĐẦU ĐÓNG GÓI TASK CVAT (NATIVE YOLO 1.1 CVAT FORMAT)")
    print(f"• Dataset Name:       {dataset_name}")
    print(f"• Thư mục ảnh nguồn:  {src_images_dir}")
    print(f"• Thư mục nhãn nguồn: {src_labels_dir}")
    print(f"• Thư mục xuất gói:   {out_pkg_dir}")
    print(f"• Số lượng classes:   {len(class_names)} -> {', '.join(class_names)}")
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

    # obj.names
    with open(temp_anno_dir / "obj.names", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")

    # obj.data
    with open(temp_anno_dir / "obj.data", "w", encoding="utf-8") as f:
        f.write(f"classes = {len(class_names)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")

    # train.txt & copy labels
    train_lines = []
    labels_copied = 0
    total_boxes = 0

    for img in image_files:
        lbl_name = f"{img.stem}.txt"
        lbl_src = src_labels_dir / lbl_name
        dest_lbl = obj_data_dir / lbl_name
        
        train_lines.append(f"data/obj_train_data/{img.name}")
        if lbl_src.exists():
            shutil.copy2(lbl_src, dest_lbl)
            labels_copied += 1
            # đếm số boxes
            with open(lbl_src, "r", encoding="utf-8") as lf:
                for line in lf:
                    if line.strip():
                        total_boxes += 1
        else:
            # Tạo file rỗng nếu là negative sample
            dest_lbl.touch()

    with open(temp_anno_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    # Nén annotations.zip
    anno_zip_path = out_pkg_dir / "annotations.zip"
    print(f"🏷️ 2. Đang nén {labels_copied} file nhãn ({total_boxes} boxes) vào {anno_zip_path.name} theo chuẩn CVAT YOLO 1.1...")
    with zipfile.ZipFile(anno_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_anno:
        for root, dirs, files in os.walk(temp_anno_dir):
            for file in files:
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(temp_anno_dir)
                zip_anno.write(abs_path, arcname=str(rel_path))

    # Dọn dẹp temp
    shutil.rmtree(temp_anno_dir)

    # 3. Tạo file schema nhãn json cho CVAT Raw tab
    cvat_schema = []
    for idx, cname in enumerate(class_names):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        cvat_schema.append({
            "name": cname,
            "color": color,
            "type": "rectangle",
            "attributes": []
        })

    schema_file = out_pkg_dir / "cvat_labels_schema.json"
    with open(schema_file, "w", encoding="utf-8") as sf:
        json.dump(cvat_schema, sf, indent=2, ensure_ascii=False)

    # 4. Tạo file hướng dẫn sử dụng nhanh README_CVAT_SETUP.md
    classes_list_text = "\n".join([f"- **`{i}`: `{name}`**" for i, name in enumerate(class_names)])
    readme_content = f"""# 🚀 HƯỚNG DẪN 3 BƯỚC NẠP DỮ LIỆU VÀO CVAT ĐỂ CHECK TAY
Thư mục gói dữ liệu: `{out_pkg_dir}`

Danh mục {len(class_names)} lớp đối tượng trong bộ dữ liệu này:
{classes_list_text}

---

### Bước 1: Tạo Task trên CVAT
1. Mở CVAT (truy cập `http://localhost:8080` hoặc `https://app.cvat.ai`).
2. Bấm **"+ Create new task"**.
3. **Name**: `{task_name}`
4. **Labels**:
   - Bấm vào tab **"Raw"**.
   - Mở file `cvat_labels_schema.json` trong thư mục này, copy toàn bộ nội dung JSON và dán vào ô cấu hình.
   - Bấm **"Done"** để CVAT nạp đủ {len(class_names)} lớp với mã màu tương ứng.
5. **Select files**: Kéo thả file `images.zip` ({images_zip_path.stat().st_size / (1024*1024):.1f} MB) vào vùng upload.
6. Bấm **"Submit & Open"** (Chờ CVAT khởi tạo Job xong trong vài giây).

---

### Bước 2: Nạp Bounding Box nháp đã sinh tự động (AI Pre-annotation)
1. Trong màn hình Task vừa tạo, bấm vào dấu 3 chấm góc trên bên phải (`...`) hoặc menu **Actions**.
2. Chọn **"Upload Annotations"** $\\rightarrow$ Chọn định dạng **"YOLO 1.1"**.
3. Chọn file `annotations.zip` trong thư mục này tải lên.
4. Xong! Mở Job ra bạn sẽ thấy **toàn bộ {len(image_files)} frames đã có sẵn {total_boxes} Bounding Box**, không phải vẽ tay từ đầu!

---

### Bước 3: Thao tác Kiểm Duyệt Nhanh (5-10s / frame)
- **Space** / **F** / **D**: Lướt qua khung hình tiếp theo / lùi khung hình.
- **N**: Vẽ thêm bounding box nếu thấy vật thể bị sót góc.
- **Click vào box + Delete**: Xóa bounding box bị che khuất quá mức hoặc sai góc nhìn.
- **Ctrl + S**: Lưu định kỳ (CVAT cũng tự động auto-save).

---

### Bước 4: Xuất kết quả sau khi duyệt xong
Khi duyệt hoàn tất:
1. Bấm **Menu** góc trái $\\rightarrow$ **Export Task Dataset**.
2. Chọn định dạng **"YOLO 1.1"** (hoặc **"CVAT for images 1.1"**).
3. Tải file ZIP về máy để cập nhật trực tiếp vào dataset chính thức!
"""
    with open(out_pkg_dir / "README_CVAT_SETUP.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\n==================================================================")
    print("✅ ĐÃ ĐÓNG GÓI HOÀN TẤT BỘ CVAT TASK PACKAGE:")
    print(f"• File ảnh:           {images_zip_path} ({images_zip_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"• File nhãn nháp:     {anno_zip_path} ({anno_zip_path.stat().st_size / 1024:.1f} KB)")
    print(f"• File nhãn mẫu:      {schema_file}")
    print(f"• Hướng dẫn chi tiết: {out_pkg_dir / 'README_CVAT_SETUP.md'}")
    print("==================================================================")

if __name__ == "__main__":
    main()
