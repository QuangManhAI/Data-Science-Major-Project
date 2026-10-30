#!/usr/bin/env python3
"""
export_unsplit_unified_dataset.py
Kịch bản tổng hợp toàn bộ 3,120 frames dữ liệu bệnh viện (17 classes) thành:
1. Thư mục dataset phẳng duy nhất chưa phân chia (Unsplit Dataset):
   `data/vision/dataset/hospital_unsplit_3120_17class/`
   ├── images/   (3,120 frames .jpg)
   ├── labels/   (3,120 files .txt nhãn YOLO 17 classes)
   ├── classes.txt
   ├── manifest.json
   └── README.md
2. Gói cài đặt hoàn chỉnh nạp trực tiếp vào CVAT (CVAT YOLO 1.1 Package):
   `data/vision/cvat_unified_3120_package/`
   ├── images.zip              (Tất cả 3,120 frames nén sẵn để tạo task)
   ├── annotations.zip         (Annotations YOLO 1.1 chuẩn CVAT với 7,903 boxes)
   ├── cvat_labels_schema.json (Schema 17 nhãn màu tương phản cao dán vào tab Raw)
   └── README_CVAT_SETUP.md    (Hướng dẫn chi tiết từng bước nạp và kiểm duyệt GT)
"""

import os
import sys
import shutil
import json
import zipfile
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "data/vision/dataset/hospital_unified_3120_17class"
DEST_UNSPLIT_DIR = REPO_ROOT / "data/vision/dataset/hospital_unsplit_3120_17class"
DEST_CVAT_DIR = REPO_ROOT / "data/vision/cvat_unified_3120_package"

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

DISTINCT_COLORS = [
    "#E6194B",  # 0: patient (Red)
    "#3CB44B",  # 1: doctor (Green)
    "#FFE119",  # 2: wheelchair (Yellow)
    "#4363D8",  # 3: queue_kiosk (Royal Blue)
    "#F58231",  # 4: reception_counter (Orange)
    "#911EB4",  # 5: sanitizer_stand (Purple)
    "#46F0F0",  # 6: pccc_station (Cyan)
    "#F032E6",  # 7: medical_waste_bin (Magenta)
    "#BCF60C",  # 8: water_dispenser (Lime)
    "#FABEBE",  # 9: wc_station (Pink)
    "#008080",  # 10: elevator_doors (Teal)
    "#E6BEFF",  # 11: hospital_bed (Lavender)
    "#9A6324",  # 12: stretcher_cart (Brown)
    "#FFFAC8",  # 13: iv_stand (Beige / Warm Yellow)
    "#800000",  # 14: medical_trolley (Maroon)
    "#AAFFC3",  # 15: vital_monitor (Mint)
    "#808000"   # 16: hospital_chair (Olive)
]


def consolidate_unsplit_dataset():
    print("==================================================================")
    print("🚀 BƯỚC 1: TỔNG HỢP DỮ LIỆU CHƯA CHIA (UNSPLIT DATASET - 3,120 FRAMES)")
    print(f"📁 Nguồn: {SOURCE_DIR}")
    print(f"🎯 Đích:  {DEST_UNSPLIT_DIR}")
    print("==================================================================")

    if DEST_UNSPLIT_DIR.exists():
        print(f"🧹 Dọn dẹp thư mục đích cũ: {DEST_UNSPLIT_DIR}")
        shutil.rmtree(DEST_UNSPLIT_DIR)

    images_dest = DEST_UNSPLIT_DIR / "images"
    labels_dest = DEST_UNSPLIT_DIR / "labels"
    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)

    class_counts = Counter()
    total_boxes = 0
    empty_label_files = 0
    copied_images = 0
    copied_labels = 0

    image_source_map = {}

    for split in ["train", "val", "test"]:
        img_src_dir = SOURCE_DIR / split / "images"
        lbl_src_dir = SOURCE_DIR / split / "labels"

        split_images = sorted(list(img_src_dir.glob("*.jpg")))
        print(f"📦 Đang gom từ split '{split}': {len(split_images)} frames...")

        for img_path in split_images:
            stem = img_path.stem
            dest_img = images_dest / img_path.name
            
            if dest_img.exists():
                raise ValueError(f"Trùng tên file ảnh: {img_path.name}")
            
            shutil.copy2(img_path, dest_img)
            copied_images += 1
            image_source_map[img_path.name] = split

            # Copy label
            lbl_name = f"{stem}.txt"
            lbl_path = lbl_src_dir / lbl_name
            dest_lbl = labels_dest / lbl_name

            if lbl_path.exists():
                content = lbl_path.read_text(encoding="utf-8").strip()
                dest_lbl.write_text(content + "\n" if content else "", encoding="utf-8")
                copied_labels += 1
                if not content:
                    empty_label_files += 1
                else:
                    for line in content.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            class_counts[cid] += 1
                            total_boxes += 1
            else:
                dest_lbl.touch()
                copied_labels += 1
                empty_label_files += 1

    print(f"\n✅ Đã tổng hợp thành công {copied_images} images và {copied_labels} labels!")
    print(f"📊 Tổng số Bounding Boxes: {total_boxes}")
    print(f"🪹 Số frames Hard Negative (ảnh sạch không có vật thể): {empty_label_files}")

    # Ghi file classes.txt
    classes_txt_path = DEST_UNSPLIT_DIR / "classes.txt"
    with open(classes_txt_path, "w", encoding="utf-8") as f:
        for cid in sorted(UNIFIED_CLASSES.keys()):
            f.write(f"{UNIFIED_CLASSES[cid]}\n")

    # Ghi manifest.json
    manifest_data = {
        "dataset_name": "hospital_unsplit_3120_17class",
        "description": "Toàn bộ 3,120 khung hình thực tế bệnh viện đã chuẩn hóa 17 nhãn, CHƯA phân chia Train/Val/Test, phục vụ đưa lên CVAT kiểm duyệt Ground Truth.",
        "total_frames": copied_images,
        "total_labels": copied_labels,
        "total_bounding_boxes": total_boxes,
        "empty_negative_frames": empty_label_files,
        "classes": {str(k): v for k, v in UNIFIED_CLASSES.items()},
        "boxes_per_class": {
            UNIFIED_CLASSES[cid]: {
                "class_id": cid,
                "box_count": class_counts[cid]
            } for cid in sorted(UNIFIED_CLASSES.keys())
        },
        "original_split_origin": {
            "from_train": sum(1 for v in image_source_map.values() if v == "train"),
            "from_val": sum(1 for v in image_source_map.values() if v == "val"),
            "from_test": sum(1 for v in image_source_map.values() if v == "test")
        }
    }

    manifest_path = DEST_UNSPLIT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # Ghi README.md
    readme_text = f"""# 🏥 BỘ DỮ LIỆU TỔNG HỢP BỆNH VIỆN 17 LỚP (CHƯA CHIA TRAIN/VAL/TEST)
- **Thư mục:** `{DEST_UNSPLIT_DIR}`
- **Quy mô:** {copied_images:,} ảnh, {total_boxes:,} bounding boxes
- **Mục đích:** Bộ dữ liệu phẳng (raw/unsplit) tổng hợp đầy đủ 100% hình ảnh thực tế lâm sàng, thiết bị y tế và ca giường bệnh trước khi phân chia, dùng để import lên CVAT rà soát và kiểm duyệt Ground Truth (GT).

## 📋 Danh mục 17 Lớp Đối Tượng
| Class ID | Tên Nhãn (Taxonomy) | Số Bounding Boxes | Mô tả Thực Tế |
|:---:|:---|:---:|:---|
"""
    for cid in sorted(UNIFIED_CLASSES.keys()):
        cname = UNIFIED_CLASSES[cid]
        cnt = class_counts[cid]
        readme_text += f"| `{cid}` | **`{cname}`** | {cnt:,} | Nhãn số {cid} |\n"

    readme_text += f"""
## 🗂️ Cấu trúc thư mục
```
hospital_unsplit_3120_17class/
├── images/          # {copied_images} file ảnh .jpg
├── labels/          # {copied_labels} file nhãn .txt (YOLO format: class_id x_center y_center w h)
├── classes.txt      # 17 tên lớp theo thứ tự class_id từ 0 đến 16
├── manifest.json    # Báo cáo chi tiết số lượng instance và phân bổ
└── README.md        # Tài liệu hướng dẫn này
```
"""
    (DEST_UNSPLIT_DIR / "README.md").write_text(readme_text, encoding="utf-8")


def build_cvat_task_package():
    print("\n==================================================================")
    print("📦 BƯỚC 2: TẠO GÓI CVAT TASK SẴN SÀNG UPLOAD (CVAT YOLO 1.1 PACKAGE)")
    print(f"🎯 Đích: {DEST_CVAT_DIR}")
    print("==================================================================")

    if DEST_CVAT_DIR.exists():
        shutil.rmtree(DEST_CVAT_DIR)
    DEST_CVAT_DIR.mkdir(parents=True, exist_ok=True)

    images_dir = DEST_UNSPLIT_DIR / "images"
    labels_dir = DEST_UNSPLIT_DIR / "labels"
    image_files = sorted(list(images_dir.glob("*.jpg")))

    # 1. Nén images.zip
    images_zip_path = DEST_CVAT_DIR / "images.zip"
    print(f"📸 1. Đang nén {len(image_files)} ảnh vào {images_zip_path.name}...")
    with zipfile.ZipFile(images_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_img:
        for idx, img in enumerate(image_files, 1):
            zip_img.write(img, arcname=img.name)
            if idx % 500 == 0 or idx == len(image_files):
                print(f"   - Đã nén {idx}/{len(image_files)} ảnh...")

    print(f"   💾 Kích thước images.zip: {images_zip_path.stat().st_size / (1024*1024):.2f} MB")

    # 2. Tạo annotations.zip theo chuẩn CVAT YOLO 1.1
    # Cấu trúc:
    # annotations.zip
    # ├── obj.data
    # ├── obj.names
    # ├── train.txt
    # └── obj_train_data/
    #     └── *.txt
    print("\n🏷️ 2. Đang đóng gói annotations.zip (CVAT YOLO 1.1 format)...")
    temp_anno = DEST_CVAT_DIR / "_temp_yolo11"
    if temp_anno.exists():
        shutil.rmtree(temp_anno)
    obj_train_data = temp_anno / "obj_train_data"
    obj_train_data.mkdir(parents=True, exist_ok=True)

    class_list = [UNIFIED_CLASSES[i] for i in sorted(UNIFIED_CLASSES.keys())]

    # obj.names
    with open(temp_anno / "obj.names", "w", encoding="utf-8") as f:
        f.write("\n".join(class_list) + "\n")

    # obj.data
    with open(temp_anno / "obj.data", "w", encoding="utf-8") as f:
        f.write(f"classes = {len(class_list)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")

    # train.txt & copy labels
    train_lines = []
    for img in image_files:
        lbl_file = labels_dir / f"{img.stem}.txt"
        dest_lbl = obj_train_data / f"{img.stem}.txt"
        train_lines.append(f"data/obj_train_data/{img.name}")
        if lbl_file.exists():
            shutil.copy2(lbl_file, dest_lbl)
        else:
            dest_lbl.touch()

    with open(temp_anno / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    anno_zip_path = DEST_CVAT_DIR / "annotations.zip"
    with zipfile.ZipFile(anno_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_anno:
        for root, dirs, files in os.walk(temp_anno):
            for file in files:
                abs_p = Path(root) / file
                rel_p = abs_p.relative_to(temp_anno)
                zip_anno.write(abs_p, arcname=str(rel_p))

    shutil.rmtree(temp_anno)
    print(f"   💾 Kích thước annotations.zip: {anno_zip_path.stat().st_size / 1024:.2f} KB")

    # 3. Tạo file cvat_labels_schema.json
    cvat_schema = []
    for idx, cname in enumerate(class_list):
        color = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]
        cvat_schema.append({
            "name": cname,
            "color": color,
            "type": "rectangle",
            "attributes": []
        })

    schema_file = DEST_CVAT_DIR / "cvat_labels_schema.json"
    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(cvat_schema, f, indent=2, ensure_ascii=False)
    print(f"   🎨 Đã tạo cvat_labels_schema.json với 17 nhãn màu tương phản cao!")

    # 4. Tạo README_CVAT_SETUP.md
    labels_quick_list = "\n".join([f"- **`{i}`: `{name}`**" for i, name in enumerate(class_list)])
    readme_cvat = f"""# 🏥 HƯỚNG DẪN CHI TIẾT NẠP TOÀN BỘ 3,120 FRAMES VÀO CVAT ĐỂ CHECK GROUND TRUTH

Thư mục gói dữ liệu CVAT:
`{DEST_CVAT_DIR}`

---

## 🎯 17 LỚP ĐỐI TƯỢNG (HOSPITAL UNIFIED TAXONOMY)
{labels_quick_list}

---

## 🚀 QUY TRÌNH 3 BƯỚC NẠP DỮ LIỆU VÀO CVAT

### BƯỚC 1: TẠO TASK TRÊN CVAT
1. Mở trình duyệt truy cập CVAT (trên `http://localhost:8080` hoặc `https://app.cvat.ai`).
2. Nhấn nút **"+ Create new task"**.
3. Điền thông tin Task:
   - **Name**: `Hospital_Unified_3120_17Class_Audit`
4. Cấu hình Nhãn (Labels):
   - Chuyển sang tab **"Raw"**.
   - Mở file [`cvat_labels_schema.json`](./cvat_labels_schema.json) trong thư mục này, copy toàn bộ nội dung JSON và dán vào ô nhập liệu.
   - Bấm **"Done"** (CVAT sẽ tạo tự động 17 nhãn kèm mã màu trực quan phân biệt).
5. Tải ảnh lên (**Select files**):
   - Kéo và thả file [`images.zip`](./images.zip) ({images_zip_path.stat().st_size / (1024*1024):.1f} MB) vào vùng upload.
6. Nhấn nút **"Submit & Open"** (CVAT xử lý giải nén và tạo task trong 10-20 giây).

---

### BƯỚC 2: NẠP TOÀN BỘ 7,903 BOUNDING BOXES NHẬP SẴN
Thay vì phải vẽ tay từ con số 0, toàn bộ 3,120 khung hình đã có sẵn 7,903 bounding boxes ground truth chất lượng cao. Bạn chỉ cần:
1. Tại trang chi tiết Task vừa tạo trên CVAT, nhấn vào biểu tượng menu **"..."** (góc phải trên) hoặc nút **"Actions"**.
2. Chọn **"Upload Annotations"** $\rightarrow$ chọn định dạng **"YOLO 1.1"**.
3. Chọn file [`annotations.zip`](./annotations.zip) trong thư mục này và tải lên.
4. Thông báo nạp thành công xuất hiện. Giờ bạn chỉ cần bấm mở **Job #...** để bắt đầu rà soát!

---

### BƯỚC 3: MẸO VÀ PHÍM TẮT CHECK TAY NHANH TRÊN CVAT (5-10s/frame)
- **`F` / `D`** hoặc **`Space`**: Tiến tới khung hình tiếp theo / Lùi lại khung hình trước.
- **`N`**: Bật chế độ vẽ Bounding Box mới (nếu phát hiện vật thể chưa được gán nhãn).
- **Phím số `0` - `9`** hoặc gõ tên nhãn: Chọn nhanh lớp đối tượng khi vẽ.
- **`Click vào box` + `Delete`**: Xóa các bounding box bị nhiễu, nhầm lẫn hoặc che khuất quá 80%.
- **`Ctrl + S`**: Lưu công việc (CVAT cũng hỗ trợ auto-save định kỳ).

---

## 📥 BƯỚC 4: XUẤT DỮ LIỆU ĐÃ DUYỆT (EXPORT SAU KHI CHECK XONG)
Khi bạn đã kiểm duyệt xong toàn bộ hoặc một phần:
1. Trong màn hình Task, bấm **Menu** (góc trái trên) $\rightarrow$ chọn **"Export Task Dataset"**.
2. Chọn định dạng **"YOLO 1.1"**.
3. Tích chọn **"Save Images"** (hoặc chỉ tải annotations zip).
4. Tải file ZIP về máy.
5. Sau đó thông báo lại cho trợ lý AI: "Tôi đã duyệt xong file zip trên CVAT, đường dẫn là ...", AI sẽ tự động kích hoạt pipeline chia Train/Val/Test chống rò rỉ dữ liệu và tiến hành huấn luyện mô hình siêu chuẩn!
"""
    (DEST_CVAT_DIR / "README_CVAT_SETUP.md").write_text(readme_cvat, encoding="utf-8")
    print(f"   📖 Đã tạo README_CVAT_SETUP.md hướng dẫn chi tiết!")

    print("\n==================================================================")
    print("✨ TẤT CẢ TÀI NGUYÊN ĐÃ SẴN SÀNG:")
    print(f"1. Thư mục dataset phẳng: {DEST_UNSPLIT_DIR}")
    print(f"2. Gói nạp CVAT:         {DEST_CVAT_DIR}")
    print("==================================================================")


if __name__ == "__main__":
    consolidate_unsplit_dataset()
    build_cvat_task_package()
