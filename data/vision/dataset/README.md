# 🏥 Danh Mục Các Bộ Dataset Thị Giác Máy Tính (Vision Datasets Catalog)

Thư mục `data/vision/dataset/` là kho lưu trữ trung tâm các bộ dữ liệu thị giác phục vụ huấn luyện và đánh giá mô hình YOLO26n cho Robot Trợ Lý Y Tế.

Mỗi bộ dataset được phân tách thành một **thư mục độc lập** (isolated subfolder) kèm cấu trúc chuẩn YOLO, manifest định danh, và tài liệu phân loại (classification taxonomy) để phục vụ việc mở rộng nhiều bộ dataset trong tương lai mà không bị lẫn lộn dữ liệu.

---

## 📂 Danh Sách Các Bộ Dataset Hiện Có

| Tên Bộ Dataset | Quy Mô | Số Lớp (Classes) | Trạng Thái / Mục Tiêu |
| :--- | :---: | :---: | :--- |
| [`hospital_gold_1387_3class/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/dataset/hospital_gold_1387_3class) | **1,313 frames** (1,679 boxes) | 3 (`patient`, `doctor`, `wheelchair`) | **Gold Standard** - 100% positive frames người đi lại/đứng/ngồi và xe lăn đã qua CVAT human verification. |
| [`hospital_props_1600_13class/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/dataset/hospital_props_1600_13class) | **1,600 frames** (~3,800 boxes) | 13 (`queue_kiosk`, `reception_counter`, `iv_stand`, `hospital_bed`, ...) | **Equipment & Props Standard** - Trích xuất tự động từ 3D Physics Mesh trong Godot 4 phục vụ dẫn đường và tránh va chạm. |

---

## 🛠️ Quy Chuẩn Lưu Trữ Dataset (Multi-Dataset Catalog)

```text
data/vision/dataset/
├── README.md                                  # Danh mục tổng hợp tất cả datasets
├── hospital_gold_1387_3class/                 # Bộ dataset 1.3k người & xe lăn
│   ├── images/                                # 1,313 file ảnh JPG
│   ├── labels/                                # 1,313 file nhãn TXT (3 classes)
│   ├── manifest.json                          # Metadata & phiên bản dataset v2.2
│   └── README.md                              # Tài liệu chi tiết dataset
├── hospital_props_1600_13class/               # Bộ dataset 1.6k trang thiết bị y tế
│   ├── images/                                # 1,600 file ảnh JPG
│   ├── labels/                                # 1,600 file nhãn TXT (13 classes)
│   ├── manifest.json                          # Metadata & chỉ mục tọa độ 3D
│   └── README.md                              # Dataset card chi tiết 13 món đồ
└── <future_dataset_name>/                     # Dataset mới tiếp theo
    ├── images/
    ├── labels/
    ├── manifest.json
    └── README.md
```

---

## 🔗 Liên Kết Liên Quan
- Dữ liệu thô bệnh nhân nằm giường (Bed Cases): [`data/vision/raw/hospital_bed_cases/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/raw/hospital_bed_cases/)
- Dữ liệu thô và bẫy rỗng (Hard Negatives): [`data/vision/raw/hard_negatives/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/raw/hard_negatives/)
- File cấu hình YOLO: [`configs/hospital_3class.yaml`](file:///home/bush/Projects/Data-Science-Major-Project/configs/hospital_3class.yaml)
- Script kiểm tra chất lượng dữ liệu: [`scripts/data_pipeline/data_quality_checks.py`](file:///home/bush/Projects/Data-Science-Major-Project/scripts/data_pipeline/data_quality_checks.py)
