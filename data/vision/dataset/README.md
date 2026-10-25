# 🏥 Danh Mục Các Bộ Dataset Thị Giác Máy Tính (Vision Datasets Catalog)

Thư mục `data/vision/dataset/` là kho lưu trữ trung tâm các bộ dữ liệu thị giác phục vụ huấn luyện và đánh giá mô hình YOLO26n cho Robot Trợ Lý Y Tế.

Mỗi bộ dataset được phân tách thành một **thư mục độc lập** (isolated subfolder) kèm cấu trúc chuẩn YOLO, manifest định danh, và tài liệu phân loại (classification taxonomy) để phục vụ việc mở rộng nhiều bộ dataset trong tương lai mà không bị lẫn lộn dữ liệu.

---

## 📂 Danh Sách Các Bộ Dataset Hiện Có

| Tên Bộ Dataset | Quy Mô | Số Lượng Hộp Nhãn | Số Lớp (Classes) | Trạng Thái / Mục Tiêu |
| :--- | :---: | :---: | :---: | :--- |
| [`hospital_gold_1387_3class/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/dataset/hospital_gold_1387_3class) | **1,313 frames** | **1,679 boxes** | 3 (`patient`, `doctor`, `wheelchair`) | **Gold Standard** - 100% positive frames người đi lại/đứng/ngồi và xe lăn đã qua CVAT human verification. |

---

## 🛠️ Quy Chuẩn Lưu Trữ Dataset Mới (Future Datasets)

Khi tạo hoặc nạp thêm các bộ dataset mới (ví dụ: dataset camera góc trần, dataset camera hồng ngoại ban đêm, dataset phát hiện té ngã, v.v.), hãy tuân thủ cấu trúc thư mục sau:

```text
data/vision/dataset/
├── README.md                                  # Danh mục tổng hợp tất cả datasets
├── hospital_gold_1387_3class/                 # Bộ dataset 1.3k hiện tại (người & xe lăn)
│   ├── images/                                # 1,313 file ảnh JPG
│   ├── labels/                                # 1,313 file nhãn TXT (chuẩn YOLO 3-class)
│   ├── manifest.json                          # Metadata & phiên bản dataset v2.2
│   └── README.md                              # Tài liệu chi tiết dataset
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
