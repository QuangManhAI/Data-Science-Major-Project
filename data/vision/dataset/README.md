# 🏥 Danh Mục Các Bộ Dataset Thị Giác Máy Tính (Vision Datasets Catalog)

Thư mục `data/vision/dataset/` là kho lưu trữ trung tâm các bộ dữ liệu thị giác phục vụ huấn luyện và đánh giá mô hình YOLO26n cho Robot Trợ Lý Y Tế.

Mỗi bộ dataset được phân tách thành một **thư mục độc lập** (isolated subfolder) kèm cấu trúc chuẩn YOLO, manifest định danh, và tài liệu phân loại (classification taxonomy) để phục vụ việc mở rộng nhiều bộ dataset trong tương lai mà không bị lẫn lộn dữ liệu.

---

## 📂 Danh Sách Các Bộ Dataset Hiện Có

| Tên Bộ Dataset | Quy Mô | Số Lượng Hộp Nhãn | Số Lớp (Classes) | Trạng Thái / Mục Tiêu |
| :--- | :---: | :---: | :---: | :--- |
| [`hospital_gold_1387_3class/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/dataset/hospital_gold_1387_3class) | **1,387 frames** | **1,753 boxes** | 3 (`patient`, `doctor`, `wheelchair`) | **Gold Standard** - 100% positive frames lâm sàng đã qua CVAT human verification. |

---

## 🛠️ Quy Chuẩn Lưu Trữ Dataset Mới (Future Datasets)

Khi tạo hoặc nạp thêm các bộ dataset mới (ví dụ: dataset camera góc trần, dataset camera hồng ngoại ban đêm, dataset phát hiện té ngã, v.v.), hãy tuân thủ cấu trúc thư mục sau:

```text
data/vision/dataset/
├── README.md                                  # Danh mục tổng hợp tất cả datasets
├── hospital_gold_1387_3class/                 # Bộ dataset 1k3 hiện tại (người & xe lăn)
│   ├── images/                                # 1,387 file ảnh JPG
│   ├── labels/                                # 1,387 file nhãn TXT (chuẩn YOLO 3-class)
│   ├── manifest.json                          # Metadata & phiên bản dataset
│   └── README.md                              # Tài liệu chi tiết dataset
└── <future_dataset_name>/                     # Dataset mới tiếp theo
    ├── images/
    ├── labels/
    ├── manifest.json
    └── README.md
```

---

## 🔗 Liên Kết Liên Quan
- Dữ liệu thô và bẫy rỗng (Hard Negatives): [`data/vision/raw/hard_negatives/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/raw/hard_negatives/)
- File cấu hình YOLO: [`configs/hospital_3class.yaml`](file:///home/bush/Projects/Data-Science-Major-Project/configs/hospital_3class.yaml)
- Script kiểm tra chất lượng dữ liệu: [`scripts/data_pipeline/data_quality_checks.py`](file:///home/bush/Projects/Data-Science-Major-Project/scripts/data_pipeline/data_quality_checks.py)
