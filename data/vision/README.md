# 🏥 QUY CHUẨN QUẢN TRỊ DỮ LIỆU THỊ GIÁC (VISION DATA MANAGEMENT)

Kiến trúc lưu trữ dữ liệu thị giác được tinh gọn theo nguyên tắc **Zero Duplication** (không nhân bản file ảnh, master images duy nhất), phân tách rõ ràng giữa dữ liệu thô sinh từ mô phỏng, dữ liệu chuẩn vàng đã qua kiểm định CVAT, và mẫu âm tính:

```text
data/vision/
├── dataset/                                   # 🌟 KHO CHỨA CÁC BỘ DATASET ĐÃ KIỂM ĐỊNH
│   ├── README.md                              # Danh mục tổng hợp tất cả datasets
│   └── hospital_gold_1387_3class/             # Bộ dataset 1.3k hiện tại (100% positive frames)
│       ├── README.md                          # Thẻ mô tả dataset (Dataset Card)
│       ├── manifest.json                      # Metadata phiên bản v2.2
│       ├── images/                            # 1,313 ảnh JPG chuẩn 640x480
│       └── labels/                            # 1,313 file nhãn TXT (chuẩn YOLO 3 classes)
│
├── raw/                                       # 📦 DỮ LIỆU THÔ & METADATA GỐC TỪ GODOT
│   ├── annotations.jsonl                      # Metadata không gian 3D chi tiết (cự ly, toạ độ, góc quay, độ che khuất)
│   ├── annotations_standard.jsonl             # Nhãn tiêu chuẩn YOLO trích xuất tự động
│   ├── manifest_unified.json                  # Báo cáo sinh dữ liệu mô phỏng từ Godot
│   ├── pre_annotation_manifest.json           # Log kiểm toán trước thẩm định CVAT
│   ├── hospital_bed_cases/                    # 74 frames bệnh nhân nằm giường (U2.8, U2.11)
│   │   ├── README.md
│   │   ├── manifest.json
│   │   ├── images/                            # 74 ảnh nằm giường
│   │   └── labels/                            # 74 file nhãn class 0
│   └── hard_negatives/                        # 133 khung hình âm tính (0 ground truth box)
│       ├── README.md                          # Thuyết minh mục đích kiểm thử chống False Positive
│       ├── negatives_manifest.json            # Chỉ mục machine-readable của 133 frames bẫy
│       ├── images/                            # 133 ảnh (standee, kiosk rỗng, giường trống, frames khuất 100%)
│       └── labels/                            # 133 file nhãn rỗng (0 byte)
│
└── cvat_task_package/                         # 🛠️ GÓI GÁN NHÃN THẨM ĐỊNH TRÊN CVAT
    ├── README_CVAT_SETUP.md                   # Hướng dẫn setup task trên CVAT local/cloud
    ├── cvat_labels_schema.json                # Schema 3 lớp đối tượng chuẩn y tế
    ├── images.zip                             # Gói nén ảnh nạp vào CVAT Task
    ├── annotations.zip                        # Nhãn nháp chuyển đổi cho CVAT
    └── Hospital_Vision_Gold_1520.zip          # Bản export gốc đã qua con người thẩm định
```

---

## 🏷️ Quy Chuẩn 3 Lớp Đối Tượng Lâm Sàng (3 Core Classes)

1. **`0: patient`** (1,168 boxes / 69.57%):
   - Bệnh nhân người lớn, người già (chống gậy / lưng còng), bệnh nhi (4 tuổi).
   - Tư thế ngồi ghế băng chờ, ngồi ho gập người 45°, buông thõng mệt mỏi.
2. **`1: doctor`** (435 boxes / 25.91%):
   - Bác sĩ mặc áo blouse trắng tương phản thấp.
   - Điều dưỡng mặc trang phục scrub xanh ngọc.
   - Bác sĩ cầm bảng kẹp hồ sơ bệnh án clipboard, bác sĩ/điều dưỡng tuần tra đối đầu.
3. **`2: wheelchair`** (76 boxes / 4.53%):
   - Xe lăn y tế 4 bánh (cả trường hợp có bệnh nhân đang ngồi và xe lăn rỗng dừng đỗ cạnh tường).

*(Lưu ý: 133 frames bẫy rỗng lưu tại `data/vision/raw/hard_negatives/` và 74 frames bệnh nhân nằm giường lưu tại `data/vision/raw/hospital_bed_cases/` để bộ dataset chính tập trung 100% vào người đi lại/đứng/ngồi và xe lăn).*
