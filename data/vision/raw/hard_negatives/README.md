# 🛑 Kho Lưu Trữ Khung Hình Bẫy & Mẫu Âm Tính (Hard Negatives)

Thư mục này lưu trữ **133 khung hình âm tính (0 ground-truth box)** được bóc tách từ tập dữ liệu gốc 1,520 frames của Godot 4, nhằm:
1. Giữ cho bộ dữ liệu huấn luyện chính [`data/vision/dataset/hospital_gold_1387_3class/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/dataset/hospital_gold_1387_3class/) đạt độ tinh khiết **100% positive frames** (chỉ chứa người và xe lăn).
2. Lưu trữ có hệ thống cả ảnh (`images/`) và nhãn rỗng (`labels/`) để làm **tập kiểm thử chống nhận diện nhầm (False Positive Stress-Testing)** hoặc bổ sung mẫu âm tính có kiểm soát (Hard Negative Mining) khi huấn luyện.

---

## 📂 Phân Loại 133 Khung Hình Âm Tính

| Nhóm | Mã Ca | Tên Ca Lâm Sàng | Số Frames | Lý Do Phân Loại Vào Hard Negatives |
| :---: | :---: | :--- | :---: | :--- |
| **Bẫy Đạo Cụ Rỗng (Empty Traps)** | **U8.8** | Bẫy Standee 2D In Hình Y Tế | 38 | Standee in hình người phẳng 2D. Tách ra để huấn luyện chống nhận diện nhầm tranh ảnh/standee thành người thật. |
| | **U8.4** | Bẫy Kiosk Bấm Số Rỗng | 38 | Trụ kiosk lấy số thứ tự khi không có người đứng bấm. |
| | **U8.6** | Bẫy Giường Bệnh Rỗng | 38 | Giường bệnh viện trải drap trắng không có bệnh nhân nằm. |
| **Khuất Hoàn Toàn / Ngoài FOV** | **U2.8** | Bệnh Nhân Nằm Giường Cấp Cứu | 1 | Khung hình bệnh nhân bị thành giường che khuất 100%. |
| | **U2.11** | Bệnh Nhân Nằm Giường Đa Góc Nhìn | 1 | Khung hình góc quay khuất hoàn toàn thân bệnh nhân. |
| | **U3.2** | Khuất Sau Xe Monitor Sinh Hiệu | 6 | Bác sĩ bị xe đẩy monitor che phủ hoàn toàn 100%. |
| | **U3.9** | Khuất Sau Vách Rèm Y Tế | 4 | Bệnh nhân đứng sau rèm kín 100%. |
| | **U5.1** | Tuần Tra Đối Đầu (1.08 m/s) | 7 | Bác sĩ đi tuần tra đã di chuyển vượt ra khỏi mép khung nhìn camera (Out of FOV). |
| **TỔNG CỘNG** | — | **8 Ca Lâm Sàng** | **133** | **Tất cả file nhãn `.txt` tương ứng đều có dung lượng 0 byte (0 box).** |

---

## 📁 Cấu Trúc Thư Mục

```text
data/vision/raw/hard_negatives/
├── README.md                      # Tài liệu thuyết minh mục đích và phân loại
├── negatives_manifest.json        # Chỉ mục machine-readable của 133 frames
├── images/                        # 133 file ảnh JPG
└── labels/                        # 133 file nhãn rỗng (0 byte TXT)
```
