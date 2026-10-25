# 🛏️ Kho Lưu Trữ Dữ Liệu Bệnh Nhân Nằm Giường (`hospital_bed_cases`)

Thư mục này lưu trữ **74 khung hình** (ảnh và nhãn) của các ca **Bệnh nhân nằm giường bệnh** (`U2.8` và `U2.11`) được bóc tách từ tập dữ liệu chính theo yêu cầu quản trị dữ liệu:

- **Mục đích tách ra**: Giữ cho bộ dữ liệu huấn luyện chính (`hospital_gold_...`) tập trung 100% vào **Người đi lại/đứng/ngồi ghế băng và Xe lăn y tế**, không bị lẫn các biến thể nằm ngang trên giường bệnh.
- **Dữ liệu được lưu trữ nguyên vẹn**: Cả ảnh gốc (`images/`) và nhãn YOLO (`labels/`) được bảo toàn đầy đủ 100% để phục vụ khi dự án mở rộng sang bài toán nhận diện Giường bệnh / Bệnh nhân nội trú.

---

## 📂 Danh Sách Các Ca Lâm Sàng Nằm Giường (74 Frames)

| Mã Ca | Tên Ca Lâm Sàng | Đối Tượng (Actor) | Số Frames | Số Boxes | Ghi Chú Kịch Bản |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **U2.8** | Bệnh Nhân Nằm Giường Cấp Cứu | `recumbent_bed_patient` | **37** | **37** | Bệnh nhân nằm ngửa trên giường cáng cấp cứu, camera chụp góc nghiêng đa hướng. |
| **U2.11** | Bệnh Nhân Nằm Giường Đa Góc Nhìn | `recumbent_bed_patient` | **37** | **37** | Bệnh nhân nằm trên đệm giường bệnh quan sát từ nhiều góc cao/thấp khác nhau. |
| **TỔNG** | **2 Ca Lâm Sàng** | — | **74** | **74** | **100% nhãn class 0 (`patient`).** |

*(Lưu ý: 2 khung hình `U2_8_0038` và `U2_11_0038` bị che khuất 100% bởi thành giường nằm trong kho [`data/vision/raw/hard_negatives/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/raw/hard_negatives/)).*

---

## 📁 Cấu Trúc Thư Mục

```text
data/vision/raw/hospital_bed_cases/
├── README.md                      # Tài liệu mô tả các ca nằm giường
├── manifest.json                  # Metadata chi tiết từng frame
├── images/                        # 74 file ảnh JPG (U2_8_0001 -> U2_11_0037)
└── labels/                        # 74 file nhãn TXT (chuẩn YOLO class 0)
```
