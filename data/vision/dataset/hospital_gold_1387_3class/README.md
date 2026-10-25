# 🏥 Dataset Card: `hospital_gold_1387_3class` (Core 1,313 Frames)

Bộ dữ liệu thị giác lâm sàng chuẩn vàng (Gold Standard) tập trung 100% vào phát hiện **Người (Bệnh nhân & Bác sĩ) và Xe lăn** trong môi trường bệnh viện thông minh.

---

## 📌 Tổng Quan Bộ Dữ Liệu

- **Tên dataset**: `hospital_gold_1387_3class` (`Hospital_Vision_Gold_Core_Positive`)
- **Phiên bản**: `2.2_pedestrian_wheelchair_only`
- **Số lượng khung hình (frames)**: **1,313 ảnh** (`.jpg`, độ phân giải 640x480)
- **Tổng số bounding box**: **1,679 instances**
- **Độ tinh khiết**: **100% positive frames** chỉ chứa người đứng/ngồi và xe lăn y tế:
  - 133 frames bẫy rỗng (standee, kiosk rỗng, giường trống, frames khuất 100%) lưu trữ tại [`data/vision/raw/hard_negatives/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/raw/hard_negatives/).
  - 74 frames bệnh nhân nằm giường bệnh (U2.8 và U2.11) lưu trữ tại [`data/vision/raw/hospital_bed_cases/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision/raw/hospital_bed_cases/).
- **Nguồn gốc kiểm định**: Sinh mô phỏng vật lý 3D từ Godot 4 + Duyệt và chuẩn hóa thủ công trên CVAT (Human-in-the-loop verification).

---

## 🎯 Hệ Thống 3 Lớp Đối Tượng (3 Core Classes)

| Class ID | Tên Lớp | Số Lượng Hộp | Tỷ Lệ (%) | Đối Tượng Lâm Sàng Bao Gồm |
| :---: | :---: | :---: | :---: | :--- |
| **`0`** | `patient` | **1,168** | **69.57%** | Bệnh nhân người lớn, cụ già chống gậy/gù lưng, bệnh nhi 4 tuổi, bệnh nhân ngồi băng ghế chờ, bệnh nhân ho rũ người. |
| **`1`** | `doctor` | **435** | **25.91%** | Bác sĩ mặc áo blouse trắng, điều dưỡng mặc đồ scrub xanh ngọc, bác sĩ cầm hồ sơ bệnh án clipboard, bác sĩ/điều dưỡng đi tuần tra hành lang. |
| **`2`** | `wheelchair` | **76** | **4.53%** | Xe lăn y tế tiêu chuẩn (cả có người bệnh nhân ngồi trên xe lăn và xe lăn rỗng dừng chờ). |

---

## 📁 Cấu Trúc Thư Mục

```text
hospital_gold_1387_3class/
├── README.md                          # Tài liệu tổng quan bộ dataset này
├── manifest.json                      # Metadata chính thức của dataset
├── images/                            # 1,313 file ảnh sạch
└── labels/                            # 1,313 file nhãn YOLO (class_id xc yc w h)
```

---

## 📊 Phân Bố Theo Nhóm Lâm Sàng (35 Positive Cases)

1. **Ngụy trang & Tương phản thấp (U1)**: 5 ca (190 frames) — Blouse trắng trùng tường, scrub xanh trùng rèm, áo hoa văn, áo tối màu.
2. **Tư thế lâm sàng đa dạng (U2)**: 7 ca (266 frames) — Ngồi ghế tựa, gập người ho 45°, buông thõng mỏi mệt, cụ già chống gậy, ngồi xe lăn, trẻ em. *(2 ca nằm giường U2.8, U2.11 đã chuyển sang raw/hospital_bed_cases/)*.
3. **Che khuất bởi đạo cụ y tế (U3)**: 8 ca (292 frames) — Khuất cột truyền dịch, cây bấm số kiosk, xe monitor, quầy lễ tân, vách rèm, góc ngoặt hành lang, ngưỡng cửa phòng bệnh.
4. **Động học & Di chuyển tuần tra (U5)**: 1 ca (31 frames) — Bác sĩ tuần tra đối đầu trực diện robot ở tốc độ 1.08 m/s.
5. **Mật độ đám đông & Tương tác xã hội (U6)**: 8 ca (304 frames) — 2 cụ già trò chuyện, bác sĩ tư vấn bệnh nhân, hàng đợi kiosk 3 người, cặp đôi thảo luận, điều dưỡng dìu cụ già, mẹ dắt con, cụm 4 người, mẹ bế em bé.
6. **Quang học & Mép biên FOV (U7)**: 5 ca (190 frames) — Cận cảnh an toàn 0.9m, chuẩn 2.5m, xa 6.0m, cắt mép trái FOV, cắt mép phải FOV.
7. **Đạo cụ xe lăn y tế (U8)**: 1 ca (38 frames) — Xe lăn rỗng đỗ hành lang (phục vụ nhận diện phương tiện hỗ trợ di chuyển).
