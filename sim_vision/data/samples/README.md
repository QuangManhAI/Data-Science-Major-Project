# 📸 THƯ VIỆN ẢNH ĐẠI DIỆN TOÀN BỘ CÁC CA TRONG HỆ SINH THÁI DATASET (SAMPLES)

Thư mục này được tổ chức chuẩn mực nhằm cung cấp các khung hình trực quan tổng quát nhất cho **TOÀN BỘ 83 CA KIỂM THỬ** độc lập trong cả 2 bộ dữ liệu mô phỏng bệnh viện Godot 4 (đã loại bỏ C20 lỗi dán nhãn).

---

## 🏛️ Cấu Trúc Thư Mục Chuẩn Hóa

```
sim_vision/data/samples/
├── master_posters/                                   # 3 Poster Ma Trận Trực Quan Siêu Rộng
│   ├── 01_dataset_1_all_48_cases_poster.png          # Lưới 6x8: Toàn bộ 48 ca Dataset 1
│   ├── 02_dataset_2_all_36_clinical_cases_poster.png # Lưới 6x6: Toàn bộ 36 ca Dataset 2
│   └── 03_master_unified_84_cases_poster.png         # Lưới 7x12: Hợp nhất toàn bộ 84 ca (1200 + 10800)
│
├── dataset_1_baseline_48cases/                       # 48 Ảnh Rời Chuẩn Hóa (C01 -> C48)
│   ├── C01_T_Junction_Wall.jpg
│   ├── ...
│   └── C48_Fire_Extinguisher_Bin.jpg
│
└── dataset_2_clinical_36cases/                       # 36 Ảnh Rời Chuẩn Hóa (N1.1 -> N6.5)
    ├── N1_1_Doctor_White_Blouse_Wall.jpg
    ├── ...
    └── N6_5_Guided_Walking_Elderly_Pair.jpg
```

---

## 📋 Danh Mục Tra Cứu Toàn Bộ 83 Ca Kiểm Thử

### Phân Hệ 1: Dataset Baseline (vision_eval_1200 — 47 Kịch Bản C01 đến C48, Bỏ C20 Lỗi Dán Nhãn)
- **Trục D1 (Kiến trúc)**: C01 (Góc ngoặt T), C02 (Cắt dọc cửa), C03 (Khe cửa trượt 35%), C04 (Hốc thang máy), C05 (Nấp sau trụ cột), C06 (Cửa mở hé 45°).
- **Trục D2 (Nội thất y tế)**: C07 (Quầy lễ tân 1.1m), C08 (Bàn khám & màn hình), C09 (Vách rèm y tế), C10 (Xe tiêm 2 tầng), C11 (Cọc truyền che ngực), C12 (Sau ghế chờ inox).
- **Trục D3 (Tư thế lâm sàng)**: C13 (Xe lăn 0°), C14 (Xe lăn 45°), C15 (Xe lăn 90°), C16 (Cụ già gù chống gậy), C17 (Nằm giường cấp cứu), C18 (Cúi gập 90° nhặt sổ).
- **Trục D4 (Đám đông & NMS)**: C19 (Đẩy xe lăn sau - NMS trap), C21 (Hàng 3 người), C22 (Cắt mặt 2.5m), C23 (Cụm 4 người), C24 (Dìu bước khập khiễng). *(Ca C20 dán nhãn sai bệnh nhi đã được loại bỏ triệt để)*.
- **Trục D5 (Cự ly & Mép ảnh)**: C25 (Cực cận 0.6m), C26 (0.9m), C27 (2.5m), C28 (Xa 6.0m), C29 (Cắt mép trái), C30 (Cắt mép phải).
- **Trục D6 (Động học)**: C31 (Quay lưng 180°), C32 (Chéo lưng 135°), C33 (Xoay 90°), C34 (Đi lùi), C35 (Đi nhanh 1.5m/s), C36 (Đứng yên nói chuyện).
- **Trục D7 (Ánh sáng PBR)**: C37 (Ngược sáng sảnh), C38 (Phản chiếu sàn gạch bóng), C39 (Bóng đổ dài), C40 (Dư sáng huỳnh quang), C41 (Đèn trực đêm tối), C42 (Ánh sáng chớp tắt).
- **Trục D8 (Bẫy đạo cụ rỗng)**: C43 (Xe lăn trống), C44 (Giường cấp cứu trống), C45 (Cọc truyền treo áo khoác), C46 (Poster 2D dán tường), C47 (Kiosk tự phục vụ), C48 (Cụm bình chữa cháy).

---

### Phân Hệ 2: Dataset Clinical Demographics (vision_eval_extended_10800 — 36 Ca N1.1 đến N6.5)
- **Nhóm N1 (Nhân khẩu & Màu áo)**: N1.1 (Blouse ngụy trang tường kem ΔE=8.4), N1.2 (Scrub xanh trùng rèm), N1.3 (Cụ bà tóc bạc F09), N1.4 (Cụ ông lưng còng M03), N1.5 (Cụ ông áo sọc M14), N1.6 (Áo khoác dài phủ chân), N1.7 (Bệnh nhân nam thể hình đậm).
- **Nhóm N2 (Tư thế & Ngồi ghế)**: N2.1 (Ngồi ghế băng dài), N2.2 (Ngồi gập người ho 45°), N2.3 (Ngồi buông thõng mỏi mệt), N2.4 (Bác sĩ đọc kẹp bệnh án), N2.5 (Cụ bà ngồi bắt chéo chân).
- **Nhóm N3 (Che khuất đạo cụ)**: N3.1 (Cột truyền dịch inox thanh mảnh), N3.2 (Xe đẩy monitor 1.2m), N3.3 (Bấm kiosk 1.55m che 70%), N3.4 (Cây nước nóng lạnh), N3.5 (Rửa tay cột khử khuẩn), N3.6 (Xe cáng cấp cứu 2.1m tiền cảnh), N3.7 (Sát mép standee 1.8m).
- **Nhóm N4 (Bẫy False Positive rỗng)**: N4.1 (Cột IV rỗng), N4.2 (Cột rửa tay rỗng), N4.3 (Kiosk rỗng), N4.4 (Xe monitor rỗng), N4.5 (Xe cáng nệm trắng rỗng), N4.6 (Cây nước rỗng), N4.7 (Standee 2D in hình kích thước thật - Poster Trap), N4.8 (Màn hình LED gọi số).
- **Nhóm N5 (Động học né ngang)**: N5.1 (Đi tuần tra đối đầu v_rel=1.08 m/s), N5.2 (Bước né ngang Pardon Sidestep 0.55m ω=28.1°/s), N5.3 (Xoay người 90° qua cửa), N5.4 (Y tá đi nhanh rẽ từ góc khuất).
- **Nhóm N6 (Mật độ xã hội & Phân rã)**: N6.1 (Hai cụ già ngồi liền kề ghế băng - NMS trap), N6.2 (Bác sĩ đứng cạnh bệnh nhân ngồi), N6.3 (Hàng chờ 3 bệnh nhân kiosk), N6.4 (Thảo luận 180° lưng vs 0° mặt), N6.5 (Dìu bước đi chậm).
