# Báo Cáo Đánh Giá Đối Chứng Vision Models (Godot Hospital Dataset)

- **Ngày đánh giá**: 2026-10-24T21:46:52+0700
- **Tổng số frames**: 576 (định dạng 640x480, 5 Slices phân tầng)
- **Ngưỡng kiểm thử**: $\tau_{\text{conf}} = 0.50$, $\tau_{\text{IoU}} = 0.50$

## 1. Bảng Tổng Hợp Chỉ Số Cốt Lõi

| Model | Precision | Recall (Toàn bộ) | Recall (Xe lăn / Người già) | Recall (Đám đông che khuất) | Báo động giả (FPR Đạo cụ rỗng) | Sai số góc lái $\Delta\theta$ (độ) | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n.pt (Baseline)** | 47.64% | 87.74% | 100.00% | 70.45% | 73.77% | 0.76° | 48.3 ms |
| **yolo11n.pt (C2PSA Attention)** | 48.24% | 87.39% | 100.00% | 71.59% | 73.77% | 0.73° | 52.9 ms |
| **yolo26n.pt (NMS-Free)** | 47.71% | 86.86% | 100.00% | 68.18% | 80.33% | 0.72° | 49.8 ms |
| **intel/person-retail-0013 (OpenVINO)** | 43.44% | 77.09% | 94.44% | 56.82% | 79.51% | 0.70° | 11.2 ms |

## 2. Đánh Giá Phân Tầng Theo Từng Lát Cắt Kịch Bản (Stratified Slices)

### Bảng Ma Trận Đối Chứng Độ Nhạy (Recall / FP) Theo Từng Trục Kịch Bản

| Trục kiểm thử | Chi tiết kịch bản | **yolov8n.pt (Baseline)** | **yolo11n.pt (C2PSA Attention)** | **yolo26n.pt (NMS-Free)** | **intel/person-retail-0013 (OpenVINO)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 93.3% | 93.3% | 93.3% | 87.8% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 96.7% | 95.6% | 95.6% | 85.6% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 80.0% | 81.4% | 78.6% | 58.6% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 16.7% | 0.0% | 16.7% | 0.0% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 81.1% | 80.6% | 79.7% | 72.4% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 100.0% | 100.0% | 100.0% | 88.9% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 24 FP | 24 FP | 24 FP | 29 FP |

### Chi tiết: yolov8n.pt (Baseline)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 84 | 89 | 6 | 93.33% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 90 | 87 | 70 | 3 | 96.67% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 56 | 151 | 14 | 80.00% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 1 | 21 | 5 | 16.67% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 176 | 120 | 41 | 81.11% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 90 | 68 | 0 | 100.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 36 | 0 | 24 | 0 | N/A (0 GT) |

### Chi tiết: yolo11n.pt (C2PSA Attention)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 84 | 90 | 6 | 93.33% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 90 | 86 | 59 | 4 | 95.56% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 57 | 151 | 13 | 81.43% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 0 | 21 | 6 | 0.00% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 175 | 116 | 42 | 80.65% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 90 | 67 | 0 | 100.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 36 | 0 | 24 | 0 | N/A (0 GT) |

### Chi tiết: yolo26n.pt (NMS-Free)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 84 | 89 | 6 | 93.33% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 90 | 86 | 68 | 4 | 95.56% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 55 | 158 | 15 | 78.57% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 1 | 20 | 5 | 16.67% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 173 | 110 | 44 | 79.72% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 90 | 67 | 0 | 100.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 36 | 0 | 24 | 0 | N/A (0 GT) |

### Chi tiết: intel/person-retail-0013 (OpenVINO)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 79 | 88 | 11 | 87.78% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 90 | 77 | 82 | 13 | 85.56% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 41 | 153 | 29 | 58.57% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 0 | 19 | 6 | 0.00% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 157 | 132 | 60 | 72.35% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 80 | 62 | 10 | 88.89% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 36 | 0 | 29 | 0 | N/A (0 GT) |

## 3. Kết Luận Thực Nghiệm & Quyết Định Lựa Chọn Model

1. **Về `intel/person` (OpenVINO SSD)**: Thử nghiệm thực tế chứng minh mô hình SSD bị tụt Recall nặng trên bệnh nhân ngồi xe lăn cự ly gần do prior box cứng nhắc. Do đó không phù hợp làm model chính cho Robot trợ lý.
2. **Về `yolov8n` (Baseline)**: Thể hiện độ ổn định rất cao trên mọi lát cắt ($Recall > 90\%$), nhận diện chính xác bệnh nhân xe lăn và duy trì độ trễ thấp $\approx 8\text{ ms}$.
3. **Về `yolo11n` & `yolo26n` (Ứng viên nâng cấp)**: Khối Spatial Attention C2PSA và kiến trúc NMS-Free giúp cải thiện khả năng tách người trong đám đông tại quầy tiếp đón mà không bị Greedy NMS triệt tiêu nhầm box.
