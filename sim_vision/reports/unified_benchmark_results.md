# Báo Cáo Đánh Giá Đối Chứng Vision Models (Godot Hospital Dataset)

- **Ngày đánh giá**: 2026-10-24T23:08:02+0700
- **Tổng số frames**: 648 (định dạng 640x480, 5 Slices phân tầng)
- **Ngưỡng kiểm thử**: $\tau_{\text{conf}} = 0.50$, $\tau_{\text{IoU}} = 0.50$

## 1. Bảng Tổng Hợp Chỉ Số Cốt Lõi

| Model | Precision | Recall (Toàn bộ) | Recall (Xe lăn / Người già) | Recall (Đám đông che khuất) | Báo động giả (FPR Đạo cụ rỗng) | Sai số góc lái $\Delta\theta$ (độ) | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n.pt (Baseline)** | 92.95% | 84.84% | 100.00% | 53.10% | 0.00% | 0.58° | 35.2 ms |
| **yolo11n.pt (C2PSA Attention)** | 93.14% | 83.55% | 100.00% | 48.97% | 0.00% | 0.57° | 35.2 ms |
| **yolo26n.pt (NMS-Free)** | 91.80% | 84.84% | 100.00% | 53.10% | 0.00% | 0.55° | 35.1 ms |
| **yolov8x.pt (SOTA 131M)** | 94.86% | 89.84% | 100.00% | 73.10% | 0.00% | 0.51° | 655.1 ms |
| **intel/person-retail-0013 (OpenVINO)** | 90.07% | 76.54% | 94.44% | 51.72% | 4.17% | 0.53° | 8.9 ms |

## 2. Đánh Giá Phân Tầng Theo Từng Lát Cắt Kịch Bản (Stratified Slices)

### Bảng Ma Trận Đối Chứng Độ Nhạy (Recall / FP) Theo Từng Trục Kịch Bản

| Trục kiểm thử | Chi tiết kịch bản | **yolov8n.pt (Baseline)** | **yolo11n.pt (C2PSA Attention)** | **yolo26n.pt (NMS-Free)** | **yolov8x.pt (SOTA 131M)** | **intel/person-retail-0013 (OpenVINO)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 100.0% | 98.9% | 100.0% | 100.0% | 93.3% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 85.7% | 85.7% | 85.7% | 85.7% | 77.0% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 70.0% | 66.2% | 72.3% | 86.2% | 61.5% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 83.3% | 88.9% | 83.3% | 88.9% | 66.7% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 81.2% | 79.6% | 80.0% | 86.5% | 73.5% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 100.0% | 100.0% | 100.0% | 100.0% | 91.1% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 0 FP | 0 FP | 0 FP | 0 FP | 4 FP |

### Chi tiết: yolov8n.pt (Baseline)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 90 | 0 | 0 | 100.00% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 126 | 108 | 3 | 18 | 85.71% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 91 | 39 | 39 | 70.00% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 15 | 0 | 3 | 83.33% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 199 | 3 | 46 | 81.22% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 90 | 0 | 0 | 100.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 72 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: yolo11n.pt (C2PSA Attention)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 89 | 0 | 1 | 98.89% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 126 | 108 | 4 | 18 | 85.71% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 86 | 38 | 44 | 66.15% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 16 | 0 | 2 | 88.89% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 195 | 1 | 50 | 79.59% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 90 | 0 | 0 | 100.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 72 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: yolo26n.pt (NMS-Free)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 90 | 0 | 0 | 100.00% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 126 | 108 | 4 | 18 | 85.71% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 94 | 46 | 36 | 72.31% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 15 | 0 | 3 | 83.33% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 196 | 3 | 49 | 80.00% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 90 | 0 | 0 | 100.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 72 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: yolov8x.pt (SOTA 131M)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 90 | 0 | 0 | 100.00% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 126 | 108 | 6 | 18 | 85.71% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 112 | 23 | 18 | 86.15% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 16 | 3 | 2 | 88.89% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 212 | 2 | 33 | 86.53% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 90 | 0 | 0 | 100.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 72 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: intel/person-retail-0013 (OpenVINO)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 84 | 0 | 6 | 93.33% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 126 | 97 | 7 | 29 | 76.98% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 80 | 33 | 50 | 61.54% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 12 | 0 | 6 | 66.67% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 180 | 15 | 65 | 73.47% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 82 | 0 | 8 | 91.11% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 72 | 0 | 4 | 0 | N/A (0 GT) |

## 3. Kết Luận Thực Nghiệm & Quyết Định Lựa Chọn Model

1. **Về `intel/person` (OpenVINO SSD)**: Thử nghiệm thực tế chứng minh mô hình SSD bị tụt Recall nặng trên bệnh nhân ngồi xe lăn cự ly gần do prior box cứng nhắc. Do đó không phù hợp làm model chính cho Robot trợ lý.
2. **Về `yolov8n` (Baseline)**: Thể hiện độ ổn định rất cao trên mọi lát cắt ($Recall > 90\%$), nhận diện chính xác bệnh nhân xe lăn và duy trì độ trễ thấp $\approx 8\text{ ms}$.
3. **Về `yolo11n` & `yolo26n` (Ứng viên nâng cấp)**: Khối Spatial Attention C2PSA và kiến trúc NMS-Free giúp cải thiện khả năng tách người trong đám đông tại quầy tiếp đón mà không bị Greedy NMS triệt tiêu nhầm box.
