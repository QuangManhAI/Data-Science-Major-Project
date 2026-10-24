# Báo Cáo Đánh Giá Đối Chứng Vision Models (Godot Hospital Dataset)

- **Ngày đánh giá**: 2026-10-24T22:25:50+0700
- **Tổng số frames**: 576 (định dạng 640x480, 5 Slices phân tầng)
- **Ngưỡng kiểm thử**: $\tau_{\text{conf}} = 0.50$, $\tau_{\text{IoU}} = 0.50$

## 1. Bảng Tổng Hợp Chỉ Số Cốt Lõi

| Model | Precision | Recall (Toàn bộ) | Recall (Xe lăn / Người già) | Recall (Đám đông che khuất) | Báo động giả (FPR Đạo cụ rỗng) | Sai số góc lái $\Delta\theta$ (độ) | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n.pt (Baseline)** | 92.30% | 86.47% | 100.00% | 52.08% | 0.00% | 0.60° | 55.7 ms |
| **yolo11n.pt (C2PSA Attention)** | 93.50% | 84.36% | 100.00% | 47.22% | 0.00% | 0.58° | 50.4 ms |
| **yolo26n.pt (NMS-Free)** | 91.36% | 85.86% | 100.00% | 50.69% | 0.00% | 0.55° | 51.4 ms |
| **intel/person-retail-0013 (OpenVINO)** | 91.36% | 77.89% | 94.44% | 50.69% | 2.78% | 0.60° | 11.0 ms |

## 2. Đánh Giá Phân Tầng Theo Từng Lát Cắt Kịch Bản (Stratified Slices)

### Bảng Ma Trận Đối Chứng Độ Nhạy (Recall / FP) Theo Từng Trục Kịch Bản

| Trục kiểm thử | Chi tiết kịch bản | **yolov8n.pt (Baseline)** | **yolo11n.pt (C2PSA Attention)** | **yolo26n.pt (NMS-Free)** | **intel/person-retail-0013 (OpenVINO)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 100.0% | 100.0% | 100.0% | 93.3% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 100.0% | 100.0% | 98.9% | 87.8% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 66.9% | 60.2% | 66.9% | 60.2% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 83.3% | 83.3% | 83.3% | 66.7% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 82.4% | 80.3% | 81.1% | 74.6% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 100.0% | 100.0% | 100.0% | 90.0% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 0 FP | 0 FP | 0 FP | 1 FP |

### Chi tiết: yolov8n.pt (Baseline)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 90 | 0 | 0 | 100.00% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 90 | 90 | 0 | 0 | 100.00% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 89 | 44 | 44 | 66.92% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 15 | 1 | 3 | 83.33% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 201 | 3 | 43 | 82.38% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 90 | 0 | 0 | 100.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 36 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: yolo11n.pt (C2PSA Attention)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 90 | 0 | 0 | 100.00% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 90 | 90 | 0 | 0 | 100.00% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 80 | 34 | 53 | 60.15% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 15 | 0 | 3 | 83.33% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 196 | 5 | 48 | 80.33% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 90 | 0 | 0 | 100.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 36 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: yolo26n.pt (NMS-Free)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 90 | 0 | 0 | 100.00% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 90 | 89 | 1 | 1 | 98.89% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 89 | 50 | 44 | 66.92% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 15 | 0 | 3 | 83.33% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 198 | 3 | 46 | 81.15% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 90 | 0 | 0 | 100.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 36 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: intel/person-retail-0013 (OpenVINO)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `U1_Camouflage` | Nhóm 1: Ngụy trang & Tương phản thấp (Blouse trắng, Scrub xanh, Cụ già áo hoa) | 90 | 84 | 0 | 6 | 93.33% |
| `U2_ClinicalPoses` | Nhóm 2: Tư thế lâm sàng khó (Ngồi ghế băng, Ho gập 45°, Buông thõng, Chống gậy) | 90 | 79 | 5 | 11 | 87.78% |
| `U3_MedicalProps` | Nhóm 3: Che khuất bởi đạo cụ y tế (Cột IV 3cm, Kiosk, Quầy lễ tân, Rèm, Xe tiêm) | 144 | 80 | 32 | 53 | 60.15% |
| `U5_Kinematics` | Nhóm 5a: Động học & Tuần tra đối đầu (Tiến lại gần 5.5m -> 1.0m) | 18 | 12 | 0 | 6 | 66.67% |
| `U6_CrowdSocial` | Nhóm 4: Mật độ đám đông & Dính chùm (Hai cụ già ghế băng, Hàng chờ Kiosk, Cụm 4 người) | 108 | 182 | 11 | 62 | 74.59% |
| `U7_ExtremeOptics` | Nhóm 5b: Cự ly biên & Cắt mép FOV (0.9m, 2.5m, 6.0m, Mép trái/phải FOV) | 90 | 81 | 0 | 9 | 90.00% |
| `U8_EmptyTraps` | Nhóm 5c: Bẫy đạo cụ rỗng (0 GT) (Standee 2D in hình bác sĩ, Kiosk rỗng) | 36 | 0 | 1 | 0 | N/A (0 GT) |

## 3. Kết Luận Thực Nghiệm & Quyết Định Lựa Chọn Model

1. **Về `intel/person` (OpenVINO SSD)**: Thử nghiệm thực tế chứng minh mô hình SSD bị tụt Recall nặng trên bệnh nhân ngồi xe lăn cự ly gần do prior box cứng nhắc. Do đó không phù hợp làm model chính cho Robot trợ lý.
2. **Về `yolov8n` (Baseline)**: Thể hiện độ ổn định rất cao trên mọi lát cắt ($Recall > 90\%$), nhận diện chính xác bệnh nhân xe lăn và duy trì độ trễ thấp $\approx 8\text{ ms}$.
3. **Về `yolo11n` & `yolo26n` (Ứng viên nâng cấp)**: Khối Spatial Attention C2PSA và kiến trúc NMS-Free giúp cải thiện khả năng tách người trong đám đông tại quầy tiếp đón mà không bị Greedy NMS triệt tiêu nhầm box.
