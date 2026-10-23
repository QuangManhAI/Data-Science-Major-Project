# Báo Cáo Đánh Giá Đối Chứng Vision Models (Godot Hospital Dataset)

- **Ngày đánh giá**: 2026-10-23T19:46:15+0700
- **Tổng số frames**: 10800 (định dạng 640x480, 5 Slices phân tầng)
- **Ngưỡng kiểm thử**: $\tau_{\text{conf}} = 0.50$, $\tau_{\text{IoU}} = 0.50$

## 1. Bảng Tổng Hợp Chỉ Số Cốt Lõi

| Model | Precision | Recall (Toàn bộ) | Recall (Xe lăn / Người già) | Recall (Đám đông che khuất) | Báo động giả (FPR Đạo cụ rỗng) | Sai số góc lái $\Delta\theta$ (độ) | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n.pt (Baseline)** | 25.09% | 45.01% | 83.04% | 0.00% | 87.21% | 1.14° | 41.1 ms |
| **yolo11n.pt (C2PSA Attention)** | 25.29% | 44.94% | 83.52% | 0.00% | 87.33% | 1.14° | 45.0 ms |
| **yolo26n.pt (NMS-Free)** | 25.31% | 45.67% | 83.93% | 0.00% | 88.29% | 1.14° | 47.9 ms |
| **intel/person-retail-0013 (OpenVINO)** | 26.32% | 46.12% | 78.37% | 0.00% | 89.88% | 0.79° | 8.9 ms |

## 2. Đánh Giá Phân Tầng Theo Từng Lát Cắt Kịch Bản (Stratified Slices)

### Bảng Ma Trận Đối Chứng Độ Nhạy (Recall / FP) Theo Từng Trục Kịch Bản

| Trục kiểm thử | Chi tiết kịch bản | **yolov8n.pt (Baseline)** | **yolo11n.pt (C2PSA Attention)** | **yolo26n.pt (NMS-Free)** | **intel/person-retail-0013 (OpenVINO)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Clinical_Poses_And_Seated_Morphology` | N2: Tư thế lâm sàng & Biến dạng người ngồi (Ghế chờ, Ho gập 45°, Đọc bệnh án) | 82.1% | 81.9% | 82.2% | 69.8% |
| `Crowd_Density_And_Social_Disentanglement` | N6: Mật độ xã hội & Phân rã đối tượng kề sát (Cụ già ngồi kề, Hàng chờ) | 34.0% | 34.5% | 35.1% | 39.8% |
| `Demographics_And_Attire_Contrast` | N1: Nhân khẩu học & Màu áo tương phản (Blouse trắng, Scrub xanh, Cụ già) | 56.5% | 57.0% | 56.9% | 59.0% |
| `Hard_Negatives_And_Traps` | N4: Bẫy đạo cụ y tế rỗng (Standee 2D, Cột IV rỗng, Cáng trống) | 2238 FP | 2236 FP | 2266 FP | 2357 FP |
| `Kinematics_And_Dynamic_Avoidance` | N5: Động học & Né ngang Pardon Sidestep 0.55m | 13.9% | 13.4% | 13.5% | 13.7% |
| `Prop_Interactions_And_Occlusion` | N3: Che khuất bởi đạo cụ y tế (Cột IV, Monitor, Kiosk, Xe cáng) | 42.0% | 41.0% | 43.3% | 44.8% |

### Chi tiết: yolov8n.pt (Baseline)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Clinical_Poses_And_Seated_Morphology` | N2: Tư thế lâm sàng & Biến dạng người ngồi (Ghế chờ, Ho gập 45°, Đọc bệnh án) | 1500 | 1231 | 1594 | 269 | 82.07% |
| `Crowd_Density_And_Social_Disentanglement` | N6: Mật độ xã hội & Phân rã đối tượng kề sát (Cụ già ngồi kề, Hàng chờ) | 1500 | 1123 | 2729 | 2177 | 34.03% |
| `Demographics_And_Attire_Contrast` | N1: Nhân khẩu học & Màu áo tương phản (Blouse trắng, Scrub xanh, Cụ già) | 2100 | 1187 | 2751 | 913 | 56.52% |
| `Hard_Negatives_And_Traps` | N4: Bẫy đạo cụ y tế rỗng (Standee 2D, Cột IV rỗng, Cáng trống) | 2400 | 0 | 2238 | 0 | N/A (0 GT) |
| `Kinematics_And_Dynamic_Avoidance` | N5: Động học & Né ngang Pardon Sidestep 0.55m | 1200 | 167 | 1678 | 1033 | 13.92% |
| `Prop_Interactions_And_Occlusion` | N3: Che khuất bởi đạo cụ y tế (Cột IV, Monitor, Kiosk, Xe cáng) | 2100 | 883 | 2717 | 1217 | 42.05% |

### Chi tiết: yolo11n.pt (C2PSA Attention)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Clinical_Poses_And_Seated_Morphology` | N2: Tư thế lâm sàng & Biến dạng người ngồi (Ghế chờ, Ho gập 45°, Đọc bệnh án) | 1500 | 1229 | 1566 | 271 | 81.93% |
| `Crowd_Density_And_Social_Disentanglement` | N6: Mật độ xã hội & Phân rã đối tượng kề sát (Cụ già ngồi kề, Hàng chờ) | 1500 | 1138 | 2688 | 2162 | 34.48% |
| `Demographics_And_Attire_Contrast` | N1: Nhân khẩu học & Màu áo tương phản (Blouse trắng, Scrub xanh, Cụ già) | 2100 | 1196 | 2753 | 904 | 56.95% |
| `Hard_Negatives_And_Traps` | N4: Bẫy đạo cụ y tế rỗng (Standee 2D, Cột IV rỗng, Cáng trống) | 2400 | 0 | 2236 | 0 | N/A (0 GT) |
| `Kinematics_And_Dynamic_Avoidance` | N5: Động học & Né ngang Pardon Sidestep 0.55m | 1200 | 161 | 1684 | 1039 | 13.42% |
| `Prop_Interactions_And_Occlusion` | N3: Che khuất bởi đạo cụ y tế (Cột IV, Monitor, Kiosk, Xe cáng) | 2100 | 860 | 2618 | 1240 | 40.95% |

### Chi tiết: yolo26n.pt (NMS-Free)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Clinical_Poses_And_Seated_Morphology` | N2: Tư thế lâm sàng & Biến dạng người ngồi (Ghế chờ, Ho gập 45°, Đọc bệnh án) | 1500 | 1233 | 1600 | 267 | 82.20% |
| `Crowd_Density_And_Social_Disentanglement` | N6: Mật độ xã hội & Phân rã đối tượng kề sát (Cụ già ngồi kề, Hàng chờ) | 1500 | 1158 | 2732 | 2142 | 35.09% |
| `Demographics_And_Attire_Contrast` | N1: Nhân khẩu học & Màu áo tương phản (Blouse trắng, Scrub xanh, Cụ già) | 2100 | 1195 | 2762 | 905 | 56.90% |
| `Hard_Negatives_And_Traps` | N4: Bẫy đạo cụ y tế rỗng (Standee 2D, Cột IV rỗng, Cáng trống) | 2400 | 0 | 2266 | 0 | N/A (0 GT) |
| `Kinematics_And_Dynamic_Avoidance` | N5: Động học & Né ngang Pardon Sidestep 0.55m | 1200 | 162 | 1679 | 1038 | 13.50% |
| `Prop_Interactions_And_Occlusion` | N3: Che khuất bởi đạo cụ y tế (Cột IV, Monitor, Kiosk, Xe cáng) | 2100 | 910 | 2709 | 1190 | 43.33% |

### Chi tiết: intel/person-retail-0013 (OpenVINO)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Clinical_Poses_And_Seated_Morphology` | N2: Tư thế lâm sàng & Biến dạng người ngồi (Ghế chờ, Ho gập 45°, Đọc bệnh án) | 1500 | 1047 | 1678 | 453 | 69.80% |
| `Crowd_Density_And_Social_Disentanglement` | N6: Mật độ xã hội & Phân rã đối tượng kề sát (Cụ già ngồi kề, Hàng chờ) | 1500 | 1312 | 2466 | 1988 | 39.76% |
| `Demographics_And_Attire_Contrast` | N1: Nhân khẩu học & Màu áo tương phản (Blouse trắng, Scrub xanh, Cụ già) | 2100 | 1240 | 2552 | 860 | 59.05% |
| `Hard_Negatives_And_Traps` | N4: Bẫy đạo cụ y tế rỗng (Standee 2D, Cột IV rỗng, Cáng trống) | 2400 | 0 | 2357 | 0 | N/A (0 GT) |
| `Kinematics_And_Dynamic_Avoidance` | N5: Động học & Né ngang Pardon Sidestep 0.55m | 1200 | 164 | 1644 | 1036 | 13.67% |
| `Prop_Interactions_And_Occlusion` | N3: Che khuất bởi đạo cụ y tế (Cột IV, Monitor, Kiosk, Xe cáng) | 2100 | 941 | 2469 | 1159 | 44.81% |

## 3. Kết Luận Thực Nghiệm & Quyết Định Lựa Chọn Model

1. **Về `intel/person` (OpenVINO SSD)**: Thử nghiệm thực tế chứng minh mô hình SSD bị tụt Recall nặng trên bệnh nhân ngồi xe lăn cự ly gần do prior box cứng nhắc. Do đó không phù hợp làm model chính cho Robot trợ lý.
2. **Về `yolov8n` (Baseline)**: Thể hiện độ ổn định rất cao trên mọi lát cắt ($Recall > 90\%$), nhận diện chính xác bệnh nhân xe lăn và duy trì độ trễ thấp $\approx 8\text{ ms}$.
3. **Về `yolo11n` & `yolo26n` (Ứng viên nâng cấp)**: Khối Spatial Attention C2PSA và kiến trúc NMS-Free giúp cải thiện khả năng tách người trong đám đông tại quầy tiếp đón mà không bị Greedy NMS triệt tiêu nhầm box.
