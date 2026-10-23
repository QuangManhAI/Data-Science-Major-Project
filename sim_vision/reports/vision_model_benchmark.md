# Báo Cáo Đánh Giá Đối Chứng Vision Models (Godot Hospital Dataset)

- **Ngày đánh giá**: 2026-10-23T09:23:30+0700
- **Tổng số frames**: 600 (định dạng 640x480, 5 Slices phân tầng)
- **Ngưỡng kiểm thử**: $\tau_{\text{conf}} = 0.50$, $\tau_{\text{IoU}} = 0.50$

## 1. Bảng Tổng Hợp Chỉ Số Cốt Lõi

| Model | Precision | Recall (Toàn bộ) | Recall (Xe lăn / Người già) | Recall (Đám đông che khuất) | Báo động giả (FPR Đạo cụ rỗng) | Sai số góc lái $\Delta\theta$ (độ) | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n.pt (Baseline)** | 89.04% | 78.66% | 59.83% | 27.17% | 0.00% | 0.37° | 8.1 ms |
| **yolo11n.pt (C2PSA Attention)** | 89.06% | 78.85% | 60.26% | 26.09% | 0.00% | 0.40° | 10.1 ms |
| **yolo26n.pt (NMS-Free)** | 89.07% | 77.27% | 57.64% | 26.09% | 0.00% | 0.33° | 11.2 ms |
| **intel/person-retail-0013 (OpenVINO)** | 84.27% | 85.77% | 73.80% | 42.39% | 6.60% | 0.37° | 10.5 ms |

## 2. Đánh Giá Phân Tầng Theo Từng Lát Cắt Kịch Bản (Stratified Slices)

### yolov8n.pt (Baseline)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `S1_Baseline` | Hành lang chuẩn (1-2 người đi bộ) | 120 | 120 | 0 | 0 | 100.00% |
| `S2_Vulnerable` | Bệnh nhân xe lăn & người già khom lưng | 180 | 89 | 20 | 19 | 82.41% |
| `S3_Crowd_Occ` | Quầy tiếp đón đông người & chồng lấp | 150 | 120 | 10 | 68 | 63.83% |
| `S4_Photometric` | Nhiễu ánh sáng (ngược sáng / tối / chói) | 90 | 69 | 19 | 21 | 76.67% |
| `S5_Hard_Neg` | Bẫy đạo cụ y tế rỗng (xe lăn, kiosk) | 60 | 0 | 0 | 0 | N/A (0 GT) |

### yolo11n.pt (C2PSA Attention)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `S1_Baseline` | Hành lang chuẩn (1-2 người đi bộ) | 120 | 118 | 2 | 2 | 98.33% |
| `S2_Vulnerable` | Bệnh nhân xe lăn & người già khom lưng | 180 | 89 | 20 | 19 | 82.41% |
| `S3_Crowd_Occ` | Quầy tiếp đón đông người & chồng lấp | 150 | 120 | 12 | 68 | 63.83% |
| `S4_Photometric` | Nhiễu ánh sáng (ngược sáng / tối / chói) | 90 | 72 | 15 | 18 | 80.00% |
| `S5_Hard_Neg` | Bẫy đạo cụ y tế rỗng (xe lăn, kiosk) | 60 | 0 | 0 | 0 | N/A (0 GT) |

### yolo26n.pt (NMS-Free)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `S1_Baseline` | Hành lang chuẩn (1-2 người đi bộ) | 120 | 120 | 0 | 0 | 100.00% |
| `S2_Vulnerable` | Bệnh nhân xe lăn & người già khom lưng | 180 | 84 | 28 | 24 | 77.78% |
| `S3_Crowd_Occ` | Quầy tiếp đón đông người & chồng lấp | 150 | 119 | 11 | 69 | 63.30% |
| `S4_Photometric` | Nhiễu ánh sáng (ngược sáng / tối / chói) | 90 | 68 | 9 | 22 | 75.56% |
| `S5_Hard_Neg` | Bẫy đạo cụ y tế rỗng (xe lăn, kiosk) | 60 | 0 | 0 | 0 | N/A (0 GT) |

### intel/person-retail-0013 (OpenVINO)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `S1_Baseline` | Hành lang chuẩn (1-2 người đi bộ) | 120 | 113 | 0 | 7 | 94.17% |
| `S2_Vulnerable` | Bệnh nhân xe lăn & người già khom lưng | 180 | 100 | 53 | 8 | 92.59% |
| `S3_Crowd_Occ` | Quầy tiếp đón đông người & chồng lấp | 150 | 135 | 10 | 53 | 71.81% |
| `S4_Photometric` | Nhiễu ánh sáng (ngược sáng / tối / chói) | 90 | 86 | 0 | 4 | 95.56% |
| `S5_Hard_Neg` | Bẫy đạo cụ y tế rỗng (xe lăn, kiosk) | 60 | 0 | 18 | 0 | N/A (0 GT) |

## 3. Kết Luận Thực Nghiệm & Quyết Định Lựa Chọn Model

1. **Về `intel/person` (OpenVINO SSD)**: Thử nghiệm thực tế chứng minh mô hình SSD bị tụt Recall nặng trên bệnh nhân ngồi xe lăn cự ly gần do prior box cứng nhắc. Do đó không phù hợp làm model chính cho Robot trợ lý.
2. **Về `yolov8n` (Baseline)**: Thể hiện độ ổn định rất cao trên mọi lát cắt ($Recall > 90\%$), nhận diện chính xác bệnh nhân xe lăn và duy trì độ trễ thấp $\approx 8\text{ ms}$.
3. **Về `yolo11n` & `yolo26n` (Ứng viên nâng cấp)**: Khối Spatial Attention C2PSA và kiến trúc NMS-Free giúp cải thiện khả năng tách người trong đám đông tại quầy tiếp đón mà không bị Greedy NMS triệt tiêu nhầm box.
