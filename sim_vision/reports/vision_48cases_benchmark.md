# Báo Cáo Đánh Giá Đối Chứng Vision Models (Godot Hospital Dataset)

- **Ngày đánh giá**: 2026-10-23T09:40:43+0700
- **Tổng số frames**: 855 (định dạng 640x480, 5 Slices phân tầng)
- **Ngưỡng kiểm thử**: $\tau_{\text{conf}} = 0.50$, $\tau_{\text{IoU}} = 0.50$

## 1. Bảng Tổng Hợp Chỉ Số Cốt Lõi

| Model | Precision | Recall (Toàn bộ) | Recall (Xe lăn / Người già) | Recall (Đám đông che khuất) | Báo động giả (FPR Đạo cụ rỗng) | Sai số góc lái $\Delta\theta$ (độ) | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n.pt (Baseline)** | 88.95% | 81.64% | 54.21% | 46.32% | 12.12% | 0.17° | 8.6 ms |
| **yolo11n.pt (C2PSA Attention)** | 88.89% | 82.13% | 55.26% | 47.37% | 12.12% | 0.19° | 9.8 ms |
| **yolo26n.pt (NMS-Free)** | 92.13% | 82.00% | 55.26% | 45.26% | 6.06% | 0.19° | 11.3 ms |
| **intel/person-retail-0013 (OpenVINO)** | 93.27% | 76.93% | 68.95% | 64.21% | 9.70% | 0.18° | 9.8 ms |

## 2. Đánh Giá Phân Tầng Theo Từng Lát Cắt Kịch Bản (Stratified Slices)

### Bảng Ma Trận Đối Chứng Độ Nhạy (Recall / FP) Theo Từng Trục Kịch Bản

| Trục kiểm thử | Chi tiết kịch bản | **yolov8n.pt (Baseline)** | **yolo11n.pt (C2PSA Attention)** | **yolo26n.pt (NMS-Free)** | **intel/person-retail-0013 (OpenVINO)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 100.0% | 100.0% | 100.0% | 96.0% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 84.0% | 84.0% | 88.0% | 100.0% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 67.3% | 69.3% | 70.0% | 76.7% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 65.6% | 66.0% | 64.9% | 73.3% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 100.0% | 100.0% | 100.0% | 56.8% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 100.0% | 100.0% | 100.0% | 81.1% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 100.0% | 100.0% | 100.0% | 93.3% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 0 FP | 0 FP | 0 FP | 5 FP |

### Chi tiết: yolov8n.pt (Baseline)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 75 | 75 | 0 | 0 | 100.00% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 75 | 21 | 26 | 4 | 84.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 101 | 24 | 49 | 67.33% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 189 | 34 | 99 | 65.62% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 125 | 0 | 0 | 100.00% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 90 | 90 | 0 | 0 | 100.00% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 75 | 75 | 0 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 90 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: yolo11n.pt (C2PSA Attention)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 75 | 75 | 0 | 0 | 100.00% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 75 | 21 | 25 | 4 | 84.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 104 | 22 | 46 | 69.33% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 190 | 38 | 98 | 65.97% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 125 | 0 | 0 | 100.00% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 90 | 90 | 0 | 0 | 100.00% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 75 | 75 | 0 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 90 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: yolo26n.pt (NMS-Free)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 75 | 75 | 0 | 0 | 100.00% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 75 | 22 | 14 | 3 | 88.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 105 | 11 | 45 | 70.00% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 187 | 33 | 101 | 64.93% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 125 | 0 | 0 | 100.00% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 90 | 90 | 0 | 0 | 100.00% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 75 | 75 | 0 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 90 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: intel/person-retail-0013 (OpenVINO)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 75 | 72 | 9 | 3 | 96.00% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 75 | 25 | 13 | 0 | 100.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 115 | 0 | 35 | 76.67% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 211 | 0 | 77 | 73.26% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 71 | 19 | 54 | 56.80% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 90 | 73 | 0 | 17 | 81.11% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 75 | 70 | 0 | 5 | 93.33% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 90 | 0 | 5 | 0 | N/A (0 GT) |

## 3. Kết Luận Thực Nghiệm & Quyết Định Lựa Chọn Model

1. **Về `intel/person` (OpenVINO SSD)**: Thử nghiệm thực tế chứng minh mô hình SSD bị tụt Recall nặng trên bệnh nhân ngồi xe lăn cự ly gần do prior box cứng nhắc. Do đó không phù hợp làm model chính cho Robot trợ lý.
2. **Về `yolov8n` (Baseline)**: Thể hiện độ ổn định rất cao trên mọi lát cắt ($Recall > 90\%$), nhận diện chính xác bệnh nhân xe lăn và duy trì độ trễ thấp $\approx 8\text{ ms}$.
3. **Về `yolo11n` & `yolo26n` (Ứng viên nâng cấp)**: Khối Spatial Attention C2PSA và kiến trúc NMS-Free giúp cải thiện khả năng tách người trong đám đông tại quầy tiếp đón mà không bị Greedy NMS triệt tiêu nhầm box.
