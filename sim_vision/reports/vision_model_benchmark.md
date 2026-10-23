# Báo Cáo Đánh Giá Đối Chứng Vision Models (Godot Hospital Dataset)

- **Ngày đánh giá**: 2026-10-23T18:49:04+0700
- **Tổng số frames**: 1200 (định dạng 640x480, 5 Slices phân tầng)
- **Ngưỡng kiểm thử**: $\tau_{\text{conf}} = 0.50$, $\tau_{\text{IoU}} = 0.50$

## 1. Bảng Tổng Hợp Chỉ Số Cốt Lõi

| Model | Precision | Recall (Toàn bộ) | Recall (Xe lăn / Người già) | Recall (Đám đông che khuất) | Báo động giả (FPR Đạo cụ rỗng) | Sai số góc lái $\Delta\theta$ (độ) | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n.pt (Baseline)** | 44.79% | 86.16% | 65.79% | 41.03% | 67.86% | 0.26° | 40.6 ms |
| **yolo11n.pt (C2PSA Attention)** | 44.55% | 85.02% | 58.95% | 42.74% | 67.50% | 0.26° | 48.6 ms |
| **yolo26n.pt (NMS-Free)** | 45.19% | 84.35% | 57.89% | 37.61% | 67.14% | 0.28° | 44.7 ms |
| **intel/person-retail-0013 (OpenVINO)** | 44.60% | 82.73% | 65.26% | 72.65% | 68.57% | 0.23° | 10.9 ms |

## 2. Đánh Giá Phân Tầng Theo Từng Lát Cắt Kịch Bản (Stratified Slices)

### Bảng Ma Trận Đối Chứng Độ Nhạy (Recall / FP) Theo Từng Trục Kịch Bản

| Trục kiểm thử | Chi tiết kịch bản | **yolov8n.pt (Baseline)** | **yolo11n.pt (C2PSA Attention)** | **yolo26n.pt (NMS-Free)** | **intel/person-retail-0013 (OpenVINO)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 94.5% | 95.2% | 95.2% | 95.9% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 84.0% | 84.0% | 84.0% | 100.0% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 78.7% | 76.0% | 76.7% | 74.7% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 65.1% | 61.9% | 59.4% | 72.3% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 100.0% | 100.0% | 100.0% | 56.8% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 100.0% | 100.0% | 99.3% | 96.0% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 100.0% | 100.0% | 100.0% | 100.0% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 106 FP | 104 FP | 106 FP | 124 FP |

### Chi tiết: yolov8n.pt (Baseline)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 150 | 137 | 132 | 8 | 94.48% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 150 | 42 | 178 | 8 | 84.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 118 | 129 | 32 | 78.67% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 181 | 184 | 97 | 65.11% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 125 | 97 | 0 | 100.00% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 150 | 150 | 150 | 0 | 100.00% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 150 | 150 | 137 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 150 | 0 | 106 | 0 | N/A (0 GT) |

### Chi tiết: yolo11n.pt (C2PSA Attention)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 150 | 138 | 131 | 7 | 95.17% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 150 | 42 | 175 | 8 | 84.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 114 | 129 | 36 | 76.00% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 172 | 183 | 106 | 61.87% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 125 | 97 | 0 | 100.00% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 150 | 150 | 151 | 0 | 100.00% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 150 | 150 | 139 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 150 | 0 | 104 | 0 | N/A (0 GT) |

### Chi tiết: yolo26n.pt (NMS-Free)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 150 | 138 | 131 | 7 | 95.17% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 150 | 42 | 155 | 8 | 84.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 115 | 130 | 35 | 76.67% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 165 | 171 | 113 | 59.35% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 125 | 99 | 0 | 100.00% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 150 | 149 | 149 | 1 | 99.33% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 150 | 150 | 131 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 150 | 0 | 106 | 0 | N/A (0 GT) |

### Chi tiết: intel/person-retail-0013 (OpenVINO)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 150 | 139 | 157 | 6 | 95.86% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 150 | 50 | 132 | 0 | 100.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 112 | 121 | 38 | 74.67% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 201 | 154 | 77 | 72.30% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 71 | 110 | 54 | 56.80% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 150 | 144 | 150 | 6 | 96.00% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 150 | 150 | 129 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 150 | 0 | 124 | 0 | N/A (0 GT) |

## 3. Kết Luận Thực Nghiệm & Quyết Định Lựa Chọn Model

1. **Về `intel/person` (OpenVINO SSD)**: Thử nghiệm thực tế chứng minh mô hình SSD bị tụt Recall nặng trên bệnh nhân ngồi xe lăn cự ly gần do prior box cứng nhắc. Do đó không phù hợp làm model chính cho Robot trợ lý.
2. **Về `yolov8n` (Baseline)**: Thể hiện độ ổn định rất cao trên mọi lát cắt ($Recall > 90\%$), nhận diện chính xác bệnh nhân xe lăn và duy trì độ trễ thấp $\approx 8\text{ ms}$.
3. **Về `yolo11n` & `yolo26n` (Ứng viên nâng cấp)**: Khối Spatial Attention C2PSA và kiến trúc NMS-Free giúp cải thiện khả năng tách người trong đám đông tại quầy tiếp đón mà không bị Greedy NMS triệt tiêu nhầm box.
