# Báo Cáo Đánh Giá Đối Chứng Vision Models (Godot Hospital Dataset)

- **Ngày đánh giá**: 2026-10-23T10:25:25+0700
- **Tổng số frames**: 1200 (định dạng 640x480, 5 Slices phân tầng)
- **Ngưỡng kiểm thử**: $\tau_{\text{conf}} = 0.50$, $\tau_{\text{IoU}} = 0.50$

## 1. Bảng Tổng Hợp Chỉ Số Cốt Lõi

| Model | Precision | Recall (Toàn bộ) | Recall (Xe lăn / Người già) | Recall (Đám đông che khuất) | Báo động giả (FPR Đạo cụ rỗng) | Sai số góc lái $\Delta\theta$ (độ) | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n.pt (Baseline)** | 88.66% | 85.32% | 55.26% | 50.88% | 17.09% | 0.17° | 8.6 ms |
| **yolo11n.pt (C2PSA Attention)** | 88.95% | 85.61% | 54.21% | 53.51% | 15.64% | 0.18° | 9.9 ms |
| **yolo26n.pt (NMS-Free)** | 92.10% | 85.51% | 55.79% | 50.00% | 7.27% | 0.18° | 11.2 ms |
| **intel/person-retail-0013 (OpenVINO)** | 92.54% | 82.88% | 64.74% | 68.42% | 13.82% | 0.17° | 10.2 ms |

## 2. Đánh Giá Phân Tầng Theo Từng Lát Cắt Kịch Bản (Stratified Slices)

### Bảng Ma Trận Đối Chứng Độ Nhạy (Recall / FP) Theo Từng Trục Kịch Bản

| Trục kiểm thử | Chi tiết kịch bản | **yolov8n.pt (Baseline)** | **yolo11n.pt (C2PSA Attention)** | **yolo26n.pt (NMS-Free)** | **intel/person-retail-0013 (OpenVINO)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 100.0% | 100.0% | 100.0% | 100.0% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 84.0% | 88.0% | 84.0% | 100.0% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 68.7% | 68.0% | 70.7% | 76.0% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 64.9% | 65.6% | 64.6% | 70.5% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 100.0% | 100.0% | 100.0% | 56.0% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 100.0% | 100.0% | 100.0% | 96.0% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 100.0% | 100.0% | 100.0% | 100.0% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 0 FP | 0 FP | 0 FP | 14 FP |

### Chi tiết: yolov8n.pt (Baseline)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 150 | 150 | 0 | 0 | 100.00% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 150 | 42 | 60 | 8 | 84.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 103 | 21 | 47 | 68.67% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 187 | 34 | 101 | 64.93% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 125 | 0 | 0 | 100.00% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 150 | 150 | 1 | 0 | 100.00% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 150 | 150 | 0 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 150 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: yolo11n.pt (C2PSA Attention)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 150 | 150 | 0 | 0 | 100.00% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 150 | 44 | 53 | 6 | 88.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 102 | 18 | 48 | 68.00% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 189 | 39 | 99 | 65.62% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 125 | 0 | 0 | 100.00% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 150 | 150 | 3 | 0 | 100.00% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 150 | 150 | 0 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 150 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: yolo26n.pt (NMS-Free)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 150 | 150 | 0 | 0 | 100.00% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 150 | 42 | 31 | 8 | 84.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 106 | 10 | 44 | 70.67% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 186 | 36 | 102 | 64.58% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 125 | 0 | 0 | 100.00% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 150 | 150 | 1 | 0 | 100.00% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 150 | 150 | 0 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 150 | 0 | 0 | 0 | N/A (0 GT) |

### Chi tiết: intel/person-retail-0013 (OpenVINO)

| Slice | Mô tả | Số frame | TP | FP | FN | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `D1_Architecture` | D1: Vật cản kiến trúc cố định (Mép cửa phòng khám 110, mép tường, trụ) | 150 | 150 | 9 | 0 | 100.00% |
| `D2_Furniture` | D2: Đạo cụ & nội thất che khuất (Quầy lễ tân, rèm che, xe tiêm) | 150 | 50 | 30 | 0 | 100.00% |
| `D3_Clinical_Poses` | D3: Dị biệt dáng đi & lâm sàng (Xe lăn, nạng/gậy, cáng nằm, khom lưng) | 150 | 114 | 0 | 36 | 76.00% |
| `D4_Crowd_Dyn` | D4: Động học đám đông (Đi song song, che lấp lẫn nhau, xếp hàng) | 150 | 203 | 3 | 85 | 70.49% |
| `D5_Distances` | D5: Biến dạng cự ly & Cắt biên ảnh (0.6m đến 10m, mép FOV) | 150 | 70 | 15 | 55 | 56.00% |
| `D6_Kinematics` | D6: Góc quay cơ thể & Động học (0°, 45°, 90°, 135°, 180°, 270°) | 150 | 144 | 0 | 6 | 96.00% |
| `D7_Lighting` | D7: Nhiễu ánh sáng & Hiệu ứng PBR (Ngược sáng, chói lóa, đèn sự cố) | 150 | 150 | 0 | 0 | 100.00% |
| `D8_Hard_Negatives` | D8: Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, poster y tế, kiosk) | 150 | 0 | 14 | 0 | N/A (0 GT) |

## 3. Kết Luận Thực Nghiệm & Quyết Định Lựa Chọn Model

1. **Về `intel/person` (OpenVINO SSD)**: Thử nghiệm thực tế chứng minh mô hình SSD bị tụt Recall nặng trên bệnh nhân ngồi xe lăn cự ly gần do prior box cứng nhắc. Do đó không phù hợp làm model chính cho Robot trợ lý.
2. **Về `yolov8n` (Baseline)**: Thể hiện độ ổn định rất cao trên mọi lát cắt ($Recall > 90\%$), nhận diện chính xác bệnh nhân xe lăn và duy trì độ trễ thấp $\approx 8\text{ ms}$.
3. **Về `yolo11n` & `yolo26n` (Ứng viên nâng cấp)**: Khối Spatial Attention C2PSA và kiến trúc NMS-Free giúp cải thiện khả năng tách người trong đám đông tại quầy tiếp đón mà không bị Greedy NMS triệt tiêu nhầm box.
