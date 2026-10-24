# 🏥 HỆ THỐNG DỮ LIỆU ĐÁNH GIÁ THỊ GIÁC ROBOT BỆNH VIỆN (SIM_VISION DATASETS)

Thư mục này trỏ trực tiếp (thông qua symlink chuẩn) về phân hệ dữ liệu thị giác chuyên biệt nằm tại thư mục gốc của dự án:
👉 [`data/vision/`](file:///home/bush/Projects/Data-Science-Major-Project/data/vision)

---

## 📂 Danh Mục Thư Mục Trong `data/vision/`

### 1. `data/vision/vision_eval_unified/` — Unified Gold Standard Dataset (648 Frames)
Bộ dữ liệu chuẩn hóa hợp nhất toàn bộ 36 ca kiểm thử lâm sàng trọng yếu:
- **Quy mô**: **648 khung hình** ($640 \times 480$ RGB JPEG) và **648 file nhãn YOLO đa lớp**.
- **Hệ thống phân lớp (Multi-Class)**:
  - `0: person` — Người (Bác sĩ, Y tá, Bệnh nhân, Cụ già).
  - `1: wheelchair` — Xe lăn bệnh viện (Hỗ trợ phát hiện kép người ngồi xe lăn và cảnh báo an toàn).
- **Cấu trúc lưu trữ**:
  - `images/`: Chứa 648 ảnh định dạng JPEG chất lượng cao.
  - `labels/`: Chứa 648 nhãn YOLO đa lớp chuẩn hóa `[class_id x_center y_center width height]`.
  - `annotations.jsonl`: File metadata chuẩn phục vụ pipeline đánh giá tự động (`eval_vision_models.py`).
  - `annotations_rich_5w1h.jsonl`: File metadata khoa học 7 phân hệ 5W1H (tọa độ 6-DoF, khoảng cách Euclid 3D, độ rọi Lux, góc ngắm camera).
  - `data.yaml`: File cấu hình dữ liệu chuẩn của Ultralytics YOLO.
  - `manifest_unified.json`: Bảng kê chi tiết phân loại 36 ca kiểm thử theo 7 nhóm kịch bản.

### 2. `data/vision/vision_eval_groundtruth_visualized/` — Visual Inspection Frames (648 Frames)
Bộ ảnh trực quan hóa hỗ trợ kỹ sư kiểm tra đối chứng trực tiếp:
- **Khung hình chữ nhật kép**:
  - 🟩 **Xanh lá cây (Green Box)**: Bounding box Ground Truth (Zero-Defect).
  - 🟦 **Xanh dương (Blue Box)**: Bounding box dự đoán của mô hình AI (`yolov8n`).
- **Phân nhóm theo 7 thư mục chuyên đề**:
  - `U1_Camouflage/`: Bác sĩ áo blouse trắng tiệp màu tường, Y tá scrub xanh trùng màu rèm, Cụ già áo hoa.
  - `U2_ClinicalPoses/`: Tư thế lâm sàng phức tạp (Ngồi ghế băng dài, Ho gập người $45^\circ$, Buông thõng mệt mỏi, Đọc bệnh án, Bệnh nhân nằm trên giường bệnh cấp cứu đủ mọi góc ngắm $360^\circ$).
  - `U3_MedicalProps/`: Che khuất bởi đạo cụ y tế (Cột truyền dịch inox thanh mảnh, Máy monitor sinh hiệu, Kiosk tự phục vụ, Cây nước, Cột khử khuẩn, Xe cáng).
  - `U5_Kinematics/`: Động học chuyển động tuần tra đối đầu cự ly biến thiên từ $5.5\text{m} \to 1.0\text{m}$.
  - `U6_CrowdSocial/`: Mật độ đám đông & Dính chùm (Hai cụ già ngồi liền kề, Hàng chờ kiosk 3 người, Cụm giao tiếp 4 người, Dìu bước khập khiễng).
  - `U7_ExtremeOptics/`: Cự ly biên cực hạn ($0.9\text{m}$, $2.5\text{m}$, $6.0\text{m}$) và cắt mép tầm nhìn Camera FOV.
  - `U8_EmptyTraps/`: Bẫy đạo cụ rỗng (Ground Truth 0 người) nhằm đo lường báo động giả (Standee 2D in hình bác sĩ kích thước thật, Kiosk rỗng, Giường trống).

---

## 📊 Báo Cáo & Công Cụ Kèm Theo

- **Báo cáo chuyên sâu trực quan các ca mạnh/yếu nhất**:
  [`sim_vision/reports/report_best_and_weakest_cases_visualization.md`](../reports/report_best_and_weakest_cases_visualization.md)
- **Báo cáo benchmark đối chứng 5 mô hình (YOLOv8n, YOLO11n, YOLO26n, YOLOv8x, Intel OpenVINO)**:
  [`sim_vision/reports/unified_benchmark_results.md`](../reports/unified_benchmark_results.md)
- **Scripts thực thi**:
  - Tạo Ground Truth: [`sim_vision/python/generate_zero_defect_groundtruth.py`](../python/generate_zero_defect_groundtruth.py)
  - Vẽ khung trực quan hóa: [`sim_vision/python/render_visualized_groundtruth.py`](../python/render_visualized_groundtruth.py)
  - Benchmark mô hình: [`sim_vision/python/eval_vision_models.py`](../python/eval_vision_models.py)
  - Tạo biểu đồ & poster: [`sim_vision/python/generate_visualization_report_assets.py`](../python/generate_visualization_report_assets.py)
