# 🏥 HỆ THỐNG DỮ LIỆU ĐÁNH GIÁ THỊ GIÁC ROBOT BỆNH VIỆN (SIM_VISION DATASETS)

Thư mục này chứa toàn bộ các bộ dữ liệu mô phỏng 3D chuẩn hóa phục vụ huấn luyện, đánh giá và kiểm thử độ bền vững (Stress-test) của các mô hình Thị Giác Máy Tính (Person Detection) trên **Robot Trợ Lý Bệnh Viện Tự Hành**.

Dữ liệu được kết xuất từ môi trường **Godot 4 Forward+ Vulkan**, tích hợp đầy đủ hệ sinh thái lâm sàng, đạo cụ y tế, động học tránh va chạm và siêu dữ liệu khoa học.

---

## 📂 Danh Mục Các Bộ Dữ Liệu

### 1. `vision_eval_1200/` — Baseline Spatial & Environmental Benchmark (1,200 Frames)
- **Mục tiêu**: Đánh giá khả năng tổng quát hóa cơ bản của mô hình trước các biến thiên không gian, cự ly và chiếu sáng.
- **Quy mô**: **1,200 khung hình chuẩn hóa** ($640 \times 480$, JPEG).
- **Phân bổ 8 Trục Kịch Bản (D1 – D8)** (Mỗi trục đúng 150 frames):
  - `D1_Static_Occ`: Che khuất bởi cấu trúc kiến trúc (Trụ cột, vách kính, mép cửa phòng 110).
  - `D2_Med_Props`: Che khuất bởi đạo cụ y tế tiền cảnh (Xe tiêm, cọc truyền dịch, giường cấp cứu).
  - `D3_Vulnerable`: Nhóm bệnh nhân yếu thế (Ngồi xe lăn, chống gậy, nằm giường bệnh).
  - `D4_Crowd_Dyn`: Động học đám đông (Người đi bộ cắt ngang, che lấp lẫn nhau, xếp hàng).
  - `D5_Distances`: Biến dạng khoảng cách ($0.6\text{m} \to 10.0\text{m}$) và cắt biên mép ảnh FOV.
  - `D6_Kinematics`: Góc quay cơ thể đa hướng ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ$).
  - `D7_Lighting`: Nhiễu môi trường quang học PBR (Đèn huỳnh quang, ngược sáng, tối mờ sự cố).
  - `D8_Hard_Negatives`: Bẫy báo động giả (Xe lăn rỗng, giường trống, poster y tế, kiosk).
- **Notebook đối chứng**: [`notebooks/01_vision_model_benchmark_visualization.ipynb`](../../notebooks/01_vision_model_benchmark_visualization.ipynb) (10 nhóm / 50 hạng mục trực quan).

---

### 2. `vision_eval_extended_10800/` — Big Data Clinical Demographics & Kinematics Benchmark (10,800 Frames)
- **Mục tiêu**: Stress-test chuyên sâu cấp độ nghiên cứu khoa học; giải quyết các thách thức thị giác lâm sàng phức tạp (ngụy trang quang học, nén tỷ lệ $H/W$, bẫy đa tạp phẳng 2D, động học né ngang $50\text{Hz}$ và triệt tiêu hộp bao NMS).
- **Quy mô**: **10,800 khung hình chuẩn hóa** ($640 \times 480$, JPEG).
- **Cấu trúc lấy mẫu 5 chiều cực đại**:
  - $36\text{ Ca kiểm thử lâm sàng độc lập (N1.1} \to \text{N6.5)}$.
  - Mỗi ca gồm đúng **300 khung hình**:
    - 15 Cự ly quan sát liên tục ($0.8\text{m} \to 7.5\text{m}$).
    - 5 Góc ngắm camera robot ($\pm 16^\circ, \pm 8^\circ, 0^\circ$).
    - 2 Độ cao cảm biến robot ($0.85\text{m}$ ngang ngực robot vs $1.25\text{m}$ ngang tầm mắt).
    - 2 Hướng quay NPC ($0^\circ$ đối diện vs $45^\circ$ nghiêng) hoặc 2 chế độ chiếu sáng ($450\text{ lux}$ chuẩn vs $220\text{ lux}$ hành lang tối).
    - *(Ca động học: 30 bước thời gian liên tục $\times$ 5 góc camera $\times$ 2 độ cao camera)*.
- **6 Phân nhóm kịch bản lâm sàng**:
  - `N1_Demographics_Attire` (2,100 frames): Bác sĩ blouse trắng ngụy trang tường ($\Delta E < 10.5$), Y tá scrub xanh trùng màu rèm, Cụ già tóc bạc áo hoa, Cụ ông gù lưng kyphosis, Áo kẻ sọc, Áo khoác dài, Thể hình đậm.
  - `N2_Clinical_Poses` (1,500 frames): Ngồi ghế băng dài, Bệnh nhân gập người ho khạc $45^\circ$, Ngồi buông thõng mỏi mệt, Bác sĩ đọc kẹp bệnh án, Ngồi bắt chéo chân.
  - `N3_Prop_Occlusion` (2,100 frames): Che khuất bởi cột truyền dịch inox, máy monitor sinh hiệu, kiosk lấy số, cây nước nóng lạnh, cột khử khuẩn, xe cáng cấp cứu, standee y tế.
  - `N4_Hard_Negatives` (2,400 frames - Ground Truth RỖNG): 8 loại đạo cụ rỗng kiểm tra False Positive (phơi bày bẫy Standee 2D Poster Trap).
  - `N5_Kinematics_Pardon` (1,200 frames): Động học tuần tra đối đầu, Bước né ngang nhường đường ("Pardon Sidestep" $0.55\text{m}$ vận tốc góc $28.1^\circ/\text{s}$), Chuyển hướng $90^\circ$ qua cửa, Rẽ bất ngờ từ góc khuất.
  - `N6_Crowd_Social` (1,500 frames): Hai cụ già ngồi liền kề ghế băng (thử thách NMS-Free YOLO26n vs Greedy NMS YOLOv8n), Bác sĩ đứng - Bệnh nhân ngồi, Hàng chờ kiosk 3 người, Cặp đôi thảo luận đối mặt, Dìu bước đi chậm.
- **Notebook đối chứng**: [`notebooks/02_clinical_demographics_vision_benchmark.ipynb`](../../notebooks/02_clinical_demographics_vision_benchmark.ipynb) (12 nhóm chuyên đề / 65 hạng mục khoa học).

---

### 3. `samples/` — Visual Validation Samples
- Thư mục lưu trữ các hình ảnh trích xuất trực quan (Visual Inspection) phục vụ tinh chỉnh tư thế tiếp xúc của NPC:
  - Hiệu chỉnh tọa độ gối nằm giường bệnh cấp cứu ($Y=+0.08\text{m}$, nghiêng đầu tựa gối $-87^\circ$).
  - Hiệu chỉnh chân tiếp xúc bàn để chân xe lăn ($Y=+0.08\text{m}$).
  - Hiệu chỉnh đệm ngồi ghế băng dài ($Y=+0.06\text{m}$, chạm sàn tự nhiên).

---

## 🏷️ Cấu Trúc Nhãn & Siêu Metadata (7 Phân Hệ)

Trong bộ dữ liệu `vision_eval_extended_10800/`, mỗi khung hình đi kèm đồng thời:
1. **Nhãn YOLO chuẩn hóa** (`labels/*.txt`): `0 <x_center> <y_center> <width> <height>` (tất cả chuẩn hóa $[0, 1]$).
2. **Siêu Metadata JSON Lines** (`annotations_rich_5w1h.jsonl`):
   - **Định danh lâm sàng**: `frame_id`, `case_id`, `group_id`, `clinical_hazard_tier` (Tier 1/2/3).
   - **Tư thế 6-DoF Camera**: `position_world`, `rotation_euler_deg`, `intrinsics` ($f_x, f_y, c_x, c_y, \text{fov}$).
   - **Quang học & Trắc sắc học**: Profile sáng, độ rọi Lux, nhiệt độ màu Kelvin, sai màu đối chứng CIE $\Delta E^*_{ab}$.
   - **Động học & Thời gian**: Timestep index, vận tốc góc $d\theta/dt$, góc ngắm bearing thực tế $\theta_{\text{bearing}}$.
   - **Thực thể & Đạo cụ**: Phân lớp nhân khẩu học, trang phục, góc gập cột sống, đạo cụ che khuất.
   - **Hình học Bounding Box**: BBox XYXY, YOLO normalized, Aspect ratio $H/W$, Area pixel, Khoảng cách 3D Euclid.
   - **Diễn giải 5W1H Narrative**: Who, What, Where, When, Why, How có thể đọc hiểu tự động bởi LLM.

---

## 📈 Tóm Tắt Kết Quả Đối Chứng 4 Mô Hình (Trên 10,800 Frames)

| Mô Hình | Recall Toàn Cục | Recall Lâm Sàng (N2) | Recall Yếu Thế | FPR Bẫy Rỗng (N4) | Sai Số Góc Lái $\Delta\theta$ | Độ Trễ P95 | Đánh Giá Triển Khai |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO26n (NMS-Free)** | **45.67%** | **82.20%** 🏆 | **83.93%** 🏆 | 88.29% | $1.14^\circ$ | $47.9\text{ ms}$ | **Khuyến nghị #1** (Tách chuẩn người ngồi sát, không bị NMS xóa nhầm) |
| **YOLO11n (C2PSA Attention)** | 44.94% | 81.93% | 83.52% | 87.33% | $1.14^\circ$ | $45.0\text{ ms}$ | Khá (Chú ý không gian tốt) |
| **YOLOv8n (Baseline DFL)** | 45.01% | 82.07% | 83.04% | 87.21% | $1.14^\circ$ | $41.1\text{ ms}$ | Ổn định (Greedy NMS xóa nhầm bệnh nhân ngồi kề) |
| **Intel OpenVINO SSD** | 46.12% | 69.80% ❌ | 78.37% ❌ | 89.88% ❌ | **$0.79^\circ$** | **$8.9\text{ ms}$** | Không đạt chuẩn lâm sàng (Trượt người ngồi cự ly gần) |

> [!NOTE]
> Báo cáo khoa học chi tiết: [`sim_vision/reports/vision_extended_benchmark.md`](../reports/vision_extended_benchmark.md)  
> Tài liệu kỹ thuật First Principles: [`docs/experiments/EXP_VISION_EXTENDED_NPC_BENCHMARK.md`](../../docs/experiments/EXP_VISION_EXTENDED_NPC_BENCHMARK.md)
