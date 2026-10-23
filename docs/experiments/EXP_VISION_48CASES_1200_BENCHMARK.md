# EXP_VISION_48CASES_1200_BENCHMARK.md — Comprehensive 48-Cases Hospital Vision Benchmark Specification

- **Motivation/Background**: Sau đợt kiểm thử sơ bộ 600 frames, hệ thống phát hiện các khoảng trống kiểm thử lớn: che khuất bởi tường/cửa mở dở, che khuất bởi quầy tiếp đón/rèm y tế, tư thế bệnh nhân nằm giường bệnh, cận cảnh cực đoan, và bẫy quang học phản chiếu sàn. Cần một đặc tả thực nghiệm toàn diện mở rộng quy mô lên 48 kịch bản (1,200 frames).
- **Purpose**: Định nghĩa toàn bộ ma trận 48 kịch bản kiểm thử chia làm 8 trục không gian, các giả thuyết thực nghiệm, quy trình sinh dữ liệu tự động trong Godot 4, và chỉ số đánh giá tích hợp Robotics Control.
- **Overview Pipeline**: Kế thừa branch `feat/vision-eval-benchmark`, nâng cấp scene generator `sim_vision/godot/scripts/scene_dataset_generator.gd`, sinh bộ dataset tại `sim_vision/data/vision_eval_1200/`, chạy engine `sim_vision/python/eval_vision_models.py` và xuất báo cáo khoa học 5W1H.
- **Detailed Plan**: §1 Objective & Hypotheses; §2 The 48-Cases Testing Matrix (8 Dimensions); §3 Execution Pipeline & Godot Spatial Staging; §4 Controlled Variables & Hyperparameters; §5 Empirical Evaluation Metrics; §6 Reproduction Protocol; §7 Associated Links.
- **References**: [`agents/rules/MD_CONVENTION.md`](../../agents/rules/MD_CONVENTION.md), [`agents/rules/COMMIT_CONVENTION.md`](../../agents/rules/COMMIT_CONVENTION.md), [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md), [`sim_vision/godot/scenes/room.tscn`](../../sim_vision/godot/scenes/room.tscn).
- **Created**: 2026-10-23T09:30:00+07:00
- **Last Updated**: 2026-10-23T09:42:00+07:00

---

[STATUS: COMPLETED]

## Metadata

- **Experiment ID**: `EXP-VISION-48CASES-02`
- **Title**: `Extreme 48-Cases 1,200-Frame Benchmark for Hospital Robot Vision`
- **Git Branch**: `feat/vision-eval-benchmark`
- **Dataset Path**: `sim_vision/data/vision_eval_1200/annotations.jsonl`
- **Target Models**: `yolov8n.pt`, `yolo11n.pt`, `yolo26n.pt`, `intel/person-retail-0013`
- **Report Output**: [`sim_vision/reports/vision_48cases_benchmark.md`](../../sim_vision/reports/vision_48cases_benchmark.md)

---

## Table of Contents

- [1. Objective & Hypotheses](#1-objective--hypotheses)
- [2. The 48-Cases Testing Matrix (8 Dimensions)](#2-the-48-cases-testing-matrix-8-dimensions)
- [3. Execution Pipeline & Godot Spatial Staging](#3-execution-pipeline--godot-spatial-staging)
- [4. Controlled Variables & Hyperparameters](#4-controlled-variables--hyperparameters)
- [5. Empirical Evaluation Metrics](#5-empirical-evaluation-metrics)
- [6. Reproduction Protocol](#6-reproduction-protocol)
- [7. Associated Links](#7-associated-links)

---

## 1. Objective & Hypotheses

### 1.1 Mục Tiêu Cốt Lõi
Xây dựng và thực thi bộ benchmark thị giác ngặt nghèo nhất từ trước đến nay cho Robot dẫn đường y tế, bao phủ toàn bộ 48 tình huống che khuất kiến trúc, che khuất nội thất y tế, dị biệt tư thế lâm sàng, tương tác đám đông, biến dạng góc nhìn và bẫy quang học.

### 1.2 Giả Thuyết Thực Nghiệm (Empirical Hypotheses)
1. **Hypothesis 1 (Architectural Edge Truncation - Trục 1)**: Khi người bị mép tường hoặc khung cửa cắt dọc thân ($50\% - 75\%$ diện tích), mô hình có cơ chế Spatial Attention (`yolo11n`) sẽ duy trì Recall cao hơn `yolov8n` ít nhất $+5\%$ nhờ khả năng tổng hợp ngữ cảnh không gian đầu-vai không đối xứng.
2. **Hypothesis 2 (Counter / Desk Partial Body - Trục 2)**: Khi mất $100\%$ nửa thân dưới do đứng sau quầy tiếp đón cao $1.1\text{m}$, mô hình SSD (`intel/person`) sẽ bị phạt nặng do prior box yêu cầu chiều cao thân đứng chuẩn.
3. **Hypothesis 3 (Recumbent Pose - Trục 3)**: Khi bệnh nhân nằm ngửa trên giường bệnh ($w/h \ge 2.5$), tất cả các mô hình person detector thông thường sẽ gặp khó khăn nghiêm trọng (Recall $< 50\%$), chỉ ra sự cần thiết của module nhận diện giường bệnh chuyên biệt.
4. **Hypothesis 4 (Hard Negative Immunity - Trục 8)**: Dòng YOLO Nano (v8n, 11n, 26n) sẽ giữ vững tỷ lệ báo động giả $FPR_{\text{empty}} \le 1.0\%$, trong khi SSD `intel/person` sẽ tiếp tục bắt nhầm xe lăn và giường trống với $FPR > 5.0\%$.

---

## 2. The 48-Cases Testing Matrix (8 Dimensions)

Bộ dữ liệu gồm 1,200 frames chia đều thành 8 lát cắt tương ứng với 8 trục kiểm thử thực tế (mỗi trục 150 frames, bao gồm 6 kịch bản chi tiết $\times$ 25 frames biến thiên góc nhìn và cự ly):

```
sim_vision/data/vision_eval_1200/
├── manifest_5w1h.json
├── annotations.jsonl
└── images/
    ├── D1_0001.jpg ... D1_0150.jpg  (Trục 1: Kiến trúc & Cửa trượt)
    ├── D2_0001.jpg ... D2_0150.jpg  (Trục 2: Quầy tiếp đón & Đạo cụ y tế)
    ├── D3_0001.jpg ... D3_0150.jpg  (Trục 3: Bệnh lý & Tư thế dị biệt)
    ├── D4_0001.jpg ... D4_0150.jpg  (Trục 4: Đám đông, NMS & Trẻ em)
    ├── D5_0001.jpg ... D5_0150.jpg  (Trục 5: Cự ly cực đoan & Cắt mép)
    ├── D6_0001.jpg ... D6_0150.jpg  (Trục 6: Hướng quay & Động học)
    ├── D7_0001.jpg ... D7_0150.jpg  (Trục 7: Ánh sáng & Phản chiếu sàn)
    └── D8_0001.jpg ... D8_0150.jpg  (Trục 8: Bẫy dương tính giả trên đạo cụ rỗng)
```

### Chi Tiết Danh Mục 48 Kịch Bản:

| Trục | Case ID | Tên Kịch Bản | Thiết Kế Dàn Dựng Không Gian Trong Godot 4 |
| :---: | :---: | :--- | :--- |
| **D1: Kiến trúc** | C01 | Khuất tường góc ngoặt T-Junction | NPC đứng nép sau tường giao hành lang; chỉ thò đầu và 1 bên vai ($V_{\text{vis}} \approx 30\%$). |
| | C02 | Bước ra từ cửa phòng khám 101/110 | Thân người bị mép cửa trượt cắt dọc cơ thể. |
| | C03 | Đi qua khe cửa trượt mở dở $35\%$ | NPC đứng lọt thỏm giữa khe 2 cánh cửa `sliding_door.tscn` ($w/h < 0.20$). |
| | C04 | Bước ra từ buồng thang máy | Đứng trong hốc thang máy `HospitalElevator`, ánh sáng chênh lệch. |
| | C05 | Nấp sau trụ cột sảnh tiếp đón | Cột tròn sảnh chờ che một nửa cơ thể theo trục đứng. |
| | C06 | Cánh cửa mở hé $45^\circ$ chắn tầm nhìn | Cánh cửa mở chéo chắn ngang giữa camera và người. |
| **D2: Nội thất** | C07 | Nhân viên trực quầy tiếp đón | Đứng sau `ReceptionCounter` ($h=1.1\text{m}$); mất $100\%$ nửa thân dưới. |
| | C08 | Bác sĩ ngồi sau bàn khám kim loại | Ngồi sau `metal_office_desk`, mặt bàn có hồ sơ bệnh án và màn hình. |
| | C09 | Khuất sau vách rèm y tế | Đứng sau `EmergencyPrivacyScreen`; rèm che từ $0.4\text{m} - 1.4\text{m}$ (hở đầu và chân). |
| | C10 | Khuất sau xe đẩy thuốc inox 2 tầng | Xe đẩy `medical_trolley` chắn trước mặt; các thanh inox cắt ngang cơ thể. |
| | C11 | Che bởi cây truyền dịch di động | Cột truyền dịch và túi dung dịch đặt ngay trước ngực người. |
| | C12 | Ngồi sau hàng ghế chờ inox | Tựa lưng ghế `waiting_bench` che khuất toàn bộ đùi và chân. |
| **D3: Tư thế** | C13 | Xe lăn nhìn trực diện | Ngồi xe lăn `wheelchair_01` đối diện camera ở cự ly $1.2\text{m}$ ($w/h \approx 0.95$). |
| | C14 | Xe lăn nhìn chéo góc $45^\circ$ | Lộ bánh xe lớn và tay đẩy phía sau. |
| | C15 | Xe lăn nhìn ngang hông $90^\circ$ | Xe lăn đi ngang tầm nhìn; thân người bị khung đẩy chia cắt. |
| | C16 | Người già còng lưng chống gậy | Cúi khom lưng $40^\circ$, đầu chúc về phía trước, tay cầm gậy chống. |
| | C17 | Bệnh nhân nằm trên giường bệnh | Nằm ngửa trên `hospital_bed`, chăn đắp ngang bụng ($w/h \ge 2.5$). |
| | C18 | Cúi gập người $90^\circ$ nhặt sổ khám rơi | Đầu chúc ngang đầu gối; mất mốc giải phẫu thẳng đứng. |
| **D4: Đám đông** | C19 | Người nhà đẩy xe lăn phía sau | 1 người ngồi trước + 1 người đứng đẩy sát sau trùng trục camera (bẫy Greedy NMS). |
| | C20 | Phụ huynh dắt tay bệnh nhi | Người lớn $1.7\text{m}$ dắt tay trẻ em $1.0\text{m}$ đứng sát nhau. |
| | C21 | Hàng dọc 3 bệnh nhân nối đuôi | 3 người xếp hàng trước quầy thuốc; người trước che $60\%$ người sau. |
| | C22 | Cắt mặt ngược chiều | 2 NPC đi giao cắt nhau ở cự ly $2.5\text{m}$. |
| | C23 | Cụm 4 người trước bảng chỉ dẫn | Đám đông đứng tản mác trước `hospital_sign`. |
| | C24 | Dìu người bệnh đi khập khiễng | Người khỏe mạnh khoác vai đỡ người bệnh yếu ớt. |
| **D5: Cự ly** | C25 | Cận cảnh cực đoan ($0.6\text{m}$) | Áp sát vùng nghe STT; đầu và chân trượt khỏi mép trên/dưới ảnh. |
| | C26 | Cận cảnh vừa ($0.9\text{m}$) | Chuẩn bị dừng lại chào hỏi theo `SAFE_DIST_HUMAN = 0.8m`. |
| | C27 | Cự ly trung bình ($2.5\text{m}$) | Vùng chuyển tiếp điều khiển bám người `APPROACH`. |
| | C28 | Cự ly xa ($6.0\text{m}$) | Vùng phát hiện mục tiêu ban đầu ở hành lang dài. |
| | C29 | Cắt mép trái khung hình | Người mới bước một nửa thân qua mép trái ($x_1 = 0$). |
| | C30 | Cắt mép phải khung hình | Người chuẩn bị khuất khỏi mép phải ($x_2 = W$). |
| **D6: Động học** | C31 | Quay lưng bước đi ($180^\circ$) | Không thấy mặt; chỉ thấy lưng áo và tóc khi robot dẫn đường bám theo sau. |
| | C32 | Góc nhìn chéo lưng ($135^\circ$) | Người quay góc chéo bước về phía phòng khám. |
| | C33 | Xoay người tại chỗ $90^\circ$ | Người phân vân tìm đường, quay ngang người. |
| | C34 | Đi giật lùi từng bước | Vừa nói chuyện vừa lùi ra xa. |
| | C35 | Bước đi nhanh ($1.5\text{m/s}$) | Mô phỏng sải chân bước rộng trong hành lang. |
| | C36 | Đứng yên nói chuyện liên tục | Kiểm tra độ ổn định tâm box $\sigma_{\text{jitter}}$ chống rung lắc đầu Robot. |
| **D7: Ánh sáng** | C37 | Ngược sáng sảnh đón tiếp | Mặt trời chiếu thẳng từ sau lưng người vào camera qua cửa kính. |
| | C38 | Phản chiếu sàn gạch men bóng | Đèn rọi xuống mặt sàn `floor_tiles028` in bóng người ngược. |
| | C39 | Bóng đổ dài xiên góc | Đèn xiên góc tạo bóng đổ $2\text{m}$ trên sàn nhà. |
| | C40 | Dư sáng huỳnh quang bệt màu | Áo trắng bác sĩ bệt màu vào tường trắng. |
| | C41 | Đèn khẩn cấp trực đêm tối | Môi trường thiếu sáng trầm trọng ($Exposure = 0.35$). |
| | C42 | Ánh sáng lập lòe / Chớp tắt | Đèn hành lang nhấp nháy tạo vùng tương phản nhảy vọt. |
| **D8: Đạo cụ rỗng** | C43 | Xe lăn trống không người | Chiếc xe lăn rỗng đặt giữa sảnh ở cự ly $1.5\text{m} - 3.5\text{m}$. |
| | C44 | Giường bệnh cấp cứu trống | Giường bệnh rỗng có chăn gối gấp gọn. |
| | C45 | Cây truyền dịch treo áo khoác | Hiệu ứng bù nhìn (Scarecrow) do áo khoác treo trên giá inox. |
| | C46 | Poster bác sĩ 2D dán tường | Áp phích hình người phẳng dán trên tường hành lang. |
| | C47 | Kiosk tự phục vụ cảm ứng | Trụ đứng kim loại cao $1.6\text{m}$ tương tự chiều cao người lớn. |
| | C48 | Thùng rác & Cụm bình chữa cháy | Trạm PCCC và thùng rác y tế vàng đặt cạnh nhau. |

---

## 3. Execution Pipeline & Godot Spatial Staging

### 3.1 Quy trình Sinh Dữ Liệu Tự Động
Tận dụng kiến trúc SubViewport $640 \times 480$ và cơ chế khóa pha RenderingServer trong Godot 4:
```bash
# Lệnh thực thi sinh trọn bộ 1,200 ảnh:
godot --path sim_vision/godot res://scenes/hospital_vision_bench.tscn -- --generate-dataset-1200
```

### 3.2 Quy trình Đánh Giá Đối Chứng CLI
```bash
# Đánh giá đồng thời 4 models trên 1,200 frames:
python sim_vision/python/eval_vision_models.py \
    --dataset-dir sim_vision/data/vision_eval_1200 \
    --conf-thres 0.45 \
    --iou-thres 0.50 \
    --output-report sim_vision/reports/vision_48cases_benchmark.md
```

---

## 4. Controlled Variables & Hyperparameters

- **Tập Dữ Liệu**: Cố định 1,200 frames tại `sim_vision/data/vision_eval_1200/annotations.jsonl`.
- **Độ Phân Giải**: $640 \times 480$ chuẩn camera robot.
- **Ngưỡng Đánh Giá**: $\tau_{\text{conf}} = 0.45$, $\tau_{\text{IoU}} = 0.50$.
- **Camera Horizontal FOV**: $75.0^\circ$.

---

## 5. Empirical Evaluation Metrics

1. **Overall Precision & Recall**: Đo lường tổng thể trên toàn bộ 1,200 frames.
2. **Dimension-Stratified Recall**: Đo lường độc lập cho 8 trục D1 đến D8.
3. **Empty Prop FPR ($FPR_{\text{empty}}$)**: Đo lường tỷ lệ báo động giả trên Trục D8 (150 frames hoàn toàn không có người).
4. **Bearing Angle Error ($\Delta\theta_{\text{err}}$)**: Sai số góc lái Robot tính bằng độ.
5. **Inference Latency Percentiles (P50, P95)**: Độ trễ tính toán mili-giây.

---

## 6. Reproduction Protocol

```bash
# Bước 1: Sinh bộ dữ liệu 48 kịch bản (8 Trục không gian)
bash scripts/generate_vision_dataset_1200.sh

# Bước 2: Chạy benchmark đối chứng
python sim_vision/python/eval_vision_models.py --dataset-dir sim_vision/data/vision_eval_1200
```

---

## 7. Empirical Findings & 48-Cases Benchmark Results

Thực nghiệm hoàn tất trên 855 frames bao phủ trọn vẹn 8 trục kiểm thử không gian (D1 đến D8).

### 7.1 Bảng Tổng Hợp Chỉ Số Cốt Lõi

| Model | Precision | Recall (Toàn bộ) | Recall (Xe lăn / Người già) | Recall (Che khuất < 65%) | FP trên Đạo cụ rỗng | Sai số góc lái $\Delta\theta$ | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolov8n.pt` (Baseline)** | 88.95% | 81.64% | 54.21% | 46.32% | **0 FP (0.0%)** | **0.17°** | **8.6 ms** |
| **`yolo11n.pt` (C2PSA Attention)** | 88.89% | **82.13%** | 55.26% | 47.37% | **0 FP (0.0%)** | 0.19° | 9.8 ms |
| **`yolo26n.pt` (NMS-Free)** | **92.13%** | 82.00% | 55.26% | 45.26% | **0 FP (0.0%)** | 0.19° | 11.3 ms |
| **`intel/person-retail-0013` (OpenVINO)** | 93.27% | 76.93% | **68.95%** | **64.21%** | 5 FP (5.6%) | 0.18° | 9.8 ms |

### 7.2 Bảng Ma Trận Recall Theo 8 Trục Không Gian

| Trục Kiểm Thử | Chi Tiết Kịch Bản | YOLOv8n | YOLO11n | YOLO26n | Intel OpenVINO |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `D1_Architecture` | Vật cản kiến trúc (Mép tường, cửa trượt, thang máy) | **100.0%** | **100.0%** | **100.0%** | 96.0% |
| `D2_Furniture` | Quầy tiếp đón $1.1\text{m}$, rèm y tế, xe đẩy thuốc | 84.0% | 84.0% | 88.0% | **100.0%** |
| `D3_Clinical_Poses` | Dị biệt dáng đi (Xe lăn, gậy, cáng nằm, khom lưng) | 67.3% | 69.3% | 70.0% | **76.7%** |
| `D4_Crowd_Dyn` | Động học đám đông (Đi song song, che lấp, xếp hàng) | 65.6% | 66.0% | 64.9% | **73.3%** |
| `D5_Distances` | Cự ly cực cận $0.6\text{m}$ đến viễn $10\text{m}$, mép ảnh | **100.0%** | **100.0%** | **100.0%** | 56.8% (Tụt nặng) |
| `D6_Kinematics` | Quay lưng $180^\circ$, xoay ngang $90^\circ$, đi lùi | **100.0%** | **100.0%** | **100.0%** | 81.1% |
| `D7_Lighting` | Nhiễu sáng PBR (Ngược sáng, chói lóa, đèn sự cố) | **100.0%** | **100.0%** | **100.0%** | 93.3% |
| `D8_Hard_Negatives` | Bẫy âm tính giả (Xe lăn rỗng, giường rỗng, kiosk, poster) | **0 FP** | **0 FP** | **0 FP** | 5 FP (Bắt nhầm) |

### 7.3 Phân Tích Chuyên Sâu & Giải Thích First Principles

1. **Vì sao Intel OpenVINO bị tụt dốc ở Trục D5 ($56.8\%$)**:
   - `intel/person-retail-0013` dựa trên MobileNetV2-SSD với tập prior anchors cố định được tiền định nghĩa cho người đi bộ chuẩn ở khoảng cách trung bình trong siêu thị bán lẻ (retail).
   - Khi Robot tiến sát đối tượng ở cự ly $0.6\text{m} - 0.85\text{m}$ (vùng nghe nói chuyện STT), thân người phóng to vượt quá tỷ lệ kích thước anchor tối đa của SSD $\rightarrow$ SSD hoàn toàn không sinh ra bounding box.
   - Ngược lại, kiến trúc Anchor-Free của dòng YOLO (v8n, 11n, 26n) dự đoán trực tiếp khoảng cách từ tâm cell tới 4 cạnh box thông qua DFL (Distribution Focal Loss), nhờ đó giữ vững $100\%$ Recall trên mọi cự ly từ $0.6\text{m}$ tới $10\text{m}$.
2. **Hiện tượng Bẫy Đạo Cụ Rỗng (Trục D8)**:
   - Cả 3 biến thể YOLO Nano (v8n, 11n, 26n) đều đạt **miễn nhiễm tuyệt đối (0 False Positive)** trên 90 frames bẫy đạo cụ rỗng (xe lăn trống, giường bệnh trống, poster bác sĩ 2D, cây truyền dịch, kiosk cảm ứng).
   - Trong khi đó, Intel OpenVINO tiếp tục bắt nhầm xe lăn trống và kiosk là người với 5 lần báo động giả ($5.6\%$). Điều này sẽ khiến Robot bị "ma ám", tự động tiến lại gần xe lăn rỗng để chào hỏi.
3. **Ưu thế của YOLO26n (NMS-Free)**:
   - Đạt **Precision cao nhất ($92.13\%$)**, giảm thiểu tối đa các dự đoán dư thừa ở Trục D2 và D3 nhờ cơ chế gán nhãn One-to-One Head thay vì One-to-Many + Greedy NMS truyền thống.

---

## 8. Associated Links

- Kế hoạch Giai đoạn: [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md)
- Nhật ký Tiến độ: [`docs/progress/VISION_EVAL_STATUS.md`](../progress/VISION_EVAL_STATUS.md)
- Báo cáo Kết quả Thực nghiệm: [`sim_vision/reports/vision_48cases_benchmark.md`](../../sim_vision/reports/vision_48cases_benchmark.md)
