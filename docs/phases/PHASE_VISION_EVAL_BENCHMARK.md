# PHASE_VISION_EVAL_BENCHMARK.md — Vision Model Evaluation & Godot Dataset Harvesting Specification

- **Motivation/Background**: Phân hệ Vision đóng vai trò cốt lõi cung cấp tọa độ BBoxes điều khiển Robot tiếp cận và giao tiếp với bệnh nhân. Tuy nhiên, việc đánh giá năng lực nhận diện người hiện chưa có bộ Ground Truth chuẩn xác trong môi trường mô phỏng 3D Godot 4, dẫn đến nguy cơ lựa chọn sai model (ví dụ các model SSD Anchor-based như `intel/person` bị trượt người ngồi xe lăn hoặc góc nhìn thấp) và không đo lường được sai số góc hướng điều khiển.
- **Purpose**: Định nghĩa toàn bộ đặc tả kiến trúc, hợp đồng I/O, nền tảng toán học chiếu hình 3D-to-2D không phình hộp, quy trình thu thập bộ dataset phân tầng 600 frames, và pipeline benchmark đối chứng đa model (`yolov8n` vs `intel/person` vs `yolo26n`/`yolo11n`).
- **Overview Pipeline**: Khởi tạo trên branch `feat/vision-eval-benchmark`, tích hợp trực tiếp giữa engine mô phỏng [`sim_vision/godot`](../../sim_vision/godot) và backend đánh giá CLI [`sim_vision/python`](../../sim_vision/python).
- **Detailed Plan**: §1 Scope & Objective; §2 Model Comparative Analysis (First Principles); §3 Mathematical Foundation (3D-to-2D Projection & Occlusion Raycast); §4 Stratified Dataset Architecture (5 Slices / 600 Frames); §5 Execution Pipeline & CLI Contracts; §6 Evaluation Metrics Suite (Robotics & Detection); §7 Acceptance Criteria & Quality Gates; §8 Associated Links.
- **References**: [`agents/rules/MD_CONVENTION.md`](../../agents/rules/MD_CONVENTION.md), [`agents/rules/RESULTS_REPORTING.md`](../../agents/rules/RESULTS_REPORTING.md), [`sim_vision/python/detector.py`](../../sim_vision/python/detector.py), [`sim_vision/python/protocol.py`](../../sim_vision/python/protocol.py).
- **Created**: 2026-10-23T09:05:00+07:00
- **Last Updated**: 2026-10-23T09:05:00+07:00

---

[STATUS: ACTIVE]

## Metadata

- **Phase ID**: `PHASE-VISION-BENCH-01`
- **Phase Name**: `Vision Model Evaluation & Automated Godot Dataset Benchmark`
- **Git Branch**: `feat/vision-eval-benchmark`
- **Status**: IN_PROGRESS
- **Target Modules**:
  - Godot Harvester: [`sim_vision/godot/scripts/scene_dataset_generator.gd`](../../sim_vision/godot/scripts/scene_dataset_generator.gd)
  - Python Evaluator: [`sim_vision/python/eval_vision_models.py`](../../sim_vision/python/eval_vision_models.py)
  - Detector Core: [`sim_vision/python/detector.py`](../../sim_vision/python/detector.py)

---

## Table of Contents

- [1. Scope & Objective](#1-scope--objective)
- [2. Model Comparative Analysis (First Principles)](#2-model-comparative-analysis-first-principles)
- [3. Mathematical Foundation (3D-to-2D Projection & Occlusion Raycast)](#3-mathematical-foundation-3d-to-2d-projection--occlusion-raycast)
- [4. Stratified Dataset Architecture (5 Slices / 600 Frames)](#4-stratified-dataset-architecture-5-slices--600-frames)
- [5. Execution Pipeline & CLI Contracts](#5-execution-pipeline--cli-contracts)
- [6. Evaluation Metrics Suite (Robotics & Detection)](#6-evaluation-metrics-suite-robotics--detection)
- [7. Acceptance Criteria & Quality Gates](#7-acceptance-criteria--quality-gates)
- [8. Associated Links](#8-associated-links)

---

## 1. Scope & Objective

### 1.1 Bối cảnh & Vấn đề Kỹ thuật
Trong hệ thống Robot trợ lý bệnh viện đa phương thức:
1. **Khớp nối Thị giác - Điều khiển**: Đầu ra bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ của [`sim_vision/python/detector.py`](../../sim_vision/python/detector.py) là tín hiệu điều khiển trực tiếp cho [`sim_vision/godot/scripts/controller.gd`](../../sim_vision/godot/scripts/controller.gd) để xoay mặt, bám theo bệnh nhân và kích hoạt hội thoại.
2. **Độ lệch miền đồ họa 3D (Synthetic-to-Real Domain Gap)**: Đồ họa Godot 4 PBR bề mặt nhẵn, ít texture tần số cao làm các model phụ thuộc cục bộ bị tụt confidence.
3. **Dị biệt tỷ lệ khung hình (Aspect Ratio Shift)**: Bệnh nhân ngồi xe lăn hoặc người già chống gậy có tỷ lệ khung hình rộng ngang ($w/h \approx 0.8 - 1.1$), khác hoàn toàn dáng người đứng chuẩn ($w/h \approx 0.35 - 0.45$).
4. **Chia sẻ tài nguyên phần cứng**: Phân hệ Vision phải chạy song song với PhoASR-Whisper (Port 8766), PhoBERT NLU, và VieNeu-TTS (Port 8767), đòi hỏi latency P95 $\le 15\text{ ms}$ trên CPU/GPU.

### 1.2 Mục tiêu cốt lõi
- Xây dựng công cụ tự động sinh bộ dữ liệu phân tầng 600 ảnh trong Godot 4 với Ground Truth chính xác 100% bằng toán học hình học, loại bỏ hoàn toàn việc gán nhãn thủ công.
- Đánh giá đối chứng đa mô hình: `yolov8n.pt` (baseline), `intel/person-detection-0202` (OpenVINO SSD), `yolo26n.pt` / `yolo11n.pt` (thế hệ mới NMS-Free).
- Đo lường tác động của sai số thị giác lên góc điều khiển Robot ($\Delta\theta_{\text{err}}$) và tỷ lệ báo động giả trên đạo cụ rỗng ($FPR_{\text{empty}}$).

---

## 2. Model Comparative Analysis (First Principles)

### 2.1 Bảng so sánh Kiến trúc
| Tiêu chí | `intel/person-0202` (OpenVINO) | `yolov8n.pt` (Baseline) | `yolo26n.pt` / `yolo11n.pt` (Ứng viên nâng cấp) |
| :--- | :--- | :--- | :--- |
| **Kiến trúc Head** | MobileNetV2 + SSD Head | Anchor-free Decoupled Head | Anchor-free + One-to-One NMS-Free Head |
| **Cơ chế Hộp neo** | Clustered Prior Boxes ($w/h \approx 0.4$) | Anchor-free (hồi quy khoảng cách $l,t,r,b$) | Anchor-free + Dual-Head STAL Loss |
| **Bệnh nhân Xe lăn** | Kém ($IoU_{\max} < 0.45$ với prior box hẹp) | Tốt (tự do co giãn 4 hướng) | Rất tốt (tập trung đặc trưng cấu trúc cơ thể) |
| **Đám đông chồng lấp** | Kém (1-2 scale heads, receptive field lẫn) | Khá (phụ thuộc lọc Greedy NMS) | Xuất sắc (loại bỏ NMS, giữ trọn 2 người trùng trục) |
| **Bẫy Đạo cụ rỗng** | Dễ bắt nhầm biển báo, kiosk | Ít bắt nhầm khi $\tau_{\text{conf}} \ge 0.45$ | Rất ít bắt nhầm nhờ attention ngữ cảnh |
| **Độ trễ suy luận** | $\approx 2 - 4\text{ ms}$ (CPU INT8) | $\approx 6 - 12\text{ ms}$ (CPU ONNX) | $\approx 4 - 8\text{ ms}$ (CPU NMS-Free, nhanh hơn 43%) |

### 2.2 Nhận định kỹ thuật
- **Loại bỏ `intel/person` (SSD)** làm detector chính cho robot dẫn đường vì kiến trúc SSD với prior box đứng thẳng không thể thích nghi với bệnh nhân ngồi xe lăn ở cự ly gần ($0.8\text{ m} - 1.8\text{ m}$). Giữ lại làm mốc baseline so sánh.
- **Giữ `yolov8n`** làm baseline vững chắc (đã tích hợp trong [`sim_vision/python/detector.py`](../../sim_vision/python/detector.py)), ép `classes=[0]` để tối ưu NMS.
- **Thử nghiệm `yolo26n` / `yolo11n`** làm ứng viên số 1 để giải quyết triệt để lỗi mất box tại quầy tiếp đón đông người và giảm tải CPU.

---

## 3. Mathematical Foundation (3D-to-2D Projection & Occlusion Raycast)

### 3.1 Chống lỗi Phình Hộp (Anti-AABB Bloat via Local OBB Sampling)
Thay vì sử dụng `global_transform * mesh.get_aabb()` (làm hộp bị phình rộng $41.4\%$ khi NPC xoay góc $45^\circ$), hệ thống lấy mẫu lưới nội suy 45 điểm trong hệ tọa độ cục bộ (Local Space) của từng bộ phận cơ thể:
$$P_{\text{local}} = AABB.\text{pos} + [AABB.\text{size}_x \cdot i_x, AABB.\text{size}_y \cdot i_y, AABB.\text{size}_z \cdot i_z]^T$$
với $i_x, i_z \in \{0.05, 0.50, 0.95\}$ và $i_y \in \{0.02, 0.25, 0.50, 0.75, 0.98\}$.

Tọa độ thế giới được tính trực tiếp:
$$P_{\text{world}} = \mathbf{T}_{\text{mesh}} \cdot P_{\text{local}}$$
Loại bỏ hoàn toàn các điểm có $Z_{\text{eye}} \le 0$ (phía sau Near Clipping Plane qua `robot_camera.is_position_behind(P)`), sau đó chiếu lên 2D:
$$(u, v) = \text{unproject\_position}(P_{\text{world}})$$

### 3.2 Tách biệt Ngữ nghĩa (Semantic Separation for Wheelchairs)
- Cơ thể sinh học của NPC: gắn vào Group `gt_body_mesh` $\rightarrow$ Tính toán Bounding Box chuẩn class `person`.
- Khung xe lăn / Gậy chống: gắn vào Group `gt_assistive_device` $\rightarrow$ Không gộp vào hộp người, chỉ ghi metadata `subcategory: wheelchair`.

### 3.3 Chỉ số Hiển thị Thực tế (True Visibility Ratio $V_{\text{vis}}$)
Bắn $M = 15$ tia vật lý từ tâm Camera Robot tới các điểm mẫu trên cơ thể NPC qua `PhysicsRayQueryParameters3D`:
$$V_{\text{vis}} = \frac{1}{M} \sum_{j=1}^{M} \mathbb{I}(\text{Raycast}(O_{\text{cam}} \rightarrow P_j) \text{ trúng Collider NPC hoặc rỗng})$$
- $V_{\text{vis}} < 0.15$: Bỏ qua (không đưa vào Ground Truth).
- $0.15 \le V_{\text{vis}} < 0.25$: Đánh dấu `ignore_eval = true` (vùng che khuất cực hạn, tương tự `iscrowd` trong COCO).
- $V_{\text{vis}} \ge 0.25$: Đưa vào đánh giá phân tầng Occluded.

---

## 4. Stratified Dataset Architecture (5 Slices / 600 Frames)

Bộ dữ liệu được tổ chức tại `data/vision_eval_600/` gồm 600 frames định dạng JPEG ($640 \times 480$) kèm file `annotations.jsonl` chuẩn 5W1H:

```
data/vision_eval_600/
├── manifest_5w1h.json
├── annotations.jsonl
└── images/
    ├── S1_0001.jpg ... S1_0120.jpg  (Hành lang tiêu chuẩn)
    ├── S2_0001.jpg ... S2_0180.jpg  (Bệnh nhân Xe lăn & Người già)
    ├── S3_0001.jpg ... S3_0150.jpg  (Quầy tiếp đón đông người & Chồng lấp)
    ├── S4_0001.jpg ... S4_0090.jpg  (Nhiễu ánh sáng bệnh viện)
    └── S5_0001.jpg ... S5_0060.jpg  (Bẫy Dương tính giả - Đạo cụ rỗng)
```

| Slice ID | Tên Kịch bản | Số lượng | Biến số Không gian & Ánh sáng | Mục tiêu Đánh giá |
| :--- | :--- | :---: | :--- | :--- |
| **S1_Baseline** | Hành lang chuẩn | 120 | 2 NPC đứng, 5 cự ly ($1.5 - 8.0\text{ m}$), 4 góc xoay NPC, 3 góc lệch cam $\Delta\psi \in \{-18^\circ, 0, +18^\circ\}$ | Baseline Recall/Precision & Latency |
| **S2_Vulnerable** | Xe lăn & Người già | 180 | 3 NPC (xe lăn, khom lưng chống gậy, ngồi ghế), 5 cự ly ($0.9 - 4.5\text{ m}$), 4 góc xoay | Đánh giá khả năng bắt dáng người dị biệt |
| **S3_Crowd_Occ** | Quầy tiếp đón đông | 150 | 3 đội hình (người nhà sau xe lăn, hàng 3 người, cắt mặt 4 người), 5 cự ly, 5 độ lệch che | Đánh giá khả năng tách người không phụ thuộc NMS |
| **S4_Photometric** | Nhiễu ánh sáng | 90 | 3 profile (ngược sáng, đèn khẩn cấp tối, dư sáng huỳnh quang), 2 NPC, 5 cự ly | Đánh giá độ bền vững trước mất tương phản |
| **S5_Hard_Neg** | Đạo cụ y tế rỗng | 60 | 4 đạo cụ (xe lăn trống, băng ca, kiosk, poster bác sĩ), 5 cự ly, 3 góc nhìn | Đo $FPR_{\text{empty}}$ (tránh robot chào ghế trống) |

---

## 5. Execution Pipeline & CLI Contracts

### 5.1 Quy trình Sinh Dữ liệu từ Godot 4
Chạy Godot Engine với offscreen render (hỗ trợ GPU thực):
```bash
# Trên desktop có màn hình (chạy tốc độ tối đa không lock vsync)
godot --path sim_vision/godot res://scenes/hospital_vision_bench.tscn -- --generate-dataset

# Trên Linux server / CI (Offscreen Vulkan)
xvfb-run -a godot --rendering-driver vulkan --path sim_vision/godot res://scenes/hospital_vision_bench.tscn -- --generate-dataset
```

### 5.2 Quy trình Đánh giá CLI Python
Chạy script đánh giá đối chứng độc lập (tuân thủ nguyên tắc Separation of Concerns, không dùng Jupyter Notebook):
```bash
python sim_vision/python/eval_vision_models.py \
    --dataset-dir data/vision_eval_600 \
    --conf-thres 0.45 \
    --iou-thres 0.50 \
    --output-report sim_vision/reports/vision_model_benchmark.md
```

---

## 6. Evaluation Metrics Suite (Robotics & Detection)

1. **Stratified Recall Nhóm Yếu thế ($Recall_{\text{wheelchair}}$)**:
   $$Recall_{\text{wheelchair}} = \frac{TP_{\text{wheelchair}}}{TP_{\text{wheelchair}} + FN_{\text{wheelchair}}} \quad (\text{với } IoU \ge 0.50)$$
2. **Sai số Góc hướng Điều khiển ($\Delta\theta_{\text{err}}$)**:
   $$\Delta\theta_{\text{err}} = \frac{|c_{x,\text{pred}} - c_{x,\text{gt}}|}{W} \times \alpha_h \quad (\text{với } \alpha_h = 75.0^\circ)$$
   Ngưỡng an toàn: $\Delta\theta_{\text{err}} \le 2.0^\circ$ để Robot không quay mặt lệch khỏi người đối thoại.
3. **Tỷ lệ Báo động giả trên Cảnh trống ($FPR_{\text{empty}}$)**:
   $$FPR_{\text{empty}} = \frac{\text{Số frame S5 có ít nhất 1 BBox } (\text{conf} \ge \tau_{\text{conf}})}{60}$$
4. **Độ trễ Suy luận (Latency Percentiles)**:
   Đo $P50, P95, P99$ tính bằng mili-giây trên CPU và GPU.

---

## 7. Acceptance Criteria & Quality Gates

| Tiêu chuẩn | Điều kiện Chấp thuận | Mục đích Hệ thống |
| :--- | :---: | :--- |
| **Dataset Completeness** | Đúng 600 frames, 0 lỗi cú pháp JSONL | Đảm bảo tính khoa học và toàn diện |
| **Ground Truth Precision** | 0 trường hợp AABB bloat $> 15\%$ | Ground Truth khớp sát cơ thể 3D |
| **Recall Xe lăn ($Recall_{\text{wheelchair}}$)** | $\ge 92.0\%$ | Không bỏ rơi bệnh nhân yếu thế |
| **Recall Đám đông ($Recall_{\text{occluded}}$)** | $\ge 85.0\%$ | Đảm bảo nhận diện tại quầy tiếp đón |
| **False Positive trên Đạo cụ rỗng ($FPR_{\text{empty}}$)** | $\le 2.0\%$ | Robot không nhầm xe lăn trống thành người |
| **Sai số góc hướng trung bình ($\Delta\theta$)** | $\le 2.0^\circ$ | Định hướng giao tiếp chính xác |
| **Độ trễ xử lý P95** | $\le 15.0\text{ ms}$ | Giữ vững tổng độ trễ hội thoại $1.21\text{ s}$ |

---

## 8. Associated Links

- Kế hoạch Thực nghiệm: [`docs/experiments/EXP_VISION_MODEL_BENCHMARK.md`](../experiments/EXP_VISION_MODEL_BENCHMARK.md)
- Nhật ký Tiến độ: [`docs/progress/VISION_EVAL_STATUS.md`](../progress/VISION_EVAL_STATUS.md)
- Module Detector Hiện tại: [`sim_vision/python/detector.py`](../../sim_vision/python/detector.py)
- Hợp đồng Giao tiếp WebSocket: [`sim_vision/python/protocol.py`](../../sim_vision/python/protocol.py)
- Báo cáo E2E Baseline Hiện tại: [`sim_vision/reports/e2e_summary.md`](../../sim_vision/reports/e2e_summary.md)
