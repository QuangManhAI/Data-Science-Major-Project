# EXP_VISION_MODEL_BENCHMARK.md — Vision Models Comparative Benchmark Experiment

- **Motivation/Background**: Cần xác định thực nghiệm mô hình thị giác máy tính tối ưu nhất (giữa `yolov8n`, `intel/person-0202`, và `yolo26n`/`yolo11n`) cho Robot trợ lý bệnh viện dựa trên 4 chỉ số cốt lõi: khả năng bắt người đi xe lăn, khả năng chống nhầm lẫn đạo cụ rỗng, sai số góc lái Robot, và độ trễ phân tán.
- **Purpose**: Định nghĩa phương án thực nghiệm có kiểm soát (single-variable controlled experiment), ma trận đánh giá đối chứng, các tham số siêu cấp, và cấu trúc báo cáo kết quả 5W1H.
- **Overview Pipeline**: Tiếp nhận dataset 600 ảnh sinh tự động từ Godot 4 tại `data/vision_eval_600/`, chạy qua script CLI [`sim_vision/python/eval_vision_models.py`](../../sim_vision/python/eval_vision_models.py), xuất metric ra JSON và Markdown.
- **Detailed Plan**: §1 Objective & Hypothesis; §2 Controlled Variables; §3 Setup & Hyperparameters; §4 Empirical Evaluation Plan (5W1H); §5 Failure Modes Analysis; §6 Reproduction Protocol; §7 Associated Links.
- **References**: [`agents/rules/MD_CONVENTION.md`](../../agents/rules/MD_CONVENTION.md), [`agents/rules/RESULTS_REPORTING.md`](../../agents/rules/RESULTS_REPORTING.md), [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md).
- **Created**: 2026-10-23T09:05:00+07:00
- **Last Updated**: 2026-10-23T09:05:00+07:00

---

[STATUS: ACTIVE]

## Metadata

- **Experiment ID**: `EXP-VISION-01`
- **Title**: `Comparative Evaluation of yolov8n vs. intel/person vs. yolo26n on Godot 4 Hospital Dataset`
- **Target Script**: [`sim_vision/python/eval_vision_models.py`](../../sim_vision/python/eval_vision_models.py)
- **Dataset Path**: `data/vision_eval_600/annotations.jsonl`
- **Report Output**: `sim_vision/reports/vision_model_benchmark.md`

---

## 1. Objective & Hypothesis

- **Hypothesis 1 (Vulnerable Pose Shift)**: Các mô hình SSD Anchor-based như `intel/person-detection-0202` sẽ bị suy giảm Recall trầm trọng ($Recall_{\text{wheelchair}} < 70\%$) khi bệnh nhân ngồi xe lăn ở cự ly gần ($0.9 - 1.8\text{ m}$) do tỷ lệ prior box ($w/h \approx 0.4$) không khớp tỷ lệ khung hình thực tế ($w/h \approx 0.9$). Trong khi đó, `yolov8n` và `yolo26n` (Anchor-Free) sẽ duy trì $Recall_{\text{wheelchair}} \ge 92\%$.
- **Hypothesis 2 (Crowd Occlusion)**: Tại quầy tiếp đón đông người (người nhà đứng đẩy sau lưng xe lăn), mô hình NMS-Free `yolo26n` sẽ vượt trội hơn `yolov8n` ít nhất $+6\%$ Recall nhờ loại bỏ hiện tượng Greedy NMS triệt tiêu nhầm box trùng trục.
- **Hypothesis 3 (Robotics Steering Error)**: Sai số góc hướng ($\Delta\theta_{\text{err}}$) của các mô hình Anchor-Free trên người ngồi xe lăn sẽ nhỏ hơn $2.0^\circ$, đảm bảo Robot không bị lắc đầu/co giật góc quay.

---

## 2. Controlled Variables (Single-Variable Principle)

- **Variable Changed**: Kiến trúc mạng nơ-ron nhận diện đối tượng (Detection Backbone & Head: `intel/person-0202` MobileNetV2-SSD vs `yolov8n.pt` Anchor-free vs `yolo26n.pt` NMS-Free).
- **Held Constant**:
  - Tập dữ liệu kiểm thử: Cố định 600 frames và Ground Truth tại `data/vision_eval_600/annotations.jsonl`.
  - Độ phân giải đầu vào thực tế từ Camera Robot: $640 \times 480$ (kèm padding/resizing tương ứng của từng model adapter).
  - Ngưỡng Confidence Threshold: $\tau_{\text{conf}} = 0.45$.
  - Ngưỡng IoU Match Threshold: $\tau_{\text{IoU}} = 0.50$.
  - Thiết bị phần cứng đánh giá: Cùng chạy trên môi trường CPU/GPU chuẩn của dự án.

---

## 3. Setup & Hyperparameters

### 3.1 Danh sách Candidate Models
1. **Model A (Legacy Baseline - OpenVINO SSD)**:
   - Tên: `intel/person-detection-0202` (hoặc `person-detection-retail-0013`)
   - Định dạng: OpenVINO IR (`.xml`, `.bin`), precision FP16/INT8.
2. **Model B (Current System Baseline - Ultralytics YOLOv8)**:
   - Tên: `yolov8n.pt` (3.2M params, 8.7 GFLOPs)
   - Thiết lập: `classes=[0]` (chỉ dự đoán person).
3. **Model C (Next-Gen Candidate - NMS-Free YOLO26 / YOLO11)**:
   - Tên: `yolo26n.pt` / `yolo11n.pt` (~2.4M - 2.6M params)
   - Thiết lập: One-to-One NMS-Free head, `classes=[0]`.

### 3.2 Bộ Tham số Đánh giá
```yaml
evaluation_config:
  dataset_root: "data/vision_eval_600"
  annotations_file: "data/vision_eval_600/annotations.jsonl"
  camera_fov_h_deg: 75.0
  conf_threshold: 0.45
  iou_threshold: 0.50
  slices:
    - S1_Baseline
    - S2_Vulnerable
    - S3_Crowd_Occ
    - S4_Photometric
    - S5_Hard_Neg
```

---

## 4. Empirical Evaluation Plan (5W1H)

### 4.1 Bảng Tổng hợp Kết quả Thực nghiệm Dự kiến

| Model Architecture | Precision | Recall (All) | Recall (Wheelchair) | Recall (Occluded) | FPR (Empty Props) | Mean $\Delta\theta_{\text{err}}$ (deg) | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `intel/person-0202` | TBD | TBD | TBD (< 70%?) | TBD | TBD | TBD | TBD |
| `yolov8n.pt` (Baseline) | TBD | TBD | TBD ($\ge 92\%$) | TBD ($\approx 82\%$) | TBD ($\le 2\%$) | TBD ($< 1.8^\circ$) | TBD ($\approx 8\text{ ms}$) |
| `yolo26n.pt` (Candidate) | TBD | TBD | TBD ($\ge 95\%$) | TBD ($\ge 88\%$) | TBD ($\le 1\%$) | TBD ($< 1.5^\circ$) | TBD ($\approx 5\text{ ms}$) |

### 4.2 Chi tiết 5W1H khi Chạy Benchmark
- **What**: Đánh giá độ chính xác, tỷ lệ bắt bệnh nhân xe lăn, tỷ lệ báo động giả trên đạo cụ rỗng, và độ lệch góc lái Robot.
- **Why**: Lựa chọn model thị giác chính thức cho phiên bản phát hành của Robot dẫn đường bệnh viện.
- **When**: Thực thi trên branch `feat/vision-eval-benchmark`.
- **Where**: Chạy offline qua CLI tại máy host; ghi kết quả vào [`sim_vision/reports/vision_model_benchmark.md`](../../sim_vision/reports/vision_model_benchmark.md).
- **Who**: Nhóm Kỹ thuật AI Phân hệ Vision.
- **How**: Chạy script [`sim_vision/python/eval_vision_models.py`](../../sim_vision/python/eval_vision_models.py) quét toàn bộ 600 ảnh JSONL.

---

## 5. Failure Modes Analysis

1. **Failure Mode 1: Trượt Xe lăn (Wheelchair False Negative)**
   - Cơ chế: Do khung xe lăn chiếm diện tích lớn, model nhầm lẫn là vật thể đồ nội thất.
   - Chỉ số cảnh báo: $Recall_{\text{wheelchair}} < 90\%$.
2. **Failure Mode 2: Báo động giả trên Đạo cụ rỗng (Empty Prop False Positive)**
   - Cơ chế: Chiếc xe lăn trống hoặc poster bác sĩ trên tường bị nhận nhầm thành người thật.
   - Chỉ số cảnh báo: $FPR_{\text{empty}} > 2\%$.
3. **Failure Mode 3: Rung lắc Tâm Box (Temporal Center Jitter)**
   - Cơ chế: Khi người đứng yên nói chuyện 3-5s, tâm box nhảy qua lại làm Robot rung giật góc quay.
   - Chỉ số cảnh báo: Độ lệch chuẩn vị trí tâm $\sigma_{\text{cx}} > 4.0\text{ px}$.

---

## 6. Reproduction Protocol

```bash
# Bước 1: Sinh bộ dữ liệu 600 ảnh từ Godot (nếu chưa có)
godot --path sim_vision/godot res://scenes/hospital_vision_bench.tscn -- --generate-dataset

# Bước 2: Chạy benchmark tự động trên các model
python sim_vision/python/eval_vision_models.py \
    --models yolov8n,intel_person_0202,yolo26n \
    --dataset-dir data/vision_eval_600 \
    --output-report sim_vision/reports/vision_model_benchmark.md
```

---

## 7. Associated Links

- Master Phase Doc: [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md)
- Live Progress Tracker: [`docs/progress/VISION_EVAL_STATUS.md`](../progress/VISION_EVAL_STATUS.md)
- Báo cáo Benchmark Hoàn chỉnh: [`sim_vision/reports/vision_model_benchmark.md`](../../sim_vision/reports/vision_model_benchmark.md)
