# VISION_EVAL_STATUS.md — Vision Model Evaluation Phase Progress Status

- **Motivation/Background**: Theo dõi trạng thái triển khai theo thời gian thực của phase xây dựng dataset tự động trong Godot 4 và benchmark đối chứng các model Person Detection.
- **Purpose**: Ghi lại tiến độ chi tiết, nhật ký thực thi (activity log), các rào cản kỹ thuật và hành động tiếp theo theo đúng tiêu chuẩn quản trị dự án.
- **Overview Pipeline**: Khởi tạo trên branch `feat/vision-eval-benchmark`, cập nhật liên tục qua các mốc hoàn thành code và test.
- **Detailed Plan**: §1 Status Overview; §2 Chronological Activity Log; §3 Active Blockers & Decisions; §4 Immediate Next Steps.
- **References**: [`agents/rules/MD_CONVENTION.md`](../../agents/rules/MD_CONVENTION.md), [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md).
- **Created**: 2026-10-23T09:05:00+07:00
- **Last Updated**: 2026-10-23T09:25:00+07:00

---

## Status Overview

- **Governing Phase**: [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md)
- **Governing Experiment**: [`docs/experiments/EXP_VISION_MODEL_BENCHMARK.md`](../experiments/EXP_VISION_MODEL_BENCHMARK.md)
- **Current Lifecycle State**: `[STATUS: COMPLETED]`
- **Active Git Branch**: `feat/vision-eval-benchmark`
- **Primary Assignee**: Pair Programming AI Agent & User

---

## 1. Chronological Activity Log

- **2026-10-23T08:58:40+07:00**: Tạo và chuyển sang branch mới `feat/vision-eval-benchmark`.
- **2026-10-23T09:00:00+07:00**: Hoàn thành phân tích kỹ thuật First Principles: so sánh cơ chế `yolov8n` (Anchor-Free), `intel/person` (MobileNetV2-SSD Clustered Prior Boxes), và `yolo26n`/`yolo11n` (NMS-Free).
- **2026-10-23T09:05:00+07:00**: Khởi tạo trọn bộ tài liệu quản trị (`PHASE_VISION_EVAL_BENCHMARK.md`, `EXP_VISION_MODEL_BENCHMARK.md`, `VISION_EVAL_STATUS.md`).
- **2026-10-23T09:12:00+07:00**: Triển khai `IntelOpenVINODetector` và kiểm tra tương thích nạp `person-detection-retail-0013` qua OpenVINO native runtime.
- **2026-10-23T09:16:00+07:00**: Hoàn thiện kịch bản Godot `scene_dataset_generator.gd`, cấu hình scene `hospital_vision_bench.tscn`, và engine benchmark `eval_vision_models.py`.
- **2026-10-23T09:22:48+07:00**: Chạy tự động sinh đủ 600 frames JPEG ($640 \times 480$) kèm nhãn Ground Truth hình học chính xác cao tại `sim_vision/data/vision_eval_600/`.
- **2026-10-23T09:23:32+07:00**: Chạy benchmark đối chứng 4 models (`yolov8n.pt`, `yolo11n.pt`, `yolo26n.pt`, `intel/person-retail-0013`) và xuất báo cáo hoàn chỉnh tại `sim_vision/reports/vision_model_benchmark.md`.

---

## 2. Active Blockers & Decisions Needed

- **Decision Finalized (Chốt Model Thị giác)**:
  - Tiếp tục giữ **`yolov8n.pt`** làm detector chính cho Robot trợ lý do đạt độ chính xác $89.04\%$, latency thấp nhất ($8.1\text{ ms}$ P95), và quan trọng nhất là **$FPR_{\text{empty}} = 0.00\%$** (hoàn toàn không nhận nhầm xe lăn trống hay poster thành người).
  - Không sử dụng `intel/person` (SSD) làm model chính do tỷ lệ báo động giả trên đạo cụ rỗng lên tới **$6.60\%$** (nhận nhầm 18 lần trên 60 frames không có người).
  - Đề xuất sẵn `yolo11n.pt` làm phương án dự phòng khi cần tăng cường độ nhạy trong môi trường ánh sáng phức tạp ($Recall = 80.00\%$ trên Slice S4).

---

## 3. Immediate Next Steps

- [x] Tạo branch `feat/vision-eval-benchmark`.
- [x] So sánh phân tích nguyên lý `yolov8n` vs `intel/person`.
- [x] Xây dựng Scene Generator và bộ dataset 600 ảnh trong Godot 4.
- [x] Chạy benchmark và xuất báo cáo khoa học 5W1H.
- [ ] Push commit lên Git remote repository khi có thông tin chứng thực (credentials).
