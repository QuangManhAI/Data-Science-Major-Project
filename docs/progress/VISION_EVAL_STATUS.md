# VISION_EVAL_STATUS.md — Vision Model Evaluation Phase Progress Status

- **Motivation/Background**: Theo dõi trạng thái triển khai theo thời gian thực của phase xây dựng dataset tự động trong Godot 4 và benchmark đối chứng các model Person Detection.
- **Purpose**: Ghi lại tiến độ chi tiết, nhật ký thực thi (activity log), các rào cản kỹ thuật và hành động tiếp theo theo đúng tiêu chuẩn quản trị dự án.
- **Overview Pipeline**: Khởi tạo trên branch `feat/vision-eval-benchmark`, cập nhật liên tục qua các mốc hoàn thành code và test.
- **Detailed Plan**: §1 Status Overview; §2 Chronological Activity Log; §3 Active Blockers & Decisions; §4 Immediate Next Steps.
- **References**: [`agents/rules/MD_CONVENTION.md`](../../agents/rules/MD_CONVENTION.md), [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md).
- **Created**: 2026-10-23T09:05:00+07:00
- **Last Updated**: 2026-10-23T09:05:00+07:00

---

## Status Overview

- **Governing Phase**: [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md)
- **Governing Experiment**: [`docs/experiments/EXP_VISION_MODEL_BENCHMARK.md`](../experiments/EXP_VISION_MODEL_BENCHMARK.md)
- **Current Lifecycle State**: `[STATUS: IN_PROGRESS]`
- **Active Git Branch**: `feat/vision-eval-benchmark`
- **Primary Assignee**: Pair Programming AI Agent & User

---

## 1. Chronological Activity Log

- **2026-10-23T08:58:40+07:00**: Tạo và chuyển sang branch mới `feat/vision-eval-benchmark`.
- **2026-10-23T09:00:00+07:00**: Hoàn thành phân tích kỹ thuật First Principles: so sánh cơ chế `yolov8n` (Anchor-Free), `intel/person` (MobileNetV2-SSD Clustered Prior Boxes), và `yolo26n`/`yolo11n` (NMS-Free).
- **2026-10-23T09:02:00+07:00**: Giải quyết 4 bẫy toán học trích xuất Bounding Box trong Godot 4 (chống phình AABB bloat, tách ngữ nghĩa xe lăn, xử lý Near-plane, và Raycast che khuất $V_{\text{vis}}$).
- **2026-10-23T09:05:00+07:00**: Khởi tạo trọn bộ tài liệu quản trị và kế hoạch thực thi chuẩn convention:
  - Technical Phase Spec: [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md)
  - Experiment Spec: [`docs/experiments/EXP_VISION_MODEL_BENCHMARK.md`](../experiments/EXP_VISION_MODEL_BENCHMARK.md)
  - Live Status Tracking: [`docs/progress/VISION_EVAL_STATUS.md`](VISION_EVAL_STATUS.md)

---

## 2. Active Blockers & Decisions Needed

- **Decision 1 (Tùy chọn tải checkpoint `intel/person-0202` / `yolo26n`)**:
  - `yolov8n.pt` đã có sẵn tại [`models/yolov8n/yolov8n.pt`](../../models/yolov8n/yolov8n.pt).
  - Đối với `intel/person` và `yolo26n`/`yolo11n`, cần script tải tự động checkpoint chuẩn về `models/` trước khi chạy benchmark.
- **Decision 2 (Scene Benchmark trong Godot)**:
  - Cần tạo scene chuyên dụng [`sim_vision/godot/scenes/hospital_vision_bench.tscn`](../../sim_vision/godot/scenes/) tích hợp các prop đã có (`reception_counter`, `hospital_bed`, `sliding_door`, `waiting_bench`) cùng prefab NPC đứng và xe lăn để script `scene_dataset_generator.gd` vận hành.

---

## 3. Immediate Next Steps

- [ ] Triển khai script Godot [`sim_vision/godot/scripts/scene_dataset_generator.gd`](../../sim_vision/godot/scripts/) hỗ trợ sinh 5 slices (600 frames).
- [ ] Triển khai script Python [`sim_vision/python/eval_vision_models.py`](../../sim_vision/python/) với adapter cho `yolov8n`, `intel/person`, và `yolo26n`.
- [ ] Tạo unit test kiểm thử adapter detector và thuật toán IoU / Bearing error trong [`sim_vision/python/tests/`](../../sim_vision/python/tests/).
- [ ] Chạy thử nghiệm trên 30 ảnh mẫu để xác thực toàn vẹn pipeline trước khi chạy đầy đủ 600 frames.
