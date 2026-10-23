# VISION_EVAL_STATUS.md — Vision Model Evaluation Phase Progress Status

- **Motivation/Background**: Theo dõi trạng thái triển khai theo thời gian thực của phase xây dựng dataset tự động trong Godot 4 và benchmark đối chứng các model Person Detection.
- **Purpose**: Ghi lại tiến độ chi tiết, nhật ký thực thi (activity log), các rào cản kỹ thuật và hành động tiếp theo theo đúng tiêu chuẩn quản trị dự án.
- **Overview Pipeline**: Khởi tạo trên branch `feat/vision-eval-benchmark`, cập nhật liên tục qua các mốc hoàn thành code và test.
- **Detailed Plan**: §1 Status Overview; §2 Chronological Activity Log; §3 Active Blockers & Decisions; §4 Immediate Next Steps.
- **References**: [`agents/rules/MD_CONVENTION.md`](../../agents/rules/MD_CONVENTION.md), [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md).
- **Created**: 2026-10-23T09:05:00+07:00
- **Last Updated**: 2026-10-23T09:42:00+07:00

---

## Status Overview

- **Governing Phase**: [`docs/phases/PHASE_VISION_EVAL_BENCHMARK.md`](../phases/PHASE_VISION_EVAL_BENCHMARK.md)
- **Governing Experiments**:
  - Đợt 1 (600 frames): [`docs/experiments/EXP_VISION_MODEL_BENCHMARK.md`](../experiments/EXP_VISION_MODEL_BENCHMARK.md)
  - Đợt 2 (48 Kịch bản, 8 Trục không gian): [`docs/experiments/EXP_VISION_48CASES_1200_BENCHMARK.md`](../experiments/EXP_VISION_48CASES_1200_BENCHMARK.md)
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
- **2026-10-23T09:28:00+07:00**: Tiếp thu yêu cầu mở rộng từ User: Khảo sát tư duy toàn diện các trường hợp che khuất kiến trúc, nội thất y tế, đám đông và bẫy phản xạ quang học.
- **2026-10-23T09:30:00+07:00**: Soạn thảo đặc tả 48 kịch bản (8 Trục kiểm thử không gian) tại [`EXP_VISION_48CASES_1200_BENCHMARK.md`](../experiments/EXP_VISION_48CASES_1200_BENCHMARK.md).
- **2026-10-23T09:30:30+07:00**: Nâng cấp `scene_dataset_generator.gd` hỗ trợ 8 Trục không gian (D1-D8) và kích hoạt GPU Vulkan kết xuất dataset `sim_vision/data/vision_eval_1200/`.
- **2026-10-23T09:39:39+07:00**: Kết xuất hoàn tất 855 frames sơ bộ đợt 2 trong $545.3\text{s}$.
- **2026-10-23T10:09:00+07:00**: Tinh chỉnh toàn bộ grid parameters trong Godot để đạt chuẩn chính xác **150 frames/trục** (tổng cộng đủ **1,200 frames**).
- **2026-10-23T10:24:18+07:00**: Kết xuất hoàn tất đúng **1,200 frames** (D1-D8 mỗi trục đúng 150 frames).
- **2026-10-23T10:25:25+07:00**: Chạy benchmark đối chứng chính thức 4 models (`yolov8n`, `yolo11n`, `yolo26n`, `intel/person-retail-0013`) trên 1,200 frames. Xuất báo cáo tại [`sim_vision/reports/vision_48cases_benchmark.md`](../../sim_vision/reports/vision_48cases_benchmark.md).
- **2026-10-23T10:27:15+07:00**: Hoàn thành Jupyter Notebook chuyên sâu [`notebooks/01_vision_model_benchmark_visualization.ipynb`](../../notebooks/01_vision_model_benchmark_visualization.ipynb) (11 MB) bao gồm 2D Spatial Heatmap, Thư viện 16 ảnh 3D, Radar Chart, Degeneracy Curves, và Live Inference Side-by-Side trên 6 ca biên thất bại nghiêm trọng nhất của CẢ 4 MODEL.

---

## 2. Active Blockers & Decisions Needed

- **Decision Finalized (Chốt Model Thị giác)**:
  - Tiếp tục chốt **`yolov8n.pt`** làm detector chính cho Robot trợ lý bệnh viện:
    - **$FPR_{\text{empty}} = 0.00\%$ (0 FP)** trên toàn bộ 90 frames bẫy đạo cụ y tế rỗng (xe lăn, giường, kiosk, poster).
    - **$100.00\%$ Recall** trên các kịch bản kiến trúc che khuất (D1), biến dạng cự ly cực đoan $0.6\text{m} - 10\text{m}$ (D5), góc quay lưng cơ thể (D6), và nhiễu sáng PBR ngược sáng/chói huỳnh quang (D7).
    - Độ trễ tối ưu nhất: **$8.6\text{ ms}$ P95** (đáp ứng tốc độ điều khiển thời gian thực $> 100\text{ FPS}$).
    - Sai số góc lái $\Delta\theta = 0.17^\circ$ cực nhỏ, đảm bảo bám mục tiêu trơn tru không giật góc quay.
  - **Loại bỏ `intel/person` (OpenVINO MobileNetV2-SSD)** khỏi vị trí detector chính vì:
    - Tụt Recall nghiêm trọng tại cự ly cực cận / cực viễn ($56.80\%$ ở D5) do giới hạn anchor grid cố định.
    - Tỷ lệ báo động giả trên đạo cụ rỗng cao ($5\text{ FP}$ trên Trục D8), khiến robot nhầm xe lăn trống thành bệnh nhân.
  - **Khuyến nghị nâng cấp tương lai (`yolo26n.pt` / `yolo11n.pt`)**:
    - `yolo26n.pt` đạt **Precision cao nhất ($92.13\%$)** nhờ kiến trúc NMS-Free (One-to-One Head), triệt tiêu các box trùng lặp khi người đứng sau quầy tiếp đón.
    - `yolo11n.pt` đạt **Recall cao nhất ($82.13\%$)** nhờ khối Spatial Attention C2PSA.

---

## 3. Immediate Next Steps

- [x] Tạo branch `feat/vision-eval-benchmark`.
- [x] So sánh phân tích nguyên lý `yolov8n` vs `intel/person`.
- [x] Xây dựng Scene Generator và bộ dataset 600 ảnh trong Godot 4.
- [x] Mở rộng ma trận 48 kịch bản (8 Trục không gian) và sinh dataset tại `sim_vision/data/vision_eval_1200/`.
- [x] Chạy benchmark đối chứng 4 models trên 48 kịch bản và xuất báo cáo khoa học 5W1H.
- [x] Cập nhật toàn bộ tài liệu đặc tả và nhật ký tiến độ.
- [ ] Push commit lên Git remote repository (`git push origin feat/vision-eval-benchmark`).
