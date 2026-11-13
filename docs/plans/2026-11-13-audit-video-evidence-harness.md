# Kế hoạch Thực hiện: Bộ Công Cụ Kiểm Toán Toàn Diện & Trích Xuất Video Bằng Chứng (Full-Scope Audit & Video Evidence Harness)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hệ thống tự động phát hiện lỗi tiềm ẩn của robot bệnh viện (nhận diện khuôn mặt, tráo người ReID, dắt người đứt khoảng cách, kẹt cửa/thang máy, đơ hội thoại), tự động đóng gói và xuất các clip video `.mp4` góc nhìn toàn cảnh 3D để người dùng nghiệm thu kiểm định, đồng thời chuẩn hóa hoàn toàn `.gitignore` cho dự án.

**Architecture:** Môi trường 3D Godot duy trì bộ đệm tròn (circular dashcam buffer ~150 JPEG frames trong RAM) ghi lại góc nhìn toàn cảnh thứ ba kèm HUD thông số. Khi module `AuditWatchdog` phát hiện vi phạm các ngưỡng lỗi (tether > 2.5m, ReID mismatch, kẹt > 5s, dialogue freeze), nó đóng gói các frame và gọi Python `cv2.VideoWriter` xuất file video `.mp4` độc lập kèm file telemetry `.json` tương ứng vào `sim_vision/reports/audit_evidence/`.

**Tech Stack:** Godot Engine 4.7.2 GDScript, Python 3.14 (.venv), OpenCV (`cv2.VideoWriter` codec `mp4v`), WebSocket JSON protocol.

**Spec:** [docs/plans/2026-11-13-audit-video-evidence-harness.md](file:///home/bush/Projects/Data-Science-Major-Project/docs/plans/2026-11-13-audit-video-evidence-harness.md)

## Global Constraints

- Không sửa đổi hoặc phá vỡ các tính năng sẵn có trên nhánh `main` (làm việc 100% trên nhánh `feat/audit-rl-engagement`).
- Không phụ thuộc vào binary `ffmpeg` hệ thống (sử dụng thư viện Python `opencv-python-headless` đã có sẵn trong `.venv`).
- Bộ đệm video trong RAM không được làm giảm FPS của mô phỏng Godot (lấy mẫu 15 FPS, nén JPEG nhẹ 75% chất lượng).
- Mọi tệp sinh ra từ quá trình audit (video `.mp4`, telemetry `.json`, thư mục tạm `/tmp/`) phải được thêm đầy đủ vào `.gitignore`.

## Review Focus

1. `DEFAULT_YOLO_PATH` nạp đúng model trọng số release mà không văng lỗi `FileNotFoundError`.
2. `dialogue_manager.gd` không bao giờ bị kẹt vĩnh viễn ở trạng thái `LISTEN` (có timeout bảo vệ 8.0s).
3. Đứt khoảng cách tether (> 2.5m) tự động sinh ra video `.mp4` có thời lượng 10-15s ghi lại khoảnh khắc lỗi.
4. Tráo người hoặc điểm tương đồng ReID rơi xuống dưới ngưỡng tự động ghi nhận mã lỗi `REID_MISMATCH`.
5. `.gitignore` lọc sạch toàn bộ video artifacts và các thư mục chưa được theo dõi (`release_models/`, `data/vision/`, `runs/`).

---

## Danh Sách Tác Vụ (Tasks)

### Task 1: Chuẩn hóa `.gitignore` và Khắc phục Lỗi Đường dẫn Mô hình YOLO

**Files:**
- Modify: `.gitignore`
- Modify: `sim_vision/python/detector.py:16-30`
- Test: `sim_vision/python/tests/test_detector.py`

- [ ] **Step 1:** Cập nhật `.gitignore` để bỏ qua các thư mục nặng và các artifacts sinh ra:
  - Thêm `release_models/` (ngoại trừ cấu hình/metadata)
  - Thêm `data/vision/`
  - Thêm `runs/`
  - Thêm `sim_vision/reports/audit_evidence/`
  - Thêm `*.mp4`, `*.avi`, `*.webm`
- [ ] **Step 2:** Kiểm tra `git status` đảm bảo working tree sạch sẽ, không còn các file rác bị track nhầm.
- [ ] **Step 3:** Trong `sim_vision/python/detector.py`, cập nhật hàm `resolve_model_path()` để tìm kiếm thêm trong thư mục `release_models/yolo26n_hospital_robot_v6_best/yolo26n_hospital_robot_best_map92.6.pt`.
- [ ] **Step 4:** Chạy kiểm thử `PYTHONPATH=sim_vision/python .venv/bin/pytest sim_vision/python/tests/test_detector.py` và kiểm tra nạp thử mô hình bằng `python -c "from detector import resolve_model_path; print(resolve_model_path())"`.
- [ ] **Step 5:** Commit thay đổi Task 1 với thông điệp: `fix(detector): resolve yolo release path and update gitignore for audit artifacts`.

---

### Task 2: Khắc phục Lỗ hổng Kẹt Vô Hạn Trạng thái Lắng nghe (`LISTEN Freeze`)

**Files:**
- Modify: `sim_vision/godot/scripts/dialogue_manager.gd:60-150`
- Test: `sim_vision/godot/tests/test_dialogue.gd`

- [ ] **Step 1:** Rà soát hàm quản lý chuyển trạng thái `_on_talk_started()` và `_set_dialogue_state()` trong `dialogue_manager.gd`.
- [ ] **Step 2:** Thêm timer bảo vệ `listen_timeout_s = 8.0` cho trạng thái `LISTEN`. Nếu sau 8.0s không nhận được tín hiệu kết thúc từ NPC, tự động gọi `_on_talk_aborted("listen timeout")` và chuyển robot về `ARRIVED` để giải phóng khóa vận tốc.
- [ ] **Step 3:** Chạy bộ kiểm thử GUT `test_dialogue.gd`:
  `/home/bush/.local/bin/godot --headless --path sim_vision/godot -s addons/gut/gut_cmdln.gd -gtest=res://tests/test_dialogue.gd`
- [ ] **Step 4:** Xác nhận tất cả test hội thoại đều PASS.
- [ ] **Step 5:** Commit thay đổi Task 2 với thông điệp: `fix(dialogue): add watchdog timeout to LISTEN state to prevent robot freeze`.

---

### Task 3: Xây dựng Module Ghép Video Python (`stitch_video.py`)

**Files:**
- Create: `scripts/audit/stitch_video.py`
- Create: `tests/test_stitch_video.py`

- [ ] **Step 1:** Viết test `tests/test_stitch_video.py` kiểm tra:
  - Nhận thư mục chứa các ảnh JPEG và thông số metadata.
  - Sử dụng `cv2.VideoWriter` mã hóa thành file `.mp4` chuẩn `mp4v`.
  - Xuất kèm file `.json` chứa metadata telemetry.
- [ ] **Step 2:** Chạy test để xác nhận test ban đầu fail (RED):
  `PYTHONPATH=. .venv/bin/pytest tests/test_stitch_video.py`
- [ ] **Step 3:** Cài đặt `scripts/audit/stitch_video.py` với CLI hỗ trợ:
  `python scripts/audit/stitch_video.py --frame-dir <dir> --output <out_path> --fps 15 --metadata <json_str>`
- [ ] **Step 4:** Chạy lại test xác nhận test PASS (GREEN).
- [ ] **Step 5:** Commit thay đổi Task 3 với thông điệp: `feat(audit): implement standalone video stitcher using opencv`.

---

### Task 4: Xây dựng Module Black-Box Watchdog & Dashcam trong Godot (`audit_watchdog.gd`)

**Files:**
- Create: `sim_vision/godot/scripts/audit_watchdog.gd`
- Modify: `sim_vision/godot/scripts/main.gd`
- Test: `sim_vision/godot/tests/test_audit_watchdog.gd`

- [ ] **Step 1:** Cài đặt `AuditWatchdog` kế thừa `Node`:
  - Giữ mảng tròn `Array[PackedByteArray]` (tối đa 150 JPEG frames từ `get_viewport().get_texture().get_image().save_jpg_to_buffer(75)`).
  - Định kỳ mỗi $0.066\text{s}$ ($15\text{ FPS}$) lấy 1 frame màn hình 3D.
  - Giám sát các điều kiện lỗi:
    + `TETHER_BREAK`: $d_{\text{tether}} > 2.5\text{m}$ liên tục $> 2.5\text{s}$ khi đang dắt.
    + `REID_MISMATCH`: ReID confidence $< 0.75$ khi đang dắt.
    + `STAGNATION`: Vận tốc $v < 0.05\text{m/s}$ kéo dài $> 5.0\text{s}$ khi chưa tới đích.
- [ ] **Step 2:** Khi phát hiện lỗi:
  - Tiếp tục đệm thêm 30 frames ($2.0\text{s}$).
  - Ghi các frame JPEG vào thư mục tạm `/tmp/audit_event_<id>/`.
  - Gọi tiến trình `stitch_video.py` chạy nền xuất video `.mp4` và file `.json` vào `sim_vision/reports/audit_evidence/`.
- [ ] **Step 3:** Viết bài kiểm thử `test_audit_watchdog.gd` kiểm tra việc kích hoạt trigger lỗi và lưu đệm.
- [ ] **Step 4:** Chạy GUT test xác nhận `test_audit_watchdog.gd` PASS.
- [ ] **Step 5:** Tích hợp `AuditWatchdog` vào `main.gd` (có thể bật/tắt qua cờ `--audit-mode` hoặc biến môi trường).
- [ ] **Step 6:** Commit thay đổi Task 4 với thông điệp: `feat(audit): integrate rolling dashcam watchdog into godot simulation`.

---

### Task 5: Xây dựng Kịch bản Kiểm toán Tự Động & Xuất Video Bằng Chứng (`run_audit.py`)

**Files:**
- Create: `scripts/audit/run_audit.py`
- Output: `sim_vision/reports/audit_evidence/AUDIT_REPORT.md` + danh sách video `.mp4`

- [ ] **Step 1:** Cài đặt `scripts/audit/run_audit.py`:
  - Tự động kiểm tra và khởi động các server phụ trợ cần thiết (YOLO :8765, STT :8766).
  - Khởi chạy Godot có giao diện 3D với cờ `--audit-mode` và chạy các kịch bản thử nghiệm:
    + **Kịch bản A:** Bệnh nhân đi chậm rẽ vào phòng khám 101/110.
    + **Kịch bản B:** Người lạ (bác sĩ/y tá) đi cắt ngang trước mặt robot khi đang dắt.
    + **Kịch bản C:** Dắt người đi thang máy lên Tầng 2.
- [ ] **Step 2:** Chạy thực tế script `python scripts/audit/run_audit.py --episodes 3`.
- [ ] **Step 3:** Kiểm tra thư mục `sim_vision/reports/audit_evidence/` để đảm bảo:
  - Các file video `fail_<ERR_TYPE>_<TIMESTAMP>.mp4` được sinh ra đầy đủ.
  - Có thể mở phát video bình thường, hiển thị rõ ràng góc nhìn thứ ba, robot, bệnh nhân và telemetry HUD.
  - File tổng kết `AUDIT_REPORT.md` được sinh ra tự động.
- [ ] **Step 4:** Commit thay đổi Task 5 với thông điệp: `feat(audit): add automated audit runner and generate video evidence report`.

---

### Task 6: Nghiệm Thu Hoàn Chỉnh & Báo Cáo Người Dùng

- [ ] **Step 1:** Kiểm tra lại toàn bộ `git status` đảm bảo không có file rác, file nhị phân video bị commit nhầm.
- [ ] **Step 2:** Chạy kiểm thử hồi quy toàn diện:
  - Pytest: `PYTHONPATH=sim_vision/python:. .venv/bin/pytest tests/ sim_vision/python/tests/`
  - GUT: `/home/bush/.local/bin/godot --headless --path sim_vision/godot -s addons/gut/gut_cmdln.gd -gconfig=res://.gutconfig.json`
- [ ] **Step 3:** Tổng hợp báo cáo kết quả kèm đường dẫn cụ thể tới các video `.mp4` để người dùng trực tiếp mở xem và đánh giá.
