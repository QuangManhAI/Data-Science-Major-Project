# Tiến độ

| Prompt | Trạng thái | Test | Ghi chú |
| :-- | :-- | :-- | :-- |
| G1 | Xong | GUT 3/3 pass | |
| G2 | Xong | pytest 36/36 pass | |
| G3 | Xong | GUT 9/9 pass | Ảnh render đạt |
| G4 | Xong | GUT 16/16 pass, gate 320/320 | Gate đạt 100% mọi mốc |
| G5 | Xong | GUT 27/27 pass | |
| G6 | Xong | GUT 41/41 pass | 5 Hz, nối lại 2.0 s |
| G7 | Xong | GUT 61/61 pass | |
| G8 | Xong | GUT 69/69 pass | H_TARGET đổi 0.65 ➔ 0.85 |
| G9 | Xong | GUT 78/78 pass | 148 dòng / 30 s |
| G10 | Xong | GUT 81/81, pytest 40/40 | Batch 10/10 |

## G1: Bootstrap & Config (2026-10-21)

**Đã xong**
- Cấu trúc thư mục theo Mục 1: `godot/{scenes,scripts,assets,tests,tools,addons}`, `python/tests`, `logs/`.
- `godot/project.godot` (Forward+, bật plugin GUT), `scenes/main.tscn` rỗng.
- `godot/scripts/config.gd`: 19 hằng số Mục 2.1, `enum RobotState`, `state_name()`.
- GUT 9.7.1, `godot/tests/test_config.gd`.
- `python/requirements.txt` có ghim phiên bản; venv `python/.venv`.
- `.gitignore`, `ASSUMPTIONS.md`, `PROGRESS.md`, `godot/assets/CREDITS.md`.

**Kiểm chứng**
- [x] `godot --headless --import` và chạy scene chính 60 frame: không có ERROR/WARNING.
- [x] `pip install -r requirements.txt` chạy được.
- [x] GUT: `test_config.gd` 3/3 pass (23 assert).

**Chạy test**
```bash
cd sim_vision/godot
godot --headless -s addons/gut/gut_cmdln.gd -gconfig=res://.gutconfig.json
```

**Vấn đề tồn tại:** chưa có.

## G2: YOLO Server (2026-10-21)

**Đã xong**
- `python/protocol.py`: `parse_frame`, `build_detections`, `build_error`, `ProtocolError` (không phụ thuộc ultralytics).
- `python/detector.py`: interface `Detector`, `normalize_boxes`, `YoloDetector` (import ultralytics trễ).
- `python/yolo_server.py`: server WebSocket `127.0.0.1:8765`, `--model`, `--port`; suy luận qua `asyncio.to_thread`; mỗi client chỉ xử lý frame mới nhất.
- `python/send_image.py`: gửi ảnh thử. Test: `python/tests/` (FakeDetector, không gọi YOLO thật).

**Kiểm chứng**
- [x] `pytest python/tests`: 36/36 pass (JSON hỏng, thiếu trường, sai kiểu, JPEG hỏng, tọa độ chuẩn hóa, sắp conf, server không dừng sau lỗi, frame dồn chỉ xử lý frame mới nhất, thiếu model thoát mã ≠ 0).
- [x] YOLO thật: `bus.jpg` ➔ 3 người, conf 0.87/0.85/0.83, 34.1 ms; `zidane.jpg` ➔ 2 người, conf 0.83/0.83, 22.1 ms (CPU).
- [x] Model không tồn tại ➔ `ERROR: model file not found: ...`, exit 1, không tự tải. (Thử bằng đường dẫn không tồn tại thay vì xóa file thật của người dùng.)

**Chạy**
```bash
cd sim_vision/python
.venv/bin/python -m pytest tests
.venv/bin/python yolo_server.py --model yolov8n.pt
.venv/bin/python send_image.py <ảnh.jpg>
```

**Vấn đề tồn tại:** chưa có.

## G3: Căn phòng (2026-10-21)

**Đã xong**
- `godot/scenes/room.tscn`: phòng 8 × 6 × 3 m, gốc ở tâm sàn; sàn, tường, trần PBR (albedo + normal + roughness, thêm AO cho sàn và trần); cửa sổ ở tường bắc; nắng (`DirectionalLight3D`) và 2 đèn trần (`OmniLight3D`), đều bật bóng; `WorldEnvironment` SDFGI + SSAO + ACES; 4 đồ vật sát tường có `StaticBody3D` + collision.
- `godot/scripts/room.gd` (`class_name Room`): `get_free_position`, `get_furniture_footprints`, `distance_to_rect`, `is_clear`.
- Asset CC0 trong `godot/assets/`, nguồn và hướng dẫn tải lại ở `godot/assets/CREDITS.md`.
- `godot/tests/test_room.gd`.

**Kiểm chứng**
- [x] Render 4 góc nhìn (cửa sổ Godot thật, không headless): sàn và tường có vân, nắng qua cửa sổ tạo mảng sáng và bóng, đồ vật có bóng, không có mảng đen.
- [x] GUT 9/9 pass (gồm 100 lần gọi `get_free_position(0.5)`: đều nằm trong phòng và cách đồ vật ≥ 0.5 m).
- Lỗi tìm được nhờ test: ghế bành xoay 45° làm hộp collision lòi ra ngoài tường ➔ đã dời ghế vào trong.

**Vấn đề tồn tại**
- Ảnh hơi sáng (trần gần như trắng). Chưa chỉnh vì chưa biết ảnh hưởng tới YOLO; sẽ xem lại ở cổng kiểm tra G4.

## G4: Người tả thực (2026-10-21)

**Đã xong**
- Model Rocketbox `Male_Adult_06` (MIT) trong `godot/assets/models/human_m06/`; hướng dẫn tải từng bước và nguồn ở `godot/assets/CREDITS.md`.
- `godot/tools/extract_human_anims.gd` ➔ `anims/idle.res`, `anims/walk.res`.
- `godot/scenes/human.tscn` (`CharacterBody3D`, material PBR cho thân, đầu, tóc; capsule va chạm; `AnimationPlayer` thư viện `human/idle`, `human/walk`) và `godot/scripts/human.gd` (`walks`, `waypoints`, 0.5 m/s).
- `godot/tools/detect_gate.tscn` + `detect_gate.gd`; `godot/tests/test_human.gd`.

**Kiểm chứng**
- [x] Render 4 hướng: da, tóc, quần áo có texture, không có màu hồng/trắng do thiếu texture; chân chạm sàn; người cao ≈ 1.75 m.
- [x] GUT: 16/16 pass (thêm 7 test cho human: collision, animation, material, chiều cao, `step_towards`, waypoint, tốc độ đi 0.5 m/s).
- [x] Cổng nhận diện (YOLO thật, `YOLO_CONF` giữ 0.4, ngưỡng tính trúng 0.5): **20/20 ở cả 16 mốc**, `infer_ms` trung bình 27.2 ms (CPU).
  - Yêu cầu ≥ 90% ở 1–4 m với yaw 0°/45°: đạt 100%.
  - Yêu cầu ≥ 70% ở 2 m với yaw 90°/180°: đạt 100%.

`logs/detect_gate.csv`:

| distance_m | yaw_deg | hits/n | mean_top_conf |
| :-- | :-- | :-- | :-- |
| 1 | 0 / 45 / 90 / 180 | 20/20 cả 4 | 0.935 / 0.919 / 0.868 / 0.936 |
| 2 | 0 / 45 / 90 / 180 | 20/20 cả 4 | 0.925 / 0.908 / 0.845 / 0.931 |
| 4 | 0 / 45 / 90 / 180 | 20/20 cả 4 | 0.906 / 0.881 / 0.821 / 0.905 |
| 6 | 0 / 45 / 90 / 180 | 20/20 cả 4 | 0.899 / 0.872 / 0.792 / 0.897 |

**Chạy cổng kiểm tra**
```bash
cd sim_vision/python && .venv/bin/python yolo_server.py --model yolov8n.pt   # cửa sổ 1
cd sim_vision/godot && godot --path . res://tools/detect_gate.tscn           # cửa sổ 2
```

**Vấn đề tồn tại**
- Ảnh mẫu `d1_yaw0` (ảnh đầu tiên của lượt chạy) bị ngược sáng: người đứng sát tường tây, xa đèn. Vẫn nhận ra (conf 0.935), nhưng phần chính diện của người tối hơn ngoài đời thật.
- Gate đạt 100% vì cảnh đơn giản (1 người, nền sáng). Chưa thử khi người bị đồ vật che khuất một phần.

## G5: Robot, Camera & Ô góc nhìn (2026-10-21)

**Đã xong**
- `godot/scenes/robot.tscn` + `godot/scripts/robot.gd` (`set_command`, `get_camera_viewport`, `front_obstacle_distance`, differential drive bằng `move_and_slide`).
- `godot/scenes/view_overlay.tscn` + `godot/scripts/view_overlay.gd` (`show_detections`, vẽ box bằng `_draw` trên node `Boxes`).
- `godot/tests/test_robot.gd` (11 test).

**Kiểm chứng**
- [x] `set_command(0.3, 0.5)`: robot đi vòng tròn, khoảng cách xa nhất tới điểm xuất phát 1.197 m (lý thuyết 1.2 m).
- [x] Chạy `V_MAX` thẳng vào tường: dừng ở x = 3.738 (giới hạn 3.75), không xuyên tường.
- [x] Ô góc nhìn hiện đúng hình camera robot, ảnh thay đổi giữa các frame (không đứng hình); box, nhãn, trạng thái, chữ tiếng Việt hiển thị đúng.
- [x] GUT 27/27 pass.

**Vấn đề tồn tại**
- Test raycast ban đầu fail khi chạy chung với test khác: robot của test trước chưa kịp giải phóng nên `move_and_slide` đẩy robot mới lùi 0.35 m. Đã sửa bằng cách đặt robot và tường của test đó ở x = 50. Không phải lỗi của `robot.gd`.
- Camera người xem đặt ngoài phòng thì bị trần che ➔ G8 sẽ cho camera này bỏ qua lớp render của trần.

## G6: Vision Client (2026-10-21)

**Đã xong**
- `godot/scripts/vision_client.gd` (`class_name VisionClient`): `setup`, `parse_message`, signal `detections_received`, `connection_changed`; chụp 5 Hz, JPEG 0.8, base64; bỏ frame cũ/trễ; tự nối lại.
- `godot/tests/test_vision_client.gd` (14 test).

**Kiểm chứng**
- [x] GUT 41/41 pass.
- [x] Server thật + cảnh (người cách robot 3 m): nhận đúng 5 kết quả/giây (giây đầu là 4 vì mới kết nối), conf ~0.92, `infer_ms` 34–51 ms.
- [x] Tắt server ➔ `connection_changed(false)` ngay. Bật lại ➔ `connection_changed(true)` sau 2.04 s kể từ lệnh bật (tính cả thời gian nạp model), dưới 3 s.
- [x] Hết lượt chạy, `sent_at` còn 0 mục.

**Vấn đề tồn tại**
- Lần chạy kiểm chứng đầu tiên không nhận được kết quả nào: script dựng cảnh gọi `robot.get_camera_viewport()` trước khi robot `_ready` nên nhận null. Đã sửa `robot.gd` (xem ASSUMPTIONS [G6]). Scene bình thường (G8) không gặp lỗi này.

## G7: Controller (2026-10-21)

**Đã xong**
- `godot/scripts/controller.gd` (`class_name Controller`, `step`, `pick_target`), đúng thứ tự điều kiện của pseudocode G7.2.
- `godot/tests/test_controller.gd`: 9 ví dụ của bảng G7.3 + 11 test biên (cx = 0/1, h ≈ 0 kẹp `V_MAX`, omega luôn trong giới hạn, h = `H_TARGET`, cận dưới vùng trễ, `t_lost` = `LOST_TIMEOUT_S`, ARRIVED vừa mất dấu, |e| = `ALIGN_TOL`, bằng conf chọn box cao hơn, hàm thuần).

**Kiểm chứng**
- [x] GUT 61/61 pass (20 test controller).
- [x] Dò vết `cx=0.55, h=0.30`: e = 0.05 ➔ omega = −2.0 × 0.05 = −0.1; v = 1.5 × (0.65 − 0.30) = 0.525; |e| ≤ 0.25 nên giữ v ➔ khớp test.
- [x] Dấu omega: `omega = -K_ANG * e` (`controller.gd`). Người bên phải (cx > 0.5, e > 0) ➔ omega < 0 ➔ quay phải, đúng Mục 2.4. Kết hợp với G5 (omega > 0 làm `rotation.y` tăng, hướng trước lệch về −X, tức bên trái) ➔ cả chuỗi điều khiển đúng chiều.

**Vấn đề tồn tại:** chưa có.

## G8: Main Scene & An toàn (2026-10-21)

**Đã xong**
- `godot/scenes/main.tscn` (Room, Human, Robot, Controller, VisionClient, ViewerCamera, UI/ViewOverlay) — scene chạy chính của project.
- `godot/scripts/main.gd` (`class_name Main`): luồng `detections_received ➔ controller.step ➔ apply_safety ➔ robot.set_command`; `apply_safety` và `pick_start_positions` là hàm thuần; `@export seed`, `@export controller_path`; phím R/Esc; signal `cycle_completed` cho G9.
- `godot/tests/test_main.gd` (8 test).

**Kiểm chứng**
- [x] Bật server, chạy scene (seed 0, người cách 4.32 m): SEARCH (quay 0.5 rad/s) ➔ thấy người ở giây 8.4 ➔ APPROACH (quay thẳng rồi tiến) ➔ ARRIVED ở giây 21.2. Ô góc nhìn hiện box `person 0.92` và trạng thái.
- [x] Tắt server giữa chừng: lệnh về (0, 0), robot đứng yên trong 3 s (dịch chuyển 0.0000 m), trạng thái giữ nguyên, ô báo "YOLO: mất kết nối".
- [x] GUT 69/69 pass.

**Vấn đề tồn tại — CẦN NGƯỜI DÙNG QUYẾT ĐỊNH**
- Robot dừng (ARRIVED) ở **khoảng cách thật 2.21 m**, trong khi G10 chỉ tính là thành công nếu dừng trong khoảng **[0.8, 1.5] m**. Theo hình học, camera cao 1.2 m, FOV dọc 70°, người cao 1.75 m thì `h = H_TARGET = 0.65` ứng với khoảng 2.1–2.2 m. Muốn dừng ở 1.5 m thì cần `h` ≈ 0.80. Đây là mâu thuẫn giữa Mục 2.1 (`H_TARGET`), G5 (FOV 70°) và tiêu chí G10. Chưa sửa gì; sẽ hỏi trước khi làm G10.

## G9: Run Logger (2026-10-21)

**Đã xong**
- `godot/scripts/run_logger.gd` (`class_name RunLogger`): `open`, `log_row`, `flush`, `close`, `format_row`, `signed_angle_deg`; ghi đệm, flush mỗi 1 s và khi thoát.
- `main.tscn` thêm node `RunLogger`; `main.gd` thêm `build_log_row`, `human_angle_deg`.
- `godot/tests/test_run_logger.gd` (9 test).

**Kiểm chứng**
- [x] GUT 78/78 pass.
- [x] Chạy scene chính 30 s (seed 0) ➔ `logs/run_20261021_170133.csv`: 148 dòng (5.04 dòng/s), đủ 13 cột đúng thứ tự.
- [x] pandas: `true_distance_m` giảm đều 4.32 ➔ 2.21 m trong suốt APPROACH (64/64 bước không tăng). `corr(h, true_distance_m)` = −0.969, `corr(cx, true_angle_deg)` = −0.999. Dòng không có detection thì `conf/cx/h` trống.

**Vấn đề tồn tại**
- Vẫn là mâu thuẫn ở G8: ARRIVED ở 2.21 m, ngoài khoảng thành công [0.8, 1.5] m của G10.

## Đổi H_TARGET 0.65 ➔ 0.85 (2026-10-21)

Người dùng chọn đổi `H_TARGET` để G10 có thể đạt. Cập nhật config, file kế hoạch (Mục 2.1, bảng G7.3, dò vết G7.4) và test; GUT 78/78 pass.

Đo lại bằng chạy thật (seed 0, `logs/run_20261021_170425.csv`): ARRIVED ở **1.46 m** (trước đây 2.21 m), sát biên trên 1.5 m của G10. Khi người ở gần thì chân ra khỏi khung hình nên `h` tăng rất chậm (0.81 ở 1.63 m, 0.825 ở 1.55 m, 0.845 ở 1.46 m); APPROACH mất 14 s, phần cuối robot chỉ nhích chậm vì v = 1.5·(0.85 − h) rất nhỏ.

## G10: Test tổng thể & README (2026-10-21)

**Đã xong**
- `godot/tools/batch_run.tscn` + `batch_run.gd` (`class_name BatchRun`, hàm thuần `is_success`, `touches`, `format_summary_row`); `godot/tests/test_batch_run.gd`.
- `python/analyze_logs.py` + `python/tests/test_analyze_logs.py`.
- `README.md` (cài đặt, lấy model người, chạy server, chạy scene, test, batch, giải thích hằng số).

**Kiểm chứng**
- [x] Batch 10 lượt (seed 0–9): **10/10 thành công** (yêu cầu ≥ 9/10).
- [x] **0 va chạm với người** ở mọi lượt.
- [x] `mean_infer_ms` = **33.9 ms** trên CPU (yêu cầu < 100 ms).
- [x] Thời gian tới đích trung bình 18.65 s (13.97 – 23.55 s).
- [x] GUT 81/81, pytest 40/40.
- [ ] "Làm theo README từ máy sạch chạy được": **chưa kiểm**, mới chạy trên máy phát triển.

`logs/batch_summary.csv`:

| seed | success | time_to_arrive_s | final_distance_m | collisions | frames | mean_infer_ms |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | true | 20.52 | 1.450 | 0 | 103 | 34.4 |
| 1 | true | 13.97 | 1.499 | 0 | 70 | 33.2 |
| 2 | true | 20.37 | 1.461 | 0 | 102 | 34.0 |
| 3 | true | 23.55 | 1.496 | 0 | 118 | 33.8 |
| 4 | true | 17.93 | 1.498 | 0 | 90 | 34.3 |
| 5 | true | 15.17 | 1.478 | 0 | 76 | 33.6 |
| 6 | true | 22.98 | 1.498 | 0 | 115 | 33.9 |
| 7 | true | 16.97 | 1.453 | 0 | 85 | 33.7 |
| 8 | true | 20.73 | 1.455 | 0 | 104 | 34.0 |
| 9 | true | 14.32 | 1.485 | 0 | 72 | 34.2 |

**Vấn đề tồn tại**
- **Biên an toàn mỏng:** cả 10 lượt dừng ở 1.45–1.50 m, 4 lượt ở 1.496–1.499 m, sát biên trên 1.5 m. Lý do: khi người ở gần thì chân ra khỏi khung hình nên `h` tăng rất chậm theo khoảng cách (xem `logs/h_vs_distance.png`). Chỉ cần nhiễu nhỏ (đổi model người, cao hơn/thấp hơn, đổi FOV) là có thể dừng ở > 1.5 m và bị tính là thất bại. Nếu cần biên rộng hơn: tăng `H_TARGET` lên ~0.88–0.90, hoặc đổi tiêu chí thành công. Chưa đổi gì, chờ người dùng quyết định.
- Phần cuối APPROACH chậm (v = 1.5·(0.85 − h) rất nhỏ khi h gần 0.85), chiếm phần lớn thời gian tới đích.
- Batch không chạy được ở chế độ headless (đã ghi trong README).

## Checklist quyết định (theo file kế hoạch)

| Bước | Tiêu chí | Kết quả |
| :-- | :-- | :-- |
| G1 | Project mở được, hằng số đúng Mục 2.1, GUT chạy | Đạt |
| G2 | Server đúng Mục 2.2, lỗi không làm dừng server, không tự tải model | Đạt |
| G3 | Phòng PBR, ánh sáng có bóng, `get_free_position` đúng | Đạt |
| G4 | Người tả thực, gate ≥ 90% tại 1–4 m, không hạ ngưỡng | Đạt (100% ở mọi mốc) |
| G5 | Robot kẹp vận tốc, ô góc nhìn đúng, box đúng tọa độ | Đạt |
| G6 | Bỏ frame cũ/trễ, message sai không crash, tự nối lại | Đạt |
| G7 | Đúng bảng ví dụ (theo H_TARGET 0.85), dấu omega đúng, hàm thuần | Đạt |
| G8 | Robot tìm và tiến lại gần, an toàn ghi đè, mất kết nối thì dừng | Đạt |
| G9 | CSV đủ cột, góc có dấu đúng, flush khi thoát | Đạt |
| G10 | ≥ 9/10 lượt thành công, không va người, README | 10/10, 0 va chạm; chưa thử README trên máy sạch |
