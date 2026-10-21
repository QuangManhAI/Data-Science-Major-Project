# Giả định

Định dạng: `[G<số>] <giả định> — <lý do>`.

## G1
- [G1] Dùng Godot 4.7.2-stable (`config/features` = "4.7") — kế hoạch yêu cầu ≥ 4.3; đây là bản người dùng đã cài.
- [G1] GUT 9.7.1 cài vào `godot/addons/gut/`, cấu hình chạy ở `godot/.gutconfig.json` — bản GUT mới nhất, hỗ trợ Godot 4.x.
- [G1] Thư mục `sim_vision/` nằm ở gốc repo `Data-Science-Major-Project/` — Mục 1 không nói rõ vị trí.
- [G1] Python 3.12 (Homebrew) trong venv `python/.venv` — `python3` của hệ thống là 3.9, thấp hơn yêu cầu 3.10+.
- [G1] Phiên bản ghim trong `requirements.txt` là bản mới nhất trên PyPI tại 2026-10-21, đã cài chung và chạy được. `ultralytics` tự kéo thêm `opencv-python` (không headless) làm phụ thuộc; hai gói cùng phiên bản nên không xung đột.
- [G1] `Config` là `class_name Config extends RefCounted`, kiểu: `IMG_W`, `IMG_H`, `CAPTURE_HZ`, `STALE_MS` là `int`, còn lại `float`/`String` — Mục 2.1 không nêu kiểu; chọn theo bản chất giá trị.
- [G1] `Config.state_name()` với giá trị không thuộc enum ➔ `push_error` và trả `""` — Mục 2.3 không nói về giá trị lạ; không bịa thêm tên trạng thái.
- [G1] `main.tscn` tạm thời chỉ có node gốc `Node3D` tên `Main` — nội dung thật làm ở G8.
- [G1] `pandas`, `matplotlib` (cần cho G10) chưa thêm vào `requirements.txt` — G1 chỉ liệt kê 6 gói; sẽ thêm ở G10.

## G2
- [G2] Thứ tự kiểm tra trường: `type` ➔ `frame_id` ➔ `jpeg_b64` ➔ `width` ➔ `height` — để khớp ví dụ `{"type":"frame","frame_id":4}` ➔ `missing field: jpeg_b64` (message này thiếu cả `width`, `height`).
- [G2] Chuỗi lỗi ngoài ví dụ: `missing field: <tên>`, `invalid field: <tên>` (sai kiểu/giá trị, kể cả `type` ≠ `"frame"`), `invalid jpeg` (base64 hỏng hoặc không giải mã được), `size mismatch` (`width`/`height` khác kích thước ảnh thật) — Mục 2.2 chỉ cho định dạng message `error`, không liệt kê nội dung `message`.
- [G2] JSON hợp lệ nhưng không phải object (vd `[1,2]`, `42`) ➔ `invalid json` — server chỉ hiểu message dạng object.
- [G2] `frame_id` là `true`/`false` hoặc số thực ➔ không hợp lệ — Mục 2.2: số nguyên.
- [G2] Box được kẹp về [0, 1], làm tròn 4 chữ số; box suy biến sau khi kẹp (x1 ≥ x2 hoặc y1 ≥ y2) bị bỏ — để luôn đúng ràng buộc `0 ≤ x1 < x2 ≤ 1`.
- [G2] `conf` làm tròn 4 chữ số, `infer_ms` làm tròn 1 chữ số — ví dụ Mục 2.2 dùng `23.5`, `21.4`.
- [G2] `YoloDetector` chạy nháp 1 lần với ảnh đen 640×480 lúc khởi động — để `infer_ms` của frame đầu không tính thời gian khởi tạo.
- [G2] Suy luận chạy trên CPU (mặc định của ultralytics trên macOS, không dùng MPS) — G10 yêu cầu `mean_infer_ms` < 100 ms trên CPU.
- [G2] "Chỉ xử lý frame mới nhất": mỗi kết nối giữ đúng 1 message đang chờ; message mới ghi đè message chờ cũ (kể cả message hỏng ghi đè frame hợp lệ đang chờ). Frame bị bỏ không có phản hồi.
- [G2] Hằng số phía Python (`WS_HOST`, `WS_PORT`, `YOLO_CONF`) đặt trong `protocol.py`, giá trị lấy từ `WS_URL`, `YOLO_CONF` của Mục 2.1 — G2 chỉ cho phép 3 file `protocol.py`, `detector.py`, `yolo_server.py`.
- [G2] `python/yolov8n.pt` là symlink tới `models/yolov8n/yolov8n.pt` (file người dùng đã cung cấp) — tránh copy file 6.5 MB.
- [G2] Thêm `python/send_image.py` để gửi ảnh thật khi kiểm chứng; ảnh người thật dùng `bus.jpg`, `zidane.jpg` có sẵn trong gói ultralytics (không tải gì thêm).
- [G2] Thêm `python/pytest.ini` (`pythonpath = .`, `asyncio_mode = auto`) — repo gốc đã có `pytest.ini` riêng trỏ tới `tests/`, nên cần file riêng cho `sim_vision/python`.

## G3
- [G3] Phòng có 1 cửa sổ 2 m × 1.4 m (x ∈ [-1, 1], y ∈ [0.9, 2.3]) ở tường bắc (z = -3), không có kính — để `DirectionalLight3D` "giả nắng qua cửa sổ" thật sự chiếu vào phòng kín có bật bóng.
- [G3] Đồ vật: bàn làm việc kim loại, ghế bành, chậu cây, xe lăn (Poly Haven, CC0) — chọn đồ hợp bối cảnh bệnh viện; xe lăn hợp với đồ án.
- [G3] Texture: sàn Tiles028, tường PaintedPlaster017, trần OfficeCeiling001 (ambientCG, CC0, bản 1K). Mapping triplanar theo tọa độ thế giới, lặp lại mỗi 2 m (`uv1_scale = 0.5`) — các mảng tường ghép liền vân, không méo.
- [G3] Collision đồ vật là 1 `BoxShape3D` bao AABB của model (đo sau khi import) — đủ cho va chạm và cho `get_free_position`; không cần lưới chính xác.
- [G3] `get_free_position(min_clearance, rng = null)` có thêm tham số tùy chọn `rng` — G8 cần đặt vị trí lặp lại được theo `seed`; gọi đúng chữ ký G3.1 vẫn chạy (ngẫu nhiên).
- [G3] "Cách đồ vật ≥ min_clearance" tính bằng khoảng cách trên mặt phẳng XZ từ điểm tới hình chiếu (AABB thế giới) của collision đồ vật; cách tường cũng ≥ `min_clearance` (khớp ví dụ `|x| ≤ 3.5`, `|z| ≤ 2.5` với 0.5).
- [G3] `min_clearance` lớn tới mức không còn vùng hợp lệ (vd ≥ 3 m) ➔ báo lỗi ngay, không thử 100 lần.
- [G3] Thông số ánh sáng/môi trường: nắng `light_energy` 2.0 chếch 35°; 2 `OmniLight3D` tại (±2, 2.7, 0), năng lượng 1.2, tầm 7 m; SDFGI (bật occlusion), SSAO, ACES, nền và ánh sáng môi trường từ `ProceduralSkyMaterial` (năng lượng 0.5) — đặc tả không cho số; chỉnh tiếp ở G4 nếu cổng nhận diện không đạt.
- [G3] `room.tscn` được sinh một lần bằng script Python tạm (không lưu trong repo); từ giờ sửa trực tiếp trong Godot editor.

## G4
- [G4] Người: Microsoft Rocketbox `Male_Adult_06` (MIT), áo đỏ, quần jean — người dùng chọn Rocketbox; áo đỏ tương phản tốt với tường và sàn sáng màu.
- [G4] Dùng file FBX gốc, Godot 4.7 import trực tiếp; không chuyển sang `.glb` — G4.1 cho `.glb`/`.gltf`, nhưng chuyển định dạng cần Blender và không làm thay đổi gì khi chạy; khi import Godot cũng tạo ra scene giống như từ glTF.
- [G4] Material gán lại trong `human.tscn` (FBX trỏ tới texture theo đường dẫn tuyệt đối trên máy tác giả). Roughness = `1 − specular` của Rocketbox. Normal map dùng nguyên bản, không lật kênh Y (ảnh render không thấy lỗi bóng đổ ngược).
- [G4] Tóc và lông mi (surface `opacity`) dùng `ALPHA_DEPTH_PRE_PASS`, hiển thị cả hai mặt — tránh tóc bị thủng hoặc sắp xếp sai thứ tự trong suốt.
- [G4] Animation `idle` = `m_idle_neutral_01` (tại chỗ), `walk` = `m_walk_neutral` (bản tách chuyển động `xy`, đi tại chỗ; script tự di chuyển 0.5 m/s). Tách ra `idle.res`/`walk.res` bằng `tools/extract_human_anims.gd`, bỏ track của 41 xương không có trong avatar.
- [G4] Gốc `human.tscn` là `CharacterBody3D` (đi bằng `move_and_slide`, không có trọng lực, y giữ nguyên) — cần va chạm để robot (G8) và raycast phát hiện được.
- [G4] Quy ước hướng: mặt người là +Z cục bộ. Khi đi, `rotation.y = atan2(dx, dz)`. Waypoint lặp vòng; `walks = true` mà có < 2 waypoint ➔ đứng yên (idle).
- [G4] Capsule va chạm: bán kính 0.25 m, cao 1.75 m — chiều cao người ≈ 1.75 m (xương đầu ở 1.59 m).
- [G4] Cổng kiểm tra: camera nhìn ngang (không chúc xuống) ở 1.2 m, đặt tại x = −3.6 nhìn về +X để đủ chỗ cho 6 m trong phòng 8 m; yaw 0° = người nhìn thẳng vào camera; ánh sáng ±10% nhân vào mọi `Light3D` của phòng; camera lệch ngẫu nhiên ±5° cả ngang và dọc; seed cố định 0.
- [G4] "Có detection conf ≥ 0.5" = detection đầu tiên (conf cao nhất) có conf ≥ 0.5. CSV thêm cột `mean_top_conf` (trung bình conf cao nhất của 20 ảnh, 0 nếu không thấy ai). Lưu 1 ảnh mẫu mỗi mốc vào `logs/detect_gate/` (chỉ tool này lưu ảnh, server thì không).
- [G4] `detect_gate` tự kết nối WebSocket tối giản (có `outbound_buffer_size` = 4 MB vì ảnh base64 vượt quá 64 KB mặc định) — VisionClient G6 chưa có. G6 cũng cần tăng buffer này.

## G5
- [G5] Thân robot: trụ cao 1.0 m, bán kính 0.25 m; đầu là hộp 0.3 × 0.25 × 0.3 m, tâm ở 1.1 m. Collision là một hình trụ cao 1.225 m (bao cả đầu). Vật liệu robot là màu trơn — yêu cầu PBR ở G3 chỉ áp cho phòng.
- [G5] Camera robot (trong `SubViewport`) bám theo đầu robot nhờ `RemoteTransform3D` đặt ở (0, 1.2, −0.16), tức mặt trước của đầu; `near = 0.05` để không thấy chính đầu robot.
- [G5] `RayCast3D` đặt ở mặt trước thân (0, 0.3, −0.25), dài 1.0 m về −Z. Độ cao 0.3 m để chạm được đồ thấp (chậu cây 0.63 m). `front_obstacle_distance` đo từ mặt trước thân robot và gọi `force_raycast_update()` để luôn có kết quả mới.
- [G5] Robot không có trọng lực, y giữ nguyên; hướng trước là −Z; mỗi khung vật lý `rotation.y += omega·dt` (omega > 0 quay trái, đúng Mục 2.4).
- [G5] Kích thước `SubViewport` đặt bằng `Config.IMG_W/IMG_H` trong `_ready`; `fov = 70` (đặc tả G5.1, không có trong Mục 2.1) ghi trong scene.
- [G5] Ô góc nhìn: viền 2 px là một `ColorRect` sáng màu phía sau ảnh; chữ dùng font mặc định của Godot (hiển thị được tiếng Việt), cỡ 13. Trạng thái robot ở góc trên trái của ô, trạng thái kết nối ở góc dưới trái (đỏ khi mất kết nối).
- [G5] Thêm `ViewOverlay.set_viewport(vp)` để gắn ảnh camera, cùng các hàm thuần `norm_to_pixel`, `label_text`, `connection_text` để test được — G5.2 chỉ yêu cầu `show_detections`.
- [G5] `Robot.get_command()` và `Robot.clamp_command()` thêm vào để test và để G8/G9 đọc lệnh đang chạy.

## G6
- [G6] Mỗi lần thử kết nối tạo một `WebSocketPeer` mới, `outbound_buffer_size` = 4 MB (mặc định 64 KB nhỏ hơn 1 ảnh JPEG base64 ~60–100 KB). Thử kết nối ngay khi bắt đầu, sau đó mỗi `RECONNECT_S` khi đang ở trạng thái CLOSED.
- [G6] Bộ đếm thời gian chụp đặt lại về 0 mỗi khi trạng thái kết nối đổi; frame đầu tiên gửi sau 1/`CAPTURE_HZ` giây kể từ lúc kết nối.
- [G6] Kết quả chỉ được nhận khi `frame_id` có trong `sent_at` (Godot đã gửi frame này) — `frame_id` lạ bị bỏ, coi như trễ.
- [G6] `sent_at` được dọn mỗi khung: bỏ frame ≤ frame đã nhận gần nhất và frame đã gửi quá `STALE_MS` (không thể nhận nữa) ➔ chỉ còn vài mục.
- [G6] Message `type == "error"` ➔ `push_warning` kèm `frame_id` và `message`; không phát signal.
- [G6] JSON của Godot đọc mọi số thành `float` ➔ `frame_id` hợp lệ là số ≥ 0 không có phần lẻ. `infer_ms` phải là số ≥ 0. `conf` cũng phải nằm trong [0, 1].
- [G6] Dùng `JSON.new().parse()` thay cho `JSON.parse_string()` — hàm sau in lỗi engine khi gặp JSON hỏng, trái với yêu cầu "`push_warning`, bỏ qua".
- [G6] Thêm `is_server_connected()`, `record_sent()`, `pending_count()`, `handle_message(text, now_ms)` và hàm thuần `is_fresh()`, `is_valid_detection()` để test logic bỏ frame cũ/trễ mà không cần socket thật. Không dùng tên `is_connected` vì `Object.is_connected` đã có.
- [G6] `Robot.get_camera_viewport()` đổi sang `get_node()` thay vì biến `@onready` — gọi trước `_ready` (vd từ script dựng cảnh) trước đây trả null.

## G7
- [G7] `Controller` kế thừa `Node` (không phải `RefCounted`) — G8 chọn bộ điều khiển qua `@export var controller_path: NodePath`, nên phải là node. `step` vẫn là hàm thuần: không dùng biến thành viên, không đọc scene hay thời gian.
- [G7] `state` truyền vào và trả ra là giá trị enum `Config.RobotState` (int); đổi sang chuỗi bằng `Config.state_name()` khi hiển thị/ghi log.
- [G7] `target` là đúng dictionary detection được chọn (không thêm `cx`, `h`) — G7.1 chỉ ghi `target: Dictionary`.
- [G7] Bằng `conf` và bằng `h` ➔ chọn detection đứng trước trong danh sách.
- [G7] Với `cx` ∈ [0, 1] thì |omega| ≤ `K_ANG`·0.5 = 1.0 < `OMEGA_MAX`, nên phép kẹp omega không bao giờ xảy ra với input hợp lệ. Test biên kiểm tra omega luôn trong giới hạn thay vì ép kẹp.

## G8
- [G8] Camera người xem là `ViewerCamera` đặt trong phòng, sát trần ở góc tây nam (−3.8, 2.85, 2.75), nhìn chéo xuống giữa phòng, FOV 80° — camera ngoài phòng bị tường và trần che.
- [G8] Người cũng quay mặt theo hướng ngẫu nhiên (cùng `rng`) — G8.1 chỉ nói robot hướng ngẫu nhiên; người quay lưng hay quay ngang vẫn được YOLO nhận ra (gate G4) và gần thực tế hơn.
- [G8] Robot và người lấy vị trí lần lượt từ cùng một `rng` (robot trước, người sau), thử tối đa 100 cặp để có khoảng cách ≥ 3 m; không được ➔ `push_error` và dùng cặp cuối.
- [G8] `time_since_seen_s` = `INF` khi chưa từng thấy người kể từ lúc bắt đầu hoặc nhấn R.
- [G8] Khi kết nối mà quá `LOST_TIMEOUT_S` không có kết quả mới, `controller.step(state, [], t)` được gọi mỗi khung cho tới khi có kết quả lại; các lần gọi này không phát `cycle_completed` (G9 chỉ ghi log khi có kết quả nhận diện).
- [G8] Mất kết nối ➔ `set_command(0, 0)`, trạng thái giữ nguyên; kết nối lại ➔ mốc "kết quả gần nhất" đặt về lúc kết nối.
- [G8] Thêm signal `cycle_completed(info)` (frame_id, dets, cmd sau an toàn, infer_ms, connected) để G9 ghi log mà không phải sửa luồng chính; thêm `human_distance()` (khoảng cách XZ giữa tâm robot và tâm người).
- [G8] Biến `@export var seed` trùng tên hàm toàn cục `seed()` của GDScript ➔ thêm `@warning_ignore("shadowed_global_identifier")` để giữ đúng tên trong đặc tả.
- [G8] Có thêm nhãn hướng dẫn "R: đặt lại vị trí   Esc: thoát" ở góc trên trái.

## G9
- [G9] `conf`, `cx`, `h` lấy từ detection mà controller chọn (`cmd.target`), không phải detection đầu tiên — để log khớp với thứ robot đang bám theo.
- [G9] `time_s` tính từ lúc `main.gd` bắt đầu (`_ready`), không reset khi nhấn R.
- [G9] `connected` ghi `true`/`false` (pandas tự đọc thành bool); `frame_id`, `n_det` là số nguyên; `state` là tên chuỗi.
- [G9] `true_angle_deg` đúng bằng ±180 (người ở ngay sau lưng) ghi là 180. Hướng trước hoặc hướng tới người không có thành phần XZ ➔ 0.
- [G9] Ngoài `NOTIFICATION_WM_CLOSE_REQUEST`, logger còn flush và đóng file trong `_exit_tree` — thoát bằng phím Esc (`get_tree().quit()`) không phát WM_CLOSE_REQUEST.
- [G9] Thư mục log mặc định là `sim_vision/logs/` (cạnh thư mục `godot/`), tự tạo nếu chưa có; tạo không được hoặc mở file không được ➔ `push_error` một lần, `log_row` không làm gì nữa.
- [G9] `main.gd` có node `RunLogger` thì ghi log, không có thì bỏ qua (`get_node_or_null`) — test và batch (G10) có thể bỏ logger.

## Thay đổi hợp đồng (2026-10-21)
- [G8➔G10] `H_TARGET` 0.65 ➔ **0.85** (người dùng quyết định). Lý do: với 0.65 robot dừng ở 2.21 m, ngoài khoảng thành công [0.8, 1.5] m của G10. Đã sửa `config.gd`, `test_config.gd`, Mục 2.1 và bảng ví dụ G7.3/G7.4 trong `docs/PROMPT/godot_robot_yolo_prompt.md`, 4 ví dụ và 3 test biên trong `test_controller.gd`. Logic controller không đổi.

## G10
- [G10] Một lượt kết thúc ngay khi thành công (ARRIVED và khoảng cách thật trong [0.8, 1.5] m) hoặc khi hết 30 s mô phỏng; `final_distance_m` là khoảng cách lúc kết thúc.
- [G10] Thời gian mô phỏng là tổng `delta` của các khung vật lý, bắt đầu tính khi `VisionClient` đã kết nối (robot đứng yên trước đó) ➔ vị trí ban đầu giống hệt khi chạy `main.tscn` cùng seed.
- [G10] `collisions` = số lần robot **chạm người** (va chạm trượt với người, hoặc tâm cách nhau ≤ tổng bán kính + 2 cm), đếm theo sườn lên — tiêu chí G10.3 là "không va chạm với người". Chạm tường/đồ vật không tính.
- [G10] `frames` = số chu kỳ nhận diện (số lần `cycle_completed`); `mean_infer_ms` = trung bình `infer_ms` của các chu kỳ đó.
- [G10] Tham số dòng lệnh `-- --runs=N --max-time=S` (mặc định 10 lượt và 30 s). Batch cần cửa sổ có render; khi `--headless` thì tự thoát với mã 2.
- [G10] `analyze_logs.py` đọc mọi file `logs/run_*.csv` ở thư mục gốc `logs/` (không đọc thư mục con) và `logs/batch_summary.csv`; biểu đồ lưu ở `logs/h_vs_distance.png`. Log chạy thử trước batch đã chuyển sang `logs/archive_pre_batch/`.
- [G10] Thêm `pandas==3.0.6`, `matplotlib==3.11.2` vào `requirements.txt`.
