# Sim_Vision: robot tìm người bằng YOLOv8n trong Godot

Một căn phòng 8 × 6 × 3 m, một người tả thực và một robot. Robot dùng camera và `yolov8n.pt` để tìm người, quay về phía người rồi tiến lại gần. Góc trên bên phải màn hình là ô góc nhìn của robot. Đây là bước đầu của đồ án robot dẫn đường bệnh viện; bộ điều khiển (`controller.gd`) tách riêng để sau này thay bằng policy RL.

```
camera robot (SubViewport 640×480) ─5 Hz, JPEG─▶ yolo_server.py (YOLOv8n) ─detections─▶ VisionClient
        ▲                                                                                   │
        └── Robot.set_command(v, omega) ◀── apply_safety ◀── Controller.step ◀──────────────┘
```

Kế hoạch đầy đủ và hợp đồng chung (message WebSocket, hằng số): `docs/PROMPT/godot_robot_yolo_prompt.md` ở gốc repo. Tiến độ và kết quả kiểm chứng: [`PROGRESS.md`](PROGRESS.md). Các giả định: [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

## 1. Cài đặt

**Yêu cầu:** macOS / Linux / Windows có GPU hỗ trợ Vulkan hoặc Metal; Godot **4.3 trở lên** (đã thử với 4.7.2); Python **3.10 trở lên** (đã thử với 3.12).

```bash
# Thư mục gốc repo:
python -m venv .venv
# Kích hoạt trên Windows:
.\.venv\Scripts\Activate.ps1
# Hoặc macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**Model YOLO:** đặt `yolov8n.pt` (do người dùng cung cấp) vào `sim_vision/python/`. Trong repo này, đó là symlink tới `models/yolov8n/yolov8n.pt`:

```bash
ln -s ../../models/yolov8n/yolov8n.pt sim_vision/python/yolov8n.pt
```

Server **không tự tải model**: thiếu file thì báo lỗi và thoát.

**Mở project Godot lần đầu** (để Godot import asset):

```bash
cd sim_vision/godot
godot --headless --import
```

Lúc import sẽ có vài dòng lỗi `Can't open file … m011_*.tga`. Đó là do file FBX của người trỏ tới texture trên máy tác giả gốc, không ảnh hưởng gì vì `human.tscn` gán material riêng.

## 2. Asset và model người

Mọi asset đều có giấy phép CC0 hoặc MIT và đã có sẵn trong `godot/assets/`. Nguồn, giấy phép và **hướng dẫn tải lại từng bước** (gồm cách lấy người Rocketbox, đổi texture, tách animation) ở [`godot/assets/CREDITS.md`](godot/assets/CREDITS.md).

- Phòng: texture ambientCG (sàn Tiles028, tường PaintedPlaster017, trần OfficeCeiling001); đồ vật Poly Haven (bàn, ghế bành, chậu cây, xe lăn).
- Người: Microsoft Rocketbox `Male_Adult_06` (MIT), cao ~1.75 m, có animation `idle` và `walk` (`godot/tools/extract_human_anims.gd` tạo file `.res`).

Kiểm tra YOLO có nhận ra người không (cổng G4, cần server đang chạy):

```bash
cd sim_vision/godot && godot --path . res://tools/detect_gate.tscn   # ➔ logs/detect_gate.csv
```

## 3. Chạy

**Cửa sổ 1: YOLO server**

```bash
cd sim_vision/python
python sim_vision/python/yolo_server.py --model models/yolov8n/yolov8n.pt     # tùy chọn: --port 8765
```

**Cửa sổ 2: scene chính**

```bash
cd sim_vision/godot
godot --path .                 # chạy scenes/main.tscn
```

- Phím **R**: đặt lại robot và người ở vị trí ngẫu nhiên (cách nhau ≥ 3 m). Phím **Esc**: thoát.
- Ô góc phải: ảnh camera robot, box `person 0.87`, trạng thái `SEARCH` / `APPROACH` / `ARRIVED`, và `YOLO: đã kết nối` / `YOLO: mất kết nối`.
- Muốn lặp lại đúng một tình huống thì đặt `seed` (≥ 0) trên node `Main` trong editor; −1 là ngẫu nhiên.
- Mỗi lần chạy ghi một file `sim_vision/logs/run_<YYYYMMDD_HHMMSS>.csv` (13 cột, gồm ground truth `true_distance_m`, `true_angle_deg`).
- Tắt server giữa chừng thì robot dừng; bật lại thì tự nối lại (thử mỗi 2 s).

Gửi thử một ảnh tới server: `python sim_vision/python/send_image.py <ảnh.jpg>`.

## 4. Test

```bash
# Python (không gọi YOLO thật, dùng FakeDetector)
pytest

# Godot (GUT, chạy headless được)
cd sim_vision/godot && godot --headless -s addons/gut/gut_cmdln.gd -gconfig=res://.gutconfig.json
```

## 5. Chạy batch và phân tích

```bash
# Cửa sổ 1: server như mục 3. Cửa sổ 2:
cd sim_vision/godot
godot --path . res://tools/batch_run.tscn                     # 10 lượt, seed 0..9
godot --path . res://tools/batch_run.tscn -- --runs=20 --max-time=30

python sim_vision/python/analyze_logs.py     # tỉ lệ thành công, thời gian TB, logs/h_vs_distance.png
```

- `logs/batch_summary.csv`: `seed, success, time_to_arrive_s, final_distance_m, collisions, frames, mean_infer_ms`.
- **Thành công** = robot ở `ARRIVED` với khoảng cách thật tới người trong [0.8, 1.5] m trước 30 giây mô phỏng. `collisions` = số lần robot chạm người.
- **Không chạy được ở chế độ `--headless`**: renderer giả của headless không vẽ ảnh nên camera robot không có hình để gửi YOLO. `batch_run` tự kiểm tra và thoát với mã 2. Cần máy có màn hình (hoặc màn hình ảo như Xvfb trên Linux, kèm GPU/driver hỗ trợ Vulkan).

## 6. Hằng số (Mục 2.1, `godot/scripts/config.gd`)

| Tên | Giá trị | Ý nghĩa |
| :-- | :-- | :-- |
| `WS_URL` | `ws://127.0.0.1:8765` | Địa chỉ YOLO server |
| `CAPTURE_HZ` | 5 | Số ảnh gửi đi mỗi giây |
| `IMG_W`, `IMG_H` | 640, 480 | Độ phân giải camera robot |
| `JPEG_QUALITY` | 0.8 | Chất lượng JPEG khi gửi |
| `YOLO_CONF` | 0.4 | Ngưỡng tin cậy phía server; box thấp hơn bị bỏ |
| `V_MAX` | 0.8 m/s | Vận tốc tiến tối đa |
| `OMEGA_MAX` | 1.5 rad/s | Vận tốc quay tối đa |
| `K_ANG` | 2.0 | Hệ số quay: `omega = −K_ANG·(cx − 0.5)` (người bên phải ➔ quay phải, omega < 0) |
| `K_LIN` | 1.5 | Hệ số tiến: `v = K_LIN·(H_TARGET − h)` |
| `H_TARGET` | 0.85 | Box cao bằng tỉ lệ này của ảnh thì coi là đã tới (≈ 1.4–1.5 m). Ban đầu 0.65, nhưng giá trị đó làm robot dừng ở ~2.2 m nên đã đổi |
| `H_HYST` | 0.10 | Độ trễ: đang `ARRIVED` chỉ quay lại `APPROACH` khi `h < H_TARGET − H_HYST`, tránh nhảy qua lại |
| `ALIGN_TOL` | 0.25 | Người lệch tâm quá mức này (theo `cx`) thì chỉ quay, không tiến |
| `LOST_TIMEOUT_S` | 1.5 s | Mất người quá thời gian này thì về `SEARCH` |
| `SEARCH_OMEGA` | 0.5 rad/s | Tốc độ quay tại chỗ khi tìm người |
| `STALE_MS` | 500 ms | Kết quả về trễ hơn mức này bị bỏ |
| `SAFE_DIST_HUMAN` | 0.8 m | Khoảng cách thật (tâm tới tâm) tới người nhỏ hơn mức này thì không cho tiến |
| `SAFE_DIST_OBSTACLE` | 0.5 m | Raycast phía trước chạm vật cản gần hơn mức này thì không cho tiến |
| `RECONNECT_S` | 2.0 s | Chu kỳ thử kết nối lại server |

## 7. Cấu trúc

```
sim_vision/
  godot/
    project.godot, .gutconfig.json
    scenes/   main, room, human, robot, view_overlay (.tscn)
    scripts/  config, room, human, robot, view_overlay, vision_client, controller, main, run_logger (.gd)
    tools/    detect_gate, batch_run (.tscn/.gd), extract_human_anims.gd
    tests/    test_*.gd (GUT)
    assets/   textures/, models/, CREDITS.md
    addons/gut/
  python/
    yolo_server.py, detector.py, protocol.py, send_image.py, analyze_logs.py
    tests/, requirements.txt, pytest.ini, yolov8n.pt (người dùng cung cấp)
  logs/       (bị gitignore) run_*.csv, batch_summary.csv, detect_gate.csv
  ASSUMPTIONS.md, PROGRESS.md, README.md
```

**Thay bộ điều khiển bằng RL:** viết node mới có cùng hàm `step(state, dets, time_since_seen_s) -> {state, v, omega, target}`, thêm vào `main.tscn` rồi trỏ `controller_path` của node `Main` tới node đó.
