# Dựng và chạy sim robot bệnh viện trên máy mới (hướng dẫn cho AI agent)

Mục tiêu: từ một máy trống, tại **thư mục gốc repo** `Data-Science-Major-Project`, tải toàn bộ model + giọng NPC từ
Hugging Face (`ManhQuangAI/Robot-tics`, private) và chạy `run_simulation.py` với robot RL dẫn bệnh nhân.

Làm tuần tự. Mỗi bước có lệnh **kiểm tra**; chỉ sang bước sau khi kiểm tra ra đúng kết quả mong đợi.
Mọi lệnh chạy từ thư mục gốc repo, trừ khi ghi khác. Ví dụ dùng macOS/Linux; Windows xem mục cuối.

---

## 0. Những gì có sẵn ở đâu

| Thứ | Nguồn |
|---|---|
| Code | GitHub `QuangManhAI/Data-Science-Major-Project`, nhánh **`feat/human-walk-animation`** |
| Model (YOLO, RL, STT, PhoBERT, VnCoreNLP, TTS, codec) | HF `ManhQuangAI/Robot-tics` (private) |
| Giọng NPC + manifest, âm nền bệnh viện, cache TTS | HF `ManhQuangAI/Robot-tics` (private), cùng repo |

Cấu trúc thư mục trên HF **giống hệt repo**: tải thẳng vào thư mục gốc là các file nằm đúng chỗ code tìm.

| Thành phần | Đường dẫn trong repo |
|---|---|
| YOLO26n detect 17 lớp (mAP50 92.6 %) | `models/yolo26n_hospital_robot_v6_best/yolo26n_hospital_robot_best_map92.6.pt` |
| RL PPO Pure-Vision V2 (21-D, không LiDAR) | `experiments/checkpoints/PPOHospital-PureVisionV2_best.pt` |
| STT PhoASR-whisper-small fine-tune | `models_finetuned/phoasr_small_hospital_best/` |
| NLU PhoBERT intent | `models_finetuned/phobert-intent/` |
| Tách từ cho PhoBERT | `models/VnCoreNLP/` (cần Java) |
| TTS VieNeu v3 Turbo ONNX | `models/Vie-neu-v3-turbo/` (dùng `onnx_update/`) |
| Codec của TTS | `models/MOSS-Audio-Tokenizer-Nano-ONNX/` |
| Giọng NPC (8 nam, 14 nữ, người lớn) | `data/processed/manifest.csv` + `data/processed/wav/` |
| Âm nền bệnh viện | `sim_vision/godot/assets/sounds/hospital_ambience.wav`, `monitor_beep.wav` |
| Cache câu trả lời TTS | `data/tts_cache/` |

---

## 1. Phần mềm hệ thống

| Cần | Phiên bản | Kiểm tra |
|---|---|---|
| Python | 3.12 (3.11–3.12 được; **không** 3.13) | `python3.12 --version` |
| Godot | **4.7.2** (bản standard, không phải .NET) | `godot --version` ➔ `4.7.2.stable...` |
| Java (JDK) | ≥ 8, khuyên 17 | `java -version` |
| ffmpeg | bất kỳ | `ffmpeg -version` (chỉ cần khi xử lý lại dữ liệu giọng) |
| git, git-lfs không cần | | |

Cài (macOS): `brew install python@3.12 openjdk@17 ffmpeg` + tải Godot 4.7.2 từ godotengine.org.
Cài (Ubuntu): `sudo apt install python3.12 python3.12-venv openjdk-17-jdk ffmpeg` + Godot 4.7.2 Linux x86_64.

Godot phải gọi được: hoặc có `godot` trong PATH, hoặc đặt `export GODOT_BIN=/đường/dẫn/tới/godot`.

**JAVA_HOME bắt buộc đặt** (code mặc định trỏ đường dẫn Homebrew của máy gốc):
```bash
# macOS (Homebrew):  export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
# Ubuntu:            export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ls "$JAVA_HOME/bin/java"          # phải tồn tại
```
Ghi `export JAVA_HOME=...` (và `GODOT_BIN` nếu dùng) vào `~/.zshrc` / `~/.bashrc` để các tiến trình con thấy.

---

## 2. Lấy code

```bash
git clone https://github.com/QuangManhAI/Data-Science-Major-Project.git
cd Data-Science-Major-Project
git checkout feat/human-walk-animation
```
Kiểm tra: `git branch --show-current` ➔ `feat/human-walk-animation`; có file `run_simulation.py`, `SETUP_FROM_HF.md`.

---

## 3. Hai môi trường Python

Launcher tự tìm `sim_vision/python/.venv` (sim, YOLO, STT, NLU, RL) và `sim_vision/python/.venv-tts` (TTS riêng
vì gói `vieneu` kéo nhiều phụ thuộc). Tạo đúng tên và đúng chỗ:

```bash
python3.12 -m venv sim_vision/python/.venv
sim_vision/python/.venv/bin/pip install -U pip
sim_vision/python/.venv/bin/pip install -r sim_vision/python/requirements.txt
sim_vision/python/.venv/bin/pip install scipy huggingface_hub

python3.12 -m venv sim_vision/python/.venv-tts
sim_vision/python/.venv-tts/bin/pip install -U pip
sim_vision/python/.venv-tts/bin/pip install "vieneu==3.8.3" websockets
```
GPU NVIDIA (tùy chọn): cài torch bản CUDA trước rồi mới `-r requirements.txt`. STT tự chọn thiết bị mps ➔ cuda ➔ cpu.

Kiểm tra:
```bash
sim_vision/python/.venv/bin/python -c "import ultralytics, torch, transformers, py_vncorenlp; print('sim venv ok')"
sim_vision/python/.venv-tts/bin/python -c "import vieneu, onnxruntime; print('tts venv ok')"
```

---

## 4. Tải model + giọng NPC từ Hugging Face

Repo private: cần token **Read** của tài khoản có quyền (huggingface.co ➔ Settings ➔ Access Tokens).
Hỏi người dùng token nếu chưa đăng nhập; không ghi token vào file nào trong repo.

```bash
sim_vision/python/.venv/bin/hf auth login            # dán token
sim_vision/python/.venv/bin/hf auth whoami           # phải in user=...
sim_vision/python/.venv/bin/hf download ManhQuangAI/Robot-tics --local-dir . --exclude "README.md" --exclude ".gitattributes"
rm -rf .cache/huggingface          # metadata của lệnh tải, không nằm trong .gitignore ➔ xóa cho sạch git status
```
Tải khoảng 2.5 GB.

Kiểm tra đủ file:
```bash
ls -la models/yolo26n_hospital_robot_v6_best/yolo26n_hospital_robot_best_map92.6.pt \
       experiments/checkpoints/PPOHospital-PureVisionV2_best.pt \
       models_finetuned/phoasr_small_hospital_best/model.safetensors \
       models_finetuned/phobert-intent/model.safetensors \
       models/VnCoreNLP/VnCoreNLP-1.2.jar models/VnCoreNLP/models/wordsegmenter/wordsegmenter.rdr \
       models/Vie-neu-v3-turbo/onnx_update/vieneu_backbone_shared.data \
       models/MOSS-Audio-Tokenizer-Nano-ONNX \
       data/processed/manifest.csv sim_vision/godot/assets/sounds/hospital_ambience.wav
ls data/processed/wav | wc -l                        # = số dòng manifest.csv trừ 1 (1669)
```

**Quan trọng — file phải là file thật, không phải symlink sang thư mục khác.** onnxruntime từ chối file `.data`
nằm ngoài thư mục model ("External data path escapes model directory"), và VnCoreNLP tìm `models/wordsegmenter`
cạnh vị trí thật của `.jar`. `hf download --local-dir` tạo file thật nên không gặp, trừ khi tự tạo symlink.

---

## 5. Chuẩn bị giọng NPC cho Godot

STT server đọc `data/processed/`; Godot đọc bản sao trong `sim_vision/godot/assets/voices/` (không có trên HF vì
tạo lại được):
```bash
cd sim_vision/python
.venv/bin/python -m voice.sync_voices
cd ../..
```
Kiểm tra: in `copy 1669, giữ 0, xóa 0 ➔ .../godot/assets/voices`; `ls sim_vision/godot/assets/voices/clips.json`.

Không chạy `python -m voice.add_dataset_voices` ở máy mới: script đó dựng lại manifest từ bộ thu âm gốc
(`data/processed/voice_dataset/`, không upload). Manifest tải về đã là kết quả cuối.

---

## 6. Import project Godot (một lần)

```bash
"${GODOT_BIN:-godot}" --headless --path sim_vision/godot --import
```
Kiểm tra: lệnh tự thoát, không có dòng `ERROR` về script. (Lần đầu mất 1–3 phút.)

---

## 7. Kiểm thử trước khi chạy

```bash
# Python (≈ 280 test, phải exit 0)
cd sim_vision/python && .venv/bin/python -m pytest tests -q -p no:warnings; echo "exit=$?"; cd ../..

# Godot GUT (phải "Passing Tests 177" / không có Failing)
cd sim_vision/godot && "${GODOT_BIN:-godot}" --headless -s addons/gut/gut_cmdln.gd -gconfig=res://.gutconfig.json -gexit 2>&1 | grep -E "^Tests|Passing|Failing"; cd ../..
```
Đã biết: `tests/test_pure_vision_obs.py::test_doctor_channel_and_avoidance` (thư mục `tests/` ở gốc) hỏng từ trước,
không liên quan sim. `tests/test_dashboard.py` cần `gradio` (không cài cũng được).

---

## 8. Chạy sim

```bash
sim_vision/python/.venv/bin/python run_simulation.py --controller rl --nlu phobert
# chỉ các ca dẫn đường:                      ... --escort-only
# không mở Godot (chỉ server, để debug):     ... --no-godot
```
Launcher bật 4 server rồi mở cửa sổ Godot. Log nằm trong `logs/` (`yolo.log`, `stt.log`, `tts.log`,
`rl_policy_server.log`, `godot.log`). Kết quả mong đợi trong terminal:
```
[Launcher] Starting YOLO server ... (model: yolo26n_hospital_robot_best_map92.6.pt)
[Launcher] -> YOLO Server is ready!
[Launcher] -> STT Server is ready!
[Launcher] -> TTS Server is ready!
[Launcher] -> RL Policy Server is ready!
[Launcher] Launching Godot: ... (controller=rl)
```
Kiểm tra sau ~60 s:
```bash
grep "obs_dim=21" logs/rl_policy_server.log                 # policy V2 đã nạp
grep "clips" logs/stt.log                                   # "... phoasr_small_hospital_best ... 1669 clips"
grep -c "^ERROR\|SCRIPT ERROR" logs/godot.log               # 0
grep "Ambience" logs/godot.log                              # nền=true, máy theo dõi có tiếng bíp=5
ls -t sim_vision/logs/dialogue_*.csv | head -1 | xargs tail -3   # lượt hội thoại: hyp_text, intent, action
```
Phím trong Godot: R đặt lại · C đổi góc nhìn · A/D xoay camera · I/Tab thông số · M bật/tắt tiếng ồn · Esc thoát.

Đánh giá tự động 25 phút (tùy chọn): `sim_vision/python/.venv/bin/python scripts/eval_godot_escort.py --minutes 25`
➔ `experiments/results/godot_eval_<thời gian>/summary.json`.

---

## 9. Lỗi thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `TTS Server exited`, log: `External data path escapes model directory` | file `.onnx`/`.data` là symlink sang thư mục khác | thay symlink bằng file thật (tải lại bằng `hf download`) |
| Robot hỏi xong luôn hiện "Xin lỗi, tôi chưa nghe rõ", NPC không có phụ đề | NLU lỗi: thiếu Java / `JAVA_HOME` sai / `wordsegmenter.rdr is not found` | đặt `JAVA_HOME` (bước 1), kiểm tra `models/VnCoreNLP/models/wordsegmenter/` |
| Robot đứng im, `rl_policy_server.log`: `mat1 and mat2 shapes cannot be multiplied (1x31 and 21x128)` | code cũ không phải nhánh V2 | `git checkout feat/human-walk-animation` |
| `Checkpoint not found` | thiếu file RL | kiểm tra `experiments/checkpoints/PPOHospital-PureVisionV2_best.pt` |
| NPC không nói | thiếu giọng | làm lại bước 4 (kiểm tra `data/processed/wav`) và bước 5 |
| `Godot executable not found` | không có trong PATH | `export GODOT_BIN=...` hoặc `--godot /đường/dẫn` |
| `port ... in use` / server không lên | lần chạy trước chưa tắt | `pkill -f run_simulation.py; pkill -f "tts_server|rl_policy_server|yolo_server|stt_server"` |
| YOLO log ghi model `yolov8n.pt` | thiếu file YOLO v6 | kiểm tra bước 4 |

---

## 10. Windows

- venv: `sim_vision\python\.venv\Scripts\python.exe` thay cho `.venv/bin/python` (launcher tự nhận cả hai).
- `JAVA_HOME`: `setx JAVA_HOME "C:\Program Files\Eclipse Adoptium\jdk-17..."` rồi mở terminal mới.
- Godot: `setx GODOT_BIN "C:\...\Godot_v4.7.2-stable_win64_console.exe"` (bản `_console` để thấy log).
- Các lệnh `grep`/`ls` ở trên: dùng PowerShell `Select-String` / `Get-ChildItem`, hoặc Git Bash.
