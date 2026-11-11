# Nâng cấp tương tác người – robot (HRI) và nhận diện người

Ngày lập: 2026-11-11 · Nhánh: `feat/human-walk-animation`

## Mục tiêu

1. Robot **giao tiếp bằng cử chỉ**: có tay, chỉ vào phòng ban / thang máy / hướng rẽ.
2. Robot **chủ động nói khi dẫn đường**: "bác nhớ đi gần tôi nhé", báo rẽ, báo tới nơi.
3. Robot **theo dõi khoảng cách** với bệnh nhân: xa thì nhắc và chờ; mất hẳn thì hỏi có tiếp tục không,
   rồi nghe câu trả lời (STT + PhoBERT, intent `tiep_tuc` / `dung_lai` đã có sẵn).
4. Robot **biết đúng người mình đang dẫn**: lúc đón thì ghi nhớ mặt + quần áo + dáng (embedding của model
   pretrained trên người thật), lúc dẫn thì bám theo embedding đó. Cùng pipeline chạy được ngoài đời thực,
   không train lại.

Nguyên tắc chung: **không đổi observation 21-D của policy RL** ➔ không phải train lại PPO. Mọi hành vi mới nằm
ở lớp "giám sát" (supervisor) bên ngoài policy: lớp này được dừng robot, nói, chỉ tay, nhưng không lái thay policy.

## Hiện trạng liên quan (đã kiểm tra trong code)

| Chỗ | Hiện tại | Vấn đề |
|---|---|---|
| `rl_policy_server.py:149` | chọn box `patient` có conf cao nhất | đông người ➔ bám nhầm người |
| `rl_policy_server.py:259` | khoảng cách = `1.20 / chiều cao box` | phụ thuộc FOV camera Godot |
| `rl_env_bridge.gd step_escorted_patient` | bệnh nhân luôn bám theo robot (script) | không bao giờ tụt lại ➔ không test được "nhắc" |
| `dialogue_manager.gd` | robot chỉ nói khi trả lời NPC | chưa có đường để robot tự nói |
| `robot.tscn` | không có tay | |

## Danh sách việc

### A. Tay robot và chỉ hướng (Godot)
- [x] A1. `robot_arms.gd`: hai tay (vai – cánh tay – bàn tay), tạo bằng code như `LidarFan`. Tư thế nghỉ, `point_at(vị trí)`,
      `lower()`, `wave()`. Quay mượt (slerp), giới hạn góc khớp (không chỉ ra sau lưng: quá ±110° thì kẹp lại).
- [x] A2. Tay nằm ở **visual layer riêng**, camera robot không render layer đó ➔ tay không lọt vào ảnh YOLO.
      Tay không có collision ➔ không ảnh hưởng tia depth / LiDAR.
- [x] A3. GUT test: chọn tay trái/phải theo hướng, kẹp góc, tay không có trong cull mask của camera robot.

### B. Robot chủ động nói (Godot + Python)
- [x] B1. `RobotAnnouncer`: robot tự nói một câu (TTS + phụ đề), có hàng đợi và cooldown, dùng `utt_id` riêng để không
      đụng lượt hội thoại. Không có TTS ➔ chỉ phụ đề.
- [x] B2. Câu thoại khi dẫn (`nlu/escort_lines.py`, một nguồn duy nhất cho Godot và pregen):
      bắt đầu ("Mời bác đi theo tôi, bác nhớ đi gần tôi nhé"), thấy đích ("Phòng … ở phía trước bên phải ạ"),
      thang máy, tới nơi, nhắc, hỏi tiếp tục, kết thúc khi từ chối.
- [x] B3. Thêm các câu đó vào `tts/pregen.py` ➔ có sẵn trong `data/tts_cache`, không chờ TTS lúc chạy.

### C. Theo dõi khoảng cách, nhắc, hỏi tiếp tục
- [x] C1. Máy trạng thái `EscortMonitor` (hàm thuần, test được):
      `OK` (≤ 2.2 m) ➔ `FAR` (> 2.8 m quá 2 s: nói nhắc, robot **dừng chờ**) ➔ `LOST` (> 5 m hoặc mất dấu quá 8 s:
      hỏi "Bác có muốn tôi dẫn tiếp không ạ?", nghe trả lời) ➔ `tiep_tuc` thì dẫn tiếp, `dung_lai` / im lặng 2 lần thì kết thúc ca.
- [x] C2. Hành vi bệnh nhân trong sim: thỉnh thoảng **tụt lại** (dừng nghỉ, đi chậm), nghe nhắc thì bước nhanh lại;
      bị hỏi thì trả lời bằng **clip thật** `tiep_tuc` / `dung_lai` (44 clip mỗi loại trong manifest).
- [x] C3. Lúc dừng chờ, robot quay về phía bệnh nhân và chỉ tay ra hiệu đi tiếp.
- [ ] C4. Log + eval: số lần nhắc, số lần hỏi, kết quả, thời gian chờ (thêm vào `scripts/eval_godot_escort.py`).

### D. Nhận diện đúng người (mặt + quần áo + dáng)
- [ ] D1. Mặt NPC: texture mặt **do AI tạo** (người không tồn tại, không dùng ảnh người thật); đa dạng màu áo / quần.
- [ ] D2. Thử model pretrained **chưa chỉnh gì**: phát hiện mặt (SCRFD/RetinaFace), embedding mặt (ArcFace),
      ReID toàn thân (OSNet). Chụp frame từ camera robot trong Godot, đo tỉ lệ nhận đúng theo khoảng cách.
      **Kết quả kém thì dừng ở đây và báo lại**, không tích hợp mù.
- [ ] D3. Ghi nhớ lúc đón (APPROACH/hội thoại): lưu embedding mặt + thân, chỉ trong RAM, xóa khi kết thúc ca.
- [ ] D4. Lúc dẫn: chọn box khớp embedding thay cho "conf cao nhất"; khớp quá thấp ➔ coi như mất dấu (đưa vào C1).
- [ ] D5. Cảnh đông (3–4 bệnh nhân gần nhau): đo tỉ lệ bám nhầm người trước và sau.

### E. Báo cáo
- [ ] E1. Cập nhật tài liệu, bảng số liệu trước/sau cho báo cáo đồ án.

## Tiến độ (2026-11-11)
- A, B, C1–C3 xong. GUT 202/202, pytest pass. Câu dẫn đường: `sim_vision/godot/data/escort_lines.json`, pregen 147 câu mới vào `data/tts_cache`.
- Chạy thật (`run_simulation.py --controller rl --nlu phobert --escort-only --lag-prob 1.0`), 2 ca tới phòng 303:
  ca 1 nhắc ➔ hỏi ➔ bệnh nhân nói "Được rồi, đi tiếp đi." ➔ PhoBERT `tiep_tuc` ➔ dẫn tiếp ➔ tới nơi;
  ca 2 nhắc 2 lần ➔ bệnh nhân tự theo kịp ➔ dẫn tiếp ➔ tới nơi. Mọi câu robot nói đều trúng cache TTS (~1 ms).
- Lỗi tìm được khi chạy thật và đã sửa: bệnh nhân đã quay lại gần (1.4 m) nhưng tia nhìn bị che ➔ robot vẫn hỏi.
  Giờ ở gần coi như theo kịp; mất dấu chỉ đếm khi đang xa.
- Còn lại: C4 (số liệu eval), báo rẽ khi đích chưa nhìn thấy, D, E.

## Thứ tự làm
A ➔ B ➔ C ➔ D ➔ E. A, B, C chỉ là Godot + câu thoại, rủi ro thấp. D là phần nặng nhất và có rủi ro kỹ thuật thật
(mặt NPC low-poly có đủ giống người để model pretrained nhận ra không), nên làm D2 để đo trước rồi mới tích hợp.

## Rủi ro / phản biện đã biết
- **Robot dừng chờ** là lệnh ghi đè ngoài policy (v = 0). Không đổi obs nên không cần train lại, nhưng phải ghi rõ
  trong báo cáo là hành vi do supervisor, không phải do RL học.
- Khoảng cách tether ở C1 tạm dùng vị trí thật trong sim; sau D4 sẽ đổi sang khoảng cách ước lượng từ camera.
- Chỉ tay chỉ có ý nghĩa khi đích **nhìn thấy được** (có line-of-sight); đích sau tường thì chỉ theo hướng đi.
- Bệnh nhân nội trú mặc đồ giống nhau ➔ ReID theo quần áo yếu; ghi vào phần hạn chế.
- Dữ liệu mặt là dữ liệu sinh trắc học (Nghị định 13/2023): chỉ so khớp trong một ca, không lưu.
