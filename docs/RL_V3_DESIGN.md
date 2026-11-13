# RL V3: robot tự học cách đối mặt, hỏi và ghi nhớ người cần giúp

Ngày: 2026-11-12 · Nhánh: `feat/rl-v3-engagement` · Mốc trước V3: commit `f22b271`

## Vì sao có V3

V2 lái tốt (6/6 bệnh nhân, 0 va chạm) nhưng phần **tương tác** đang là luật viết tay hoặc không có:
robot dừng nói chuyện mà không quay về phía người nói, không chủ động hỏi, không ngẩng đầu để thấy mặt người đứng gần.
Bug đã gặp: robot quay lưng, NPC đứng dậy nói, robot chỉ nghe rồi dẫn đi mà không thấy mặt ➔ không nhớ được ai.

Môi trường train V1/V2 còn tự chuyển sang dẫn đường ngay khi robot tới đủ gần (`hospital_env_pure_vision.py` 1b):
không có bước hỏi, đối mặt, ghi nhớ. V3 đưa các bước đó vào môi trường train để PPO **tự học**.

## Phân chia: cảm biến (không học) và quyết định (RL học)

| Cảm biến (có sẵn, không học) | Quyết định (V3 học) |
|---|---|
| Micro mảng: có tiếng người (VAD) + hướng giọng nói (DOA) | Nghe tiếng gọi ➔ quay lại hay đi tiếp |
| YuNet: mặt, vị trí mặt trong ảnh | Ngẩng / cúi đầu để mặt nằm trong khung |
| YOLO + ReID: box người, đúng người | Hỏi "Cần tôi giúp gì không ạ?" lúc nào, hỏi ai |
| Depth camera: vật cản | Lúc nói chuyện đứng yên, giữ mặt người trong khung đến khi ghi nhớ xong |

Cử chỉ tay vẫn là luật, gắn theo hành động "hỏi" (vẫy) và "dẫn" (chỉ), không đưa vào RL.

## Quan sát: 31-D (21-D của V2 + 10 mới)

| Chỉ số | Ý nghĩa | Nguồn khi chạy Godot / robot thật |
|---|---|---|
| 0–20 | Giữ nguyên V2 (depth 5 hướng, box bệnh nhân, pha, đích, vận tốc, tether) | như V2 |
| 21 | Có tiếng người nói / gọi (0/1) | VAD micro (sim: NPC đang phát clip) |
| 22, 23 | Hướng giọng nói so với hướng robot (sin, cos); không có tiếng ➔ 0, 0 | DOA micro mảng (sim: vị trí NPC đang nói) |
| 24 | Thấy mặt (0/1) | YuNet |
| 25, 26 | Tâm mặt trong ảnh (cx, cy ∈ −0.5..0.5; cy > 0 là mặt nằm thấp) | YuNet |
| 27 | Cỡ mặt (chiều cao box mặt / chiều cao ảnh) | YuNet |
| 28 | Góc ngẩng đầu hiện tại (−1 = cúi 15°, 0 = ngang, 1 = ngẩng 30°) | encoder servo cổ |
| 29 | Thời gian từ lần hỏi gần nhất / 5 s (1 = lâu rồi / chưa hỏi) | trạng thái robot |
| 30 | Đang nói chuyện (0/1) | trạng thái hội thoại |

Mặt được chọn: người đang nói chuyện với robot; nếu không có ai thì mặt gần giữa khung nhất.

## Hành động: 4-D (giữ kiểu liên tục ∈ [−1, 1] để dùng lại mạng PPO)

| Chỉ số | Ý nghĩa |
|---|---|
| 0, 1 | v, ω như V2 |
| 2 | Góc ngẩng đầu mục tiêu: a < 0 ➔ cúi tới 15°, a > 0 ➔ ngẩng tới 30°; đầu quay tối đa 60°/s |
| 3 | Hỏi: a > 0.5 ➔ robot nói "Cần tôi giúp gì không ạ?" (khoá 3 s giữa hai lần) |

## Mô hình người trong môi trường train

- Mỗi bệnh nhân: **cần giúp** (80%) hay **không** (20%, hỏi sẽ từ chối); **tự gọi** robot (50% số người cần giúp)
  hay **chờ được hỏi**; hướng mặt ngẫu nhiên khi ngồi chờ, quay về robot (90°/s) khi gọi / được hỏi / đang nói chuyện.
- **Người tự gọi**: robot trong vòng 4 m ➔ gọi 2.5 s, nghỉ 4 s, tối đa 3 lần rồi thôi.
- **Thấy mặt** khi: trong FOV ngang của camera, trong FOV dọc sau khi tính góc ngẩng (mặt ngồi cao 1.15 m, đứng 1.6 m,
  camera 1.2 m), người đó quay mặt về robot (lệch ≤ 60°), ≤ 4.5 m (xác suất giảm từ 3 m), không bị che.
  Người đứng sát robot (< ~0.7 m) chỉ thấy mặt khi robot ngẩng đầu.
- **Bắt đầu nói chuyện** khi: robot hỏi mà mặt gần giữa khung là người cần giúp, cách ≤ 2.5 m; hoặc người đang gọi
  thấy được mặt và cách ≤ 2.5 m (không cần hỏi lại); hoặc **người tự tới nói**: người cần giúp, robot dừng / đi chậm
  (< 0.35 m/s) trong vòng 2 m ➔ họ tự nói (0.4 lần/s), kể cả khi robot không nhìn họ (như `npc.gd` trong Godot).
  Một nửa số lần họ đứng dậy bước sát lại (0.6–1.3 m, lệch tới ±100° so với hướng robot).
- **Nói chuyện** 3–5 s, robot phải đứng yên. Mặt người đó trong khung đủ 1 s ➔ **ghi nhớ được**. Hết giờ mà chưa ghi
  nhớ ➔ chờ thêm tối đa 4 s; vẫn không ➔ dẫn đi nhưng bị phạt (robot không biết mình đang dẫn ai).
- Sau nói chuyện ➔ dẫn đường như V2.

## Thưởng (cộng thêm vào thưởng V2)

| Sự kiện | Thưởng |
|---|---|
| Ghi nhớ được mặt người đang nói chuyện | +1.0 (một lần) |
| Lúc nói chuyện: thấy mặt / không thấy mặt | +0.03 / −0.03 mỗi bước |
| Kết thúc nói chuyện mà chưa ghi nhớ được | −1.0 |
| Lúc nói chuyện mà robot đi (v > 0.05) | −0.05 mỗi bước |
| Hỏi khi không có mặt ai trong 2.5 m | −0.5 |
| Hỏi lại người đã từ chối / hỏi trong lúc khoá | −0.3 |
| Hỏi người không cần giúp (họ từ chối) | −0.1 |
| Có người đang gọi trong 4 m mà robot không thấy mặt họ | −0.01 mỗi bước |
| Người gọi bỏ cuộc (gọi 3 lần không ai đáp) | −1.0 |
| Đang đi nhanh (v > 0.2) mà đầu không để ngang | −0.003 × |góc ngẩng chuẩn hoá| mỗi bước |
| Lúc nói chuyện: góc tới người nói nhỏ đi / lớn lên | +0.3 / −0.3 mỗi rad (shaping theo thế năng, không đổi chính sách tối ưu) |

Thưởng V2 (đến nơi +10, giao bệnh nhân, va chạm, tether, …) giữ nguyên. `R_ENGAGED` (+2) giờ chỉ đến sau khi nói
chuyện xong, tức là sau khi đối mặt và (thường là) ghi nhớ được.

## Train

- `src/training/hospital_env_engage_v3.py` (kế thừa V2), `src/training/train_engage_v3.py`.
- Khởi tạo từ V2 (`PPOHospital-PureVisionV2_best.pt`): chép trọng số 21 đầu vào / 2 đầu ra cũ, phần mới khởi tạo 0
  ➔ phần lái không phải học lại từ đầu.
- Thầy (luật) cho V3 dẫn dắt 300 ca đầu rồi tắt dần, như V2; luật thầy = bản luật hiện có trong Godot.
- 2000 ca × 700 s như V2; chọn checkpoint tốt nhất theo đánh giá cố định.
- So sánh với **thầy luật** trên cùng bộ ca đánh giá: số người được giúp, tỉ lệ ghi nhớ được mặt, số lần hỏi thừa,
  số người gọi bị bỏ qua.

## Đưa vào Godot

- `robot.gd`: đầu (và camera trước) ngẩng / cúi được; `rl_policy_server.py`: quan sát 31-D, hành động 4-D.
- Godot gửi thêm: tiếng nói (NPC đang phát clip + vị trí ➔ DOA), box mặt YuNet, góc đầu, trạng thái hỏi / nói chuyện.
- Hành động "hỏi" ➔ robot nói câu hỏi, vẫy tay; NPC cần giúp đáp bằng clip yêu cầu (luồng hội thoại hiện có).
- Bản luật hiện tại (`_face_speaker`, …) giữ lại làm baseline, bật bằng `--controller rl --policy v2`.

## Rủi ro

- Mô hình người trong Python đơn giản hơn Godot (không có hoạt hình, mặt luôn "thấy được" khi đủ điều kiện hình học).
- Thêm 2 hành động làm PPO khó hội tụ hơn ➔ dùng thầy + khởi tạo từ V2; nếu không hội tụ thì giảm dần (bỏ ngẩng đầu
  khỏi RL, để luật).
- Camera ngẩng lên thì ảnh YOLO khác lúc train YOLO ➔ phạt ngẩng đầu khi đang đi để đầu về ngang khi di chuyển.

## Kết quả và các lần sửa

1. **Train lần 1** (2000 ca, khởi tạo từ V2): trên môi trường train, V3 dẫn tới nơi 5.17/6 người, luật 3.03; va chạm 0 so
   với 11.8. Trong Godot: chỉ 2/5 hội thoại ghi nhớ đúng người.
2. **Soi quan sát Godot gửi** (`RL_OBS_DUMP` trong `rl_policy_server.py`): NPC trong Godot tự nói khi robot ở gần, kể cả
   đứng sát bên cạnh / sau lưng; YOLO mất box người đứng sát ➔ server đổi pha sang SEARCH giữa lúc nói chuyện. Cả hai
   không có lúc train. Sửa: thêm "người tự tới nói" vào môi trường train; server giữ pha APPROACH khi đang nói chuyện;
   Godot không lấy mặt người khác khi người nói ở ngoài khung.
3. **Train tiếp 1000 ca** trên môi trường mới: không cải thiện (người tự tới nói: ghi nhớ 83–86%, luật 100%).
   Nguyên nhân: policy không dùng dấu của hướng giọng nói, luôn quay trái (người ở sau-phải ➔ quay cả vòng, dừng khi người
   vẫn sau lưng). Trọng số vào DOA khởi tạo 0, std của ω đã co về 0.11, thưởng ghi nhớ chỉ đến cuối hội thoại.
   Sửa: thêm shaping theo góc tới người nói, đặt lại std của ω = 0.3 khi train tiếp.

## Vòng 9 (2026-11-13): thoát khỏi người chắn lối, quay hẳn về người nói

### Vì sao

Log `run_1335`, `run_1427`, `run_1500`: dẫn xong robot kẹt trong phòng, đẩy vào bệnh nhân vừa giao; ca đầu `run_1500` ghi nhớ
nhầm người. Nguyên nhân tìm được:

1. **Godot che người vừa giao khỏi depth** (`robot.depth_ignore` vẫn trỏ vào họ ở pha quay về) ➔ 3 sector giữa báo "trống"
   trong khi người đứng cách 0.63 m. Lỗi harness, không phải policy.
2. **Lúc train bệnh nhân đi chếch trước robot, trong Godot họ đi sau lưng** (camera trước chỉ thấy 4–8% thời gian) ➔ tới nơi họ
   đứng giữa robot và cửa. Policy chưa từng gặp cảnh này. Đưa đúng cảnh đó vào môi trường train, model vòng 8 còn 1.9 người
   / ca, 31 va chạm, đẩy vào người 53 s / ca.
3. **Lúc nói chuyện policy vẫn bị kéo về điểm tuần tra** (ô "đích" 13–15) ➔ quay về giọng nói rồi dừng khi còn lệch ~60°;
   người bên hông / sau lưng chỉ ghi nhớ được 42–44%.
4. Godot ép robot dừng khi cách người gần nhất 1.85 m (kể cả người đi ngang) và không cho policy quay lúc đứng chờ.
5. Camera 86° không có trí nhớ: người vừa ra khỏi khung (đứng bên hông) là "không tồn tại" với policy.
6. Tâm box bệnh nhân server gửi (−0.5..0.5) bằng nửa lúc train (−1..1); pha quay về Godot gửi `CALL_STAFF`, lúc train là
   `SEARCH` + "đang quay về".

### Thay đổi

Quan sát 35 ➔ **45-D** (`hospital_env_engage_v3.py`, dựng bằng cùng code ở `rl_policy_server.py`, lớp dùng chung
`src/training/local_memory.py`):

| Chỉ số | Ý nghĩa | Nguồn |
|---|---|---|
| 13–15 lúc nói chuyện | "đích" = hướng giọng nói đã nhớ ➔ quay về người nói dùng lại kỹ năng "quay về đích" | DOA + odometry |
| 18–20 | tether tính cả camera sau (bệnh nhân thường đi sau lưng) | ReID trước / sau (`v3.tether`) |
| 35–40 | trí nhớ vật cản: độ gần ở 6 sector ngoài khung nhìn (trái .. sau .. phải), từ depth 8 s gần nhất + odometry | depth + odometry |
| 41 | tín hiệu gỡ kẹt: ra lệnh đi 0.4 s mà không nhúc nhích ➔ cửa sổ 2.2 s, dấu = phía nên quay (cố định trong cửa sổ) | odometry so với lệnh |
| 42–44 | box người to nhất trong khung [thấy, cx, h], bất kể là ai (tường và người đều lấp depth, chỉ người mới xin đường được) | YOLO |

Hành động "hỏi" (a[3]) thêm nghĩa thứ hai: với người đã gặp, hoặc khi robot đang bận (dẫn / quay về), nó là
**"Xin lỗi, anh chị cho tôi đi nhờ một chút ạ"** với người đứng ngay trước mặt (≤ 1.5 m, ±40°); 85% số người bước sang bên
(đang đứng trong ô cửa thì bước hẳn ra). Robot chủ động xin đường, không trông vào việc NPC tự né.

Mô hình người lúc train: 60% bệnh nhân đi sau lưng robot theo vết đường, 50% dừng lại 5–18 s một lần; giao xong họ đứng tại
chỗ (50% sau 3–20 s mới đi tiếp vào phòng); NPC nhân viên chỉ tự né 50% số lần. Thưởng thêm: −0.2 / bước khi lái vào người
đứng chắn, +0.02 × độ chính giữa của mặt lúc nói chuyện.

Train (`src/training/train_engage_v3_skills.py`, rollout song song): trộn ca đầy đủ với 2 bài tập ngắn, **người tự tới
nói** và **thoát khỏi phòng** (`setup_exit_drill`). Ở các "trạng thái kỹ năng" (đang nói chuyện, đang gỡ kẹt, bài tập) luật
thầy cho nhãn bắt chước (DAgger) cạnh loss PPO, trọng số giảm dần tới sàn; lái thường để PPO tự lo. Thầy chỉ dùng quan sát
(không có trạng thái ẩn) nên policy bắt chước được chính xác.

Godot: bỏ che người vừa giao khỏi depth; V3 bỏ luật ép dừng 1.85 m, policy tự quay / ngẩng đầu khi đứng chờ;
`_robot_ask()` thêm nhánh xin đường, `HospitalNpc.make_way()` / `Human.make_way()`; gửi thêm `v3.held`, `v3.tether`.

### Kết quả (môi trường train mới, 40 ca × 700 s + 200 lượt mỗi bài tập, cùng seed)

| Chỉ số | Vòng 8 | Vòng 9 | Luật thầy |
|---|---|---|---|
| Người dẫn tới nơi / ca (trung bình 20% số người từ chối) | 1.94 | 4.95 | 4.48 |
| Ca phục vụ hết mọi người | 12% | 85% | 78% |
| Va chạm / ca | 30.9 | 0.82 | 1.05 |
| Thời gian đẩy vào người / ca | 53.5 s | 1.6 s | 5.3 s |
| Người tự tới nói: ghi nhớ được mặt (bên hông / sau lưng) | 87% (83%) | 99% (99%) | 98% (98%) |
| Thoát khỏi phòng qua người vừa giao | 35% | 100% | 98% |
| Hỏi thừa / ca | 2.19 | 0.12 | 0.98 |

Số vòng 8 đo trên 16 ca / 100 lượt (dòng `[INIT policy]` của log train). Vòng 9 train 3 chặng × 6000 ca
(`experiments/runs/20261113_154054`, `_180145`, `_184022`; chặng 2–3 tăng tỉ lệ ca đầy đủ lên 50%), model cài là best của
chặng 3. Chặng 2 chạy qua `rl_policy_server.py` (quan sát dựng từ thông điệp như Godot gửi) thoát phòng 30/31 lượt.

### Chưa xong / rủi ro

- Chưa đo trong Godot: cần chạy `run_simulation.py --policy v3` và đọc log.
- Người gọi bị bỏ qua tới mức bỏ cuộc: 0.72 / ca.
- Robot chưa tự đứng chờ khi bệnh nhân dừng lại giữa đường (Godot vẫn có luật `EscortMonitor` giữ robot).
- YOLO nhận người áo đỏ (`human_m06`) thành `pccc_station` ở gần ➔ không có box người ➔ không ghi nhớ / xin đường theo box được.
- Câu `excuse_me` chưa có giọng sinh sẵn (`python -m tts.pregen`).
- Model vòng 8 giữ ở `experiments/checkpoints/PPOHospital-EngageV3_round8_35d.pt` (server vẫn chạy được model 35-D).
