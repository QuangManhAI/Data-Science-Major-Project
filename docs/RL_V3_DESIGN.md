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
