# V3: luật vs học (môi trường train V3, cùng các ca đánh giá)

30 ca × 700 s, người ngẫu nhiên (cần giúp / không, tự gọi / chờ hỏi, hướng mặt). Checkpoint: `PPOHospital-EngageV3_best.pt`.

| Chỉ số | Luật (Godot hiện tại) | V3 học |
|---|---|---|
| Người được dẫn tới nơi / ca (6) | 3.17 | 5.20 |
| Ca giúp đủ 6 người | 30% | 83% |
| Hội thoại có ghi nhớ được mặt | 100% | 100% |
| Thời gian thấy mặt khi nói chuyện | 94% | 86% |
| Người tự tới nói (có thể bên cạnh / sau lưng): ghi nhớ được | 100% | 100% |
| Số lần hỏi / ca | 3.70 | 7.57 |
| Hỏi thừa / ca (không ai, hỏi lại) | 0.00 | 2.77 |
| Người gọi bị bỏ qua tới mức bỏ cuộc / ca | 0.60 | 0.00 |
| Bỏ đi giữa lúc nói chuyện / ca | 0.00 | 0.00 |
| Va chạm / ca | 12.53 | 0.00 |
| Tốc độ dẫn (m/s) | 0.36 | 0.46 |
| Tổng thưởng | 35.42 | 100.57 |
