# Báo cáo Đánh giá Chuyên sâu Mô hình ASR (PhoASR-Whisper-Small)

- **Model Checkpoint:** `best`
- **Tập kiểm thử:** `test` (1726 câu, 117.51 phút âm thanh)
- **Phần cứng:** `NVIDIA GeForce RTX 4060 Laptop GPU` (Batch size: 16, Num beams: 1)
- **Thời gian thực thi:** 2026-11-12 08:46:28

---

## 1. Kết quả Tổng quan & So sánh Baseline

| Chỉ số | Kết quả Checkpoint Epoch 1 | Baseline Chưa Fine-tune | Đánh giá / Tiến bộ |
|:---|:---:|:---:|:---:|
| **WER Chuẩn hóa Tổng thể** | **11.69%** | **19.10%** | Cần cải thiện |
| **CER Chuẩn hóa** | **8.59%** | 12.90% | Đạt |
| **Tỷ lệ Câu Đúng 100% (Sent Acc)** | **62.11%** | 41.20% | Khá |
| **Macro WER theo Người nói** | **12.65%** | 17.90% | Đạt |
| **Real-Time Factor (RTF)** | **0.0273** | 0.0540 (MPS) | ✅ Đạt chuẩn Realtime (<0.08) |

### Chi tiết Phân rã Sai số (Levenshtein S-D-I):
- **Thay thế (Substitutions - S):** 1045 từ
- **Nuốt từ (Deletions - D):** 192 từ
- **Thêm từ / Ảo giác (Insertions - I):** 456 từ
- **Tổng số từ trong nhãn:** 14488 từ

---

## 2. Phân tầng theo Chế độ Thu âm (Mode)

| Chế độ | Số mẫu | WER (%) | CER (%) | Sent Acc (%) | Ghi chú |
|:---|:---:|:---:|:---:|:---:|:---|
| **Đọc chuẩn (`_r`)** | 883 | **6.87%** | 4.57% | 71.80% | Phát âm chuẩn theo kịch bản |
| **Tự nhiên (`_n`)** | 823 | **17.01%** | 12.99% | 51.03% | Tự nhiên, có nuốt chữ/nói nhanh |

---

## 3. Phân tầng theo Người nói (Speaker Breakdown)

| Speaker ID | Số câu | Thời lượng (phút) | WER (%) | CER (%) | Sent Acc (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `LH_F30_02` | 70 | 7.2 | **0.56%** | 0.14% | 97.14% |
| `LH_A2` | 80 | 5.9 | **0.60%** | 0.40% | 95.00% |
| `LH_B` | 80 | 6.1 | **1.13%** | 0.72% | 92.50% |
| `LH_A` | 80 | 4.3 | **2.70%** | 1.82% | 83.75% |
| `LH_B2` | 80 | 5.6 | **2.97%** | 2.28% | 78.75% |
| `HT_F20` | 10 | 0.7 | **3.33%** | 2.88% | 80.00% |
| `X_F30` | 70 | 4.3 | **4.10%** | 2.55% | 81.43% |
| `LH_B3` | 80 | 4.5 | **4.11%** | 2.46% | 77.50% |
| `LH_B8` | 79 | 5.2 | **5.31%** | 3.99% | 73.42% |
| `LH_F30_03` | 70 | 5.2 | **5.97%** | 4.40% | 75.71% |
| `LH_F30_04` | 70 | 4.9 | **6.72%** | 4.58% | 72.86% |
| `LH_B7` | 80 | 5.8 | **6.94%** | 4.84% | 62.50% |
| `LH_F30_01` | 70 | 5.7 | **7.28%** | 5.05% | 62.86% |
| `PU_M10` | 25 | 1.8 | **7.98%** | 4.73% | 64.00% |
| `APP_c5de7e` | 22 | 1.1 | **8.33%** | 5.78% | 45.45% |
| `LH_M30` | 70 | 4.6 | **9.14%** | 5.69% | 64.29% |
| `APP_f91703` | 80 | 3.8 | **9.76%** | 7.34% | 70.00% |
| `PU_F10` | 25 | 1.8 | **10.11%** | 6.35% | 52.00% |
| `APP_ba76d1` | 8 | 0.7 | **10.64%** | 8.93% | 50.00% |
| `APP_1603b1` | 80 | 5.0 | **15.44%** | 12.50% | 48.75% |
| `LH_A4` | 1 | 0.1 | **16.67%** | 14.58% | 0.00% |
| `HT_M20` | 70 | 5.3 | **18.15%** | 15.82% | 50.00% |
| `LH_B1` | 80 | 5.9 | **19.55%** | 11.81% | 26.25% |
| `PU_F20_A` | 80 | 6.7 | **19.97%** | 17.44% | 40.00% |
| `PU_F20_B` | 80 | 6.8 | **23.80%** | 19.94% | 35.00% |
| `APP_5ac354` | 80 | 3.9 | **26.88%** | 19.73% | 33.75% |
| `APP_2ff99b` | 11 | 0.6 | **32.08%** | 19.18% | 27.27% |
| `LH_A3` | 80 | 3.6 | **40.84%** | 28.05% | 13.75% |
| `APP_112ba0` | 15 | 0.8 | **45.81%** | 28.86% | 6.67% |

---

## 4. Phân tầng theo Độ dài & Đặc tính Câu hỏi

| Phân nhóm | Số câu | WER (%) | Sent Acc (%) |
|:---|:---:|:---:|:---:|
| **Câu chứa số phòng/tầng (`has_digit = True`)** | 89 | **11.33%** | 52.81% |
| **Câu không chứa số (`has_digit = False`)** | 1637 | **11.71%** | 62.61% |
| **Thời lượng: <2s (ngắn)** | 43 | **37.85%** | 41.86% |
| **Thời lượng: 2-4s (vừa)** | 822 | **12.13%** | 62.90% |
| **Thời lượng: >4s (dài)** | 861 | **10.55%** | 62.37% |

---

## 5. Độ nhạy Từ vựng Chuyên ngành Bệnh viện (Domain Lexicon Recall)

| Từ khóa Bệnh viện | Số lần xuất hiện | Số lần nhận đúng | Tỷ lệ Nhận đúng (Recall) |
|:---|:---:|:---:|:---:|
| `x quang` | 17 | 14 | **82.35%** |
| `bảo hiểm` | 85 | 82 | **96.47%** |
| `siêu âm` | 58 | 49 | **84.48%** |
| `khoa sản` | 56 | 51 | **91.07%** |
| `khoa nhi` | 44 | 37 | **84.09%** |
| `cấp cứu` | 14 | 11 | **78.57%** |
| `phòng khám` | 16 | 16 | **100.00%** |
| `robot` | 14 | 12 | **85.71%** |
| `thang máy` | 30 | 24 | **80.00%** |
| `nhà thuốc` | 70 | 63 | **90.00%** |

---

## 6. Phân tích Các Trường hợp Lỗi Nặng Nhất (Top Worst Samples)

| # | Speaker | Mode | WER | Nhãn chuẩn hóa (Reference) | Mô hình nhận dạng (Hypothesis) |
|:---:|:---:|:---:|:---:|:---|:---|
| 1 | `PU_F20_A` | `n` | 180.0% | "quầy thu ngân ở đâu" | "cho tôi hỏi có đòi thu ngân ở tại nền bá thầu" |
| 2 | `PU_F20_A` | `n` | 140.0% | "khoa cấp cứu ở đâu" | "cho mày hỏi khoa thuốc cú nằm ở đâu vậy" |
| 3 | `PU_F20_B` | `n` | 133.3% | "cảm ơn nhé" | "ok ok mình cảm ơn" |
| 4 | `PU_F20_A` | `n` | 125.0% | "nhà thuốc ở đâu" | "cho tôi hỏi nhà thuốc nằm ở đâu vậy" |
| 5 | `LH_A3` | `n` | 100.0% | "nhà thuốc ở đâu" | "dạ vâng" |
| 6 | `LH_A3` | `n` | 100.0% | "gọi nhân viên giúp tôi với" | "những ký kết tối đấy" |
| 7 | `LH_A3` | `n` | 100.0% | "đi chậm lại giúp tôi" | "là em chết rồi" |
| 8 | `APP_5ac354` | `n` | 100.0% | "thang máy gần nhất ở đâu" | "thế mấy người mỹ có đau" |
| 9 | `APP_5ac354` | `n` | 100.0% | "cho mình hỏi nhà vệ sinh nam ở đâu" | "minh bé ơi cho mình hỏi chân sứ nó đau nhỉ" |
| 10 | `APP_5ac354` | `n` | 100.0% | "quầy thu ngân ở đâu" | "này chú ớ đâu nhỉ" |
