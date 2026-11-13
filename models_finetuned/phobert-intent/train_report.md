# Báo cáo fine-tune PhoBERT-base-v2 (G19)

Ngày: 2026-11-14 01:38. Thiết bị: mps. Dữ liệu: train 2643 câu, val 360 câu (`data/nlu`). Không dùng tập test.

## Siêu tham số

| Tham số | Giá trị |
| :-- | :-- |
| lr | 2e-05 |
| batch | 16 |
| epochs | 10 |
| warmup | 0.1 |
| weight_decay | 0.01 |
| patience | 3 |
| seed | 42 |
| max_len | 64 |
| grad_clip | 1.0 |
| optimizer | AdamW, lịch tuyến tính có warmup, không fp16 |

## Theo epoch

| epoch | train_loss | val_loss | val_macro_f1 | giây |
| --: | --: | --: | --: | --: |
| 1 | 2.7842 | 2.0205 | 0.9189 | 87 |
| 2 | 1.3177 | 0.5633 | 1.0000 | 87 |
| 3 | 0.3933 | 0.1538 | 1.0000 | 86 |
| 4 | 0.1435 | 0.0731 | 1.0000 | 88 |
| 5 | 0.0800 | 0.0460 | 1.0000 | 88 |
| 6 | 0.0553 | 0.0337 | 1.0000 | 88 |
| 7 | 0.0429 | 0.0271 | 1.0000 | 88 |
| 8 | 0.0360 | 0.0233 | 1.0000 | 86 |
| 9 | 0.0323 | 0.0213 | 1.0000 | 87 |
| 10 | 0.0301 | 0.0207 | 1.0000 | 87 |

Checkpoint tốt nhất: epoch 10 (macro-F1 val 1.0000). Early stopping theo macro-F1 val; bằng điểm ➔ chọn epoch có val loss thấp hơn.

## Kết quả trên val (checkpoint tốt nhất)

- Accuracy: 1.0000
- Macro-F1: 1.0000
- Ngưỡng tin cậy chọn trên val: **0.6** (macro-F1 khi coi `max_prob < ngưỡng` là `unknown`: 1.0000). Bằng điểm ➔ chọn ngưỡng gần 0.6 nhất.

| ngưỡng | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| macro-F1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### F1 từng lớp

| intent | F1 | số câu val |
| :-- | --: | --: |
| bao_hiem | 1.0000 | 18 |
| cam_on | 1.0000 | 18 |
| cap_cuu | 1.0000 | 18 |
| dan_duong | 1.0000 | 18 |
| dang_ky_kham | 1.0000 | 18 |
| dieu_chinh_am_luong | 1.0000 | 18 |
| dieu_chinh_toc_do | 1.0000 | 18 |
| dung_lai | 1.0000 | 18 |
| gap_nhan_vien | 1.0000 | 18 |
| ho_tro_di_chuyen | 1.0000 | 18 |
| hoi_khoa | 1.0000 | 18 |
| hoi_phong | 1.0000 | 18 |
| khong_ho_tro | 1.0000 | 18 |
| nha_thuoc | 1.0000 | 18 |
| nhac_lai | 1.0000 | 18 |
| thanh_toan | 1.0000 | 18 |
| thoi_gian_cho | 1.0000 | 18 |
| tien_ich | 1.0000 | 18 |
| tiep_tuc | 1.0000 | 18 |
| unknown | 1.0000 | 18 |

### Ma trận nhầm lẫn

| thật \ dự đoán | bao_hiem | cam_on | cap_cuu | dan_duon | dang_ky_ | dieu_chi | dieu_chi | dung_lai | gap_nhan | ho_tro_d | hoi_khoa | hoi_phon | khong_ho | nha_thuo | nhac_lai | thanh_to | thoi_gia | tien_ich | tiep_tuc | unknown |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| bao_hiem | 18 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| cam_on | · | 18 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| cap_cuu | · | · | 18 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| dan_duong | · | · | · | 18 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| dang_ky_kham | · | · | · | · | 18 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| dieu_chinh_am_luong | · | · | · | · | · | 18 | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| dieu_chinh_toc_do | · | · | · | · | · | · | 18 | · | · | · | · | · | · | · | · | · | · | · | · | · |
| dung_lai | · | · | · | · | · | · | · | 18 | · | · | · | · | · | · | · | · | · | · | · | · |
| gap_nhan_vien | · | · | · | · | · | · | · | · | 18 | · | · | · | · | · | · | · | · | · | · | · |
| ho_tro_di_chuyen | · | · | · | · | · | · | · | · | · | 18 | · | · | · | · | · | · | · | · | · | · |
| hoi_khoa | · | · | · | · | · | · | · | · | · | · | 18 | · | · | · | · | · | · | · | · | · |
| hoi_phong | · | · | · | · | · | · | · | · | · | · | · | 18 | · | · | · | · | · | · | · | · |
| khong_ho_tro | · | · | · | · | · | · | · | · | · | · | · | · | 18 | · | · | · | · | · | · | · |
| nha_thuoc | · | · | · | · | · | · | · | · | · | · | · | · | · | 18 | · | · | · | · | · | · |
| nhac_lai | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 18 | · | · | · | · | · |
| thanh_toan | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 18 | · | · | · | · |
| thoi_gian_cho | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 18 | · | · | · |
| tien_ich | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 18 | · | · |
| tiep_tuc | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 18 | · |
| unknown | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 18 |

### Cặp hay nhầm

- Không có câu val nào sai.
## Nạp lại từ thư mục đã lưu

Cùng dự đoán trên 100.0% câu val; chênh lệch xác suất lớn nhất 0.00e+00.
