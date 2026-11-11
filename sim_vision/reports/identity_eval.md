# Đo nhận diện đúng người trên camera robot (Godot)

Ghi nhớ từ 1 frame / người (lúc nói chuyện: đứng, nhìn thẳng, 2.0 m). 10 người, 1120 frame, 1076 frame thử. Đúng = giống người đã ghi nhớ hơn 9 người còn lại.

## Phát hiện (YOLO của robot, lớp patient/doctor) và mặt (YuNet) theo khoảng cách

| Khoảng cách (m) | YOLO thấy người | Thấy mặt (khi nhìn trước) | Cỡ mặt (px) |
|---|---|---|---|
| 1.0 | 90% | 90% | 72 |
| 1.5 | 91% | 90% | 45 |
| 2.0 | 98% | 100% | 33 |
| 3.0 | 100% | 100% | 22 |
| 4.0 | 100% | 92% | 14 |
| 5.0 | 100% | 57% | 12 |
| 6.0 | 99% | 15% | 10 |

## Tỉ lệ bám đúng người theo khoảng cách

| Khoảng cách (m) | Toàn thân OSNet x0_25 | n |
|---|---|---|
| 1.0 | 88% | 144 |
| 1.5 | 100% | 146 |
| 2.0 | 100% | 147 |
| 3.0 | 99% | 160 |
| 4.0 | 99% | 160 |
| 5.0 | 99% | 160 |
| 6.0 | 97% | 159 |

## Tỉ lệ bám đúng người theo hướng nhìn

| Hướng | Toàn thân OSNet x0_25 | n |
|---|---|---|
| ngang (90°) | 97% | 274 |
| sau (135–180°) | 96% | 404 |
| trước (0–45°) | 100% | 398 |

## Tình huống dẫn đường (bệnh nhân đi phía trước, quay lưng, 1–3 m)

| Cách | Bám đúng |
|---|---|
| Toàn thân OSNet x0_25 | 94% (n=112) |

## Ngưỡng chấp nhận (lúc chạy: không box nào vượt ngưỡng ➔ coi như mất dấu, không bám người lạ)

Ngưỡng = phân vị 99% điểm giống của người lạ giống nhất ➔ ≤ 1% frame nhận nhầm người lạ.

| Model | Ngưỡng cosine | Nhận đúng người (≥ ngưỡng) | Nhận đúng, lúc quay lưng 1–3 m |
|---|---|---|---|
| OSNet x0_25 | 0.78 | 67% | 62% |

## Theo từng người (toàn thân)

| Người | OSNet x0_25 | Hay nhầm với (x0_25) |
|---|---|---|
| human_m06 | 100% | – |
| medical_doctor | 100% | – |
| medical_nurse | 98% | medical_doctor |
| patient_elder_f09 | 100% | – |
| patient_elder_m03 | 99% | patient_elder_f09 |
| patient_elder_m14 | 100% | – |
| patient_f02 | 95% | patient_m02 |
| patient_f08 | 98% | medical_doctor |
| patient_m02 | 96% | medical_doctor |
| patient_m05 | 91% | patient_elder_f09 |

---

# Đo nhận diện đúng người trên camera robot (Godot)

Ghi nhớ từ 2 frame / người (lúc nói chuyện nhìn thẳng + lúc vừa quay lưng đi theo robot, 2.0 m). 10 người, 1120 frame, 1067 frame thử. Đúng = giống người đã ghi nhớ hơn 9 người còn lại.

## Phát hiện (YOLO của robot, lớp patient/doctor) và mặt (YuNet) theo khoảng cách

| Khoảng cách (m) | YOLO thấy người | Thấy mặt (khi nhìn trước) | Cỡ mặt (px) |
|---|---|---|---|
| 1.0 | 90% | 90% | 72 |
| 1.5 | 91% | 90% | 45 |
| 2.0 | 98% | 100% | 33 |
| 3.0 | 100% | 100% | 22 |
| 4.0 | 100% | 92% | 14 |
| 5.0 | 100% | 57% | 12 |
| 6.0 | 99% | 15% | 10 |

## Tỉ lệ bám đúng người theo khoảng cách

| Khoảng cách (m) | Toàn thân OSNet x0_25 | n |
|---|---|---|
| 1.0 | 90% | 144 |
| 1.5 | 100% | 146 |
| 2.0 | 100% | 138 |
| 3.0 | 100% | 160 |
| 4.0 | 100% | 160 |
| 5.0 | 100% | 160 |
| 6.0 | 99% | 159 |

## Tỉ lệ bám đúng người theo hướng nhìn

| Hướng | Toàn thân OSNet x0_25 | n |
|---|---|---|
| ngang (90°) | 96% | 274 |
| sau (135–180°) | 98% | 395 |
| trước (0–45°) | 100% | 398 |

## Tình huống dẫn đường (bệnh nhân đi phía trước, quay lưng, 1–3 m)

| Cách | Bám đúng |
|---|---|
| Toàn thân OSNet x0_25 | 99% (n=112) |

## Ngưỡng chấp nhận (lúc chạy: không box nào vượt ngưỡng ➔ coi như mất dấu, không bám người lạ)

Ngưỡng = phân vị 99% điểm giống của người lạ giống nhất ➔ ≤ 1% frame nhận nhầm người lạ.

| Model | Ngưỡng cosine | Nhận đúng người (≥ ngưỡng) | Nhận đúng, lúc quay lưng 1–3 m |
|---|---|---|---|
| OSNet x0_25 | 0.79 | 81% | 88% |

## Theo từng người (toàn thân)

| Người | OSNet x0_25 | Hay nhầm với (x0_25) |
|---|---|---|
| human_m06 | 100% | – |
| medical_doctor | 96% | medical_nurse |
| medical_nurse | 100% | – |
| patient_elder_f09 | 100% | – |
| patient_elder_m03 | 100% | – |
| patient_elder_m14 | 100% | – |
| patient_f02 | 94% | patient_m02 |
| patient_f08 | 99% | medical_doctor |
| patient_m02 | 97% | medical_doctor |
| patient_m05 | 98% | patient_elder_f09 |
