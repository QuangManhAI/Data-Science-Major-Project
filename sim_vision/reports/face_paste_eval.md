# Dán mặt người thật lên NPC: nhận diện bằng mặt vs toàn thân

Camera robot (1.2 m, 640×480), hành lang tầng 1, 10 NPC, 1–6 m, 8 hướng, đứng / đang đi.
Mặt: YuNet + SFace (ngưỡng 0.363). Toàn thân: OSNet x0.25. Box người: YOLO của robot.

## Mặt Rocketbox gốc

Ghi nhớ được mặt: 10/10 người (frame nhìn thẳng 2 m).

| Khoảng cách (m) | YuNet thấy mặt (nhìn trước) | Nhận đúng bằng mặt (nhìn trước) | Nhận đúng bằng toàn thân (mọi hướng) |
|---|---|---|---|
| 1.0 | 100% | 100% | 88% |
| 1.5 | 100% | 100% | 100% |
| 2.0 | 100% | 100% | 100% |
| 3.0 | 100% | 80% | 99% |
| 4.0 | 92% | 27% | 99% |
| 5.0 | 57% | 12% | 99% |
| 6.0 | 15% | 0% | 97% |

| Hướng | Thấy mặt | Nhận đúng bằng mặt | Nhận đúng bằng toàn thân |
|---|---|---|---|
| ngang (90°) | 58% | 15% | 97% |
| sau (135–180°) | 18% | 0% | 96% |
| trước (0–45°) | 79% | 58% | 100% |

| Tình huống | Bằng mặt | Bằng toàn thân |
|---|---|---|
| Nói chuyện (nhìn trước, 1–3 m) | 94% | 100% |
| Dẫn đường (quay lưng, đang đi, 1–3 m) | 1% | 94% |

## Dán ảnh mặt người thật

Ghi nhớ được mặt: 10/10 người (frame nhìn thẳng 2 m).

| Khoảng cách (m) | YuNet thấy mặt (nhìn trước) | Nhận đúng bằng mặt (nhìn trước) | Nhận đúng bằng toàn thân (mọi hướng) |
|---|---|---|---|
| 1.0 | 100% | 93% | 85% |
| 1.5 | 100% | 94% | 100% |
| 2.0 | 100% | 90% | 100% |
| 3.0 | 93% | 58% | 100% |
| 4.0 | 85% | 30% | 99% |
| 5.0 | 55% | 8% | 98% |
| 6.0 | 17% | 3% | 94% |

| Hướng | Thấy mặt | Nhận đúng bằng mặt | Nhận đúng bằng toàn thân |
|---|---|---|---|
| ngang (90°) | 62% | 2% | 96% |
| sau (135–180°) | 19% | 0% | 94% |
| trước (0–45°) | 77% | 52% | 100% |

| Tình huống | Bằng mặt | Bằng toàn thân |
|---|---|---|
| Nói chuyện (nhìn trước, 1–3 m) | 83% | 100% |
| Dẫn đường (quay lưng, đang đi, 1–3 m) | 0% | 95% |
