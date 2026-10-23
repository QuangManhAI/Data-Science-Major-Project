# EXP_VISION_EXTENDED_NPC_BENCHMARK.md — Mở Rộng Benchmark Thị Giác với Hệ Sinh Thái NPC & Đạo Cụ Lâm Sàng Mới

- **Motivation/Background**: Sau khi sáp nhập nhánh `quang-manh`, môi trường mô phỏng bệnh viện Godot 4 đã được bổ sung đầy đủ hệ sinh thái nhân vật lâm sàng (Bác sĩ áo blouse trắng, Điều dưỡng áo scrub xanh, 3 phân lớp bệnh nhân cao tuổi, 4 bệnh nhân trưởng thành), các tư thế ngồi ghế chờ/ho/kiểm tra bệnh án, cơ chế động học né ngang "pardon sidestep", cùng 8 đạo cụ y tế mới (cột truyền dịch, máy monitor, cáng xe đẩy, kiosk xếp hàng, cột sát khuẩn, standee thông báo). Cần thiết lập bộ kịch bản mở rộng và đánh giá lại độ bền vững của 4 mô hình thị giác robot (`yolov8n`, `yolo11n`, `yolo26n`, `intel/person-retail-0013`).
- **Purpose**: Đặc tả toàn diện các ca kiểm thử mới (6 nhóm N1–N6), lập checklist triển khai tích hợp vào Godot Dataset Generator, thu thập tập dữ liệu mở rộng chuẩn hóa, thực thi benchmark đối chứng và cập nhật ma trận chỉ số kỹ thuật phục vụ triển khai robot trợ lý bệnh viện.
- **Hypothesis**:
  1. *Giả thuyết H1 (Màu áo & Ngụy trang)*: Bác sĩ mặc blouse trắng đứng trước tường trắng sẽ làm giảm Precision của Intel SSD do mất độ tương phản biên dạng, trong khi các dòng YOLO (Anchor-Free DFL) vẫn giữ được Recall $\ge 85\%$.
  2. *Giả thuyết H2 (Tư thế ngồi ghế & ho)*: Bệnh nhân ngồi ghế băng dài và gập người ho (`m_sit_cough`) làm giảm chiều cao hộp bao xuống $H_{\text{box}} \approx 0.45 - 0.50$, khiến MobileNetV2-SSD trượt mục tiêu ($Recall < 60\%$), đòi hỏi YOLOv8n/11n/26n phải thích ứng hình thái.
  3. *Giả thuyết H3 (Bẫy đạo cụ rỗng mới)*: Cột truyền dịch inox (`iv_stand`) và Standee in hình y tế (`info_standee`) sẽ tạo báo động giả nghiêm trọng trên Intel SSD ($FPR > 10\%$), trong khi các mô hình YOLOv8n/11n/26n duy trì $FPR \approx 0\%$.
  4. *Giả thuyết H4 (Cụm người ngồi liền kề)*: YOLO26n (NMS-Free Head) sẽ vượt trội so với YOLOv8n (Greedy NMS) trong việc phân tách 2 cụ già ngồi sát nhau trên ghế chờ mà không bị nuốt mất hộp bao.
- **Detailed Plan**:
  - §1 Ma Trận Phân Loại 6 Nhóm Ca Mới (N1–N6, 32 kịch bản chi tiết)
  - §2 Checklist Kế Hoạch Triển Khai Thực Hiện
  - §3 Kiến Trúc Mở Rộng Bộ Dataset Generator Godot 4
  - §4 Tiêu Chuẩn Đánh Giá Chỉ Số Đối Chứng Cả 4 Model
  - §5 Nhật Ký Thực Nghiệm & Kết Quả Đo Lường
- **References**: [`EXP_VISION_48CASES_1200_BENCHMARK.md`](EXP_VISION_48CASES_1200_BENCHMARK.md), [`sim_vision/reports/vision_48cases_benchmark.md`](../../sim_vision/reports/vision_48cases_benchmark.md), [`docs/AI-Human-logs/Vuong.md`](../AI-Human-logs/Vuong.md).
- **Created**: 2026-10-23T17:40:00+07:00
- **Status**: `[STATUS: IN_PROGRESS]`

---

## 1. Ma Trận Phân Loại 6 Nhóm Ca Mới (32 Kịch Bản Lâm Sàng Chi Tiết)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  MA TRẬN HỆ THỐNG KIỂM THỬ MỞ RỘNG (N1-N6)                             │
├────────────────────────────────┬───────────────────────────────┬───────────────────────────────────────┤
│ Nhóm N1: Nhân Khẩu & Màu Áo    │ Nhóm N2: Tư Thế Ngồi & Lâm Sàng│ Nhóm N3: Tương Tác & Che Khuất Đạo Cụ │
│  • N1.1 Blouse trắng tường kem │  • N2.1 Ngồi ghế băng dài     │  • N3.1 Đang truyền dịch (IV Stand)   │
│  • N1.2 Scrub xanh rèm y tế    │  • N2.2 Ngồi gập người ho     │  • N3.2 Bên máy monitor sinh hiệu     │
│  • N1.3 Cụ bà tóc bạc áo hoa   │  • N2.3 Ngồi buông thõng mỏi  │  • N3.3 Đang bấm Kiosk lấy số khám    │
│  • N1.4 Cụ ông lưng còng       │  • N2.4 Đọc kẹp bệnh án ngực  │  • N3.4 Lấy nước tại cây nước         │
│  • N1.5 Cụ ông áo kẻ sọc       │  • N2.5 Ngồi vắt chéo chân    │  • N3.5 Rửa tay cột sát khuẩn         │
│  • N1.6 Bệnh nhân áo dài/váy   │                               │  • N3.6 Đẩy xe cáng cấp cứu dài 2.1m  │
│  • N1.7 Bệnh nhân thể hình đậm │                               │  • N3.7 Đứng cạnh Standee thông báo   │
├────────────────────────────────┼───────────────────────────────┼───────────────────────────────────────┤
│ Nhóm N4: Bẫy Đạo Cụ Rỗng Mới   │ Nhóm N5: Động Học & Né Ngang  │ Nhóm N6: Đa Người & Xã Hội Bệnh Viện  │
│  • N4.1 Cột truyền dịch rỗng   │  • N5.1 Đi tuần tra đối đầu   │  • N6.1 Hai cụ già ngồi liền kề ghế   │
│  • N4.2 Cột khử khuẩn rỗng     │  • N5.2 Né ngang Pardon 0.55m │  • N6.2 Cặp Bác sĩ đứng - Bệnh nhân   │
│  • N4.3 Kiosk bấm số rỗng      │  • N5.3 Xoay thân qua cửa 90° │  • N6.3 Hàng chờ trước Kiosk khám     │
│  • N4.4 Máy monitor rỗng       │  • N5.4 Đi vội rẽ góc hành lang│ • N6.4 Bác sĩ & Y tá đối mặt 180°    │
│  • N4.5 Xe cáng cấp cứu rỗng   │                               │  • N6.5 Người nhà dìu bệnh nhân đi    │
│  • N4.6 Cây nước nóng lạnh rỗng│                               │                                       │
│  • N4.7 Standee in hình người  │                               │                                       │
│  • N4.8 Màn hình gọi số tường  │                               │                                       │
└────────────────────────────────┴───────────────────────────────┴───────────────────────────────────────┘
```

---

## 2. Checklist Kế Hoạch Triển Khai Thực Hiện (Implementation Checklist)

- [ ] **Giai đoạn 1: Lập Kế Hoạch & Thiết Kế Kiến Trúc (Planning & Design)**
  - [x] Lập tài liệu đặc tả `EXP_VISION_EXTENDED_NPC_BENCHMARK.md` và artifact planning.
  - [ ] Khảo sát các tài nguyên mô hình và hoạt ảnh thực tế trong `sim_vision/godot/scenes/props/`.
  - [ ] Thiết lập danh sách các thực thể mục tiêu và vị trí tọa độ dàn dựng (Staging positions) trong hành lang bệnh viện.

- [ ] **Giai đoạn 2: Tích Hợp Bộ Sinh Dữ Liệu Godot 4 (Generator Implementation)**
  - [ ] Nâng cấp scene `hospital_vision_bench.tscn`: bổ sung các instance NPC mới và đạo cụ mới vào các Pool tương ứng (`ActorsPool`, `ObstacleProps`, `HardNegativesPool`).
  - [ ] Mở rộng script `scene_dataset_generator.gd` với các hàm sinh chuyên biệt:
    - `_generate_d9_medical_attire()` (Nhóm N1: Blouse, Scrub, Elder textures)
    - `_generate_d10_seated_clinical_poses()` (Nhóm N2: Seated, Coughing, Documents)
    - `_generate_d11_prop_interactions()` (Nhóm N3: IV stand, Monitor, Kiosk, Stretcher)
    - `_generate_d12_new_hard_negatives()` (Nhóm N4: Empty IV, Kiosk, Standee poster)
    - `_generate_d13_dynamic_pardon_sidestep()` (Nhóm N5 & N6: Dynamic walking, sidestep, dense pairs)
  - [ ] Đảm bảo tính toán chính xác phép chiếu 8 đỉnh OBB và 12 tia Raycast đo che khuất $V_{\text{vis}}$.

- [ ] **Giai đoạn 3: Kết Xuất Dữ Liệu & Gán Nhãn Khách Quan (Data Generation)**
  - [ ] Chạy thực thi Godot 4 kết xuất tập dữ liệu mở rộng tại `sim_vision/data/vision_eval_extended/`.
  - [ ] Kiểm tra tính toàn vẹn của tập ảnh JPEG ($640 \times 480$) và file nhãn `annotations.jsonl`.
  - [ ] Xuất file `manifest_5w1h.json` ghi nhận đầy đủ siêu dữ liệu kỹ thuật.

- [ ] **Giai đoạn 4: Chạy Benchmark Đối Chứng CẢ 4 MÔ HÌNH (Multi-Model Evaluation)**
  - [ ] Chạy suy luận đối chứng 4 mô hình: `yolov8n.pt`, `yolo11n.pt`, `yolo26n.pt`, `person-detection-retail-0013.xml`.
  - [ ] Đánh giá các chỉ số cốt lõi:
    - Precision / Recall / F1-Score toàn diện.
    - Recall phân tách trên người ngồi ghế ($Recall_{\text{seated}}$) và người gập ho ($Recall_{\text{cough}}$).
    - Tỷ lệ báo động giả trên cột truyền dịch và standee in hình ($FPR_{\text{iv}}$, $FPR_{\text{standee}}$).
    - Độ ổn định góc lái bám mục tiêu ($\Delta\theta^\circ$).
    - Độ trễ suy luận P50/P95 (ms) so với chu kỳ điều khiển 20ms.

- [ ] **Giai đoạn 5: Trực Quan Hóa & Báo Cáo Khoa Học (Visualization & Reporting)**
  - [ ] Xuất báo cáo markdown hoàn chỉnh tại `sim_vision/reports/vision_extended_npc_benchmark.md`.
  - [ ] Nâng cấp Jupyter Notebook trực quan hóa các ca biên mới (Side-by-side grids cả 4 models trên các ca khó nhất).
  - [ ] Tạo Git commit chuẩn Conventional Commits kèm trailer `Companion-Doc:`.

---

## 3. Tiêu Chuẩn Đánh Giá Chỉ Số Đối Chứng (Evaluation Metrics & Pass Criteria)

1. **Kháng Báo Động Giả (False Alarm Immunity)**:
   - $FPR_{\text{iv\_stand}} = 0.00\%$ và $FPR_{\text{standee}} = 0.00\%$ trên nhóm Hard Negatives mới. Mô hình đạt chuẩn không được phép nhầm lẫn cột truyền dịch hay standee phẳng thành người.
2. **Độ Nhạy Người Ngồi Ghế (Seated Person Sensitivity)**:
   - $Recall_{\text{seated}} \ge 80.00\%$ khi bệnh nhân ngồi trên ghế chờ băng dài.
3. **Phân Tách Đám Đông Liền Kề (Dense Pair Separation)**:
   - Tỷ lệ tách đúng cả 2 đối tượng khi hai bệnh nhân ngồi sát nhau $\ge 80.00\%$ (Thử thách NMS-Free).
4. **Ngân Sách Điều Khiển 50Hz (Real-time Latency Budget)**:
   - Độ trễ $P95 \le 20.0\text{ ms}$ trên phần cứng mục tiêu để đảm bảo không làm trễ chu kỳ điều khiển robot.
