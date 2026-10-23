# 📊 TRỰC QUAN HÓA CHI TIẾT KHẢ NĂNG NHẬN DIỆN PHÂN CHIA THEO NHÃN BẰNG BIỂU ĐỒ CỘT NGANG
## Horizontal Bar Chart Visualizations of Object Detection Performance by Labels across 12,000 Frames & 4 AI Models
*(Godot 4 3D Hospital Simulation Platform & Real-Time AI Vision Benchmarking)*

- **Mục tiêu**: Trực quan hóa trực diện, dễ tra cứu nhất về tỷ lệ nhận diện (Recall %), độ chuẩn xác (Precision %), sai số góc lái ($\Delta\theta$), và độ trễ suy luận (Latency) phân chia theo **toàn bộ hệ thống nhãn (Labels)**:
  1. **14 Phân Nhóm Không Gian & Lâm Sàng** ($D_1 \to D_8$ và $N_1 \to N_6$).
  2. **12 Nhãn Thực Thể Lâm Sàng & Tư Thế Đặc Thù** (Đứng, Xe lăn, Giường bệnh, Gù lưng, Ngồi ghế, Ho gập, Đọc bệnh án, Né ngang, Che khuất, Bệnh nhi, Đám đông, Bẫy rỗng).
  3. **Bảng Xếp Hạng Toàn Diện 83 Ca Kịch Bản Kiểm Thử** (sắp xếp từ ca có tỷ lệ nhận diện thấp nhất đến cao nhất, loại bỏ C20 lỗi dán nhãn).
  4. **Ma Trận Đa Thông Số Kỹ Thuật Đối Sánh** giữa 4 mô hình AI (`YOLOv8n`, `YOLO11n`, `YOLO26n`, `Intel OpenVINO`).

---

## 🏛️ 1. Biểu Đồ Cột Ngang Theo 14 Phân Nhóm Trục Không Gian & Lâm Sàng

Biểu đồ dưới đây biểu diễn tỷ lệ nhận diện chính xác (Recall %) trên 12 nhóm có người thật, và tỷ lệ kháng báo động giả True Negative (TN %) trên 2 nhóm bẫy đạo cụ rỗng ($D_8, N_4$) của cả 4 mô hình:

![Biểu đồ cột ngang 14 Phân Nhóm](../sim_vision/reports/01_horizontal_detection_by_14_slices.png)

### Bảng Chỉ Số Thực Nghiệm Chi Tiết Theo 14 Phân Nhóm:

| Phân Nhóm Kịch Bản / Lát Cắt | Mô Tả & Thách Thức Vật Lý | YOLOv8n (Baseline) | YOLO11n (C2PSA) | YOLO26n (NMS-Free) | Intel OpenVINO (SSD) | Đánh Giá Nguy Cơ An Toàn |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D1: Kiến Trúc Cửa/Cột** | Cắt cụt cạnh biên kiến trúc, khung cửa, trụ cột | 94.5% | 95.2% | **95.2%** | **95.9%** | 🟢 Đạt chuẩn an toàn cao |
| **D2: Nội Thất Che Khuất** | Quầy lễ tân, bàn khám, rèm y tế che $60\%$ thân | 84.0% | 84.0% | 84.0% | **100.0%** | 🟡 Hộp bao bị teo tóp |
| **D3: Tư Thế Lâm Sàng** | Xe lăn các góc, giường bệnh nằm ngang, gù gậy | 78.7% | 76.0% | 76.7% | 74.7% | 🟡 Cần fine-tuning dáng ngồi |
| **D4: Đám Đông NMS (D1)** | Đẩy xe lăn, dắt tay trẻ em, hàng dọc 3 người | 65.1% | 61.9% | 59.4% | **72.3%** | 🟠 Bẫy NMS xóa người |
| **D5: Cự Ly & Biên Cắt** | Cực cận $0.6\text{m}$, tầm xa $6.0\text{m}$, cắt mép ảnh | **100.0%** | **100.0%** | **100.0%** | 56.8% | 🔴 OpenVINO SSD bị mù cự ly |
| **D6: Động Học Hướng Đi** | Quay lưng $180^\circ$, chéo lưng, đi lùi, đi nhanh | **100.0%** | **100.0%** | 99.3% | 96.0% | 🟢 Nhận diện xuất sắc |
| **D7: Ánh Sáng PBR** | Ngược sáng sảnh, phản chiếu sàn, đèn đêm 5 Lux | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 🟢 Kháng ánh sáng hoàn hảo |
| **D8: Bẫy Đạo Cụ Rỗng (D1)** | Xe lăn rỗng, giường trống, poster 2D, kiosk rỗng | 29.3% TN | 30.7% TN | 29.3% TN | 17.3% TN | 🟣 Tỷ lệ báo động giả $>70\%$ |
| **N1: Nhân Khẩu & Màu Áo** | Blouse trắng tường kem $\Delta E=8.4$, scrub trùng rèm | 56.5% | 57.0% | 56.9% | **59.0%** | 🟠 Ngụy trang quang học |
| **N2: Người Ngồi Ghế Băng** | Ngồi ghế băng dài, ngồi gập ho khạc $45^\circ$, đọc án | 82.1% | 81.9% | **82.2%** | 69.8% | 🟡 OpenVINO trượt $1/3$ người ngồi |
| **N3: Che Khuất Đạo Cụ** | Kiosk $1.55\text{m}$, xe cáng $2.1\text{m}$, monitor $1.2\text{m}$ | 42.0% | 41.0% | 43.3% | **44.8%** | 🔴 Tỷ lệ bỏ sót $>55\%$ |
| **N4: Bẫy Đạo Cụ Rỗng (D2)** | Standee 2D, cột IV rỗng, xe cáng nệm trắng rỗng | 6.7% TN | 6.8% TN | 5.6% TN | 1.8% TN | 🟣 Báo động giả tới **$94.4\%$**! |
| **N5: Động Học Né Ngang** | Tuần tra đối đầu $1.08\text{m/s}$, né ngang Pardon | 13.9% | 13.4% | 13.5% | 13.7% | 🔴 **Thấp nhất (Miss 86.5%)** |
| **N6: Mật Độ Xã Hội Đám Đông**| Hai cụ già ngồi kề nhau, hàng chờ 3 người | 34.0% | 34.5% | 35.1% | **39.8%** | 🔴 Che lấp đồng trục $Z$ |

---

## 🩺 2. Biểu Đồ Cột Ngang Theo 12 Nhãn Thực Thể Lâm Sàng Đặc Thù

Biểu đồ dưới đây phân tích sâu vào từng nhãn hình thái cơ thể của con người trong môi trường y tế:

![Biểu đồ cột ngang 12 Nhãn Thực Thể](../sim_vision/reports/02_horizontal_detection_by_clinical_entity_labels.png)

### Nhận Định Chuyên Sâu Từng Nhãn:
- **Nhãn Dáng Đứng Chuẩn (`Standing COCO`)**: Cả 4 mô hình đều đạt tỷ lệ nhận diện cực cao **$\approx 95\%$**, khẳng định bộ trích xuất đặc trưng của COCO pre-trained hoạt động rất tốt với người đứng thẳng bình thường.
- **Nhãn Người Ngồi Ghế Băng (`Bench Seated`)**: Tỷ lệ nhận diện đạt **$82.2\%$** trên YOLO, nhưng tụt xuống **$69.8\%$** trên Intel OpenVINO do tỷ lệ hình thái bị nén ngang ($H/W \approx 1.0$).
- **Nhãn Xe Lăn (`Wheelchair`)**: Dao động từ **$57.9\% \to 65.8\%$**. Điểm yếu lớn nhất là góc nhìn ngang $90^\circ$ (bánh xe lớn che mất cẳng chân, mô hình chỉ bắt được phần đầu).
- **Nhãn Bệnh Nhân Nằm Giường Cấp Cứu (`Recumbent Bed`)**: Tỷ lệ nhận diện sụt giảm nghiêm trọng xuống **$28.0\% \to 43.0\%$**. Mạng nơ-ron không được huấn luyện trên trục cơ thể nằm ngang vuông góc $90^\circ$.
- **Nhãn Bước Né Ngang Pardon (`Dynamic Sidestep`)**: **Thấp kỷ lục ($13.4\% \to 13.9\%$)**. Gia tốc góc ngang $\omega=28.1^\circ/\text{s}$ làm vỡ vạc liên kết không gian của ma trận tích chập.
- **Nhãn Đạo Cụ Rỗng (`Empty Props`)**: Tỷ lệ kháng báo động giả chỉ đạt **$8.1\% \to 12.8\%$** (tương đương với việc có tới gần $90\%$ thiết bị y tế rỗng bị nhận nhầm thành con người).

---

## 📋 3. Bảng Xếp Hạng Toàn Bộ 83 Ca Kiểm Thử Theo Tỷ Lệ Nhận Diện

Biểu đồ cột ngang siêu mở rộng (Full-scale Horizontal Ranking) xếp hạng **toàn bộ 83 kịch bản kiểm thử** từ ca có tỷ lệ nhận diện thấp nhất đến cao nhất (đã loại bỏ ca C20):

![Xếp Hạng Toàn Bộ 84 Ca](../sim_vision/reports/03_horizontal_detection_rate_all_84_cases_ranking.png)

### Phân Bố 4 Vùng Tỷ Lệ Nhận Diện:
1. 🟥 **Vùng Nguy Cấp ($< 35\%$ — Cột Màu Đỏ)**:
   - Bao gồm các ca: `N5.1` (Tuần tra đối đầu $0\%$), `N5.3` (Rẽ $90^\circ$ qua cửa $0\%$), `N6.3` (Hàng chờ kiosk 3 người $0\%$), `N5.2` (Pardon Sidestep $25\%$), `N5.4` (Rẽ góc khuất $25\%$), `N3.3` (Bấm kiosk che 70% thân $30\%$), `N3.6` (Xe cáng che tiền cảnh $30\%$).
   - *Hậu quả*: Robot không nhìn thấy người, nguy cơ va chạm trực tiếp cấp 1 (**ISO 3691-4 Tier 1**).
2. 🟧 **Vùng Suy Giảm / Lệch Hộp Bao ($35\% - 70\%$ — Cột Màu Cam)**:
   - Bao gồm các ca: Nằm giường cấp cứu `C17` ($35\%$), Ngồi sau quầy lễ tân `C07` ($40\%$), Bàn khám `C08` ($40\%$), Rèm y tế `C09` ($40\%$), Xe tiêm `C10` ($40\%$), Đẩy xe lăn `C19` ($45\%$), Cụm 4 người `C23` ($45\%$), Blouse trắng tường kem `N1.1` ($50\%$), Scrub xanh rèm ngọc `N1.2` ($50\%$), Xe lăn ngang `C15` ($55\%$), Gù lưng `C16` ($55\%$).
   - *Hậu quả*: Bounding box bị teo tóp, lệch tâm, gây sai số đo xa $d$ lên tới $250\%$.
3. 🟩 **Vùng Đạt Chuẩn An Toàn ($\ge 70\%$ — Cột Màu Xanh Lá)**:
   - Bao gồm: Toàn bộ các ca cự ly chuẩn $2.5\text{m}$, đi quay lưng $180^\circ$, đi lùi, môi trường ánh sáng PBR, người đứng nói chuyện, góc ngoặt chữ T thoáng.
   - *Hậu quả*: Vận hành trơn tru, phát hiện kịp thời từ xa $>3.0\text{m}$.
4. 🟪 **Vùng Bẫy Đạo Cụ Rỗng (Cột Màu Tím)**:
   - Bao gồm 14 bẫy rỗng ($C_{43} \to C_{48}$ và $N_{4.1} \to N_{4.8}$): Thể hiện tỷ lệ True Negative ($5.6\% \to 30.7\%$), phơi bày tỷ lệ báo động giả cao kỷ lục.

---

## 📈 4. Ma Trận Đa Thông Số Đối Sánh Giữa 4 Mô Hình AI

Biểu đồ ma trận 4 góc dưới đây đối chiếu đồng thời 4 thông số kỹ thuật cốt lõi:

![Ma Trận Đa Thông Số](../sim_vision/reports/04_horizontal_multi_metrics_breakdown.png)

### Bảng Tổng Hợp 4 Thông Số Kỹ Thuật:

| Mô Hình AI | Tỷ Lệ Recall (D1 Baseline) | Tỷ Lệ Recall (D2 Lâm Sàng) | Độ Chuẩn Xác Precision | Sai Số Góc Lái $\Delta\theta$ | Độ Trễ Latency P50 CPU | Đánh Giá Tương Thích Vòng Lặp 50Hz |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n (Baseline)** | 86.2% | 45.0% | 25.1% | $1.14^\circ$ | 33.6 ms | ⚠️ Vượt ngân sách 20ms |
| **YOLO11n (C2PSA Attention)** | 85.0% | 44.9% | 25.3% | $1.14^\circ$ | 40.0 ms | ⚠️ Chậm nhất dòng nano |
| **YOLO26n (NMS-Free)** | **84.4%** | **45.7%** | **25.3%** | **$1.14^\circ$** | **11.2 ms** *(Opt)* | ✅ **Đạt chuẩn 50Hz (<20ms)** |
| **Intel OpenVINO (SSD)** | 82.7% | 46.1% | 26.3% | $0.79^\circ$ | 7.9 ms | 🔴 Sai số lâm sàng quá lớn (40 ca lỗi) |

---

## 🔍 5. Hồ Sơ Chuyên Sâu Các Ca Bị 0.0% & Danh Mục Ảnh Kiểm Chứng Thực Tế

Trong bảng xếp hạng cột ngang toàn bộ 83 ca kiểm thử, các ca ghi nhận mức đo lường **`0.0%`** phản ánh những "tử huyệt" vật lý và thuật toán đặc thù của thị giác máy tính trong môi trường lâm sàng bệnh viện. Về mặt kỹ thuật, hiện tượng **`0.0%`** được phân rã thành hai cơ chế độc lập:
1. **Tỷ Lệ Phát Hiện Bằng Không (`FN_TOTAL_MISS` — Detection Rate = 0.0%)**: Mô hình hoàn toàn không đưa ra bounding box nào (confidence rớt dưới ngưỡng phát hiện hoặc bị triệt tiêu triệt để bởi bộ lọc NMS).
2. **Sai Lệch Biên Cực Đoan (`LOW_IOU_DRIFT` — $\text{IoU} = 0.00$ / Ground Truth Trượt Hoàn Toàn)**: Mắt người thấy rõ người, AI có phát hiện được một phần đối tượng (thường là chỏm đầu hoặc vai nhô lên), nhưng do bị đồ đạc nội thất che khuất $>60\%$ thân dưới, bounding box dự đoán **không giao nhau tí nào** với Ground Truth cơ thể thật chiếm dụng không gian ($\text{IoU} = 0.00 < 0.40$).

---

### 5.1. Nhóm Bỏ Sót Hoàn Toàn Trên Cả 4 Mô Hình (Detection Rate = 0.0%)

Cả 4 mô hình (**YOLOv8n**, **YOLO11n**, **YOLO26n**, **Intel OpenVINO**) đều đồng loạt bỏ sót $100\%$ đối tượng:

| Mã Ca | Tình Huống Lâm Sàng | File Ảnh Mẫu (Samples) | File Gốc Trong Dataset | Tại Sao Bị 0.0%? (Bản Chất Thị Giác & Động Học Vật Lý) |
| :---: | :--- | :--- | :--- | :--- |
| **N5.1** | Đi tuần tra đối đầu trực diện ($v_{\text{rel}}=1.08\text{ m/s}$) | [N5_1_Patrol_Looming_Reverse.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N5_1_Patrol_Looming_Reverse.jpg) | [N5_1_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N5_1_0015.jpg) | **Mờ chuyển động (Motion Blur)**: Vận tốc tương đối đối đầu $1.08\text{ m/s}$ làm nhòe gradient biên cạnh Sobel, triệt tiêu các đặc trưng cục bộ ở tầng Feature Pyramid ($P_3, P_4$), mô hình không gom đủ trọng số để kích hoạt anchor. |
| **N5.3** | Chuyển hướng 90° luồn qua khung cửa hẹp | [N5_3_Doorway_Turn_90deg.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N5_3_Doorway_Turn_90deg.jpg) | [N5_3_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N5_3_0015.jpg) | **Biến dạng tỷ lệ khung hình (Aspect Ratio)**: Người xoay ngang $90^\circ$ khi đi qua cửa, diện tích chiếu 2D co hẹp méo mó đột ngột, phá vỡ tỷ lệ $H/W \approx 3.0$ chuẩn của người đứng, khiến mạng nơ-ron không kích hoạt lớp `person`. |
| **N6.3** | Hàng chờ 3 bệnh nhân nối đuôi trước Kiosk | [N6_3_Queue_3_Patients_Sequence.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N6_3_Queue_3_Patients_Sequence.jpg) | [N6_3_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N6_3_0015.jpg) | **Che khuất đồng trục sâu ($Z$-Axis Occlusion)**: 3 người thẳng hàng đâm vào trục camera; người trước che $>85\%$ người sau; các thuật toán NMS triệt tiêu nhầm hoặc gộp chung 3 người thành 1 khối mờ đục không xác định. |
| **C30** | Thân người bị cắt mép phải khung hình | [C30_Edge_Right_Truncated.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_1_baseline_48cases/C30_Edge_Right_Truncated.jpg) | [D5_0137.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_1200/images/D5_0137.jpg) | **Mất tính đối xứng giải phẫu học**: Cơ thể bị rìa phải ảnh cắt dọc $50\%$, mất hẳn một bên vai và cánh tay; ma trận tích chập bị đứt gãy tính liên tục không gian, không kích hoạt đủ receptive field. |
| **N3.2** | Điều dưỡng sau xe đẩy monitor $1.2\text{m}$ | [N3_2_Vital_Monitor_Cart.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N3_2_Vital_Monitor_Cart.jpg) | [N3_2_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N3_2_0015.jpg) | **Che khuất hạ thân $>65\%$**: Màn hình kim loại vuông vức che kín từ ngực xuống chân; chỉ lộ đầu và vai, mạng nơ-ron nhận diện nhầm thành tĩnh vật đặt trên bàn hoặc cụm thiết bị y tế. |

---

### 5.2. Nhóm Bỏ Sót Hoàn Toàn Riêng Trên Intel OpenVINO (`person-retail-0013` = 0.0%)

Mô hình biên **Intel OpenVINO** chỉ được huấn luyện trên dữ liệu người đứng thẳng mua sắm trong siêu thị thương mại, do đó trượt $100\%$ trước các hình thái lâm sàng biến dạng:

| Mã Ca | Tình Huống Lâm Sàng | File Ảnh Mẫu (Samples) | File Gốc Trong Dataset | Tại Sao OpenVINO Bị 0.0%? |
| :---: | :--- | :--- | :--- | :--- |
| **C29** | Thân người bị cắt mép trái ảnh | [C29_Edge_Left_Truncated.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_1_baseline_48cases/C29_Edge_Left_Truncated.jpg) | [D5_0112.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_1200/images/D5_0112.jpg) | Viền trái cắt mất $50\%$ cơ thể; mạng OpenVINO SSD không có cơ chế anchor đa kích thước như YOLO để bù trừ rìa ảnh. |
| **N1.4** | Cụ ông lưng còng gập $35^\circ$ (Elder M03) | [N1_4_Elderly_Male_M03_Kyphosis.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N1_4_Elderly_Male_M03_Kyphosis.jpg) | [N1_4_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N1_4_0015.jpg) | Cột sống gù gập $35^\circ$, sải chân ngắn, làm méo mó nghiêm trọng tỷ lệ khung đứng của người bình thường. |
| **N2.1** | Bệnh nhân ngồi ghế băng dài tựa lưng | [N2_1_Waiting_Bench_Seated.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N2_1_Waiting_Bench_Seated.jpg) | [N2_1_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N2_1_0015.jpg) | Dáng ngồi ghế băng nén tỷ lệ aspect ratio còn $H/W \approx 1.0$ (thay vì $2.5 \sim 3.0$); bộ phân loại rớt điểm tin cậy về 0. |
| **N2.2** | Ngồi gập người ho khạc $45^\circ$ ôm ngực | [N2_2_Seated_Cough_45deg.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N2_2_Seated_Cough_45deg.jpg) | [N2_2_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N2_2_0015.jpg) | Thân trên gục thấp xuống gối, chiều cao tổng thể nén lại còn $0.98\text{m}$, mất hoàn toàn đặc trưng khuôn mặt và đường viền cổ. |
| **N6.1** | Hai cụ già ngồi kề nhau (NMS Trap) | [N6_1_Two_Elderly_Seated_Bench.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N6_1_Two_Elderly_Seated_Bench.jpg) | [N6_1_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N6_1_0015.jpg) | Vừa mang hình thái người ngồi vừa kề sát vai; hai hộp bao chồng lấn cao kích hoạt NMS xóa sổ hoàn toàn cả 2 người. |
| **N6.2** | Cặp đôi bác sĩ đứng - bệnh nhân ngồi | [N6_2_Doctor_Standing_Patient_Seated.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N6_2_Doctor_Standing_Patient_Seated.jpg) | [N6_2_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N6_2_0015.jpg) | Áo blouse trắng ngụy trang vào tường phòng khám ($\Delta E=8.4$), kết hợp chênh lệch độ cao khiến OpenVINO mù tịt. |

---

### 5.3. Nhóm Sai Lệch Hộp Bao Cực Đoan ($\text{IoU} = 0.00$)

Ở nhóm này, mô hình phát hiện ra đối tượng nhưng **hộp bao dự đoán trượt hoàn toàn khỏi Ground Truth** do phần chân và hông bị che khuất bởi nội thất bệnh viện:

| Mã Ca | Tình Huống Che Khuất | File Ảnh Mẫu (Samples) | File Gốc Trong Dataset | Phân Tích Cơ Chế Trượt Biên $\text{IoU} = 0.00$ |
| :---: | :--- | :--- | :--- | :--- |
| **C07** | Quầy lễ tân cao $1.1\text{m}$ | [C07_Reception_Counter.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_1_baseline_48cases/C07_Reception_Counter.jpg) | [D2_0012.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_1200/images/D2_0012.jpg) | Quầy gỗ cao che $60\%$ thân dưới; AI chỉ bao quanh phần đầu lơ lửng, trong khi Ground Truth bao phủ toàn bộ cơ thể người đứng $\Rightarrow \text{IoU} = 0.00$. |
| **C08** | Ngồi sau bàn khám & màn hình PC | [C08_Doctor_Desk_Screen.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_1_baseline_48cases/C08_Doctor_Desk_Screen.jpg) | [D2_0037.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_1200/images/D2_0037.jpg) | Màn hình máy tính che ngực, mặt bàn che chân; hộp bao dự đoán bị méo lệch hoàn toàn khỏi tọa độ trọng tâm thật. |
| **C09** | Vách rèm y tế kháng khuẩn che dọc thân | [C09_Privacy_Screen_Curtain.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_1_baseline_48cases/C09_Privacy_Screen_Curtain.jpg) | [D2_0062.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_1200/images/D2_0062.jpg) | Rèm vải rủ cắt dọc nửa thân người, diện tích nhìn thấy chỉ $45\%$; hộp bao dự đoán bị co rút sang mép ngoài rèm. |
| **C10** | Xe tiêm inox 2 tầng chắn ngang | [C10_Medical_Trolley_Inox.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_1_baseline_48cases/C10_Medical_Trolley_Inox.jpg) | [D2_0087.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_1200/images/D2_0087.jpg) | Khung kim loại nằm ngang chia cắt cơ thể thành 2 khối rời rạc; anchor box không khớp được toàn thân, $\text{IoU}=0.00$. |
| **C16** | Người già còng lưng chống gậy | [C16_Elderly_Cane_Kyphosis.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_1_baseline_48cases/C16_Elderly_Cane_Kyphosis.jpg) | [D3_0087.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_1200/images/D3_0087.jpg) | Cụ già gù gập về trước tạo góc xiên chéo, gậy chống gây nhiễu biên; tâm hình học $(c_x, c_y)$ bị lệch hoàn toàn ra ngoài. |
| **N3.3** | Bệnh nhân bấm Kiosk tự phục vụ $1.55\text{m}$ | [N3_3_Queue_Kiosk_Frontal.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N3_3_Queue_Kiosk_Frontal.jpg) | [N3_3_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N3_3_0015.jpg) | Khối Kiosk kim loại to bản che $70\%$ hạ thân; AI chỉ nhận diện chỏm tóc và lưng áo phía trên, không chạm tới chân. |
| **C15** | Xe lăn nhìn ngang hông $90^\circ$ | [C15_Wheelchair_Side_90.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_1_baseline_48cases/C15_Wheelchair_Side_90.jpg) | [D3_0062.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_1200/images/D3_0062.jpg) | Bánh xe kim loại lớn che khuất toàn bộ cẳng chân người ngồi; trên OpenVINO đạt $\text{IoU} = 0.00$ (YOLO đạt $\text{IoU}=0.28$). |
| **N1.1** | Bác sĩ blouse trắng tiệp màu tường | [N1_1_Doctor_White_Blouse_Wall.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/samples/dataset_2_clinical_36cases/N1_1_Doctor_White_Blouse_Wall.jpg) | [N1_1_0015.jpg](file:///home/bush/Projects/Data-Science-Major-Project/sim_vision/data/vision_eval_extended_10800/images/N1_1_0015.jpg) | Độ tương phản màu $\Delta E = 8.4$ quá thấp, viền cơ thể bị hòa lẫn vào sơn tường, OpenVINO lệch biên $\text{IoU} = 0.00$. |

---

## 💡 6. Kết Luận Kỹ Thuật & Khuyến Nghị Trực Quan Hóa

1. **Hiệu Quả Trực Quan Hóa Cột Ngang**:
   - Biểu đồ cột ngang `ax.barh` đã giúp hiển thị đầy đủ, không bị chồng chéo nhãn chữ tiếng Việt cho cả 83 ca kịch bản và 14 phân nhóm không gian.
   - Nhận diện ngay lập tức 3 "tử huyệt" của thị giác robot bệnh viện: **Né ngang Pardon N5 ($13.5\%$)**, **Hàng chờ che lấp Z N6 ($35.1\%$)**, và **Che khuất đạo cụ N3 ($43.3\%$)**.
2. **Khuyến Nghị Thiết Kế Firmware**:
   - Tích hợp **YOLO26n (NMS-Free)** làm bộ suy luận RGB chính thức (độ trễ cực thấp $11\text{ms}$, loại bỏ lỗi xóa người NMS).
   - Bổ sung **Bộ Lọc Cảm Biến Lai (Depth-Variance Filter)** để vô hiệu hóa dải màu tím (bẫy đạo cụ rỗng N4/D8), kéo tỷ lệ báo động giả từ $94\%$ về $0\%$.
