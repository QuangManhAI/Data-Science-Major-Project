# 🏥 Dataset Card: `hospital_props_1600_15class`

Bộ dữ liệu thị giác nhận diện **15 nhóm Trang thiết bị, Ghế chờ, Xe lăn & Mốc dịch vụ y tế cốt lõi** phục vụ Robot Trợ Lý Bệnh Viện (Autonomous Hospital Assistant Robot).
Toàn bộ 1,600 frames đã được **thẩm định, chỉnh sửa và bổ sung thủ công 100% trên CVAT**.

---

## 📌 Tổng Quan Bộ Dữ Liệu

- **Tên dataset**: `hospital_props_1600_15class`
- **Phiên bản**: `2.0_human_verified_cvat` (Đã qua kiểm duyệt và bổ sung nhãn tay trên CVAT)
- **Quy mô**: **1,600 frames** (Độ phân giải $640 \times 480$)
- **Tổng số bounding box**: **3,496 instances** (trung bình 2.19 vật thể/frame)
- **Cơ chế gán nhãn**: Khởi tạo tự động bằng 3D Physics Mesh Projection trong Godot 4 + Thẩm định, nắn chỉnh và gắn thêm vật thể thực tế bởi con người trên công cụ CVAT.
- **Mục đích sử dụng**: Phục vụ Dẫn đường ngữ nghĩa (Semantic Wayfinding) và Tránh va chạm vật cản nhạy cảm (Safety Collision Avoidance).

---

## 🎯 Danh Mục 15 Lớp Đối Tượng (15 Classes)

| Class ID | Tên Lớp | Số lượng Boxes | Vai Trò Lâm Sàng & Hành Vi Robot |
| :---: | :--- | :---: | :--- |
| **`0`** | `queue_kiosk` | **204** | Cây bấm số lấy phiếu tự động tại sảnh chờ (Dẫn bệnh nhân tới lấy số). |
| **`1`** | `reception_counter` | **313** | Quầy tiếp đón / lễ tân / thanh toán viện phí. |
| **`2`** | `sanitizer_stand` | **277** | Trụ rửa tay sát khuẩn tự động tại sảnh và hành lang. |
| **`3`** | `pccc_station` | **91** | Trạm PCCC / Hộp bình chữa cháy gắn vách tường. |
| **`4`** | `medical_waste_bin` | **397** | Thùng rác y tế màu vàng/đen tại các buồng bệnh. |
| **`5`** | `water_dispenser` | **139** | Cây nước uống nóng lạnh phục vụ bệnh nhân tại sảnh. |
| **`6`** | `wc_station` | **76** | Khu vực nhà vệ sinh bệnh viện (Chỉ dẫn khi bệnh nhân hỏi). |
| **`7`** | `elevator_doors` | **72** | Cửa thang máy bệnh viện (Chỉ dẫn di chuyển liên tầng). |
| **`8`** | `hospital_bed` | **430** | Giường bệnh viện trong phòng cấp cứu và phòng khám. |
| **`9`** | `stretcher_cart` | **34** | Cáng cáng cấp cứu gurney (Luật nhường đường ưu tiên). |
| **`10`** | `iv_stand` | **181** | Cọc truyền dịch Inox (Vật cản nhạy cảm cần né rộng vòng cung). |
| **`11`** | `medical_trolley` | **345** | Xe tiêm thuốc inox 2 tầng của điều dưỡng (Vật cản di động). |
| **`12`** | `vital_monitor` | **251** | Xe đẩy máy monitor đo sinh hiệu / nhịp tim / huyết áp. |
| **`13`** | `wheelchair` | **275** | Xe lăn người bệnh (Vật thể di động cần nhận diện tránh va quẹt). |
| **`14`** | `hospital_chair` | **411** | Ghế ngồi chờ / dãy ghế phòng khám (Hỗ trợ tìm chỗ ngồi cho người mệt). |

---

## 📁 Cấu Trúc Thư Mục (Flat Structure)

```text
hospital_props_1600_15class/
├── README.md                          # Thẻ mô tả bộ dữ liệu này
├── manifest.json                      # Metadata phiên bản và phân bố lớp chi tiết
├── images/                            # 1,600 file ảnh JPG (P_0001.jpg -> P_1600.jpg)
└── labels/                            # 1,600 file nhãn TXT chuẩn YOLO (class_id xc yc w h)
```

---

## ⚙️ File Cấu Hình YOLO

- Đường dẫn file config: [`configs/hospital_props_15class.yaml`](file:///home/bush/Projects/Data-Science-Major-Project/configs/hospital_props_15class.yaml)
- Lệnh huấn luyện đề xuất:
  ```bash
  python sim_vision/python/train_hospital_vision.py \
    --model models/yolo26n/yolo26n.pt \
    --data configs/hospital_props_15class.yaml \
    --epochs 50 --batch 16 --name yolo26n_hospital_props_15class_v1
  ```
