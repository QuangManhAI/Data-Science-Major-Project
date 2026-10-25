# 🏥 Dataset Card: `hospital_props_1600_13class`

Bộ dữ liệu thị giác nhận diện **13 nhóm Trang thiết bị & Mốc dịch vụ y tế cốt lõi** phục vụ Robot Trợ Lý Bệnh Viện (Autonomous Hospital Assistant Robot).

---

## 📌 Tổng Quan Bộ Dữ Liệu

- **Tên dataset**: `hospital_props_1600_13class`
- **Phiên bản**: `1.0_unified_flat` (Định dạng phẳng, chưa chia Train/Val/Test theo yêu cầu người dùng)
- **Quy mô**: **1,600 frames** (Độ phân giải $640 \times 480$)
- **Tổng số bounding box**: **2,310 instances** (trung bình 1.44 vật thể/frame)
- **Cơ chế gán nhãn**: Tự động 100% bằng thuật toán chiếu 3D Physics Mesh / Collision AABB kết hợp Raycast kiểm tra che khuất trong Godot 4.
- **Mục đích sử dụng**: Phục vụ Dẫn đường ngữ nghĩa (Semantic Wayfinding) và Tránh va chạm vật cản nhạy cảm (Safety Collision Avoidance).

---

## 🎯 Danh Mục 13 Lớp Đối Tượng (13 Classes)

| Class ID | Tên Lớp | File 3D Mesh | Vai Trò Lâm Sàng & Hành Vi Robot |
| :---: | :--- | :--- | :--- |
| **`0`** | `queue_kiosk` | `props/queue_kiosk.tscn` | Cây bấm số lấy phiếu tự động tại sảnh chờ (Dẫn bệnh nhân tới lấy số). |
| **`1`** | `reception_counter` | `props/reception_counter.tscn` | Quầy tiếp đón / lễ tân / thanh toán viện phí. |
| **`2`** | `sanitizer_stand` | `props/sanitizer_stand.tscn` | Trụ rửa tay sát khuẩn tự động tại sảnh và hành lang. |
| **`3`** | `pccc_station` | `props/pccc_station.tscn` | Trạm PCCC / Hộp bình chữa cháy gắn vách tường. |
| **`4`** | `medical_waste_bin` | `props/medical_waste_bin.tscn` | Thùng rác y tế màu vàng/đen tại các buồng bệnh. |
| **`5`** | `water_dispenser` | `props/water_dispenser.tscn` | Cây nước uống nóng lạnh phục vụ bệnh nhân tại sảnh. |
| **`6`** | `wc_station` | `props/wc_station.tscn` | Khu vực nhà vệ sinh bệnh viện (Chỉ dẫn khi bệnh nhân hỏi). |
| **`7`** | `elevator_doors` | `props/elevator_doors.tscn` | Cửa thang máy bệnh viện (Chỉ dẫn di chuyển liên tầng). |
| **`8`** | `hospital_bed` | `props/hospital_bed.tscn` | Giường bệnh viện trong phòng cấp cứu và phòng khám. |
| **`9`** | `stretcher_cart` | `props/stretcher_cart.tscn` | Cáng cáng cấp cứu gurney (Luật nhường đường ưu tiên). |
| **`10`** | `iv_stand` | `props/iv_stand.tscn` | Cọc truyền dịch Inox (Vật cản nhạy cảm cần né rộng vòng cung). |
| **`11`** | `medical_trolley` | `props/medical_trolley.tscn` | Xe tiêm thuốc inox 2 tầng của điều dưỡng (Vật cản di động). |
| **`12`** | `vital_monitor` | `props/vital_monitor.tscn` | Xe đẩy máy monitor đo sinh hiệu / nhịp tim / huyết áp. |

---

## 📁 Cấu Trúc Thư Mục (Flat Structure)

```text
hospital_props_1600_13class/
├── README.md                          # Thẻ mô tả bộ dữ liệu này
├── manifest.json                      # Metadata phiên bản và phân bố lớp
├── images/                            # 1,600 file ảnh JPG (P_0001.jpg -> P_1600.jpg)
└── labels/                            # 1,600 file nhãn TXT chuẩn YOLO (class_id xc yc w h)
```

---

## ⚙️ File Cấu Hình YOLO

- Đường dẫn file config: [`configs/hospital_props_13class.yaml`](file:///home/bush/Projects/Data-Science-Major-Project/configs/hospital_props_13class.yaml)
- Lệnh huấn luyện đề xuất:
  ```bash
  python sim_vision/python/train_hospital_vision.py \
    --model models/yolo26n/yolo26n.pt \
    --data configs/hospital_props_13class.yaml \
    --epochs 40 --batch 16 --name yolo26n_hospital_props_v1
  ```
