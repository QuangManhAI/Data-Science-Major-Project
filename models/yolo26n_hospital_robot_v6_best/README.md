# 🤖 YOLO26n Hospital Robot Perception Model (Production Release)

- **Tên mô hình chính thức**: `yolo26n_hospital_robot_best_map92.6`
- **Phiên bản**: v6 Golden Architecture (Official Final Release)
- **Kiến trúc**: YOLO26n Nano (2.39M parameters, 5.4 GFLOPs @ 640 / 8.5 GFLOPs @ 800)
- **Bộ dữ liệu huấn luyện**: 4,901 frames chuẩn vàng (3,715 Train, 655 Val, 884 Test Benchmark)
- **Kiểm toán bảo mật**: 100% Zero Data Leakage (SHA-256)

---

## 📊 Kết Quả Nghiệm Thu (884 Frames Benchmark - 3,082 Boxes CVAT)
- **Overall mAP@50**: **92.63%**
- **Overall Precision**: **95.13%**
- **Overall Recall**: **87.57%**
- **Tốc độ suy luận**: 5.38 ms (~186 FPS) trên GPU Laptop RTX 3050

---

## 📂 Danh Sách Tệp Trong Thư Mục Này
1. `yolo26n_hospital_robot_best_map92.6.pt`: Tệp trọng số gốc PyTorch (5.3 MB).
2. `yolo26n_hospital_robot_best_map92.6.onnx`: Tệp ONNX chuẩn (9.5 MB), dùng trực tiếp cho C++, Python, Godot, ROS2, TensorRT, OpenVINO.
3. `classes_17.json`: Danh sách 17 nhãn ID và Class Name.

---

## 🏷️ Danh Sách 17 Lớp Đối Tượng
```json
{
  "0": "patient",
  "1": "doctor",
  "2": "wheelchair",
  "3": "queue_kiosk",
  "4": "reception_counter",
  "5": "sanitizer_stand",
  "6": "pccc_station",
  "7": "medical_waste_bin",
  "8": "water_dispenser",
  "9": "wc_station",
  "10": "elevator_doors",
  "11": "hospital_bed",
  "12": "stretcher_cart",
  "13": "iv_stand",
  "14": "medical_trolley",
  "15": "vital_monitor",
  "16": "hospital_chair"
}
```

---

## 🚀 Hướng Dẫn Sử Dụng Nhanh (Python)
```python
from ultralytics import YOLO

# 1. Nạp mô hình PyTorch
model = YOLO("yolo26n_hospital_robot_best_map92.6.pt")

# 2. Hoặc nạp mô hình ONNX
# model = YOLO("yolo26n_hospital_robot_best_map92.6.onnx")

# 3. Dự đoán trên ảnh camera robot
results = model.predict(source="camera_frame.jpg", imgsz=800, conf=0.25)
for r in results:
    r.show()
```
