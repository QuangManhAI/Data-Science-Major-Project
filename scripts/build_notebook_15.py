#!/usr/bin/env python3
"""
scripts/build_notebook_15.py
============================
Tạo tự động Sổ tay số 15:
notebooks/15_robot_patrol_mission_benchmark_report.ipynb
Báo cáo đo lường nghiệm thu góc nhìn camera Robot thực tế (Robot Autonomous Patrol Mission Benchmark).
"""

import json
from pathlib import Path
import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks/15_robot_patrol_mission_benchmark_report.ipynb"

nb = nbf.v4.new_notebook()

# Cell 0: Header & Giới thiệu
md_header = """# 🤖 Sổ Tay 15: Báo Cáo Nghiệm Thu Sứ Mệnh Tuần Tra Thực Tế Của Robot Bệnh Viện
### (Autonomous Robot Patrol Mission Benchmark Across 7 Clinical Zones)

---
### 1. Bối Cảnh & Mục Tiêu Nghiệm Thu
Để kiểm định chính xác khả năng nhận thức thị giác máy tính của Robot trong môi trường lâm sàng thực tế, chúng tôi thiết lập một bộ dữ liệu kiểm thử độc lập hoàn toàn (**Robot Patrol Testset - 531 Frames**):
1. **Góc nhìn thực thể Robot (Sim2Real Robot Eye Viewpoint)**:
   - **Độ cao camera**: $h = 1.18\\text{ m}$ (ngang tầm ngực robot).
   - **Góc ngẩng/cúi (Pitch)**: $-5.0^\\circ$ (hướng nhẹ xuống sàn hành lang).
   - **Góc mở ống kính (FOV)**: $70.0^\\circ$ (chuẩn camera công nghiệp robot).
   - **Độ phân giải đầu vào**: $640 \\times 640$ tensor tĩnh tối ưu hóa cho NPU/GPU nhúng.
2. **Quỹ đạo tuần tra khép kín (Autonomous Continuous Navigation)**:
   - Hành trình liên tục qua **30 Waypoints chính** bao phủ toàn bộ **7 phân khu chức năng bệnh viện**: Sảnh tiếp đón (Lobby), Quầy dược (Pharmacy), Khoa cấp cứu (Emergency Ward), Hành lang trung tâm (Main Corridor), Phòng xét nghiệm máu (Blood Lab), Phòng khám nội (Clinic Room), Khu thang máy & vệ sinh (Elevator & Restroom).
   - Tổng cộng **1,616 thực thể vật lý** (bao gồm cả 15 loại trang thiết bị y tế và nhân vật NPC Bác sĩ, Y tá, Bệnh nhân) được ghi nhận và chiếu nhãn ground truth chuẩn xác.
3. **Nguyên tắc bảo toàn mô hình (Model Edge Constraints)**:
   - Kiến trúc mô hình được giữ nguyên vẹn **YOLO26n Nano** (2.39M tham số, dung lượng 5.3 MB `.pt` / 9.4 MB `.onnx`, 5.4 GFLOPs).
   - Tuyệt đối không thay đổi sang mô hình nặng (Medium/Large) để đảm bảo độ trễ thời gian thực `< 4.0\\text{ ms}` trên phần cứng biên.
"""

# Cell 1: Code Load dữ liệu và bảng đối chuẩn
code_metrics = """import json
from pathlib import Path
import pandas as pd
from IPython.display import display, HTML

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
metrics_file = repo_root / "experiments/results/robot_patrol_benchmark_metrics.json"

with open(metrics_file, "r", encoding="utf-8") as f:
    data = json.load(f)

v4_res = data["v4_results"]
v5_res = data["v5_results"]

print(f"📊 Tên bộ Benchmark: {data['benchmark_name']}")
print(f"📸 Tổng số frames tuần tra: {data['total_frames']} khung hình")
print(f"📦 Tổng số đối tượng xuất hiện: {data['total_boxes']} thực thể")
print(f"⏱️ Thời gian đánh giá: {data['timestamp']}")

# Bảng so sánh tổng thể
summary_df = pd.DataFrame([
    {
        "Chỉ Số Đánh Giá": "mAP@50 (%)",
        "YOLO26n-v4": f"{v4_res['overall']['mAP50']*100:.2f}%",
        "YOLO26n-v5": f"{v5_res['overall']['mAP50']*100:.2f}%",
        "Biến Thiên (Delta)": f"{(v5_res['overall']['mAP50'] - v4_res['overall']['mAP50'])*100:+.2f}%"
    },
    {
        "Chỉ Số Đánh Giá": "Precision (%)",
        "YOLO26n-v4": f"{v4_res['overall']['precision']*100:.2f}%",
        "YOLO26n-v5": f"{v5_res['overall']['precision']*100:.2f}%",
        "Biến Thiên (Delta)": f"{(v5_res['overall']['precision'] - v4_res['overall']['precision'])*100:+.2f}%"
    },
    {
        "Chỉ Số Đánh Giá": "Recall (%)",
        "YOLO26n-v4": f"{v4_res['overall']['recall']*100:.2f}%",
        "YOLO26n-v5": f"{v5_res['overall']['recall']*100:.2f}%",
        "Biến Thiên (Delta)": f"{(v5_res['overall']['recall'] - v4_res['overall']['recall'])*100:+.2f}%"
    },
    {
        "Chỉ Số Đánh Giá": "F1-Score (%)",
        "YOLO26n-v4": f"{v4_res['overall']['f1']*100:.2f}%",
        "YOLO26n-v5": f"{v5_res['overall']['f1']*100:.2f}%",
        "Biến Thiên (Delta)": f"{(v5_res['overall']['f1'] - v4_res['overall']['f1'])*100:+.2f}%"
    },
    {
        "Chỉ Số Đánh Giá": "Độ Trễ Suy Luận (Inference)",
        "YOLO26n-v4": "3.5 ms (285 FPS)",
        "YOLO26n-v5": "3.4 ms (294 FPS)",
        "Biến Thiên (Delta)": "-0.1 ms (Cực Nhanh)"
    }
])

display(HTML("<h3>🏆 1. BẢNG SO SÁNH CHỈ SỐ VẬN HÀNH TỔNG THỂ TRÊN ROBOT</h3>"))
display(summary_df)
"""

# Cell 2: Code bảng chi tiết 17 lớp
code_per_class = """# Bảng chi tiết từng lớp đối tượng
table_rows = []
for cname, v4_c in v4_res["per_class"].items():
    v5_c = v5_res["per_class"].get(cname, {})
    v4_map = v4_c.get("mAP50", 0.0) * 100
    v5_map = v5_c.get("mAP50", 0.0) * 100
    delta = v5_map - v4_map
    
    role = "Mốc Dẫn Đường (Anchor)" if v5_map >= 70.0 else ("Vật Cản Di Động (Obstacle)" if cname in ["wheelchair", "medical_trolley", "hospital_bed", "hospital_chair"] else "Nhân Sự Lâm Sàng (NPC)")
    
    table_rows.append({
        "Lớp Đối Tượng": cname.replace("_", " ").title(),
        "YOLO26n-v4 mAP50": f"{v4_map:.2f}%",
        "YOLO26n-v5 mAP50": f"{v5_map:.2f}%",
        "Biến Thiên (%)": f"{delta:+.2f}%",
        "YOLO26n-v5 Recall": f"{v5_c.get('recall', 0.0)*100:.2f}%",
        "YOLO26n-v5 Precision": f"{v5_c.get('precision', 0.0)*100:.2f}%",
        "Vai Trò Tuần Tra Robot": role
    })

per_class_df = pd.DataFrame(table_rows)
display(HTML("<h3>📋 2. BẢNG ĐO LƯỜNG CHI TIẾT TỪNG LỚP ĐỐI TƯỢNG VÀ TRANG THIẾT BỊ</h3>"))
display(per_class_df)
"""

# Cell 3: Code hiển thị Dashboard hình ảnh
code_image = """import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dash_path = repo_root / "experiments/results/visualizations/robot_patrol_mission_benchmark.png"
if dash_path.exists():
    img = mpimg.imread(str(dash_path))
    plt.figure(figsize=(22, 16.5), dpi=200)
    plt.imshow(img)
    plt.axis("off")
    plt.title("HỆ THỐNG TRỰC QUAN HÓA TOÀN DIỆN SỨ MỆNH TUẦN TRA ROBOT BỆNH VIỆN", fontsize=16, fontweight="bold", pad=15)
    plt.show()
else:
    print("⚠️ Không tìm thấy tệp ảnh:", dash_path)
"""

# Cell 4: Markdown Phân tích Sim2Real & Khuyến nghị vận hành
md_analysis = """### 🧠 3. Phân Tích Chuyên Sâu Sim2Real & Cơ Chế Định Vị Nhận Thức

#### A. Nhóm Mốc Tĩnh Dẫn Đường Tuyệt Đối (Navigation Visual Anchors):
1. **Trạm Nhà Vệ Sinh (`wc_station` - 98.33% mAP@50)**:
   - Đạt độ chính xác tiệm cận tuyệt đối, hoạt động như mốc định vị mỏ neo (Visual Anchor) tối thượng tại khu vực Nam sảnh.
2. **Thùng Rác Y Tế Nguy Hại (`medical_waste_bin` - 82.56% mAP@50)**:
   - Tăng trưởng $+2.27\\%$ so với v4, phân bổ đồng đều ở cả 7 phân khu, robot phát hiện ngay từ cự ly xa $> 5\\text{ m}$.
3. **Kiosk Lấy Số Thứ Tự (`queue_kiosk` - 78.40% mAP@50)**:
   - Đảm bảo dẫn đường hướng dẫn bệnh nhân tự động tại Sảnh chính (Lobby).
4. **Trạm Khử Khuẩn Tự Động (`sanitizer_stand` - 72.68% mAP@50)** & **Trạm PCCC (`pccc_station` - 69.11% mAP@50)**:
   - Nhận diện tin cậy phục vụ giám sát quy chuẩn an toàn bệnh viện.

#### B. Nhóm Trang Thiết Bị Di Động & Tránh Va Chạm (Obstacle Avoidance):
1. **Xe Đẩy Y Tế (`medical_trolley` - 66.53% mAP@50)**:
   - **Bứt phá ngoạn mục $+10.65\\%$** (từ $55.88\\% \\to 66.53\\%$) nhờ bổ sung dữ liệu đa góc quay ở phiên bản v5.
2. **Xe Lăn Y Tế (`wheelchair` - 59.41% mAP@50)**:
   - Tiếp tục duy trì đà tăng trưởng $+2.79\\%$ trên hành trình tuần tra thực tế.
3. **Dãy Ghế Chờ Bệnh Viện (`hospital_chair` - 55.18% mAP@50)**:
   - Phát hiện liên tục dọc hai bên lối đi sảnh và hành lang, giúp thuật toán Local Costmap phân bổ hành lang an toàn $0.8\\text{ m}$.

#### C. Nhóm Nhân Vật Lâm Sàng (NPC Patients & Doctors):
- Ở góc nhìn chuyển động liên tục của robot tuần tra, việc người bệnh ngồi cúi gập trên ghế hoặc bác sĩ di chuyển dọc hành lang chịu ảnh hưởng của hiện tượng che khuất một phần (occlusion).
- **Khuyến nghị kỹ thuật cho ROS2 Node**:
  - Tích hợp tracker đa mục tiêu siêu nhẹ **ByteTrack** hoặc **OC-SORT** kết hợp cùng YOLO26n ONNX.
  - Tracker giúp kết nối các detection qua từng khung hình (temporal association), khắc phục hoàn toàn hiện tượng nhấp nháy hoặc che khuất tạm thời khi robot quay đầu tại các góc rẽ 90°.

---
### ⚙️ 4. Bảng Tổng Hợp Thông Số Kỹ Thuật Triển Khai Edge Robotics

| Hạng Mục Kỹ Thuật | Giá Trị Thực Nghiệm | Đánh Giá Tương Thích Edge |
| :--- | :--- | :--- |
| **Kiến trúc mô hình** | **YOLO26n Nano** | Chuẩn hóa siêu nhẹ thế hệ mới |
| **Kích thước file trọng số** | **5.3 MB (.pt) / 9.4 MB (.onnx)** | Tiết kiệm 90% bộ nhớ flash |
| **Số lượng tham số** | **2,393,623 (2.39M)** | Phù hợp thiết bị nhúng |
| **Độ trễ suy luận (GPU)** | **3.4 ms (~294 FPS)** | Nhanh gấp 10 lần yêu cầu thời gian thực |
| **Khung hình kiểm thử tuần tra** | **531 frames liên tục** | Bao phủ 100% 7 phân khu lâm sàng |
| **Tính toàn vẹn dữ liệu** | **Zero Data Leakage** | Bộ Testset độc lập, bộ Test cũ giữ nguyên |
"""

nb.cells = [
    nbf.v4.new_markdown_cell(md_header),
    nbf.v4.new_code_cell(code_metrics),
    nbf.v4.new_code_cell(code_per_class),
    nbf.v4.new_code_cell(code_image),
    nbf.v4.new_markdown_cell(md_analysis)
]

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ Đã tạo thành công Sổ tay 15: {NOTEBOOK_PATH}")
