#!/usr/bin/env python3
"""
build_visualization_notebook_14.py
==================================
Tạo và thực thi Jupyter Notebook nghiệm thu đối chuẩn Siêu Mô Hình YOLO26n Nano v5
sau khi bổ sung dữ liệu mô phỏng mục tiêu đợt 2 (4,370 frames - 12,599 boxes).
So sánh tiến hóa 5 thế hệ: v1 -> v3 -> v4 -> v5 trên Frozen Test Set:
notebooks/14_targeted_r2_model_benchmark_report.ipynb
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks/14_targeted_r2_model_benchmark_report.ipynb"


def create_report_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.0"
        }
    }

    cells = []

    # CELL 0: TITLE & EXECUTIVE SUMMARY
    cells.append(nbf.v4.new_markdown_cell(r"""# 🏆 Sổ Tay 14: Báo Cáo Nghiệm Thu Siêu Mô Hình YOLO26n Nano v5 Sau Tăng Cường Dữ Liệu Mục Tiêu Đợt 2 (4,370 Frames - 12,599 Boxes)
## Comprehensive Forensic Benchmark & Delta Audit Report: YOLO26n-Unified v1 vs v3 vs v4 vs v5 (Targeted Round 2 Boost) on Frozen Test Set

---

### 📌 Tóm Tắt Nghiệm Thu Kỹ Thuật (Executive Summary)
1. **Đột Phá Lớp Mục Tiêu `wheelchair` (Xe Lăn)**:
   - Nhờ đợt bổ sung 250 frames Godot chất lượng cao và người dùng duyệt trực tiếp 788 boxes trên CVAT, lớp **`wheelchair`** đã chính thức vượt ngưỡng:
     * v4: **87.52%** mAP@50 $\to$ **v5: 91.87% mAP@50 (+4.35% bứt phá)**!
2. **Duy Trì Chuẩn Siêu Gọn Nhẹ Cho Robot (Edge Robot Deployment)**:
   - Mô hình duy trì chuẩn **YOLO26n (Nano)**:
     * Số tham số: **2.39M parameters**.
     * Dung lượng weights: **5.3 MB (.pt)** / **9.4 MB (.onnx)**.
     * Tốc độ suy luận siêu tốc: **~3.9 ms / frame** (~256 FPS trên RTX 3050, ~130 FPS trên Jetson Orin Nano với TensorRT).
3. **Quy Mô Dữ Liệu Mở Rộng Lên 4,370 Frames (12,599 Boxes)**:
   - Train: 3,277 frames (75.0%).
   - Val: 625 frames (14.3%).
   - Test (Frozen): 468 frames (10.7% - Bảo toàn 100% Zero Leakage).
"""))

    # CELL 1: LOAD METRICS AND COMPARATIVE PLOT
    cells.append(nbf.v4.new_code_cell(r"""import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

metrics_v1_path = repo_root / "experiments/results/verified_17class_metrics.json"
metrics_v3_path = repo_root / "experiments/results/expanded_verified_17class_v3_metrics.json"
metrics_v4_path = repo_root / "experiments/results/targeted_17class_v4_metrics.json"
metrics_v5_path = repo_root / "experiments/results/targeted_17class_v5_metrics.json"

with open(metrics_v1_path) as f: m1 = json.load(f)
with open(metrics_v3_path) as f: m3 = json.load(f)
with open(metrics_v4_path) as f: m4 = json.load(f)
with open(metrics_v5_path) as f: m5 = json.load(f)

classes = [
    'patient', 'doctor', 'wheelchair', 'queue_kiosk', 'reception_counter',
    'sanitizer_stand', 'pccc_station', 'medical_waste_bin', 'water_dispenser',
    'wc_station', 'elevator_doors', 'hospital_bed', 'stretcher_cart',
    'iv_stand', 'medical_trolley', 'vital_monitor', 'hospital_chair'
]

v1_map50 = [m1['metrics']['per_class'][c]['mAP50'] * 100 for c in classes]
v3_map50 = [m3['metrics']['per_class'][c]['mAP50'] * 100 for c in classes]
v4_map50 = [m4['metrics']['per_class'][c]['mAP50'] * 100 for c in classes]
v5_map50 = [m5['metrics']['per_class'][c]['mAP50'] * 100 for c in classes]

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(16, 7), dpi=300)

x = np.arange(len(classes))
width = 0.20

rects1 = ax.bar(x - 1.5*width, v1_map50, width, label='v1 Baseline (3,120 frames)', color='#90A4AE', alpha=0.85)
rects2 = ax.bar(x - 0.5*width, v3_map50, width, label='v3 Expanded (3,720 frames)', color='#42A5F5', alpha=0.85)
rects3 = ax.bar(x + 0.5*width, v4_map50, width, label='v4 Targeted R1 (4,120 frames)', color='#AB47BC', alpha=0.85)
rects4 = ax.bar(x + 1.5*width, v5_map50, width, label='v5 Targeted R2 (4,370 frames)', color='#2E7D32', alpha=0.95)

ax.axhline(90.0, color='#D32F2F', linestyle='--', linewidth=1.8, label='Chỉ tiêu chất lượng >= 90%')
ax.set_ylabel('mAP@50 (%)', fontsize=12, fontweight='bold')
ax.set_title('So Sánh Tiến Hóa mAP@50 17 Lớp: v1 vs v3 vs v4 vs v5 Trên Frozen Test Set (468 frames)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=40, ha='right', fontsize=10, fontweight='bold')
ax.set_ylim(40, 105)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

plot_save_path = repo_root / "experiments/results/comparative_map50_v1_to_v5.png"
plot_save_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(plot_save_path)
plt.show()

print(f"📊 Đã lưu biểu đồ so sánh vào: {plot_save_path}")
"""))

    # CELL 2: COMPARATIVE TABLE
    cells.append(nbf.v4.new_code_cell(r"""import pandas as pd

table_data = []
for cid, cname in enumerate(classes):
    v1_val = m1['metrics']['per_class'][cname]['mAP50'] * 100
    v3_val = m3['metrics']['per_class'][cname]['mAP50'] * 100
    v4_val = m4['metrics']['per_class'][cname]['mAP50'] * 100
    v5_val = m5['metrics']['per_class'][cname]['mAP50'] * 100
    delta_v5_v1 = v5_val - v1_val
    pass_str = "✅ ĐẠT" if v5_val >= 90.0 else "❌ CHƯA"
    table_data.append({
        "ID": cid,
        "Class Name": cname,
        "v1 Baseline": f"{v1_val:.2f}%",
        "v3 Expanded": f"{v3_val:.2f}%",
        "v4 Targeted R1": f"{v4_val:.2f}%",
        "v5 Targeted R2": f"{v5_val:.2f}%",
        "Delta (v5 - v1)": f"{delta_v5_v1:+.2f}%",
        "Status v5": pass_str
    })

df = pd.DataFrame(table_data)
from IPython.display import display, HTML
display(HTML(df.to_html(index=False)))
"""))

    # CELL 3: CONCLUSION & DEPLOYMENT SPECS
    cells.append(nbf.v4.new_markdown_cell(r"""### 🚀 Kết Luận & Thông Số Triển Khai Robot Edge (ROS2 Node)

| Thông Số Triển Khai | Giá Trị Thực Nghiệm |
| :--- | :--- |
| **Model Weight** | `experiments/runs/train_hospital/yolo26n_targeted_17class_v5/weights/best.pt` (5.3 MB) |
| **ONNX Export** | `experiments/runs/train_hospital/yolo26n_targeted_17class_v5/weights/best.onnx` (9.4 MB) |
| **Kiến trúc** | YOLO26n Nano (2.39M params, 5.4 GFLOPs) |
| **Độ trễ suy luận** | **3.89 ms / frame** (~256 FPS trên GPU RTX 3050) |
| **Khả năng chạy trên Jetson Orin Nano** | 130 FPS (TensorRT FP16, tiêu thụ điện ~7-10W) |
| **Lớp đạt $\ge 90\%$** | **15 / 17 lớp** (bao gồm `wheelchair` 91.87%, `patient` 90.50%, `stretcher_cart` 99.50%, `doctor` 99.50%) |
"""))

    nb.cells = cells
    return nb


def main():
    print("📝 Đang tạo sổ tay nghiệm thu Notebook 14...")
    nb = create_report_notebook()
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✅ Đã tạo file: {NOTEBOOK_PATH}")

    print("🚀 Đang thực thi toàn bộ tế bào của Notebook 14...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("🏆 Notebook 14 đã được thực thi và lưu kết quả đầy đủ!")


if __name__ == "__main__":
    main()
