#!/usr/bin/env python3
"""
build_visualization_notebook_10.py
Tạo và thực thi Jupyter Notebook nghiệm thu toàn diện Siêu Mô Hình YOLO26n-Unified (17 Classes)
trên bộ nhãn Ground Truth đã được người dùng kiểm duyệt CVAT:
notebooks/10_final_verified_model_benchmark_report.ipynb
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks/10_final_verified_model_benchmark_report.ipynb"


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

    # ==========================================================================
    # CELL 0: TITLE & EXECUTIVE SUMMARY
    # ==========================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""# 🏆 Sổ Tay 10: Báo Cáo Nghiệm Thu Siêu Mô Hình YOLO26n 17 Lớp (Nhãn Ground Truth CVAT & Zero Data Leakage)
## Comprehensive Forensic Benchmark & Audit Report of YOLO26n-Unified (17 Hospital Classes) on Frozen Test Set

---

### 📌 Tóm Tắt Nghiệm Thu Kỹ Thuật (Executive Summary)
1. **Bộ Dữ Liệu Thực Tế Đã Kiểm Duyệt CVAT**:
   - Quy mô: **3,120 frames thực tế bệnh viện** (2,184 Train / 468 Val / 468 Test).
   - Tổng cộng **8,337 Ground Truth bounding boxes** đã được người dùng rà soát, tinh chỉnh và bổ sung trực tiếp trên CVAT (+434 boxes mới).
   - Đạt chuẩn **Zero Data Leakage 100%** qua Kiến trúc Phân chia Đa tầng (Case-Level Patient Isolation, Viewpoint Disjointness, MD5 Collision = 0).
2. **Kích Hoạt Hệ Thống Tăng Cường Đa Dạng (Full Augmentation Suite)**:
   - Kích hoạt **Copy-Paste (0.30)**, **Mosaic (1.00)**, **MixUp (0.15)**, **Multi-Scale Zooming (scale=0.50)**, **Perspective Distortion (0.0005)**, kết hợp siết chặt **Box Regression Gain (box=8.0)**.
   - Cơ chế `close_mosaic=10`: Tắt ghép ảnh ở 10 epochs cuối để mô hình tinh chỉnh mịn trên ảnh thực tế.
3. **Hiệu Năng Vượt Trội Trên Tập Test Đóng Băng (468 frames, 1,162 Ground Truth Boxes)**:
   - **Precision**: **93.63%** *(Vượt chuẩn $\ge 90\%$)*
   - **Recall**: **90.50%** *(Vượt chuẩn $\ge 90\%$)*
   - **F1-Score**: **92.04%** *(Vượt chuẩn $\ge 90\%$)*
   - **mAP@50 Tổng thể**: **93.24%** *(Vượt chuẩn $\ge 90\%$)*
   - **mAP@50-95 Tổng thể**: **64.66%**
   - **Tốc độ Inference**: **3.83 ms / frame** (~**261 FPS**) trên GPU NVIDIA RTX 3050 Laptop.
"""))

    # ==========================================================================
    # CELL 1: SETUP & LOAD METRICS
    # ==========================================================================
    cells.append(nbf.v4.new_code_cell(r"""import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / "configs").exists():
    REPO_ROOT = REPO_ROOT.parent

METRICS_PATH = REPO_ROOT / "experiments/results/verified_17class_metrics.json"
RUN_DIR = REPO_ROOT / "experiments/runs/train_hospital/yolo26n_verified_17class_v1"
DATASET_DIR = REPO_ROOT / "data/vision/dataset/hospital_unified_3120_17class"

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics_data = json.load(f)

print(f"✅ Đã nạp thành công dữ liệu đo lường: {METRICS_PATH.name}")
print(f"• Model Name:     {metrics_data['model_name']}")
print(f"• Dataset:        {metrics_data['dataset']}")
print(f"• Test Frames:    {metrics_data['test_frames']} frames")
print(f"• Overall mAP@50: {metrics_data['metrics']['mAP50']*100:.2f}%")
print(f"• Precision:      {metrics_data['metrics']['precision']*100:.2f}%")
print(f"• Recall:         {metrics_data['metrics']['recall']*100:.2f}%")
print(f"• F1-Score:       {metrics_data['metrics']['f1']*100:.2f}%")
"""))

    # ==========================================================================
    # CELL 2: 17-CLASS BENCHMARK TABLE & EVALUATION
    # ==========================================================================
    cells.append(nbf.v4.new_code_cell(r"""per_class = metrics_data["metrics"]["per_class"]

vn_names = {
    0: "Bệnh nhân (patient)",
    1: "Bác sĩ / Nhân viên (doctor)",
    2: "Xe lăn (wheelchair)",
    3: "Kiosk bấm số (queue_kiosk)",
    4: "Quầy tiếp đón (reception_counter)",
    5: "Trụ sát khuẩn (sanitizer_stand)",
    6: "Trụ cứu hỏa (pccc_station)",
    7: "Thùng rác y tế (medical_waste_bin)",
    8: "Cây nước uống (water_dispenser)",
    9: "Khu vệ sinh WC (wc_station)",
    10: "Cửa thang máy (elevator_doors)",
    11: "Giường bệnh (hospital_bed)",
    12: "Cáng cứu thương (stretcher_cart)",
    13: "Cột truyền dịch (iv_stand)",
    14: "Xe đẩy thuốc (medical_trolley)",
    15: "Monitor sinh tồn (vital_monitor)",
    16: "Ghế chờ bệnh viện (hospital_chair)"
}

test_instances = {
    0: 148, 1: 76, 2: 88, 3: 33, 4: 47, 5: 210, 6: 85, 7: 181,
    8: 25, 9: 10, 10: 13, 11: 47, 12: 5, 13: 11, 14: 38, 15: 27, 16: 118
}

table_rows = []
for cname, stats in per_class.items():
    cid = stats["class_id"]
    map50 = stats["mAP50"] * 100
    map50_95 = stats["mAP50_95"] * 100
    p = stats["precision"] * 100
    r = stats["recall"] * 100
    f1 = stats["f1"] * 100
    instances = test_instances.get(cid, 0)
    passed = map50 >= 90.0

    table_rows.append({
        "Class ID": cid,
        "Tên Lớp (Anh)": cname,
        "Tên Lớp (Việt)": vn_names.get(cid, cname),
        "Test Boxes": instances,
        "mAP@50 (%)": map50,
        "mAP@50-95 (%)": map50_95,
        "Precision (%)": p,
        "Recall (%)": r,
        "F1-Score (%)": f1,
        "Chuẩn >= 90%": "✅ ĐẠT" if passed else "❌ CHƯA ĐẠT"
    })

df_audit = pd.DataFrame(table_rows).sort_values(by="mAP@50 (%)", ascending=False).reset_index(drop=True)
passed_count = (df_audit["Chuẩn >= 90%"] == "✅ ĐẠT").sum()

print(f"📊 TỔNG KẾT NGHIỆM THU 17 LỚP: {passed_count}/17 LỚP ĐẠT CHUẨN >= 90% mAP@50")
df_audit
"""))

    # ==========================================================================
    # CELL 3: VISUAL RANKING BAR CHART
    # ==========================================================================
    cells.append(nbf.v4.new_code_cell(r"""plt.figure(figsize=(14, 8))
colors = ["#2f855a" if v >= 90.0 else "#dd6b20" for v in df_audit["mAP@50 (%)"]]

bars = plt.barh(df_audit["Tên Lớp (Việt)"], df_audit["mAP@50 (%)"], color=colors, edgecolor='#1a202c', alpha=0.9, height=0.68)
plt.gca().invert_yaxis()
plt.axvline(90.0, color='#e53e3e', linestyle='--', lw=2, label='Ngưỡng Nghiệm Thu (90.0%)')

for bar in bars:
    w = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    plt.text(w + 0.8, y, f"{w:.1f}%", va='center', ha='left', fontsize=10, fontweight='bold', color='#1a202c')

plt.title("Xếp Hạng Năng Lực Nhận Diện 17 Lớp Đối Tượng Bệnh Viện (mAP@50 trên Tập Test Đóng Băng)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("mAP@50 (%)", fontsize=12)
plt.xlim(0, 108)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.show()
"""))

    # ==========================================================================
    # CELL 4: TRAINING DYNAMICS & LOSS CURVES
    # ==========================================================================
    cells.append(nbf.v4.new_code_cell(r"""results_csv = RUN_DIR / "results.csv"
if results_csv.exists():
    df_train = pd.read_csv(results_csv)
    df_train.columns = [c.strip() for c in df_train.columns]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = df_train['epoch']

    # 1. Box Loss
    ax = axes[0, 0]
    ax.plot(epochs, df_train['train/box_loss'], label='Train Box Loss', color='#3182ce', lw=2)
    ax.plot(epochs, df_train['val/box_loss'], label='Val Box Loss', color='#e53e3e', lw=2)
    ax.set_title("Box Regression Loss (Tối ưu hóa Bounding Box)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    # 2. Classification Loss
    ax = axes[0, 1]
    ax.plot(epochs, df_train['train/cls_loss'], label='Train Cls Loss', color='#805ad5', lw=2)
    ax.plot(epochs, df_train['val/cls_loss'], label='Val Cls Loss', color='#dd6b20', lw=2)
    ax.set_title("Classification Loss (Phân loại 17 Lớp)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    # 3. Precision & Recall
    ax = axes[1, 0]
    ax.plot(epochs, df_train['metrics/precision(B)'], label='Val Precision', color='#2b6cb0', lw=2)
    ax.plot(epochs, df_train['metrics/recall(B)'], label='Val Recall', color='#d69e2e', lw=2)
    ax.set_title("Precision & Recall Trajectory", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    # 4. Validation mAP
    ax = axes[1, 1]
    ax.plot(epochs, df_train['metrics/mAP50(B)'], label='Val mAP@50', color='#38a169', lw=2.5)
    ax.plot(epochs, df_train['metrics/mAP50-95(B)'], label='Val mAP@50-95', color='#319795', lw=2)
    ax.set_title("Validation Accuracy Trajectory (mAP)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.suptitle("Động Học Huấn Luyện Siêu Mô Hình YOLO26n (45 Epochs, Full Augmentation)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Không tìm thấy results.csv")
"""))

    # ==========================================================================
    # CELL 5: CONFUSION MATRIX & CURVES
    # ==========================================================================
    cells.append(nbf.v4.new_code_cell(r"""fig, axes = plt.subplots(1, 2, figsize=(18, 8))

cm_path = RUN_DIR / "confusion_matrix_normalized.png"
if cm_path.exists():
    axes[0].imshow(Image.open(cm_path))
    axes[0].set_title("Ma Trận Nhầm Lẫn Chuẩn Hóa (Normalized Confusion Matrix)", fontsize=13, fontweight='bold')
    axes[0].axis('off')

pr_path = RUN_DIR / "BoxPR_curve.png"
if pr_path.exists():
    axes[1].imshow(Image.open(pr_path))
    axes[1].set_title("Đường Cong Precision-Recall (PR Curve across 17 Classes)", fontsize=13, fontweight='bold')
    axes[1].axis('off')

plt.tight_layout()
plt.show()
"""))

    # ==========================================================================
    # CELL 6: FORENSIC ROOT CAUSE ANALYSIS & CONCLUSION
    # ==========================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""---

## 🔍 Phân Tích Pháp Y Chuyên Sâu: Hiện Trạng & Tiến Bộ Vượt Bậc Của 17 Lớp

### 🌟 1. Nhóm 13 Lớp Đạt Chuẩn Xuất Sắc ($\ge 90\%$ mAP@50):
1. **`doctor` (99.5%)**, **`wc_station` (99.5%)**, **`elevator_doors` (99.5%)**, **`sanitizer_stand` (99.4%)**, **`queue_kiosk` (99.1%)**, **`reception_counter` (98.7%)**, **`pccc_station` (98.2%)**, **`medical_waste_bin` (97.9%)**: Nhận diện gần như hoàn hảo tuyệt đối.
2. **`hospital_bed` (97.52%)**: Bước nhảy vọt thần kỳ! Tăng từ $91.0\%$ lên $97.52\%$ nhờ bổ sung 47 boxes Ground Truth ca giường bệnh thực tế.
3. **`vital_monitor` (95.20%)** và **`hospital_chair` (94.43%)**: Đạt độ ổn định cực cao.
4. **`water_dispenser` (90.64%)** và **`medical_trolley` (90.57%)**: Chính thức vượt qua ranh giới 90% thành công!

---

### ⚠️ 2. Báo Cáo Minh Bạch Về 4 Lớp Dưới Ngưỡng 90%:
- **`patient` (89.84% mAP@50)**:
  * Điểm thực tế đạt **89.84%** (chỉ cách ngưỡng 90% đúng 0.16%). Bệnh nhân ở các ca khám đông người thường bị che khuất một phần bởi bác sĩ đứng khám.
- **`wheelchair` (88.69% mAP@50)**:
  * Đạt **88.69%** (thiếu 1.31%). Các góc nhìn từ sau lưng xe lăn khi người ngồi che khuất bánh xe làm giảm nhẹ độ tin cậy.
- **`stretcher_cart` (79.50% mAP@50)**:
  * Precision đạt rất cao (**92.27%**), nhưng tập Test Đóng Băng **chỉ có duy nhất 5 chiếc cáng** trong toàn bộ kho dữ liệu thực tế (1 chiếc bị che khuất làm tỷ lệ Recall giảm 20%).
- **`iv_stand` (66.67% mAP@50)**:
  * **Precision đạt gần như tuyệt đối: 99.88%** (Mô hình hoàn toàn không báo động giả!).
  * Nhãn mới từ CVAT đã giúp loại trừ các viền giả của đợt thử nghiệm trước. Khó khăn duy nhất là cọc inox mảnh 2cm ở cự ly xa >3.5m trong ảnh 640x640 chỉ chiếm 2-3 pixel bề rộng, khiến Recall dừng ở 63.64%.

---

## 🏁 Kết Luận Nghiệm Thu & Khuyến Nghị
- **Mô hình đạt điểm tổng thể ấn tượng**: Precision **93.63%**, Recall **90.50%**, F1-Score **92.04%**, mAP@50 **93.24%**.
- **Độ an toàn Edge-AI thời gian thực**: Tốc độ xử lý **3.83 ms** (~**261 FPS**), đáp ứng vượt trội mọi tiêu chuẩn triển khai thực tế trên robot và camera giám sát bệnh viện!
"""))

    nb.cells = cells
    return nb


def main():
    print("==================================================================")
    print("📝 BẮT ĐẦU XUẤT SỔ TAY BÁO CÁO NGHIỆM THU: 10_final_verified_model_benchmark_report.ipynb")
    print(f"🎯 Đích: {NOTEBOOK_PATH}")
    print("==================================================================")

    nb = create_report_notebook()
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"✅ Đã tạo cấu trúc sổ tay. Bắt đầu thực thi tính toán và vẽ đồ thị...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"🎉 ĐÃ THỰC THI THÀNH CÔNG VÀ LƯU SỔ TAY BÁO CÁO TẠI: {NOTEBOOK_PATH}")
    print(f"💾 Dung lượng sổ tay: {NOTEBOOK_PATH.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
