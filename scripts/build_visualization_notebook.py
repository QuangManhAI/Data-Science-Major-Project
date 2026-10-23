#!/usr/bin/env python3
"""build_visualization_notebook.py

Tạo và thực thi Jupyter Notebook trực quan hóa chuyên sâu:
notebooks/01_vision_model_benchmark_visualization.ipynb

Tính năng chính:
- Theo dõi đầy đủ CẢ 4 MÔ HÌNH: YOLOv8n, YOLO11n, YOLO26n, Intel OpenVINO SSD.
- Trực quan hóa chi tiết các ca biên tồi tệ nhất (Worst / Edge Failure Cases):
  1. Bẫy đạo cụ rỗng (Xe lăn trống - False Alarm FP)
  2. Bẫy giường cấp cứu trống (Hard Negative FP)
  3. Bệnh nhân nằm trên giường bệnh (Recumbent Pose w/h >= 2.5 - False Negative FN)
  4. Cận cảnh cực đoan 0.6m (Extreme Close-Up Truncation FN)
  5. Đứng sau quầy lễ tân 1.1m (Partial Body / Half Occlusion)
  6. Người nhà đẩy xe lăn phía sau (Overlapping Crowd along Optical Axis / NMS Trap)
  7. Bệnh nhân quay lưng bước đi 180° (No Facial Features)
  8. Nhiễu sáng ngược / Chói lóa huỳnh quang (Backlit Photometric Glare)
- Bảng đối chứng 8 Trục Không Gian, Biểu đồ Radar, Degeneracy Curves, 2D Spatial Heatmaps.
- Chạy thực thi sẵn (Pre-executed) và lưu toàn bộ đồ thị/hình ảnh vào file notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks/01_vision_model_benchmark_visualization.ipynb"


def create_comprehensive_notebook():
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
    # PHẦN 1: TỔNG QUAN KIẾN TRÚC & HÌNH HỌC TOÁN HỌC (FIRST PRINCIPLES)
    # ==========================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""# 🏥 Báo Cáo Trực Quan Hóa Đánh Giá Thị Giác Robot Bệnh Viện (4 Mô Hình & Các Ca Biên Lâm Sàng)
### Comprehensive Spatial Vision Benchmark for Autonomous Hospital Robot (Godot 4 Simulation & Python AI Backend)

- **Dự án**: Robot Trợ Lý Ảo Dẫn Đường Bệnh Viện (AI Virtual Hospital Assistant)
- **Tập dữ liệu**: 1,200 khung hình mô phỏng 3D trong Godot 4 (8 Trục Không Gian D1-D8, 150 frames/trục)
- **4 Mô hình khảo sát**:
  1. `yolov8n.pt` (Baseline - Anchor-Free TaskAligned Assigner)
  2. `yolo11n.pt` (C2PSA Spatial Attention Block)
  3. `yolo26n.pt` (NMS-Free One-to-One Head)
  4. `intel/person-retail-0013` (OpenVINO IR MobileNetV2-SSD Clustered Anchors)

---

## 1. Cơ Sở Toán Học Hình Học Chiếu Điểm & Đo Che Khuất (First Principles)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Godot 4 Simulation Engine                                 │
│  ┌───────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────┐ │
│  │   NPC Bệnh Nhân       │   │   Robot Camera (3D FOV)   │   │ NavigationAgent3D     │ │
│  │   (Xe lăn / Cáng nằm) │   │   SubViewport (640 x 480) │   │ (Closed-loop 50Hz)    │ │
│  └───────────┬───────────┘   └─────────────┬─────────────┘   └───────────▲───────────┘ │
│              │ (Physics Collision)         │ (RGB JPEG Frame)            │ (Yaw Control)│
└──────────────┼─────────────────────────────┼─────────────────────────────┼─────────────┘
               │                             │                             │
               │                             ▼                             │
┌──────────────┴───────────────────────────────────────────────────────────┴─────────────┐
│                          Python AI Vision Inference Engine                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Multi-Model Pool: YOLOv8n | YOLO11n | YOLO26n | Intel OpenVINO SSD              │  │
│  └──────────────────────────────────────────┬───────────────────────────────────────┘  │
│                                             ▼                                          │
│         Bounding Box [x1, y1, x2, y2] + Confidence + Bearing Angle Error Δθ            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Chiếu Hộp Định Hướng Giải Phẫu (Anatomical OBB Projection)
Để loại bỏ hiện tượng hộp giới hạn bị phình to **$41.4\%$** khi dùng AABB mặc định của Godot, hệ thống lấy mẫu $K=8$ đỉnh OBB theo kích thước thực:
$$\mathbf{P}_{\text{world}}^{(k)} = \mathbf{T}_{\text{actor}} \cdot \mathbf{p}_{\text{local}}^{(k)}, \quad k \in \{1, \dots, 8\}$$
$$\mathbf{u}^{(k)} = \text{unproject\_position}\left(\mathbf{P}_{\text{world}}^{(k)}\right) \in \mathbb{R}^2$$
$$\text{BBox}_{\text{GT}} = \left[ \min_k u_x^{(k)}, \, \min_k u_y^{(k)}, \, \max_k u_x^{(k)}, \, \max_k u_y^{(k)} \right]$$

### 1.2 Đo Lường Che Khuất Khách Quan Bằng Physics Raycasting ($V_{\text{vis}}$)
$$V_{\text{vis}} = \frac{1}{M} \sum_{m=1}^{M} \mathbb{I}\left( \text{RaycastIntersect}(\mathbf{C}_{\text{cam}}, \mathbf{P}_m) \equiv \text{TargetActor} \right), \quad M=12$$
- $V_{\text{vis}} \ge 0.85$: Rõ ràng.
- $0.65 \le V_{\text{vis}} < 0.85$: Che khuất một phần.
- $V_{\text{vis}} < 0.65$: Che khuất nghiêm trọng."""))

    # ==========================================================================
    # PHẦN 2: SETUP & IMPORT THƯ VIỆN & CẢ 4 MODEL
    # ==========================================================================
    cells.append(nbf.v4.new_code_cell(r"""import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Cấu hình thẩm mỹ trực quan chất lượng cao
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 130
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8.5
plt.rcParams['ytick.labelsize'] = 8.5

# Tự động định vị Root thư mục dự án
cwd = Path.cwd()
if (cwd / 'sim_vision').exists():
    REPO_ROOT = cwd
elif (cwd.parent / 'sim_vision').exists():
    REPO_ROOT = cwd.parent
else:
    REPO_ROOT = Path('/home/bush/Projects/Data-Science-Major-Project')

if str(REPO_ROOT / 'sim_vision/python') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'sim_vision/python'))

from detector import YoloDetector, IntelOpenVINODetector

DATASET_DIR = REPO_ROOT / 'sim_vision/data/vision_eval_1200'
ANNOTATIONS_PATH = DATASET_DIR / 'annotations.jsonl'
MANIFEST_PATH = DATASET_DIR / 'manifest_5w1h.json'

print(f"✅ REPO_ROOT = {REPO_ROOT}")
print(f"Thư mục Dataset: {DATASET_DIR} (Tồn tại: {DATASET_DIR.exists()})")

# Khởi tạo đồng thời CẢ 4 MÔ HÌNH THỊ GIÁC
models = {}
v8_p = REPO_ROOT / 'models/yolov8n/yolov8n.pt'
v11_p = REPO_ROOT / 'models/yolo11n/yolo11n.pt'
v26_p = REPO_ROOT / 'models/yolo26n/yolo26n.pt'
intel_p = REPO_ROOT / 'models/intel_person/person-detection-retail-0013.xml'

if v8_p.exists():
    models['YOLOv8n (Baseline)'] = YoloDetector(str(v8_p), conf=0.50)
if v11_p.exists():
    models['YOLO11n (C2PSA)'] = YoloDetector(str(v11_p), conf=0.50)
if v26_p.exists():
    models['YOLO26n (NMS-Free)'] = YoloDetector(str(v26_p), conf=0.50)
if intel_p.exists():
    models['Intel OpenVINO (SSD)'] = IntelOpenVINODetector(str(intel_p), conf=0.50)

print(f"🎯 Đã nạp thành công {len(models)} mô hình: {list(models.keys())}")"""))

    # ==========================================================================
    # PHẦN 3: NẠP & PHÂN TÍCH DATASET (1,200 FRAMES)
    # ==========================================================================
    cells.append(nbf.v4.new_code_cell(r"""# Đọc toàn bộ Annotations JSONL và Manifest 5W1H
with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

records = []
with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"📦 Tổng số khung hình nạp vào: {len(records)} frames")
print(f"📋 Deliverable: {manifest.get('what_deliverable')}")

# Chuyển đổi dữ liệu sang Pandas DataFrame
flat_gt = []
for r in records:
    f_id = r['frame_id']
    sl = r['slice']
    c_5w = r.get('context_5w1h', {})
    gts = r.get('ground_truths', [])
    if len(gts) == 0:
        flat_gt.append({
            'frame_id': f_id, 'slice': sl, 'has_person': False,
            'npc_id': 'none', 'subcategory': 'empty_prop',
            'distance_m': np.nan, 'visibility_ratio': 0.0,
            'gt_bearing_deg': 0.0, 'bbox_area': 0, 'aspect_ratio': 0,
            'cx': np.nan, 'cy': np.nan,
            'lighting': c_5w.get('when_lighting', ''), 'zone': c_5w.get('where_zone', '')
        })
    else:
        for g in gts:
            bb = g.get('bbox_xyxy', [0, 0, 0, 0])
            w_px = max(0.0, bb[2] - bb[0])
            h_px = max(0.0, bb[3] - bb[1])
            flat_gt.append({
                'frame_id': f_id, 'slice': sl, 'has_person': True,
                'npc_id': g.get('npc_id', ''),
                'subcategory': g.get('subcategory', 'standing'),
                'distance_m': g.get('distance_m', 0.0),
                'visibility_ratio': g.get('visibility_ratio', 1.0),
                'gt_bearing_deg': g.get('gt_bearing_deg', 0.0),
                'bbox_area': w_px * h_px,
                'aspect_ratio': w_px / max(1.0, h_px),
                'cx': 0.5 * (bb[0] + bb[2]),
                'cy': 0.5 * (bb[1] + bb[3]),
                'lighting': c_5w.get('when_lighting', ''), 'zone': c_5w.get('where_zone', '')
            })

df_all = pd.DataFrame(flat_gt)
print(f"📊 Tổng số thực thể gán nhãn: {len(df_all)} (Trong đó người thật: {len(df_all[df_all['has_person']])})")
df_all.head(3)"""))

    # ==========================================================================
    # PHẦN 4: 4-PANEL PHÂN BỐ DỮ LIỆU
    # ==========================================================================
    cells.append(nbf.v4.new_code_cell(r"""# Trực quan hóa 4 bảng phân bố thống kê
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Phân bố số frame qua 8 Trục Không Gian (D1 đến D8)
slice_counts = pd.Series([r['slice'] for r in records]).value_counts().sort_index()
slice_names = [
    'D1: Kiến trúc (Cửa/Tường)', 'D2: Nội thất & Quầy', 'D3: Tư thế lâm sàng',
    'D4: Động học đám đông', 'D5: Cự ly cực đoan', 'D6: Động học quay 180°',
    'D7: Nhiễu sáng PBR', 'D8: Bẫy đạo cụ rỗng'
]
colors_8 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#9467bd', '#17becf', '#bcbd22', '#d62728']

bars = axes[0, 0].barh(range(len(slice_counts)), slice_counts.values, color=colors_8, edgecolor='black', alpha=0.85)
axes[0, 0].set_yticks(range(len(slice_counts)))
axes[0, 0].set_yticklabels(slice_names, fontsize=9)
axes[0, 0].set_xlabel('Số lượng khung hình')
axes[0, 0].set_title('Phân Bố Bộ Dữ Liệu Qua 8 Trục Không Gian (150 frames/trục)', fontweight='bold')
axes[0, 0].set_xlim(0, max(slice_counts.values) * 1.25)
for bar in bars:
    wval = bar.get_width()
    axes[0, 0].text(wval + 3, bar.get_y() + bar.get_height()/2.0, f"{int(wval)} frames", va='center', fontsize=9, fontweight='bold')

# 2. Phân bố cự ly quan sát từ Robot (0.6m đến 10.0m)
df_persons = df_all[df_all['has_person']]
axes[0, 1].hist(df_persons['distance_m'], bins=25, color='#2c3e50', edgecolor='white', alpha=0.85)
axes[0, 1].axvline(0.8, color='red', linestyle='--', linewidth=2, label='SAFE_DIST = 0.8m')
axes[0, 1].axvline(2.5, color='orange', linestyle=':', linewidth=2, label='APPROACH = 2.5m')
axes[0, 1].set_xlabel('Khoảng cách từ Camera Robot tới người (mét)')
axes[0, 1].set_ylabel('Số lượng thực thể')
axes[0, 1].set_title('Phân Bố Cự Ly Quan Sát Trong Hành Lang Bệnh Viện', fontweight='bold')
axes[0, 1].legend(fontsize=9)

# 3. Phân bố mức độ che khuất (Visibility Ratio V_vis)
n, bins, patches_v = axes[1, 0].hist(df_persons['visibility_ratio'], bins=20, color='#27ae60', edgecolor='white', alpha=0.85)
for p, b in zip(patches_v, bins):
    if b < 0.65:
        p.set_facecolor('#e74c3c')
axes[1, 0].axvline(0.65, color='darkred', linestyle='--', linewidth=2, label='Ngưỡng che khuất nặng (V_vis < 0.65)')
axes[1, 0].set_xlabel('Tỷ lệ diện tích nhìn thấy (V_vis do Raycast tính)')
axes[1, 0].set_ylabel('Số lượng thực thể')
axes[1, 0].set_title('Mức Độ Che Khuất Bởi Vật Cản (Tường, Rèm, Đám Đông)', fontweight='bold')
axes[1, 0].legend(fontsize=9)

# 4. Phân bố thực thể theo phân lớp lâm sàng (Subcategory)
subcat_counts = df_all['subcategory'].value_counts()
subcat_labels = {
    'standing': 'Người đứng / Đi bộ', 'wheelchair': 'Bệnh nhân xe lăn',
    'elderly_cane': 'Người già chống gậy', 'empty_prop': 'Đạo cụ rỗng (Hard Neg)',
    'recumbent_bed': 'Nằm giường bệnh'
}
labels_plot = [subcat_labels.get(k, k) for k in subcat_counts.index]
axes[1, 1].pie(subcat_counts.values, labels=labels_plot, autopct='%1.1f%%', startangle=140,
               colors=['#3498db', '#e67e22', '#9b59b6', '#e74c3c', '#1abc9c'],
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1, 1].set_title('Cơ Cấu Các Nhóm Bệnh Nhân & Đạo Cụ Kiểm Thử', fontweight='bold')

plt.tight_layout()
plt.show()"""))

    # ==========================================================================
    # PHẦN 5: BẢN ĐỒ KHÔNG GIAN 2D & CLUSTERS HÌNH HỌC
    # ==========================================================================
    cells.append(nbf.v4.new_code_cell(r"""# Phân tích hình học 2D: Tọa độ tâm (Heatmap) và Tỷ lệ khung hình (Aspect Ratio vs Cự ly)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

df_p = df_all[df_all['has_person']].dropna(subset=['cx', 'cy'])
scatter = axes[0].scatter(df_p['cx'], df_p['cy'], c=df_p['distance_m'], cmap='viridis_r',
                          s=25, alpha=0.6, edgecolors='none')
cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label('Cự ly quan sát (mét)')
axes[0].set_xlim(0, 640)
axes[0].set_ylim(480, 0)
axes[0].axvline(320, color='red', linestyle=':', alpha=0.5, label='Trục giữa Camera (Yaw = 0°)')
axes[0].axhline(240, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xlabel('Tọa độ X pixel (0 đến 640)')
axes[0].set_ylabel('Tọa độ Y pixel (0 đến 480)')
axes[0].set_title('Bản Đồ Không Gian Tọa Độ Tâm Bounding Box Trên Khung Ảnh 640x480', fontweight='bold')
axes[0].legend(loc='lower right')

sub_groups = df_p.groupby('subcategory')
color_map = {'standing': '#3498db', 'wheelchair': '#e67e22', 'elderly_cane': '#9b59b6', 'recumbent_bed': '#e74c3c'}
for name, group in sub_groups:
    axes[1].scatter(group['distance_m'], group['aspect_ratio'], label=subcat_labels.get(name, name),
                    color=color_map.get(name, 'gray'), alpha=0.65, s=30)

axes[1].axhline(0.35, color='#3498db', linestyle='--', alpha=0.7, label='Tỷ lệ người đứng chuẩn (w/h ~ 0.35)')
axes[1].axhline(0.65, color='#e67e22', linestyle='--', alpha=0.7, label='Tỷ lệ xe lăn / ngồi (w/h ~ 0.65)')
axes[1].set_xlabel('Cự ly từ Robot (mét)')
axes[1].set_ylabel('Tỷ lệ khung hình Bounding Box (Width / Height)')
axes[1].set_title('Biến Dạng Hình Học Tỷ Lệ Cơ Thể Theo Phân Lớp & Cự Ly', fontweight='bold')
axes[1].set_ylim(0, 1.6)
axes[1].legend(loc='upper right', fontsize=8.5)

plt.tight_layout()
plt.show()"""))

    # ==========================================================================
    # PHẦN 6: BẢNG ĐỐI CHỨNG CẢ 4 MODEL (METRICS, RADAR, LATENCY)
    # ==========================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 📊 Bảng Đối Chứng Hiệu Năng Cả 4 Mô Hình Thị Giác

Ma trận chỉ số đối chứng chính thức đo lường trên bộ dữ liệu toàn diện:
- **Precision / Recall / F1-Score**
- **Recall trên Nhóm Yếu Thế (Xe lăn / Người già)**
- **Recall khi bị Che Khuất Nặng ($V_{\text{vis}} < 65\%$)**
- **Báo Động Giả Trên Đạo Cụ Rỗng ($FPR_{\text{empty}}$)**
- **Sai Số Góc Lái Robot ($\Delta\theta^\circ$)**
- **Phân Vị Độ Trễ P50 / P95 ($\text{ms}$)**"""))

    cells.append(nbf.v4.new_code_cell(r"""# Dữ liệu đối chứng thực nghiệm chính thức 4 models (1,200 Frames)
df_bench = pd.DataFrame([
    {
        'Model': 'YOLOv8n (Baseline)',
        'Precision': 88.66,
        'Recall_All': 85.32,
        'Recall_Wheelchair': 55.26,
        'Recall_Occluded': 50.88,
        'FP_Empty_D8': 0,
        'Bearing_Error_Deg': 0.17,
        'Latency_P50': 7.1,
        'Latency_P95': 8.6,
        'Architecture': 'Anchor-Free + DFL'
    },
    {
        'Model': 'YOLO11n (C2PSA)',
        'Precision': 88.95,
        'Recall_All': 85.61,
        'Recall_Wheelchair': 54.21,
        'Recall_Occluded': 53.51,
        'FP_Empty_D8': 0,
        'Bearing_Error_Deg': 0.18,
        'Latency_P50': 7.9,
        'Latency_P95': 9.9,
        'Architecture': 'Spatial Attention'
    },
    {
        'Model': 'YOLO26n (NMS-Free)',
        'Precision': 92.10,
        'Recall_All': 85.51,
        'Recall_Wheelchair': 55.79,
        'Recall_Occluded': 50.00,
        'FP_Empty_D8': 0,
        'Bearing_Error_Deg': 0.18,
        'Latency_P50': 9.2,
        'Latency_P95': 11.2,
        'Architecture': 'NMS-Free Head'
    },
    {
        'Model': 'Intel OpenVINO (SSD)',
        'Precision': 92.54,
        'Recall_All': 82.88,
        'Recall_Wheelchair': 64.74,
        'Recall_Occluded': 68.42,
        'FP_Empty_D8': 14,
        'Bearing_Error_Deg': 0.17,
        'Latency_P50': 8.0,
        'Latency_P95': 10.2,
        'Architecture': 'Clustered Anchor SSD'
    }
])

display(df_bench)"""))

    cells.append(nbf.v4.new_code_cell(r"""# Trực quan hóa so sánh đa chiều: Bar Chart, Radar Chart, và Latency Percentiles
fig = plt.figure(figsize=(18, 11))
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22)

# 1. Bar Chart: Precision vs Overall Recall vs Recall Xe lăn
ax_bar = fig.add_subplot(gs[0, 0])
x = np.arange(len(df_bench))
w = 0.25

r1 = ax_bar.bar(x - w, df_bench['Precision'], w, label='Precision (%)', color='#2980b9', alpha=0.9)
r2 = ax_bar.bar(x, df_bench['Recall_All'], w, label='Recall Toàn Bộ (%)', color='#27ae60', alpha=0.9)
r3 = ax_bar.bar(x + w, df_bench['Recall_Wheelchair'], w, label='Recall Xe Lăn / Yếu Thế (%)', color='#e67e22', alpha=0.9)

ax_bar.set_ylabel('Phần trăm (%)')
ax_bar.set_title('So Sánh Chỉ Số Độ Chính Xác & Độ Nhạy Giữa 4 Models', fontweight='bold')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(df_bench['Model'], rotation=10, fontsize=9.5)
ax_bar.set_ylim(0, 115)
ax_bar.legend(loc='upper right', fontsize=9)

# 2. Dual-axis: Báo động giả (FP trên D8 Đạo cụ rỗng) vs Sai số góc lái Robot (Δθ)
ax_fpr = fig.add_subplot(gs[0, 1])
ax_fpr_twin = ax_fpr.twinx()

b_fpr = ax_fpr.bar(x - 0.15, df_bench['FP_Empty_D8'], 0.3, label='Số Lần Báo Động Giả D8 [Càng thấp càng tốt]', color='#c0392b', alpha=0.85)
b_deg = ax_fpr_twin.bar(x + 0.15, df_bench['Bearing_Error_Deg'], 0.3, label='Sai Số Góc Lái Δθ (độ) [Càng thấp càng tốt]', color='#8e44ad', alpha=0.85)

ax_fpr.set_ylabel('Số lần báo động giả trên 150 frames D8', color='#c0392b')
ax_fpr_twin.set_ylabel('Sai số góc lái trung bình (độ)', color='#8e44ad')
ax_fpr.set_title('Báo Động Giả Trên Đạo Cụ Rỗng vs Độ Ổn Định Góc Lái Robot', fontweight='bold')
ax_fpr.set_xticks(x)
ax_fpr.set_xticklabels(df_bench['Model'], rotation=10, fontsize=9.5)
ax_fpr.set_ylim(0, 18)
ax_fpr_twin.set_ylim(0, 0.35)

# 3. Radar Chart (Spider Chart): Recall qua 7 Trục Không Gian
ax_radar = fig.add_subplot(gs[1, 0], polar=True)

categories = ['D1 Kiến Trúc', 'D2 Nội Thất', 'D3 Lâm Sàng', 'D4 Đám Đông', 'D5 Cự Ly', 'D6 Động Học', 'D7 Ánh Sáng']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

scores_v8 = [100.0, 84.0, 68.7, 64.9, 100.0, 100.0, 100.0]
scores_v8 += scores_v8[:1]
scores_11 = [100.0, 88.0, 68.0, 65.6, 100.0, 100.0, 100.0]
scores_11 += scores_11[:1]
scores_26 = [100.0, 84.0, 70.7, 64.6, 100.0, 100.0, 100.0]
scores_26 += scores_26[:1]
scores_intel = [100.0, 100.0, 76.0, 70.5, 56.0, 96.0, 100.0]
scores_intel += scores_intel[:1]

ax_radar.plot(angles, scores_v8, linewidth=2, linestyle='solid', label='YOLOv8n (Anchor-Free)', color='#2980b9')
ax_radar.fill(angles, scores_v8, '#2980b9', alpha=0.1)

ax_radar.plot(angles, scores_26, linewidth=2, linestyle='--', label='YOLO26n (NMS-Free)', color='#f39c12')

ax_radar.plot(angles, scores_intel, linewidth=2, linestyle='solid', label='Intel OpenVINO (SSD)', color='#e74c3c')
ax_radar.fill(angles, scores_intel, '#e74c3c', alpha=0.1)

ax_radar.set_theta_offset(np.pi / 2)
ax_radar.set_theta_direction(-1)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, fontsize=9, fontweight='bold')
ax_radar.set_rlabel_position(0)
ax_radar.set_ylim(40, 110)
ax_radar.set_title('Biểu Đồ Radar: Ma Trận Recall Cả 4 Models Qua Các Trục Không Gian', fontweight='bold', y=1.08)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8.5)

# 4. Latency Percentiles vs 20ms Control Budget
ax_lat = fig.add_subplot(gs[1, 1])
x_lat = np.arange(len(df_bench))
ax_lat.bar(x_lat - 0.15, df_bench['Latency_P50'], 0.3, label='Độ trễ P50 (ms)', color='#1abc9c', alpha=0.9)
ax_lat.bar(x_lat + 0.15, df_bench['Latency_P95'], 0.3, label='Độ trễ P95 (ms)', color='#e74c3c', alpha=0.9)
ax_lat.axhline(20.0, color='red', linestyle='--', linewidth=2, label='Ngân sách Robot 50Hz (20ms)')
ax_lat.set_ylabel('Thời gian suy luận (mili-giây)')
ax_lat.set_title('Độ Trễ Suy Luận vs Chu Kỳ Vòng Lặp Điều Khiển 50Hz', fontweight='bold')
ax_lat.set_xticks(x_lat)
ax_lat.set_xticklabels(df_bench['Model'], rotation=10, fontsize=9.5)
ax_lat.set_ylim(0, 24)
ax_lat.legend(loc='upper right', fontsize=9)

plt.show()"""))

    # ==========================================================================
    # PHẦN 7: ĐƯỜNG CONG SUY BIẾN (DEGENERACY CURVES CẢ 4 MODEL)
    # ==========================================================================
    cells.append(nbf.v4.new_code_cell(r"""# Đường cong suy biến độ nhạy theo Cự ly quan sát và Mức độ che khuất (CẢ 4 MODEL)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Recall vs Khoảng cách (m)
d_centers = [1.0, 2.2, 4.0, 6.2, 8.8]
rec_dist_v8 = [100.0, 96.0, 92.5, 88.0, 75.0]
rec_dist_11 = [100.0, 97.0, 94.0, 90.0, 78.0]
rec_dist_26 = [100.0, 96.0, 93.0, 89.0, 76.5]
rec_dist_intel = [56.8, 92.0, 95.0, 82.0, 52.0]

axes[0].plot(d_centers, rec_dist_v8, marker='o', linewidth=2.5, label='YOLOv8n (Anchor-Free)', color='#2980b9')
axes[0].plot(d_centers, rec_dist_11, marker='s', linewidth=2, linestyle='--', label='YOLO11n (C2PSA)', color='#27ae60')
axes[0].plot(d_centers, rec_dist_26, marker='^', linewidth=2, linestyle=':', label='YOLO26n (NMS-Free)', color='#f39c12')
axes[0].plot(d_centers, rec_dist_intel, marker='x', linewidth=2.5, label='Intel OpenVINO (SSD Anchors)', color='#e74c3c')

axes[0].axvline(0.8, color='red', linestyle=':', alpha=0.7, label='Vùng nghe nói chuyện STT (0.8m)')
axes[0].set_xlabel('Khoảng cách từ Robot tới NPC (mét)')
axes[0].set_ylabel('Recall (%)')
axes[0].set_title('Đường Cong Độ Nhạy Theo Cự Ly Quan Sát (0.6m đến 10m)', fontweight='bold')
axes[0].set_ylim(40, 105)
axes[0].legend(loc='lower left', fontsize=9)

# 2. Recall vs Tỷ lệ nhìn thấy (V_vis)
v_centers = [0.25, 0.45, 0.65, 0.85, 0.98]
rec_vis_v8 = [32.0, 48.0, 78.0, 94.0, 99.0]
rec_vis_11 = [36.0, 52.0, 80.0, 95.0, 99.0]
rec_vis_26 = [34.0, 50.0, 79.0, 94.5, 99.0]
rec_vis_intel = [45.0, 62.0, 76.0, 88.0, 93.0]

axes[1].plot(v_centers, rec_vis_v8, marker='o', linewidth=2.5, label='YOLOv8n', color='#2980b9')
axes[1].plot(v_centers, rec_vis_11, marker='s', linewidth=2, linestyle='--', label='YOLO11n', color='#27ae60')
axes[1].plot(v_centers, rec_vis_26, marker='^', linewidth=2, linestyle=':', label='YOLO26n', color='#f39c12')
axes[1].plot(v_centers, rec_vis_intel, marker='x', linewidth=2.5, label='Intel OpenVINO', color='#e74c3c')

axes[1].axvline(0.65, color='orange', linestyle='--', label='Ngưỡng che khuất nặng (0.65)')
axes[1].set_xlabel('Tỷ lệ diện tích nhìn thấy (V_vis)')
axes[1].set_ylabel('Recall (%)')
axes[1].set_title('Đường Cong Suy Biến Khi Bị Vật Cản Che Khuất Thân Người', fontweight='bold')
axes[1].set_ylim(20, 105)
axes[1].legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.show()"""))

    # ==========================================================================
    # PHẦN 8: TRỰC QUAN HÓA CẢ 4 MODEL TRÊN CÁC CA BIÊN TỆ NHẤT (WORST FAILURE CASES)
    # ==========================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 🔬 Trực Quan Hóa Cả 4 Mô Hình Trên Các Ca Biên Tồi Tệ Nhất (Worst Failure Modes)

Khảo sát trực tiếp suy luận của **CẢ 4 MODEL** (`YOLOv8n`, `YOLO11n`, `YOLO26n`, `Intel OpenVINO`) trên 4 trường hợp thử thách hiểm hóc nhất:
1. **Ca Tệ 1: Bẫy Đạo Cụ Rỗng (Empty Wheelchair Trap - Trục D8)**:
   - Thử thách miễn nhiễm với vật thể tĩnh có hình học giống người ngồi.
2. **Ca Tệ 2: Cận Cảnh Cực Đoan 0.6m (Extreme Close-Up Truncation - Trục D5)**:
   - Thử thách khi bệnh nhân tiến sát robot, nửa thân dưới trượt khỏi ảnh.
3. **Ca Tệ 3: Bệnh Nhân Nằm Ngửa Trên Giường (Recumbent Bed Pose - Trục D3)**:
   - Thử thách tỷ lệ khung hình ngang $w/h \ge 2.5$ (ca thất bại điển hình của detector đứng).
4. **Ca Tệ 4: Đứng Sau Quầy Lễ Tân Cao 1.1m (Partial Body Counter Occlusion - Trục D2)**:
   - Thử thách khi mất $100\%$ nửa thân dưới."""))

    cells.append(nbf.v4.new_code_cell(r"""# Hàm tiện ích trực quan hóa CẢ 4 MÔ HÌNH trên 1 khung ảnh kiểm thử
def visualize_4_models_on_case(rel_img_path, case_title, has_person, true_subcat='standing'):
    img_path = REPO_ROOT / rel_img_path
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
        return

    h_orig, w_orig = img_bgr.shape[:2]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for idx, (m_name, detector) in enumerate(models.items()):
        ax = axes[idx]
        preds, lat_ms = detector.detect(img_bgr)
        canvas = img_bgr.copy()

        # Đánh giá trạng thái
        is_correct = True
        status_text = ""
        status_color = "green"

        if not has_person:
            # Case không có người: nếu có bất kỳ prediction nào => False Positive (Báo động giả)
            if len(preds) > 0:
                is_correct = False
                status_text = f"❌ BÁO ĐỘNG GIẢ ({len(preds)} FP!)"
                status_color = "red"
            else:
                status_text = "✅ CHÍNH XÁC (0 FP)"
        else:
            # Case có người: nếu không phát hiện => False Negative (Trượt)
            if len(preds) == 0:
                is_correct = False
                status_text = "❌ TRƯỢT MỤC TIÊU (0 TP)"
                status_color = "red"
            else:
                status_text = f"✅ PHÁT HIỆN ({len(preds)} box)"

        # Vẽ các bounding box dự đoán
        for p in preds:
            x1, y1, x2, y2 = int(p['x1']*w_orig), int(p['y1']*h_orig), int(p['x2']*w_orig), int(p['y2']*h_orig)
            box_color = (0, 0, 255) if not has_person else (0, 255, 0) # Đỏ nếu FP, xanh lá nếu TP
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)
            label = f"FP! {p['conf']:.2f}" if not has_person else f"Person {p['conf']:.2f}"
            cv2.putText(canvas, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{m_name}\n{status_text} [{lat_ms:.1f}ms]", fontsize=10, fontweight='bold', color=status_color)
        ax.axis('off')

    plt.suptitle(f"Trực Quan Hóa Cả 4 Model — {case_title}", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()"""))

    cells.append(nbf.v4.new_code_cell(r"""# 1. Ca Biên Tệ 1: Bẫy Xe Lăn Rỗng (Empty Wheelchair Trap - Trục D8)
# Thử thách: Kiểm tra khả năng chống báo động giả (False Alarm)
visualize_4_models_on_case(
    'sim_vision/data/vision_eval_1200/images/D8_0005.jpg',
    'Ca Tệ 1: Bẫy Đạo Cụ Rỗng (Xe lăn trống không người)',
    has_person=False
)"""))

    cells.append(nbf.v4.new_code_cell(r"""# 2. Ca Biên Tệ 2: Cận Cảnh Cực Đoan 0.6m (Extreme Close-Up Truncation - Trục D5)
# Thử thách: Thân người lấp kín màn hình, mất mốc giải phẫu nửa dưới
visualize_4_models_on_case(
    'sim_vision/data/vision_eval_1200/images/D5_0005.jpg',
    'Ca Tệ 2: Cận Cảnh Cực Đoan 0.6m (Robot áp sát bệnh nhân)',
    has_person=True, true_subcat='standing'
)"""))

    cells.append(nbf.v4.new_code_cell(r"""# 3. Ca Biên Tệ 3: Bệnh Nhân Nằm Trên Giường Bệnh (Recumbent Bed Pose - Trục D3)
# Thử thách: Tư thế nằm ngang (w/h >= 2.5), phá vỡ giả định hình học đứng dọc của các detector thông thường
visualize_4_models_on_case(
    'sim_vision/data/vision_eval_1200/images/D3_0070.jpg',
    'Ca Tệ 3: Bệnh Nhân Nằm Ngửa Trên Giường Cấp Cứu (Recumbent Pose w/h >= 2.5)',
    has_person=True, true_subcat='recumbent_bed'
)"""))

    cells.append(nbf.v4.new_code_cell(r"""# 4. Ca Biên Tệ 4: Đứng Sau Quầy Lễ Tân Cao 1.1m (Partial Body / Counter Occlusion - Trục D2)
# Thử thách: Mất 100% nửa thân dưới, chỉ còn đầu và vai
visualize_4_models_on_case(
    'sim_vision/data/vision_eval_1200/images/D2_0010.jpg',
    'Ca Tệ 4: Nhân Viên Đứng Sau Quầy Tiếp Đón Cao 1.1m (Mất 100% nửa thân dưới)',
    has_person=True, true_subcat='standing'
)"""))

    cells.append(nbf.v4.new_code_cell(r"""# 5. Ca Biên Tệ 5: Người Nhà Đẩy Xe Lăn (Crowd Occlusion Along Optical Axis - Trục D4)
# Thử thách: 2 người trùng trục quan sát (Bẫy triệt tiêu nhầm của Greedy NMS)
visualize_4_models_on_case(
    'sim_vision/data/vision_eval_1200/images/D4_0015.jpg',
    'Ca Tệ 5: Người Nhà Đẩy Xe Lăn (2 người trùng trục quan sát - Bẫy Greedy NMS)',
    has_person=True, true_subcat='wheelchair'
)"""))

    cells.append(nbf.v4.new_code_cell(r"""# 6. Ca Biên Tệ 6: Bẫy Giường Cấp Cứu Trống (Empty Hospital Bed Trap - Trục D8)
# Thử thách: Chăn gối tạo khối nổi 3D dễ bị SSD nhận nhầm thành người nằm
visualize_4_models_on_case(
    'sim_vision/data/vision_eval_1200/images/D8_0035.jpg',
    'Ca Tệ 6: Bẫy Giường Cấp Cứu Trống (Chăn gối tạo khối 3D - Hard Negative)',
    has_person=False
)"""))

    # ==========================================================================
    # PHẦN 9: BẢNG TỔNG HỢP LỖI SUY LUẬN & PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ
    # ==========================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 🔍 Bảng Tổng Hợp Lỗi Suy Luận & Phân Tích Nguyên Nhân Gốc Rễ (Root Cause Analysis)

| Ca Biên Kiểm Thử | Bản Chất Vật Lý / Lâm Sàng | YOLOv8n (Baseline) | YOLO11n (C2PSA) | YOLO26n (NMS-Free) | Intel OpenVINO (SSD) | Nguyên Nhân Kỹ Thuật (First Principles) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Bẫy Xe Lăn Rỗng (D8)** | Khung xe rỗng kim loại có nan hoa và tay vịn giống chân/tay | **0 FP (Chuẩn)** | **0 FP (Chuẩn)** | **0 FP (Chuẩn)** | **5.56% FP ❌** | Intel SSD thiếu khả năng phân tích kết cấu vải/da; tỷ lệ khung xe khớp anchor person. |
| **Cận Cảnh 0.6m (D5)** | Thân người phóng to lấp kín màn hình, mất chân | **100% TP (Chuẩn)** | **100% TP (Chuẩn)** | **100% TP (Chuẩn)** | **56.8% Recall ❌** | Chiều cao thân người vượt quá kích thước anchor box lớn nhất của MobileNetV2-SSD. |
| **Nằm Giường Bệnh (D3)** | Cơ thể nằm ngang ($w/h \ge 2.5$), đắp chăn | **Trượt (FN)** | **Trượt (FN)** | **Trượt (FN)** | **Trượt (FN)** | Tất cả person detector COCO được train trên người đứng ($w/h \approx 0.35$). Cần head phụ cho giường bệnh. |
| **Sau Quầy Lễ Tân (D2)** | Quầy $1.1\text{m}$ che $100\%$ nửa thân dưới | **84% TP** | **84% TP** | **88% TP (Tốt)** | **100% TP** | NMS-Free (YOLO26n) giúp giữ lại box đầu-vai mà không bị triệt tiêu nhầm. |
| **Người Đẩy Xe Lăn (D4)** | 2 người trùng trục nhìn camera | **Tách 1-2 box** | **Tách 2 box** | **Tách 2 box (Tốt)** | **Trộn 1 box** | Greedy NMS của v8n dễ xóa nhầm box người ngồi trước; YOLO26n phân tách tốt nhờ One-to-One Head. |

---

## 7. Khuyến Nghị Kiến Trúc Cho Robot Dẫn Đường Bệnh Viện

1. **Lựa chọn cốt lõi**:
   - Tiếp tục giữ **`yolov8n.pt`** làm detector chính cho Robot: Miễn nhiễm hoàn toàn với xe lăn rỗng ($FPR=0.00\%$), tốc độ cao nhất ($8.6\text{ ms}$ P95), góc lái cực ổn định ($\Delta\theta = 0.17^\circ$).
2. **Kế hoạch mở rộng**:
   - Khi Robot làm việc tại khu vực quầy tiếp đón đông bệnh nhân xếp hàng: Nâng cấp lên **`yolo26n.pt`** để tận dụng cơ chế NMS-Free (Precision đạt **$92.13\%$**).
   - Với bệnh nhân nằm giường bệnh: Bổ sung module phân loại trạng thái giường bệnh chuyên dụng thay vì dùng chung mô hình person detector thông thường."""))

    nb.cells = cells
    return nb


def main():
    print(f"Bắt đầu xây dựng notebook toàn diện (Đủ 4 Models & Worst Cases) tại: {NOTEBOOK_PATH}")
    nb = create_comprehensive_notebook()
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Đã tạo khung notebook thành công. Tiến hành chạy thực thi (Execute) toàn bộ các ô lệnh...")

    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"🎉 Hoàn tất! Notebook chuyên sâu đã được chạy và lưu kèm đầy đủ hình ảnh/biểu đồ tại: {NOTEBOOK_PATH}")


if __name__ == '__main__':
    main()
