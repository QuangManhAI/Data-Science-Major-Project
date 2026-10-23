#!/usr/bin/env python3
"""create_dataset_overview_samples.py

Tạo các ảnh overview đại diện chuẩn nhất cho toàn bộ hệ sinh thái dataset:
1. Grid tổng quan 8 Trục Không Gian Baseline (D1 - D8): baseline_8_axes_overview.png
2. Grid tổng quan 6 Nhóm Lâm Sàng Mở Rộng (N1 - N6, 36 ca kiểm thử): clinical_36_cases_overview.png
3. Panel 6 ca kịch bản tiêu biểu nhất (Key Representative Cases): key_clinical_scenarios_grid.png
4. Copy các ảnh đại diện vào sim_vision/data/samples/overview_cases/
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "sim_vision/data/samples"
OVERVIEW_DIR = SAMPLES_DIR / "overview_cases"
OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)

DATA_1200_DIR = REPO_ROOT / "sim_vision/data/vision_eval_1200/images"
DATA_10800_DIR = REPO_ROOT / "sim_vision/data/vision_eval_extended_10800/images"

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

# ------------------------------------------------------------------------------
# 1. TẠO GRID 8 TRỤC BASELINE (D1 - D8)
# ------------------------------------------------------------------------------
d_axes = [
    ("D1_0015.jpg", "D1: Vật Cản Kiến Trúc (Doorway/Trụ)"),
    ("D2_0015.jpg", "D2: Đạo Cụ Y Tế Che Khuất (Xe Tiêm/Cột)"),
    ("D3_0015.jpg", "D3: Dáng Đi Lâm Sàng (Xe Lăn/Nạng/Giường)"),
    ("D4_0015.jpg", "D4: Động Học Đám Đông (Cắt Ngang/Xếp Hàng)"),
    ("D5_0015.jpg", "D5: Biến Dạng Cự Ly (0.6m - 10.0m)"),
    ("D6_0015.jpg", "D6: Góc Quay Đa Hướng (0° - 180°)"),
    ("D7_0015.jpg", "D7: Nhiễu Ánh Sáng PBR (Chói/Mờ)"),
    ("D8_0015.jpg", "D8: Bẫy Đạo Cụ Rỗng (Xe Lăn/Giường Trống)")
]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for idx, (img_name, label) in enumerate(d_axes):
    img_p = DATA_1200_DIR / img_name
    ax = axes[idx]
    if img_p.exists():
        im = cv2.imread(str(img_p))
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im_rgb)
        # Lưu bản copy lẻ
        cv2.imwrite(str(OVERVIEW_DIR / f"baseline_{img_name}"), im)
    else:
        ax.set_facecolor("#333333")
        ax.text(0.5, 0.5, "Image Missing", ha="center", color="white")
    ax.set_title(label, fontsize=11, weight="bold", pad=8)
    ax.axis("off")

plt.suptitle("🏥 BỘ DATASET 01: 8 TRỤC KHÔNG GIAN BASELINE (vision_eval_1200 — 1,200 FRAMES)", fontsize=14, weight="bold", y=0.98)
plt.tight_layout()
out_baseline_grid = SAMPLES_DIR / "dataset_01_baseline_8_axes_overview.png"
plt.savefig(out_baseline_grid, dpi=160, bbox_inches="tight")
plt.close()
print(f"✅ Đã tạo: {out_baseline_grid}")


# ------------------------------------------------------------------------------
# 2. TẠO GRID 6 NHÓM LÂM SÀNG TIÊU BIỂU (N1 - N6)
# ------------------------------------------------------------------------------
key_cases = [
    ("N1_1_0015.jpg", "N1.1: Bác Sĩ Blouse Ngụy Trang Tường (ΔE=8.4)", "#1e3a8a"),
    ("N1.3: Cụ Bà Tóc Bạc Áo Hoa (Elder F09)", "N1_3_0015.jpg", "#1e3a8a"),
    ("N2_2_0015.jpg", "N2.2: Bệnh Nhân Ngồi Gập Người Ho 45°", "#065f46"),
    ("N2_4_0015.jpg", "N2.4: Bác Sĩ Cúi Đầu Đọc Bệnh Án Clipboard", "#065f46"),
    ("N3_1_0015.jpg", "N3.1: Che Khuất Cột Truyền Dịch Inox (Slender)", "#0f766e"),
    ("N3_3_0015.jpg", "N3.3: Bệnh Nhân Đứng Bấm Kiosk Lấy Số (70% Occ)", "#0f766e"),
    ("N4_1_0015.jpg", "N4.1: Bẫy Cột Truyền Dịch Rỗng (False Positive)", "#991b1b"),
    ("N4_7_0015.jpg", "N4.7: Bẫy Standee In Hình Y Tế 2D (Poster Trap)", "#991b1b"),
    ("N5_2_0015.jpg", "N5.2: Bước Né Ngang Nhường Đường (Pardon 0.55m)", "#854d0e"),
    ("N5_4_0015.jpg", "N5.4: Điều Dưỡng Rẽ Nhanh Từ Góc Khuất", "#854d0e"),
    ("N6_1_0015.jpg", "N6.1: Hai Cụ Già Ngồi Kề Ghế Băng (NMS Trap)", "#581c87"),
    ("N6_3_0015.jpg", "N6.3: Hàng Chờ 3 Người Trước Kiosk (Axial Chain)", "#581c87"),
]

fig, axes = plt.subplots(3, 4, figsize=(18, 13))
axes = axes.flatten()

for idx, (img_name, label, color) in enumerate([
    ("N1_1_0015.jpg", "N1.1: Bác Sĩ Blouse Ngụy Trang Tường", "#1e3a8a"),
    ("N1_3_0015.jpg", "N1.3: Cụ Bà Tóc Bạc Áo Hoa (F09)", "#1e3a8a"),
    ("N2_2_0015.jpg", "N2.2: Ngồi Gập Người Ho Khạc 45°", "#065f46"),
    ("N2_4_0015.jpg", "N2.4: Bác Sĩ Đọc Kẹp Bệnh Án", "#065f46"),
    ("N3_1_0015.jpg", "N3.1: Che Khuất Cột Truyền Dịch Inox", "#0f766e"),
    ("N3_3_0015.jpg", "N3.3: Bệnh Nhân Đứng Bấm Kiosk Lấy Số", "#0f766e"),
    ("N4_1_0015.jpg", "N4.1: Bẫy Cột Truyền Dịch Rỗng (FP Trap)", "#991b1b"),
    ("N4_7_0015.jpg", "N4.7: Bẫy Standee 2D Poster Trap", "#991b1b"),
    ("N5_2_0015.jpg", "N5.2: Bước Né Ngang Pardon Sidestep", "#854d0e"),
    ("N5_4_0015.jpg", "N5.4: Y Tá Rẽ Nhanh Từ Góc Khuất", "#854d0e"),
    ("N6_1_0015.jpg", "N6.1: Hai Cụ Già Ngồi Kề Nhau (NMS Trap)", "#581c87"),
    ("N6_3_0015.jpg", "N6.3: Hàng Chờ 3 Bệnh Nhân Kiosk", "#581c87")
]):
    img_p = DATA_10800_DIR / img_name
    ax = axes[idx]
    if img_p.exists():
        im = cv2.imread(str(img_p))
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im_rgb)
        cv2.imwrite(str(OVERVIEW_DIR / f"extended_{img_name}"), im)
    else:
        ax.set_facecolor("#333333")
        ax.text(0.5, 0.5, "Image Missing", ha="center", color="white")
    ax.set_title(label, fontsize=10.5, weight="bold", color=color, pad=8)
    ax.axis("off")

plt.suptitle("🏥 BỘ DATASET 02: 12 CA LÂM SÀNG KHẮC NGHIỆT ĐẠI DIỆN (vision_eval_extended_10800)", fontsize=14, weight="bold", y=0.98)
plt.tight_layout()
out_extended_grid = SAMPLES_DIR / "dataset_02_clinical_12_representative_overview.png"
plt.savefig(out_extended_grid, dpi=160, bbox_inches="tight")
plt.close()
print(f"✅ Đã tạo: {out_extended_grid}")


# ------------------------------------------------------------------------------
# 3. TẠO POSTER LƯỚI 6x6 TỔNG QUAN TOÀN BỘ 36 CA KIỂM THỬ (N1.1 -> N6.5)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(6, 6, figsize=(22, 20))
axes = axes.flatten()

case_ids = [
    "N1_1", "N1_2", "N1_3", "N1_4", "N1_5", "N1_6",
    "N1_7", "N2_1", "N2_2", "N2_3", "N2_4", "N2_5",
    "N3_1", "N3_2", "N3_3", "N3_4", "N3_5", "N3_6",
    "N3_7", "N4_1", "N4_2", "N4_3", "N4_4", "N4_5",
    "N4_6", "N4_7", "N4_8", "N5_1", "N5_2", "N5_3",
    "N5_4", "N6_1", "N6_2", "N6_3", "N6_4", "N6_5"
]

case_titles = [
    "N1.1 Blouse Trắng", "N1.2 Scrub Xanh", "N1.3 Cụ Bà F09", "N1.4 Cụ Ông Gù", "N1.5 Cụ Ông Sọc", "N1.6 Áo Khoác Dài",
    "N1.7 Nam Đậm", "N2.1 Ghế Băng", "N2.2 Ho Khạc 45°", "N2.3 Mỏi Mệt", "N2.4 Đọc Bệnh Án", "N2.5 Bắt Chéo Chân",
    "N3.1 Cột IV", "N3.2 Xe Monitor", "N3.3 Kiosk 1.55m", "N3.4 Cây Nước", "N3.5 Khử Khuẩn", "N3.6 Xe Cáng Đẩy",
    "N3.7 Standee Mép", "N4.1 Bẫy Cột IV", "N4.2 Bẫy Cột Rửa", "N4.3 Bẫy Kiosk", "N4.4 Bẫy Monitor", "N4.5 Bẫy Cáng Nệm",
    "N4.6 Bẫy Cây Nước", "N4.7 Bẫy Poster 2D", "N4.8 Bẫy Màn LED", "N5.1 Đi Tuần Tra", "N5.2 Né Pardon", "N5.3 Xoay Cửa 90°",
    "N5.4 Rẽ Góc Khuất", "N6.1 Hai Cụ Già Kề", "N6.2 Bác Sĩ - BN", "N6.3 Hàng Chờ 3N", "N6.4 Thảo Luận 180°", "N6.5 Dìu Đi Chậm"
]

for idx, cid in enumerate(case_ids):
    img_p = DATA_10800_DIR / f"{cid}_0015.jpg"
    ax = axes[idx]
    if img_p.exists():
        im = cv2.imread(str(img_p))
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im_rgb)
    else:
        ax.set_facecolor("#222222")
    ax.set_title(case_titles[idx], fontsize=8.5, weight="bold", pad=4)
    ax.axis("off")

plt.suptitle("🏥 POSTER MA TRẬN TOÀN BỘ 36 CA KIỂM THỬ LÂM SÀNG (Mỗi ca 300 frames — Tổng 10,800 frames)", fontsize=16, weight="bold", y=0.99)
plt.tight_layout()
out_matrix_36 = SAMPLES_DIR / "dataset_02_full_36_clinical_cases_matrix.png"
plt.savefig(out_matrix_36, dpi=160, bbox_inches="tight")
plt.close()
print(f"✅ Đã tạo: {out_matrix_36}")
