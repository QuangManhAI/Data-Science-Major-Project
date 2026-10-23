#!/usr/bin/env python3
"""reorganize_samples_and_generate_all_cases.py

Dọn dẹp triệt để thư mục sim_vision/data/samples/ và tái cấu trúc toàn diện:
1. Trích xuất toàn bộ 48 ca kịch bản của Dataset 1 (vision_eval_1200) -> dataset_1_baseline_48cases/
2. Trích xuất toàn bộ 36 ca lâm sàng của Dataset 2 (vision_eval_extended_10800) -> dataset_2_clinical_36cases/
3. Kết xuất 3 tấm Master Posters trực quan độ phân giải cao:
   - 01_dataset_1_all_48_cases_poster.png (6 hàng x 8 cột = 48 ô)
   - 02_dataset_2_all_36_clinical_cases_poster.png (6 hàng x 6 cột = 36 ô)
   - 03_master_unified_84_cases_poster.png (7 hàng x 12 cột = 84 ô hợp nhất toàn bộ)
4. Xuất sim_vision/data/samples/README.md tra cứu chi tiết 84 ca kiểm thử.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_ROOT = REPO_ROOT / "sim_vision/data/samples"
DATA1_IMG_DIR = REPO_ROOT / "sim_vision/data/vision_eval_1200/images"
DATA2_IMG_DIR = REPO_ROOT / "sim_vision/data/vision_eval_extended_10800/images"

DIR_D1 = SAMPLES_ROOT / "dataset_1_baseline_48cases"
DIR_D2 = SAMPLES_ROOT / "dataset_2_clinical_36cases"
DIR_POSTERS = SAMPLES_ROOT / "master_posters"

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

# ==============================================================================
# BƯỚC 1: DỌN DẸP SẠCH THƯ MỤC SAMPLES
# ==============================================================================
print("1. Đang dọn dẹp sạch sẽ thư mục samples...")
if SAMPLES_ROOT.exists():
    shutil.rmtree(SAMPLES_ROOT)

SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
DIR_D1.mkdir(parents=True, exist_ok=True)
DIR_D2.mkdir(parents=True, exist_ok=True)
DIR_POSTERS.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DANH MỤC 48 CA CỦA DATASET 1 (vision_eval_1200)
# ==============================================================================
DATASET_1_CASES = [
    # D1: Kiến Trúc
    ("C01", "D1", 12, "T_Junction_Wall", "C01: Khuất Tường Góc Ngoặt"),
    ("C02", "D1", 37, "Door_Frame_Clinic", "C02: Cắt Dọc Khung Cửa Phòng"),
    ("C03", "D1", 62, "Sliding_Door_35pct", "C03: Khe Cửa Trượt Mở Dở 35%"),
    ("C04", "D1", 87, "Elevator_Alcove", "C04: Hốc Buồng Thang Máy"),
    ("C05", "D1", 112, "Lobby_Round_Column", "C05: Nấp Sau Trụ Cột Sảnh"),
    ("C06", "D1", 137, "Door_Ajar_45deg", "C06: Cánh Cửa Mở Hé 45°"),
    # D2: Nội Thất & Đạo Cụ
    ("C07", "D2", 12, "Reception_Counter", "C07: Quầy Lễ Tân Cao 1.1m"),
    ("C08", "D2", 37, "Doctor_Desk_Screen", "C08: Ngồi Sau Bàn Khám & Màn Hình"),
    ("C09", "D2", 62, "Privacy_Screen_Curtain", "C09: Khuất Sau Vách Rèm Y Tế"),
    ("C10", "D2", 87, "Medical_Trolley_Inox", "C10: Khuất Sau Xe Đẩy Thuốc 2 Tầng"),
    ("C11", "D2", 112, "IV_Stand_Chest_Block", "C11: Cọc Truyền Dịch Che Ngực"),
    ("C12", "D2", 137, "Waiting_Bench_Back", "C12: Ngồi Sau Hàng Ghế Chờ Inox"),
    # D3: Tư Thế Lâm Sàng
    ("C13", "D3", 12, "Wheelchair_Frontal", "C13: Xe Lăn Nhìn Trực Diện (0°)"),
    ("C14", "D3", 37, "Wheelchair_Oblique_45", "C14: Xe Lăn Nhìn Chéo Góc 45°"),
    ("C15", "D3", 62, "Wheelchair_Side_90", "C15: Xe Lăn Nhìn Ngang Hông 90°"),
    ("C16", "D3", 87, "Elderly_Cane_Kyphosis", "C16: Người Già Còng Lưng Chống Gậy"),
    ("C17", "D3", 112, "Bed_Patient_Recumbent", "C17: Bệnh Nhân Nằm Giường Cấp Cứu"),
    ("C18", "D3", 137, "Bent_Over_90deg", "C18: Cúi Gập 90° Nhặt Sổ Khám"),
    # D4: Đám Đông & NMS
    ("C19", "D4", 12, "Wheelchair_Pushed_Rear", "C19: Đẩy Xe Lăn Sát Sau (NMS Trap)"),
    ("C20", "D4", 37, "Parent_Child_Holding", "C20: Phụ Huynh Dắt Bệnh Nhi"),
    ("C21", "D4", 62, "Queue_3_Axial_Sequence", "C21: Hàng Dọc 3 Người Nối Đuôi"),
    ("C22", "D4", 87, "Cross_Walk_Intersect", "C22: Cắt Mặt Ngược Chiều 2.5m"),
    ("C23", "D4", 112, "Crowd_Signboard_Cluster", "C23: Cụm 4 Người Trước Bảng Chỉ Dẫn"),
    ("C24", "D4", 137, "Guiding_Limping_Couple", "C24: Dìu Người Bệnh Đi Khập Khiễng"),
    # D5: Cự Ly & Cắt Biên
    ("C25", "D5", 12, "Extreme_Close_0.6m", "C25: Cận Cảnh Cực Đoan (0.6m)"),
    ("C26", "D5", 37, "Close_Safety_0.9m", "C26: Cận Cảnh Vùng An Toàn (0.9m)"),
    ("C27", "D5", 62, "Mid_Range_2.5m", "C27: Cự Ly Trung Bình (2.5m)"),
    ("C28", "D5", 87, "Far_Range_6.0m", "C28: Cự Ly Xa Dọc Hành Lang (6.0m)"),
    ("C29", "D5", 112, "Edge_Left_Truncated", "C29: Cắt Mép Trái Khung Hình"),
    ("C30", "D5", 137, "Edge_Right_Truncated", "C30: Cắt Mép Phải Khung Hình"),
    # D6: Động Học & Xoay Hướng
    ("C31", "D6", 12, "Walk_Away_180deg", "C31: Quay Lưng Bước Đi (180°)"),
    ("C32", "D6", 37, "Turn_Oblique_135deg", "C32: Góc Nhìn Chéo Lưng (135°)"),
    ("C33", "D6", 62, "Lateral_Profile_90deg", "C33: Xoay Người Tại Chỗ (90°)"),
    ("C34", "D6", 87, "Walking_Backwards", "C34: Đi Giật Lùi Từng Bước"),
    ("C35", "D6", 112, "Fast_Paced_Walk", "C35: Bước Đi Nhanh (1.5 m/s)"),
    ("C36", "D6", 137, "Stationary_Talking", "C36: Đứng Yên Nói Chuyện Liên Tục"),
    # D7: Ánh Sáng PBR
    ("C37", "D7", 12, "Backlight_Lobby_Glass", "C37: Ngược Sáng Cửa Kính Sảnh"),
    ("C38", "D7", 37, "Floor_Specular_Reflection", "C38: Phản Chiếu Sàn Men Bóng"),
    ("C39", "D7", 62, "Long_Oblique_Shadow", "C39: Bóng Đổ Xiên Dài 2m"),
    ("C40", "D7", 87, "Fluorescent_Overexposure", "C40: Dư Sáng Huỳnh Quang Bệt Màu"),
    ("C41", "D7", 112, "Emergency_Night_Dim", "C41: Đèn Trực Đêm Thiếu Sáng"),
    ("C42", "D7", 137, "Flickering_Light_Spike", "C42: Ánh Sáng Lập Lòe Chớp Tắt"),
    # D8: Bẫy Đạo Cụ Rỗng (False Positives)
    ("C43", "D8", 12, "Empty_Wheelchair", "C43: Xe Lăn Trống Không Người"),
    ("C44", "D8", 37, "Empty_Hospital_Bed", "C44: Giường Bệnh Cấp Cứu Trống"),
    ("C45", "D8", 62, "IV_Stand_Coat_Scarecrow", "C45: Cọc Truyền Treo Áo Khoác"),
    ("C46", "D8", 87, "Doctor_Poster_Wall", "C46: Poster Bác Sĩ 2D Dán Tường"),
    ("C47", "D8", 112, "Empty_Touch_Kiosk", "C47: Kiosk Tự Phục Vụ 1.6m"),
    ("C48", "D8", 137, "Fire_Extinguisher_Bin", "C48: Cụm Bình Chữa Cháy & Thùng Rác")
]

# ==============================================================================
# DANH MỤC 36 CA CỦA DATASET 2 (vision_eval_extended_10800)
# ==============================================================================
DATASET_2_CASES = [
    # N1: Nhân Khẩu & Màu Áo
    ("N1.1", "N1_1", 15, "Doctor_White_Blouse_Wall", "N1.1: Bác Sĩ Blouse Trắng Ngụy Trang (ΔE=8.4)"),
    ("N1.2", "N1_2", 15, "Nurse_Teal_Scrub_Curtain", "N1.2: Y Tá Scrub Xanh Trùng Rèm Y Tế"),
    ("N1.3", "N1_3", 15, "Elderly_Female_F09_GrayHair", "N1.3: Cụ Bà Tóc Bạc Áo Hoa (Elder F09)"),
    ("N1.4", "N1_4", 15, "Elderly_Male_M03_Kyphosis", "N1.4: Cụ Ông Lưng Còng Nghiêng 35°"),
    ("N1.5", "N1_5", 15, "Elderly_Male_M14_Striped", "N1.5: Cụ Ông Áo Kẻ Sọc Nhiễu Tần Số"),
    ("N1.6", "N1_6", 15, "Female_Patient_Long_Coat", "N1.6: Bệnh Nhân Nữ Áo Khoác Dài Phủ Chân"),
    ("N1.7", "N1_7", 15, "Male_Patient_Heavy_Build", "N1.7: Bệnh Nhân Nam Thể Hình Đậm Vai Rộng"),
    # N2: Tư Thế Lâm Sàng & Ghế
    ("N2.1", "N2_1", 15, "Waiting_Bench_Seated", "N2.1: Ngồi Ghế Băng Dài Tựa Lưng"),
    ("N2.2", "N2_2", 15, "Seated_Cough_45deg", "N2.2: Ngồi Gập Người Ho Khạc 45°"),
    ("N2.3", "N2_3", 15, "Seated_Exhausted_Limp", "N2.3: Ngồi Buông Thõng Mỏi Mệt"),
    ("N2.4", "N2_4", 15, "Doctor_Clipboard_Reading", "N2.4: Bác Sĩ Cúi Đầu Đọc Bệnh Án"),
    ("N2.5", "N2_5", 15, "Elderly_Crossed_Legs", "N2.5: Cụ Bà Ngồi Bắt Chéo Chân Chữ Ngũ"),
    # N3: Che Khuất Đạo Cụ Y Tế
    ("N3.1", "N3_1", 15, "IV_Stand_Slender_Pole", "N3.1: Che Cột Truyền Dịch Inox Mảnh"),
    ("N3.2", "N3_2", 15, "Vital_Monitor_Cart", "N3.2: Điều Dưỡng Sau Xe Đẩy Monitor 1.2m"),
    ("N3.3", "N3_3", 15, "Queue_Kiosk_Frontal", "N3.3: Bệnh Nhân Bấm Kiosk 1.55m (70% Occ)"),
    ("N3.4", "N3_4", 15, "Water_Dispenser_Flank", "N3.4: Bệnh Nhân Lấy Nước Cây Nóng Lạnh"),
    ("N3.5", "N3_5", 15, "Sanitizer_Stand_Arms", "N3.5: Rửa Tay Cột Khử Khuẩn (Tay Duỗi)"),
    ("N3.6", "N3_6", 15, "Stretcher_Cart_Mobile", "N3.6: Hộ Lý Đẩy Xe Cáng Cấp Cứu 2.1m"),
    ("N3.7", "N3_7", 15, "Info_Standee_Edge", "N3.7: Bệnh Nhân Đứng Sát Mép Standee 1.8m"),
    # N4: Bẫy Đạo Cụ Rỗng (False Positives)
    ("N4.1", "N4_1", 15, "Empty_IV_Stand_Trap", "N4.1: Bẫy Cột Truyền Dịch Inox Rỗng"),
    ("N4.2", "N4_2", 15, "Empty_Sanitizer_Trap", "N4.2: Bẫy Cột Khử Khuẩn Tự Động Rỗng"),
    ("N4.3", "N4_3", 15, "Empty_Queue_Kiosk_Trap", "N4.3: Bẫy Kiosk Bấm Số Khám Rỗng"),
    ("N4.4", "N4_4", 15, "Empty_Vital_Monitor_Trap", "N4.4: Bẫy Xe Đẩy Máy Monitor Rỗng"),
    ("N4.5", "N4_5", 15, "Empty_Stretcher_Bed_Trap", "N4.5: Bẫy Xe Cáng Cấp Cứu Nệm Trắng Rỗng"),
    ("N4.6", "N4_6", 15, "Empty_Water_Dispenser_Trap", "N4.6: Bẫy Cây Nước Uống Đơn Lẻ"),
    ("N4.7", "N4_7", 15, "Standee_2D_Poster_Trap", "N4.7: Bẫy Standee 2D In Hình Y Tế (Poster Trap)"),
    ("N4.8", "N4_8", 15, "Empty_Queue_Display_Trap", "N4.8: Bẫy Màn Hình LED Gọi Số Treo Tường"),
    # N5: Động Học & Né Ngang
    ("N5.1", "N5_1", 15, "Patrol_Looming_Reverse", "N5.1: Đi Tuần Tra Đối Đầu (v_rel=1.08 m/s)"),
    ("N5.2", "N5_2", 15, "Pardon_Sidestep_0.55m", "N5.2: Bước Né Ngang Nhường Đường (0.55m)"),
    ("N5.3", "N5_3", 15, "Doorway_Turn_90deg", "N5.3: Chuyển Hướng 90° Qua Ngưỡng Cửa"),
    ("N5.4", "N5_4", 15, "Corner_Sudden_Emergence", "N5.4: Điều Dưỡng Rẽ Bất Ngờ Từ Góc Khuất"),
    # N6: Mật Độ Xã Hội & Phân Rã
    ("N6.1", "N6_1", 15, "Two_Elderly_Seated_Bench", "N6.1: Hai Cụ Già Ngồi Kề Nhau (NMS Trap)"),
    ("N6.2", "N6_2", 15, "Doctor_Standing_Patient_Seated", "N6.2: Cặp Đôi Bác Sĩ Đứng - Bệnh Nhân Ngồi"),
    ("N6.3", "N6_3", 15, "Queue_3_Patients_Sequence", "N6.3: Hàng Chờ 3 Bệnh Nhân Trước Kiosk"),
    ("N6.4", "N6_4", 15, "Consultation_180deg_Facing", "N6.4: Thảo Luận: 180° Quay Lưng vs 0° Mặt"),
    ("N6.5", "N6_5", 15, "Guided_Walking_Elderly_Pair", "N6.5: Bệnh Nhân Được Người Nhà Dìu Đi Chậm")
]

# ==============================================================================
# BƯỚC 2: SAO CHÉP VÀ ĐẶT TÊN CHUẨN XÁC TỪNG FILE ĐẠI DIỆN
# ==============================================================================
print("2. Trích xuất toàn bộ 48 ca Dataset 1 vào dataset_1_baseline_48cases/...")
d1_images = []
for cid, axis, frame_idx, slug, title in DATASET_1_CASES:
    src_name = f"{axis}_{frame_idx:04d}.jpg"
    src_p = DATA1_IMG_DIR / src_name
    dst_name = f"{cid}_{slug}.jpg"
    dst_p = DIR_D1 / dst_name
    if src_p.exists():
        im = cv2.imread(str(src_p))
        cv2.imwrite(str(dst_p), im)
        d1_images.append((dst_p, title))
    else:
        print(f"⚠️ Thiếu file: {src_p}")

print(f"✅ Đã trích xuất đủ {len(d1_images)}/48 ca Dataset 1.")

print("3. Trích xuất toàn bộ 36 ca Dataset 2 vào dataset_2_clinical_36cases/...")
d2_images = []
for cid_dot, prefix, frame_idx, slug, title in DATASET_2_CASES:
    src_name = f"{prefix}_{frame_idx:04d}.jpg"
    src_p = DATA2_IMG_DIR / src_name
    dst_name = f"{prefix}_{slug}.jpg"
    dst_p = DIR_D2 / dst_name
    if src_p.exists():
        im = cv2.imread(str(src_p))
        cv2.imwrite(str(dst_p), im)
        d2_images.append((dst_p, title))
    else:
        print(f"⚠️ Thiếu file: {src_p}")

print(f"✅ Đã trích xuất đủ {len(d2_images)}/36 ca Dataset 2.")

# ==============================================================================
# BƯỚC 3: KẾT XUẤT 3 MASTER POSTERS SIÊU ĐẸP
# ==============================================================================
print("4. Đang tạo Poster 1: Toàn bộ 48 ca Dataset 1 (6 hàng x 8 cột)...")
fig, axes = plt.subplots(6, 8, figsize=(26, 18))
axes = axes.flatten()

for idx, (img_path, title) in enumerate(d1_images):
    ax = axes[idx]
    im = cv2.imread(str(img_path))
    if im is not None:
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im_rgb)
    ax.set_title(title, fontsize=8, weight="bold", pad=3, color="#1e293b")
    ax.axis("off")

plt.suptitle("🏥 BỘ DATASET 01: TOÀN BỘ 48 CA KỊCH BẢN BASELINE (C01 - C48 | vision_eval_1200)", fontsize=16, weight="bold", y=0.99)
plt.tight_layout()
poster1_path = DIR_POSTERS / "01_dataset_1_all_48_cases_poster.png"
plt.savefig(poster1_path, dpi=160, bbox_inches="tight")
plt.close()
print(f"✅ Đã lưu Poster 1: {poster1_path}")

print("5. Đang tạo Poster 2: Toàn bộ 36 ca Dataset 2 (6 hàng x 6 cột)...")
fig, axes = plt.subplots(6, 6, figsize=(22, 20))
axes = axes.flatten()

for idx, (img_path, title) in enumerate(d2_images):
    ax = axes[idx]
    im = cv2.imread(str(img_path))
    if im is not None:
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im_rgb)
    # Tô màu tiêu đề theo nhóm
    gid = title[:2]
    c_map = {"N1": "#1e3a8a", "N2": "#065f46", "N3": "#0f766e", "N4": "#991b1b", "N5": "#854d0e", "N6": "#581c87"}
    ax.set_title(title, fontsize=8.5, weight="bold", pad=3, color=c_map.get(gid, "black"))
    ax.axis("off")

plt.suptitle("🏥 BỘ DATASET 02: TOÀN BỘ 36 CA LÂM SÀNG MỞ RỘNG (N1.1 - N6.5 | vision_eval_extended_10800)", fontsize=16, weight="bold", y=0.99)
plt.tight_layout()
poster2_path = DIR_POSTERS / "02_dataset_2_all_36_clinical_cases_poster.png"
plt.savefig(poster2_path, dpi=160, bbox_inches="tight")
plt.close()
print(f"✅ Đã lưu Poster 2: {poster2_path}")

print("6. Đang tạo Master Poster 3: Hợp nhất toàn bộ 84 ca của CẢ 2 DATASETS (7 hàng x 12 cột)...")
fig, axes = plt.subplots(7, 12, figsize=(36, 20))
axes = axes.flatten()

all_84_cases = d1_images + d2_images
for idx, (img_path, title) in enumerate(all_84_cases):
    ax = axes[idx]
    im = cv2.imread(str(img_path))
    if im is not None:
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im_rgb)
    ax.set_title(title, fontsize=6.8, weight="bold", pad=2)
    ax.axis("off")

plt.suptitle("🏥 MASTER POSTER: HỢP NHẤT TOÀN BỘ 84 CA KIỂM THỬ THỊ GIÁC (DATASET 1: 48 CASES + DATASET 2: 36 CASES)", fontsize=18, weight="bold", y=0.995)
plt.tight_layout()
poster3_path = DIR_POSTERS / "03_master_unified_84_cases_poster.png"
plt.savefig(poster3_path, dpi=140, bbox_inches="tight")
plt.close()
print(f"✅ Đã lưu Master Poster 3: {poster3_path}")

# ==============================================================================
# BƯỚC 4: XUẤT TÀI LIỆU HƯỚNG DẪN TRA CỨU sim_vision/data/samples/README.md
# ==============================================================================
readme_content = f"""# 📸 THƯ VIỆN ẢNH ĐẠI DIỆN TOÀN BỘ CÁC CA TRONG HỆ SINH THÁI DATASET (SAMPLES)

Thư mục này được tổ chức chuẩn mực nhằm cung cấp các khung hình trực quan tổng quát nhất cho **TOÀN BỘ 84 CA KIỂM THỬ** độc lập trong cả 2 bộ dữ liệu mô phỏng bệnh viện Godot 4.

---

## 🏛️ Cấu Trúc Thư Mục Chuẩn Hóa

```
sim_vision/data/samples/
├── master_posters/                                   # 3 Poster Ma Trận Trực Quan Siêu Rộng
│   ├── 01_dataset_1_all_48_cases_poster.png          # Lưới 6x8: Toàn bộ 48 ca Dataset 1
│   ├── 02_dataset_2_all_36_clinical_cases_poster.png # Lưới 6x6: Toàn bộ 36 ca Dataset 2
│   └── 03_master_unified_84_cases_poster.png         # Lưới 7x12: Hợp nhất toàn bộ 84 ca (1200 + 10800)
│
├── dataset_1_baseline_48cases/                       # 48 Ảnh Rời Chuẩn Hóa (C01 -> C48)
│   ├── C01_T_Junction_Wall.jpg
│   ├── ...
│   └── C48_Fire_Extinguisher_Bin.jpg
│
└── dataset_2_clinical_36cases/                       # 36 Ảnh Rời Chuẩn Hóa (N1.1 -> N6.5)
    ├── N1_1_Doctor_White_Blouse_Wall.jpg
    ├── ...
    └── N6_5_Guided_Walking_Elderly_Pair.jpg
```

---

## 📋 Danh Mục Tra Cứu Toàn Bộ 84 Ca Kiểm Thử

### Phân Hệ 1: Dataset Baseline (vision_eval_1200 — 48 Kịch Bản C01 đến C48)
- **Trục D1 (Kiến trúc)**: C01 (Góc ngoặt T), C02 (Cắt dọc cửa), C03 (Khe cửa trượt 35%), C04 (Hốc thang máy), C05 (Nấp sau trụ cột), C06 (Cửa mở hé 45°).
- **Trục D2 (Nội thất y tế)**: C07 (Quầy lễ tân 1.1m), C08 (Bàn khám & màn hình), C09 (Vách rèm y tế), C10 (Xe tiêm 2 tầng), C11 (Cọc truyền che ngực), C12 (Sau ghế chờ inox).
- **Trục D3 (Tư thế lâm sàng)**: C13 (Xe lăn 0°), C14 (Xe lăn 45°), C15 (Xe lăn 90°), C16 (Cụ già gù chống gậy), C17 (Nằm giường cấp cứu), C18 (Cúi gập 90° nhặt sổ).
- **Trục D4 (Đám đông & NMS)**: C19 (Đẩy xe lăn sau - NMS trap), C20 (Dắt tay trẻ em), C21 (Hàng 3 người), C22 (Cắt mặt 2.5m), C23 (Cụm 4 người), C24 (Dìu bước khập khiễng).
- **Trục D5 (Cự ly & Mép ảnh)**: C25 (Cực cận 0.6m), C26 (0.9m), C27 (2.5m), C28 (Xa 6.0m), C29 (Cắt mép trái), C30 (Cắt mép phải).
- **Trục D6 (Động học)**: C31 (Quay lưng 180°), C32 (Chéo lưng 135°), C33 (Xoay 90°), C34 (Đi lùi), C35 (Đi nhanh 1.5m/s), C36 (Đứng yên nói chuyện).
- **Trục D7 (Ánh sáng PBR)**: C37 (Ngược sáng sảnh), C38 (Phản chiếu sàn gạch bóng), C39 (Bóng đổ dài), C40 (Dư sáng huỳnh quang), C41 (Đèn trực đêm tối), C42 (Ánh sáng chớp tắt).
- **Trục D8 (Bẫy đạo cụ rỗng)**: C43 (Xe lăn trống), C44 (Giường cấp cứu trống), C45 (Cọc truyền treo áo khoác), C46 (Poster 2D dán tường), C47 (Kiosk tự phục vụ), C48 (Cụm bình chữa cháy).

---

### Phân Hệ 2: Dataset Clinical Demographics (vision_eval_extended_10800 — 36 Ca N1.1 đến N6.5)
- **Nhóm N1 (Nhân khẩu & Màu áo)**: N1.1 (Blouse ngụy trang tường kem ΔE=8.4), N1.2 (Scrub xanh trùng rèm), N1.3 (Cụ bà tóc bạc F09), N1.4 (Cụ ông lưng còng M03), N1.5 (Cụ ông áo sọc M14), N1.6 (Áo khoác dài phủ chân), N1.7 (Bệnh nhân nam thể hình đậm).
- **Nhóm N2 (Tư thế & Ngồi ghế)**: N2.1 (Ngồi ghế băng dài), N2.2 (Ngồi gập người ho 45°), N2.3 (Ngồi buông thõng mỏi mệt), N2.4 (Bác sĩ đọc kẹp bệnh án), N2.5 (Cụ bà ngồi bắt chéo chân).
- **Nhóm N3 (Che khuất đạo cụ)**: N3.1 (Cột truyền dịch inox thanh mảnh), N3.2 (Xe đẩy monitor 1.2m), N3.3 (Bấm kiosk 1.55m che 70%), N3.4 (Cây nước nóng lạnh), N3.5 (Rửa tay cột khử khuẩn), N3.6 (Xe cáng cấp cứu 2.1m tiền cảnh), N3.7 (Sát mép standee 1.8m).
- **Nhóm N4 (Bẫy False Positive rỗng)**: N4.1 (Cột IV rỗng), N4.2 (Cột rửa tay rỗng), N4.3 (Kiosk rỗng), N4.4 (Xe monitor rỗng), N4.5 (Xe cáng nệm trắng rỗng), N4.6 (Cây nước rỗng), N4.7 (Standee 2D in hình kích thước thật - Poster Trap), N4.8 (Màn hình LED gọi số).
- **Nhóm N5 (Động học né ngang)**: N5.1 (Đi tuần tra đối đầu v_rel=1.08 m/s), N5.2 (Bước né ngang Pardon Sidestep 0.55m ω=28.1°/s), N5.3 (Xoay người 90° qua cửa), N5.4 (Y tá đi nhanh rẽ từ góc khuất).
- **Nhóm N6 (Mật độ xã hội & Phân rã)**: N6.1 (Hai cụ già ngồi liền kề ghế băng - NMS trap), N6.2 (Bác sĩ đứng cạnh bệnh nhân ngồi), N6.3 (Hàng chờ 3 bệnh nhân kiosk), N6.4 (Thảo luận 180° lưng vs 0° mặt), N6.5 (Dìu bước đi chậm).
"""

with open(SAMPLES_ROOT / "README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"✅ Đã tạo sim_vision/data/samples/README.md.")
"""
"""
