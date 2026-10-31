#!/usr/bin/env python3
"""
scripts/data_pipeline/merge_robot_testsets.py
=============================================
Hợp nhất 2 bộ kiểm thử robot tuần tra:
1. Bộ cũ (Patrol 531 frames - đã được người dùng kiểm duyệt trên CVAT)
2. Bộ mới (Active Inspection 623 frames - đã lọc sạch 100% khung hình trùng lặp)

Tạo thành Bộ Test Chuẩn Toàn Diện (Unified Robot Inspection Benchmark: 1,154 frames)
"""

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

OLD_DIR = REPO_ROOT / "data/vision/dataset/hospital_unified_golden_4901_17class/test"
NEW_DIR = REPO_ROOT / "data/vision/dataset/hospital_robot_inspection_testset"
OUT_DIR = REPO_ROOT / "data/vision/dataset/hospital_robot_unified_benchmark_1154"

def main():
    print("==================================================================")
    print("🤝 TIẾN HÀNH GỘP 2 BỘ KIỂM THỬ ROBOT (1,154 KHUNG HÌNH KHÔNG TRÙNG LẶP)")
    print(f"📁 Nguồn 1 (Bộ cũ CVAT-verified): {OLD_DIR}")
    print(f"📁 Nguồn 2 (Bộ mới Deduplicated):  {NEW_DIR}")
    print(f"📂 Đích:                          {OUT_DIR}")
    print("==================================================================")

    out_img = OUT_DIR / "images"
    out_lbl = OUT_DIR / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # Copy nguồn 1 (Bộ cũ)
    old_imgs = sorted(list((OLD_DIR / "images").glob("*.jpg")))
    print(f"1️⃣ Đang chép {len(old_imgs)} ảnh từ bộ cũ...")
    for img in old_imgs:
        shutil.copy2(img, out_img / img.name)
        lbl = (OLD_DIR / "labels") / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, out_lbl / lbl.name)

    # Copy nguồn 2 (Bộ mới đã lọc sạch)
    new_imgs = sorted(list((NEW_DIR / "images").glob("*.jpg")))
    print(f"2️⃣ Đang chép {len(new_imgs)} ảnh từ bộ mới đã lọc...")
    for img in new_imgs:
        shutil.copy2(img, out_img / img.name)
        lbl = (NEW_DIR / "labels") / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, out_lbl / lbl.name)

    total_imgs = len(list(out_img.glob("*.jpg")))
    total_lbls = len(list(out_lbl.glob("*.txt")))

    # Thống kê hộp nhãn
    CLASS_NAMES = [
        "patient", "doctor", "wheelchair", "queue_kiosk", "reception_counter",
        "sanitizer_stand", "pccc_station", "medical_waste_bin", "water_dispenser",
        "wc_station", "elevator_doors", "hospital_bed", "stretcher_cart",
        "iv_stand", "medical_trolley", "vital_monitor", "hospital_chair"
    ]
    counts = {c: 0 for c in CLASS_NAMES}
    total_boxes = 0

    for lbl_file in out_lbl.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    if cid < len(CLASS_NAMES):
                        counts[CLASS_NAMES[cid]] += 1
                        total_boxes += 1

    print("==================================================================")
    print(f"✅ HỢP NHẤT THÀNH CÔNG! Tổng frames: {total_imgs} | Tổng boxes: {total_boxes}")
    print("📊 Phân bổ hộp nhãn của 17 lớp trong bộ test hợp nhất:")
    for c, cnt in counts.items():
        print(f"   • {c:<20}: {cnt:4d} boxes")
    print("==================================================================")

if __name__ == "__main__":
    main()
