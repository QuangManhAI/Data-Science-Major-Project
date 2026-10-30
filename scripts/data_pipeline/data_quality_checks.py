#!/usr/bin/env python3
"""
data_quality_checks.py
Automated Quality Assurance & Data Integrity Validator for Hospital Vision Dataset.
Kiểm tra toàn diện trước khi train hoặc release dataset:
1. Bbox Sanity: Tọa độ hợp lệ [0, 1], w > 0, h > 0, aspect ratio không dị dạng.
2. Label Consistency: Mỗi ảnh có 1 label, mỗi label có 1 ảnh, không class ID lạ.
3. Data Leakage: Không trùng lặp stem/file giữa Train, Val và Test.
4. Dataset Balance: Thống kê phân bố lớp và kích thước mẫu.
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter

def check_dataset(dataset_dir: Path):
    print("==================================================================")
    print("🛡️ BẮT ĐẦU KIỂM TRA CHẤT LƯỢNG DỮ LIỆU TỰ ĐỘNG (DATA QUALITY CHECKS)")
    print(f"• Thư mục Dataset: {dataset_dir}")
    print("==================================================================")

    # Đọc manifest.json nếu có để lấy schema classes động
    valid_classes = {0, 1, 2}
    class_names = {0: "patient", 1: "doctor", 2: "wheelchair"}
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                m_data = json.load(mf)
                if "classes" in m_data:
                    valid_classes = set()
                    class_names = {}
                    for k, v in m_data["classes"].items():
                        try:
                            cid = int(k)
                            valid_classes.add(cid)
                            class_names[cid] = str(v)
                        except ValueError:
                            cid = int(v)
                            valid_classes.add(cid)
                            class_names[cid] = str(k)
        except Exception:
            pass

    # Kiểm tra xem đang ở chế độ đã chia train/val/test hay chỉ là dataset đơn lẻ
    has_splits = (dataset_dir / "train" / "images").exists() and (dataset_dir / "val" / "images").exists() and (dataset_dir / "test" / "images").exists()
    is_unified = not has_splits and (dataset_dir / "images").exists() and (dataset_dir / "labels").exists()
    splits = ["train", "val", "test"] if has_splits else ["unified"]
    errors = []
    warnings = []
    split_stems = {}
    stats = {}

    for split in splits:
        if is_unified:
            img_dir = dataset_dir / "images"
            lbl_dir = dataset_dir / "labels"
        else:
            img_dir = dataset_dir / split / "images"
            lbl_dir = dataset_dir / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            errors.append(f"❌ Thiếu thư mục: {img_dir} hoặc {lbl_dir}")
            continue

        images = {p.stem: p for p in img_dir.glob("*.jpg")}
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")}
        split_stems[split] = set(images.keys())

        # Check orphan images/labels
        orphan_images = set(images.keys()) - set(labels.keys())
        orphan_labels = set(labels.keys()) - set(images.keys())

        if orphan_images:
            warnings.append(f"[{split}] Có {len(orphan_images)} ảnh không có file nhãn txt (ảnh negative/empty).")
        if orphan_labels:
            errors.append(f"[{split}] Có {len(orphan_labels)} file nhãn mồ côi (không có ảnh tương ứng)!")

        # Check label file content
        box_count = 0
        class_counter = Counter()

        for stem, lbl_path in labels.items():
            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for line_idx, line in enumerate(lines, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 5:
                    errors.append(f"[{split} - {stem}.txt:L{line_idx}] Dòng nhãn thiếu trường: '{line.strip()}'")
                    continue

                try:
                    c_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                except ValueError:
                    errors.append(f"[{split} - {stem}.txt:L{line_idx}] Giá trị không phải số: '{line.strip()}'")
                    continue

                if c_id not in valid_classes:
                    warnings.append(f"[{split} - {stem}.txt] Class ID lạ: {c_id} (hợp lệ: {sorted(list(valid_classes))})")

                # Bbox Bounds Check
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                    errors.append(f"[{split} - {stem}.txt] Tâm box [xc, yc] vượt ngưỡng [0, 1]: ({xc}, {yc})")
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    errors.append(f"[{split} - {stem}.txt] Kích thước box [w, h] bất thường: ({w}, {h})")

                # Aspect ratio check
                aspect_ratio = max(w / h, h / w) if min(w, h) > 0 else 999
                if aspect_ratio > 12.0:
                    warnings.append(f"[{split} - {stem}.txt] Hộp có tỉ lệ cạnh cực dị (ratio={aspect_ratio:.1f}): w={w:.4f}, h={h:.4f}")

                class_counter[c_id] += 1
                box_count += 1

        stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": box_count,
            "classes": dict(class_counter)
        }

    # Data Leakage Check (Giao giữa các splits nếu ở chế độ split)
    print("\n🔍 1. KIỂM TRA CHỐNG THẤM DỮ LIỆU (DATA LEAKAGE CHECK):")
    if is_unified:
        print("  ℹ️ Đang ở chế độ UNIFIED DATASET (Toàn bộ dữ liệu nằm trong 1 folder, chưa chia train/val/test).")
    else:
        train_stems = split_stems.get("train", set())
        val_stems = split_stems.get("val", set())
        test_stems = split_stems.get("test", set())

        train_val_leak = train_stems & val_stems
        train_test_leak = train_stems & test_stems
        val_test_leak = val_stems & test_stems

        if train_val_leak:
            errors.append(f"🔴 Data Leakage: Có {len(train_val_leak)} file xuất hiện ở cả Train và Val!")
        if train_test_leak:
            errors.append(f"🔴 CRITICAL DATA LEAKAGE: Có {len(train_test_leak)} file xuất hiện ở cả Train và Test!")
        if val_test_leak:
            errors.append(f"🔴 Data Leakage: Có {len(val_test_leak)} file xuất hiện ở cả Val và Test!")

        if not train_val_leak and not train_test_leak and not val_test_leak:
            print("  ✅ 100% CLEAN: Hoàn toàn không có rò rỉ dữ liệu giữa Train, Val và Test.")
        else:
            print("  ❌ PHÁT HIỆN RÒ RỈ DỮ LIỆU!")

        # Class Representation Check across splits
        for split in ["train", "val", "test"]:
            split_classes = set(stats.get(split, {}).get("classes", {}).keys())
            missing_in_split = valid_classes - split_classes
            if missing_in_split:
                missing_names = [class_names.get(c, str(c)) for c in missing_in_split]
                errors.append(f"[{split}] THIẾU LỚP (Zero Representation): Các lớp sau có 0 mẫu trong {split}: {missing_names}")
            else:
                print(f"  ✅ [{split.upper()}] Đầy đủ 100% các lớp đối tượng ({len(valid_classes)} classes > 0 instances).")

    # Summary Output
    print("\n📊 2. THỐNG KÊ PHÂN BỐ BỘ DỮ LIỆU:")
    total_imgs = sum(s["images"] for s in stats.values())
    total_boxes = sum(s["boxes"] for s in stats.values())

    for split in splits:
        s = stats.get(split, {})
        n_img = s.get("images", 0)
        n_box = s.get("boxes", 0)
        pct = (n_img / total_imgs * 100) if total_imgs > 0 else 0
        classes = s.get("classes", {})
        class_str = ", ".join([f"{class_names.get(cid, str(cid))}: {cnt}" for cid, cnt in sorted(classes.items())])
        print(f"  • {split.upper():<5}: {n_img:4d} ảnh ({pct:5.1f}%) | {n_box:4d} boxes | {class_str}")

    print(f"  -------------------------------------------------------------")
    print(f"  • TỔNG CỘNG: {total_imgs} ảnh, {total_boxes} boxes đã được kiểm chứng.")

    print("\n📋 3. KẾT LUẬN KIỂM ĐỊNH:")
    if errors:
        print(f"  ❌ PHÁT HIỆN {len(errors)} LỖI CẦN SỬA:")
        for e in errors[:10]:
            print(f"     - {e}")
        return False
    else:
        print("  ✅ PASS: 0 lỗi cấu trúc dữ liệu.")

    if warnings:
        print(f"  ⚠️ Cảnh báo ({len(warnings)} thông báo):")
        for w in warnings[:5]:
            print(f"     - {w}")
    else:
        print("  ✅ 0 cảnh báo. Bộ dữ liệu hoàn hảo.")

    print("==================================================================")
    return True

if __name__ == "__main__":
    import argparse
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng và tính toàn vẹn dữ liệu thị giác")
    parser.add_argument("--dir", type=str, default=str(repo_root / "data/vision/dataset/hospital_gold_1387_3class"), help="Thư mục dataset cần kiểm tra")
    args = parser.parse_args()

    passed = check_dataset(Path(args.dir))
    sys.exit(0 if passed else 1)
