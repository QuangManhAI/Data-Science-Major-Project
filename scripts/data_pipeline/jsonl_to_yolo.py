#!/usr/bin/env python3
"""
jsonl_to_yolo.py
Script chuyển đổi nhanh từ file metadata đa tầng `annotations.jsonl` (Phase 0)
sang định dạng file nhãn chuẩn YOLO (.txt) gồm 5 giá trị:
<class_id> <x_center> <y_center> <width> <height> (chuẩn hóa 0.0 - 1.0).
"""

import os
import json
import argparse
from pathlib import Path

def convert_jsonl_to_yolo(jsonl_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not jsonl_path.exists():
        print(f"❌ Không tìm thấy file: {jsonl_path}")
        return

    print("==================================================================")
    print("🔄 CHUYỂN ĐỔI METADATA ANNOTATIONS.JSONL SANG FILE NHÃN YOLO (.TXT)")
    print(f"• File nguồn: {jsonl_path}")
    print(f"• Thư mục đích: {output_dir}")
    print("==================================================================")

    total_frames = 0
    total_boxes = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            frame_id = data.get("frame_id", "")
            if not frame_id:
                img_p = Path(data.get("image_path", ""))
                frame_id = img_p.stem

            if not frame_id:
                continue

            yolo_lines = []
            img_w = data.get("image", {}).get("width", 640.0)
            img_h = data.get("image", {}).get("height", 480.0)

            anns = data.get("annotations", []) or data.get("ground_truth_annotations", [])
            for ann in anns:
                c_id = ann.get("category_id", 0)
                # Chỉ lấy 0 (person) và 1 (wheelchair)
                if c_id not in (0, 1):
                    continue

                if "yolo_normalized" in ann:
                    xc, yc, w, h = ann["yolo_normalized"]
                elif "bbox_xyxy_px" in ann:
                    x1, y1, x2, y2 = ann["bbox_xyxy_px"]
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    xc = (x1 + x2) / (2.0 * img_w)
                    yc = (y1 + y2) / (2.0 * img_h)
                elif "visible_bbox_xyxy" in ann:
                    x1, y1, x2, y2 = ann["visible_bbox_xyxy"]
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    xc = (x1 + x2) / (2.0 * img_w)
                    yc = (y1 + y2) / (2.0 * img_h)
                else:
                    continue

                # Kẹp giá trị vào khoảng [0, 1]
                xc = max(0.0, min(1.0, float(xc)))
                yc = max(0.0, min(1.0, float(yc)))
                w = max(0.0, min(1.0, float(w)))
                h = max(0.0, min(1.0, float(h)))

                yolo_lines.append(f"{c_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                total_boxes += 1

            out_file = output_dir / f"{frame_id}.txt"
            with open(out_file, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            total_frames += 1

    print(f"\n✅ Đã chuyển đổi thành công {total_frames} frames ({total_boxes} boxes).")
    print("==================================================================")

def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Chuyển annotations.jsonl sang file nhãn YOLO")
    parser.add_argument("--jsonl", type=str, default=str(repo_root / "data/vision/raw/annotations.jsonl"), help="File annotations.jsonl")
    parser.add_argument("--output", type=str, default=str(repo_root / "data/vision/raw/labels"), help="Thư mục xuất nhãn txt")
    args = parser.parse_args()

    convert_jsonl_to_yolo(Path(args.jsonl), Path(args.output))

if __name__ == "__main__":
    main()
