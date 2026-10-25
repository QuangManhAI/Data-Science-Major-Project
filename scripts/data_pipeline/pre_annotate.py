#!/usr/bin/env python3
"""
pre_annotate.py
Active Learning Pre-annotation tool using YOLO26n.
Chạy inference mô hình YOLO26n trên tập ảnh mới cần review, tự động sinh nhãn nháp (YOLO txt format)
và phân loại theo 3 tầng độ tin cậy (High / Medium / Low) phục vụ Active Learning Loop.
"""

import argparse
import json
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    repo_root = Path(__file__).resolve().parents[2]
    
    parser = argparse.ArgumentParser(description="Tự động sinh nhãn nháp cho Active Learning bằng YOLO26n")
    parser.add_argument("--model", type=str, default=str(repo_root / "models/yolo26n/yolo26n_hospital_best.pt"), help="File weights YOLO26n")
    parser.add_argument("--input-dir", type=str, default=str(repo_root / "data/vision/raw/images"), help="Thư mục ảnh cần quét nhãn")
    parser.add_argument("--output-labels", type=str, default=str(repo_root / "data/vision/raw/labels_pre_annotated"), help="Thư mục xuất nhãn nháp txt")
    parser.add_argument("--high-conf", type=float, default=0.85, help="Ngưỡng tin cậy cao (Auto-accept candidate)")
    parser.add_argument("--low-conf", type=float, default=0.40, help="Ngưỡng tin cậy thấp (Cần human check kỹ)")
    parser.add_argument("--device", type=str, default="0", help="Thiết bị tính toán")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        fallback = repo_root / "experiments/runs/train_hospital/yolo26n_hospital_v1/weights/best.pt"
        if fallback.exists():
            model_path = fallback
        else:
            print(f"❌ Không tìm thấy model weights tại {model_path}")
            return

    input_dir = Path(args.input_dir)
    output_labels = Path(args.output_labels)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    if not images:
        print(f"⚠️ Thư mục {input_dir} không có ảnh nào để quét nhãn.")
        return

    print("==================================================================")
    print("🤖 BẮT ĐẦU QUÉT NHÃN TỰ ĐỘNG BẰNG YOLO26n (ACTIVE LEARNING)")
    print(f"• Model: {model_path}")
    print(f"• Thư mục ảnh ({len(images)} ảnh): {input_dir}")
    print(f"• Thư mục xuất nhãn: {output_labels}")
    print(f"• Phân tầng: High >= {args.high_conf} | Low < {args.low_conf}")
    print("==================================================================")

    model = YOLO(str(model_path))
    
    stats = {
        "total_images": len(images),
        "total_boxes": 0,
        "high_confidence_frames": 0,
        "medium_confidence_frames": 0,
        "low_confidence_or_empty_frames": 0,
        "class_counts": {0: 0, 1: 0} # 0: person, 1: wheelchair
    }

    tier_manifest = {
        "high_confidence": [],
        "medium_confidence": [],
        "low_confidence_or_empty": []
    }

    for img_path in images:
        results = model(img_path, conf=0.15, device=args.device, verbose=False)[0]
        boxes = results.boxes
        
        label_file = output_labels / f"{img_path.stem}.txt"
        yolo_lines = []
        min_conf = 1.0
        max_conf = 0.0

        if len(boxes) > 0:
            for cls_id, xywhn, conf in zip(boxes.cls.cpu().numpy(), boxes.xywhn.cpu().numpy(), boxes.conf.cpu().numpy()):
                c_id = int(cls_id)
                if c_id in (0, 1):
                    x, y, w, h = xywhn
                    yolo_lines.append(f"{c_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    stats["class_counts"][c_id] = stats["class_counts"].get(c_id, 0) + 1
                    stats["total_boxes"] += 1
                    min_conf = min(min_conf, float(conf))
                    max_conf = max(max_conf, float(conf))
        
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        # Phân loại độ tin cậy của frame
        if len(yolo_lines) == 0 or min_conf < args.low_conf:
            stats["low_confidence_or_empty_frames"] += 1
            tier_manifest["low_confidence_or_empty"].append({
                "file": img_path.name,
                "n_boxes": len(yolo_lines),
                "min_conf": round(min_conf, 3) if len(yolo_lines) > 0 else 0.0
            })
        elif min_conf >= args.high_conf:
            stats["high_confidence_frames"] += 1
            tier_manifest["high_confidence"].append({
                "file": img_path.name,
                "n_boxes": len(yolo_lines),
                "min_conf": round(min_conf, 3)
            })
        else:
            stats["medium_confidence_frames"] += 1
            tier_manifest["medium_confidence"].append({
                "file": img_path.name,
                "n_boxes": len(yolo_lines),
                "min_conf": round(min_conf, 3)
            })

    # Lưu báo cáo tổng kết
    manifest_path = output_labels.parent / "pre_annotation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "tiers": tier_manifest}, f, indent=2, ensure_ascii=False)

    print("\n==================================================================")
    print("✅ HOÀN TẤT QUÉT NHÃN NHÁP CHO ACTIVE LEARNING:")
    print(f"• Tổng số frames:                    {stats['total_images']}")
    print(f"• Tổng số bounding boxes sinh ra:   {stats['total_boxes']} (Person: {stats['class_counts'].get(0, 0)}, Wheelchair: {stats['class_counts'].get(1, 0)})")
    print(f"• Tầng tin cậy cao (High Conf >= {args.high_conf}): {stats['high_confidence_frames']} frames ({stats['high_confidence_frames']/stats['total_images']*100:.1f}%)")
    print(f"• Tầng cần review (Medium Conf):     {stats['medium_confidence_frames']} frames ({stats['medium_confidence_frames']/stats['total_images']*100:.1f}%)")
    print(f"• Tầng khó/trống (Low Conf < {args.low_conf}):  {stats['low_confidence_or_empty_frames']} frames ({stats['low_confidence_or_empty_frames']/stats['total_images']*100:.1f}%)")
    print(f"• Manifest lưu tại:                  {manifest_path}")
    print("==================================================================")

if __name__ == "__main__":
    main()
