#!/usr/bin/env python3
"""
active_learning_select.py
Active Learning Uncertainty & Hard-Case Sampling:
Lọc và chọn lọc các frames có độ bất định cao nhất (Uncertainty Sampling / Least Confidence / Margin)
hoặc các ca khó (hard negatives, occlusion, unusual aspect ratios) để ưu tiên gán nhãn và review trước.
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO

def main():
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Chọn lọc frames bất định cho Active Learning")
    parser.add_argument("--model", type=str, default=str(repo_root / "models/yolo26n/yolo26n_hospital_best.pt"), help="Weights model YOLO26n")
    parser.add_argument("--input-dir", type=str, default=str(repo_root / "data/vision/raw/images"), help="Thư mục ảnh cần chọn lọc")
    parser.add_argument("--top-k", type=int, default=50, help="Số lượng frames khó nhất cần trích xuất")
    parser.add_argument("--device", type=str, default="0", help="Thiết bị tính toán")
    parser.add_argument("--output-json", type=str, default=str(repo_root / "data/vision/in_review/hard_cases_selected.json"), help="File kết quả chọn lọc")
    args = parser.parse_args()

    model_path = Path(args.model)
    input_dir = Path(args.input_dir)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"❌ Không tìm thấy model weights tại: {model_path}")
        return

    images = list(input_dir.glob("*.jpg"))
    if not images:
        print(f"❌ Không tìm thấy ảnh nào tại {input_dir}")
        return

    print("==================================================================")
    print("🎯 BẮT ĐẦU CHỌN LỌC DỮ LIỆU BẤT ĐỊNH (ACTIVE LEARNING SELECTION)")
    print(f"• Model: {model_path}")
    print(f"• Thư mục đầu vào: {input_dir} ({len(images)} ảnh)")
    print(f"• Chỉ tiêu chọn lọc: Top {args.top_k} frames bất định nhất")
    print("==================================================================")

    model = YOLO(str(model_path))
    scored_frames = []

    for img_path in images:
        res = model(img_path, conf=0.10, device=args.device, verbose=False)[0]
        boxes = res.boxes

        if len(boxes) == 0:
            # Ảnh không detect ra gì: Có thể là true negative hoặc total miss
            scored_frames.append({
                "file": img_path.name,
                "uncertainty_score": 0.50,
                "reason": "empty_prediction_candidate_negative_or_miss",
                "max_conf": 0.0,
                "n_boxes": 0
            })
        else:
            confs = boxes.conf.cpu().numpy()
            # Bất định cao nhất khi confidence nằm gần ngưỡng phân định (0.40 - 0.60)
            uncertainty = float(1.0 - np.abs(confs - 0.50).min() * 2.0)
            scored_frames.append({
                "file": img_path.name,
                "uncertainty_score": round(uncertainty, 4),
                "reason": "marginal_confidence_prediction",
                "max_conf": round(float(confs.max()), 4),
                "n_boxes": len(boxes)
            })

    # Sắp xếp điểm bất định giảm dần
    scored_frames.sort(key=lambda x: x["uncertainty_score"], reverse=True)
    selected_hard_cases = scored_frames[:args.top_k]

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_evaluated": len(images),
            "top_k": args.top_k,
            "selected_cases": selected_hard_cases
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Đã chọn lọc thành công {len(selected_hard_cases)} frames khó nhất.")
    print(f"• Kết quả lưu tại: {out_json}")
    print(f"• Điểm bất định cao nhất: {selected_hard_cases[0]['uncertainty_score']} ({selected_hard_cases[0]['file']})")
    print("==================================================================")

if __name__ == "__main__":
    import numpy as np
    main()
