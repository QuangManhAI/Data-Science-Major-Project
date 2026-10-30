#!/usr/bin/env python3
"""
train_unified_super_model.py
Huấn luyện Siêu Mô Hình Duy Nhất: YOLO26n-Unified (17 Classes)
trên bộ dữ liệu hợp nhất hospital_unified_3120_17class (3,120 frames).

Đặc tả:
- 30 Epochs | Batch 16 | Imgsz 640 | GPU RTX 3050 Laptop (AMP Mixed Precision).
- Đánh giá khách quan trên tập Test Đóng Băng (469 frames, 1,164 instances).
- Lưu kết quả vào experiments/results/unified_17class_metrics.json.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]


def extract_metrics(val_results, class_names):
    metrics = {
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50_95": round(float(val_results.box.map), 4),
        "precision": round(float(val_results.box.mp), 4),
        "recall": round(float(val_results.box.mr), 4),
        "f1": round(float(2 * (val_results.box.mp * val_results.box.mr) / max(1e-6, (val_results.box.mp + val_results.box.mr))), 4),
        "speed": {k: round(v, 2) for k, v in getattr(val_results, "speed", {}).items()},
        "per_class": {}
    }
    
    if hasattr(val_results.box, "maps") and len(val_results.box.maps) > 0:
        for idx, (cid, cname) in enumerate(class_names.items()):
            if idx < len(val_results.box.maps):
                map_cls = val_results.box.maps[idx]
                metrics["per_class"][cname] = {
                    "class_id": int(cid),
                    "mAP50_95": round(float(map_cls), 4)
                }
    return metrics


def main():
    default_device = "0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Huấn luyện Siêu Mô Hình YOLO26n-Unified 17 Classes")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện")
    parser.add_argument("--batch", type=int, default=16, help="Kích thước batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Độ phân giải đầu vào")
    parser.add_argument("--device", type=str, default=default_device, help="Thiết bị tính toán")
    parser.add_argument("--model", type=str, default=str(REPO_ROOT / "models/yolo26n/yolo26n.pt"), help="File weights khởi tạo")
    args = parser.parse_args()

    initial_weights = Path(args.model)
    if not initial_weights.exists():
        initial_weights = REPO_ROOT / "weights/yolo26n.pt"

    data_yaml = REPO_ROOT / "configs/hospital_unified_17class.yaml"
    project_dir = REPO_ROOT / "experiments/runs/train_hospital"
    run_name = "yolo26n_unified_17class_v1"

    print("==================================================================")
    print("🌟 BẮT ĐẦU HUẤN LUYỆN SIÊU MÔ HÌNH HỢP NHẤT: YOLO26n-Unified (17 CLASSES)")
    print(f"• Weights khởi tạo: {initial_weights}")
    print(f"• Dataset Config:   {data_yaml}")
    print(f"• Thiết bị:         {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"• Epochs:           {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print("==================================================================")

    start_time = time.time()
    model = YOLO(str(initial_weights))

    # 1. Huấn luyện Siêu Mô Hình
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
        seed=42
    )

    elapsed = round(time.time() - start_time, 1)
    best_weights = project_dir / run_name / "weights/best.pt"
    print(f"\n✅ Siêu Mô Hình 17 Lớp huấn luyện hoàn tất sau {elapsed}s! Weights: {best_weights}")

    # 2. Đánh giá trên tập Test Đóng Băng (469 frames, 1,164 instances)
    print("\n🧊 Đang đánh giá Siêu Mô Hình trên tập Test Đóng Băng (469 frames)...")
    best_model = YOLO(str(best_weights))
    val_test = best_model.val(
        data=str(data_yaml),
        split="test",
        project=str(project_dir / run_name),
        name="eval_frozen_test",
        exist_ok=True,
        plots=True,
        verbose=True,
        device=args.device
    )

    manifest_file = REPO_ROOT / "data/vision/dataset/hospital_unified_3120_17class/manifest.json"
    class_names = {}
    if manifest_file.exists():
        with open(manifest_file, "r") as f:
            m = json.load(f)
            class_names = {int(k): str(v) for k, v in m.get("classes", {}).items()}

    test_metrics = extract_metrics(val_test, class_names)
    test_metrics["training_time_sec"] = elapsed
    test_metrics["best_weights"] = str(best_weights)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": "YOLO26n-Unified-17Class",
        "dataset": "hospital_unified_3120_17class",
        "total_classes": len(class_names),
        "test_frames": 469,
        "hardware": {
            "device": args.device,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "imgsz": args.imgsz,
            "initial_weights": str(initial_weights)
        },
        "metrics": test_metrics
    }

    out_json = REPO_ROOT / "experiments/results/unified_17class_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("🎉 KẾT QUẢ ĐÁNH GIÁ SIÊU MÔ HÌNH 17 LỚP TRÊN TẬP TEST ĐÓNG BĂNG:")
    print(f"• mAP@50:    {test_metrics['mAP50']*100:.2f}%")
    print(f"• mAP@50-95: {test_metrics['mAP50_95']*100:.2f}%")
    print(f"• Precision: {test_metrics['precision']*100:.2f}%")
    print(f"• Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"• F1-Score:  {test_metrics['f1']*100:.2f}%")
    print(f"📝 Kết quả chi tiết đã lưu tại: {out_json}")
    print("==================================================================")


if __name__ == "__main__":
    main()
