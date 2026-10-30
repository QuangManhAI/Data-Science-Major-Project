#!/usr/bin/env python3
"""
train_dual_baselines.py
Điều phối tự động quy trình huấn luyện & đánh giá song hành 2 mô hình YOLO26n Baseline:
1. Model A: YOLO26n-Clinical (configs/hospital_3class.yaml - 3 Classes)
2. Model B: YOLO26n-Props (configs/hospital_props_15class.yaml - 15 Classes)

Đặc tả:
- Huấn luyện 30 epochs trên GPU RTX 3050 (AMP Mixed Precision).
- Đánh giá khách quan trên tập Test đóng băng (split='test').
- Lưu trữ toàn bộ chỉ số vào experiments/results/dual_baseline_metrics.json.
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
    """Trích xuất metrics từ đối tượng kết quả val() của Ultralytics."""
    metrics = {
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50_95": round(float(val_results.box.map), 4),
        "precision": round(float(val_results.box.mp), 4),
        "recall": round(float(val_results.box.mr), 4),
        "f1": round(float(2 * (val_results.box.mp * val_results.box.mr) / max(1e-6, (val_results.box.mp + val_results.box.mr))), 4),
        "per_class": {}
    }
    
    # Per-class mAP
    if hasattr(val_results.box, "maps") and len(val_results.box.maps) > 0:
        for idx, (cid, cname) in enumerate(class_names.items()):
            if idx < len(val_results.box.maps):
                map50_cls = val_results.box.maps[idx] # mAP50-95 per class
                metrics["per_class"][cname] = {
                    "class_id": int(cid),
                    "mAP50_95": round(float(map50_cls), 4)
                }
    return metrics


def train_model_a(args, initial_weights):
    print("\n" + "=" * 70)
    print("🏥 [1/2] BẮT ĐẦU HUẤN LUYỆN MODEL A: YOLO26n-Clinical (3 Classes)")
    print("=" * 70)
    data_yaml = REPO_ROOT / "configs/hospital_3class.yaml"
    project_dir = REPO_ROOT / "experiments/runs/train_hospital"
    run_name = "yolo26n_clinical_baseline_v1"

    start_time = time.time()
    model = YOLO(str(initial_weights))

    print(f"• Dataset: {data_yaml}")
    print(f"• Epochs: {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz} | Device: {args.device}")

    # 1. Huấn luyện
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
    print(f"\n✅ Model A huấn luyện hoàn tất sau {elapsed}s! Weights: {best_weights}")

    # 2. Đánh giá trên tập Test Đóng Băng (Frozen Test Set)
    print("\n🧊 Đang đánh giá Model A trên tập Test Đóng Băng (198 frames)...")
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

    class_names = {0: "patient", 1: "doctor", 2: "wheelchair"}
    test_metrics = extract_metrics(val_test, class_names)
    test_metrics["training_time_sec"] = elapsed
    test_metrics["best_weights"] = str(best_weights)
    print(f"📊 Model A Frozen Test Results: mAP@50={test_metrics['mAP50']}, mAP@50-95={test_metrics['mAP50_95']}")
    return test_metrics


def train_model_b(args, initial_weights):
    print("\n" + "=" * 70)
    print("🛋️ [2/2] BẮT ĐẦU HUẤN LUYỆN MODEL B: YOLO26n-Props (15 Classes)")
    print("=" * 70)
    data_yaml = REPO_ROOT / "configs/hospital_props_15class.yaml"
    project_dir = REPO_ROOT / "experiments/runs/train_hospital"
    run_name = "yolo26n_props_baseline_v1"

    start_time = time.time()
    model = YOLO(str(initial_weights))

    print(f"• Dataset: {data_yaml}")
    print(f"• Epochs: {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz} | Device: {args.device}")

    # 1. Huấn luyện
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
    print(f"\n✅ Model B huấn luyện hoàn tất sau {elapsed}s! Weights: {best_weights}")

    # 2. Đánh giá trên tập Test Đóng Băng (Frozen Test Set)
    print("\n🧊 Đang đánh giá Model B trên tập Test Đóng Băng (240 frames)...")
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

    manifest_file = REPO_ROOT / "data/vision/dataset/hospital_props_1600_15class/manifest.json"
    class_names = {}
    if manifest_file.exists():
        with open(manifest_file, "r") as f:
            m = json.load(f)
            class_names = {int(k): str(v) for k, v in m.get("classes", {}).items()}

    test_metrics = extract_metrics(val_test, class_names)
    test_metrics["training_time_sec"] = elapsed
    test_metrics["best_weights"] = str(best_weights)
    print(f"📊 Model B Frozen Test Results: mAP@50={test_metrics['mAP50']}, mAP@50-95={test_metrics['mAP50_95']}")
    return test_metrics


def main():
    default_device = "0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Điều phối huấn luyện 2 mô hình YOLO26n Baseline")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện")
    parser.add_argument("--batch", type=int, default=16, help="Kích thước batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Độ phân giải đầu vào")
    parser.add_argument("--device", type=str, default=default_device, help="Thiết bị tính toán")
    parser.add_argument("--model", type=str, default=str(REPO_ROOT / "models/yolo26n/yolo26n.pt"), help="File weights khởi tạo")
    args = parser.parse_args()

    initial_weights = Path(args.model)
    if not initial_weights.exists():
        initial_weights = REPO_ROOT / "weights/yolo26n.pt"

    print("==================================================================")
    print("🚀 BẮT ĐẦU CHƯƠNG TRÌNH HUẤN LUYỆN DUAL YOLO26n BASELINE BENCHMARK")
    print(f"• Weights khởi tạo: {initial_weights}")
    print(f"• Thiết bị:         {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"• Epochs:           {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print("==================================================================")

    # 1. Huấn luyện Model A
    metrics_a = train_model_a(args, initial_weights)

    # 2. Huấn luyện Model B
    metrics_b = train_model_b(args, initial_weights)

    # 3. Tổng hợp kết quả
    total_metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        "model_a_clinical_3class": metrics_a,
        "model_b_props_15class": metrics_b
    }

    out_json = REPO_ROOT / "experiments/results/dual_baseline_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(total_metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"🎉 HOÀN TẤT HUẤN LUYỆN & ĐÁNH GIÁ 2 MÔ HÌNH BASELINE THÀNH CÔNG!")
    print(f"📝 Kết quả tổng hợp đã lưu tại: {out_json}")
    print("==================================================================")


if __name__ == "__main__":
    main()
