#!/usr/bin/env python3
"""
train_verified_super_model.py
Huấn luyện Siêu Mô Hình YOLO26n-Unified (17 Classes) trên bộ dữ liệu Ground Truth đã kiểm duyệt CVAT:
- Dataset: hospital_unified_3120_17class (2,184 Train / 468 Val / 468 Test) - Zero Data Leakage.
- Cấu hình Full Augmentation: Copy-Paste (0.3), Multi-Scale (0.5), Mosaic (1.0), MixUp (0.15),
  Perspective (0.0005), close_mosaic=10, box=8.0.
- Epochs: 45 | Batch: 16 | Imgsz: 640 | GPU RTX 3050 Laptop.
- Đánh giá độc lập trên tập Test Đóng Băng (468 frames, 1,162 Ground Truth boxes).
- Lưu kết quả vào: experiments/results/verified_17class_metrics.json.
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

UNIFIED_CLASSES = {
    0: "patient",
    1: "doctor",
    2: "wheelchair",
    3: "queue_kiosk",
    4: "reception_counter",
    5: "sanitizer_stand",
    6: "pccc_station",
    7: "medical_waste_bin",
    8: "water_dispenser",
    9: "wc_station",
    10: "elevator_doors",
    11: "hospital_bed",
    12: "stretcher_cart",
    13: "iv_stand",
    14: "medical_trolley",
    15: "vital_monitor",
    16: "hospital_chair"
}


def run_training():
    parser = argparse.ArgumentParser(description="Huấn luyện Siêu Mô Hình YOLO26n 17 Classes với Full Augmentation")
    parser.add_argument("--epochs", type=int, default=45, help="Số epochs huấn luyện")
    parser.add_argument("--batch", type=int, default=16, help="Kích thước batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Độ phân giải đầu vào")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Thiết bị tính toán")
    parser.add_argument("--model", type=str, default=str(REPO_ROOT / "models/yolo26n/yolo26n.pt"), help="File weights khởi tạo")
    parser.add_argument("--data", type=str, default=str(REPO_ROOT / "configs/hospital_unified_17class.yaml"), help="File YAML cấu hình dataset")
    parser.add_argument("--run-name", type=str, default="yolo26n_verified_17class_v1", help="Tên phiên chạy huấn luyện")
    parser.add_argument("--save-json", type=str, default=str(REPO_ROOT / "experiments/results/verified_17class_metrics.json"), help="Đường dẫn file JSON lưu kết quả")
    parser.add_argument("--box", type=float, default=9.5, help="Hệ số Box Loss gain (tăng độ thắt chặt bounding box)")
    parser.add_argument("--dfl", type=float, default=1.8, help="Hệ số DFL gain")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Số epochs cuối tắt mosaic để hội tụ ổn định")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục huấn luyện từ checkpoint last.pt")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    project_dir = REPO_ROOT / "experiments/runs/train_hospital"
    run_name = args.run_name
    run_dir = project_dir / run_name

    print("==================================================================")
    print("🚀 BẮT ĐẦU HUẤN LUYỆN SIÊU MÔ HÌNH YOLO26n-UNIFIED 17 CLASSES (FULL AUG)")
    print(f"• Dataset Config:   {data_yaml}")
    print(f"• Weights khởi tạo: {args.model}")
    print(f"• Thiết bị:         {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"• Epochs:           {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print("• Các kỹ thuật Augmentation kích hoạt:")
    print("    - Copy-Paste:     0.30")
    print("    - Mosaic:         1.00 (tắt ở 10 epochs cuối)")
    print("    - Multi-Scale:    scale=0.50")
    print("    - MixUp:          0.15")
    print("    - Perspective:    0.0005")
    print("    - Box Loss Gain:  8.0")
    print("==================================================================")

    start_time = time.time()
    
    # 1. Huấn luyện mô hình
    if args.resume and (run_dir / "weights/last.pt").exists():
        print(f"\n🔄 KÍCH HOẠT RESUME: Đang khôi phục và tiếp tục từ {run_dir / 'weights/last.pt'}...")
        model = YOLO(str(run_dir / "weights/last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.002,
            lrf=0.01,
            warmup_epochs=3.0,
            box=args.box,
            cls=1.0,
            dfl=args.dfl,
            copy_paste=0.3,
            mosaic=1.0,
            close_mosaic=args.close_mosaic,
            mixup=0.15,
            scale=0.5,
            perspective=0.0005,
            degrees=10.0,
            translate=0.1,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,
            patience=15,
            save=True,
            plots=True,
            verbose=True
        )

    train_elapsed = time.time() - start_time
    print(f"\n✅ HUẤN LUYỆN HOÀN TẤT trong {train_elapsed/60:.2f} phút!")

    best_weights = run_dir / "weights/best.pt"
    if not best_weights.exists():
        best_weights = run_dir / "weights/last.pt"

    print(f"\n🧊 2. ĐÁNH GIÁ KHÁCH QUAN TRÊN TẬP TEST ĐÓNG BĂNG (split='test')...")
    eval_model = YOLO(str(best_weights))
    test_results = eval_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        verbose=True
    )

    # Trích xuất số liệu chi tiết từng lớp
    box = test_results.box
    map50_all = box.map50
    map50_95_all = box.map
    p_all = box.mp
    r_all = box.mr
    f1_all = 2 * (p_all * r_all) / max(1e-6, (p_all + r_all))

    per_class_stats = {}
    ap_indices = list(box.ap_class_index) if hasattr(box, "ap_class_index") and box.ap_class_index is not None else []
    ap50 = getattr(box, "ap50", None)
    maps = getattr(box, "maps", None)
    p = getattr(box, "p", None)
    r = getattr(box, "r", None)
    f1 = getattr(box, "f1", None)

    for cid in sorted(UNIFIED_CLASSES.keys()):
        cname = UNIFIED_CLASSES[cid]
        if cid in ap_indices:
            idx = ap_indices.index(cid)
            c_map50 = float(ap50[idx]) if ap50 is not None and idx < len(ap50) else 0.0
            c_map50_95 = float(maps[idx]) if maps is not None and idx < len(maps) else 0.0
            c_p = float(p[idx]) if p is not None and idx < len(p) else 0.0
            c_r = float(r[idx]) if r is not None and idx < len(r) else 0.0
            c_f1 = float(f1[idx]) if f1 is not None and idx < len(f1) else 0.0
        else:
            c_map50 = c_map50_95 = c_p = c_r = c_f1 = 0.0

        per_class_stats[cname] = {
            "class_id": cid,
            "mAP50": round(c_map50, 4),
            "mAP50_95": round(c_map50_95, 4),
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "f1": round(c_f1, 4),
            "pass_90": bool(c_map50 >= 0.90)
        }

    results_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": "YOLO26n-Unified-17Class-VerifiedGT",
        "dataset": "hospital_unified_3120_17class",
        "total_classes": 17,
        "test_frames": 468,
        "hardware": {
            "device": str(args.device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "imgsz": args.imgsz,
            "optimizer": "AdamW",
            "box_gain": args.box,
            "dfl_gain": args.dfl,
            "copy_paste": 0.3,
            "mosaic": 1.0,
            "close_mosaic": args.close_mosaic,
            "mixup": 0.15,
            "scale": 0.5,
            "initial_weights": str(args.model)
        },
        "metrics": {
            "mAP50": round(float(map50_all), 4),
            "mAP50_95": round(float(map50_95_all), 4),
            "precision": round(float(p_all), 4),
            "recall": round(float(r_all), 4),
            "f1": round(float(f1_all), 4),
            "speed": {k: round(v, 2) for k, v in getattr(test_results, "speed", {}).items()},
            "classes_above_90_count": sum(1 for v in per_class_stats.values() if v["pass_90"]),
            "per_class": per_class_stats
        },
        "training_time_sec": round(train_elapsed, 1),
        "best_weights": str(best_weights)
    }

    results_json_path = Path(args.save_json).resolve()
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Đã lưu kết quả đo lường vào: {results_json_path}")
    print("\n==================================================================")
    print("🏆 BÁO CÁO NGHIỆM THU TẬP TEST ĐÓNG BĂNG (17 CLASSES):")
    print("==================================================================")
    print(f"• mAP@50 Tổng thể:    {map50_all*100:.2f}%")
    print(f"• mAP@50-95 Tổng thể: {map50_95_all*100:.2f}%")
    print(f"• Precision:          {p_all*100:.2f}%")
    print(f"• Recall:             {r_all*100:.2f}%")
    print(f"• F1-Score:           {f1_all*100:.2f}%")
    print(f"• Số lớp đạt >= 90%:  {results_data['metrics']['classes_above_90_count']}/17 lớp")
    print("-" * 65)
    print(f"{'Class':<5} {'Tên Lớp':<19} {'mAP@50':<10} {'mAP@50-95':<12} {'Recall':<10} {'Trạng Thái'}")
    print("-" * 65)
    for cname, stats in per_class_stats.items():
        pass_str = "✅ ĐẠT" if stats["pass_90"] else "❌ CHƯA ĐẠT"
        print(f"{stats['class_id']:<5} {cname:<19} {stats['mAP50']*100:<9.1f}% {stats['mAP50_95']*100:<11.1f}% {stats['recall']*100:<9.1f}% {pass_str}")
    print("==================================================================")


if __name__ == "__main__":
    run_training()
