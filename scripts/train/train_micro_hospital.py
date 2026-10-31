#!/usr/bin/env python3
"""
scripts/train/train_micro_hospital.py
======================================
Huấn luyện Mô hình Cực Hạn Siêu Nhẹ YOLO26-Micro (0.62M params, 3.17 GFLOPs):
- Cắt bỏ tầng Attention C2PSA, nén kênh đặc trưng tầng sâu.
- Warm-start chuyển giao tri thức từ checkpoint v6 (domain transfer).
- Cosine Annealing learning rate + tăng cường cls loss để kéo Recall.
- Đánh giá trực tiếp sau huấn luyện trên Toàn Bộ 884 Frames Benchmark (3,082 boxes).
- Xuất kết quả vào: experiments/results/yolo26_micro_benchmark_metrics.json
"""

import os
import sys
import time
import json
from pathlib import Path
import torch
from ultralytics import YOLO

# Đặt biến môi trường chống xung đột luồng và clock
os.environ["OMP_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "configs/hospital_unified_golden_17class.yaml"
BENCHMARK_YAML = REPO_ROOT / "configs/hospital_robot_final_benchmark_884.yaml"
INIT_MODEL = REPO_ROOT / "models/yolo26_micro_init.pt"
PROJECT_DIR = REPO_ROOT / "experiments/runs/train_hospital"
RUN_NAME = "yolo26_micro_golden_17class"
RUN_DIR = PROJECT_DIR / RUN_NAME
SAVE_METRICS_JSON = REPO_ROOT / "experiments/results/yolo26_micro_benchmark_metrics.json"

CLASSES = {
    0: "patient", 1: "doctor", 2: "wheelchair", 3: "queue_kiosk", 4: "reception_counter",
    5: "sanitizer_stand", 6: "pccc_station", 7: "medical_waste_bin", 8: "water_dispenser",
    9: "wc_station", 10: "elevator_doors", 11: "hospital_bed", 12: "stretcher_cart",
    13: "iv_stand", 14: "medical_trolley", 15: "vital_monitor", 16: "hospital_chair"
}


def main():
    print("==================================================================")
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH SIÊU NHẸ YOLO26-MICRO (0.62M PARAMS)")
    print(f"• Dataset YAML:    {DATA_YAML}")
    print(f"• Model gốc:       {INIT_MODEL} (0.62M params, 3.17 GFLOPs)")
    print(f"• Thư mục đích:    {RUN_DIR}")
    print(f"• Thiết bị:        GPU CUDA ({torch.cuda.get_device_name(0)})")
    print("• Quy mô dữ liệu:  Train: 3,715 frames | Val: 655 frames | Test: 884 frames")
    print("==================================================================")

    start_time = time.time()

    model = YOLO(str(INIT_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=40,
        batch=16,
        imgsz=640,
        device="0",
        workers=0,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=2.5,
        box=9.5,
        cls=1.5,
        dfl=2.0,
        copy_paste=0.15,
        mosaic=1.0,
        close_mosaic=12,
        mixup=0.08,
        scale=0.4,
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
    print(f"\n✅ HUẤN LUYỆN YOLO26-MICRO HOÀN TẤT TRONG {train_elapsed/60:.2f} PHÚT!")

    best_pt = RUN_DIR / "weights/best.pt"
    if not best_pt.exists():
        best_pt = RUN_DIR / "weights/last.pt"

    print(f"\n🧊 TIẾN HÀNH ĐÁNH GIÁ NGHIỆM THU TRÊN BỘ TEST BENCHMARK 884 FRAMES (3,082 BOXES)...")
    eval_model = YOLO(str(best_pt))
    test_results = eval_model.val(
        data=str(BENCHMARK_YAML),
        split="test",
        imgsz=640,
        device="0",
        workers=0,
        plots=True,
        verbose=True
    )

    metrics_dict = test_results.results_dict
    mp = float(metrics_dict.get("metrics/precision(B)", 0.0))
    mr = float(metrics_dict.get("metrics/recall(B)", 0.0))
    map50 = float(metrics_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(metrics_dict.get("metrics/mAP50-95(B)", 0.0))

    per_class_results = {}
    if hasattr(test_results, "box") and hasattr(test_results.box, "ap50"):
        ap50_list = test_results.box.ap50
        ap_list = test_results.box.ap
        p_list = test_results.box.p
        r_list = test_results.box.r

        for cls_id, cls_name in CLASSES.items():
            if cls_id < len(ap50_list):
                c_p = float(p_list[cls_id]) if cls_id < len(p_list) else 0.0
                c_r = float(r_list[cls_id]) if cls_id < len(r_list) else 0.0
                c_map50 = float(ap50_list[cls_id])
                c_map = float(ap_list[cls_id]) if cls_id < len(ap_list) else 0.0
                per_class_results[cls_name] = {
                    "precision": round(c_p * 100, 2),
                    "recall": round(c_r * 100, 2),
                    "map50": round(c_map50 * 100, 2),
                    "map50_95": round(c_map * 100, 2)
                }

    speed_info = test_results.speed
    infer_ms = float(speed_info.get("inference", 0.0))
    fps = round(1000.0 / infer_ms, 1) if infer_ms > 0 else 0.0

    output_payload = {
        "model_name": "YOLO26-Micro (Hospital Simulator)",
        "model_path": str(best_pt),
        "total_params": 622959,
        "gflops": 3.17,
        "imgsz": 640,
        "train_time_minutes": round(train_elapsed / 60, 2),
        "overall": {
            "precision": round(mp * 100, 2),
            "recall": round(mr * 100, 2),
            "map50": round(map50 * 100, 2),
            "map50_95": round(map50_95 * 100, 2),
            "inference_ms": round(infer_ms, 2),
            "fps": fps
        },
        "per_class": per_class_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    SAVE_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n==================================================================")
    print("🏆 KẾT QUẢ NGHIỆM THU MÔ HÌNH SIÊU NHẸ YOLO26-MICRO TRÊN 884 FRAMES:")
    print("==================================================================")
    print(f"• Overall Precision: {mp*100:.2f}%")
    print(f"• Overall Recall:    {mr*100:.2f}%")
    print(f"• Overall mAP@50:    {map50*100:.2f}%")
    print(f"• Overall mAP@50-95: {map50_95*100:.2f}%")
    print(f"• Tốc độ suy luận:   {infer_ms:.2f} ms ({fps} FPS)")
    print(f"• Đã lưu kết quả:    {SAVE_METRICS_JSON}")
    print("==================================================================")


if __name__ == "__main__":
    main()
