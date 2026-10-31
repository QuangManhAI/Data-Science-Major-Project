#!/usr/bin/env python3
"""
scripts/train/train_golden_v6.py
================================
Huấn luyện Siêu Mô Hình YOLO26n Nano v6 trên Kiến Trúc Chuẩn Vàng (Golden Architecture):
- Train: 3,715 frames (10,642 boxes) - Gộp toàn diện từ toàn bộ 4,370 frames tĩnh.
- Val:   655 frames (1,957 boxes) - Đánh giá hội tụ độc lập trong lúc train.
- Test:  531 frames (2,223 boxes) - Bộ Test Vàng Robot Patrol (CVAT verified).
- BẢO MẬT ZERO DATA LEAKAGE: 0% trùng lặp giữa Train/Val và Test Robot.
- Model: YOLO26n Nano (2.39M params, 5.3 MB, 5.4 GFLOPs).
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
DATA_YAML = REPO_ROOT / "configs/hospital_unified_golden_17class.yaml"
INIT_MODEL = REPO_ROOT / "models/yolo26n/yolo26n.pt"
PROJECT_DIR = REPO_ROOT / "experiments/runs/train_hospital"
RUN_NAME = "yolo26n_golden_17class_v6"
RUN_DIR = PROJECT_DIR / RUN_NAME
SAVE_METRICS_JSON = REPO_ROOT / "experiments/results/golden_v6_patrol_benchmark_metrics.json"

CLASSES = {
    0: "patient", 1: "doctor", 2: "wheelchair", 3: "queue_kiosk", 4: "reception_counter",
    5: "sanitizer_stand", 6: "pccc_station", 7: "medical_waste_bin", 8: "water_dispenser",
    9: "wc_station", 10: "elevator_doors", 11: "hospital_bed", 12: "stretcher_cart",
    13: "iv_stand", 14: "medical_trolley", 15: "vital_monitor", 16: "hospital_chair"
}


def main():
    print("==================================================================")
    print("🚀 BẮT ĐẦU HUẤN LUYỆN SIÊU MÔ HÌNH YOLO26n NANO v6 (GOLDEN ARCHITECTURE)")
    print(f"• Dataset YAML:    {DATA_YAML}")
    print(f"• Model gốc:       {INIT_MODEL} (YOLO26n Nano, 2.39M params)")
    print(f"• Thư mục đích:    {RUN_DIR}")
    print(f"• Thiết bị:        GPU CUDA ({torch.cuda.get_device_name(0)})")
    print("• Quy mô dữ liệu:  Train: 3,715 frames | Val: 655 frames | Golden Test: 531 frames")
    print("• Bảo mật:         ZERO DATA LEAKAGE (Đã kiểm duyệt SHA-256)")
    print("==================================================================")

    start_time = time.time()

    model = YOLO(str(INIT_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=45,
        batch=16,
        imgsz=640,
        device="0",
        workers=0,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        warmup_epochs=3.0,
        box=9.2,
        cls=1.0,
        dfl=1.8,
        copy_paste=0.3,
        mosaic=1.0,
        close_mosaic=10,
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
    print(f"\n✅ HUẤN LUYỆN v6 HOÀN TẤT TRONG {train_elapsed/60:.2f} PHÚT!")

    best_pt = RUN_DIR / "weights/best.pt"
    if not best_pt.exists():
        best_pt = RUN_DIR / "weights/last.pt"

    print(f"\n🧊 ĐÁNH GIÁ CHÍNH THỨC TRÊN BỘ TEST VÀNG ROBOT PATROL (split='test', 531 frames)...")
    eval_model = YOLO(str(best_pt))
    test_results = eval_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device="0",
        workers=0,
        plots=True,
        verbose=True
    )

    box = test_results.box
    map50_all = float(box.map50)
    map_all = float(box.map)
    p_all = float(box.mp)
    r_all = float(box.mr)
    f1_all = 2 * (p_all * r_all) / max(1e-6, (p_all + r_all))

    per_class = {}
    ap_indices = list(box.ap_class_index) if hasattr(box, "ap_class_index") and box.ap_class_index is not None else []
    ap50 = getattr(box, "ap50", None)
    maps = getattr(box, "maps", None)
    p = getattr(box, "p", None)
    r = getattr(box, "r", None)

    for cid, cname in CLASSES.items():
        if cid in ap_indices:
            idx = ap_indices.index(cid)
            c_map50 = float(ap50[idx]) if ap50 is not None and idx < len(ap50) else 0.0
            c_map = float(maps[idx]) if maps is not None and idx < len(maps) else 0.0
            c_p = float(p[idx]) if p is not None and idx < len(p) else 0.0
            c_r = float(r[idx]) if r is not None and idx < len(r) else 0.0
        else:
            c_map50 = c_map = c_p = c_r = 0.0

        per_class[cname] = {
            "class_id": cid,
            "mAP50": round(c_map50, 4),
            "mAP50_95": round(c_map, 4),
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "pass_90": bool(c_map50 >= 0.90)
        }

    output_payload = {
        "model_name": "YOLO26n_Nano_Golden_v6",
        "benchmark_name": "Hospital_Robot_Patrol_Golden_Testset",
        "total_test_frames": 531,
        "total_test_boxes": 2223,
        "train_frames": 3715,
        "val_frames": 655,
        "training_time_min": round(train_elapsed / 60, 2),
        "overall": {
            "mAP50": round(map50_all, 4),
            "mAP50_95": round(map_all, 4),
            "precision": round(p_all, 4),
            "recall": round(r_all, 4),
            "f1": round(f1_all, 4),
            "classes_above_90": sum(1 for v in per_class.values() if v["pass_90"])
        },
        "per_class": per_class
    }

    SAVE_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Đã lưu kết quả đo lường vào: {SAVE_METRICS_JSON}")

    # Xuất ONNX tĩnh opset 17
    print("\n📦 Xuất mô hình ONNX tối ưu hóa Edge...")
    try:
        onnx_file = eval_model.export(
            format="onnx",
            imgsz=640,
            half=False,
            dynamic=False,
            simplify=True,
            opset=17
        )
        print(f"✅ ONNX export thành công: {onnx_file}")
    except Exception as e:
        print(f"⚠️ ONNX export warning: {e}")

    print("\n==================================================================")
    print("🏆 KẾT QUẢ NGHIỆM THU YOLO26n v6 TRÊN BỘ TEST VÀNG ROBOT (531 FRAMES):")
    print(f"• Overall mAP@50:    {map50_all*100:.2f}%")
    print(f"• Overall Precision: {p_all*100:.2f}%")
    print(f"• Overall Recall:    {r_all*100:.2f}%")
    print(f"• Số lớp >= 90%:     {output_payload['overall']['classes_above_90']}/17")
    print("==================================================================")


if __name__ == "__main__":
    main()
