#!/usr/bin/env python3
"""
scripts/train/train_golden_v7_boost.py
======================================
Huấn luyện YOLO26n v7-MaxAccuracy (Kéo độ chính xác lên đỉnh cao):
1. Domain Warm-Start: Nạp 100% trọng số từ checkpoint v6 (yolo26n_golden_17class_v6).
2. Nâng phân giải lên imgsz=800 kết hợp multi_scale=0.15 để nhận diện sắc nét vật thể mảnh/xa.
3. Tăng gấp đôi trọng số phân loại (cls=2.0) để phạt nặng lỗi bỏ sót cáng cứu thương và cây nước.
4. Tăng cường Copy-Paste (0.3) và đóng sớm Mosaic 15 epochs để học trọn vẹn hình thái thực.
5. Tự động đánh giá nghiệm thu đối chuẩn song song ở cả 2 độ phân giải 640 và 800 trên 884 frames.
"""

import os
import sys
import time
import json
from pathlib import Path
import torch
from ultralytics import YOLO

os.environ["OMP_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "configs/hospital_unified_golden_17class.yaml"
BENCHMARK_YAML = REPO_ROOT / "configs/hospital_robot_final_benchmark_884.yaml"
V6_BEST_PT = REPO_ROOT / "experiments/runs/train_hospital/yolo26n_golden_17class_v6/weights/best.pt"
PROJECT_DIR = REPO_ROOT / "experiments/runs/train_hospital"
RUN_NAME = "yolo26n_golden_17class_v7_boost"
RUN_DIR = PROJECT_DIR / RUN_NAME
SAVE_METRICS_JSON = REPO_ROOT / "experiments/results/v7_boost_benchmark_metrics.json"

CLASSES = {
    0: "patient", 1: "doctor", 2: "wheelchair", 3: "queue_kiosk", 4: "reception_counter",
    5: "sanitizer_stand", 6: "pccc_station", 7: "medical_waste_bin", 8: "water_dispenser",
    9: "wc_station", 10: "elevator_doors", 11: "hospital_bed", 12: "stretcher_cart",
    13: "iv_stand", 14: "medical_trolley", 15: "vital_monitor", 16: "hospital_chair"
}


def main():
    if not V6_BEST_PT.exists():
        print(f"❌ Không tìm thấy checkpoint v6: {V6_BEST_PT}")
        return

    print("==================================================================")
    print("🚀 BẮT ĐẦU HUẤN LUYỆN YOLO26n v7-MAXACCURACY (GOLDEN ARCHITECTURE)")
    print(f"• Trọng số nền tảng: {V6_BEST_PT} (v6 91.03% mAP)")
    print(f"• Dataset YAML:       {DATA_YAML}")
    print(f"• Thư mục đích:       {RUN_DIR}")
    print(f"• Thiết bị:           GPU CUDA ({torch.cuda.get_device_name(0)})")
    print("• Kích thước ảnh:     800x800 (Tăng 56% pixel cho chi tiết nhỏ)")
    print("• Mục tiêu:           Đẩy mAP@50 lên 93.5% - 95.0% & giải quyết 2 lớp yếu")
    print("==================================================================")

    start_time = time.time()

    model = YOLO(str(V6_BEST_PT))
    model.train(
        data=str(DATA_YAML),
        epochs=35,
        batch=12,
        imgsz=800,
        device="0",
        workers=0,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0006,
        lrf=0.02,
        cos_lr=True,
        warmup_epochs=2.0,
        box=9.5,
        cls=2.0,            # Gấp đôi trọng số phân loại để kéo Recall
        dfl=2.0,
        copy_paste=0.3,     # Nhân bản vật thể hiếm
        mosaic=1.0,
        close_mosaic=15,    # Đóng sớm mosaic 15 epochs
        mixup=0.05,
        scale=0.5,
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
    print(f"\n✅ HUẤN LUYỆN v7-BOOST HOÀN TẤT TRONG {train_elapsed/60:.2f} PHÚT!")

    best_pt = RUN_DIR / "weights/best.pt"
    if not best_pt.exists():
        best_pt = RUN_DIR / "weights/last.pt"

    eval_model = YOLO(str(best_pt))

    # Đánh giá 1: Tại imgsz=800 (độ nét cao)
    print(f"\n🧊 [ĐÁNH GIÁ 1] NGHIỆM THU TẠI IMGSZ=800 TRÊN 884 TEST FRAMES...")
    res_800 = eval_model.val(
        data=str(BENCHMARK_YAML),
        split="test",
        imgsz=800,
        device="0",
        workers=0,
        plots=True,
        verbose=True
    )

    # Đánh giá 2: Tại imgsz=640 (chuẩn siêu tốc độ)
    print(f"\n⚡ [ĐÁNH GIÁ 2] NGHIỆM THU TẠI IMGSZ=640 TRÊN 884 TEST FRAMES...")
    res_640 = eval_model.val(
        data=str(BENCHMARK_YAML),
        split="test",
        imgsz=640,
        device="0",
        workers=0,
        plots=False,
        verbose=False
    )

    def extract_metrics(res, sz):
        d = res.results_dict
        per_c = {}
        if hasattr(res, "box") and hasattr(res.box, "ap50"):
            for cid, cname in CLASSES.items():
                if cid < len(res.box.ap50):
                    per_c[cname] = {
                        "precision": round(float(res.box.p[cid]) * 100, 2),
                        "recall": round(float(res.box.r[cid]) * 100, 2),
                        "map50": round(float(res.box.ap50[cid]) * 100, 2),
                        "map50_95": round(float(res.box.ap[cid]) * 100, 2)
                    }
        speed = res.speed.get("inference", 0.0)
        return {
            "imgsz": sz,
            "precision": round(float(d.get("metrics/precision(B)", 0.0)) * 100, 2),
            "recall": round(float(d.get("metrics/recall(B)", 0.0)) * 100, 2),
            "map50": round(float(d.get("metrics/mAP50(B)", 0.0)) * 100, 2),
            "map50_95": round(float(d.get("metrics/mAP50-95(B)", 0.0)) * 100, 2),
            "inference_ms": round(float(speed), 2),
            "fps": round(1000.0 / float(speed), 1) if float(speed) > 0 else 0.0,
            "per_class": per_c
        }

    m800 = extract_metrics(res_800, 800)
    m640 = extract_metrics(res_640, 640)

    output_payload = {
        "model_name": "YOLO26n_Golden_v7_Boost",
        "model_path": str(best_pt),
        "total_params": 2393623,
        "gflops": 5.4,
        "train_time_minutes": round(train_elapsed / 60, 2),
        "benchmark_800": m800,
        "benchmark_640": m640,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    SAVE_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n==================================================================")
    print("🏆 KẾT QUẢ NGHIỆM THU ĐỈNH CAO YOLO26n v7-MAXACCURACY (884 FRAMES):")
    print("==================================================================")
    print(f"• Tại imgsz=800: mAP@50 = {m800['map50']:.2f}% | Recall = {m800['recall']:.2f}% | Latency = {m800['inference_ms']} ms ({m800['fps']} FPS)")
    print(f"• Tại imgsz=640: mAP@50 = {m640['map50']:.2f}% | Recall = {m640['recall']:.2f}% | Latency = {m640['inference_ms']} ms ({m640['fps']} FPS)")
    print(f"• File kết quả:  {SAVE_METRICS_JSON}")
    print("==================================================================")


if __name__ == "__main__":
    main()
