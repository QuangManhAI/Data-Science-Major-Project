#!/usr/bin/env python3
import json
import time
from pathlib import Path
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
best_weights = REPO_ROOT / "experiments/runs/train_hospital/yolo26n_targeted_17class_v4/weights/best.pt"
data_yaml = REPO_ROOT / "configs/hospital_unified_targeted_17class.yaml"
save_json = REPO_ROOT / "experiments/results/targeted_17class_v4_metrics.json"

UNIFIED_CLASSES = {
    0: "patient", 1: "doctor", 2: "wheelchair", 3: "queue_kiosk", 4: "reception_counter",
    5: "sanitizer_stand", 6: "pccc_station", 7: "medical_waste_bin", 8: "water_dispenser",
    9: "wc_station", 10: "elevator_doors", 11: "hospital_bed", 12: "stretcher_cart",
    13: "iv_stand", 14: "medical_trolley", 15: "vital_monitor", 16: "hospital_chair"
}

def main():
    print("🧊 ĐÁNH GIÁ TRÊN TẬP TEST ĐÓNG BĂNG (split='test')...")
    eval_model = YOLO(str(best_weights))
    test_results = eval_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=16,
        device="0",
        plots=True,
        verbose=True
    )

    box = test_results.box
    map50_all = float(box.map50)
    map50_95_all = float(box.map)
    p_all = float(box.mp)
    r_all = float(box.mr)
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

    above_90 = sum(1 for v in per_class_stats.values() if v["pass_90"])

    results_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": "YOLO26n-Targeted-17Class-v4",
        "dataset": "hospital_unified_targeted_4120_17class",
        "total_classes": 17,
        "test_frames": 468,
        "hardware": {
            "device": "0",
            "device_name": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        },
        "training_config": {
            "epochs": 45,
            "batch_size": 16,
            "imgsz": 640,
            "optimizer": "AdamW",
            "box_gain": 8.8,
            "dfl_gain": 1.6,
            "copy_paste": 0.3,
            "mosaic": 1.0,
            "close_mosaic": 10,
            "mixup": 0.15,
            "scale": 0.5,
            "initial_weights": "models/yolo26n/yolo26n.pt"
        },
        "metrics": {
            "mAP50": round(map50_all, 4),
            "mAP50_95": round(map50_95_all, 4),
            "precision": round(p_all, 4),
            "recall": round(r_all, 4),
            "f1": round(f1_all, 4),
            "speed": {k: round(v, 2) for k, v in getattr(test_results, "speed", {}).items()},
            "classes_above_90_count": above_90,
            "per_class": per_class_stats
        },
        "best_weights": str(best_weights)
    }

    save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Đã lưu kết quả đo lường vào: {save_json}")
    print("\n==================================================================")
    print("🏆 BÁO CÁO NGHIỆM THU TẬP TEST ĐÓNG BĂNG (17 CLASSES) - v4:")
    print("==================================================================")
    print(f"• mAP@50 Tổng thể:    {map50_all*100:.2f}%")
    print(f"• mAP@50-95 Tổng thể: {map50_95_all*100:.2f}%")
    print(f"• Precision:          {p_all*100:.2f}%")
    print(f"• Recall:             {r_all*100:.2f}%")
    print(f"• F1-Score:           {f1_all*100:.2f}%")
    print(f"• Số lớp đạt >= 90%:  {above_90}/17 lớp")
    print("-" * 65)
    for cname, stats in per_class_stats.items():
        pass_str = "✅ ĐẠT" if stats["pass_90"] else "❌ CHƯA ĐẠT"
        print(f"{stats['class_id']:<5} {cname:<19} {stats['mAP50']*100:<9.1f}% {stats['mAP50_95']*100:<11.1f}% {stats['recall']*100:<9.1f}% {pass_str}")
    print("==================================================================")

if __name__ == "__main__":
    main()
