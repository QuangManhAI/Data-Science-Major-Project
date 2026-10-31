#!/usr/bin/env python3
"""
scripts/eval_robot_patrol.py
============================
Đánh giá nghiệm thu đối chuẩn song song giữa YOLO26n v4 và YOLO26n v5
trực tiếp trên Bộ Test Hành Trình Tuần Tra Thực Tế Của Robot (531 frames):
- Đo lường mAP@50, Precision, Recall, F1.
- So sánh bước nhảy vọt giữa 2 phiên bản trên góc nhìn camera robot di chuyển thực tế.
- Xuất kết quả vào: experiments/results/robot_patrol_benchmark_metrics.json
"""

import json
import time
from pathlib import Path
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML = REPO_ROOT / "configs/hospital_robot_patrol_test.yaml"
V4_WEIGHTS = REPO_ROOT / "experiments/runs/train_hospital/yolo26n_targeted_17class_v4/weights/best.pt"
V5_WEIGHTS = REPO_ROOT / "experiments/runs/train_hospital/yolo26n_targeted_17class_v5/weights/best.pt"
SAVE_JSON = REPO_ROOT / "experiments/results/robot_patrol_benchmark_metrics.json"

CLASSES = {
    0: "patient", 1: "doctor", 2: "wheelchair", 3: "queue_kiosk", 4: "reception_counter",
    5: "sanitizer_stand", 6: "pccc_station", 7: "medical_waste_bin", 8: "water_dispenser",
    9: "wc_station", 10: "elevator_doors", 11: "hospital_bed", 12: "stretcher_cart",
    13: "iv_stand", 14: "medical_trolley", 15: "vital_monitor", 16: "hospital_chair"
}


def evaluate_model(weights_path: Path, model_tag: str):
    print(f"\n🔍 Đang đánh giá {model_tag} ({weights_path.name})...")
    model = YOLO(str(weights_path))
    results = model.val(
        data=str(CONFIG_YAML),
        split="test",
        imgsz=640,
        batch=16,
        workers=0,
        device="0",
        plots=False,
        verbose=False
    )

    box = results.box
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

    return {
        "model_tag": model_tag,
        "weights": str(weights_path),
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


def main():
    print("==================================================================")
    print("🤖 ĐÁNH GIÁ ĐỐI CHUẨN TRÊN BỘ TEST HÀNH TRÌNH ROBOT TUẦN TRA (531 FRAMES)")
    print("==================================================================")

    res_v4 = evaluate_model(V4_WEIGHTS, "YOLO26n-v4 (Targeted R1)")
    res_v5 = evaluate_model(V5_WEIGHTS, "YOLO26n-v5 (Targeted R2 - Latest)")

    actual_boxes = 0
    lbl_dir = REPO_ROOT / "data/vision/dataset/hospital_robot_patrol_testset/labels"
    for lf in lbl_dir.glob("*.txt"):
        with open(lf, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    actual_boxes += 1

    output_payload = {
        "benchmark_name": "Hospital_Robot_Patrol_Continuous_Testset",
        "total_frames": 531,
        "total_boxes": actual_boxes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "v4_results": res_v4,
        "v5_results": res_v5
    }

    SAVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Đã lưu kết quả đo lường vào: {SAVE_JSON}")
    print("\n==================================================================")
    print("🏆 BẢNG SO SÁNH HIỆU NĂNG GÓC NHÌN ROBOT THỰC TẾ (531 PATROL FRAMES):")
    print("==================================================================")
    print(f"{'Chỉ số / Lớp':<22} {'YOLO26n v4':<15} {'YOLO26n v5':<15} {'Biến thiên (Delta)':<15}")
    print("-" * 68)
    print(f"{'Overall mAP@50':<22} {res_v4['overall']['mAP50']*100:<14.2f}% {res_v5['overall']['mAP50']*100:<14.2f}% {res_v5['overall']['mAP50']*100 - res_v4['overall']['mAP50']*100:+6.2f}%")
    print(f"{'Overall Recall':<22} {res_v4['overall']['recall']*100:<14.2f}% {res_v5['overall']['recall']*100:<14.2f}% {res_v5['overall']['recall']*100 - res_v4['overall']['recall']*100:+6.2f}%")
    print(f"{'Overall Precision':<22} {res_v4['overall']['precision']*100:<14.2f}% {res_v5['overall']['precision']*100:<14.2f}% {res_v5['overall']['precision']*100 - res_v4['overall']['precision']*100:+6.2f}%")
    print(f"{'Số lớp >= 90%':<22} {res_v4['overall']['classes_above_90']}/17 {'':<9} {res_v5['overall']['classes_above_90']}/17")
    print("-" * 68)
    for cname in CLASSES.values():
        v4_val = res_v4['per_class'][cname]['mAP50'] * 100
        v5_val = res_v5['per_class'][cname]['mAP50'] * 100
        delta = v5_val - v4_val
        star = "⭐" if cname in ["wheelchair", "iv_stand", "patient"] else "  "
        print(f"{star} {cname:<19} {v4_val:6.2f}%        {v5_val:6.2f}%        {delta:+6.2f}%")
    print("==================================================================")


if __name__ == "__main__":
    main()
