#!/usr/bin/env python3
"""extract_detailed_benchmark_stats.py
Trích xuất thống kê toàn diện về toàn bộ fail cases và % nhận diện thấp
trên cả 2 datasets (12,000 frames) x 4 models.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

with open(REPO_ROOT / "sim_vision/reports/vision_model_benchmark.json") as f:
    d1 = json.load(f)

print("=== DATASET 1 (1200 frames — 8 Trục Baseline D1 đến D8) ===")
d1_table = []
for m in d1:
    m_name = m["model_name"]
    p_ov = m["precision"] * 100
    r_ov = m["recall_overall"] * 100
    r_wc = m.get("recall_wheelchair", 0) * 100
    r_occ = m.get("recall_occluded", 0) * 100
    fpr = m.get("fpr_empty", 0) * 100
    lat = m.get("latency_p50_ms", 0)
    print(f"\n[Model: {m_name}] P={p_ov:.1f}% | R={r_ov:.1f}% | R_Wheelchair={r_wc:.1f}% | R_Occ={r_occ:.1f}% | FPR_Empty={fpr:.1f}% | Latency={lat:.1f}ms")
    for sl, metrics in m["slice_metrics"].items():
        rec = metrics.get("recall", metrics.get("recall_overall", 0)) * 100
        fpr_sl = metrics.get("fpr_empty", 0) * 100
        tp = metrics.get("tp", 0)
        fn = metrics.get("fn", 0)
        fp = metrics.get("fp", 0)
        print(f"  {sl:25s}: Recall={rec:5.1f}% | FPR_Empty={fpr_sl:5.1f}% | TP={tp:4d}, FN={fn:4d}, FP={fp:4d}")

with open(REPO_ROOT / "sim_vision/reports/vision_extended_benchmark.json") as f:
    d2 = json.load(f)

print("\n=== DATASET 2 (10,800 frames — 6 Phân Nhóm Lâm Sàng N1 đến N6) ===")
for m in d2:
    m_name = m["model_name"]
    p_ov = m["precision"] * 100
    r_ov = m["recall_overall"] * 100
    fpr = m.get("fpr_empty", 0) * 100
    lat = m.get("latency_p50_ms", 0)
    print(f"\n[Model: {m_name}] P={p_ov:.1f}% | R={r_ov:.1f}% | FPR_Empty={fpr:.1f}% | Latency={lat:.1f}ms")
    for sl, metrics in m["slice_metrics"].items():
        rec = metrics.get("recall", metrics.get("recall_overall", 0)) * 100
        fpr_sl = metrics.get("fpr_empty", 0) * 100
        tp = metrics.get("tp", 0)
        fn = metrics.get("fn", 0)
        fp = metrics.get("fp", 0)
        print(f"  {sl:30s}: Recall={rec:5.1f}% | FPR_Empty={fpr_sl:5.1f}% | TP={tp:4d}, FN={fn:4d}, FP={fp:4d}")
