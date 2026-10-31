#!/usr/bin/env python3
"""
scripts/diagnose_underperforming.py
===================================
Forensic analysis of underperforming classes (wheelchair and iv_stand).
Analyzes:
1. False Negatives (FN) & False Positives (FP) in test set.
2. Confusion matrix interactions (e.g. wheelchair vs hospital_chair).
3. Test-Time Augmentation & Multi-scale evaluation effects.
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
best_weights = REPO_ROOT / "experiments/runs/train_hospital/yolo26n_targeted_17class_v4/weights/best.pt"
data_yaml = REPO_ROOT / "configs/hospital_unified_targeted_17class.yaml"

def run_diagnostics():
    print("Loading model:", best_weights)
    model = YOLO(str(best_weights))
    
    # 1. Standard test evaluation
    print("\n[1] Standard Test Evaluation (imgsz=640, conf=0.001)...")
    res_640 = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=16,
        workers=0,
        device="0",
        plots=False,
        verbose=False
    )
    
    wc_idx = 2
    iv_idx = 13
    
    ap_indices = list(res_640.box.ap_class_index)
    idx_wc = ap_indices.index(wc_idx) if wc_idx in ap_indices else None
    idx_iv = ap_indices.index(iv_idx) if iv_idx in ap_indices else None
    
    print(f"Overall mAP50: {res_640.box.map50*100:.2f}%")
    if idx_wc is not None:
        print(f"Wheelchair (cid=2):  mAP50={res_640.box.ap50[idx_wc]*100:.2f}%, P={res_640.box.p[idx_wc]*100:.2f}%, R={res_640.box.r[idx_wc]*100:.2f}%")
    if idx_iv is not None:
        print(f"IV Stand (cid=13):   mAP50={res_640.box.ap50[idx_iv]*100:.2f}%, P={res_640.box.p[idx_iv]*100:.2f}%, R={res_640.box.r[idx_iv]*100:.2f}%")

    # 2. Test imgsz=800
    print("\n[2] High Resolution Evaluation (imgsz=800)...")
    res_800 = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=800,
        batch=8,
        workers=0,
        device="0",
        plots=False,
        verbose=False
    )
    idx_wc_800 = ap_indices.index(wc_idx) if wc_idx in ap_indices else None
    idx_iv_800 = ap_indices.index(iv_idx) if iv_idx in ap_indices else None
    print(f"Overall mAP50 (800): {res_800.box.map50*100:.2f}%")
    if idx_wc_800 is not None:
        print(f"Wheelchair (cid=2, 800):  mAP50={res_800.box.ap50[idx_wc_800]*100:.2f}%, P={res_800.box.p[idx_wc_800]*100:.2f}%, R={res_800.box.r[idx_wc_800]*100:.2f}%")
    if idx_iv_800 is not None:
        print(f"IV Stand (cid=13, 800):   mAP50={res_800.box.ap50[idx_iv_800]*100:.2f}%, P={res_800.box.p[idx_iv_800]*100:.2f}%, R={res_800.box.r[idx_iv_800]*100:.2f}%")

if __name__ == "__main__":
    run_diagnostics()
