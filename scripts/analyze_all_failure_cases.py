#!/usr/bin/env python3
"""analyze_all_failure_cases.py
Phân tích chi tiết toàn bộ các ca phát hiện tệ (Failures / Degradations)
trên cả 84 ca kiểm thử x 4 mô hình AI đối chứng.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "sim_vision/python"))
from detector import YoloDetector, IntelOpenVINODetector
from scripts.reorganize_samples_and_generate_all_cases import DATASET_1_CASES, DATASET_2_CASES

v8 = YoloDetector(str(REPO_ROOT / "models/yolov8n/yolov8n.pt"), conf=0.45)
v11 = YoloDetector(str(REPO_ROOT / "models/yolo11n/yolo11n.pt"), conf=0.45)
v26 = YoloDetector(str(REPO_ROOT / "models/yolo26n/yolo26n.pt"), conf=0.45)
intel = IntelOpenVINODetector(str(REPO_ROOT / "models/intel_person/person-detection-retail-0013.xml"), conf=0.45)
models = {"YOLOv8n": v8, "YOLO11n": v11, "YOLO26n": v26, "Intel OpenVINO": intel}

ann1_map = {}
with open(REPO_ROOT / "sim_vision/data/vision_eval_1200/annotations.jsonl") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            ann1_map[d["frame_id"]] = d

ann2_map = {}
with open(REPO_ROOT / "sim_vision/data/vision_eval_extended_10800/annotations.jsonl") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            ann2_map[d["frame_id"]] = d


def compute_iou(b1, b2):
    xx1 = max(b1[0], b2[0])
    yy1 = max(b1[1], b2[1])
    xx2 = min(b1[2], b2[2])
    yy2 = min(b1[3], b2[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


bad_cases = []

# Dataset 1
for cid, sl, f_idx, slug, title in DATASET_1_CASES:
    is_empty = (sl == "D8")
    group = f"{sl}_Baseline"
    img_path = REPO_ROOT / f"sim_vision/data/samples/dataset_1_baseline_48cases/{cid}_{slug}.jpg"
    frame_id = f"{sl}_{f_idx:04d}"
    gts = ann1_map.get(frame_id, {}).get("ground_truths", [])
    img = cv2.imread(str(img_path))
    h_orig, w_orig = img.shape[:2]

    case_res = {
        "dataset": "Dataset 1 (Baseline)",
        "cid": cid,
        "title": title,
        "group": group,
        "is_empty": is_empty,
        "gts_count": len(gts),
        "failures": {}
    }
    has_issue = False

    for m_name, det in models.items():
        preds, _ = det.detect(img)
        if is_empty:
            if len(preds) > 0:
                has_issue = True
                max_c = max([p["conf"] for p in preds])
                case_res["failures"][m_name] = {
                    "type": "FP_HALLUCINATION",
                    "details": f"{len(preds)} FP boxes, max_conf={max_c:.2f}"
                }
        else:
            if len(preds) == 0:
                has_issue = True
                case_res["failures"][m_name] = {
                    "type": "FN_TOTAL_MISS",
                    "details": f"Bỏ sót hoàn toàn 0/{len(gts)} người"
                }
            elif len(preds) < len(gts):
                has_issue = True
                case_res["failures"][m_name] = {
                    "type": "FN_PARTIAL_NMS",
                    "details": f"Bỏ sót {len(gts)-len(preds)} người ({len(preds)}/{len(gts)} detected)"
                }
            else:
                ious = []
                for g in gts:
                    best = max([
                        compute_iou(
                            [p["x1"]*w_orig, p["y1"]*h_orig, p["x2"]*w_orig, p["y2"]*h_orig],
                            g["bbox_xyxy"]
                        ) for p in preds
                    ])
                    ious.append(best)
                min_iou = min(ious) if ious else 0
                if min_iou < 0.40:
                    has_issue = True
                    case_res["failures"][m_name] = {
                        "type": "LOW_IOU_DRIFT",
                        "details": f"Sai lệch biên hộp bao (IoU={min_iou:.2f} < 0.40)"
                    }

    if has_issue:
        bad_cases.append(case_res)

# Dataset 2
for cid, case_dir, f_idx, slug, title in DATASET_2_CASES:
    is_empty = (cid.startswith("N4"))
    group = f"{cid.split('.')[0]}_Clinical"
    slug_name = cid.replace(".", "_")
    img_path = REPO_ROOT / f"sim_vision/data/samples/dataset_2_clinical_36cases/{slug_name}_{slug}.jpg"
    frame_id = f"{case_dir}_{f_idx:04d}"
    gts = ann2_map.get(frame_id, {}).get("ground_truth_annotations", [])
    img = cv2.imread(str(img_path))
    h_orig, w_orig = img.shape[:2]

    case_res = {
        "dataset": "Dataset 2 (Clinical)",
        "cid": cid,
        "title": title,
        "group": group,
        "is_empty": is_empty,
        "gts_count": len(gts),
        "failures": {}
    }
    has_issue = False

    for m_name, det in models.items():
        preds, _ = det.detect(img)
        if is_empty:
            if len(preds) > 0:
                has_issue = True
                max_c = max([p["conf"] for p in preds])
                case_res["failures"][m_name] = {
                    "type": "FP_HALLUCINATION",
                    "details": f"{len(preds)} FP boxes, max_conf={max_c:.2f}"
                }
        else:
            if len(preds) == 0:
                has_issue = True
                case_res["failures"][m_name] = {
                    "type": "FN_TOTAL_MISS",
                    "details": f"Bỏ sót hoàn toàn 0/{len(gts)} người"
                }
            elif len(preds) < len(gts):
                has_issue = True
                case_res["failures"][m_name] = {
                    "type": "FN_PARTIAL_NMS",
                    "details": f"Bỏ sót {len(gts)-len(preds)} người ({len(preds)}/{len(gts)} detected)"
                }
            else:
                ious = []
                for g in gts:
                    best = max([
                        compute_iou(
                            [p["x1"]*w_orig, p["y1"]*h_orig, p["x2"]*w_orig, p["y2"]*h_orig],
                            g["bbox_xyxy_px"]
                        ) for p in preds
                    ])
                    ious.append(best)
                min_iou = min(ious) if ious else 0
                if min_iou < 0.40:
                    has_issue = True
                    case_res["failures"][m_name] = {
                        "type": "LOW_IOU_DRIFT",
                        "details": f"Sai lệch biên hộp bao (IoU={min_iou:.2f} < 0.40)"
                    }

    if has_issue:
        bad_cases.append(case_res)

output_json = REPO_ROOT / "sim_vision/reports/bad_detection_cases_summary.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(bad_cases, f, indent=2, ensure_ascii=False)

print(f"Tổng số ca kiểm thử có ít nhất 1 mô hình gặp lỗi phát hiện: {len(bad_cases)} / 84 ca")
print(f"Đã lưu chi tiết phân tích lỗi tại: {output_json}")

# Thống kê phân loại lỗi
types = {}
model_errs = {m: 0 for m in models}
for b in bad_cases:
    for m, info in b["failures"].items():
        t = info["type"]
        types[t] = types.get(t, 0) + 1
        model_errs[m] += 1

print("\nPhân bố loại lỗi phát hiện:")
for t, cnt in types.items():
    print(f"  - {t}: {cnt} lần")

print("\nSố lượng ca lỗi theo từng mô hình:")
for m, cnt in model_errs.items():
    print(f"  - {m}: {cnt} ca lỗi")
