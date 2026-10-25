#!/usr/bin/env python3
"""
audit_quality.py
Script kiểm định chất lượng nhãn Ground Truth (Human vs AI, hoặc Gold vs Pre-annotated).
Tính toán độ trùng khớp IoU, phát hiện nhãn lệch, nhãn sót (FN) và nhãn ảo (FP),
kiểm tra xem đợt dữ liệu có đạt tỷ lệ Audit Pass >= 95% theo yêu cầu của VISION_DEEP_PLAN.md hay không.
"""

import argparse
import json
from pathlib import Path
import numpy as np

def compute_iou(box1, box2):
    # box: [xc, yc, w, h] normalized
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def read_yolo_file(file_path):
    boxes = []
    if not file_path.exists():
        return boxes
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append({"cls": cls_id, "bbox": [xc, yc, w, h]})
    return boxes

def main():
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Kiểm định chất lượng gán nhãn")
    parser.add_argument("--gt-dir", type=str, default=str(repo_root / "data/vision/raw/labels"), help="Thư mục nhãn Ground Truth (Godot Physics/Human)")
    parser.add_argument("--pred-dir", type=str, default=str(repo_root / "data/vision/in_review/labels"), help="Thư mục nhãn AI dự đoán / Pre-annotated")
    parser.add_argument("--iou-thresh", type=float, default=0.70, help="Ngưỡng IoU coi là khớp đúng")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    pred_files = list(pred_dir.glob("*.txt"))
    if not pred_files:
        print(f"❌ Không tìm thấy file nhãn nháp nào tại {pred_dir}")
        return

    # Chỉ kiểm định những file có trong tập review
    audit_files = [f for f in pred_files if (gt_dir / f.name).exists()]
    if not audit_files:
        print(f"❌ Không có file nào trùng khớp giữa {pred_dir} và {gt_dir}")
        return

    print("==================================================================")
    print("🔬 BẮT ĐẦU KIỂM ĐỊNH CHẤT LƯỢNG NHÃN (QUALITY AUDIT)")
    print(f"• Số lượng frames trong đợt kiểm định: {len(audit_files)}")
    print(f"• Ground Truth Dir: {gt_dir}")
    print(f"• Predicted Dir:    {pred_dir}")
    print(f"• Ngưỡng IoU Pass:  {args.iou_thresh}")
    print("==================================================================")

    total_gt_boxes = 0
    total_pred_boxes = 0
    matched_boxes = 0
    ious = []
    failed_frames = []

    for pred_file in audit_files:
        gt_file = gt_dir / pred_file.name
        gt_boxes = read_yolo_file(gt_file)
        pred_boxes = read_yolo_file(pred_file)

        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(pred_boxes)

        matched_in_frame = 0
        used_preds = set()

        for g in gt_boxes:
            best_iou = 0.0
            best_p_idx = -1
            for p_idx, p in enumerate(pred_boxes):
                if p_idx in used_preds:
                    continue
                if p["cls"] == g["cls"]:
                    iou = compute_iou(g["bbox"], p["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_p_idx = p_idx
            
            if best_iou >= args.iou_thresh:
                matched_boxes += 1
                matched_in_frame += 1
                used_preds.add(best_p_idx)
                ious.append(best_iou)

        if len(gt_boxes) > 0 and matched_in_frame < len(gt_boxes):
            failed_frames.append({
                "frame": gt_file.stem,
                "gt_count": len(gt_boxes),
                "matched": matched_in_frame,
                "missing": len(gt_boxes) - matched_in_frame
            })

    precision = matched_boxes / total_pred_boxes if total_pred_boxes > 0 else 0
    recall = matched_boxes / total_gt_boxes if total_gt_boxes > 0 else 0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    pass_rate = (len(audit_files) - len(failed_frames)) / len(audit_files) * 100 if audit_files else 0

    print("\n==================================================================")
    print("🏆 BÁO CÁO KIỂM ĐỊNH CHẤT LƯỢNG (AUDIT REPORT):")
    print(f"• Tổng số frames kiểm định:    {len(audit_files)}")
    print(f"• Tỷ lệ Pass Frame (100% khớp): {pass_rate:.1f}%")
    print(f"• Độ khớp IoU trung bình:     {mean_iou * 100:.2f}%")
    print(f"• Precision đối chiếu:         {precision * 100:.2f}%")
    print(f"• Recall đối chiếu:            {recall * 100:.2f}%")
    print(f"• Số frame cần can thiệp tay:  {len(failed_frames)}")
    if pass_rate >= 95.0:
        print("✅ KẾT LUẬN: ĐẠT TIÊU CHUẨN AUDIT (Pass Rate >= 95%)! An toàn cho Auto-Accept.")
    else:
        print("⚠️ KẾT LUẬN: CHƯA ĐẠT CHỈ TIÊU 95%. Yêu cầu Human-in-the-Loop review tiếp tục.")
    print("==================================================================")

if __name__ == "__main__":
    main()
