#!/usr/bin/env python3
"""
render_visualized_groundtruth.py
Vẽ song song Bounding Box Ground Truth Đa Lớp (Person: Xanh Lá / Wheelchair: Vàng Gold)
và Model Prediction (Cyan / Đỏ) cùng siêu dữ liệu trực quan lên toàn bộ ảnh (648 frames)
để người dùng đối chứng trực tiếp từng khung hình.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import shutil
from ultralytics import YOLO

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(areaA + areaB - interArea)

def render_groundtruth_and_predictions(model_path: str = "models/yolo26n/yolo26n.pt", conf_thres: float = 0.25, iou_thres: float = 0.50):
    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = repo_root / "data/vision/vision_eval_unified"
    if not dataset_dir.exists():
        dataset_dir = repo_root / "data/vision_eval_unified"
    img_dir = dataset_dir / "images"
    jsonl_file = dataset_dir / "annotations_rich_5w1h.jsonl"
    if not jsonl_file.exists():
        jsonl_file = dataset_dir / "annotations.jsonl"

    out_vis_dir = repo_root / "data/vision/vision_eval_groundtruth_visualized"

    # Reset output directory
    if out_vis_dir.exists():
        shutil.rmtree(out_vis_dir)
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    # Load Model (YOLO26n / specified)
    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = repo_root / model_path
    
    model_name_label = model_file.stem.upper()
    print("==================================================================")
    print(f"🤖 Đang nạp mô hình Object Detection: {model_file.name} ({model_name_label})...")
    model = YOLO(str(model_file))
    print(f"📁 Thư mục xuất ảnh trực quan: {out_vis_dir}")
    print("==================================================================")

    # Load annotations
    annotations = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                annotations[data["frame_id"]] = data

    print(f"🎨 BẮT ĐẦU VẼ ĐỐI CHỨNG DUAL-BOX (GROUND TRUTH MULTI-CLASS + {model_name_label}) CHO {len(annotations)} FRAMES...")
    t0 = time.time()

    rendered_count = 0
    group_counts = {}

    total_person_gt_all = 0
    total_wheelchair_gt_all = 0
    total_tp_all = 0
    total_fp_all = 0
    total_fn_all = 0

    for fid, meta in annotations.items():
        img_p = img_dir / f"{fid}.jpg"
        if not img_p.exists():
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        gid = meta.get("group_id", "Other")
        cid = meta.get("case_id", "Unknown")
        name_vi = meta.get("name_vi", "")
        is_hard_neg = meta.get("is_hard_negative", False) or cid.startswith("U8")

        # Subdirectory for group
        target_group_dir = out_vis_dir / gid
        target_group_dir.mkdir(parents=True, exist_ok=True)

        gts = meta.get("ground_truth_annotations", [])
        if not gts and "zero_defect_boxes" in meta:
            gts = meta["zero_defect_boxes"]

        # Phân tách Person GT và Wheelchair GT
        person_gts = [g for g in gts if g.get("category_id", 0) == 0]
        wheelchair_gts = [g for g in gts if g.get("category_id", 0) == 1]

        total_person_gt_all += len(person_gts)
        total_wheelchair_gt_all += len(wheelchair_gts)

        # Run model inference on this image (Person class = 0)
        res = model(str(img_p), conf=conf_thres, classes=[0], verbose=False)[0]
        preds = []
        for box in res.boxes:
            b_coords = box.xyxy[0].cpu().numpy().tolist()
            b_conf = float(box.conf[0].cpu().numpy())
            preds.append({
                "bbox": b_coords,
                "conf": b_conf,
                "matched_gt_idx": -1,
                "matched_iou": 0.0
            })

        # Match person predictions with person GT
        gt_matched = [False] * len(person_gts)
        matched_gt_ious = [0.0] * len(person_gts)

        for p_idx, pred in enumerate(preds):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, gt in enumerate(person_gts):
                gt_box = gt.get("bbox_xyxy_px") or gt.get("bbox_xyxy", [])
                if len(gt_box) < 4:
                    continue
                iou = compute_iou(gt_box, pred["bbox"])
                if iou >= iou_thres and iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_g_idx >= 0 and not gt_matched[best_g_idx]:
                gt_matched[best_g_idx] = True
                matched_gt_ious[best_g_idx] = best_iou
                pred["matched_gt_idx"] = best_g_idx
                pred["matched_iou"] = best_iou

        tp_count = sum(1 for p in preds if p["matched_gt_idx"] >= 0)
        fp_count = len(preds) - tp_count
        fn_count = len(person_gts) - sum(gt_matched)

        total_tp_all += tp_count
        total_fp_all += fp_count
        total_fn_all += fn_count

        # ==============================================================================
        # DRAW TOP HUD HEADER
        # ==============================================================================
        cv2.rectangle(img, (0, 0), (w, 46), (20, 20, 20), -1)
        # Line 1: Case Title
        title_text = f"[{cid}] {name_vi} | Frame: {fid}"
        cv2.putText(img, title_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Line 2: Legend & Stats bar
        wc_info = f"GT WC [GOLD]: {len(wheelchair_gts)}  |  " if wheelchair_gts else ""
        hud_legend = (
            f"GT Person: {len(person_gts)}  |  "
            f"{wc_info}"
            f"{model_name_label} TP [CYAN]: {tp_count}  |  "
            f"FP [RED]: {fp_count}  |  "
            f"FN [MAGENTA]: {fn_count}"
        )
        cv2.putText(img, hud_legend, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 240, 255), 1, cv2.LINE_AA)

        # ==============================================================================
        # DRAW BOTTOM FOOTER
        # ==============================================================================
        cv2.rectangle(img, (0, h - 26), (w, h), (20, 20, 20), -1)
        ctx = meta.get("context", {})
        dist = ctx.get("distance_m", "?")
        yaw = ctx.get("cam_yaw_deg", "?")
        cam_h = ctx.get("cam_height_m", "?")
        dist_str = f"{dist:.1f}m" if isinstance(dist, (int, float)) else str(dist)
        yaw_str = f"{yaw:+.1f}deg" if isinstance(yaw, (int, float)) else str(yaw)
        cam_h_str = f"{cam_h:.2f}m" if isinstance(cam_h, (int, float)) else str(cam_h)

        status_str = "MATCH (PERFECT)" if (tp_count == len(person_gts) and fp_count == 0) else ("MISSED / FN" if fn_count > 0 else "FALSE POSITIVE")
        if is_hard_neg:
            status_str = "PASS (0 FP)" if fp_count == 0 else "FAIL (FP TRIGGERED)"

        footer_text = f"Cam: d={dist_str}, yaw={yaw_str}, h={cam_h_str} | Status: {status_str} | Conf>={conf_thres:.2f}"
        footer_color = (120, 255, 120) if "PASS" in status_str or "PERFECT" in status_str else (120, 160, 255)
        cv2.putText(img, footer_text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.43, footer_color, 1, cv2.LINE_AA)

        # ==============================================================================
        # CASE 0 PERSON GT / HARD NEGATIVE (U8.2, U8.4, U8.6, U8.8)
        # ==============================================================================
        if is_hard_neg and len(person_gts) == 0:
            if cid == "U8.2":
                if fp_count == 0:
                    banner_bg = (30, 100, 30)
                    banner_text = f"GT: 0 PERSON (1 WHEELCHAIR TRAP) | {model_name_label}: 0 PERSON DETECTIONS [TRUE NEGATIVE PASS]"
                else:
                    banner_bg = (20, 30, 180)
                    banner_text = f"GT: 0 PERSON (1 WHEELCHAIR TRAP) | {model_name_label}: {fp_count} FALSE POSITIVE HALLUCINATED!"
            else:
                if fp_count == 0:
                    banner_bg = (30, 100, 30)
                    banner_text = f"GT: 0 PERSON | {model_name_label}: 0 DETECTIONS [TRUE NEGATIVE PASS]"
                else:
                    banner_bg = (20, 30, 180)
                    banner_text = f"GT: 0 PERSON | {model_name_label}: {fp_count} FALSE POSITIVE DETECTED!"
            
            text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 2)[0]
            cx, cy = w // 2, h // 2
            bx1 = cx - text_size[0] // 2 - 10
            by1 = cy - text_size[1] // 2 - 6
            bx2 = cx + text_size[0] // 2 + 10
            by2 = cy + text_size[1] // 2 + 6
            cv2.rectangle(img, (bx1, by1), (bx2, by2), banner_bg, -1)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
            cv2.putText(img, banner_text, (bx1 + 10, by2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # ==============================================================================
        # 1. DRAW WHEELCHAIR GROUND TRUTH BOXES (THICK GOLD/AMBER SOLID LINE)
        # ==============================================================================
        for w_idx, w_gt in enumerate(wheelchair_gts):
            w_box = w_gt.get("bbox_xyxy_px") or w_gt.get("bbox_xyxy", [])
            if len(w_box) < 4:
                continue
            wx1, wy1, wx2, wy2 = [int(round(v)) for v in w_box[:4]]
            wx1 = max(0, min(w - 1, wx1))
            wy1 = max(0, min(h - 1, wy1))
            wx2 = max(0, min(w - 1, wx2))
            wy2 = max(0, min(h - 1, wy2))

            wc_color = (0, 215, 255)  # Bright Gold / Amber (BGR)
            cv2.rectangle(img, (wx1, wy1), (wx2, wy2), wc_color, 2)

            w_tag = f"GT #{w_idx+1} WHEELCHAIR [GOLD]"
            (wtw, wth), _ = cv2.getTextSize(w_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            wty1 = max(48, wy1 - wth - 6)
            wty2 = wty1 + wth + 6
            wtx2 = min(w, wx1 + wtw + 6)
            cv2.rectangle(img, (wx1, wty1), (wtx2, wty2), wc_color, -1)
            cv2.putText(img, w_tag, (wx1 + 3, wty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

        # ==============================================================================
        # 2. DRAW PERSON GROUND TRUTH BOXES (THICK GREEN SOLID LINE / MAGENTA IF MISSED)
        # ==============================================================================
        for g_idx, gt in enumerate(person_gts):
            box = gt.get("bbox_xyxy_px") or gt.get("bbox_xyxy")
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            is_hit = gt_matched[g_idx]
            gt_color = (0, 230, 0) if is_hit else (180, 0, 255)  # Green if detected, Magenta if missed

            # Box viền đậm
            cv2.rectangle(img, (x1, y1), (x2, y2), gt_color, 2)

            # Tag nhãn GT ở phía trên bên trái box
            subcat = gt.get("subcategory", "person")
            match_status = f"MATCH IoU:{matched_gt_ious[g_idx]:.2f}" if is_hit else "MISSED (FN)"
            tag_str = f"GT #{g_idx+1} {subcat} | {match_status}"
            (tw, th), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)

            ty1 = max(48, y1 - th - 6)
            ty2 = ty1 + th + 6
            tx2 = min(w, x1 + tw + 6)

            cv2.rectangle(img, (x1, ty1), (tx2, ty2), gt_color, -1)
            cv2.putText(img, tag_str, (x1 + 3, ty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

        # ==============================================================================
        # 3. DRAW MODEL PERSON PREDICTIONS (CYAN IF MATCH, ORANGE/RED IF FP)
        # ==============================================================================
        for p_idx, pred in enumerate(preds):
            b = pred["bbox"]
            px1, py1, px2, py2 = [int(round(v)) for v in b]
            px1 = max(0, min(w - 1, px1))
            py1 = max(0, min(h - 1, py1))
            px2 = max(0, min(w - 1, px2))
            py2 = max(0, min(h - 1, py2))

            is_tp = pred["matched_gt_idx"] >= 0
            if is_tp:
                pred_color = (255, 210, 0)  # Cyan/Sky Blue (BGR)
                tag_label = f"{model_name_label}: {pred['conf']:.2f} (IoU:{pred['matched_iou']:.2f}) [TP]"
            else:
                pred_color = (0, 80, 255)  # Orange-Red (BGR)
                tag_label = f"{model_name_label}: {pred['conf']:.2f} [FP]"

            # Vẽ viền nét đứt hoặc viền mỏng 2px
            cv2.rectangle(img, (px1, py1), (px2, py2), pred_color, 2)

            # Tag nhãn Model ở phía dưới bên trái box (để không đè lên tag GT ở trên)
            (ptw, pth), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            pty1 = min(h - 28, py2)
            pty2 = pty1 + pth + 6
            ptx2 = min(w, px1 + ptw + 6)

            cv2.rectangle(img, (px1, pty1), (ptx2, pty2), pred_color, -1)
            cv2.putText(img, tag_label, (px1 + 3, pty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

        # Lưu ảnh
        out_path = target_group_dir / f"{fid}.jpg"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rendered_count += 1
        group_counts[gid] = group_counts.get(gid, 0) + 1

        if rendered_count % 100 == 0 or rendered_count == len(annotations):
            print(f"[Render] Đã xuất {rendered_count}/{len(annotations)} ảnh đối chứng...")

    elapsed = time.time() - t0
    overall_prec = total_tp_all / max(1, total_tp_all + total_fp_all)
    overall_rec = total_tp_all / max(1, total_person_gt_all)

    print("==================================================================")
    print(f"✅ HOÀN TẤT VẼ ĐỐI CHỨNG DUAL-BOX {rendered_count} ẢNH TRONG {elapsed:.1f}s!")
    print(f"📊 Thống kê sơ bộ với {model_name_label}:")
    print(f"  • Tổng Person GT Boxes: {total_person_gt_all}")
    print(f"  • Tổng Wheelchair GT Boxes: {total_wheelchair_gt_all}")
    print(f"  • True Positives (Khớp đúng): {total_tp_all} ({overall_rec:.2%})")
    print(f"  • False Positives (Báo sai): {total_fp_all}")
    print(f"  • False Negatives (Bỏ sót): {total_fn_all}")
    print(f"  • Precision: {overall_prec:.2%} | Recall: {overall_rec:.2%}")
    print("📁 Phân bổ theo thư mục lát cắt:")
    for gid, cnt in sorted(group_counts.items()):
        print(f"  • {gid}: {cnt} frames ({cnt // 18} ca)")
    print("==================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Ground Truth & Model Detections Visual Inspection")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Đường dẫn file trọng số model (mặc định: yolo26n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy confidence threshold (mặc định: 0.25)")
    parser.add_argument("--iou", type=float, default=0.50, help="Ngưỡng IoU matching (mặc định: 0.50)")
    args = parser.parse_args()

    render_groundtruth_and_predictions(model_path=args.model, conf_thres=args.conf, iou_thres=args.iou)
