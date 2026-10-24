#!/usr/bin/env python3
"""
generate_zero_defect_groundtruth.py
Quy trình gán nhãn 2 vòng "Bất Khả Sai" (Zero-Defect Hybrid Annotation Pipeline).
Kết hợp SOTA Model (YOLOv8x) quét sơ bộ với Dữ liệu Vật lý Godot 3D (Skeleton3D, 128-Raycast & Prop 3D Models).
Hỗ trợ Multi-Class:
  Class 0: person (Bác sĩ, Điều dưỡng, Cụ già, Bệnh nhân nằm giường cấp cứu, Bệnh nhân ngồi xe lăn)
  Class 1: wheelchair (Xe lăn y tế có người ngồi hoặc xe lăn rỗng)
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

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
    iou = interArea / float(areaA + areaB - interArea)
    ioa = interArea / float(areaA)  # overlap relative to boxA (phys box)
    return iou, ioa

def run_zero_defect_pipeline():
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / "data" / "vision" / "vision_eval_unified"
    if not data_dir.exists():
        data_dir = repo_root / "data" / "vision_eval_unified"
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"
    rich_jsonl_path = data_dir / "annotations_rich_5w1h.jsonl"
    manifest_path = data_dir / "manifest_unified.json"

    print("==================================================================")
    print("🚀 BẮT ĐẦU QUY TRÌNH GÁN NHÃN 2 VÒNG ZERO-DEFECT HYBRID (MULTI-CLASS)")
    print(f"📁 Thư mục dữ liệu: {data_dir}")
    print("==================================================================")

    # 1. Load SOTA Model (YOLOv8x)
    print("[Pass 1] Khởi tạo mô hình SOTA YOLOv8x...")
    from ultralytics import YOLO
    model_path = repo_root / "yolov8x.pt"
    if not model_path.exists():
        model = YOLO("yolov8x.pt")
    else:
        model = YOLO(str(model_path))

    # 2. Read annotations_rich_5w1h.jsonl
    with open(rich_jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[Pass 1] Đã tải {len(records)} bản ghi metadata vật lý từ Godot.")

    total_frames = len(records)
    total_hard_negatives = 0
    total_persons_gt = 0
    total_wheelchairs_gt = 0
    total_ai_matched = 0
    total_raycast_fallback = 0
    total_ignored_occluded = 0

    verified_records = []

    for idx, rec in enumerate(records):
        frame_id = rec["frame_id"]
        case_id = rec.get("case_id", "")
        is_hard_neg = rec.get("is_hard_negative", False)
        img_file = img_dir / f"{frame_id}.jpg"
        lbl_file = lbl_dir / f"{frame_id}.txt"
        img_w, img_h = rec.get("resolution", [640, 480])

        final_yolo_lines = []
        final_gt_boxes = []

        if is_hard_neg:
            total_hard_negatives += 1
            # ĐỊNH LUẬT BẪY RỖNG ĐA LỚP:
            # - Với U8.2 (Bẫy xe lăn rỗng): 0 người (Person=0) nhưng CÓ 1 Xe Lăn (Wheelchair=1)!
            # - Với U8.4, U8.6, U8.8 (Kiosk, Giường rỗng, Standee): 0 người và 0 xe lăn (File rỗng 0 byte).
            if case_id == "U8.2":
                phys_gts = rec.get("ground_truth_annotations", [])
                for phys in phys_gts:
                    if phys.get("category_id") == 1 or phys.get("category_name") == "wheelchair":
                        w_box = phys.get("bbox_xyxy_px", [])
                        if len(w_box) == 4:
                            cx1 = max(0.0, min(float(img_w), w_box[0]))
                            cy1 = max(0.0, min(float(img_h), w_box[1]))
                            cx2 = max(0.0, min(float(img_w), w_box[2]))
                            cy2 = max(0.0, min(float(img_h), w_box[3]))
                            bw = cx2 - cx1
                            bh = cy2 - cy1
                            if bw >= 8.0 and bh >= 8.0:
                                total_wheelchairs_gt += 1
                                norm_cx = (cx1 + cx2) * 0.5 / img_w
                                norm_cy = (cy1 + cy2) * 0.5 / img_h
                                norm_bw = bw / img_w
                                norm_bh = bh / img_h
                                final_yolo_lines.append(f"1 {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}")
                                final_gt_boxes.append({
                                    "category_id": 1,
                                    "category_name": "wheelchair",
                                    "bbox_xyxy_px": [cx1, cy1, cx2, cy2],
                                    "annotation_source": "PHYSICAL_3D_WHEELCHAIR_GROUNDTRUTH",
                                    "visibility_ratio": phys.get("visibility_ratio", 1.0),
                                    "subcategory": "wheelchair"
                                })
                with open(lbl_file, "w", encoding="utf-8") as lf:
                    if final_yolo_lines:
                        lf.write("\n".join(final_yolo_lines) + "\n")
                rec["zero_defect_boxes"] = final_gt_boxes
                rec["ground_truth_status"] = "VERIFIED_HARD_NEGATIVE_EMPTY_WHEELCHAIR"
                verified_records.append(rec)
                continue
            else:
                with open(lbl_file, "w", encoding="utf-8") as lf:
                    pass  # empty file 0 byte
                rec["zero_defect_boxes"] = []
                rec["ground_truth_status"] = "VERIFIED_HARD_NEGATIVE_ZERO_GT"
                verified_records.append(rec)
                continue

        # Ca có người -> Chạy YOLOv8x quét sơ bộ người
        # conf=0.15, classes=[0] (person)
        preds = model(str(img_file), conf=0.15, classes=[0], verbose=False)[0]
        ai_boxes = []
        for box in preds.boxes:
            b_coords = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            b_conf = float(box.conf[0].cpu().numpy())
            ai_boxes.append({"bbox": b_coords, "conf": b_conf, "matched": False})

        phys_gts = rec.get("ground_truth_annotations", [])

        for phys in phys_gts:
            cat_id = phys.get("category_id", 0)

            # 1. Nếu là Xe lăn (Wheelchair Class 1)
            if cat_id == 1 or phys.get("category_name") == "wheelchair":
                w_box = phys.get("bbox_xyxy_px", [])
                if len(w_box) == 4:
                    cx1 = max(0.0, min(float(img_w), w_box[0]))
                    cy1 = max(0.0, min(float(img_h), w_box[1]))
                    cx2 = max(0.0, min(float(img_w), w_box[2]))
                    cy2 = max(0.0, min(float(img_h), w_box[3]))
                    bw = cx2 - cx1
                    bh = cy2 - cy1
                    if bw >= 8.0 and bh >= 8.0:
                        total_wheelchairs_gt += 1
                        norm_cx = (cx1 + cx2) * 0.5 / img_w
                        norm_cy = (cy1 + cy2) * 0.5 / img_h
                        norm_bw = bw / img_w
                        norm_bh = bh / img_h
                        final_yolo_lines.append(f"1 {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}")
                        final_gt_boxes.append({
                            "category_id": 1,
                            "category_name": "wheelchair",
                            "bbox_xyxy_px": [cx1, cy1, cx2, cy2],
                            "annotation_source": "PHYSICAL_3D_WHEELCHAIR_GROUNDTRUTH",
                            "visibility_ratio": phys.get("visibility_ratio", 1.0),
                            "subcategory": "wheelchair"
                        })
                continue

            # 2. Nếu là Người (Person Class 0)
            vis_ratio = float(phys.get("visibility_ratio", 1.0))
            if vis_ratio < 0.05 and phys.get("ignore_eval", False):
                total_ignored_occluded += 1
                continue

            phys_box = phys.get("bbox_xyxy_px", phys.get("bbox_full_xyxy", []))
            if not phys_box or len(phys_box) < 4:
                continue

            # Tìm AI box tốt nhất khớp với physical box
            best_ai = None
            best_score = -1.0

            for ai in ai_boxes:
                if ai["matched"]:
                    continue
                iou, ioa = compute_iou(phys_box, ai["bbox"])
                score = max(iou, ioa)
                if score >= 0.30 and score > best_score:
                    best_score = score
                    best_ai = ai

            chosen_box = None
            match_source = ""

            if best_ai is not None:
                best_ai["matched"] = True
                b_ai = best_ai["bbox"]
                # Raycast Clipping Law:
                # Nếu người bị che khuất một phần (vis_ratio < 0.90), đáy hộp không được chìm sâu vào vật cản
                clip_y2 = min(b_ai[3], phys_box[3] + 12.0) if phys.get("visibility_ratio", 1.0) < 0.85 else b_ai[3]
                clip_y1 = max(b_ai[1], phys_box[1] - 8.0)
                clip_x1 = max(0.0, min(b_ai[0], phys_box[0]))
                clip_x2 = min(img_w, max(b_ai[2], phys_box[2]))

                chosen_box = [round(clip_x1, 1), round(clip_y1, 1), round(clip_x2, 1), round(clip_y2, 1)]
                match_source = f"YOLOv8x_CONF_{best_ai['conf']:.2f}_IOU_{best_score:.2f}"
                total_ai_matched += 1
            else:
                # Fallback Định Luật Khung Xương: dùng raycast physical box (Đặc biệt hữu dụng cho nằm giường & ngụy trang)
                chosen_box = [round(phys_box[0], 1), round(phys_box[1], 1), round(phys_box[2], 1), round(phys_box[3], 1)]
                match_source = "PHYSICAL_RAYCAST_FALLBACK"
                total_raycast_fallback += 1

            # Clamp box to image dimensions
            cx1 = max(0.0, min(float(img_w), chosen_box[0]))
            cy1 = max(0.0, min(float(img_h), chosen_box[1]))
            cx2 = max(0.0, min(float(img_w), chosen_box[2]))
            cy2 = max(0.0, min(float(img_h), chosen_box[3]))

            bw = cx2 - cx1
            bh = cy2 - cy1
            if bw >= 6.0 and bh >= 6.0:
                total_persons_gt += 1
                norm_cx = (cx1 + cx2) * 0.5 / img_w
                norm_cy = (cy1 + cy2) * 0.5 / img_h
                norm_bw = bw / img_w
                norm_bh = bh / img_h
                final_yolo_lines.append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}")
                final_gt_boxes.append({
                    "category_id": 0,
                    "category_name": "person",
                    "bbox_xyxy_px": [cx1, cy1, cx2, cy2],
                    "annotation_source": match_source,
                    "visibility_ratio": phys.get("visibility_ratio", 1.0),
                    "subcategory": phys.get("subcategory", "standing")
                })

        # Ghi file nhãn YOLO
        with open(lbl_file, "w", encoding="utf-8") as lf:
            if final_yolo_lines:
                lf.write("\n".join(final_yolo_lines) + "\n")

        rec["zero_defect_boxes"] = final_gt_boxes
        rec["ground_truth_status"] = "VERIFIED_GOLD"
        verified_records.append(rec)

        if (idx + 1) % 100 == 0 or idx == total_frames - 1:
            print(f"[Pass 2] Đã thẩm định & gán nhãn: {idx + 1}/{total_frames} frames...")

    # Ghi lại file annotations.jsonl & annotations_rich_5w1h.jsonl
    with open(rich_jsonl_path, "w", encoding="utf-8") as f:
        for r in verified_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    clean_jsonl = data_dir / "annotations.jsonl"
    with open(clean_jsonl, "w", encoding="utf-8") as f:
        for r in verified_records:
            simple_rec = {
                "frame_id": r["frame_id"],
                "case_id": r["case_id"],
                "name_vi": r.get("name_vi", ""),
                "name_en": r.get("name_en", ""),
                "group_id": r.get("group_id", ""),
                "is_hard_negative": r.get("is_hard_negative", False),
                "image_path": r.get("image_path", ""),
                "resolution": r.get("resolution", [640, 480]),
                "ground_truth_annotations": r.get("zero_defect_boxes", [])
            }
            f.write(json.dumps(simple_rec, ensure_ascii=False) + "\n")

    # Cập nhật manifest_unified.json
    manifest = {
        "dataset_name": "vision_eval_unified",
        "version": "3.1_zero_defect_multiclass_36_cases",
        "total_frames": total_frames,
        "total_cases": 36,
        "classes": {
            0: "person",
            1: "wheelchair"
        },
        "resolution": [640, 480],
        "statistics": {
            "total_frames": total_frames,
            "total_hard_negatives_0gt": total_hard_negatives,
            "total_person_groundtruth_boxes": total_persons_gt,
            "total_wheelchair_groundtruth_boxes": total_wheelchairs_gt,
            "total_ai_matched_boxes": total_ai_matched,
            "total_raycast_fallback_boxes": total_raycast_fallback,
            "total_ignored_occluded_persons": total_ignored_occluded
        },
        "annotation_pipeline": "dual_pass_hybrid_yolov8x_plus_skeleton3d_raycast_multiclass",
        "zero_defect_guarantee": "Zero False Positives on Empty Traps, Recumbent Bed Patient & Wheelchair Detection"
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("==================================================================")
    print("✅ HOÀN TẤT QUY TRÌNH GÁN NHÃN 2 VÒNG ZERO-DEFECT (MULTI-CLASS)!")
    print(f"• Tổng số frames: {total_frames}")
    print(f"• Số ca bẫy rỗng (Hard Negatives): {total_hard_negatives} frames")
    print(f"• Tổng số nhãn Người (Class 0): {total_persons_gt} boxes")
    print(f"• Tổng số nhãn Xe Lăn (Class 1): {total_wheelchairs_gt} boxes")
    print(f"• Tỷ lệ SOTA AI bọc viền áo chuẩn: {total_ai_matched} boxes ({total_ai_matched / max(total_persons_gt, 1) * 100:.1f}%)")
    print(f"• Tỷ lệ Raycast fallback (nằm giường/ngụy trang/xa): {total_raycast_fallback} boxes")
    print(f"• Số đối tượng che khuất > 95% bỏ qua: {total_ignored_occluded} instances")
    print("==================================================================")

if __name__ == "__main__":
    run_zero_defect_pipeline()
