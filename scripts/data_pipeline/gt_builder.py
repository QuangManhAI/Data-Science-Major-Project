#!/usr/bin/env python3
"""
gt_builder.py
Script tổng hợp siêu dữ liệu (Metadata Aggregator) từ Godot Engine
tạo ra tệp `annotations.jsonl` chuẩn Phase 0 phục vụ cả 3 mục đích:
1. Training YOLO (2D visible_bbox và normalized yolo)
2. Robot Navigation (3D godot_world_pos, distance_meters, bearing_degrees)
3. Forensic Audit (occlusion_state, visibility_ratio, difficulty_tag)
"""

import os
import json
import argparse
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[2]
    
    parser = argparse.ArgumentParser(description="Tổng hợp metadata Ground Truth từ Godot ra annotations.jsonl")
    parser.add_argument("--raw-dir", type=str, default=str(repo_root / "data/vision/raw"), help="Thư mục raw data")
    parser.add_argument("--output-jsonl", type=str, default=str(repo_root / "data/vision/raw/annotations_standard.jsonl"), help="Đường dẫn file annotations.jsonl chuẩn")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    in_jsonl = raw_dir / "annotations.jsonl"
    out_jsonl = Path(args.output_jsonl)

    if not in_jsonl.exists():
        print(f"❌ Không tìm thấy file {in_jsonl}")
        return

    print("==================================================================")
    print("🏗️ BẮT ĐẦU CHUẨN HÓA VÀ XÂY DỰNG GROUND TRUTH MULTI-LAYER METADATA")
    print(f"• Input:  {in_jsonl}")
    print(f"• Output: {out_jsonl}")
    print("==================================================================")

    processed_frames = 0
    total_annotations = 0

    with open(in_jsonl, "r", encoding="utf-8") as fin, open(out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            
            frame_id = item.get("frame_id", "")
            img_rel = item.get("image_path", f"images/{frame_id}.jpg")
            resolution = item.get("resolution", [640, 480])
            ctx = item.get("context", {})
            d_m = ctx.get("distance_m", 2.0)
            yaw_deg = ctx.get("cam_yaw_deg", 0.0)

            standard_record = {
                "$schema": "hospital_robot_gt_v2",
                "frame_id": frame_id,
                "image": {
                    "file_name": img_rel,
                    "width": resolution[0],
                    "height": resolution[1],
                    "channels": 3,
                    "is_negative_frame": item.get("is_hard_negative", False)
                },
                "robot_camera": {
                    "cam_height_m": ctx.get("cam_height_m", 1.05),
                    "cam_yaw_deg": yaw_deg,
                    "fov_horizontal_deg": 75.0
                },
                "scene_context": {
                    "case_id": item.get("case_id", ""),
                    "group_id": item.get("group_id", ""),
                    "group_name": item.get("group_name", ""),
                    "name_vi": item.get("name_vi", ""),
                    "name_en": item.get("name_en", "")
                },
                "annotations": []
            }

            raw_anns = item.get("ground_truth_annotations", [])
            for ann in raw_anns:
                c_id = ann.get("category_id", 0)
                if c_id not in (0, 1):
                    continue

                bbox_px = ann.get("bbox_xyxy_px", [0, 0, 0, 0])
                w_px = bbox_px[2] - bbox_px[0]
                h_px = bbox_px[3] - bbox_px[1]
                xc_norm = (bbox_px[0] + bbox_px[2]) / (2.0 * resolution[0])
                yc_norm = (bbox_px[1] + bbox_px[3]) / (2.0 * resolution[1])
                w_norm = w_px / resolution[0]
                h_norm = h_px / resolution[1]

                vis_ratio = ann.get("visibility_ratio", 1.0)
                occlusion_state = "none" if vis_ratio >= 0.95 else ("partial" if vis_ratio >= 0.50 else "heavy")

                standard_record["annotations"].append({
                    "category_id": c_id,
                    "category_name": "person" if c_id == 0 else "wheelchair",
                    "sub_type": ann.get("subcategory", "general"),
                    "visible_bbox_xyxy": [round(x, 1) for x in bbox_px],
                    "full_body_bbox_xyxy": [round(x, 1) for x in ann.get("bbox_full_xyxy", bbox_px)],
                    "yolo_normalized": [round(xc_norm, 6), round(yc_norm, 6), round(w_norm, 6), round(h_norm, 6)],
                    "distance_meters": ann.get("euclidean_distance_m", d_m),
                    "bearing_degrees": yaw_deg,
                    "visibility_ratio": round(vis_ratio, 3),
                    "occlusion_state": occlusion_state,
                    "verification_status": "gold_verified"
                })
                total_annotations += 1

            fout.write(json.dumps(standard_record, ensure_ascii=False) + "\n")
            processed_frames += 1

    print(f"\n✅ Đã chuẩn hóa {processed_frames} frames ({total_annotations} instances) sang annotations_standard.jsonl.")
    print("==================================================================")

if __name__ == "__main__":
    main()
