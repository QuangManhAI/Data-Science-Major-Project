#!/usr/bin/env python3
"""
render_visualized_groundtruth.py
Vẽ bounding box Ground Truth (xanh lá) cùng siêu dữ liệu trực quan lên toàn bộ ảnh
để người dùng duyệt mắt chay (manual visual inspection).
"""

import os
import json
import time
from pathlib import Path
import cv2
import numpy as np
import shutil

def render_groundtruth_visualizations():
    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = repo_root / "sim_vision/data/vision_eval_unified"
    img_dir = dataset_dir / "images"
    jsonl_file = dataset_dir / "annotations.jsonl"
    out_vis_dir = repo_root / "sim_vision/data/vision_eval_groundtruth_visualized"

    # Reset output directory
    if out_vis_dir.exists():
        shutil.rmtree(out_vis_dir)
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annotations = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                annotations[data["frame_id"]] = data

    print("==================================================================")
    print(f"🎨 BẮT ĐẦU VẼ TRỰC QUAN HÓA GROUND TRUTH CHO {len(annotations)} FRAMES...")
    print(f"📁 Thư mục đích: {out_vis_dir}")
    print("==================================================================")
    t0 = time.time()

    rendered_count = 0
    group_counts = {}

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

        # Header bar
        cv2.rectangle(img, (0, 0), (w, 32), (25, 25, 25), -1)
        title_text = f"[{cid}] {name_vi} | Frame: {fid}"
        cv2.putText(img, title_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Info bar at bottom
        cv2.rectangle(img, (0, h - 26), (w, h), (25, 25, 25), -1)
        ctx = meta.get("context", {})
        dist = ctx.get("distance_m", "?")
        yaw = ctx.get("cam_yaw_deg", "?")
        cam_h = ctx.get("cam_height_m", "?")
        dist_str = f"{dist:.1f}m" if isinstance(dist, (int, float)) else str(dist)
        yaw_str = f"{yaw:+.1f}deg" if isinstance(yaw, (int, float)) else str(yaw)
        cam_h_str = f"{cam_h:.2f}m" if isinstance(cam_h, (int, float)) else str(cam_h)

        footer_text = f"Cam: d={dist_str}, yaw={yaw_str}, h={cam_h_str} | Verified GT: {len(gts)} person(s)"
        cv2.putText(img, footer_text, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        if is_hard_neg or len(gts) == 0:
            # Banner giữa ảnh cho ca 0 GT
            banner_bg = (160, 60, 20) if is_hard_neg else (50, 50, 50)
            banner_text = "GT: 0 PERSON (BAY DAO CU RONG / ZERO-GT HARD NEGATIVE)" if is_hard_neg else "GT: 0 PERSON (OCCLUDED > 80% / IGNORED)"
            text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
            cx, cy = w // 2, h // 2
            bx1 = cx - text_size[0] // 2 - 12
            by1 = cy - text_size[1] // 2 - 8
            bx2 = cx + text_size[0] // 2 + 12
            by2 = cy + text_size[1] // 2 + 8
            cv2.rectangle(img, (bx1, by1), (bx2, by2), banner_bg, -1)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
            cv2.putText(img, banner_text, (bx1 + 12, by2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Vẽ các box Ground Truth
            for idx, gt in enumerate(gts):
                box = gt.get("bbox_xyxy_px") or gt.get("bbox_xyxy")
                if not box or len(box) < 4:
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                # Box viền xanh lá đậm (BGR: 0, 220, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)

                # Tag nhãn
                vis_ratio = gt.get("visibility_ratio", 1.0)
                subcat = gt.get("subcategory", "person")
                tag_str = f"#{idx+1} {subcat} (Vis:{int(vis_ratio*100)}%)"
                (tw, th), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)

                ty1 = max(34, y1 - th - 6)
                ty2 = ty1 + th + 6
                tx2 = min(w, x1 + tw + 8)

                cv2.rectangle(img, (x1, ty1), (tx2, ty2), (0, 180, 0), -1)
                cv2.putText(img, tag_str, (x1 + 4, ty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # Lưu ảnh
        out_path = target_group_dir / f"{fid}.jpg"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rendered_count += 1
        group_counts[gid] = group_counts.get(gid, 0) + 1

        if rendered_count % 100 == 0 or rendered_count == len(annotations):
            print(f"[Render] Đã xuất {rendered_count}/{len(annotations)} ảnh trực quan...")

    elapsed = time.time() - t0
    print("==================================================================")
    print(f"✅ HOÀN TẤT VẼ TRỰC QUAN HÓA {rendered_count} ẢNH TRONG {elapsed:.1f}s!")
    for gid, cnt in sorted(group_counts.items()):
        print(f"  • {gid}: {cnt} frames ({cnt // 18} ca)")
    print("==================================================================")

if __name__ == "__main__":
    render_groundtruth_visualizations()
