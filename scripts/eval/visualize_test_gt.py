#!/usr/bin/env python3
"""
visualize_test_gt.py
Script trích xuất và trực quan hóa toàn bộ Ground Truth của tập Test đóng băng.
Vẽ bounding box, nhãn class, khoảng cách cự ly 3D và khung tên ca kiểm thử lên từng ảnh.
Xuất ảnh ra thư mục `data/vision/dataset/test_visualized_gt/` kèm trang HTML gallery để người dùng tự kiểm định bằng mắt.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[2]
    test_img_dir = repo_root / "data" / "vision" / "dataset" / "test" / "images"
    test_lbl_dir = repo_root / "data" / "vision" / "dataset" / "test" / "labels"
    out_dir = repo_root / "data" / "vision" / "dataset" / "test_visualized_gt"
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = repo_root / "data" / "vision" / "raw" / "annotations.jsonl"
    meta_by_frame = {}
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                meta_by_frame[d.get("frame_id", "")] = d

    image_files = sorted(list(test_img_dir.glob("*.jpg")))
    print("==================================================================")
    print("🎨 BẮT ĐẦU VẼ VÀ XUẤT ẢNH GROUND TRUTH TẬP TEST ĐỂ KIỂM ĐỊNH TAY")
    print(f"• Thư mục ảnh Test gốc: {test_img_dir} ({len(image_files)} ảnh)")
    print(f"• Thư mục xuất kiểm định: {out_dir}")
    print("==================================================================")

    class_names = {0: "person", 1: "wheelchair"}
    class_colors = {
        0: (0, 255, 0),    # Bright Green for person
        1: (255, 255, 0),  # Cyan for wheelchair
    }

    gallery_items = []

    for img_path in image_files:
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        meta = meta_by_frame.get(stem, {})
        case_name_vi = meta.get("name_vi", stem)
        group_name = meta.get("group_name", "Clinical Test")

        lbl_path = test_lbl_dir / f"{stem}.txt"
        boxes = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        boxes.append({"cls": c_id, "bbox": (x1, y1, x2, y2), "norm": (xc, yc, bw, bh)})

        # Vẽ các bounding box Ground Truth
        for idx, b in enumerate(boxes):
            c_id = b["cls"]
            x1, y1, x2, y2 = b["bbox"]
            color = class_colors.get(c_id, (0, 255, 0))
            c_name = class_names.get(c_id, f"cls_{c_id}")

            # Lấy cự ly 3D từ metadata nếu có
            dist_str = ""
            gt_anns = meta.get("ground_truth_annotations", [])
            if idx < len(gt_anns):
                d_m = gt_anns[idx].get("euclidean_distance_m")
                subcat = gt_anns[idx].get("subcategory", "")
                if d_m is not None:
                    dist_str = f" | {d_m:.1f}m"
                if subcat:
                    dist_str += f" ({subcat})"

            # Vẽ khung chữ nhật
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Vẽ label tag
            label_text = f"GT: {c_name}{dist_str}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            tag_y1 = max(0, y1 - th - 6)
            tag_y2 = y1
            tag_x2 = min(w, x1 + tw + 8)
            cv2.rectangle(img, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(img, label_text, (x1 + 4, y1 - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # Vẽ tọa độ góc dưới
            coord_str = f"[{x1},{y1},{x2},{y2}]"
            cv2.putText(img, coord_str, (x1 + 4, min(h - 6, y2 - 4)), font, 0.38, color, 1, cv2.LINE_AA)

        # Vẽ Header Banner trên cùng
        banner_h = 32
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        header_text = f"[{stem}] {case_name_vi} | GT Boxes: {len(boxes)}"
        cv2.putText(img, header_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Lưu ảnh xuất ra
        out_file = out_dir / f"{stem}_gt.jpg"
        cv2.imwrite(str(out_file), img)

        gallery_items.append({
            "filename": f"{stem}_gt.jpg",
            "frame_id": stem,
            "case_name": case_name_vi,
            "n_boxes": len(boxes),
            "classes": [class_names.get(b["cls"], str(b["cls"])) for b in boxes]
        })

    # Tạo file HTML Gallery để người dùng mở bằng trình duyệt kiểm tra trực quan
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Visual Ground Truth Inspection - Test Set (90 Frames)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
        h1 {{ text-align: center; color: #4caf50; margin-bottom: 8px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 24px; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 16px; max-width: 1600px; margin: 0 auto; }}
        .card {{ background: #1e1e1e; border: 1px solid #333; border-radius: 8px; overflow: hidden; transition: transform 0.15s, border-color 0.15s; }}
        .card:hover {{ transform: scale(1.02); border-color: #4caf50; }}
        .card img {{ width: 100%; height: auto; display: block; border-bottom: 1px solid #2a2a2a; }}
        .info {{ padding: 10px 12px; font-size: 13px; }}
        .frame-id {{ font-weight: bold; color: #4caf50; font-family: monospace; font-size: 14px; }}
        .case-title {{ color: #ccc; margin: 4px 0; }}
        .badges {{ margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; }}
        .badge {{ padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .badge-person {{ background: rgba(0, 255, 0, 0.15); color: #00ff00; border: 1px solid #00ff00; }}
        .badge-wheelchair {{ background: rgba(0, 255, 255, 0.15); color: #00ffff; border: 1px solid #00ffff; }}
        .badge-count {{ background: #333; color: #aaa; }}
    </style>
</head>
<body>
    <h1>🔬 KIỂM ĐỊNH GROUND TRUTH TẬP TEST (90 FRAMES ĐÓNG BĂNG)</h1>
    <div class="subtitle">Thư mục xuất ảnh: <code>data/vision/dataset/test_visualized_gt/</code> | Toạ độ trích xuất trực tiếp từ 3D Physics Mesh của Godot 4</div>
    <div class="grid">
"""
    for item in gallery_items:
        badges_html = f'<span class="badge badge-count">{item["n_boxes"]} boxes</span>'
        for cls in item["classes"]:
            cls_class = "badge-person" if cls == "person" else "badge-wheelchair"
            badges_html += f'<span class="badge {cls_class}">{cls}</span>'

        html_content += f"""
        <div class="card">
            <a href="{item['filename']}" target="_blank">
                <img src="{item['filename']}" alt="{item['frame_id']}" loading="lazy"/>
            </a>
            <div class="info">
                <div class="frame-id">{item['frame_id']}</div>
                <div class="case-title">{item['case_name']}</div>
                <div class="badges">{badges_html}</div>
            </div>
        </div>
"""
    html_content += """
    </div>
</body>
</html>
"""
    html_path = out_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ Đã xuất toàn bộ {len(gallery_items)} ảnh kiểm định vào: {out_dir}")
    print(f"✅ Đã tạo trang web xem nhanh tại: {html_path}")
    print("==================================================================")

if __name__ == "__main__":
    main()
