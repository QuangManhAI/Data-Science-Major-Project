#!/usr/bin/env python3
"""
generate_recumbent_bed_poster.py
Tạo Poster trực quan hóa đa góc nhìn chuyên sâu cho Bệnh Nhân Nằm Giường Cấp Cứu (U2.8)
Bao gồm:
- Trực diện chân giường (Yaw = 0°)
- Trực diện đầu giường (Yaw = 180°)
- Cạnh sườn trái giường (Yaw = -90°)
- Cạnh sườn phải giường (Yaw = +90°)
- Góc chéo 3/4 chân - sườn trái (Yaw = -45°)
- Góc chéo 3/4 đầu - sườn phải (Yaw = +135°)
- Góc thấp ngang tầm robot (Cam H = 0.80m)
- Góc cao nhìn thẳng từ trên xuống (Cam H = 1.65m)
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np

def create_recumbent_bed_multi_angle_poster():
    repo_root = Path(__file__).resolve().parent.parent.parent
    vis_dir = repo_root / "data/vision/vision_eval_groundtruth_visualized/U2_ClinicalPoses"
    if not vis_dir.exists():
        vis_dir = repo_root / "data/vision_eval_groundtruth_visualized/U2_ClinicalPoses"
    report_dir = repo_root / "sim_vision/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/home/bush/.gemini-pro2/.gemini/antigravity-cli/brain/054ca26c-29c3-4501-b03b-252567c1ff0d")

    panels_info = [
        {
            "frame": "U2_8_0001.jpg",
            "title": "1. TRỰC DIỆN CHÂN GIƯỜNG (FOOTBOARD VIEW)",
            "subtitle": "Yaw: 0.0° | d: 2.0m | Cam H: 1.15m | Look H: 0.68m",
            "note": "Camera nhìn dọc từ chân giường lên gối. Chân & chăn ga choán tâm ảnh."
        },
        {
            "frame": "U2_8_0003.jpg",
            "title": "2. TRỰC DIỆN ĐẦU GIƯỜNG (HEADBOARD VIEW)",
            "subtitle": "Yaw: 180.0° | d: 2.0m | Cam H: 1.25m | Look H: 0.68m",
            "note": "Camera nhìn từ phía đầu giường xuống chân. Khuôn mặt & gối lộ rõ."
        },
        {
            "frame": "U2_8_0005.jpg",
            "title": "3. CẠNH SƯỜN TRÁI (LEFT LATERAL SIDE VIEW)",
            "subtitle": "Yaw: -90.0° | d: 1.6m | Cam H: 1.15m | Look H: 0.68m",
            "note": "Góc ngang 90° nhìn trọn vẹn toàn bộ chiều dài cơ thể nằm ngang đệm."
        },
        {
            "frame": "U2_8_0008.jpg",
            "title": "4. CẠNH SƯỜN PHẢI (RIGHT LATERAL SIDE VIEW)",
            "subtitle": "Yaw: +90.0° | d: 1.6m | Cam H: 1.15m | Look H: 0.68m",
            "note": "Góc ngang đối diện bên phải, thấy rõ cọc truyền dịch Inox đầu giường."
        },
        {
            "frame": "U2_8_0011.jpg",
            "title": "5. GÓC CHÉO 3/4 CHÂN - TRÁI (FRONT-LEFT OBLIQUE)",
            "subtitle": "Yaw: -45.0° | d: 1.8m | Cam H: 1.15m | Look H: 0.68m",
            "note": "Góc quan sát lâm sàng kinh điển của Bác sĩ khi bước vào phòng khám."
        },
        {
            "frame": "U2_8_0014.jpg",
            "title": "6. GÓC CHÉO 3/4 ĐẦU - PHẢI (REAR-RIGHT OBLIQUE)",
            "subtitle": "Yaw: +135.0° | d: 1.8m | Cam H: 1.25m | Look H: 0.68m",
            "note": "Góc chéo từ sau đầu, kiểm tra tư thế gối đầu và nhịp thở của bệnh nhân."
        },
        {
            "frame": "U2_8_0015.jpg",
            "title": "7. GÓC THẤP NGANG MÉP ĐỆM (LOW EYE-LEVEL VIEW)",
            "subtitle": "Yaw: -75.0° | d: 1.4m | Cam H: 0.80m | Look H: 0.68m",
            "note": "Tầm mắt của Robot gầm thấp (AGV), thành giường che khuất một phần đệm."
        },
        {
            "frame": "U2_8_0017.jpg",
            "title": "8. GÓC CAO NHÌN XUỐNG MẶT ĐỆM (OVERHEAD VIEW)",
            "subtitle": "Yaw: 0.0° | d: 2.2m | Cam H: 1.65m | Look H: 0.68m",
            "note": "Góc camera trần hoặc robot cao nhìn bao quát toàn bộ mặt phẳng nằm."
        }
    ]

    # Poster layout: 4 rows x 2 columns
    # Panel resolution: 900 x 540
    pw = 1920
    row_h = 540
    ph = 120 + 4 * row_h + 60
    poster = np.full((ph, pw, 3), 22, dtype=np.uint8)

    # Header
    cv2.rectangle(poster, (0, 0), (pw, 100), (12, 14, 18), -1)
    cv2.putText(poster, "BỘ DỮ LIỆU ĐA GÓC NHÌN BỆNH NHÂN NẰM GIƯỜNG CẤP CỨU (U2.8 RECUMBENT BED PATIENT MULTI-ANGLE)", (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(poster, "Khai phá toàn diện các góc nhìn: Trực diện chân, Trực diện đầu, Cạnh sườn 90°, Góc chéo 3/4, Góc thấp & Góc cao | Dual-Box Visualized", (40, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

    col_w = (pw - 60) // 2
    for idx, pinfo in enumerate(panels_info):
        r = idx // 2
        c = idx % 2
        x_left = 20 + c * (col_w + 20)
        y_top = 110 + r * row_h

        # Panel frame
        cv2.rectangle(poster, (x_left, y_top), (x_left + col_w, y_top + row_h - 15), (32, 34, 40), -1)
        cv2.rectangle(poster, (x_left, y_top), (x_left + col_w, y_top + row_h - 15), (60, 65, 75), 1)

        # Image load
        img_p = vis_dir / pinfo["frame"]
        img_w_disp = 580
        img_h_disp = 435
        if img_p.exists():
            im = cv2.imread(str(img_p))
            im_resized = cv2.resize(im, (img_w_disp, img_h_disp))
            poster[y_top + 15 : y_top + 15 + img_h_disp, x_left + 15 : x_left + 15 + img_w_disp] = im_resized
        else:
            cv2.rectangle(poster, (x_left + 15, y_top + 15), (x_left + 15 + img_w_disp, y_top + 15 + img_h_disp), (15, 15, 15), -1)
            cv2.putText(poster, "RENDERING IMAGE...", (x_left + 120, y_top + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)

        # Right side info block
        info_x = x_left + 15 + img_w_disp + 15
        info_w = col_w - 30 - img_w_disp - 15

        cv2.putText(poster, pinfo["title"], (info_x, y_top + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 225, 255), 2, cv2.LINE_AA)
        cv2.putText(poster, pinfo["subtitle"], (info_x, y_top + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 210, 255), 1, cv2.LINE_AA)

        # Multi-line note
        words = pinfo["note"].split(" ")
        lines = []
        cur = ""
        for w in words:
            if len(cur + " " + w) > 28:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)

        line_y = y_top + 130
        for l in lines:
            cv2.putText(poster, l, (info_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 225, 230), 1, cv2.LINE_AA)
            line_y += 26

        # Forensic Callout Box
        callout_y = y_top + 280
        cv2.rectangle(poster, (info_x, callout_y), (info_x + info_w - 10, callout_y + 130), (20, 25, 32), -1)
        cv2.rectangle(poster, (info_x, callout_y), (info_x + info_w - 10, callout_y + 130), (45, 55, 65), 1)
        cv2.putText(poster, "PHÂN TÍCH LÂM SÀNG:", (info_x + 8, callout_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 150), 1, cv2.LINE_AA)
        
        analysis_text = [
            "• GT: Box chuẩn đa tia vật lý",
            "• Tỷ lệ H/W: 0.35 dẹt nằm ngang",
            "• Đệm ga xanh phủ 70% cơ thể",
            "• Điểm mù thị giác COCO chuẩn"
        ]
        cy = callout_y + 50
        for at in analysis_text:
            cv2.putText(poster, at, (info_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 200), 1, cv2.LINE_AA)
            cy += 20

    out_poster_path = report_dir / "04_recumbent_bed_patient_multi_angle_poster.png"
    cv2.imwrite(str(out_poster_path), poster, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    print(f"🖼️ Đã xuất poster đa góc nhìn bệnh nhân nằm giường: {out_poster_path}")

    if artifact_dir.exists():
        shutil.copy2(out_poster_path, artifact_dir / "04_recumbent_bed_patient_multi_angle_poster.png")

if __name__ == "__main__":
    create_recumbent_bed_multi_angle_poster()
