#!/usr/bin/env python3
"""
convert_to_3class.py
Chuyển đổi nhãn dataset từ 2 classes (person, wheelchair) sang 3 classes lâm sàng:
0: patient (bệnh nhân, người lớn, người già, trẻ em)
1: doctor (bác sĩ, điều dưỡng, nhân viên y tế)
2: wheelchair (xe lăn y tế)
Dựa trên metadata gốc và phân tích IoU hình học của từng frame.
"""

import json
from pathlib import Path
from collections import Counter

def bbox_iou(boxA, boxB):
    xA1, yA1 = boxA[0] - boxA[2]/2, boxA[1] - boxA[3]/2
    xA2, yA2 = boxA[0] + boxA[2]/2, boxA[1] + boxA[3]/2
    xB1, yB1 = boxB[0] - boxB[2]/2, boxB[1] - boxB[3]/2
    xB2, yB2 = boxB[0] + boxB[2]/2, boxB[1] + boxB[3]/2

    interX1, interY1 = max(xA1, xB1), max(yA1, yB1)
    interX2, interY2 = min(xA2, xB2), min(yA2, yB2)
    interArea = max(0, interX2 - interX1) * max(0, interY2 - interY1)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    return interArea / max(areaA + areaB - interArea, 1e-6)

def main():
    repo_root = Path(__file__).resolve().parents[2]
    raw_jsonl = repo_root / "data/vision/raw/annotations.jsonl"
    lbl_dir = repo_root / "data/vision/dataset/hospital_gold_1387_3class/labels"

    print("==================================================================")
    print("🏥 CHUYỂN ĐỔI DATASET SANG 3 CLASSES LÂM SÀNG:")
    print("   0: patient (bệnh nhân / trẻ em / người nhà)")
    print("   1: doctor (bác sĩ / điều dưỡng)")
    print("   2: wheelchair (xe lăn y tế)")
    print("==================================================================")

    # 1. Đọc metadata raw
    raw_data = {}
    with open(raw_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                raw_data[d["frame_id"]] = d

    # 2. Quy tắc phân loại
    pure_doctor_cases = {"U1.1", "U1.2", "U2.4", "U3.2", "U3.8", "U3.10", "U5.1"}
    case_acts = {
        "U6.2": ["doctor_male", "patient_f02"],
        "U6.4": ["doctor_male", "patient_m02"],
        "U6.5": ["elder_f09", "nurse_female"],
        "U6.7": ["doctor_male", "nurse_female", "patient_m05"],
    }
    doctor_actors = {"doctor_male", "nurse_female", "doctor_clipboard"}

    class_counts = Counter()
    modified_files = 0

    for txt_p in sorted(lbl_dir.glob("*.txt")):
        fid = txt_p.stem
        parts = fid.split("_")
        case_id = f"{parts[0]}.{parts[1]}"
        raw_item = raw_data.get(fid, {})
        raw_anns = raw_item.get("ground_truth_annotations", [])

        with open(txt_p, "r", encoding="utf-8") as fp:
            lines = [l.strip().split() for l in fp if l.strip()]

        new_lines = []
        for line in lines:
            old_cls = int(line[0])
            xc, yc, w, h = map(float, line[1:5])
            box_norm = [xc, yc, w, h]

            if old_cls == 1:
                # Wheelchair -> class 2
                new_cls = 2
            elif case_id in pure_doctor_cases:
                new_cls = 1 # doctor
            elif case_id in case_acts:
                best_iou = -1.0
                best_idx = 0
                for idx, ra in enumerate(raw_anns):
                    r_px = ra.get("bbox_xyxy_px", [0, 0, 0, 0])
                    r_xc = (r_px[0] + r_px[2]) / 1280.0
                    r_yc = (r_px[1] + r_px[3]) / 960.0
                    r_w = (r_px[2] - r_px[0]) / 640.0
                    r_h = (r_px[3] - r_px[1]) / 480.0
                    iou = bbox_iou(box_norm, [r_xc, r_yc, r_w, r_h])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                acts_list = case_acts[case_id]
                act_name = acts_list[best_idx] if best_idx < len(acts_list) else "patient"
                new_cls = 1 if act_name in doctor_actors else 0
            else:
                new_cls = 0 # patient

            class_counts[new_cls] += 1
            new_lines.append(f"{new_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        with open(txt_p, "w", encoding="utf-8") as fp:
            fp.write("\n".join(new_lines) + ("\n" if new_lines else ""))
        modified_files += 1

    print(f"✅ Đã cập nhật thành công {modified_files} file nhãn trong {lbl_dir}")
    print(f"📊 Phân bố 3 Classes mới:")
    print(f"   • 0: patient:    {class_counts[0]:4d} boxes ({class_counts[0]/sum(class_counts.values())*100:.1f}%)")
    print(f"   • 1: doctor:     {class_counts[1]:4d} boxes ({class_counts[1]/sum(class_counts.values())*100:.1f}%)")
    print(f"   • 2: wheelchair: {class_counts[2]:4d} boxes ({class_counts[2]/sum(class_counts.values())*100:.1f}%)")
    print(f"   • TỔNG CỘNG:    {sum(class_counts.values()):4d} boxes")
    print("==================================================================")

if __name__ == "__main__":
    main()
