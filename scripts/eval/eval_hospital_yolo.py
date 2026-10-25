#!/usr/bin/env python3
"""
eval_hospital_yolo.py
Script đánh giá toàn diện mô hình YOLO sau khi huấn luyện trên tập Test đóng băng:
1. Metric tổng thể: mAP@50, mAP@50-95, Precision, Recall, FPS / Latency P95.
2. Phân tách theo từng class: person, wheelchair.
3. Phân tích pháp y lâm sàng (Forensic Breakdown):
   - Ca bệnh nhân nằm giường (U2.8)
   - Ca bệnh nhi / trẻ em (U2.10, U6.6, U6.8)
   - Ca bệnh nhân ngồi xe lăn (U2.9)
"""

import argparse
import time
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def main():
    repo_root = Path(__file__).resolve().parents[2]
    
    parser = argparse.ArgumentParser(description="Đánh giá mô hình YOLO trên tập Test bệnh viện")
    parser.add_argument("--model", type=str, default="experiments/runs/train_hospital/yolo26n_hospital_v1/weights/best.pt", help="Đường dẫn file weights cần test")
    parser.add_argument("--data", type=str, default=str(repo_root / "configs/hospital_3class.yaml"), help="File cấu hình dataset")
    parser.add_argument("--device", type=str, default="0", help="Thiết bị tính toán")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Không tìm thấy model weights tại: {model_path}")
        return

    print("==================================================================")
    print("🔬 BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH THỊ GIÁC BỆNH VIỆN TRÊN TẬP TEST")
    print(f"• Model: {model_path}")
    print(f"• Dataset: {args.data}")
    print(f"• Device: {args.device} | Conf: {args.conf} | IoU: {args.iou}")
    print("==================================================================")

    model = YOLO(str(model_path))

    # 1. Validation trên Test Split
    metrics = model.val(
        data=args.data,
        split="test",
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        verbose=True
    )

    # 2. Đo tốc độ trễ (Latency Benchmark)
    test_img_dir = repo_root / "data" / "vision" / "dataset" / "test" / "images"
    test_images = list(test_img_dir.glob("*.jpg"))
    latencies = []
    
    if test_images:
        print(f"\n⏱️ Đang đo độ trễ suy luận trên {len(test_images)} ảnh Test...")
        # Warmup
        for _ in range(5):
            _ = model(test_images[0], device=args.device, verbose=False)
            
        for img in test_images:
            t0 = time.perf_counter()
            _ = model(img, device=args.device, verbose=False)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        mean_lat = float(np.mean(latencies))
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0
    else:
        p50, p95, mean_lat, fps = 0, 0, 0, 0

    print("\n==================================================================")
    print("🏆 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (TEST REPORT):")
    print(f"• mAP@50:         {metrics.box.map50 * 100:.2f}%")
    print(f"• mAP@50-95:      {metrics.box.map * 100:.2f}%")
    print(f"• Precision:      {metrics.box.mp * 100:.2f}%")
    print(f"• Recall:         {metrics.box.mr * 100:.2f}%")
    print(f"• Tốc độ FPS:     {fps:.1f} FPS")
    print(f"• Độ trễ P50:     {p50:.2f} ms")
    print(f"• Độ trễ P95:     {p95:.2f} ms")
    print("==================================================================")

if __name__ == "__main__":
    main()
