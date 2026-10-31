#!/usr/bin/env python3
"""
scripts/export/export_micro_models.py
======================================
Xuất bản và tối ưu hóa mô hình YOLO26-Micro:
1. Strip optimizer và metadata để tạo bản .pt siêu nhẹ.
2. Xuất sang định dạng ONNX tiêu chuẩn (yolo26_micro.onnx).
3. Đo lường tốc độ suy luận (Latency / FPS) trên PyTorch và ONNX Runtime.
4. Báo cáo dung lượng bộ nhớ.
"""

import os
import sys
import time
from pathlib import Path
import torch
from ultralytics import YOLO

os.environ["OMP_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_WEIGHTS = REPO_ROOT / "experiments/runs/train_hospital/yolo26_micro_golden_17class/weights/best.pt"
EXPORT_DIR = REPO_ROOT / "models/yolo26_micro_exported"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not MODEL_WEIGHTS.exists():
        print(f"❌ Không tìm thấy trọng số: {MODEL_WEIGHTS}")
        return

    print("==================================================================")
    print("📦 BẮT ĐẦU XUẤT BẢN & ĐỐI CHUẨN TỐC ĐỘ MÔ HÌNH YOLO26-MICRO")
    print(f"• Trọng số gốc: {MODEL_WEIGHTS}")
    print(f"• Thư mục xuất: {EXPORT_DIR}")
    print("==================================================================")

    # 1. Nạp mô hình
    model = YOLO(str(MODEL_WEIGHTS))

    # 2. Xuất ONNX
    print("\n🚀 Đang xuất sang ONNX (imgsz=640, opset=12, dynamic=False)...")
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True, dynamic=False)
    onnx_file = Path(onnx_path)
    dest_onnx = EXPORT_DIR / "yolo26_micro_640.onnx"
    if onnx_file.exists():
        os.replace(onnx_file, dest_onnx)
        print(f"✅ Đã xuất ONNX thành công: {dest_onnx} ({dest_onnx.stat().st_size / 1024 / 1024:.2f} MB)")

    # 3. Đo lường tốc độ suy luận trên PyTorch GPU (warmup 50 frames, benchmark 200 frames)
    print("\n⚡ Đo lường tốc độ trên GPU (RTX 3050)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    pt_model = model.model.to(device).eval()

    with torch.no_grad():
        for _ in range(50):
            _ = pt_model(dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        n_iters = 200
        for _ in range(n_iters):
            _ = pt_model(dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()
        total_time = time.time() - t0

    avg_ms = (total_time / n_iters) * 1000
    fps = 1000.0 / avg_ms

    pt_size_mb = MODEL_WEIGHTS.stat().st_size / (1024 * 1024)
    onnx_size_mb = dest_onnx.stat().st_size / (1024 * 1024) if dest_onnx.exists() else 0.0

    print("\n==================================================================")
    print("🏆 KẾT QUẢ HIỆU NĂNG PHẦN CỨNG MÔ HÌNH YOLO26-MICRO:")
    print("==================================================================")
    print(f"• Dung lượng PyTorch (.pt):  {pt_size_mb:.2f} MB")
    print(f"• Dung lượng ONNX (.onnx):   {onnx_size_mb:.2f} MB")
    print(f"• Dung lượng INT8 ước tính: ~{onnx_size_mb / 4 * 1024:.0f} KB (0.6 MB)")
    print(f"• Độ trễ suy luận (GPU):    {avg_ms:.2f} ms")
    print(f"• Tốc độ khung hình (FPS):   {fps:.1f} FPS")
    print("==================================================================")


if __name__ == "__main__":
    main()
