#!/usr/bin/env python3
"""
train_hospital_vision.py
Script huấn luyện / tinh chỉnh (fine-tuning) các mô hình YOLO (yolo26n, yolov8n, yolo11n)
trên bộ dữ liệu thị giác bệnh viện chuẩn xác (Hospital Vision Gold Dataset).
"""

import argparse
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    default_device = "0" if torch.cuda.is_available() else "cpu"
    repo_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="Huấn luyện mô hình YOLO trên bộ dữ liệu thị giác bệnh viện")
    parser.add_argument("--model", type=str, default="models/yolo26n/yolo26n.pt", help="Mô hình khởi tạo (mặc định yolo26n)")
    parser.add_argument("--data", type=str, default=str(repo_root / "configs/hospital_3class.yaml"), help="File cấu hình dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện")
    parser.add_argument("--batch", type=int, default=16, help="Kích thước batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Độ phân giải đầu vào (mặc định 640)")
    parser.add_argument("--device", type=str, default=default_device, help=f"Thiết bị tính toán (mặc định '{default_device}')")
    parser.add_argument("--project", type=str, default="experiments/runs/train_hospital", help="Thư mục lưu weights đầu ra")
    parser.add_argument("--name", type=str, default="yolo26n_hospital_v1", help="Tên phiên bản huấn luyện")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        data_yaml = repo_root / "configs/hospital_3class.yaml"

    if not data_yaml.exists():
        print(f"❌ Lỗi: Không tìm thấy file cấu hình dữ liệu tại {data_yaml}")
        sys.exit(1)

    print("==================================================================")
    print("🚀 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN THỊ GIÁC ROBOT BỆNH VIỆN")
    print(f"• Mô hình xuất phát: {args.model}")
    print(f"• File cấu hình dữ liệu: {data_yaml}")
    print(f"• Epochs: {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print(f"• Thiết bị tính toán: {args.device}")
    print("==================================================================")

    # Khởi tạo mô hình
    model = YOLO(args.model)

    # Chạy huấn luyện
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
        save=True,
        verbose=True
    )

    print("==================================================================")
    print(f"✅ Huấn luyện hoàn tất! Weights tốt nhất lưu tại: {args.project}/{args.name}/weights/best.pt")
    print("==================================================================")

if __name__ == "__main__":
    main()
