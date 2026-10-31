#!/usr/bin/env python3
"""
scripts/export_robot_model.py
=============================
Utility to export trained YOLO26n models to edge-ready formats for robotics:
- ONNX (opset 17, simplified with onnxslim, static BCHW for TensorRT/ROS2)
- Engine (TensorRT FP16 if on Jetson Orin Nano with CUDA/TensorRT installed)

Usage:
  python3 scripts/export_robot_model.py \
      --weights experiments/runs/train_hospital/yolo26n_targeted_17class_v4/weights/best.pt \
      --format onnx \
      --imgsz 640
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Export YOLO26n model for robot deployment")
    parser.add_argument(
        "--weights",
        type=str,
        default="experiments/runs/train_hospital/yolo26n_targeted_17class_v4/weights/best.pt",
        help="Path to trained PyTorch weights (.pt)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "engine", "torchscript"],
        help="Target export format"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch/resolution support")
    args = parser.parse_args()

    weights_file = Path(args.weights)
    if not weights_file.exists():
        print(f"Error: Weights file '{weights_file}' not found.")
        sys.exit(1)

    print(f"Loading YOLO26n model: {weights_file}")
    model = YOLO(str(weights_file))

    print(f"Starting export to format='{args.format}', imgsz={args.imgsz}, half={args.half}, dynamic={args.dynamic}...")
    export_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=True,
        opset=17
    )
    print(f"Successfully exported model to: {export_path}")

if __name__ == "__main__":
    main()
