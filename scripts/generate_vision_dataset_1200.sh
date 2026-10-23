#!/usr/bin/env bash
# scripts/generate_vision_dataset_1200.sh
# Sinh bộ dữ liệu đánh giá thị giác toàn diện 48 kịch bản (8 Trục không gian) trong Godot 4

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== [Vision Benchmark] Khởi động sinh dataset 48 kịch bản trong Godot 4 ==="

GODOT_BIN="${REPO_ROOT}/.local/bin/godot"
if [ ! -f "$GODOT_BIN" ]; then
    if [ -f "/home/bush/.gemini-pro2/.local/bin/godot" ]; then
        GODOT_BIN="/home/bush/.gemini-pro2/.local/bin/godot"
    elif command -v godot &>/dev/null; then
        GODOT_BIN="godot"
    fi
fi

echo "Sử dụng Godot binary: $GODOT_BIN"

# Chạy Godot với cờ sinh dataset 48 kịch bản (8 Trục không gian)
"$GODOT_BIN" --path sim_vision/godot res://scenes/hospital_vision_bench.tscn -- --generate-dataset-1200

echo "=== [Vision Benchmark] Hoàn tất sinh dataset tại sim_vision/data/vision_eval_1200/ ==="
