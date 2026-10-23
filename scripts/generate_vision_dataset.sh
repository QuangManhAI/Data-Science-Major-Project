#!/usr/bin/env bash
# scripts/generate_vision_dataset.sh
# Sinh bộ dữ liệu đánh giá thị giác 600 frames trong Godot 4 (Offscreen Render)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== [Vision Benchmark] Khởi động sinh dataset 600 ảnh trong Godot 4 ==="

GODOT_BIN="${REPO_ROOT}/.local/bin/godot"
if [ ! -f "$GODOT_BIN" ]; then
    if [ -f "/home/bush/.gemini-pro2/.local/bin/godot" ]; then
        GODOT_BIN="/home/bush/.gemini-pro2/.local/bin/godot"
    elif command -v godot &>/dev/null; then
        GODOT_BIN="godot"
    fi
fi

echo "Sử dụng Godot binary: $GODOT_BIN"

# Chạy Godot với Scene Benchmark chuyên biệt
"$GODOT_BIN" --path sim_vision/godot res://scenes/hospital_vision_bench.tscn -- --generate-dataset

echo "=== [Vision Benchmark] Hoàn tất sinh dataset tại data/vision_eval_600/ ==="
