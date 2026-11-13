import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Ensure scripts directory is importable
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit.stitch_video import stitch_frames_to_video


def test_stitch_frames_to_video():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        # Create 10 dummy JPEG frames (640x480)
        import cv2
        for i in range(10):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), img)

        out_mp4 = tmp_path / "output.mp4"
        meta = {
            "error_type": "TETHER_BREAK",
            "timestamp": "20261113_113000",
            "tether_distance": 2.85,
            "robot_state": "GUIDE_ESCORT",
        }

        mp4_path, json_path = stitch_frames_to_video(
            frame_dir=frames_dir,
            output_mp4=out_mp4,
            fps=15.0,
            metadata=meta,
        )

        assert mp4_path.is_file()
        assert mp4_path.stat().st_size > 1000
        assert json_path is not None
        assert json_path.is_file()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["error_type"] == "TETHER_BREAK"
        assert data["tether_distance"] == 2.85
        assert data["frame_count"] == 10
        assert data["duration_s"] == pytest.approx(10 / 15.0, 0.05)
