"""Standalone Video Stitcher for Robot Audit Evidence.

Reads a directory of buffered JPEG/PNG frames captured during an audit event,
stitches them into a playable MP4 video (using OpenCV VideoWriter),
and saves companion JSON telemetry metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("stitch_video")


def stitch_frames_to_video(
    frame_dir: str | Path,
    output_mp4: str | Path,
    fps: float = 15.0,
    metadata: Optional[dict[str, Any]] = None,
) -> Tuple[Path, Optional[Path]]:
    """Stitches images in `frame_dir` into `output_mp4` and writes metadata JSON."""
    frame_dir = Path(frame_dir)
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Find and sort all image frames
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_files: list[Path] = []
    for ext in extensions:
        image_files.extend(frame_dir.glob(ext))
    image_files.sort(key=lambda p: p.name)

    if not image_files:
        raise ValueError(f"No image frames found in directory: {frame_dir}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {image_files[0]}")
    height, width, _ = first_frame.shape

    # FourCC codec: mp4v is universally supported in OpenCV on Linux/Windows/macOS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter for output: {output_mp4}")

    written_count = 0
    try:
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize if dimension differs
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height))
                writer.write(img)
                written_count += 1
    finally:
        writer.release()

    log.info("Successfully stitched %d frames to %s (%.1f FPS, %dx%d)", written_count, output_mp4, fps, width, height)

    # Write companion JSON metadata
    json_path: Optional[Path] = None
    if metadata is not None:
        meta_dict = dict(metadata)
        meta_dict["frame_count"] = written_count
        meta_dict["fps"] = fps
        meta_dict["width"] = width
        meta_dict["height"] = height
        meta_dict["duration_s"] = written_count / fps if fps > 0 else 0.0
        meta_dict["video_file"] = output_mp4.name

        json_path = output_mp4.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, ensure_ascii=False)
        log.info("Wrote audit metadata to %s", json_path)

    return output_mp4, json_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch captured frames into an audit MP4 video.")
    parser.add_argument("--frame-dir", required=True, help="Directory containing JPEG/PNG frames")
    parser.add_argument("--output", required=True, help="Path to output MP4 file")
    parser.add_argument("--fps", type=float, default=15.0, help="Frames per second (default: 15.0)")
    parser.add_argument("--metadata", default="{}", help="JSON string of metadata to attach")
    parser.add_argument("--metadata-file", default=None, help="Path to JSON file containing metadata")
    parser.add_argument("--delete-frames", action="store_true", help="Delete frame directory after stitching")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    meta: dict[str, Any] = {}
    if args.metadata_file and os.path.isfile(args.metadata_file):
        with open(args.metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
    elif args.metadata:
        try:
            meta = json.loads(args.metadata)
        except Exception as e:
            log.warning("Could not parse --metadata JSON: %s", e)

    out_mp4, out_json = stitch_frames_to_video(
        frame_dir=args.frame_dir,
        output_mp4=args.output,
        fps=args.fps,
        metadata=meta,
    )

    if args.delete_frames:
        import shutil
        shutil.rmtree(args.frame_dir, ignore_errors=True)
        log.info("Cleaned up temporary frame directory: %s", args.frame_dir)

    print(f"OUTPUT_VIDEO={out_mp4}")
    if out_json:
        print(f"OUTPUT_METADATA={out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
