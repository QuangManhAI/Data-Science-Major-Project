"""Interactive Visual Mock-Training Runner for Hospital Pure-Vision RL.

Runs the EXACT pure-vision PPO training loop (environment stepping, observation pipeline,
actor-critic forward passes, GAE advantage calculation, and PPO loss backprop) with:
1. Non-Headless Interactive GUI:
   - 2D Floorplan (rooms, doors, robot chassis, 75-deg FOV cone, 5 sectors, patient, doctor)
   - Simulated Camera View (640x360 with YOLO bounding boxes and sector clearance HUD)
   - Real-time PPO Training Telemetry HUD
2. Mock-Train Safety:
   - Checkpoint saving is DISABLED by default. No checkpoint files on disk will be overwritten
     or altered, while weights are updated live in memory.
3. Interactive Controls:
   - [SPACE] : Pause / Resume
   - [+ / -] : Speed up / Slow down playback
   - [F]     : Switch Floor view (Floor 1 -> 2 -> 3 -> Auto)
   - [Q/ESC] : Clean exit

Usage:
  # Quick smoke visual test (2 short episodes):
  python scripts/visual_mock_train.py --smoke

  # Real-time visual mock-training at 30 FPS starting from latest fine-tuned checkpoint:
  python scripts/visual_mock_train.py --fps 30 --episodes 20

  # High-speed visualization (60 FPS):
  python scripts/visual_mock_train.py --fps 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_pure_vision_rl import run_train_pure_vision


def find_latest_checkpoint() -> Path | None:
    """Find the best or latest pure-vision checkpoint in experiments/runs/."""
    runs_dir = REPO_ROOT / "experiments" / "runs"
    if not runs_dir.exists():
        return None

    # Search for latest run containing PPOHospital-PureVision_best.pt
    candidates = list(runs_dir.glob("**/checkpoints/PPOHospital-PureVision_best.pt"))
    if not candidates:
        candidates = list(runs_dir.glob("**/checkpoints/*best.pt"))
    if not candidates:
        return None

    # Sort by directory modification time descending
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-Headless Visual Mock-Training for Pure-Vision Hospital Robot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes to simulate")
    parser.add_argument("--fps", type=int, default=30, help="Target visual display FPS")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file (defaults to latest)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Hardware device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shift-duration", type=float, default=180.0, help="Shift duration in seconds")
    parser.add_argument("--smoke", action="store_true", help="Run 2 quick micro-episodes for instant testing")
    parser.add_argument("--save-checkpoints", action="store_true", help="Force enabling checkpoint saving (disables mock-train)")

    args = parser.parse_args()

    # Resolve resume checkpoint
    resume_path = args.resume
    if resume_path is None:
        latest = find_latest_checkpoint()
        if latest is not None:
            resume_path = str(latest)
            print(f"[INFO] Auto-selected latest trained checkpoint: {latest}")
        else:
            print("[INFO] No existing checkpoint found; initializing from scratch.")

    is_mock = not args.save_checkpoints

    print("=" * 76)
    print("  HOSPITAL ROBOT PURE-VISION : VISUAL MOCK TRAINING")
    print("=" * 76)
    print(f"  Visual GUI      : ENABLED (Non-Headless, Target {args.fps} FPS)")
    print(f"  Mock Train      : {'ENABLED (Checkpoints will NOT be saved to disk)' if is_mock else 'DISABLED (Saving checkpoints)'}")
    print(f"  Initial Weights : {resume_path or 'Scratch (Random Initialization)'}")
    print(f"  Episodes        : {args.episodes if not args.smoke else '2 (Smoke Mode)'}")
    print(f"  Device          : {args.device}")
    print("-" * 76)
    print("  Interactive Controls:")
    print("    [SPACE]   : Pause / Play simulation")
    print("    [+] / [-] : Speed up / Slow down rendering delay")
    print("    [F]       : Toggle Floor 1 / Floor 2 / Floor 3 / Auto")
    print("    [Q / ESC] : Quit visual training cleanly")
    print("=" * 76)

    run_train_pure_vision(
        episodes=args.episodes,
        start_episode=1,
        resume_from=resume_path,
        device_name=args.device,
        seed=args.seed,
        smoke=args.smoke,
        shift_duration_s=args.shift_duration,
        escort_curriculum=True,
        visual=True,
        mock_train=is_mock,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
