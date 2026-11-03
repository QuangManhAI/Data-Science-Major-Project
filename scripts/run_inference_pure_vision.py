"""Interactive Pure-Vision (Zero LiDAR) RL Policy Inference & Visual Demonstration.

Runs real-time policy evaluation of the trained PPO agent on the continuous multi-patient hospital environment.

Features:
1. Interactive Non-Headless Visual GUI:
   - Left Panel (740x740): Top-down hospital floorplan with rooms, doors, robot chassis, 75° camera FOV cone,
     5 radial optical sectors, unserved patients, companion escort waypoint & elastic tether, and patrolling doctor.
   - Right Top (640x360): Simulated robot front camera view with YOLO bounding boxes and sector clearance HUD.
   - Right Bottom (640x380): Live telemetry HUD (speeds, active phase, patients helped X/6, tether distance, goal).
2. Pure Inference:
   - Strictly torch.no_grad() forward passes with deterministic actor mean actions (or stochastic sampling).
   - Zero training / backprop overhead.
3. Interactive Controls:
   - [SPACE] : Pause / Resume simulation
   - [+ / -] : Speed up / Slow down playback FPS
   - [F]     : Switch Floor view (Floor 1 -> Floor 2 -> Floor 3 -> Auto-follow)
   - [R]     : Reset episode immediately
   - [Q/ESC] : Quit cleanly

Usage:
  # Run visual inference using the latest best checkpoint at 30 FPS:
  python scripts/run_inference_pure_vision.py

  # Run at 60 FPS:
  python scripts/run_inference_pure_vision.py --fps 60

  # Run with explicit checkpoint:
  python scripts/run_inference_pure_vision.py --checkpoint experiments/runs/20261103_100811_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.ppo_agent import ACT_DIM, OBS_DIM_PURE_VISION, PPOHospitalAgent
from src.training.hospital_env_pure_vision import HospitalPureVisionEnv
from src.ui.visual_env_renderer import VisualEnvRenderer


def find_latest_checkpoint() -> Path | None:
    """Find the best or latest pure-vision checkpoint in experiments/runs/."""
    runs_dir = REPO_ROOT / "experiments" / "runs"
    if not runs_dir.exists():
        return None

    candidates = list(runs_dir.glob("**/checkpoints/PPOHospital-PureVision_best.pt"))
    if not candidates:
        candidates = list(runs_dir.glob("**/checkpoints/*best.pt"))
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def run_inference(
    checkpoint_path: str | Path | None = None,
    episodes: int = 10,
    max_steps: int = 10000,
    fps: int = 30,
    device_name: str = "auto",
    deterministic: bool = True,
    seed: int = 42,
) -> None:
    # 1. Resolve Device
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # 2. Resolve Checkpoint
    if checkpoint_path is None:
        ckpt_file = find_latest_checkpoint()
        if ckpt_file is None:
            raise FileNotFoundError("No trained checkpoint found in experiments/runs/!")
    else:
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    print("=" * 78)
    print("  HOSPITAL HELPER ROBOT — PURE-VISION POLICY INFERENCE")
    print("=" * 78)
    print(f"  Checkpoint      : {ckpt_file.name}")
    print(f"  Full Path       : {ckpt_file}")
    print(f"  Target Device   : {device.type.upper()} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"  Episodes        : {episodes} shifts (up to {max_steps} steps / {max_steps*0.2:.0f}s each)")
    print(f"  Action Mode     : {'Deterministic (Actor Mean)' if deterministic else 'Stochastic (Sampled)'}")
    print(f"  Visual Display  : Interactive Non-Headless GUI ({fps} FPS)")
    print("-" * 78)
    print("  Controls:")
    print("    [SPACE]   : Pause / Resume simulation")
    print("    [+] / [-] : Speed up / Slow down playback FPS")
    print("    [F]       : Toggle Floor 1 / Floor 2 / Floor 3 / Auto-follow")
    print("    [R]       : Restart / Reset current episode")
    print("    [Q / ESC] : Quit inference")
    print("=" * 78)

    # 3. Load Agent
    agent = PPOHospitalAgent(obs_dim=OBS_DIM_PURE_VISION, act_dim=ACT_DIM).to(device)
    ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)
    state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
    agent.load_state_dict(state_dict)
    agent.eval()

    # 4. Initialize Visual Renderer
    renderer = VisualEnvRenderer(
        window_name=f"Hospital Pure-Vision Robot Inference [{ckpt_file.stem}]",
        target_fps=fps,
    )

    total_helped = 0
    total_episodes_run = 0

    try:
        for ep in range(1, episodes + 1):
            total_episodes_run += 1
            env = HospitalPureVisionEnv(
                shift_duration_s=max_steps * 0.2,
                dt=0.2,
                seed=seed + ep * 100,
                env_mode="full",
                stagnation_timeout_s=60.0,
                stagnation_penalty=-10.0,
                escort_only=True,
                domain_randomization=False,
            )

            obs = env.reset()
            done = False
            ep_reward = 0.0
            step_idx = 0

            print(f"\n--- Starting Shift {ep}/{episodes} (Robot at Entrance X={env.robot_x:.1f}, Z={env.robot_z:.1f}) ---")

            while not done:
                step_idx += 1
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    features = agent.actor_backbone(obs_t)
                    action_mean = torch.tanh(agent.actor_mean(features))
                    if deterministic:
                        action = action_mean.squeeze(0).cpu().numpy()
                    else:
                        action_logstd = agent.actor_logstd.expand_as(action_mean)
                        dist = torch.distributions.Normal(action_mean, action_logstd.exp())
                        action = dist.sample().squeeze(0).cpu().numpy()

                action = np.clip(action, -1.0, 1.0)
                next_obs, reward, done, info = env.step(action)
                ep_reward += reward

                # Log notable events
                if info.get("patient_engaged"):
                    p_idx = env.active_patient_idx
                    dest = env.patients[p_idx]["destination"] if p_idx is not None else "Unknown"
                    print(f"  [Step {step_idx:4d}] Engaged Patient #{p_idx + 1} -> Heading to {dest}")
                elif info.get("elevator_transit"):
                    cur_fl = math.floor(env.robot_x / 50.0) + 1
                    print(f"  [Step {step_idx:4d}] Rode Elevator to Floor {cur_fl} (Target: {env.goal_key})")
                elif info.get("guide_completed"):
                    print(f"  [Step {step_idx:4d}] Delivered Patient at {env.goal_key}! Helped: {env.patients_helped}/{len(env.patients)}")
                elif info.get("return_transit"):
                    print(f"  [Step {step_idx:4d}] Returned via Elevator to Floor 1 Lobby to resume patrol")

                # Render frame
                keep_going = renderer.render_step(
                    env=env,
                    action=action,
                    reward=reward,
                    step_idx=step_idx,
                    episode_idx=ep,
                    total_episodes=episodes,
                    info=info,
                    is_mock_train=False,
                )

                if not keep_going:
                    print("\n[INFO] User exited visual renderer.")
                    return

                obs = next_obs

            tasks = env.patients_helped
            total_helped += tasks
            exit_reason = info.get("exit_reason", "UNKNOWN")
            print(f"Shift {ep} Finished | Helped: {tasks}/6 | Total Reward: {ep_reward:+.1f} | Exit: {exit_reason} ({step_idx} steps)")

    finally:
        cv2.destroyAllWindows()
        print("\n" + "=" * 78)
        print("  INFERENCE SUMMARY")
        print("=" * 78)
        print(f"  Shifts Simulated    : {total_episodes_run}")
        print(f"  Total Patients Helped: {total_helped}")
        print(f"  Average Helped/Shift : {total_helped / max(1, total_episodes_run):.2f}")
        print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Interactive Pure-Vision RL Inference Demonstration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file (defaults to latest best)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of shifts to simulate")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per shift (10000 steps = 2000s)")
    parser.add_argument("--fps", type=int, default=30, help="Visual display target FPS")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Hardware device")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically instead of deterministic mean")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        device_name=args.device,
        deterministic=not args.stochastic,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
