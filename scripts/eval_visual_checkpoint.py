"""Evaluation and Diagnostic Benchmark for Pure Visual Search RL Checkpoints.

Computes comprehensive empirical metrics across held-out evaluation seeds:
- Visual spotting rate (% shifts where patient was visually acquired)
- Mean search time before visual target acquisition (seconds)
- Mean patients helped per 240s shift
- Floor breakdown: Escort successes across Floor 1, Floor 2, and Floor 3
- Collision rate and safety metrics
- Stagnation rate comparison
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.ppo_agent import ACT_DIM, OBS_DIM, PPOHospitalAgent
from src.training.hospital_env import TaskPhase, floor_of
from src.training.hospital_env_vision import HospitalVisualShiftEnv

DEFAULT_CHECKPOINT = (
    "experiments/runs/20261030_114837_PPOHospital-VisualSearch-3floor/checkpoints/"
    "PPOHospital-VisualSearch-3floor_best.pt"
)
EVAL_SEED_BASE = 500_000


def load_agent(checkpoint_path: Path) -> PPOHospitalAgent:
    agent = PPOHospitalAgent(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def run_benchmark(
    checkpoint_path: Path,
    n_episodes: int = 50,
    stagnation_timeout_s: float = 60.0,
    stagnation_penalty: float = -10.0,
    shift_duration_s: float = 240.0,
    verbose: bool = True,
) -> dict[str, Any]:
    agent = load_agent(checkpoint_path)

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / "eval_visual_benchmark.log").open("a", encoding="utf-8")

    def log_print(msg: str = "") -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    r_team_list: list[float] = []
    tasks_list: list[int] = []
    collisions_list: list[int] = []
    stagnation_count = 0
    spotted_count = 0
    search_times_s: list[float] = []
    floor_helped = {1: 0, 2: 0, 3: 0}

    log_print("=" * 80)
    log_print(f"BENCHMARKING PURE VISUAL RL CHECKPOINT: {checkpoint_path.name}")
    log_print(f"Episodes: {n_episodes} | Shift Duration: {shift_duration_s:.1f}s | Seed Base: {EVAL_SEED_BASE}")
    log_print("=" * 80)

    for ep in range(n_episodes):
        seed = EVAL_SEED_BASE + ep * 7919
        env = HospitalVisualShiftEnv(
            shift_duration_s=shift_duration_s,
            dt=0.2,
            seed=seed,
            env_mode="solo",
            stagnation_timeout_s=stagnation_timeout_s,
            stagnation_penalty=stagnation_penalty,
            escort_only=True,
            domain_randomization=False,
        )
        obs = env.reset(seed=seed)
        done = False
        info: dict[str, Any] = {}
        ever_spotted = False
        first_spot_time_s: float | None = None
        step = 0

        while not done:
            step += 1
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True
                )
            a = action.squeeze(0).numpy()
            obs, r, done, info = env.step(a)

            if info.get("has_det", 0.0) > 0.5:
                ever_spotted = True
                if first_spot_time_s is None:
                    first_spot_time_s = step * env.dt

            if info.get("event") == "guide_completed":
                fl = floor_of(env.robot_x)
                floor_helped[fl] = floor_helped.get(fl, 0) + 1

        r_team = float(info.get("cumulative_team_reward", 0.0))
        tasks = int(info.get("patients_helped", 0))
        cols = int(info.get("collisions", 0))
        is_stag = info.get("exit_reason") == "STAGNATION_TIMEOUT"

        r_team_list.append(r_team)
        tasks_list.append(tasks)
        collisions_list.append(cols)
        if is_stag:
            stagnation_count += 1
        if ever_spotted:
            spotted_count += 1
        if first_spot_time_s is not None:
            search_times_s.append(first_spot_time_s)

        if verbose and (ep < 10 or (ep + 1) % 10 == 0 or ep == n_episodes - 1):
            spot_str = f"{first_spot_time_s:4.1f}s" if first_spot_time_s is not None else "  NONE"
            log_print(
                f"[Shift {ep + 1:02d}/{n_episodes:02d}] R_team: {r_team:+6.1f} | Helped: {tasks} | "
                f"Col: {cols:2d} | Spot: {spot_str} | Exit: {info.get('exit_reason', 'N/A')}"
            )

    mean_r = float(np.mean(r_team_list))
    std_r = float(np.std(r_team_list))
    mean_tasks = float(np.mean(tasks_list))
    std_tasks = float(np.std(tasks_list))
    mean_cols = float(np.mean(collisions_list))
    spot_rate = spotted_count / n_episodes
    stag_rate = stagnation_count / n_episodes
    mean_search_time = float(np.mean(search_times_s)) if search_times_s else 0.0

    log_print("-" * 80)
    log_print("BENCHMARK SUMMARY RESULTS:")
    log_print(f"  Team Reward (R_team)      : {mean_r:+.2f} +/- {std_r:.2f}")
    log_print(f"  Patients Helped / Shift   : {mean_tasks:.2f} +/- {std_tasks:.2f} (Total: {sum(tasks_list)})")
    log_print(f"  Visual Spotting Rate      : {spot_rate * 100:.1f}% ({spotted_count}/{n_episodes})")
    log_print(f"  Mean Search Latency       : {mean_search_time:.2f} s")
    log_print(f"  Stagnation Rate           : {stag_rate * 100:.1f}% ({stagnation_count}/{n_episodes})")
    log_print(f"  Mean Collisions / Shift   : {mean_cols:.2f}")
    log_print(f"  Escorts Completed by Floor: Floor 1: {floor_helped[1]}, Floor 2: {floor_helped[2]}, Floor 3: {floor_helped[3]}")
    log_print("=" * 80)
    try:
        log_file.close()
    except Exception:
        pass

    return {
        "mean_r_team": mean_r,
        "std_r_team": std_r,
        "mean_tasks": mean_tasks,
        "std_tasks": std_tasks,
        "total_tasks": sum(tasks_list),
        "spotting_rate": spot_rate,
        "mean_search_time_s": mean_search_time,
        "stagnation_rate": stag_rate,
        "mean_collisions": mean_cols,
        "floor_breakdown": floor_helped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate visual RL checkpoint")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to checkpoint .pt")
    parser.add_argument("--episodes", type=int, default=50, help="Number of benchmark episodes")
    parser.add_argument("--stagnation-timeout", type=float, default=60.0)
    parser.add_argument("--shift-duration", type=float, default=240.0)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path

    if not ckpt_path.exists():
        print(f"Error: checkpoint {ckpt_path} not found!")
        sys.exit(1)

    run_benchmark(
        checkpoint_path=ckpt_path,
        n_episodes=args.episodes,
        stagnation_timeout_s=args.stagnation_timeout,
        shift_duration_s=args.shift_duration,
    )


if __name__ == "__main__":
    main()
