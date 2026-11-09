"""Headless benchmark and comparative evaluation between:
1. Manh V2 (21-D pure-vision depth policy): experiments/runs/manh_tune_11_9/PPOHospital-PureVisionV2_best.pt
2. Our Best V1 (29-D pure-vision policy): experiments/runs/20261106_075341_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.ppo_agent import ACT_DIM, OBS_DIM_PURE_VISION, OBS_DIM_PURE_VISION_V2, PPOHospitalAgent
from src.training.hospital_env_pure_vision import HospitalPureVisionEnv
from src.training.hospital_env_pure_vision_v2 import HospitalPureVisionV2Env


def load_checkpoint(ckpt_path: Path, expected_dim: int | None = None) -> PPOHospitalAgent:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    first_weight = state_dict.get("actor_backbone.0.weight")
    in_dim = first_weight.shape[1] if first_weight is not None else (expected_dim or 21)
    agent = PPOHospitalAgent(obs_dim=in_dim, act_dim=ACT_DIM)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


@torch.no_grad()
def evaluate_v2(
    agent: PPOHospitalAgent,
    seeds: list[int],
    shift_s: float = 700.0,
    domain_randomization: bool = False,
) -> list[dict]:
    results = []
    for seed in seeds:
        env = HospitalPureVisionV2Env(
            seed=seed,
            shift_duration_s=shift_s,
            dt=0.2,
            domain_randomization=domain_randomization,
        )
        obs, done = env.reset(), False
        step_count = 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = torch.tanh(agent.actor_mean(agent.actor_backbone(obs_t))).squeeze(0).numpy()
            obs, _, done, info = env.step(action)
            step_count += 1
        
        results.append({
            "seed": seed,
            "reward": float(info.get("cumulative_team_reward", 0.0)),
            "helped": int(info.get("patients_helped", 0)),
            "collisions": int(info.get("collisions", 0)),
            "escort_speed": float(info.get("escort_speed", 0.0)),
            "steps": step_count,
            "duration_s": round(step_count * 0.2, 1),
            "exit_reason": info.get("exit_reason", "UNKNOWN"),
            "all_served": bool(info.get("exit_reason") == "ALL_PATIENTS_SERVED"),
        })
    return results


@torch.no_grad()
def evaluate_v1(
    agent: PPOHospitalAgent,
    seeds: list[int],
    shift_s: float = 700.0,
    domain_randomization: bool = False,
) -> list[dict]:
    results = []
    for seed in seeds:
        env = HospitalPureVisionEnv(
            curriculum_stage=1,
            shift_duration_s=shift_s,
            dt=0.2,
            seed=seed,
            env_mode="full",
            stagnation_timeout_s=90.0,
            stagnation_penalty=-10.0,
            escort_only=True,
            domain_randomization=domain_randomization,
        )
        obs, done = env.reset(), False
        step_count = 0
        escort_speeds = []
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = torch.tanh(agent.actor_mean(agent.actor_backbone(obs_t))).squeeze(0).numpy()
            obs, _, done, info = env.step(action)
            step_count += 1
            if getattr(env, "task_phase", None) == getattr(env, "GUIDE_ESCORT", 2):
                escort_speeds.append(float(getattr(env, "robot_v", 0.0)))

        mean_speed = float(np.mean(escort_speeds)) if escort_speeds else 0.0
        helped = int(info.get("patients_helped", 0))
        results.append({
            "seed": seed,
            "reward": float(info.get("cumulative_team_reward", 0.0)),
            "helped": helped,
            "collisions": int(info.get("collisions", 0)),
            "escort_speed": mean_speed,
            "steps": step_count,
            "duration_s": round(step_count * 0.2, 1),
            "exit_reason": info.get("exit_reason", "UNKNOWN"),
            "all_served": bool(helped >= 6 or info.get("exit_reason") == "ALL_PATIENTS_SERVED"),
        })
    return results


def summarize(name: str, res: list[dict]) -> dict:
    helped = [r["helped"] for r in res]
    cols = [r["collisions"] for r in res]
    speeds = [r["escort_speed"] for r in res]
    rewards = [r["reward"] for r in res]
    durations = [r["duration_s"] for r in res]
    all_served = sum(1 for r in res if r["all_served"])
    stagnated = sum(1 for r in res if r["exit_reason"] == "STAGNATION_TIMEOUT")

    return {
        "model": name,
        "episodes": len(res),
        "mean_helped": float(np.mean(helped)),
        "std_helped": float(np.std(helped)),
        "all_served_rate": round(all_served / len(res) * 100.0, 1),
        "all_served_count": f"{all_served}/{len(res)}",
        "mean_collisions": float(np.mean(cols)),
        "zero_collision_rate": round(sum(1 for c in cols if c == 0) / len(res) * 100.0, 1),
        "mean_escort_speed": float(np.mean(speeds)),
        "mean_reward": float(np.mean(rewards)),
        "mean_duration_s": float(np.mean(durations)),
        "stagnation_rate": round(stagnated / len(res) * 100.0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Headless comparative evaluation between Manh V2 and Our Best V1")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes per model")
    parser.add_argument("--base-seed", type=int, default=123_000_000, help="Base seed for evaluation")
    parser.add_argument("--shift", type=float, default=700.0, help="Shift duration in seconds")
    parser.add_argument("--dr", action="store_true", help="Enable domain randomization")
    args = parser.parse_args()

    v2_path = Path("experiments/runs/manh_tune_11_9/PPOHospital-PureVisionV2_best.pt")
    v1_path = Path("experiments/runs/20261106_075341_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt")

    print("=" * 80)
    print(" HEADLESS COMPARATIVE EVALUATION OF PURE-VISION CHECKPOINTS")
    print("=" * 80)
    print(f" Model A (Manh V2):     {v2_path} (obs_dim=21)")
    print(f" Model B (Our Best V1): {v1_path} (obs_dim=29)")
    print(f" Test Configuration:    {args.episodes} episodes | Base Seed: {args.base_seed} | Shift: {args.shift}s")
    print(f" Domain Randomization:  {'ENABLED' if args.dr else 'DISABLED (Nominal Physics)'}")
    print("=" * 80)

    agent_v2 = load_checkpoint(v2_path, expected_dim=OBS_DIM_PURE_VISION_V2)
    agent_v1 = load_checkpoint(v1_path, expected_dim=OBS_DIM_PURE_VISION)

    seeds = [args.base_seed + i for i in range(args.episodes)]

    print("\n[1/2] Running headless evaluation for Model A (Manh V2)...")
    t0 = time.time()
    res_v2 = evaluate_v2(agent_v2, seeds, shift_s=args.shift, domain_randomization=args.dr)
    t_v2 = time.time() - t0
    sum_v2 = summarize("Manh V2 (21-D Depth)", res_v2)
    print(f"  Completed in {t_v2:.1f}s | Mean Helped: {sum_v2['mean_helped']:.2f}/6 | Mean Col: {sum_v2['mean_collisions']:.2f}")

    print("\n[2/2] Running headless evaluation for Model B (Our Best V1)...")
    t0 = time.time()
    res_v1 = evaluate_v1(agent_v1, seeds, shift_s=args.shift, domain_randomization=args.dr)
    t_v1 = time.time() - t0
    sum_v1 = summarize("Our Best V1 (29-D Feature)", res_v1)
    print(f"  Completed in {t_v1:.1f}s | Mean Helped: {sum_v1['mean_helped']:.2f}/6 | Mean Col: {sum_v1['mean_collisions']:.2f}")

    print("\n" + "=" * 80)
    print(" 📊 HEADLESS BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    fmt = "{:<26} | {:<18} | {:<18}"
    print(fmt.format("Metric", "Manh V2 (21-D)", "Our Best V1 (29-D)"))
    print("-" * 80)
    print(fmt.format("Patients Helped / Shift", f"{sum_v2['mean_helped']:.2f} ± {sum_v2['std_helped']:.2f} / 6", f"{sum_v1['mean_helped']:.2f} ± {sum_v1['std_helped']:.2f} / 6"))
    print(fmt.format("All 6 Patients Served", f"{sum_v2['all_served_count']} ({sum_v2['all_served_rate']}%)", f"{sum_v1['all_served_count']} ({sum_v1['all_served_rate']}%)"))
    print(fmt.format("Mean Collisions / Shift", f"{sum_v2['mean_collisions']:.2f}", f"{sum_v1['mean_collisions']:.2f}"))
    print(fmt.format("Zero-Collision Shifts", f"{sum_v2['zero_collision_rate']}%", f"{sum_v1['zero_collision_rate']}%"))
    print(fmt.format("Escort Cruising Speed", f"{sum_v2['mean_escort_speed']:.3f} m/s", f"{sum_v1['mean_escort_speed']:.3f} m/s"))
    print(fmt.format("Mean Shift Duration", f"{sum_v2['mean_duration_s']:.1f} s", f"{sum_v1['mean_duration_s']:.1f} s"))
    print(fmt.format("Stagnation Abort Rate", f"{sum_v2['stagnation_rate']}%", f"{sum_v1['stagnation_rate']}%"))
    print(fmt.format("Mean Team Reward", f"{sum_v2['mean_reward']:+.1f}", f"{sum_v1['mean_reward']:+.1f}"))
    print("=" * 80)

    out_file = Path("experiments") / "benchmark_v1_vs_v2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary_v2": sum_v2, "summary_v1": sum_v1, "episodes_v2": res_v2, "episodes_v1": res_v1}, f, indent=2)
    print(f"\n[INFO] Detailed raw episode metrics saved to: {out_file}")


if __name__ == "__main__":
    main()
