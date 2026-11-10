"""Dedicated PPO Training Script for Pure Visual-Search RL Agent (2000 Episodes).

Trains the robot to actively explore the hospital environment, visually acquire
patients via simulated pinhole camera / YOLOv8 bounding box features (zero-oracle
during search), engage dialogue, and escort patients across Floors 1, 2, and 3.

Warm-started from the 3-floor escort checkpoint:
    experiments/runs/20261027_183421_PPOHospital-3floor-escort-2k/checkpoints/PPOHospital-3floor-escort_best.pt

Usage:
    python -m src.training.train_visual_rl --episodes 2000 --device cuda
    python -m src.training.train_visual_rl --smoke --device cuda
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.ppo_agent import ACT_DIM, OBS_DIM, PPOHospitalAgent
from src.training.hospital_env_vision import HospitalVisualShiftEnv
from src.utils.checkpoint_utils import capture_rng_state, get_git_commit, seed_everything


DEFAULT_WARMSTART_PATH = (
    "experiments/runs/20261027_183421_PPOHospital-3floor-escort-2k/checkpoints/PPOHospital-3floor-escort_best.pt"
)

EVAL_SEED_BASE = 950_000_000
EVAL_COLLISION_WEIGHT = 1.0


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable MM:SS or HH:MM:SS."""
    if seconds < 0 or not np.isfinite(seconds):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m:02d}m{s:02d}s"


def resolve_device(device_name: str) -> torch.device:
    """Resolve --device into a torch.device."""
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but no CUDA device is available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but Apple MPS is not available.")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"


def evaluate_visual_policy(
    agent: PPOHospitalAgent,
    n_episodes: int,
    stagnation_timeout_s: float,
    stagnation_penalty: float,
    shift_duration_s: float,
    env_mode: str = "solo",
    escort_only: bool = True,
) -> dict[str, float]:
    """Deterministic rollouts on fixed held-out seeds for visual evaluation."""
    r_team, tasks, cols, stag = [], [], [], 0
    spotted_count = 0
    for i in range(n_episodes):
        env = HospitalVisualShiftEnv(
            shift_duration_s=shift_duration_s,
            dt=0.2,
            seed=EVAL_SEED_BASE + i * 7919,
            env_mode=env_mode,
            stagnation_timeout_s=stagnation_timeout_s,
            stagnation_penalty=stagnation_penalty,
            escort_only=escort_only,
            domain_randomization=False,
        )
        obs = env.reset()
        done = False
        info: dict[str, Any] = {}
        ever_spotted = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True
                )
            obs, _, done, info = env.step(action.squeeze(0).numpy())
            if info.get("has_det", 0.0) > 0.5:
                ever_spotted = True
        r_team.append(float(info.get("cumulative_team_reward", 0.0)))
        tasks.append(int(info.get("patients_helped", 0)))
        cols.append(int(info.get("collisions", 0)))
        stag += int(info.get("exit_reason") == "STAGNATION_TIMEOUT")
        if ever_spotted:
            spotted_count += 1
    mean_r, mean_col = float(np.mean(r_team)), float(np.mean(cols))
    return {
        "score": mean_r - EVAL_COLLISION_WEIGHT * mean_col,
        "r_team": mean_r,
        "tasks": float(np.mean(tasks)),
        "collisions": mean_col,
        "stagnation_rate": stag / max(1, n_episodes),
        "spotting_rate": spotted_count / max(1, n_episodes),
    }


def run_train_visual_rl(
    episodes: int = 2000,
    warmstart_path: str | Path | None = DEFAULT_WARMSTART_PATH,
    resume_from: str | Path | None = None,
    stagnation_timeout_s: float = 60.0,
    stagnation_penalty: float = -10.0,
    shift_duration_s: float = 240.0,
    update_freq_episodes: int = 5,
    ppo_epochs: int = 4,
    batch_size: int = 256,
    lr_init: float = 1e-4,
    lr_min: float = 3e-5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    seed: int = 42,
    device_name: str = "auto",
    experiments_root: str | Path = "experiments",
    eval_every: int = 200,
    eval_episodes: int = 20,
    patience: int = 0,
    bc_enabled: bool = True,
    env_mode: str = "solo",
    escort_only: bool = True,
    smoke: bool = False,
) -> Path:
    seed_everything(seed)
    device = resolve_device(device_name)

    if smoke:
        episodes = 2
        shift_duration_s = 4.0  # 20 steps
        eval_every = 2
        eval_episodes = 1
        update_freq_episodes = 1
        ppo_epochs = 1
        batch_size = 32

    agent = PPOHospitalAgent(obs_dim=OBS_DIM, act_dim=ACT_DIM).to(device)
    rollout_agent = PPOHospitalAgent(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    rollout_agent.eval()

    total_updates = max(1, episodes // update_freq_episodes)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr_init, eps=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=lr_min
    )

    rolling_20 = collections.deque(maxlen=20)
    best_roll20 = -float("inf")
    peak_r_team = -float("inf")
    start_ep = 1
    warmstart_loaded = False
    best_eval = -float("inf")
    evals_without_improve = 0
    ckpt_prefix = "PPOHospital-VisualSearch-3floor"

    if resume_from:
        r_path = Path(resume_from)
        if r_path.is_file():
            ckpt_file = r_path
            run_dir = r_path.parent.parent
        else:
            run_dir = r_path
            candidates = [
                f"{ckpt_prefix}_last.pt",
                f"{ckpt_prefix}_best.pt",
            ]
            ckpt_file = next(
                (run_dir / "checkpoints" / c for c in candidates if (run_dir / "checkpoints" / c).exists()),
                run_dir / "checkpoints" / candidates[0],
            )

        ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)
        agent.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass

        start_ep = int(ckpt.get("episode", 0)) + 1
        best_roll20 = float(ckpt.get("best_roll20", -float("inf")))
        best_eval = float(ckpt.get("best_eval_score", -float("inf")))
        peak_r_team = float(ckpt.get("peak_r_team", -float("inf")))
        ckpt_dir = run_dir / "checkpoints"
        metrics_file = run_dir / "metrics.jsonl"

        remaining_updates = max(1, (episodes - start_ep + 1) // update_freq_episodes)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_updates, eta_min=lr_min
        )

        if metrics_file.exists():
            with metrics_file.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if "r_team" in data:
                            rolling_20.append(float(data["r_team"]))
                    except Exception:
                        pass
        log_file = (run_dir / "train.log").open("a", encoding="utf-8")

        def log_print(msg: str = "") -> None:
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

        log_print(f" Resumed Training    : Continuing from Ep {start_ep} in {run_dir.name}")
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(experiments_root) / "runs" / f"{timestamp}_{ckpt_prefix}"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = run_dir / "metrics.jsonl"
        log_file = (run_dir / "train.log").open("a", encoding="utf-8")

        def log_print(msg: str = "") -> None:
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

        if warmstart_path:
            w_path = Path(warmstart_path)
            if w_path.exists():
                checkpoint = torch.load(w_path, map_location=device, weights_only=True)
                if "model_state_dict" in checkpoint:
                    agent.load_state_dict(checkpoint["model_state_dict"])
                else:
                    agent.load_state_dict(checkpoint)
                warmstart_loaded = True
                log_print(f" Warm-start Weights  : Loaded successfully from {w_path.name}")
            else:
                log_print(f" Warm-start Warning  : File not found at {w_path}. Starting from scratch.")
        else:
            log_print(" Training Mode       : From Scratch (Random initialization)")

    regular_save_interval = max(1, episodes // 5)

    log_print("=" * 86)
    log_print(f" PPO PURE VISUAL-SEARCH TRAINING -- {episodes} EPISODES (Start Ep: {start_ep})")
    log_print(f" Zero-Oracle Mode    : Patient coordinates MASKED to 0 during search")
    log_print(f" Vision Sensor       : 70° VFOV Pinhole Camera Model (Simulated YOLOv8)")
    log_print(f" Output Directory    : {run_dir}")
    log_print(f" Compute Device      : {device} ({describe_device(device)}) | Rollout: cpu")
    log_print(f" Escort Only Mode    : {escort_only} (100% Escort Tasks across Floors 1, 2, 3)")
    log_print(f" Stagnation Timeout  : {stagnation_timeout_s:.1f}s | Penalty: {stagnation_penalty:+.1f}")
    log_print(f" Shift Duration      : {shift_duration_s:.1f}s ({int(shift_duration_s / 0.2)} steps)")
    log_print(f" PPO Batch Update    : Every {update_freq_episodes} episodes | PPO Epochs: {ppo_epochs} | Batch: {batch_size}")
    log_print(f" Learning Rate       : {lr_init:.1e} -> {lr_min:.1e} (Cosine Annealing)")
    log_print(f" Checkpoint Saving   : Every {regular_save_interval} episodes (5 periodic checkpoints) + best/last")

    batch_obs: list[np.ndarray] = []
    batch_act: list[np.ndarray] = []
    batch_teacher: list[np.ndarray] = []
    batch_logp: list[float] = []
    batch_adv: list[np.ndarray] = []
    batch_ret: list[np.ndarray] = []

    total_helped = 0
    total_collisions = 0
    stagnation_timeouts = 0
    completed_shifts = 0

    run_config = {
        "episodes": episodes,
        "warmstart_path": str(warmstart_path) if warmstart_path else None,
        "warmstart_loaded": warmstart_loaded,
        "stagnation_timeout_s": stagnation_timeout_s,
        "stagnation_penalty": stagnation_penalty,
        "shift_duration_s": shift_duration_s,
        "update_freq_episodes": update_freq_episodes,
        "ppo_epochs": ppo_epochs,
        "batch_size": batch_size,
        "lr_init": lr_init,
        "lr_min": lr_min,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_coef": clip_coef,
        "seed": seed,
        "device": str(device),
        "eval_every": eval_every,
        "eval_episodes": eval_episodes,
        "eval_collision_weight": EVAL_COLLISION_WEIGHT,
        "patience": patience,
        "bc_enabled": bc_enabled,
        "env_mode": env_mode,
        "escort_only": escort_only,
        "pure_visual": True,
    }

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    log_print("-" * 86)
    log_print("EPISODE LOGS (Real-time single-line flush):")
    log_print("-" * 86)

    rollout_agent.load_state_dict(agent.state_dict())
    ep_width = len(str(episodes))

    t_start = time.time()
    completed_in_run = 0

    for ep in range(start_ep, episodes + 1):
        if warmstart_loaded:
            bc_weight = max(0.0, 0.05 * (1.0 - (ep - 1) / 300.0)) if bc_enabled else 0.0
        else:
            bc_weight = max(0.0, 0.15 * (1.0 - (ep - 1) / 500.0)) if bc_enabled else 0.0

        env = HospitalVisualShiftEnv(
            shift_duration_s=shift_duration_s,
            dt=0.2,
            seed=seed + ep * 79,
            env_mode=env_mode,
            stagnation_timeout_s=stagnation_timeout_s,
            stagnation_penalty=stagnation_penalty,
            escort_only=escort_only,
            domain_randomization=True,
        )
        obs = env.reset()

        ep_obs: list[np.ndarray] = []
        ep_act: list[np.ndarray] = []
        ep_teacher: list[np.ndarray] = []
        ep_logp: list[float] = []
        ep_val: list[float] = []
        ep_rew: list[float] = []

        done = False
        info: dict[str, Any] = {}
        ever_spotted = False

        while not done:
            teacher_act = env.teacher_action()
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_t, logp_t, _, val_t = rollout_agent.get_action_and_value(obs_t)

            policy_act = action_t.squeeze(0).numpy()
            exec_act = np.clip((1.0 - bc_weight) * policy_act + bc_weight * teacher_act, -1.0, 1.0)

            ep_obs.append(obs)
            ep_act.append(policy_act)
            ep_teacher.append(teacher_act)
            ep_logp.append(float(logp_t.item()))
            ep_val.append(float(val_t.item()))

            next_obs, reward, done, info = env.step(exec_act)
            if info.get("has_det", 0.0) > 0.5:
                ever_spotted = True

            ep_rew.append(float(reward))
            obs = next_obs

        # Exact GAE for completed episode
        ep_len = len(ep_rew)
        ep_adv = np.zeros(ep_len, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(ep_len)):
            nextnonterminal = 0.0 if t == ep_len - 1 else 1.0
            nextval = 0.0 if t == ep_len - 1 else ep_val[t + 1]
            delta = ep_rew[t] + gamma * nextval * nextnonterminal - ep_val[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            ep_adv[t] = lastgaelam
        ep_ret = ep_adv + np.asarray(ep_val, dtype=np.float32)

        batch_obs.extend(ep_obs)
        batch_act.extend(ep_act)
        batch_teacher.extend(ep_teacher)
        batch_logp.extend(ep_logp)
        batch_adv.append(ep_adv)
        batch_ret.append(ep_ret)

        r_team = float(info.get("cumulative_team_reward", 0.0))
        tasks = int(info.get("patients_helped", 0))
        collisions = int(info.get("collisions", 0))
        exit_reason = str(info.get("exit_reason", "UNKNOWN"))
        steps = int(env.step_count)
        duration_s = steps * 0.2

        rolling_20.append(r_team)
        roll20_mean = float(np.mean(rolling_20))
        if r_team > peak_r_team:
            peak_r_team = r_team

        total_helped += tasks
        total_collisions += collisions
        if exit_reason == "STAGNATION_TIMEOUT":
            stagnation_timeouts += 1
        elif exit_reason == "SHIFT_COMPLETED":
            completed_shifts += 1

        completed_in_run += 1
        elapsed_run = time.time() - t_start
        rate = completed_in_run / max(1e-6, elapsed_run)
        remaining_eps = episodes - ep
        eta_s = remaining_eps / rate if rate > 0 else 0.0

        log_print(
            f"Ep {ep:{ep_width}d}/{episodes} | "
            f"R_team: {r_team:+6.1f} (avg20: {roll20_mean:+6.1f}, peak: {peak_r_team:+6.1f}) | "
            f"Helped: {tasks} (total: {total_helped}) | "
            f"Col: {collisions} | "
            f"Spotted: {'YES' if ever_spotted else 'NO '} | "
            f"Exit: {exit_reason:<18s} | "
            f"{steps:4d} steps ({duration_s:4.1f}s) | "
            f"ETA: {format_eta(eta_s)}"
        )

        with metrics_file.open("a", encoding="utf-8") as f:
            rec = {
                "episode": ep,
                "r_team": r_team,
                "roll20_mean": roll20_mean,
                "patients_helped": tasks,
                "collisions": collisions,
                "spotted": ever_spotted,
                "steps": steps,
                "exit_reason": exit_reason,
                "lr": scheduler.get_last_lr()[0],
            }
            f.write(json.dumps(rec) + "\n")

        # PPO Batch Update
        if ep % update_freq_episodes == 0 and batch_obs:
            b_obs = torch.as_tensor(np.asarray(batch_obs), dtype=torch.float32, device=device)
            b_act = torch.as_tensor(np.asarray(batch_act), dtype=torch.float32, device=device)
            b_tea = torch.as_tensor(np.asarray(batch_teacher), dtype=torch.float32, device=device)
            b_logp = torch.as_tensor(np.asarray(batch_logp), dtype=torch.float32, device=device)

            flat_adv = np.concatenate(batch_adv)
            flat_ret = np.concatenate(batch_ret)
            adv_mean, adv_std = flat_adv.mean(), flat_adv.std() + 1e-8
            b_adv = torch.as_tensor((flat_adv - adv_mean) / adv_std, dtype=torch.float32, device=device)
            b_ret = torch.as_tensor(flat_ret, dtype=torch.float32, device=device)

            n_samples = b_obs.shape[0]
            indices = np.arange(n_samples)

            for _ in range(ppo_epochs):
                np.random.shuffle(indices)
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    mb_idx = indices[start_idx:end_idx]

                    _, new_logp, entropy, new_val = agent.get_action_and_value(b_obs[mb_idx], b_act[mb_idx])
                    logratio = new_logp - b_logp[mb_idx]
                    ratio = torch.exp(logratio)

                    mb_adv = b_adv[mb_idx]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((new_val.squeeze(-1) - b_ret[mb_idx]) ** 2).mean()
                    ent_loss = -entropy.mean()

                    loss = pg_loss + vf_coef * v_loss + ent_coef * ent_loss

                    if bc_weight > 0.0:
                        pred_mean = torch.tanh(agent.actor_mean(agent.actor_backbone(b_obs[mb_idx])))
                        bc_loss = ((pred_mean - b_tea[mb_idx]) ** 2).mean()
                        loss = loss + (0.2 * bc_weight) * bc_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()

            scheduler.step()
            rollout_agent.load_state_dict(agent.state_dict())

            batch_obs.clear()
            batch_act.clear()
            batch_teacher.clear()
            batch_logp.clear()
            batch_adv.clear()
            batch_ret.clear()

        # Evaluation & Checkpoints
        if ep % eval_every == 0 or ep == episodes:
            eval_metrics = evaluate_visual_policy(
                agent=rollout_agent,
                n_episodes=eval_episodes,
                stagnation_timeout_s=stagnation_timeout_s,
                stagnation_penalty=stagnation_penalty,
                shift_duration_s=shift_duration_s,
                env_mode=env_mode,
                escort_only=escort_only,
            )
            log_print("-" * 86)
            log_print(
                f"[EVAL @ Ep {ep}] Score: {eval_metrics['score']:.1f} | "
                f"R_team: {eval_metrics['r_team']:.1f} | "
                f"Tasks: {eval_metrics['tasks']:.2f} | "
                f"Col: {eval_metrics['collisions']:.2f} | "
                f"Spotting Rate: {eval_metrics['spotting_rate']*100:.1f}% | "
                f"Stagnation: {eval_metrics['stagnation_rate']*100:.1f}%"
            )
            log_print("-" * 86)

            ckpt_data = {
                "episode": ep,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_roll20": best_roll20,
                "best_eval_score": best_eval,
                "peak_r_team": peak_r_team,
                "git_commit": get_git_commit(),
                "rng_state": capture_rng_state(seed),
            }

            torch.save(ckpt_data, ckpt_dir / f"{ckpt_prefix}_last.pt")

            if eval_metrics["score"] > best_eval:
                best_eval = eval_metrics["score"]
                torch.save(ckpt_data, ckpt_dir / f"{ckpt_prefix}_best.pt")
                log_print(f" ★ New Best Eval Checkpoint! Score: {best_eval:.2f}")

        # Periodic regular checkpoint saving (adapts to total episodes, saving 5 evenly spaced checkpoints)
        if ep % regular_save_interval == 0:
            periodic_ckpt_data = {
                "episode": ep,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_roll20": best_roll20,
                "best_eval_score": best_eval,
                "peak_r_team": peak_r_team,
                "git_commit": get_git_commit(),
                "rng_state": capture_rng_state(seed),
            }
            periodic_ckpt_path = ckpt_dir / f"{ckpt_prefix}_ep{ep}.pt"
            torch.save(periodic_ckpt_data, periodic_ckpt_path)
            log_print(f" [CKPT] Saved periodic checkpoint ({ep}/{episodes}): {periodic_ckpt_path.name}")

    log_print("=" * 86)
    log_print(f" Training complete! Total episodes: {episodes}")
    log_print(f" Final Best Checkpoint: {ckpt_dir / f'{ckpt_prefix}_best.pt'}")
    try:
        log_file.close()
    except Exception:
        pass
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO Pure Visual-Search Training for Hospital Robot")
    parser.add_argument("--episodes", type=int, default=2000, help="Total episodes to train")
    parser.add_argument("--warmstart", type=str, default=DEFAULT_WARMSTART_PATH, help="Path to warm-start checkpoint")
    parser.add_argument("--no-warmstart", action="store_true", help="Start training from scratch")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from run dir or checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Run 2-episode smoke test")
    parser.add_argument("--stagnation-timeout", type=float, default=60.0)
    parser.add_argument("--stagnation-penalty", type=float, default=-10.0)
    parser.add_argument("--shift-duration", type=float, default=240.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    warmstart = None if args.no_warmstart else args.warmstart
    run_train_visual_rl(
        episodes=args.episodes,
        warmstart_path=warmstart,
        resume_from=args.resume,
        device_name=args.device,
        seed=args.seed,
        smoke=args.smoke,
        stagnation_timeout_s=args.stagnation_timeout,
        stagnation_penalty=args.stagnation_penalty,
        shift_duration_s=args.shift_duration,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":
    main()
