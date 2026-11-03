"""PPO Pure-Vision (YOLO Only, Zero LiDAR) Training for Hospital Robot.

Key Architectural Highlights:
- OBS_DIM = 29 (NO physical LiDAR sensors!)
- 5-Sector Visual Depth Map (Far-Left, Left, Center, Right, Far-Right)
- Dedicated Patient & Doctor Perception Channels
- Doctor Proximity Evasion (-5.0 false engagement penalty)
- Visual Obstacle Proximity Repulsion (-0.50 * sector_h when h > 0.50)
- Physical bumper collision penalty (-5.0)
- Teacher Behavioral Cloning Warmup decaying over first 200 episodes
- High-throughput GPU execution on NVIDIA RTX 4060
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.models.ppo_agent import ACT_DIM, OBS_DIM_PURE_VISION, PPOHospitalAgent
from src.training.hospital_env_pure_vision import HospitalPureVisionEnv

EVAL_SEED_BASE = 970_000_000
EVAL_COLLISION_WEIGHT = 1.0


def format_eta(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m:02d}m{s:02d}s"


def resolve_device(device_name: str) -> torch.device:
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


def get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def evaluate_pure_vision_policy(
    agent: PPOHospitalAgent,
    n_episodes: int,
    stagnation_timeout_s: float,
    stagnation_penalty: float,
    shift_duration_s: float,
    env_mode: str = "full",
    escort_only: bool = True,
) -> dict[str, float]:
    r_team, tasks, cols, stag = [], [], [], 0
    spotted_count = 0

    for i in range(n_episodes):
        eval_seed = EVAL_SEED_BASE + i
        env = HospitalPureVisionEnv(
            curriculum_stage=1,
            shift_duration_s=shift_duration_s,
            dt=0.2,
            seed=eval_seed,
            env_mode=env_mode,
            stagnation_timeout_s=stagnation_timeout_s,
            stagnation_penalty=stagnation_penalty,
            escort_only=escort_only,
            domain_randomization=False,
        )
        obs = env.reset()
        done = False
        ever_spotted = False
        dev = next(agent.parameters()).device
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                action_mean = agent.actor_mean(agent.actor_backbone(obs_t))
                action = torch.tanh(action_mean).squeeze(0).cpu().numpy()
            obs, _, done, info = env.step(action)
            if info.get("has_det", 0.0) > 0.5:
                ever_spotted = True

        r_team.append(float(info.get("cumulative_team_reward", 0.0)))
        tasks.append(float(info.get("patients_helped", 0)))
        cols.append(float(info.get("collisions", 0)))
        if info.get("exit_reason") == "STAGNATION_TIMEOUT":
            stag += 1
        if ever_spotted:
            spotted_count += 1

    mean_r = float(np.mean(r_team))
    mean_col = float(np.mean(cols))
    score = mean_r - EVAL_COLLISION_WEIGHT * mean_col
    return {
        "score": score,
        "r_team": mean_r,
        "tasks": float(np.mean(tasks)),
        "collisions": mean_col,
        "stagnation_rate": stag / max(1, n_episodes),
        "spotting_rate": spotted_count / max(1, n_episodes),
    }


def run_train_pure_vision(
    episodes: int = 2000,
    start_episode: int = 1,
    resume_from: str | Path | None = None,
    stagnation_timeout_s: float = 60.0,
    stagnation_penalty: float = -10.0,
    shift_duration_s: float = 300.0,
    steps_per_epoch: int = 10000,
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
    eval_episodes: int = 15,
    bc_enabled: bool = True,
    escort_curriculum: bool = True,
    env_mode: str = "full",
    smoke: bool = False,
    visual: bool = False,
    mock_train: bool = False,
    fps: int = 30,
) -> Path:
    if smoke:
        episodes = 2
        shift_duration_s = 6.0
        steps_per_epoch = 60
        update_freq_episodes = 1
        eval_every = 2
        eval_episodes = 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device_name)

    agent = PPOHospitalAgent(obs_dim=OBS_DIM_PURE_VISION, act_dim=ACT_DIM).to(device)
    rollout_agent = PPOHospitalAgent(obs_dim=OBS_DIM_PURE_VISION, act_dim=ACT_DIM).to(device)
    rollout_agent.eval()

    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr_init, eps=1e-5)
    approx_steps_per_ep = max(30, int(shift_duration_s / 0.2) * 0.75)
    total_steps_est = episodes * approx_steps_per_ep
    if steps_per_epoch > 0:
        total_updates = max(1, int(total_steps_est / steps_per_epoch))
    else:
        total_updates = max(1, episodes // update_freq_episodes)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=lr_min
    )

    rolling_20: collections.deque[float] = collections.deque(maxlen=20)
    best_roll20 = -float("inf")
    peak_r_team = -float("inf")
    total_helped = 0
    total_collisions = 0
    stagnation_timeouts = 0
    completed_shifts = 0
    completed_in_run = 0

    start_ep = start_episode
    end_ep = start_ep + episodes - 1
    best_eval = -float("inf")
    ckpt_prefix = "PPOHospital-PureVision"

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

    if resume_from:
        r_path = Path(resume_from)
        ckpt_file = r_path if r_path.is_file() else (r_path / "checkpoints" / f"{ckpt_prefix}_best.pt")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)
        agent.load_state_dict(ckpt["model_state_dict"])
        log_print(f" Initialized from Checkpoint : Loaded weights from {ckpt_file.name}")

    rollout_agent.load_state_dict(agent.state_dict())
    regular_save_interval = max(1, episodes // 5)

    log_print("=" * 86)
    log_print(" PPO PURE-VISION HOSPITAL ROBOT TRAINING (YOLO Only, Zero LiDAR)")
    log_print("=" * 86)
    log_print(f" Target Device       : {device.type.upper()} ({describe_device(device)})")
    log_print(f" Training Span       : Ep {start_ep} -> Ep {end_ep} ({episodes} episodes)")
    log_print(f" Observation Space   : 29 Dims (5-Sector Visual Depth + Patient/Doctor Channels)")
    log_print(f" Environment Mode    : {env_mode.upper()} (Crowded Hospital Multi-Floor Navigation)")
    log_print(f" Shift Duration      : {shift_duration_s:.1f}s ({int(shift_duration_s / 0.2)} steps)")
    if steps_per_epoch > 0:
        log_print(f" Steps per Epoch     : {steps_per_epoch:,d} steps (~{max(1, round(steps_per_epoch / (shift_duration_s / 0.2)))} shifts/epoch)")
    else:
        log_print(f" Update Frequency    : Every {update_freq_episodes} shifts")
    log_print(f" Stagnation Timeout  : {stagnation_timeout_s:.1f}s (Penalty: {stagnation_penalty:.1f})")
    log_print(f" Periodic Checkpoint : Every {regular_save_interval} episodes (Target: 5 checkpoints)")
    if mock_train:
        log_print(" Mock-Train Mode     : ENABLED (Weights updated in memory, NO checkpoints saved to disk)")
    if visual:
        log_print(f" Visual GUI Renderer : ENABLED (Target FPS: {fps})")
    log_print(f" Output Directory    : {run_dir}")
    log_print("=" * 86)

    renderer = None
    if visual:
        try:
            from src.ui.visual_env_renderer import VisualEnvRenderer
            renderer = VisualEnvRenderer(
                window_name="Hospital Pure-Vision RL Training Visualizer",
                target_fps=fps,
            )
        except Exception as e:
            log_print(f"[WARN] Failed to initialize VisualEnvRenderer: {e}. Falling back to headless.")
            renderer = None

    t_start = time.time()
    batch_obs, batch_act, batch_teacher, batch_logp = [], [], [], []
    batch_adv, batch_ret = [], []
    ep_width = len(str(end_ep))
    user_exit = False

    for ep in range(start_ep, end_ep + 1):
        ep_seed = seed + ep * 17
        env = HospitalPureVisionEnv(
            curriculum_stage=1,
            shift_duration_s=shift_duration_s,
            dt=0.2,
            seed=ep_seed,
            env_mode=env_mode,
            stagnation_timeout_s=stagnation_timeout_s,
            stagnation_penalty=stagnation_penalty,
            escort_only=True,
            domain_randomization=True,
        )

        # Behavioral Cloning Teacher Warmup (Decays from 0.20 to 0.0 over warmup_span episodes)
        warmup_span = 100 if resume_from else 200
        progress_in_warmup = max(0.0, min(1.0, (ep - start_ep) / float(warmup_span)))
        prior_weight = 0.20 * (1.0 - progress_in_warmup) if bc_enabled else 0.0

        obs = env.reset()
        done = False
        ep_obs, ep_act, ep_teacher, ep_logp, ep_rew, ep_val = [], [], [], [], [], []
        ever_spotted = False

        while not done:
            teacher_act = env.teacher_action()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                features = rollout_agent.actor_backbone(obs_t)
                action_mean = torch.tanh(rollout_agent.actor_mean(features))
                action_logstd = rollout_agent.actor_logstd.expand_as(action_mean)
                dist = Normal(action_mean, action_logstd.exp())
                pi_action = dist.sample()
                logp = dist.log_prob(pi_action).sum(-1)
                value = rollout_agent.critic(obs_t).squeeze(-1)

            pi_act_np = pi_action.squeeze(0).cpu().numpy()

            if prior_weight > 0.0:
                blended_act = (1.0 - prior_weight) * pi_act_np + prior_weight * teacher_act
                executed_act = np.clip(blended_act, -1.0, 1.0)
            else:
                executed_act = np.clip(pi_act_np, -1.0, 1.0)

            next_obs, reward, done, info = env.step(executed_act)

            if renderer is not None:
                keep_going = renderer.render_step(
                    env=env,
                    action=executed_act,
                    reward=reward,
                    step_idx=env.step_count,
                    episode_idx=ep,
                    total_episodes=end_ep,
                    info=info,
                    is_mock_train=mock_train,
                )
                if not keep_going:
                    user_exit = True
                    break

            if info.get("has_det", 0.0) > 0.5:
                ever_spotted = True

            ep_obs.append(np.nan_to_num(obs, nan=0.0).astype(np.float32))
            ep_act.append(executed_act)
            ep_teacher.append(teacher_act)
            ep_logp.append(float(logp.item()))
            ep_rew.append(float(np.nan_to_num(reward, nan=0.0)))
            val_num = float(np.nan_to_num(value.item(), nan=0.0))
            ep_val.append(val_num)
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
        ep_adv = np.nan_to_num(ep_adv, nan=0.0)
        ep_ret = np.nan_to_num(ep_adv + np.asarray(ep_val, dtype=np.float32), nan=0.0)

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
        remaining_eps = end_ep - ep
        eta_s = remaining_eps / rate if rate > 0 else 0.0

        log_print(
            f"Ep {ep:{ep_width}d}/{end_ep} | "
            f"R_team: {r_team:+6.1f} (avg20: {roll20_mean:+6.1f}, peak: {peak_r_team:+6.1f}) | "
            f"Helped: {tasks}/6 (total: {total_helped}) | "
            f"Col: {collisions} | "
            f"Spotted: {'YES' if ever_spotted else 'NO '} | "
            f"Exit: {exit_reason:<18s} | "
            f"{steps:4d} steps ({duration_s:4.1f}s) | "
            f"ETA: {format_eta(eta_s)}"
        )

        if user_exit:
            log_print("\n[INFO] User requested clean exit from visual GUI.")
            break

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
        should_update = (len(batch_obs) >= steps_per_epoch) if steps_per_epoch > 0 else (ep % update_freq_episodes == 0)
        if (should_update or ep == end_ep) and batch_obs:
            b_obs = torch.as_tensor(np.asarray(batch_obs), dtype=torch.float32, device=device)
            b_act = torch.as_tensor(np.asarray(batch_act), dtype=torch.float32, device=device)
            b_tea = torch.as_tensor(np.asarray(batch_teacher), dtype=torch.float32, device=device)
            b_logp = torch.as_tensor(np.asarray(batch_logp), dtype=torch.float32, device=device)
            flat_adv = np.concatenate(batch_adv, axis=0)
            flat_ret = np.concatenate(batch_ret, axis=0)

            flat_adv = np.nan_to_num(flat_adv, nan=0.0, posinf=0.0, neginf=0.0)
            flat_ret = np.nan_to_num(flat_ret, nan=0.0, posinf=0.0, neginf=0.0)
            adv_std = float(flat_adv.std())
            if adv_std > 1e-4:
                flat_adv = (flat_adv - flat_adv.mean()) / (adv_std + 1e-8)
            else:
                flat_adv = flat_adv - flat_adv.mean()
            flat_adv = np.clip(flat_adv, -5.0, 5.0)

            b_adv = torch.as_tensor(flat_adv, dtype=torch.float32, device=device)
            b_ret = torch.as_tensor(flat_ret, dtype=torch.float32, device=device)

            dataset_size = b_obs.shape[0]
            indices = np.arange(dataset_size)

            update_interrupted = False
            for _ in range(ppo_epochs):
                if update_interrupted:
                    break
                np.random.shuffle(indices)
                for start_idx in range(0, dataset_size, batch_size):
                    end_idx = min(start_idx + batch_size, dataset_size)
                    batch_idx = indices[start_idx:end_idx]

                    m_obs = b_obs[batch_idx]
                    m_act = b_act[batch_idx]
                    m_tea = b_tea[batch_idx]
                    m_old_logp = b_logp[batch_idx]
                    m_adv = b_adv[batch_idx]
                    m_ret = b_ret[batch_idx]

                    features = agent.actor_backbone(m_obs)
                    action_mean = torch.tanh(agent.actor_mean(features))
                    action_logstd = agent.actor_logstd.expand_as(action_mean)
                    dist = Normal(action_mean, action_logstd.exp())

                    new_logp = dist.log_prob(m_act).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()
                    new_val = agent.critic(m_obs).squeeze(-1)

                    ratio = torch.exp(new_logp - m_old_logp)
                    surr1 = ratio * m_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * m_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    v_loss = 0.5 * ((new_val - m_ret) ** 2).mean()
                    bc_loss = ((action_mean - m_tea) ** 2).mean()

                    last_policy_loss = float(policy_loss.item())
                    last_v_loss = float(v_loss.item())
                    last_entropy = float(entropy.item())
                    last_bc_loss = float(bc_loss.item())

                    total_loss = (
                        policy_loss
                        + vf_coef * v_loss
                        - ent_coef * entropy
                        + 0.5 * prior_weight * bc_loss
                    )

                    if not torch.isfinite(total_loss):
                        log_print(f"[WARN] Non-finite loss at episode {ep}: skipping step.")
                        optimizer.zero_grad()
                        continue

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)

                    params_ok = all(torch.isfinite(p).all() for p in agent.parameters())
                    if params_ok:
                        optimizer.step()
                    else:
                        log_print(f"[WARN] Non-finite parameter detected at episode {ep}: rolling back.")
                        agent.load_state_dict(rollout_agent.state_dict())
                        optimizer.zero_grad()
                        update_interrupted = True
                        break

            scheduler.step()
            rollout_agent.load_state_dict(agent.state_dict())
            if renderer is not None:
                renderer.update_ppo_stats({
                    "policy_loss": last_policy_loss if 'last_policy_loss' in locals() else 0.0,
                    "value_loss": last_v_loss if 'last_v_loss' in locals() else 0.0,
                    "entropy": last_entropy if 'last_entropy' in locals() else 0.0,
                    "bc_loss": last_bc_loss if 'last_bc_loss' in locals() else 0.0,
                    "lr": float(scheduler.get_last_lr()[0]),
                })

            batch_obs.clear()
            batch_act.clear()
            batch_teacher.clear()
            batch_logp.clear()
            batch_adv.clear()
            batch_ret.clear()

        # Save Best Model on rolling-20 peak
        if len(rolling_20) >= 10 and roll20_mean > best_roll20:
            best_roll20 = roll20_mean
            if not mock_train:
                ckpt_path = ckpt_dir / f"{ckpt_prefix}_best.pt"
                torch.save(
                    {
                        "episode": ep,
                        "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_roll20": best_roll20,
                        "best_eval_score": best_eval,
                        "peak_r_team": peak_r_team,
                        "git_commit": get_git_commit(),
                    },
                    ckpt_path,
                )
            else:
                log_print(f" [MOCK-TRAIN] Peak avg20 reward reached ({best_roll20:+.1f}) | Checkpoint saving bypassed.")

        # Periodic Evaluation
        if ep % eval_every == 0:
            agent.eval()
            eval_metrics = evaluate_pure_vision_policy(
                agent=agent,
                n_episodes=eval_episodes,
                stagnation_timeout_s=stagnation_timeout_s,
                stagnation_penalty=stagnation_penalty,
                shift_duration_s=shift_duration_s,
                env_mode=env_mode,
            )
            agent.train()
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

        # Regular Checkpoint Saving (Target ~5 checkpoints)
        if ep % regular_save_interval == 0 or ep == end_ep:
            if not mock_train:
                periodic_ckpt_path = ckpt_dir / f"{ckpt_prefix}_ep{ep}.pt"
                torch.save(
                    {
                        "episode": ep,
                        "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_roll20": best_roll20,
                        "best_eval_score": best_eval,
                        "peak_r_team": peak_r_team,
                        "git_commit": get_git_commit(),
                    },
                    periodic_ckpt_path,
                )
                log_print(f" [CKPT] Saved periodic checkpoint ({ep - start_ep + 1}/{episodes}): {periodic_ckpt_path.name}")
            else:
                log_print(f" [MOCK-TRAIN] Episode {ep} completed | Periodic checkpoint saving bypassed.")

    if renderer is not None:
        renderer.close()

    log_print("=" * 86)
    log_print(f" Pure-Vision Training complete! Episodes completed: {completed_in_run} (Final Ep: {end_ep})")
    if not mock_train:
        log_print(f" Final Best Checkpoint: {ckpt_dir / f'{ckpt_prefix}_best.pt'}")
    else:
        log_print(" [MOCK-TRAIN] Mock training finished cleanly without modifying checkpoint files.")
    try:
        log_file.close()
    except Exception:
        pass

    # Generate training curves plot and print summary table
    try:
        from scripts.plot_training_metrics import load_metrics, plot_metrics, print_summary_table
        if metrics_file.exists():
            data = load_metrics(metrics_file)
            print_summary_table(data)
            curve_png = run_dir / "training_curves.png"
            plot_metrics(data, curve_png, title_suffix=f"({run_dir.name})", show=False)
            print(f"[INFO] Training performance curves successfully saved to: {curve_png}")
    except Exception as e:
        print(f"[WARN] Failed to auto-plot training metrics: {e}")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO Pure-Vision (YOLO Only, Zero LiDAR) Training")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--start-episode", type=int, default=1)
    parser.add_argument("--resume", "--resume-from", dest="resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--escort-curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stagnation-timeout", type=float, default=60.0)
    parser.add_argument("--stagnation-penalty", type=float, default=-10.0)
    parser.add_argument("--shift-duration", type=float, default=300.0)
    parser.add_argument("--steps-per-epoch", type=int, default=10000, help="Rollout environment steps gathered per PPO update epoch (default: 10000)")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=15)
    parser.add_argument("--visual", "--render", dest="visual", action="store_true", help="Launch interactive non-headless visual GUI")
    parser.add_argument("--mock-train", "--no-save", dest="mock_train", action="store_true", help="Execute complete training pipeline without saving checkpoints")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for visual rendering")
    args = parser.parse_args()

    run_train_pure_vision(
        episodes=args.episodes,
        start_episode=args.start_episode,
        resume_from=args.resume,
        device_name=args.device,
        seed=args.seed,
        smoke=args.smoke,
        escort_curriculum=args.escort_curriculum,
        stagnation_timeout_s=args.stagnation_timeout,
        stagnation_penalty=args.stagnation_penalty,
        shift_duration_s=args.shift_duration,
        steps_per_epoch=args.steps_per_epoch,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        visual=args.visual,
        mock_train=args.mock_train,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
