"""PPO training for the Pure-Vision V2 policy (21-D depth-sector observation, redesigned reward).

Fixes vs train_pure_vision_rl.py (V1):
- Stored action and its log-prob are the SAME (executed, pre-clip) action -> correct PPO ratio.
- Time-limit truncation (SHIFT_COMPLETED) bootstraps V(s_T) instead of treating it as terminal.
- Best checkpoint is chosen by a deterministic held-out evaluation, not by the noisy training avg20.
- Teacher (rule-based controller on the same observation) only warms up the first episodes, then 0.
- gamma = 0.995 so delivery rewards ~250-500 steps away are not discounted to nothing.

Usage:
  python src/training/train_pure_vision_v2.py --episodes 2000
  python src/training/train_pure_vision_v2.py --smoke
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.models.ppo_agent import ACT_DIM, OBS_DIM_PURE_VISION_V2, PPOHospitalAgent
from src.training.hospital_env_pure_vision_v2 import GAMMA_V2, HospitalPureVisionV2Env

CKPT_PREFIX = "PPOHospital-PureVisionV2"
EVAL_SEED_BASE = 970_000_000


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def make_env(seed: int, shift_s: float, dr: bool) -> HospitalPureVisionV2Env:
    return HospitalPureVisionV2Env(
        curriculum_stage=1, shift_duration_s=shift_s, dt=0.2, seed=seed, env_mode="full",
        stagnation_timeout_s=90.0, stagnation_penalty=0.0, escort_only=True, domain_randomization=dr,
    )


def policy_mean(agent: PPOHospitalAgent, obs_t: torch.Tensor) -> torch.Tensor:
    return torch.tanh(agent.actor_mean(agent.actor_backbone(obs_t)))


@torch.no_grad()
def evaluate(agent: PPOHospitalAgent | None, n: int, shift_s: float, teacher: bool = False) -> dict[str, float]:
    """Deterministic policy (or the rule-based teacher) on fixed held-out seeds."""
    R, H, C, V, full = [], [], [], [], 0
    for i in range(n):
        env = make_env(EVAL_SEED_BASE + i, shift_s, dr=False)
        obs, done = env.reset(), False
        while not done:
            if teacher:
                a = env.teacher_action()
            else:
                a = policy_mean(agent, torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
            obs, _, done, info = env.step(a)
        R.append(info["cumulative_team_reward"]); H.append(info["patients_helped"]); C.append(info["collisions"])
        V.append(info["escort_speed"]); full += int(info.get("exit_reason") == "ALL_PATIENTS_SERVED")
    return {"return": float(np.mean(R)), "helped": float(np.mean(H)), "collisions": float(np.mean(C)),
            "escort_speed": float(np.mean(V)), "all_served_rate": full / max(1, n)}


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    agent = PPOHospitalAgent(obs_dim=OBS_DIM_PURE_VISION_V2, act_dim=ACT_DIM)
    if args.resume:
        agent.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True)["model_state_dict"])
    opt = torch.optim.AdamW(agent.parameters(), lr=args.lr, eps=1e-5, weight_decay=0.0)
    est_updates = max(1, int(args.episodes * args.shift / 0.2 * 0.8 / args.steps_per_update))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=est_updates, eta_min=args.lr * 0.1)

    run_dir = Path(args.out) / "runs" / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{CKPT_PREFIX}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_f = (run_dir / "train.log").open("a", encoding="utf-8")
    metrics_f = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    def save(name: str, ep: int, extra: dict) -> None:
        torch.save({"episode": ep, "model_state_dict": agent.state_dict(), "optimizer_state_dict": opt.state_dict(),
                    "obs_dim": OBS_DIM_PURE_VISION_V2, "git_commit": git_commit(), **extra}, run_dir / "checkpoints" / name)

    log(f"PPO Pure-Vision V2 | episodes={args.episodes} shift={args.shift}s gamma={GAMMA_V2} lr={args.lr} "
        f"teacher {args.teacher_start:.2f}->0 over {args.teacher_episodes} eps | out={run_dir}")
    base = evaluate(None, args.eval_episodes, args.shift, teacher=True)
    log(f"[BASELINE rule-based teacher] {json.dumps({k: round(v, 3) for k, v in base.items()})}")

    buf: dict[str, list] = collections.defaultdict(list)
    roll = collections.deque(maxlen=20)
    best_score, t0 = -float("inf"), time.time()
    gamma, lam = GAMMA_V2, args.gae_lambda

    for ep in range(1, args.episodes + 1):
        w = args.teacher_start * max(0.0, 1.0 - (ep - 1) / max(1, args.teacher_episodes))
        env = make_env(args.seed + ep * 17, args.shift, dr=True)
        obs, done = env.reset(), False
        e_obs, e_act, e_tea, e_logp, e_rew, e_val = [], [], [], [], [], []
        while not done:
            teacher = env.teacher_action() if w > 0.0 else np.zeros(ACT_DIM, dtype=np.float32)
            obs_t = torch.as_tensor(obs).unsqueeze(0)
            with torch.no_grad():
                mean = policy_mean(agent, obs_t)
                dist = Normal(mean, agent.actor_logstd.expand_as(mean).exp())
                a_pi = dist.sample().squeeze(0).numpy()
                a = (1.0 - w) * a_pi + w * teacher if w > 0.0 else a_pi   # stored == evaluated action
                logp = float(dist.log_prob(torch.as_tensor(a).unsqueeze(0)).sum(-1))
                val = float(agent.critic(obs_t))
            next_obs, r, done, info = env.step(np.clip(a, -1.0, 1.0))
            e_obs.append(obs); e_act.append(a.astype(np.float32)); e_tea.append(teacher)
            e_logp.append(logp); e_rew.append(r); e_val.append(val)
            obs = next_obs

        # GAE; bootstrap through the time limit
        last_v = 0.0
        if info.get("truncated"):
            with torch.no_grad():
                last_v = float(agent.critic(torch.as_tensor(obs).unsqueeze(0)))
        adv = np.zeros(len(e_rew), dtype=np.float32)
        gae, next_v = 0.0, last_v
        for t in reversed(range(len(e_rew))):
            delta = e_rew[t] + gamma * next_v - e_val[t]
            gae = delta + gamma * lam * gae
            adv[t], next_v = gae, e_val[t]
        for k, v in (("obs", e_obs), ("act", e_act), ("tea", e_tea), ("logp", e_logp)):
            buf[k].extend(v)
        buf["adv"].extend(adv.tolist())
        buf["ret"].extend((adv + np.asarray(e_val, dtype=np.float32)).tolist())
        buf["w"].extend([w] * len(e_rew))

        R = info["cumulative_team_reward"]
        roll.append(R)
        rec = {"episode": ep, "return": R, "roll20": float(np.mean(roll)), "helped": info["patients_helped"],
               "collisions": info["collisions"], "escort_speed": info["escort_speed"], "steps": env.step_count,
               "exit": info.get("exit_reason", ""), "teacher_w": w, "lr": sched.get_last_lr()[0]}
        metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
        el = time.time() - t0
        log(f"Ep {ep:5d}/{args.episodes} | R {R:+7.2f} (avg20 {rec['roll20']:+7.2f}) | helped {rec['helped']}/6 | "
            f"col {rec['collisions']:3d} | v_esc {rec['escort_speed']:.2f} | {rec['exit']:<19s} | w {w:.2f} | "
            f"ETA {el / ep * (args.episodes - ep) / 60:5.1f} min")

        # PPO update
        if len(buf["obs"]) >= args.steps_per_update or ep == args.episodes:
            b_obs = torch.as_tensor(np.asarray(buf["obs"]))
            b_act = torch.as_tensor(np.asarray(buf["act"]))
            b_tea = torch.as_tensor(np.asarray(buf["tea"]))
            b_logp = torch.as_tensor(np.asarray(buf["logp"], dtype=np.float32))
            b_adv = torch.as_tensor(np.asarray(buf["adv"], dtype=np.float32))
            b_ret = torch.as_tensor(np.asarray(buf["ret"], dtype=np.float32))
            b_w = torch.as_tensor(np.asarray(buf["w"], dtype=np.float32))
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            n = b_obs.shape[0]
            stats = collections.defaultdict(list)
            for _ in range(args.ppo_epochs):
                perm = torch.randperm(n)
                for s in range(0, n, args.batch_size):
                    idx = perm[s:s + args.batch_size]
                    mean = policy_mean(agent, b_obs[idx])
                    dist = Normal(mean, agent.actor_logstd.expand_as(mean).exp())
                    new_logp = dist.log_prob(b_act[idx]).sum(-1)
                    ratio = torch.exp(new_logp - b_logp[idx])
                    pg = -torch.min(ratio * b_adv[idx], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b_adv[idx]).mean()
                    v_loss = 0.5 * ((agent.critic(b_obs[idx]).squeeze(-1) - b_ret[idx]) ** 2).mean()
                    ent = dist.entropy().sum(-1).mean()
                    bc = (b_w[idx] * ((mean - b_tea[idx]) ** 2).sum(-1)).mean()
                    loss = pg + args.vf_coef * v_loss - args.ent_coef * ent + args.bc_coef * bc
                    if not torch.isfinite(loss):
                        continue
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    opt.step()
                    with torch.no_grad():
                        agent.actor_logstd.clamp_(-2.5, 0.0)
                    stats["pg"].append(pg.item()); stats["v"].append(v_loss.item()); stats["ent"].append(ent.item())
                    stats["kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
            sched.step()
            log(f"  [PPO] n={n} pg {np.mean(stats['pg']):+.4f} v {np.mean(stats['v']):.4f} ent {np.mean(stats['ent']):.3f} "
                f"kl {np.mean(stats['kl']):.4f} std {agent.actor_logstd.exp().detach().numpy().round(3).tolist()} lr {sched.get_last_lr()[0]:.2e}")
            buf.clear()

        if ep % args.eval_every == 0 or ep == args.episodes:
            ev = evaluate(agent, args.eval_episodes, args.shift)
            score = ev["return"]
            tag = ""
            if score > best_score:
                best_score, tag = score, "  <- new best"
                save(f"{CKPT_PREFIX}_best.pt", ep, {"eval": ev})
            log(f"[EVAL @ ep {ep}] {json.dumps({k: round(v, 3) for k, v in ev.items()})}{tag}")
            metrics_f.write(json.dumps({"episode": ep, "eval": ev}) + "\n"); metrics_f.flush()
    save(f"{CKPT_PREFIX}_last.pt", args.episodes, {})
    log(f"Done. Best eval return {best_score:.2f}. Checkpoints in {run_dir / 'checkpoints'}")
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="PPO Pure-Vision V2 training")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--shift", type=float, default=700.0, help="shift length in seconds (dt = 0.2 s)")
    p.add_argument("--steps-per-update", type=int, default=8192)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.002)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--bc-coef", type=float, default=1.0)
    p.add_argument("--teacher-start", type=float, default=0.5)
    p.add_argument("--teacher-episodes", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--out", type=str, default="experiments")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.episodes, args.shift, args.steps_per_update, args.eval_every, args.eval_episodes = 3, 20.0, 150, 3, 2
    train(args)


if __name__ == "__main__":
    main()
