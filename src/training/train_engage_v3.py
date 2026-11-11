"""PPO training for the Engagement V3 policy (31-D obs, 4-D action) — docs/RL_V3_DESIGN.md.

Same PPO as train_pure_vision_v2.py, plus:
- warm start from the V2 checkpoint (old 21 inputs / 2 outputs copied, new ones start at 0, "ask" biased off);
- teacher mixing by switching (with prob. w the whole teacher action is executed) instead of blending: the "ask"
  action is a threshold (> 0.5), and a 50/50 blend of teacher 1.0 and policy -0.96 never crosses it;
- evaluation also reports the engagement metrics (face enrolment, wasted asks, callers who gave up).

Usage:
  python src/training/train_engage_v3.py --episodes 2000
  python src/training/train_engage_v3.py --smoke
"""

from __future__ import annotations

import argparse
import math
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

from src.models.ppo_agent import PPOHospitalAgent
from src.training.hospital_env_engage_v3 import ACT_DIM_V3, OBS_DIM_V3, HospitalEngageV3Env
from src.training.hospital_env_pure_vision_v2 import GAMMA_V2, OBS_DIM_V2

CKPT_PREFIX = "PPOHospital-EngageV3"
EVAL_SEED_BASE = 970_000_000
V2_CKPT = ROOT_DIR / "experiments" / "checkpoints" / "PPOHospital-PureVisionV2_best.pt"
ASK_BIAS = -2.0          # tanh(-2) = -0.96: no asking until the policy learns it


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def make_env(seed: int, shift_s: float, dr: bool) -> HospitalEngageV3Env:
    return HospitalEngageV3Env(
        curriculum_stage=1, shift_duration_s=shift_s, dt=0.2, seed=seed, env_mode="full",
        stagnation_timeout_s=90.0, stagnation_penalty=0.0, escort_only=True, domain_randomization=dr,
    )


def policy_mean(agent: PPOHospitalAgent, obs_t: torch.Tensor) -> torch.Tensor:
    return torch.tanh(agent.actor_mean(agent.actor_backbone(obs_t)))


def warm_start_from_v2(agent: PPOHospitalAgent, ckpt: Path) -> None:
    """Copy the V2 network into the wider V3 one: new inputs / outputs start neutral."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
    with torch.no_grad():
        for net in ("actor_backbone", "critic"):
            w = getattr(agent, net)[0].weight
            w.zero_()
            w[:, :OBS_DIM_V2] = sd[f"{net}.0.weight"]
            getattr(agent, net)[0].bias.copy_(sd[f"{net}.0.bias"])
        agent.actor_backbone[2].load_state_dict({"weight": sd["actor_backbone.2.weight"], "bias": sd["actor_backbone.2.bias"]})
        for k in ("2", "4"):
            agent.critic[int(k)].load_state_dict({"weight": sd[f"critic.{k}.weight"], "bias": sd[f"critic.{k}.bias"]})
        agent.actor_mean.weight.zero_()
        agent.actor_mean.bias.zero_()
        agent.actor_mean.weight[:2] = sd["actor_mean.weight"]
        agent.actor_mean.bias[:2] = sd["actor_mean.bias"]
        agent.actor_mean.bias[3] = ASK_BIAS
        agent.actor_logstd[:, :2] = sd["actor_logstd"]
        agent.actor_logstd[:, 2:] = -0.5


def load_widening(agent: PPOHospitalAgent, ckpt: Path) -> None:
    """Load a V3 checkpoint; if it has fewer observation inputs (e.g. 31-D before the 'face already met' input), copy
    the old input columns and start the new ones at 0 -> same behaviour as the old policy at the start."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
    own = agent.state_dict()
    with torch.no_grad():
        for k, v in sd.items():
            if own[k].shape == v.shape:
                own[k].copy_(v)
            elif k.endswith("0.weight") and own[k].shape[0] == v.shape[0] and own[k].shape[1] > v.shape[1]:
                own[k].zero_()
                own[k][:, : v.shape[1]] = v
            else:
                raise ValueError(f"cannot widen {k}: {tuple(v.shape)} -> {tuple(own[k].shape)}")


@torch.no_grad()
def evaluate(agent: PPOHospitalAgent | None, n: int, shift_s: float, teacher: bool = False) -> dict[str, float]:
    """Deterministic policy (or the rule-based teacher) on fixed held-out seeds (people / callers randomised)."""
    R, H, C, V, full = [], [], [], [], 0
    st = collections.Counter()
    for i in range(n):
        env = make_env(EVAL_SEED_BASE + i, shift_s, dr=True)
        env.rng = np.random.default_rng(EVAL_SEED_BASE + i)
        obs, done = env.reset(seed=EVAL_SEED_BASE + i), False
        while not done:
            a = env.teacher_action() if teacher else policy_mean(agent, torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
            obs, _, done, info = env.step(a)
        R.append(info["cumulative_team_reward"]); H.append(info["patients_helped"]); C.append(info["collisions"])
        V.append(info["escort_speed"]); full += int(info.get("exit_reason") == "ALL_PATIENTS_SERVED")
        st.update(info["v3"])
    dlg = max(1, st["dialogues"])
    return {"return": float(np.mean(R)), "helped": float(np.mean(H)), "collisions": float(np.mean(C)),
            "escort_speed": float(np.mean(V)), "all_served_rate": full / max(1, n),
            "enroll_rate": st["enrolled"] / dlg, "face_frac": st["face_steps"] / max(1, st["dialogue_steps"]),
            "walkup_enroll_rate": st.get("walkup_enrolled", 0) / max(1, st.get("walkup", 0)),
            "asks_per_ep": st["asks"] / n, "wasted_asks_per_ep": (st["asks_empty"] + st["asks_repeat"]) / n,
            "callers_gave_up_per_ep": st["callers_gave_up"] / n, "aborted_per_ep": st["aborted"] / n}


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    rng = np.random.default_rng(args.seed)
    agent = PPOHospitalAgent(obs_dim=OBS_DIM_V3, act_dim=ACT_DIM_V3)
    if args.resume:
        load_widening(agent, Path(args.resume))
        if args.reset_omega_std > 0.0:      # re-open exploration of ω (it had collapsed before learning the voice side)
            with torch.no_grad():
                agent.actor_logstd[0, 1] = math.log(args.reset_omega_std)
    elif not args.from_scratch:
        warm_start_from_v2(agent, Path(args.v2_ckpt))
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
                    "obs_dim": OBS_DIM_V3, "act_dim": ACT_DIM_V3, "git_commit": git_commit(), **extra},
                   run_dir / "checkpoints" / name)

    log(f"PPO Engagement V3 | episodes={args.episodes} shift={args.shift}s gamma={GAMMA_V2} lr={args.lr} "
        f"teacher {args.teacher_start:.2f}->0 over {args.teacher_episodes} eps (switching) | "
        f"init={'resume' if args.resume else ('scratch' if args.from_scratch else 'V2 warm start')} | out={run_dir}")
    base = evaluate(None, args.eval_episodes, args.shift, teacher=True)
    log(f"[BASELINE rule-based teacher] {json.dumps({k: round(v, 3) for k, v in base.items()})}")
    init = evaluate(agent, args.eval_episodes, args.shift)
    log(f"[INIT policy] {json.dumps({k: round(v, 3) for k, v in init.items()})}")

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
            teacher = env.teacher_action() if w > 0.0 else np.zeros(ACT_DIM_V3, dtype=np.float32)
            obs_t = torch.as_tensor(obs).unsqueeze(0)
            with torch.no_grad():
                mean = policy_mean(agent, obs_t)
                dist = Normal(mean, agent.actor_logstd.expand_as(mean).exp())
                a_pi = dist.sample().squeeze(0).numpy()
                a = teacher.copy() if (w > 0.0 and rng.random() < w) else a_pi   # stored == executed action
                logp = float(dist.log_prob(torch.as_tensor(a).unsqueeze(0)).sum(-1))
                val = float(agent.critic(obs_t))
            next_obs, r, done, info = env.step(np.clip(a, -1.0, 1.0))
            e_obs.append(obs); e_act.append(a.astype(np.float32)); e_tea.append(teacher)
            e_logp.append(logp); e_rew.append(r); e_val.append(val)
            obs = next_obs

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
        v3 = info["v3"]
        rec = {"episode": ep, "return": R, "roll20": float(np.mean(roll)), "helped": info["patients_helped"],
               "collisions": info["collisions"], "escort_speed": info["escort_speed"], "steps": env.step_count,
               "exit": info.get("exit_reason", ""), "teacher_w": w, "lr": sched.get_last_lr()[0], "v3": v3}
        metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
        el = time.time() - t0
        log(f"Ep {ep:5d}/{args.episodes} | R {R:+7.2f} (avg20 {rec['roll20']:+7.2f}) | helped {rec['helped']}/6 | "
            f"col {rec['collisions']:3d} | dlg {v3['dialogues']} enr {v3['enrolled']} ask {v3['asks']} "
            f"waste {v3['asks_empty'] + v3['asks_repeat']} gaveup {v3['callers_gave_up']} | {rec['exit']:<19s} | "
            f"w {w:.2f} | ETA {el / ep * (args.episodes - ep) / 60:5.1f} min")

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
    p = argparse.ArgumentParser(description="PPO Engagement V3 training")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--shift", type=float, default=700.0)
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
    p.add_argument("--reset-omega-std", type=float, default=0.0, help="with --resume: set the ω action std to this")
    p.add_argument("--v2-ckpt", type=str, default=str(V2_CKPT))
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--out", type=str, default="experiments")
    p.add_argument("--smoke", action="store_true", help="3 short episodes, 1 eval")
    args = p.parse_args()
    if args.smoke:
        args.episodes, args.shift, args.steps_per_update = 3, 60.0, 512
        args.eval_every, args.eval_episodes, args.teacher_episodes = 3, 2, 2
        args.out = str(Path(args.out) / "smoke")
    train(args)


if __name__ == "__main__":
    main()
