"""PPO for Engagement V3 with skill drills, a skill teacher and parallel rollouts (round 9 of docs/RL_V3_DESIGN.md).

Why another trainer: after 8 rounds with train_engage_v3.py the policy still
- stopped turning to a speaker at its side / behind it half way (walk-up enrolment ~55%), and
- could not get out of a room past the person it had just delivered (15-50%), and pushed into them.
Both are rare in a 700 s shift, so PPO hardly ever sees them. Here:

- episodes are a mix of full shifts and two short drills: "walk-up" (someone next to the robot starts talking) and
  "exit" (get back to the corridor past the delivered person, HospitalEngageV3Env.setup_exit_drill);
- in *skill states* (talking, stalled / getting clear, drills) the rule teacher of the env supplies an imitation
  target (DAgger: the teacher labels the states the policy itself visits) next to the PPO loss; the weight decays to a
  small floor, so PPO can do better than the rules but does not forget them. Ordinary driving is left to PPO alone
  (there the policy is better than the rules);
- rollouts and evaluation run in worker processes.

Usage:
  python src/training/train_engage_v3_skills.py --resume experiments/checkpoints/PPOHospital-EngageV3_best.pt
  python src/training/train_engage_v3_skills.py --smoke
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import multiprocessing as mp
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
from src.training.hospital_env import TaskPhase
from src.training.hospital_env_engage_v3 import ACT_DIM_V3, OBS_DIM_V3, OBS_STALL, HospitalEngageV3Env
from src.training.hospital_env_pure_vision_v2 import GAMMA_V2
from src.training.train_engage_v3 import CKPT_PREFIX, EVAL_SEED_BASE, git_commit, load_widening, policy_mean, warm_start_from_v2, V2_CKPT

KINDS = ("shift", "walkup", "exit")
PROBE_SEED = 5_000


def make_env(seed: int, seconds: float, mode: str = "full", dr: bool = True) -> HospitalEngageV3Env:
    return HospitalEngageV3Env(curriculum_stage=1, shift_duration_s=seconds, dt=0.2, seed=seed, env_mode=mode,
                               stagnation_timeout_s=90.0, stagnation_penalty=0.0, escort_only=True,
                               domain_randomization=dr)


def setup_walkup_probe(i: int) -> HospitalEngageV3Env:
    """Fixed test: one person needing help 0.8-1.9 m from the robot at any bearing, nobody else near."""
    rng = np.random.default_rng(PROBE_SEED + i)
    env = make_env(PROBE_SEED + i, 120.0, mode="solo", dr=True)
    env.reset(seed=PROBE_SEED + i)
    p = env.patients[0]
    p.update(needs_help=True, caller=False)
    for q in env.patients[1:]:
        q["x"] += 30.0
    b, d = float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(0.8, 1.9))
    env.robot_yaw = float(rng.uniform(-math.pi, math.pi))
    env.robot_x = p["x"] + math.sin(env.robot_yaw + b) * d
    env.robot_z = p["z"] + math.cos(env.robot_yaw + b) * d
    p["face_yaw"] = float(rng.uniform(-math.pi, math.pi))
    env.mem.reset()
    env.last_obs_v2 = env._get_obs()
    return env


def skill_state(env: HospitalEngageV3Env, obs: np.ndarray, kind: str) -> bool:
    """States in which the rule teacher is a good target: talking, stalled / getting clear, and the drills."""
    return kind != "shift" or env.dialogue is not None or obs[OBS_STALL] != 0.0


# --------------------------------------------------------------------------- workers
_AGENT: PPOHospitalAgent | None = None


def _init_worker() -> None:
    global _AGENT
    torch.set_num_threads(1)
    _AGENT = PPOHospitalAgent(obs_dim=OBS_DIM_V3, act_dim=ACT_DIM_V3)
    _AGENT.eval()


def _load(sd: dict[str, np.ndarray]) -> PPOHospitalAgent:
    assert _AGENT is not None
    _AGENT.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
    return _AGENT


@torch.no_grad()
def rollout(task: dict) -> dict:
    """One training episode. task: sd, kind, seed, seconds, teacher_w (prob. of executing the teacher in skill states)."""
    agent = _load(task["sd"])
    kind, seed = task["kind"], task["seed"]
    rng = np.random.default_rng(seed)
    env = make_env(seed, task["seconds"])
    obs = env.reset()
    if kind == "walkup" and not env.setup_walkup_drill():
        kind = "shift"
    if kind == "exit" and not env.setup_exit_drill():
        kind = "shift"
    obs = env.last_obs_v2
    std = agent.actor_logstd.exp()
    O, A, T, LP, R, V, M = [], [], [], [], [], [], []
    done, info = False, {}
    while not done:
        skill = skill_state(env, obs, kind)
        teacher = env.teacher_action()                      # always called: it keeps its own manoeuvre state
        obs_t = torch.as_tensor(obs).unsqueeze(0)
        mean = policy_mean(agent, obs_t)
        dist = Normal(mean, std.expand_as(mean))
        a = dist.sample().squeeze(0).numpy()
        if skill and task["teacher_w"] > 0.0 and rng.random() < task["teacher_w"]:
            a = teacher.copy()
        O.append(obs); A.append(a.astype(np.float32)); T.append(teacher); M.append(1.0 if skill else 0.0)
        LP.append(float(dist.log_prob(torch.as_tensor(a).unsqueeze(0)).sum(-1)))
        V.append(float(agent.critic(obs_t)))
        obs, r, done, info = env.step(np.clip(a, -1.0, 1.0))
        R.append(r)
    last_v = float(agent.critic(torch.as_tensor(obs).unsqueeze(0))) if info.get("truncated") else 0.0
    gamma, lam = GAMMA_V2, task["lam"]
    adv = np.zeros(len(R), dtype=np.float32)
    gae, next_v = 0.0, last_v
    for t in reversed(range(len(R))):
        delta = R[t] + gamma * next_v - V[t]
        gae = delta + gamma * lam * gae
        adv[t], next_v = gae, V[t]
    return {"obs": np.asarray(O, dtype=np.float32), "act": np.asarray(A, dtype=np.float32),
            "tea": np.asarray(T, dtype=np.float32), "logp": np.asarray(LP, dtype=np.float32), "adv": adv,
            "ret": adv + np.asarray(V, dtype=np.float32), "mask": np.asarray(M, dtype=np.float32),
            "kind": kind, "return": float(np.sum(R)), "helped": info.get("patients_helped", 0),
            "collisions": info.get("collisions", 0), "exit": info.get("exit_reason", ""), "v3": info["v3"]}


@torch.no_grad()
def eval_task(task: dict) -> dict:
    """One evaluation episode with the deterministic policy (or the teacher): kind = shift / walkup / exit."""
    agent = _load(task["sd"]) if task.get("sd") is not None else None
    kind, i = task["kind"], task["i"]

    def act(env: HospitalEngageV3Env) -> np.ndarray:
        tea = env.teacher_action()
        if agent is None:
            return tea
        return policy_mean(agent, torch.as_tensor(env.last_obs_v2).unsqueeze(0)).squeeze(0).numpy()

    if kind == "shift":
        seed = EVAL_SEED_BASE + i
        env = make_env(seed, task["seconds"])
        env.rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        done, info = False, {}
        while not done:
            _, _, done, info = env.step(act(env))
        return {"kind": kind, "return": info["cumulative_team_reward"], "helped": info["patients_helped"],
                "collisions": info["collisions"], "escort_speed": info["escort_speed"],
                "all_served": int(info.get("exit_reason") == "ALL_PATIENTS_SERVED"), "v3": info["v3"]}
    if kind == "walkup":
        env = setup_walkup_probe(i)
        p = env.patients[0]
        b0 = abs(env._rel_angle(p["x"], p["z"])[0])
        info = {}
        for _ in range(150):
            _, _, done, info = env.step(act(env))
            if done or env.task_phase == TaskPhase.GUIDE_ESCORT:
                break
        v3 = info["v3"]
        return {"kind": kind, "dialogue": int(v3["dialogues"] >= 1), "enrolled": int(v3["enrolled"] >= 1),
                "side": int(b0 >= math.radians(45.0)), "centre": v3["centre_sum"] / max(1, v3["face_steps"])}
    env = make_env(PROBE_SEED + i, 40.0)
    env.reset(seed=PROBE_SEED + i)
    assert env.setup_exit_drill()
    done, info, k = False, {}, 0
    while not done:
        _, _, done, info = env.step(act(env))
        k += 1
    v3 = info["v3"]
    return {"kind": kind, "ok": int(v3["exit_ok"] >= 1), "time": k * env.dt, "push": v3["push_steps"],
            "collisions": info["collisions"], "excuse": v3["excuse"]}


def summarise(rows: list[dict]) -> dict[str, float]:
    sh = [r for r in rows if r["kind"] == "shift"]
    wu = [r for r in rows if r["kind"] == "walkup"]
    ex = [r for r in rows if r["kind"] == "exit"]
    out: dict[str, float] = {}
    if sh:
        st = collections.Counter()
        for r in sh:
            st.update(r["v3"])
        n, dlg = len(sh), max(1, st["dialogues"])
        out.update({"return": float(np.mean([r["return"] for r in sh])), "helped": float(np.mean([r["helped"] for r in sh])),
                    "collisions": float(np.mean([r["collisions"] for r in sh])),
                    "escort_speed": float(np.mean([r["escort_speed"] for r in sh])),
                    "all_served_rate": float(np.mean([r["all_served"] for r in sh])),
                    "enroll_rate": st["enrolled"] / dlg, "face_frac": st["face_steps"] / max(1, st["dialogue_steps"]),
                    "face_centred": st["centre_sum"] / max(1, st["face_steps"]),
                    "asks_per_ep": st["asks"] / n, "wasted_asks_per_ep": (st["asks_empty"] + st["asks_repeat"]) / n,
                    "callers_gave_up_per_ep": st["callers_gave_up"] / n, "push_s_per_ep": st["push_steps"] * 0.2 / n})
    if wu:
        dl = [r for r in wu if r["dialogue"]]
        side = [r for r in dl if r["side"]]
        out.update({"walkup_dialogue": float(np.mean([r["dialogue"] for r in wu])),
                    "walkup_enrolled": float(np.mean([r["enrolled"] for r in dl])) if dl else 0.0,
                    "walkup_enrolled_side": float(np.mean([r["enrolled"] for r in side])) if side else 0.0})
    if ex:
        ok = [r for r in ex if r["ok"]]
        out.update({"exit_ok": float(np.mean([r["ok"] for r in ex])),
                    "exit_time_s": float(np.mean([r["time"] for r in ok])) if ok else 0.0,
                    "exit_push_s": float(np.mean([r["push"] for r in ex])) * 0.2,
                    "exit_collisions": float(np.mean([r["collisions"] for r in ex]))})
    return out


def selection_score(ev: dict[str, float]) -> float:
    """Checkpoint choice: the shift return plus the two skills that the return alone hardly measures."""
    return (ev.get("return", 0.0) + 40.0 * ev.get("walkup_enrolled", 0.0) + 40.0 * ev.get("exit_ok", 0.0)
            - 5.0 * ev.get("exit_push_s", 0.0) - 3.0 * ev.get("wasted_asks_per_ep", 0.0))


def evaluate(pool, agent: PPOHospitalAgent | None, n_shift: int, n_probe: int, shift_s: float) -> dict[str, float]:
    sd = {k: v.numpy() for k, v in agent.state_dict().items()} if agent is not None else None
    tasks = [{"sd": sd, "kind": "shift", "i": i, "seconds": shift_s} for i in range(n_shift)]
    tasks += [{"sd": sd, "kind": k, "i": i} for k in ("walkup", "exit") for i in range(n_probe)]
    return summarise(pool.map(eval_task, tasks, chunksize=1))


# --------------------------------------------------------------------------- training
def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    rng = np.random.default_rng(args.seed)
    agent = PPOHospitalAgent(obs_dim=OBS_DIM_V3, act_dim=ACT_DIM_V3)
    if args.resume:
        load_widening(agent, Path(args.resume))
    else:
        warm_start_from_v2(agent, Path(args.v2_ckpt))
    with torch.no_grad():                                    # re-open exploration (stds had collapsed)
        agent.actor_logstd[0, 1] = max(float(agent.actor_logstd[0, 1]), math.log(args.init_std))
        agent.actor_logstd[0, 2] = max(float(agent.actor_logstd[0, 2]), math.log(args.init_std))
    opt = torch.optim.AdamW(agent.parameters(), lr=args.lr, eps=1e-5, weight_decay=0.0)
    iters = max(1, args.episodes // args.batch_episodes)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=args.lr * 0.1)

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

    fr = np.array([args.frac_shift, args.frac_walkup, args.frac_exit], dtype=np.float64)
    fr /= fr.sum()
    secs = {"shift": args.shift, "walkup": args.drill_s, "exit": args.exit_s}
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker)
    try:
        log(f"PPO Engagement V3 skills | episodes={args.episodes} ({iters} iters x {args.batch_episodes}) mix shift/walkup/exit="
            f"{fr.round(2).tolist()} shift={args.shift}s lr={args.lr} bc {args.bc_start}->{args.bc_end} "
            f"teacher_w {args.teacher_start}->0 over {args.teacher_frac:.0%} | obs={OBS_DIM_V3} | "
            f"init={args.resume or 'V2 warm start'} | out={run_dir}")
        base = evaluate(pool, None, args.eval_episodes, args.probe_n, args.shift)
        log(f"[BASELINE rule teacher] {json.dumps({k: round(v, 3) for k, v in base.items()})}")
        init = evaluate(pool, agent, args.eval_episodes, args.probe_n, args.shift)
        log(f"[INIT policy] {json.dumps({k: round(v, 3) for k, v in init.items()})}  score {selection_score(init):.1f}")
        metrics_f.write(json.dumps({"episode": 0, "baseline": base, "eval": init}) + "\n"); metrics_f.flush()

        best_score, t0, ep = -float("inf"), time.time(), 0
        for it in range(1, iters + 1):
            prog = (it - 1) / max(1, iters - 1)
            teacher_w = args.teacher_start * max(0.0, 1.0 - prog / max(1e-6, args.teacher_frac))
            bc_coef = args.bc_end + (args.bc_start - args.bc_end) * max(0.0, 1.0 - prog / 0.6)
            sd = {k: v.numpy() for k, v in agent.state_dict().items()}
            tasks = [{"sd": sd, "kind": str(rng.choice(KINDS, p=fr)), "seed": args.seed + (ep + j) * 17,
                      "teacher_w": teacher_w, "lam": args.gae_lambda} for j in range(args.batch_episodes)]
            for tk in tasks:
                tk["seconds"] = secs[tk["kind"]]
            eps = pool.map(rollout, tasks, chunksize=1)
            ep += len(eps)

            b = {k: torch.as_tensor(np.concatenate([e[k] for e in eps])) for k in ("obs", "act", "tea", "logp", "adv", "ret", "mask")}
            b["adv"] = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
            n = b["obs"].shape[0]
            stats = collections.defaultdict(list)
            for _ in range(args.ppo_epochs):
                perm = torch.randperm(n)
                for s in range(0, n, args.batch_size):
                    idx = perm[s:s + args.batch_size]
                    mean = policy_mean(agent, b["obs"][idx])
                    dist = Normal(mean, agent.actor_logstd.expand_as(mean).exp())
                    ratio = torch.exp(dist.log_prob(b["act"][idx]).sum(-1) - b["logp"][idx])
                    pg = -torch.min(ratio * b["adv"][idx], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b["adv"][idx]).mean()
                    v_loss = 0.5 * ((agent.critic(b["obs"][idx]).squeeze(-1) - b["ret"][idx]) ** 2).mean()
                    ent = dist.entropy().sum(-1).mean()
                    bc = (b["mask"][idx] * ((mean - b["tea"][idx]) ** 2).sum(-1)).sum() / b["mask"][idx].sum().clamp(min=1.0)
                    loss = pg + args.vf_coef * v_loss - args.ent_coef * ent + bc_coef * bc
                    if not torch.isfinite(loss):
                        continue
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    opt.step()
                    with torch.no_grad():
                        agent.actor_logstd.clamp_(-2.5, 0.0)
                    stats["pg"].append(pg.item()); stats["v"].append(v_loss.item()); stats["bc"].append(bc.item())
                    stats["kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
            sched.step()

            by = collections.defaultdict(list)
            for e in eps:
                by[e["kind"]].append(e)
            parts = []
            if by["shift"]:
                parts.append(f"shift R {np.mean([e['return'] for e in by['shift']]):+6.1f} helped {np.mean([e['helped'] for e in by['shift']]):.1f} col {np.mean([e['collisions'] for e in by['shift']]):.1f}")
            if by["walkup"]:
                dl = [e for e in by["walkup"] if e["v3"]["dialogues"] > 0]
                parts.append(f"walkup enr {np.mean([e['v3']['enrolled'] > 0 for e in dl]) if dl else 0:.2f}")
            if by["exit"]:
                parts.append(f"exit ok {np.mean([e['v3']['exit_ok'] for e in by['exit']]):.2f} push {np.mean([e['v3']['push_steps'] for e in by['exit']]) * 0.2:.1f}s")
            el = time.time() - t0
            log(f"It {it:4d}/{iters} ep {ep:5d} | {' | '.join(parts)} | n={n} pg {np.mean(stats['pg']):+.3f} v {np.mean(stats['v']):.3f} "
                f"bc {np.mean(stats['bc']):.3f} kl {np.mean(stats['kl']):.4f} std {agent.actor_logstd.exp().detach().numpy().round(2).tolist()[0]} "
                f"| w {teacher_w:.2f} bc_c {bc_coef:.2f} | ETA {el / it * (iters - it) / 60:5.1f} min")
            metrics_f.write(json.dumps({"iter": it, "episode": ep, "n": n, "teacher_w": teacher_w, "bc_coef": bc_coef,
                                        "train": {k: [{"return": e["return"], "v3": e["v3"]} for e in v] for k, v in by.items()}}) + "\n")
            metrics_f.flush()

            if it % args.eval_every == 0 or it == iters:
                ev = evaluate(pool, agent, args.eval_episodes, args.probe_n, args.shift)
                score = selection_score(ev)
                tag = ""
                if score > best_score:
                    best_score, tag = score, "  <- new best"
                    save(f"{CKPT_PREFIX}_best.pt", ep, {"eval": ev})
                log(f"[EVAL @ ep {ep}] score {score:.1f} {json.dumps({k: round(v, 3) for k, v in ev.items()})}{tag}")
                metrics_f.write(json.dumps({"episode": ep, "eval": ev, "score": score}) + "\n"); metrics_f.flush()
        save(f"{CKPT_PREFIX}_last.pt", ep, {})
        log(f"Done. Best score {best_score:.2f}. Checkpoints in {run_dir / 'checkpoints'}")
    finally:
        pool.close()
        pool.join()
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="PPO Engagement V3 with skill drills")
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--batch-episodes", type=int, default=40)
    p.add_argument("--shift", type=float, default=700.0)
    p.add_argument("--drill-s", type=float, default=30.0)
    p.add_argument("--exit-s", type=float, default=40.0)
    p.add_argument("--frac-shift", type=float, default=0.2)
    p.add_argument("--frac-walkup", type=float, default=0.4)
    p.add_argument("--frac-exit", type=float, default=0.4)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.002)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--bc-start", type=float, default=2.0, help="imitation weight in skill states at the start")
    p.add_argument("--bc-end", type=float, default=0.3, help="... and its floor (reached at 60% of training)")
    p.add_argument("--teacher-start", type=float, default=0.5, help="prob. the teacher acts in skill states at the start")
    p.add_argument("--teacher-frac", type=float, default=0.4, help="share of training over which that prob. goes to 0")
    p.add_argument("--init-std", type=float, default=0.3)
    p.add_argument("--eval-every", type=int, default=10, help="iterations")
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--probe-n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--v2-ckpt", type=str, default=str(V2_CKPT))
    p.add_argument("--out", type=str, default="experiments")
    p.add_argument("--smoke", action="store_true", help="2 tiny iterations")
    args = p.parse_args()
    if args.smoke:
        args.episodes, args.batch_episodes, args.shift, args.eval_every = 8, 4, 40.0, 2
        args.eval_episodes, args.probe_n, args.workers = 2, 4, 2
        args.out = str(Path(args.out) / "smoke")
    train(args)


if __name__ == "__main__":
    main()
