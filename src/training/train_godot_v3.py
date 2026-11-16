"""Train policy Engagement V3 NGAY TRONG GODOT (không còn khoảng cách env Python ↔ Godot).

Godot là môi trường thật: YOLO thật trên ảnh render, STT (PhoWhisper) + PhoBERT thật trên clip giọng thật, NPC thật,
va chạm vật lý thật, hội thoại / luật hộ tống y như lúc chạy demo. Godot chạy lockstep (SimClock, `--train-env`):
mỗi bước policy 0.2 s giờ sim, đứng chờ server trả lời ➔ nhanh hơn thời gian thực nhưng không lệch giờ.

    sim_vision/python/.venv/bin/python -m src.training.train_godot_v3 --envs 3 --eval-envs 1 --hours 6

Tiến trình:
- run_simulation.py --no-godot: YOLO (8765), STT + PhoBERT (8766), TTS (8767).
- Server này (mặc định 8770): /train = ca train (hành động lấy mẫu, gom rollout), /eval = ca đánh giá (tất định).
- N bản Godot `--train-env`, mỗi bản chạy ca này tới ca khác (nạp lại scene, seed mới); mỗi kết nối = một ca.
Quan sát dựng bằng đúng rl_policy_server.PolicyRunner (như lúc chạy thật); thưởng: src/training/godot_reward.py.
Checkpoint chọn theo điểm ca /eval trong Godot (godot_reward.episode_score), không theo env Python.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "sim_vision" / "python"
GODOT_DIR = REPO_ROOT / "sim_vision" / "godot"
for p in (REPO_ROOT, PYTHON_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from websockets.asyncio.server import serve  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402

from rl_policy_server import PolicyRunner, load_agent  # noqa: E402
from src.training.godot_reward import GodotReward, episode_score  # noqa: E402
from src.training.hospital_env_pure_vision_v2 import GAMMA_V2  # noqa: E402
from src.training.train_engage_v3 import git_commit, policy_mean  # noqa: E402

CKPT_PREFIX = "PPOHospital-EngageV3-Godot"
YOLO_PORT0 = 8781
DEFAULT_INIT = REPO_ROOT / "experiments" / "checkpoints" / "PPOHospital-EngageV3_best.pt"


class Fragment:
    """Các bước của một ca từ lần cập nhật PPO trước; `pending` = bước đã hành động, chưa biết thưởng."""

    def __init__(self) -> None:
        self.O: list[np.ndarray] = []
        self.A: list[np.ndarray] = []
        self.LP: list[float] = []
        self.V: list[float] = []
        self.R: list[float] = []
        self.pending: tuple | None = None

    def commit(self, r: float) -> None:
        if self.pending is None:
            return
        o, a, lp, v = self.pending
        self.O.append(o); self.A.append(a); self.LP.append(lp); self.V.append(v); self.R.append(r)
        self.pending = None

    def cut(self, bootstrap: float, lam: float) -> dict | None:
        """Tính GAE cho phần đã có thưởng, trả batch và làm rỗng (giữ `pending`)."""
        n = len(self.R)
        if n == 0:
            return None
        adv = np.zeros(n, dtype=np.float32)
        gae, next_v = 0.0, bootstrap
        for t in reversed(range(n)):
            delta = self.R[t] + GAMMA_V2 * next_v - self.V[t]
            gae = delta + GAMMA_V2 * lam * gae
            adv[t], next_v = gae, self.V[t]
        out = {"obs": np.asarray(self.O, dtype=np.float32), "act": np.asarray(self.A, dtype=np.float32),
               "logp": np.asarray(self.LP, dtype=np.float32), "adv": adv,
               "ret": adv + np.asarray(self.V, dtype=np.float32)}
        self.O, self.A, self.LP, self.V, self.R = [], [], [], [], []
        return out


class Episode:
    def __init__(self, agent, kind: str, peer: str) -> None:
        self.runner = PolicyRunner(agent, "godot")
        self.reward = GodotReward()
        self.frag = Fragment()
        self.kind = kind
        self.peer = peer
        self.ep_id = None
        self.steps = 0
        self.t_sim = 0.0
        self.t0 = time.time()
        self.finished = False


class GodotTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        torch.manual_seed(args.seed)
        self.agent = load_agent(Path(args.init))
        self.agent.train()
        with torch.no_grad():
            ls = self.agent.actor_logstd
            ls[0, 0] = max(float(ls[0, 0]), math.log(args.init_std_v))
            ls[0, 1] = max(float(ls[0, 1]), math.log(args.init_std))
            if ls.shape[1] > 2:
                ls[0, 2] = max(float(ls[0, 2]), math.log(args.init_std))
        self.opt = torch.optim.AdamW(self.agent.parameters(), lr=args.lr, eps=1e-5, weight_decay=0.0)
        self.episodes: dict[int, Episode] = {}
        self.done_batches: list[dict] = []
        self.updates = 0
        self.total_steps = 0
        self.train_summaries: list[dict] = []
        self.eval_summaries: list[dict] = []
        self.best_eval = -float("inf")
        self.t_start = time.time()
        self.stop = asyncio.Event()

        tag = f"eval_{Path(args.init).stem}" if args.eval_only else CKPT_PREFIX
        self.run_dir = Path(args.out) / "runs" / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{tag}"
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.log_f = (self.run_dir / "train.log").open("a", encoding="utf-8")
        self.metrics_f = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")

    # ------------------------------------------------------------------ utils
    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def metric(self, row: dict) -> None:
        self.metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.metrics_f.flush()

    def save(self, name: str, extra: dict) -> None:
        torch.save({"episode": len(self.train_summaries), "updates": self.updates, "steps": self.total_steps,
                    "model_state_dict": self.agent.state_dict(), "optimizer_state_dict": self.opt.state_dict(),
                    "obs_dim": self.agent.obs_dim, "act_dim": self.agent.act_dim, "git_commit": git_commit(),
                    "init": str(self.args.init), **extra}, self.run_dir / "checkpoints" / name)

    def ready_steps(self) -> int:
        return sum(len(e.frag.R) for e in self.episodes.values()) + sum(len(b["adv"]) for b in self.done_batches)

    # --------------------------------------------------------------- stepping
    @torch.no_grad()
    def on_msg(self, ep: Episode, msg: dict) -> dict:
        obs = ep.runner.observe(msg)
        train = msg.get("train") or {}
        if ep.ep_id is None:
            ep.ep_id = train.get("ep")
        ep.t_sim = float(train.get("t", ep.t_sim))
        r = ep.reward.step(train, msg.get("depth")) if train else 0.0
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if ep.kind == "eval":
            ep.steps += 1
            mean = policy_mean(self.agent, obs_t)[0]
            if train.get("done"):
                self.finish(ep, "done")
            return ep.runner.respond(msg, mean.numpy().tolist())
        ep.frag.commit(r)
        value = float(self.agent.critic(obs_t))
        if train.get("done"):
            b = ep.frag.cut(value, self.args.gae_lambda)
            if b is not None:
                self.done_batches.append(b)
            self.finish(ep, "done")
            return ep.runner.respond(msg, [-1.0, 0.0, 0.0, 0.0])
        if self.ready_steps() >= self.args.rollout_steps:
            self.update(cur=ep, cur_value=value)
            value = float(self.agent.critic(obs_t))
        mean = policy_mean(self.agent, obs_t)
        dist = Normal(mean, self.agent.actor_logstd.expand_as(mean).exp())
        a = dist.sample()
        lp = float(dist.log_prob(a).sum(-1))
        a_np = a[0].numpy().astype(np.float32)
        ep.frag.pending = (obs.astype(np.float32), a_np, lp, value)
        ep.steps += 1
        self.total_steps += 1
        return ep.runner.respond(msg, np.clip(a_np, -1.0, 1.0).tolist())

    def finish(self, ep: Episode, why: str) -> None:
        if ep.finished:
            return
        ep.finished = True
        s = ep.reward.summary()
        s.update({"kind": ep.kind, "ep": ep.ep_id, "peer": ep.peer, "steps": ep.steps, "t_sim": round(ep.t_sim, 1),
                  "wall_s": round(time.time() - ep.t0, 1), "why": why, "score": episode_score(s), "updates": self.updates})
        if ep.t_sim < 0.5 * self.args.shift_s:
            s["short"] = True                         # ca bị cắt (Godot thoát / mất kết nối): không tính vào điểm
        (self.eval_summaries if ep.kind == "eval" else self.train_summaries).append(s)
        self.metric({"episode": s})
        self.log(f"{ep.kind:5s} ca {ep.ep_id} ({ep.peer}) {why}: {s['t_sim']:.0f}s sim / {s['wall_s']:.0f}s thật | "
                 f"giúp {s['helped']:.0f} (dẫn tới {s.get('delivered', 0):.0f}) nhận việc {s.get('task_started', 0):.0f} "
                 f"thấy {s.get('spotted', 0):.0f} | hỏi thừa {s.get('ask_wasted', 0):.0f} quay lại hỏi {s.get('reengaged', 0):.0f} "
                 f"bỏ dở {s.get('escort_given_up', 0):.0f} kẹt {s.get('stagnation', 0):.0f} va {s.get('collisions', 0):.0f} "
                 f"đẩy {s.get('push_steps', 0) * 0.2:.0f}s | R {s['return']:+.1f} điểm {s['score']:+.1f}")
        if ep.kind == "eval" and not s.get("short"):
            if self.args.eval_only:
                if len([e for e in self.eval_summaries if not e.get("short")]) >= self.args.eval_episodes:
                    self.stop.set()
            else:
                self.check_best()

    def check_best(self) -> None:
        full = [e for e in self.eval_summaries if not e.get("short")]
        recent = [e for e in full if e["updates"] >= self.updates - self.args.eval_window_updates][-self.args.eval_window:]
        if len(recent) < min(self.args.eval_window, 2):
            return
        sc = float(np.mean([e["score"] for e in recent]))
        if sc > self.best_eval:
            self.best_eval = sc
            self.save(f"{CKPT_PREFIX}_best.pt", {"eval_score": sc, "eval": recent})
            self.log(f"  <- best mới: điểm eval Godot {sc:+.1f} (trung bình {len(recent)} ca gần nhất)")

    # ------------------------------------------------------------------- PPO
    def update(self, cur: Episode | None = None, cur_value: float = 0.0) -> None:
        a = self.args
        batches = list(self.done_batches)
        self.done_batches = []
        for ep in self.episodes.values():
            if ep.kind != "train":
                continue
            boot = cur_value if ep is cur else (ep.frag.pending[3] if ep.frag.pending is not None else 0.0)
            b = ep.frag.cut(boot, a.gae_lambda)
            if b is not None:
                batches.append(b)
        if not batches:
            return
        b = {k: torch.as_tensor(np.concatenate([x[k] for x in batches])) for k in ("obs", "act", "logp", "adv", "ret")}
        b["adv"] = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        n = b["obs"].shape[0]
        vf_only = self.updates < a.vf_warmup
        stats = collections.defaultdict(list)
        t0 = time.time()
        with torch.enable_grad():
            for epoch in range(a.ppo_epochs * (3 if vf_only else 1)):
                if stats["kl"] and np.mean(stats["kl"][-max(1, n // a.batch_size):]) > a.target_kl:
                    break
                perm = torch.randperm(n)
                for s in range(0, n, a.batch_size):
                    idx = perm[s:s + a.batch_size]
                    mean = policy_mean(self.agent, b["obs"][idx])
                    dist = Normal(mean, self.agent.actor_logstd.expand_as(mean).exp())
                    ratio = torch.exp(dist.log_prob(b["act"][idx]).sum(-1) - b["logp"][idx])
                    adv = b["adv"][idx]
                    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
                    v_loss = 0.5 * ((self.agent.critic(b["obs"][idx]).squeeze(-1) - b["ret"][idx]) ** 2).mean()
                    ent = dist.entropy().sum(-1).mean()
                    loss = a.vf_coef * v_loss if vf_only else pg + a.vf_coef * v_loss - a.ent_coef * ent
                    if not torch.isfinite(loss):
                        continue
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                    self.opt.step()
                    with torch.no_grad():
                        self.agent.actor_logstd.clamp_(-2.5, 0.0)
                    stats["pg"].append(pg.item()); stats["v"].append(v_loss.item())
                    if not vf_only:
                        stats["kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
        self.updates += 1
        recent = [e for e in self.train_summaries[-8:] if not e.get("short")]
        el = (time.time() - self.t_start) / 3600
        self.log(f"UPDATE {self.updates} {'(chỉ critic) ' if vf_only else ''}| n={n} bước tổng {self.total_steps} | "
                 f"pg {np.mean(stats['pg']):+.3f} v {np.mean(stats['v']):.3f} kl {np.mean(stats['kl']) if stats['kl'] else 0:.4f} "
                 f"std {self.agent.actor_logstd.exp().detach().numpy().round(2).tolist()[0]} | "
                 f"ca train gần đây: điểm {np.mean([e['score'] for e in recent]) if recent else float('nan'):+.1f} "
                 f"giúp {np.mean([e['helped'] for e in recent]) if recent else float('nan'):.1f} | {time.time() - t0:.1f}s | {el:.2f} h")
        self.metric({"update": self.updates, "n": n, "steps": self.total_steps,
                     **{k: float(np.mean(v)) for k, v in stats.items()}})
        self.save(f"{CKPT_PREFIX}_last.pt", {})
        if self.updates % a.save_every == 0:
            self.save(f"{CKPT_PREFIX}_u{self.updates:04d}.pt", {})
        if (a.max_updates and self.updates >= a.max_updates) or el >= a.hours:
            self.stop.set()

    # ------------------------------------------------------------- serving
    async def handle(self, ws) -> None:
        path = getattr(getattr(ws, "request", None), "path", "/train") or "/train"
        kind = "eval" if path.startswith("/eval") else "train"
        ep = Episode(self.agent, kind, f"{ws.remote_address[1]}")
        self.episodes[id(ep)] = ep
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict) or msg.get("type") != "rl_state":
                    continue
                await ws.send(json.dumps(self.on_msg(ep, msg)))
        except ConnectionClosed:
            pass
        finally:
            self.episodes.pop(id(ep), None)
            if ep.steps > 0 and not ep.finished:
                if ep.kind == "train":
                    b = ep.frag.cut(ep.frag.pending[3] if ep.frag.pending is not None else 0.0, self.args.gae_lambda)
                    if b is not None:
                        self.done_batches.append(b)
                self.finish(ep, "ngắt")


# ---------------------------------------------------------------- processes
def find_godot() -> str:
    for c in (os.environ.get("GODOT"), str(Path.home() / ".local" / "bin" / "godot"), "/opt/homebrew/bin/godot"):
        if c and Path(c).is_file():
            return c
    return "godot"


def wait_port(port: int, timeout: float = 240.0) -> bool:
    from websockets.sync.client import connect
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with connect(f"ws://127.0.0.1:{port}", open_timeout=1.0, close_timeout=1.0):
                return True
        except Exception:
            time.sleep(1.0)
    return False


def yolo_port(args: argparse.Namespace, i: int, kind: str) -> int:
    """Mỗi bản Godot một YOLO server (một server phục vụ tuần tự ~26 ms / ảnh ➔ là cổ chai khi chạy song song)."""
    return YOLO_PORT0 + i + (args.envs if kind == "eval" else 0)


def launch_yolo(port: int, log_dir: Path) -> subprocess.Popen:
    py = PYTHON_DIR / ".venv" / "bin" / "python"
    model = REPO_ROOT / "models" / "yolo26n_hospital_robot_v6_best" / "yolo26n_hospital_robot_best_map92.6.pt"
    cmd = [str(py if py.is_file() else sys.executable), "yolo_server.py", "--port", str(port)]
    if model.is_file():
        cmd += ["--model", str(model)]
    return subprocess.Popen(cmd, cwd=str(PYTHON_DIR), env={**os.environ, "PYTHONUNBUFFERED": "1"},
                            stdout=(log_dir / f"yolo_{port}.log").open("w"), stderr=subprocess.STDOUT)


def launch_godot(args: argparse.Namespace, i: int, kind: str, log_dir: Path) -> subprocess.Popen:
    seed = (1000 if kind == "eval" else 0) + args.seed * 10 + i
    cmd = [find_godot(), "--path", str(GODOT_DIR), "--fixed-fps", str(args.fps), "--disable-vsync",
           "--resolution", "320x240", "--", "--controller=rl", "--train-env", f"--env-url=ws://127.0.0.1:{args.port}/{kind}",
           f"--seed={seed}", f"--shift-s={args.shift_s}", f"--capture-hz={args.capture_hz}", f"--lag-prob={args.lag_prob}",
           f"--yolo-url=ws://127.0.0.1:{yolo_port(args, i, kind)}"]
    f = (log_dir / f"godot_{kind}{i}.log").open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(GODOT_DIR))


async def amain(args: argparse.Namespace) -> None:
    tr = GodotTrainer(args)
    log_dir = tr.run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    procs: list[subprocess.Popen] = []      # YOLO servers
    gprocs: list[subprocess.Popen] = []     # Godot
    servers = None
    try:
        if not args.no_servers:
            py = PYTHON_DIR / ".venv" / "bin" / "python"
            servers = subprocess.Popen([str(py if py.is_file() else sys.executable), str(REPO_ROOT / "run_simulation.py"),
                                        "--no-godot", "--no-yolo", "--controller", "fsm", "--nlu", args.nlu, "--log-dir", str(log_dir)],
                                       stdout=(log_dir / "servers.log").open("w"), stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
            for kind, n in (("train", args.envs), ("eval", args.eval_envs)):
                for i in range(n):
                    procs.append(launch_yolo(yolo_port(args, i, kind), log_dir))
            for port in [8766] + [yolo_port(args, i, "train") for i in range(args.envs)] \
                    + [yolo_port(args, i, "eval") for i in range(args.eval_envs)]:
                if not wait_port(port):
                    raise RuntimeError(f"server cổng {port} không lên (xem {log_dir})")
            tr.log("YOLO + STT/PhoBERT (+ TTS nếu có) đã sẵn sàng")
        tr.log(f"Train trong Godot | init {args.init} | {args.envs} ca train + {args.eval_envs} ca eval song song, "
               f"ca {args.shift_s:.0f}s sim, {args.fps} fps, camera {args.capture_hz} Hz | rollout {args.rollout_steps} bước, "
               f"lr {args.lr}, std v {args.init_std_v} ω {args.init_std} | out {tr.run_dir}")
        tr.metric({"args": vars(args)})
        async with serve(tr.handle, "127.0.0.1", args.port, max_size=2**24, ping_interval=None):
            specs = [("train", i) for i in range(args.envs)] + [("eval", i) for i in range(args.eval_envs)]
            for kind, i in specs:
                gprocs.append(launch_godot(args, i, kind, log_dir))
                await asyncio.sleep(3.0)
            while not tr.stop.is_set():
                for j, (kind, i) in enumerate(specs):
                    if gprocs[j].poll() is not None:
                        tr.log(f"Godot {kind}{i} thoát (mã {gprocs[j].returncode}) ➔ chạy lại")
                        gprocs[j] = launch_godot(args, i, kind, log_dir)
                try:
                    await asyncio.wait_for(tr.stop.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
        tr.log(f"Dừng sau {tr.updates} lần cập nhật, {tr.total_steps} bước. Checkpoint: {tr.run_dir / 'checkpoints'}")
    finally:
        for p in gprocs + procs:
            if p.poll() is None:
                p.terminate()
        if servers is not None and servers.poll() is None:
            servers.terminate()
            try:
                servers.wait(timeout=10)
            except subprocess.TimeoutExpired:
                servers.kill()


def main() -> None:
    p = argparse.ArgumentParser(description="PPO Engagement V3 trong Godot (lockstep, YOLO / STT / NLU thật)")
    p.add_argument("--init", default=str(DEFAULT_INIT))
    p.add_argument("--envs", type=int, default=3)
    p.add_argument("--eval-envs", type=int, default=1)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--shift-s", type=float, default=480.0, help="độ dài một ca (giờ sim)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--capture-hz", type=float, default=10.0, help="camera trước gửi YOLO (chạy demo nên dùng cùng giá trị)")
    p.add_argument("--lag-prob", type=float, default=0.7)
    p.add_argument("--nlu", default="phobert")
    p.add_argument("--no-servers", action="store_true", help="YOLO / STT / TTS đã chạy sẵn")
    p.add_argument("--rollout-steps", type=int, default=4096)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--vf-warmup", type=int, default=2, help="số lần cập nhật đầu chỉ học critic (thang thưởng mới)")
    p.add_argument("--init-std", type=float, default=0.25)
    p.add_argument("--init-std-v", type=float, default=0.12)
    p.add_argument("--eval-window", type=int, default=3)
    p.add_argument("--eval-window-updates", type=int, default=6)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--max-updates", type=int, default=0)
    p.add_argument("--eval-only", action="store_true", help="chỉ chạy ca /eval của --init (đối chứng cùng seed), không train")
    p.add_argument("--eval-episodes", type=int, default=6)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=str(REPO_ROOT / "experiments"))
    args = p.parse_args()
    if args.eval_only:
        args.envs, args.hours = 0, 1e9
    torch.set_num_threads(2)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
