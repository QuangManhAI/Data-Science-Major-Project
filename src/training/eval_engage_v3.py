"""Compare the rule-based interaction (teacher = current Godot rules) with the learned V3 policy on the same
held-out seeds (docs/RL_V3_DESIGN.md). Usage:
  python src/training/eval_engage_v3.py --ckpt experiments/checkpoints/PPOHospital-EngageV3_best.pt [--episodes 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from src.models.ppo_agent import PPOHospitalAgent
from src.training.hospital_env_engage_v3 import ACT_DIM_V3, OBS_DIM_V3
from src.training.train_engage_v3 import evaluate

ROWS = [("helped", "Người được dẫn tới nơi / ca (6)"), ("all_served_rate", "Ca giúp đủ 6 người"),
        ("enroll_rate", "Hội thoại có ghi nhớ được mặt"), ("face_frac", "Thời gian thấy mặt khi nói chuyện"),
        ("walkup_enroll_rate", "Người tự tới nói (có thể bên cạnh / sau lưng): ghi nhớ được"),
        ("asks_per_ep", "Số lần hỏi / ca"), ("wasted_asks_per_ep", "Hỏi thừa / ca (không ai, hỏi lại)"),
        ("callers_gave_up_per_ep", "Người gọi bị bỏ qua tới mức bỏ cuộc / ca"),
        ("aborted_per_ep", "Bỏ đi giữa lúc nói chuyện / ca"), ("collisions", "Va chạm / ca"),
        ("escort_speed", "Tốc độ dẫn (m/s)"), ("return", "Tổng thưởng")]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--shift", type=float, default=700.0)
    p.add_argument("--out", type=Path, default=ROOT_DIR / "experiments" / "results" / "engage_v3_eval.md")
    a = p.parse_args()
    agent = PPOHospitalAgent(obs_dim=OBS_DIM_V3, act_dim=ACT_DIM_V3)
    agent.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    agent.eval()
    rules = evaluate(None, a.episodes, a.shift, teacher=True)
    learned = evaluate(agent, a.episodes, a.shift)
    lines = ["# V3: luật vs học (môi trường train V3, cùng các ca đánh giá)", "",
             f"{a.episodes} ca × {a.shift:.0f} s, người ngẫu nhiên (cần giúp / không, tự gọi / chờ hỏi, hướng mặt). "
             f"Checkpoint: `{a.ckpt.name}`.", "", "| Chỉ số | Luật (Godot hiện tại) | V3 học |", "|---|---|---|"]
    for k, name in ROWS:
        fmt = "{:.0%}" if k in ("all_served_rate", "enroll_rate", "face_frac", "walkup_enroll_rate") else "{:.2f}"
        lines.append(f"| {name} | {fmt.format(rules[k])} | {fmt.format(learned[k])} |")
    md = "\n".join(lines) + "\n"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    (a.out.with_suffix(".json")).write_text(json.dumps({"rules": rules, "learned": learned}, indent=1), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
