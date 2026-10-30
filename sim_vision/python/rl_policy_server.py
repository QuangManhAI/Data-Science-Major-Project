"""RL policy server (port 8768): Godot gửi trạng thái thô, server trả lệnh (v, omega) từ PPO.

Quan sát 31-D được dựng bằng chính `HospitalShiftEnv._get_obs` (cùng định tuyến hành lang,
YOLO mô phỏng, chốt SEARCH ➔ APPROACH) để khớp tuyệt đối với lúc train; chỉ 17 tia LiDAR
lấy từ Godot (`--lidar godot`, mặc định) hoặc tính lại từ tường 2D lúc train (`--lidar sim`).

Request : {"type": "rl_state", "id": int, "robot": [x, z, yaw, v, omega], "lidar": [17 x float],
           "patient": [x, z], "phase": 0|1|2|3, "goal": [x, z], "task_id": int}
          task_id đổi (bệnh nhân mới) ➔ bỏ chốt APPROACH, về SEARCH như env lúc train.
Response: {"type": "rl_action", "id": int, "v": float, "omega": float, "a": [a_v, a_omega], "phase": int}

Chạy: python rl_policy_server.py [--checkpoint experiments/checkpoints/PPOHospital-full_strongest.pt] [--port 8768]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.ppo_agent import ACT_DIM, OBS_DIM, PPOHospitalAgent  # noqa: E402
from src.training.hospital_env import (  # noqa: E402
    ESCORT_WALK_SPEED,
    OMEGA_MAX,
    V_MAX,
    HospitalShiftEnv,
    TaskPhase,
    floor_of,
    signed_angle_rad,
)
from src.training.hospital_env_vision import PUBLIC_PATROL_WAYPOINTS  # noqa: E402
from src.training.camera_vision_model import CAMERA_HFOV_RAD  # noqa: E402

log = logging.getLogger("rl_policy_server")

RL_WS_HOST = "127.0.0.1"
RL_WS_PORT = 8768
# Strongest full-mode policy (8 NPCs, grid planner, dock start); .pt files are git-ignored, copy it in when cloning
DEFAULT_CHECKPOINT = "experiments/checkpoints/PPOHospital-full_strongest.pt"


class PolicyRunner:
    """Giữ một env Python làm 'bộ dựng quan sát' cho mỗi kết nối Godot."""

    def __init__(self, agent: PPOHospitalAgent, lidar_source: str, trace: bool = False) -> None:
        self.agent = agent
        self.lidar_source = lidar_source
        self.trace = trace
        self.n_steps = 0
        self.env = HospitalShiftEnv(curriculum_stage=1, env_mode="solo")
        self.env.reset()
        self.task_id: int | None = None
        self.patrol_wp_idx: int = 0
        self.lost_det_frames: int = 0
        self.last_vis_goal: tuple[float, float, float] = (0.05, 0.0, 1.0)

    def act(self, msg: dict) -> dict:
        env = self.env
        x, z, yaw, v, omega = (float(t) for t in msg["robot"])
        env.robot_x, env.robot_z, env.robot_yaw = x, z, yaw
        env.robot_v, env.robot_omega = v, omega
        # Zero-oracle: msg["patient"] is NEVER used during SEARCH or APPROACH.
        env.patient_x, env.patient_z = (float(t) for t in msg["patient"])

        phase = int(msg["phase"])
        task_id = int(msg.get("task_id", 0))
        if task_id != self.task_id:
            self.task_id = task_id
            self.lost_det_frames = 0
            env.task_phase = TaskPhase.SEARCH_PATIENT
        if phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF):
            env.task_phase = TaskPhase(phase)
            final_x, final_z = (float(t) for t in msg["goal"])
            env.set_nav_target(final_x, final_z)  # sub-goal + path length on the same grid planner as training
        else:
            if env.task_phase not in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
                env.task_phase = TaskPhase.SEARCH_PATIENT
            final_x, final_z = env.patient_x, env.patient_z

        obs = env._get_obs()
        if self.lidar_source == "godot":
            lidar = np.asarray(msg["lidar"], dtype=np.float32)
            if lidar.shape == (17,):
                obs[0:17] = np.clip(lidar, 0.0, 1.0)

        # Zero-Oracle Pure Visual Search & Approach:
        if env.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
            has_det = False
            if "yolo" in msg and len(msg["yolo"]) == 4:
                obs[17:21] = np.asarray(msg["yolo"], dtype=np.float32)
                if obs[17] > 0.5:
                    has_det = True
                    env.task_phase = TaskPhase.APPROACH_PATIENT
                    obs[21:25] = 0.0
                    obs[22] = 1.0  # one-hot APPROACH_PATIENT

            if has_det:
                self.lost_det_frames = 0
                cx_off = float(obs[18])
                h = float(obs[19])
                d_vis = float(np.clip(1.20 / max(h, 0.05), 0.8, 15.0))
                angle_vis = float(-math.atan(cx_off * math.tan(CAMERA_HFOV_RAD / 2.0)))
                obs[25] = float(np.clip(d_vis / 25.0, 0.0, 1.0))
                obs[26] = math.sin(angle_vis)
                obs[27] = math.cos(angle_vis)
                self.last_vis_goal = (obs[25], obs[26], obs[27])
            elif env.task_phase == TaskPhase.APPROACH_PATIENT and self.lost_det_frames < 10:
                # Giữ hướng tiếp cận thị giác trong 10 frame (~1s) để không bị giật quay lại tuần tra khi YOLO nhấp nháy hoặc góc nhìn hẹp
                self.lost_det_frames += 1
                obs[21:25] = 0.0
                obs[22] = 1.0  # one-hot APPROACH_PATIENT
                obs[25], obs[26], obs[27] = self.last_vis_goal
            else:
                self.lost_det_frames = 0
                env.task_phase = TaskPhase.SEARCH_PATIENT
                obs[21:25] = 0.0
                obs[21] = 1.0  # one-hot SEARCH_PATIENT
                fl = floor_of(x)
                waypoints = PUBLIC_PATROL_WAYPOINTS.get(fl, PUBLIC_PATROL_WAYPOINTS[1])
                pw_x, pw_z = waypoints[self.patrol_wp_idx % len(waypoints)]
                if math.hypot(pw_x - x, pw_z - z) <= 1.5:
                    self.patrol_wp_idx = (self.patrol_wp_idx + 1) % len(waypoints)
                    pw_x, pw_z = waypoints[self.patrol_wp_idx % len(waypoints)]
                env.set_nav_target(pw_x, pw_z)
                to_gx = env.goal_x - x
                to_gz = env.goal_z - z
                fwd_x = -math.sin(yaw)
                fwd_z = -math.cos(yaw)
                angle_goal = signed_angle_rad(fwd_x, fwd_z, to_gx, to_gz)
                obs[25] = float(np.clip(env.nav_remaining / 25.0, 0.0, 1.0))
                obs[26] = math.sin(angle_goal)
                obs[27] = math.cos(angle_goal)

        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True
            )
        a_v, a_omega = (float(t) for t in action[0])
        v_out = float(np.clip((a_v + 1.0) * 0.5 * (ESCORT_WALK_SPEED if env.task_phase == TaskPhase.GUIDE_ESCORT else V_MAX), 0.0, ESCORT_WALK_SPEED if env.task_phase == TaskPhase.GUIDE_ESCORT else V_MAX))
        omega_out = float(np.clip(a_omega * OMEGA_MAX, -OMEGA_MAX, OMEGA_MAX))
        self.n_steps += 1
        if self.trace and (env.task_phase == TaskPhase.GUIDE_ESCORT or self.n_steps % 5 == 0):
            d_p = np.hypot(env.patient_x - x, env.patient_z - z)
            d_g = np.hypot(final_x - x, final_z - z)
            print(
                f"[RL_SERVER] step={self.n_steps:<4d} phase={env.task_phase.name:<13s} "
                f"Robot=({x:6.2f}, {z:6.2f}) Patient=({env.patient_x:6.2f}, {env.patient_z:6.2f}) "
                f"d_pat={d_p:4.2f}m d_goal={d_g:4.2f}m | Action: v={v_out:.2f} w={omega_out:+.2f} (a=[{a_v:+.2f}, {a_omega:+.2f}])",
                flush=True,
            )
        return {
            "type": "rl_action",
            "id": msg.get("id", -1),
            "v": v_out,
            "omega": omega_out,
            "a": [a_v, a_omega],
            "phase": int(env.task_phase),
        }


def load_agent(checkpoint: Path) -> PPOHospitalAgent:
    agent = PPOHospitalAgent(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    agent.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    agent.eval()
    log.info("loaded %s (episode=%s)", checkpoint, ckpt.get("episode", "?"))
    return agent


async def handle_connection(ws: ServerConnection, agent: PPOHospitalAgent, lidar_source: str, trace: bool) -> None:
    runner = PolicyRunner(agent, lidar_source, trace)
    log.info("client connected")
    try:
        async for text in ws:
            try:
                msg = json.loads(text)
                if msg.get("type") != "rl_state":
                    raise ValueError(f"unexpected type {msg.get('type')!r}")
                reply = runner.act(msg)
            except (ValueError, KeyError, TypeError) as e:
                log.warning("bad message: %s", e)
                reply = {"type": "error", "message": str(e)}
            await ws.send(json.dumps(reply))
    except ConnectionClosed:
        pass
    log.info("client disconnected")


async def main_async(args: argparse.Namespace) -> None:
    agent = load_agent(Path(args.checkpoint))
    async with serve(lambda ws: handle_connection(ws, agent, args.lidar, args.trace), args.host, args.port) as server:
        log.info("listening on ws://%s:%d (lidar=%s)", args.host, args.port, args.lidar)
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO policy server for Godot RL controller")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / DEFAULT_CHECKPOINT))
    parser.add_argument("--host", default=RL_WS_HOST)
    parser.add_argument("--port", type=int, default=RL_WS_PORT)
    parser.add_argument("--lidar", choices=["godot", "sim"], default="godot",
                        help="godot: dùng LiDAR thật từ Godot; sim: tính lại từ tường 2D lúc train")
    parser.add_argument("--trace", action="store_true", help="Log trạng thái mỗi 1 s (5 bước)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
