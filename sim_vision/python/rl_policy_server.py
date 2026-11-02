"""RL policy server (port 8768): Godot gửi trạng thái thô, server trả lệnh (v, omega) từ PPO.

Hỗ trợ tự động:
- Policy thuần thị giác Pure Vision 29-D (5-sector visual depth, patient/doctor channels, ZERO LiDAR).
- Policy lai Hybrid 37-D (17-ray LiDAR + YOLO person).
- Policy cũ Legacy 31-D.
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

from src.models.ppo_agent import (  # noqa: E402
    ACT_DIM,
    OBS_DIM_PURE_VISION,
    PPOHospitalAgent,
)
from src.training.camera_vision_model import CAMERA_HFOV_RAD  # noqa: E402
from src.training.hospital_env import (  # noqa: E402
    ESCORT_WALK_SPEED,
    OMEGA_MAX,
    V_MAX,
    HospitalShiftEnv,
    TaskPhase,
    floor_of,
    signed_angle_rad,
)
from src.training.hospital_env_pure_vision import (  # noqa: E402
    HospitalPureVisionEnv,
    compute_5_sector_depth,
)
from src.training.hospital_env_vision import (  # noqa: E402
    PUBLIC_PATROL_WAYPOINTS,
    HospitalVisualShiftEnv,
)

log = logging.getLogger("rl_policy_server")

RL_WS_HOST = "127.0.0.1"
RL_WS_PORT = 8768
DEFAULT_CHECKPOINT = "experiments/checkpoints/PPOHospital-full_strongest.pt"


class PolicyRunner:
    """Bộ dựng quan sát và suy luận PPO cho mỗi kết nối Godot."""

    def __init__(self, agent: PPOHospitalAgent, lidar_source: str, trace: bool = False) -> None:
        self.agent = agent
        self.obs_dim = agent.obs_dim
        self.lidar_source = lidar_source
        self.trace = trace
        self.n_steps = 0
        if self.obs_dim == 29:
            self.env = HospitalPureVisionEnv(curriculum_stage=1, env_mode="solo")
        else:
            self.env = HospitalVisualShiftEnv(curriculum_stage=1, env_mode="solo")
        self.env.reset()
        self.task_id: int | None = None
        self.patrol_wp_idx: int = 0
        self.lost_det_frames: int = 0
        self.last_vis_goal: tuple[float, float, float] = (0.05, 0.0, 1.0)
        self.last_sectors: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]

    def act(self, msg: dict) -> dict:
        env = self.env
        x, z, yaw, v, omega = (float(t) for t in msg["robot"])
        env.robot_x, env.robot_z, env.robot_yaw = x, z, yaw
        env.robot_v, env.robot_omega = v, omega
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
            env.custom_nav_target = (final_x, final_z)
            env.set_nav_target(final_x, final_z)
        else:
            env.custom_nav_target = None
            if env.task_phase not in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
                env.task_phase = TaskPhase.SEARCH_PATIENT
            final_x, final_z = env.patient_x, env.patient_z

        dets = msg.get("dets", msg.get("yolo_dets", []))

        # =========================================================================
        # 1. PURE VISION MODE (29-D, ZERO LIDAR)
        # =========================================================================
        if self.obs_dim == 29:
            obs = env._get_obs()

            # Parse multi-class YOLO detections from Godot
            patient_box = None
            doctor_box = None
            obstacle_boxes = []

            for d in dets:
                cid = int(d.get("class_id", -1))
                cname = str(d.get("class_name", ""))
                if cid == 0 or cname == "patient":
                    if patient_box is None or float(d.get("conf", 0)) > float(patient_box.get("conf", 0)):
                        patient_box = d
                elif cid == 1 or cname == "doctor":
                    if doctor_box is None or float(d.get("conf", 0)) > float(doctor_box.get("conf", 0)):
                        doctor_box = d
                    obstacle_boxes.append(d)
                else:
                    obstacle_boxes.append(d)

            # Fallback if single legacy YOLO feature array was sent
            if patient_box is None and "yolo" in msg and len(msg["yolo"]) == 4 and float(msg["yolo"][0]) > 0.5:
                patient_box = {
                    "cx": float(msg["yolo"][1]),
                    "h": float(msg["yolo"][2]),
                    "conf": float(msg["yolo"][3]),
                }

            # Update 5-Sector Visual Depth Map
            from src.training.hospital_env import _STRUCTURE
            sectors = compute_5_sector_depth(
                robot_x=x,
                robot_z=z,
                robot_yaw=yaw,
                fl=floor_of(x),
                structure_walls=_STRUCTURE,
                obstacle_boxes=obstacle_boxes,
            )
            obs[0:5] = sectors
            self.last_sectors = sectors.tolist()

            # Patient channel [5..8]
            has_pat = False
            if patient_box is not None:
                has_pat = True
                obs[5] = 1.0
                obs[6] = float(patient_box.get("cx", (float(patient_box.get("x1", 0.0)) + float(patient_box.get("x2", 0.0))) / 2.0 - 0.5))
                obs[7] = float(patient_box.get("h", float(patient_box.get("y2", 0.0)) - float(patient_box.get("y1", 0.0))))
                obs[8] = float(patient_box.get("conf", 0.9))

            # Doctor channel [9..12]
            if doctor_box is not None:
                obs[9] = 1.0
                obs[10] = float(doctor_box.get("cx", (float(doctor_box.get("x1", 0.0)) + float(doctor_box.get("x2", 0.0))) / 2.0 - 0.5))
                obs[11] = float(doctor_box.get("h", float(doctor_box.get("y2", 0.0)) - float(doctor_box.get("y1", 0.0))))
                obs[12] = float(doctor_box.get("conf", 0.9))
            else:
                obs[9:13] = 0.0

            # Target approach vs search switching (ONLY for patient, NEVER for doctor!)
            if env.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
                if has_pat:
                    self.lost_det_frames = 0
                    env.task_phase = TaskPhase.APPROACH_PATIENT
                    obs[13:17] = 0.0
                    obs[14] = 1.0  # APPROACH_PATIENT
                    cx_off = float(obs[6])
                    h = float(obs[7])
                    d_vis = float(np.clip(1.20 / max(h, 0.05), 0.8, 15.0))
                    angle_vis = float(-math.atan(cx_off * math.tan(CAMERA_HFOV_RAD / 2.0)))
                    obs[17] = float(np.clip(d_vis / 25.0, 0.0, 1.0))
                    obs[18] = math.sin(angle_vis)
                    obs[19] = math.cos(angle_vis)
                    self.last_vis_goal = (obs[17], obs[18], obs[19])
                elif env.task_phase == TaskPhase.APPROACH_PATIENT and self.lost_det_frames < 10:
                    self.lost_det_frames += 1
                    obs[13:17] = 0.0
                    obs[14] = 1.0
                    obs[17], obs[18], obs[19] = self.last_vis_goal
                else:
                    self.lost_det_frames = 0
                    env.task_phase = TaskPhase.SEARCH_PATIENT
                    obs[13:17] = 0.0
                    obs[13] = 1.0  # SEARCH_PATIENT
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
                    obs[17] = float(np.clip(env.nav_remaining / 25.0, 0.0, 1.0))
                    obs[18] = math.sin(angle_goal)
                    obs[19] = math.cos(angle_goal)

        # =========================================================================
        # 2. HYBRID / LEGACY MODE (31-D or 37-D with LiDAR)
        # =========================================================================
        else:
            obs = env._get_obs()
            if self.lidar_source == "godot":
                lidar = np.asarray(msg.get("lidar", []), dtype=np.float32)
                if lidar.shape == (17,):
                    obs[0:17] = np.clip(lidar, 0.0, 1.0)

            if env.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
                has_det = False
                if "yolo" in msg and len(msg["yolo"]) == 4:
                    obs[17:21] = np.asarray(msg["yolo"], dtype=np.float32)
                    if obs[17] > 0.5:
                        has_det = True
                        env.task_phase = TaskPhase.APPROACH_PATIENT
                        obs[21:25] = 0.0
                        obs[22] = 1.0

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
                    self.lost_det_frames += 1
                    obs[21:25] = 0.0
                    obs[22] = 1.0
                    obs[25], obs[26], obs[27] = self.last_vis_goal
                else:
                    self.lost_det_frames = 0
                    env.task_phase = TaskPhase.SEARCH_PATIENT
                    obs[21:25] = 0.0
                    obs[21] = 1.0
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

        # Forward through Actor network
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True
            )
        a_v, a_omega = (float(t) for t in action[0])
        max_v = ESCORT_WALK_SPEED if env.task_phase == TaskPhase.GUIDE_ESCORT else V_MAX
        v_out = float(np.clip((a_v + 1.0) * 0.5 * max_v, 0.0, max_v))
        omega_out = float(np.clip(a_omega * OMEGA_MAX, -OMEGA_MAX, OMEGA_MAX))
        self.n_steps += 1

        if self.trace and (env.task_phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF) or self.n_steps % 5 == 0):
            d_p = np.hypot(env.patient_x - x, env.patient_z - z)
            d_g = np.hypot(final_x - x, final_z - z)
            print(
                f"[RL_SERVER] step={self.n_steps:<4d} phase={env.task_phase.name:<13s} "
                f"Robot=({x:6.2f}, {z:6.2f}) Patient=({env.patient_x:6.2f}, {env.patient_z:6.2f}) "
                f"d_pat={d_p:4.2f}m d_goal={d_g:4.2f}m | Action: v={v_out:.2f} w={omega_out:+.2f} | Sectors={self.last_sectors}",
                flush=True,
            )

        return {
            "type": "rl_action",
            "id": msg.get("id", -1),
            "v": v_out,
            "omega": omega_out,
            "a": [a_v, a_omega],
            "phase": int(env.task_phase),
            "sectors": self.last_sectors,
        }


def load_agent(checkpoint: Path) -> PPOHospitalAgent:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Dynamic observation dimension detection
    first_weight = state_dict.get("actor_backbone.0.weight")
    in_dim = first_weight.shape[1] if first_weight is not None else OBS_DIM_PURE_VISION
    agent = PPOHospitalAgent(obs_dim=in_dim, act_dim=ACT_DIM)
    agent.load_state_dict(state_dict)
    agent.eval()
    log.info("Loaded checkpoint %s (in_dim=%d, episode=%s)", checkpoint, in_dim, ckpt.get("episode", "?"))
    return agent


async def handle_connection(ws: ServerConnection, runner: PolicyRunner) -> None:
    async for raw in ws:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict) or msg.get("type") != "rl_state":
            continue
        reply = runner.act(msg)
        await ws.send(json.dumps(reply))


async def serve_forever(checkpoint: Path, lidar_source: str, port: int, trace: bool) -> None:
    agent = load_agent(checkpoint)
    runner = PolicyRunner(agent, lidar_source, trace)
    server = await serve(lambda ws: handle_connection(ws, runner), RL_WS_HOST, port, max_size=2**24)
    log.info("Listening on ws://%s:%d (obs_dim=%d, lidar=%s)", RL_WS_HOST, port, agent.obs_dim, lidar_source)
    await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO Hospital Policy WebSocket Server")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lidar", default="godot", choices=["godot", "sim"])
    parser.add_argument("--port", type=int, default=RL_WS_PORT)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        log.error("Checkpoint not found: %s", ckpt_path)
        return 1

    try:
        asyncio.run(serve_forever(ckpt_path, args.lidar, args.port, args.trace))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
