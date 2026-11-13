"""RL policy server (port 8768): Godot gửi trạng thái thô, server trả lệnh (v, omega) từ PPO.

Hỗ trợ tự động:
- Policy Pure Vision V2 21-D (camera depth 5 sector + patient + phase + goal + tether), xem hospital_env_pure_vision_v2.py.
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
import os
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
from src.training.hospital_env_pure_vision_v2 import (  # noqa: E402
    DEPTH_HFOV_RAD,
    DEPTH_MAX_RANGE_M,
    DEPTH_NUM_RAYS,
    OBS_DIM_V2,
    depth_rays_to_sectors,
    remap_v1_obs,
    tether_features,
)
from src.training.hospital_env import ROBOT_RADIUS_M  # noqa: E402
from src.training.hospital_env_engage_v3 import DOA_HOLDS_MEMORY, OBS_DIM_V3, VOICE_MEMORY_S  # noqa: E402
from src.training.local_memory import (  # noqa: E402
    ObstacleMemory, PatientTrack, RecoveryCue, flank_proximity, person_slots,
)

OBS_DIM_V3_R8 = 35               # tới vòng 8: chưa có trí nhớ vật cản / kẹt / box người
# 31 = trước "mặt đã gặp"; 32 = trước "nhớ hướng giọng nói"; 35 = vòng 8; OBS_DIM_V3 (45) = vòng 9
OBS_DIM_V3_R9 = 45               # vòng 9-10: + trí nhớ vật cản, gỡ kẹt, box người to nhất
# OBS_DIM_V3 (59) = vòng 11: + người trong khung có cờ "đã gặp", nhớ chỗ bệnh nhân đang dẫn (quay lại hỏi)
V3_DIMS = (31, 32, OBS_DIM_V3_R8, OBS_DIM_V3_R9, OBS_DIM_V3)
STEP_DT_S = 0.2
from src.training.hospital_env_vision import (  # noqa: E402
    PUBLIC_PATROL_WAYPOINTS,
    HospitalVisualShiftEnv,
)

log = logging.getLogger("rl_policy_server")

RL_WS_HOST = "127.0.0.1"
RL_WS_PORT = 8768
DEFAULT_CHECKPOINT = "experiments/checkpoints/PPOHospital-PureVisionV2_best.pt"


class PolicyRunner:
    """Bộ dựng quan sát và suy luận PPO cho mỗi kết nối Godot."""

    def __init__(self, agent: PPOHospitalAgent, lidar_source: str, trace: bool = False) -> None:
        self.agent = agent
        self.obs_dim = agent.obs_dim
        self.lidar_source = lidar_source
        self.trace = trace
        self.n_steps = 0
        if self.obs_dim in (29, OBS_DIM_V2, *V3_DIMS):
            self.env = HospitalPureVisionEnv(curriculum_stage=1, env_mode="solo")
        else:
            self.env = HospitalVisualShiftEnv(curriculum_stage=1, env_mode="solo")
        self.env.reset()
        self.task_id: int | None = None
        self.patrol_wp_idx: int = 0
        self.lost_det_frames: int = 0
        self.last_vis_goal: tuple[float, float, float] = (0.05, 0.0, 1.0)
        self.last_sectors: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.tether_last_d: float = 1.0
        self.tether_unseen_s: float = 0.0
        # Vòng 9 (45-D): cùng lớp với môi trường train ➔ quan sát không lệch nhau
        self.round9: bool = self.obs_dim >= OBS_DIM_V3_R9
        self.round11: bool = self.obs_dim >= OBS_DIM_V3
        self.track = PatientTrack()
        self.mem = ObstacleMemory(DEPTH_HFOV_RAD, DEPTH_MAX_RANGE_M, ROBOT_RADIUS_M)
        self.stall = RecoveryCue()
        self.last_cmd_v: float = 0.0

    def _to_v2(self, obs29: np.ndarray, msg: dict) -> np.ndarray:
        """29-D pipeline (patient / phase / goal / kinematics) -> 21-D V2 obs with camera depth + tether."""
        sectors = depth_rays_to_sectors(depth_from_msg(msg))
        self.last_sectors = sectors.tolist()
        id_tether = (msg.get("v3") or {}).get("tether") if self.round9 else None
        if self.env.task_phase == TaskPhase.GUIDE_ESCORT and id_tether is not None and len(id_tether) >= 3:
            # Người đang dẫn theo ReID ở camera trước HOẶC sau (bệnh nhân thường đi sau lưng robot): [m, thấy, số giây mất dấu]
            tether = tether_features(float(np.clip(float(id_tether[0]), 0.5, 5.0)), float(id_tether[1]) > 0.5, float(id_tether[2]))
        elif self.env.task_phase == TaskPhase.GUIDE_ESCORT:
            seen = bool(obs29[5] > 0.5)
            if seen:
                self.tether_last_d = float(np.clip(1.20 / max(float(obs29[7]), 0.05), 0.5, 5.0))
                self.tether_unseen_s = 0.0
            tether = tether_features(self.tether_last_d, seen, self.tether_unseen_s)
            if not seen:
                self.tether_unseen_s += 0.2
        else:
            self.tether_last_d, self.tether_unseen_s = 1.0, 0.0
            tether = np.zeros(3, dtype=np.float32)
        return remap_v1_obs(obs29, sectors, tether)

    def act(self, msg: dict) -> dict:
        env = self.env
        x, z, yaw, v, omega = (float(t) for t in msg["robot"])
        env.robot_x, env.robot_z, env.robot_yaw = x, z, yaw
        env.robot_v, env.robot_omega = v, omega
        env.patient_x, env.patient_z = (float(t) for t in msg["patient"])

        phase = int(msg["phase"])
        # V3: đang nói chuyện ➔ giữ pha APPROACH như lúc train (YOLO mất box người sát robot không đổi sang SEARCH)
        in_dialogue = bool((msg.get("v3") or {}).get("dialogue"))
        task_id = int(msg.get("task_id", 0))
        if task_id != self.task_id:
            self.task_id = task_id
            self.lost_det_frames = 0
            self.track.reset()
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
        if self.obs_dim in (29, OBS_DIM_V2, *V3_DIMS):
            obs = env._get_obs()

            # Parse multi-class YOLO detections from Godot
            patient_box = None
            doctor_box = None
            door_box = None
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
                elif cid == 5 or cname == "door":
                    if door_box is None or float(d.get("conf", 0)) > float(door_box.get("conf", 0)):
                        door_box = d
                else:
                    obstacle_boxes.append(d)

            # Fallback if single legacy YOLO feature array was sent
            if patient_box is None and "yolo" in msg and len(msg["yolo"]) == 4 and float(msg["yolo"][0]) > 0.5:
                patient_box = {
                    "cx": float(msg["yolo"][1]),
                    "h": float(msg["yolo"][2]),
                    "conf": float(msg["yolo"][3]),
                }

            # Update 5-Sector Visual Depth Map (Strict Sim-to-Real: Zero map raycasts)
            sectors = compute_5_sector_depth(
                robot_x=x,
                robot_z=z,
                robot_yaw=yaw,
                fl=floor_of(x),
                structure_walls=None,
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
                if self.round9:
                    # Lúc train tâm box theo camera_vision_model là -1..1 (mép ảnh = ±1); box YOLO ở đây là -0.5..0.5
                    # ➔ trước vòng 9 policy thấy bệnh nhân lệch chỉ bằng nửa thực tế.
                    obs[6] = float(np.clip(2.0 * obs[6], -1.0, 1.0))
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
                elif (env.task_phase == TaskPhase.APPROACH_PATIENT and self.lost_det_frames < 10) or in_dialogue:
                    self.lost_det_frames += 1
                    env.task_phase = TaskPhase.APPROACH_PATIENT
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
            else:
                # Active navigation toward goal (GUIDE_ESCORT or CALL_STAFF / solo transit)
                env.set_nav_target(final_x, final_z)
                to_gx = env.goal_x - x
                to_gz = env.goal_z - z
                fwd_x = -math.sin(yaw)
                fwd_z = -math.cos(yaw)
                angle_goal = signed_angle_rad(fwd_x, fwd_z, to_gx, to_gz)
                obs[17] = float(np.clip(env.nav_remaining / 25.0, 0.0, 1.0))
                obs[18] = math.sin(angle_goal)
                obs[19] = math.cos(angle_goal)

            # Robot kinematics [20..21]
            obs[20] = float(np.clip(v / V_MAX, 0.0, 1.0))
            obs[21] = float(np.clip(omega / OMEGA_MAX, -1.0, 1.0))

            # Patient tether [22..24] (Strictly for GUIDE_ESCORT with escorted patient)
            if env.task_phase == TaskPhase.GUIDE_ESCORT:
                if patient_box is not None:
                    h_pat = float(patient_box.get("h", float(patient_box.get("y2", 0.0)) - float(patient_box.get("y1", 0.0))))
                    d_vis_tether = float(np.clip(1.20 / max(h_pat, 0.05), 0.5, 5.0))
                    obs[22] = float(np.clip(d_vis_tether / 5.0, 0.0, 1.0))
                    obs[23] = 1.0 if d_vis_tether > 2.0 else 0.0
                    obs[24] = 1.0 if d_vis_tether > 3.5 else 0.0
                else:
                    # Patient lost from camera FOV
                    obs[22] = 1.0
                    obs[23] = 1.0
                    obs[24] = 1.0
            else:
                # In CALL_STAFF (solo return/transit), SEARCH, APPROACH: tether disabled
                obs[22:25] = 0.0

            # Doorway geometry [25..28] (Strictly from YOLO 'door' detection, zero oracle _DOORS)
            if door_box is not None:
                door_cx = float(door_box.get("cx", (float(door_box.get("x1", 0.0)) + float(door_box.get("x2", 0.0))) / 2.0 - 0.5))
                door_h = float(door_box.get("h", float(door_box.get("y2", 0.0)) - float(door_box.get("y1", 0.0))))
                d_vis_door = float(np.clip(1.20 / max(door_h, 0.05), 0.5, 10.0))
                angle_door = float(-math.atan(door_cx * math.tan(CAMERA_HFOV_RAD / 2.0)))
                obs[25] = float(np.clip(d_vis_door / 10.0, 0.0, 1.0))
                obs[26] = math.sin(angle_door)
                obs[27] = math.cos(angle_door)
                obs[28] = 1.0 if d_vis_door < 2.0 else 0.0
            else:
                obs[25:29] = 0.0

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
                elif (env.task_phase == TaskPhase.APPROACH_PATIENT and self.lost_det_frames < 10) or in_dialogue:
                    self.lost_det_frames += 1
                    env.task_phase = TaskPhase.APPROACH_PATIENT
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

        returning = self.round9 and phase == TaskPhase.CALL_STAFF
        if returning:
            # Đi một mình tới đích (quay về thang máy / đi gọi nhân viên): lúc train là pha SEARCH + "đang quay về", không
            # nhận bệnh nhân mới. Pha CALL_STAFF chưa từng có lúc train escort-only.
            obs[5:9] = 0.0
            obs[13:17] = 0.0
            obs[13] = 1.0
        if self.obs_dim in (OBS_DIM_V2, *V3_DIMS):
            obs = self._to_v2(obs, msg)
        if self.obs_dim in V3_DIMS:
            v3 = msg.get("v3") or {}
            feats = v3_features(v3, hold_doa=DOA_HOLDS_MEMORY and self.obs_dim >= 35)
            obs = np.concatenate([obs, feats[: min(self.obs_dim, OBS_DIM_V3_R8) - OBS_DIM_V2]]).astype(np.float32)
        if self.round9:
            v3 = msg.get("v3") or {}
            self.mem.update(x, z, yaw, depth_from_msg(msg), self.n_steps * STEP_DT_S)
            if in_dialogue and obs[34] < 1.0:
                obs[13], obs[14], obs[15] = 0.0, -obs[32], obs[33]   # đang nói chuyện: "đích" là hướng người nói
            mem = self.mem.sectors(x, z, yaw)
            # Tín hiệu gỡ kẹt: lệnh đi của bước trước mà robot không nhúc nhích (bị giữ đứng yên lúc nói chuyện thì không tính)
            left, right = flank_proximity(obs[0:5], mem)
            cue = self.stall.update(self.last_cmd_v, v, STEP_DT_S, not (in_dialogue or bool(v3.get("held"))), left, right,
                                    math.atan2(float(obs[14]), float(obs[15])))
            if self.round11:
                items = [(float(p[0]), float(p[1]), float(p[2]) > 0.5) for p in v3.get("persons", [])] \
                    if "persons" in v3 else [(float(b[1]), float(b[2]), False) for b in [largest_person_box(dets)] if b[0] > 0.5]
                slot0, extra = person_slots(items)
                t = self.n_steps * STEP_DT_S
                if env.task_phase == TaskPhase.GUIDE_ESCORT:
                    idp = v3.get("id_pos")
                    if idp is not None and len(idp) >= 3 and float(idp[0]) > 0.5:
                        self.track.update((float(idp[1]), float(idp[2])), t)
                    track = self.track.obs(x, z, yaw, t)
                else:
                    self.track.reset()
                    track = np.zeros(5, dtype=np.float32)
                obs = np.concatenate([obs, mem, [cue], slot0, extra, track]).astype(np.float32)
            else:
                obs = np.concatenate([obs, mem, [cue], largest_person_box(dets)]).astype(np.float32)

        self.last_obs = obs                    # test so khớp với môi trường train (tests/test_policy_v3.py)
        # Forward through Actor network
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True
            )
        a_all = [float(t) for t in action[0]]
        if OBS_DUMP_PATH:                       # debug sim-to-sim: quan sát thật Godot gửi (so với lúc train)
            with open(OBS_DUMP_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": self.n_steps, "obs": [round(float(o), 4) for o in obs],
                                    "a": [round(t, 4) for t in a_all]}) + "\n")
        a_v, a_omega = a_all[0], a_all[1]
        max_v = ESCORT_WALK_SPEED if env.task_phase == TaskPhase.GUIDE_ESCORT else V_MAX
        v_out = float(np.clip((a_v + 1.0) * 0.5 * max_v, 0.0, max_v))
        omega_out = float(np.clip(a_omega * OMEGA_MAX, -OMEGA_MAX, OMEGA_MAX))
        self.last_cmd_v = v_out
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

        reply = {
            "type": "rl_action",
            "id": msg.get("id", -1),
            "v": v_out,
            "omega": omega_out,
            "a": a_all,
            "phase": int(env.task_phase),
            "sectors": self.last_sectors,
        }
        if len(a_all) >= 4:   # V3: đầu ngẩng / cúi (mục tiêu, [-1, 1]) và hỏi "Cần tôi giúp gì không ạ?"
            reply["tilt"] = float(np.clip(a_all[2], -1.0, 1.0))
            reply["ask"] = bool(a_all[3] > 0.5)
            reply["returns"] = self.round11     # policy tự quay lại hỏi bệnh nhân dừng lại (Godot tắt luật tự hỏi)
        return reply


OBS_DUMP_PATH = os.environ.get("RL_OBS_DUMP", "")


def v3_features(v3: dict, hold_doa: bool = DOA_HOLDS_MEMORY) -> np.ndarray:
    """10 quan sát tương tác của V3 (docs/RL_V3_DESIGN.md) từ Godot:
    voice [active, bearing_rad (+ phải)], face [seen, cx, cy, size] (cx, cy ∈ −0.5..0.5), tilt (−1..1),
    since_ask_s, dialogue (0/1); face[4] = mặt đó là người đã gặp (đã giúp / đã từ chối);
    voice_mem [valid, bearing_rad (+ phải, đã xoay theo odometry), age_s] = hướng giọng nói nghe lần cuối.
    hold_doa: lúc im lặng, ô DOA giữ hướng đã nhớ (thử ở vòng 8, kém hơn ➔ tắt, theo DOA_HOLDS_MEMORY)."""
    voice = list(v3.get("voice", [0.0, 0.0])) + [0.0, 0.0]
    face = list(v3.get("face", [0.0, 0.0, 0.0, 0.0, 0.0])) + [0.0] * 5
    active = 1.0 if float(voice[0]) > 0.5 else 0.0
    ang = float(voice[1])
    mem = list(v3.get("voice_mem", [0.0, 0.0, 99.0])) + [0.0, 0.0, 99.0]
    mem_ok = float(mem[0]) > 0.5 and float(mem[2]) <= VOICE_MEMORY_S
    mem_ang = float(mem[1])
    if active:
        doa = (math.sin(ang), math.cos(ang))
    elif hold_doa and mem_ok:
        doa = (math.sin(mem_ang), math.cos(mem_ang))
    else:
        doa = (0.0, 0.0)
    return np.array([
        active, doa[0], doa[1],
        1.0 if float(face[0]) > 0.5 else 0.0, float(face[1]), float(face[2]), float(face[3]),
        float(np.clip(v3.get("tilt", 0.0), -1.0, 1.0)),
        float(np.clip(float(v3.get("since_ask_s", 99.0)) / 5.0, 0.0, 1.0)),
        1.0 if v3.get("dialogue") else 0.0,
        1.0 if float(face[0]) > 0.5 and float(face[4]) > 0.5 else 0.0,
        math.sin(mem_ang) if mem_ok else 0.0, math.cos(mem_ang) if mem_ok else 0.0,
        float(mem[2]) / VOICE_MEMORY_S if mem_ok else 1.0,
    ], dtype=np.float32)


def largest_person_box(dets: list) -> np.ndarray:
    """[thấy, cx (-0.5..0.5), h] của box người to nhất ở camera trước (YOLO lớp patient / doctor), bất kể là ai."""
    best = (0.0, 0.0, 0.0)
    for d in dets:
        if int(d.get("class_id", -1)) not in (0, 1) and str(d.get("class_name", "")) not in ("patient", "doctor"):
            continue
        h = float(d.get("y2", 0.0)) - float(d.get("y1", 0.0))
        if h > best[2]:
            best = (1.0, (float(d.get("x1", 0.0)) + float(d.get("x2", 0.0))) * 0.5 - 0.5, h)
    return np.asarray(best, dtype=np.float32)


def depth_from_msg(msg: dict) -> np.ndarray:
    """15 depth distances (m). Prefer Godot's camera-depth fan; fall back to the LiDAR rays inside the HFOV."""
    depth = msg.get("depth")
    if depth is not None and len(depth) == DEPTH_NUM_RAYS:
        return np.clip(np.asarray(depth, dtype=np.float32), 0.0, DEPTH_MAX_RANGE_M)
    lidar = np.asarray(msg.get("lidar", []), dtype=np.float32) * 6.0
    if lidar.size < 2:
        return np.full(DEPTH_NUM_RAYS, DEPTH_MAX_RANGE_M, dtype=np.float32)
    lidar_ang = np.linspace(-100.0, 100.0, lidar.size)
    want = np.degrees(np.linspace(-DEPTH_HFOV_RAD / 2.0, DEPTH_HFOV_RAD / 2.0, DEPTH_NUM_RAYS))
    return np.clip(np.interp(want, lidar_ang, lidar), 0.0, DEPTH_MAX_RANGE_M).astype(np.float32)


def load_agent(checkpoint: Path) -> PPOHospitalAgent:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Dynamic observation dimension detection
    first_weight = state_dict.get("actor_backbone.0.weight")
    in_dim = first_weight.shape[1] if first_weight is not None else OBS_DIM_PURE_VISION
    mean_w = state_dict.get("actor_mean.weight")
    act_dim = mean_w.shape[0] if mean_w is not None else ACT_DIM
    agent = PPOHospitalAgent(obs_dim=in_dim, act_dim=act_dim)
    agent.load_state_dict(state_dict)
    agent.eval()
    log.info("Loaded checkpoint %s (in_dim=%d, act_dim=%d, episode=%s)", checkpoint, in_dim, act_dim, ckpt.get("episode", "?"))
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
    if not ckpt_path.is_absolute() and not ckpt_path.is_file():
        ckpt_path = REPO_ROOT / ckpt_path  # launcher chạy với cwd=sim_vision/python
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
