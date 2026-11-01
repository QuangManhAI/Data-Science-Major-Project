"""Pure Vision (YOLO Only, Zero LiDAR) Hospital Environment.

Features:
- OBS_DIM = 29 (No LiDAR!)
  * [0..4]   : 5-Sector Visual Depth Map (Far-Left, Left, Center, Right, Far-Right)
  * [5..8]   : Patient Channel [seen, cx_off, h, w]
  * [9..12]  : Doctor Channel [doc_seen, doc_cx_off, doc_h, doc_w]
  * [13..16] : 4D One-hot Task Phase [SEARCH, APPROACH, GUIDE_ESCORT, CALL_STAFF]
  * [17..19] : 3D Nav Goal Polar Vector [goal_dist_norm, sin_theta, cos_theta]
  * [20..21] : 2D Robot Kinematics [v / V_MAX, omega / OMEGA_MAX]
  * [22..24] : 3D Patient Tether Status [d_tether_norm, tether_tight, tether_broken]
  * [25..28] : 4D Doorway Geometry [door_dist_norm, sin_door_angle, cos_door_angle, door_open]

- Reward Remapping:
  * Strict Doctor Avoidance (-5.0 false engagement penalty)
  * Continuous Visual Obstacle Repulsion (-0.50 * sector_h when h > 0.50)
  * Physical Collision Penalty (-5.0)
  * Free Search Penalty (-0.04 / step while in SEARCH_PATIENT)
  * Patient Spotted (+5.0, once per patient)
  * Patient Engaged (+50.0, strictly for patient class)
  * Guide Arrived (+80.0 at destination clinic)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.models.ppo_agent import OBS_DIM_PURE_VISION, OMEGA_MAX, V_MAX
from src.training.hospital_env import (
    _DOORS,
    DESTINATION_POSITIONS,
    ESCORT_WALK_SPEED,
    TaskPhase,
    _STRUCTURE,
    cast_rays,
    floor_of,
    signed_angle_rad,
)
from src.training.hospital_env_vision import (
    CAMERA_HFOV_RAD,
    PATIENT_TALK_DIST_M_VIS,
    PUBLIC_PATROL_WAYPOINTS,
    TARGET_BOX_H,
    HospitalVisualShiftEnv,
    compute_pinhole_detection,
)

OBS_DIM = OBS_DIM_PURE_VISION  # 29
PENALTY_DOCTOR_PROXIMITY = -5.0
PENALTY_FREE_SEARCH_TIME = -0.04
REWARD_VISUAL_SPOTTED = 5.0
REWARD_PATIENT_ENGAGED_VIS = 50.0
REWARD_GUIDE_ARRIVED_VIS = 80.0
WEIGHT_VISUAL_OBSTACLE = 0.50

# 5 Sector center angles in camera coordinate frame (radians) across 75 deg FOV
SECTOR_ANGLES_RAD = np.array(
    [
        math.radians(-30.0),  # Far-Left
        math.radians(-15.0),  # Left
        math.radians(0.0),    # Center
        math.radians(+15.0),  # Right
        math.radians(+30.0),  # Far-Right
    ],
    dtype=np.float32,
)


def compute_5_sector_depth(
    robot_x: float,
    robot_z: float,
    robot_yaw: float,
    fl: int,
    structure_walls: dict[int, np.ndarray],
    obstacle_boxes: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Compute 5-sector visual obstacle proximity [0..1]^5.

    Combines:
    1. Optical pinhole raycast to hospital walls at sector ray bearings.
    2. Any detected 2D obstacle bounding boxes mapped into horizontal sectors.
    """
    sectors = np.zeros(5, dtype=np.float32)

    # 1. Optical Wall Raycast across the 5 sectors
    if fl in structure_walls:
        walls = structure_walls[fl]
        ray_yaws = robot_yaw + SECTOR_ANGLES_RAD
        # Robot fwd in world is (-sin(yaw), -cos(yaw))
        ux = -np.sin(ray_yaws)
        uz = -np.cos(ray_yaws)
        hits = cast_rays(robot_x, robot_z, ux, uz, 8.0, walls)
        # Wall distance converts to optical height proxy: h = 1.20 / d_wall
        for i in range(5):
            val = float(hits[i])
            if not math.isfinite(val):
                val = 8.0
            d_w = max(0.2, min(val, 25.0))
            h_wall = float(np.clip(1.20 / d_w, 0.0, 1.0))
            sectors[i] = h_wall

    # 2. Obstacle Bounding Boxes (if provided, e.g. from YOLO detections)
    if obstacle_boxes:
        for b in obstacle_boxes:
            cx_val = b.get("cx", (float(b.get("x1", 0.0)) + float(b.get("x2", 0.0))) / 2.0 - 0.5)
            h_val = b.get("h", float(b.get("y2", 0.0)) - float(b.get("y1", 0.0)))
            cx = float(cx_val) if math.isfinite(cx_val) else 0.0
            h = float(h_val) if math.isfinite(h_val) else 0.0
            # Map cx in [-0.5, 0.5] to sector index 0..4
            sec_idx = int(np.clip(math.floor((cx + 0.5) * 5.0), 0, 4))
            sectors[sec_idx] = max(sectors[sec_idx], float(np.clip(h, 0.0, 1.0)))

    return np.nan_to_num(np.clip(sectors, 0.0, 1.0), nan=0.0)


def get_nearest_door(robot_x: float, robot_z: float) -> tuple[float, float, float]:
    fl = floor_of(robot_x)
    best_dist = 999.0
    best_dx = robot_x
    best_dz = robot_z
    for (floor, side), (centres, _) in _DOORS.items():
        if floor == fl:
            z_line = -2.0 if side == "north" else 2.0
            for c in centres:
                d = math.hypot(robot_x - c, robot_z - z_line)
                if d < best_dist:
                    best_dist = d
                    best_dx = c
                    best_dz = z_line
    return best_dx, best_dz, best_dist


class HospitalPureVisionEnv(HospitalVisualShiftEnv):
    """Pure Vision (YOLO Only, Zero LiDAR) Hospital Shift Environment."""

    def __init__(
        self,
        curriculum_stage: int = 1,
        shift_duration_s: float = 300.0,
        dt: float = 0.2,
        seed: int = 42,
        env_mode: str = "full",
        stagnation_timeout_s: float = 60.0,
        stagnation_penalty: float = -10.0,
        escort_only: bool = True,
        domain_randomization: bool = True,
    ) -> None:
        super().__init__(
            curriculum_stage=curriculum_stage,
            shift_duration_s=shift_duration_s,
            dt=dt,
            seed=seed,
            env_mode=env_mode,
            stagnation_timeout_s=stagnation_timeout_s,
            stagnation_penalty=stagnation_penalty,
            escort_only=escort_only,
            domain_randomization=domain_randomization,
        )
        self.last_doctor_det: dict[str, float] = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
        self.last_sectors: np.ndarray = np.zeros(5, dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = super().reset(seed=seed)
        self.last_doctor_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
        self.last_sectors = np.zeros(5, dtype=np.float32)
        return self._get_obs()

    def _compute_doctor_det(self) -> tuple[float, float, float, float]:
        """Compute pinhole camera detection for patrolling doctor obstacle."""
        if not getattr(self, "doctor_active", False):
            self.last_doctor_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
            return 0.0, 0.0, 0.0, 0.0

        if floor_of(self.doctor_x) != floor_of(self.robot_x):
            self.last_doctor_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
            return 0.0, 0.0, 0.0, 0.0

        det = compute_pinhole_detection(
            robot_x=self.robot_x,
            robot_z=self.robot_z,
            robot_yaw=self.robot_yaw,
            target_x=self.doctor_x,
            target_z=self.doctor_z,
            target_seated=False,
            is_blocked_fn=self._check_los_occluded,
            domain_randomization=self.domain_randomization,
            rng=self.rng,
        )
        self.last_doctor_det = det
        conf = float(np.clip(0.95 - 0.04 * math.hypot(self.doctor_x - self.robot_x, self.doctor_z - self.robot_z), 0.45, 0.98)) if det["seen"] > 0.5 else 0.0
        return det["seen"], det["cx"], det["h"], conf

    def _get_obs(self) -> np.ndarray:
        """Build 29-D pure-vision observation vector. ZERO LiDAR."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        fl = floor_of(self.robot_x)

        # 1. Compute 5-Sector Visual Depth Map [0..4]
        # Include doctor bounding box as an obstacle for sector clearance
        doc_seen, doc_cx, doc_h, doc_conf = self._compute_doctor_det()
        obs_boxes = []
        if doc_seen > 0.5:
            obs_boxes.append({"cx": doc_cx, "h": doc_h})

        sectors = compute_5_sector_depth(
            robot_x=self.robot_x,
            robot_z=self.robot_z,
            robot_yaw=self.robot_yaw,
            fl=fl,
            structure_walls=_STRUCTURE,
            obstacle_boxes=obs_boxes,
        )
        self.last_sectors = sectors
        obs[0:5] = sectors

        # 2. Patient Channel [5..8]
        pat_seen, pat_cx, pat_h, pat_conf = self._compute_yolo_det()
        if self.task_phase == TaskPhase.SEARCH_PATIENT and pat_seen > 0.5:
            self.task_phase = TaskPhase.APPROACH_PATIENT

        obs[5] = pat_seen
        obs[6] = pat_cx
        obs[7] = pat_h
        obs[8] = pat_conf

        # 3. Doctor Channel [9..12]
        obs[9] = doc_seen
        obs[10] = doc_cx
        obs[11] = doc_h
        obs[12] = doc_conf

        # 4. One-Hot Task Phase [13..16]
        phase_idx = int(np.clip(int(self.task_phase), 0, 3))
        obs[13 + phase_idx] = 1.0

        # 5. Nav Goal Target [17..19]
        if self.task_phase == TaskPhase.APPROACH_PATIENT and pat_seen > 0.5:
            d_vis = float(np.clip(1.20 / max(pat_h, 0.05), 0.8, 15.0))
            angle_vis = float(-math.atan(pat_cx * math.tan(CAMERA_HFOV_RAD / 2.0)))
            obs[17] = float(np.clip(d_vis / 25.0, 0.0, 1.0))
            obs[18] = math.sin(angle_vis)
            obs[19] = math.cos(angle_vis)
        elif self.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
            waypoints = PUBLIC_PATROL_WAYPOINTS.get(fl, PUBLIC_PATROL_WAYPOINTS[1])
            pw_x, pw_z = waypoints[self.patrol_wp_idx % len(waypoints)]
            if math.hypot(pw_x - self.robot_x, pw_z - self.robot_z) <= 1.5:
                self.patrol_wp_idx = (self.patrol_wp_idx + 1) % len(waypoints)
                pw_x, pw_z = waypoints[self.patrol_wp_idx % len(waypoints)]
            self.set_nav_target(pw_x, pw_z)
            to_gx = self.goal_x - self.robot_x
            to_gz = self.goal_z - self.robot_z
            fwd_x = -math.sin(self.robot_yaw)
            fwd_z = -math.cos(self.robot_yaw)
            angle_goal = signed_angle_rad(fwd_x, fwd_z, to_gx, to_gz)
            obs[17] = float(np.clip(self.nav_remaining / 25.0, 0.0, 1.0))
            obs[18] = math.sin(angle_goal)
            obs[19] = math.cos(angle_goal)
        else:
            to_gx = self.goal_x - self.robot_x
            to_gz = self.goal_z - self.robot_z
            fwd_x = -math.sin(self.robot_yaw)
            fwd_z = -math.cos(self.robot_yaw)
            angle_goal = signed_angle_rad(fwd_x, fwd_z, to_gx, to_gz)
            obs[17] = float(np.clip(self.nav_remaining / 25.0, 0.0, 1.0))
            obs[18] = math.sin(angle_goal)
            obs[19] = math.cos(angle_goal)

        # 6. Robot Kinematics [20..21]
        obs[20] = float(np.clip(self.robot_v / V_MAX, 0.0, 1.0))
        obs[21] = float(np.clip(self.robot_omega / OMEGA_MAX, -1.0, 1.0))

        # 7. Patient Tether Status [22..24]
        if self.task_phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF):
            d_p = math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
            obs[22] = float(np.clip(d_p / 5.0, 0.0, 1.0))
            obs[23] = 1.0 if d_p > 2.0 else 0.0
            obs[24] = 1.0 if d_p > 3.5 else 0.0
        else:
            obs[22] = 0.0
            obs[23] = 0.0
            obs[24] = 0.0

        # 8. Doorway Geometry [25..28]
        door_x, door_z, d_door = get_nearest_door(self.robot_x, self.robot_z)
        to_dx = door_x - self.robot_x
        to_dz = door_z - self.robot_z
        fwd_x = -math.sin(self.robot_yaw)
        fwd_z = -math.cos(self.robot_yaw)
        angle_door = signed_angle_rad(fwd_x, fwd_z, to_dx, to_dz)
        obs[25] = float(np.clip(d_door / 10.0, 0.0, 1.0))
        obs[26] = math.sin(angle_door)
        obs[27] = math.cos(angle_door)
        obs[28] = 1.0 if d_door < 2.0 else 0.0

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        # Free search step penalty
        free_search_penalty = (
            PENALTY_FREE_SEARCH_TIME if self.task_phase == TaskPhase.SEARCH_PATIENT else 0.0
        )

        obs, reward, done, info = super().step(action)
        reward += free_search_penalty

        # 1. Continuous Visual Obstacle Proximity Repulsion
        # When any visual sector exceeds 0.50 (obstacle closer than ~1.2m)
        obs_sectors = obs[0:5]
        for s_val in obs_sectors:
            if s_val > 0.50:
                reward += -WEIGHT_VISUAL_OBSTACLE * (s_val - 0.50)

        # 2. Strict Doctor Avoidance Penalty
        # If robot faces doctor at close range (doc_seen > 0.5 and doc_h > 0.35)
        doc_seen = obs[9]
        doc_h = obs[11]
        if doc_seen > 0.5 and doc_h > 0.35:
            reward += PENALTY_DOCTOR_PROXIMITY * (doc_h / 0.50)

        if not math.isfinite(reward):
            reward = 0.0
        reward = float(np.clip(reward, -50.0, 100.0))

        if not math.isfinite(self.cumulative_team_reward):
            self.cumulative_team_reward = 0.0

        info["cumulative_team_reward"] = self.cumulative_team_reward
        info["visual_sectors"] = obs_sectors.tolist()
        return obs, reward, done, info

    def teacher_action(self) -> np.ndarray:
        """Pure-vision teacher action for behavioral cloning warmup."""
        obs = self._get_obs()
        sectors = obs[0:5]
        pat_seen = obs[5] > 0.5
        pat_cx = obs[6]
        pat_h = obs[7]
        doc_seen = obs[9] > 0.5
        doc_cx = obs[10]
        doc_h = obs[11]

        # 1. Doctor Evasion: if doctor is right in front (Center sector / doc_h > 0.3)
        if doc_seen and doc_h > 0.25 and abs(doc_cx) < 0.25:
            # Turn away from doctor
            turn_dir = -1.0 if doc_cx > 0.0 else 1.0
            omega = turn_dir * 0.8
            v = 0.2
            a_v = (v / (0.5 * V_MAX)) - 1.0
            a_omega = omega / OMEGA_MAX
            return np.array([a_v, a_omega], dtype=np.float32)

        # 2. Patient Approach: steer towards patient bounding box
        if self.task_phase == TaskPhase.APPROACH_PATIENT and pat_seen:
            k_p = 2.0
            omega = float(np.clip(-k_p * pat_cx, -OMEGA_MAX, OMEGA_MAX))
            if pat_h >= TARGET_BOX_H:
                v = 0.0
            else:
                align_factor = max(0.0, 1.0 - abs(pat_cx) / 0.30)
                v = float(np.clip(0.60 * align_factor, 0.15, V_MAX))
            a_v = (v / (0.5 * V_MAX)) - 1.0
            a_omega = omega / OMEGA_MAX
            return np.array([a_v, a_omega], dtype=np.float32)

        # 3. Escort or Corridor Navigation: sector-based obstacle avoidance
        goal_angle = math.atan2(obs[18], obs[19])
        v_des = ESCORT_WALK_SPEED if self.task_phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF) else V_MAX
        omega_des = float(np.clip(goal_angle * 1.5, -OMEGA_MAX, OMEGA_MAX))

        # Check visual sectors: if Center (sector 2) blocked (h > 0.45)
        if sectors[2] > 0.45:
            # Steer towards clearer side (compare left vs right sectors)
            left_clear = max(sectors[0], sectors[1])
            right_clear = max(sectors[3], sectors[4])
            steer = +0.8 if left_clear < right_clear else -0.8
            omega_des = steer
            v_des = 0.20

        a_v = (v_des / (0.5 * V_MAX)) - 1.0
        a_omega = omega_des / OMEGA_MAX
        return np.array([a_v, a_omega], dtype=np.float32)
