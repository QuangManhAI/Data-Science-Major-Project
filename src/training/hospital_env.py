"""Hospital Shift RL Environment matching Godot `RLEnvBridge` (31-D obs, 2-D action, R_team).

Mirrors the exact 24m x 18m Floor-1 geometry (`sim_vision/godot/scenes/room.tscn`),
17-ray 200-deg LiDAR fan (`robot.gd`), Smart Floor-1 + Elevator + Staff routing
(`rl_env_bridge.gd`), and coupled elderly patient escort tether (0.48 m/s, 2.5m max tether).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from src.models.ppo_agent import OBS_DIM, OMEGA_MAX, V_MAX


class TaskPhase(IntEnum):
    SEARCH_PATIENT = 0
    APPROACH_PATIENT = 1
    GUIDE_ESCORT = 2
    CALL_STAFF = 3


# Shared Team Rewards (R_team) — identical to RLEnvBridge in GDScript
REWARD_ANSWER = 10.0
REWARD_CALL_STAFF = 20.0
REWARD_GUIDE_ARRIVED = 30.0
REWARD_PATIENT_ENGAGED = 2.0

# Dense shaping & safety constants
SHAPING_PROGRESS_SCALE = 1.5
BONUS_TETHER_GOOD = 0.03
PENALTY_TETHER_STRETCH = -0.08
PENALTY_STEP_TIME = -0.01
PENALTY_COLLISION = -5.0

# Physical thresholds (meters & m/s)
TETHER_IDEAL_M = 1.6
TETHER_MAX_M = 2.5
GOAL_ARRIVE_ROBOT_M = 1.4
GOAL_ARRIVE_PATIENT_M = 2.0
COLLISION_LIDAR_M = 0.35
PATIENT_TALK_DIST_M = 1.35

ELDERLY_PATIENT_SPEED = 0.48
ESCORT_WALK_SPEED = 0.52

LIDAR_NUM_RAYS = 17
LIDAR_FOV_DEG = 200.0
LIDAR_MAX_RANGE = 6.0

# Floor-1 Destination Anchors (matching RLEnvBridge.DESTINATION_POSITIONS)
DESTINATION_POSITIONS: dict[str, tuple[float, float]] = {
    "101": (-7.0, -5.5),
    "110": (1.0, -5.5),
    "112": (9.0, -5.5),
    "nha_thuoc": (-7.5, 5.5),
    "tiep_don": (0.0, 4.5),
    "bhyt": (0.0, 4.5),
    "thu_ngan": (0.0, 4.5),
    "cap_cuu": (-7.0, -5.5),
    "xet_nghiem": (1.0, -5.5),
    "nha_ve_sinh": (7.5, 5.5),
    "thang_may": (10.5, 0.0),
    "staff": (-1.8, 3.8),
}

FLOOR1_DIRECT_SLOTS = {
    "101",
    "110",
    "112",
    "nha_thuoc",
    "tiep_don",
    "bhyt",
    "thu_ngan",
    "cap_cuu",
    "xet_nghiem",
    "nha_ve_sinh",
    "thang_may",
}

# Patient request distribution in a hospital shift
PATIENT_SCENARIOS: list[tuple[str, str, int]] = [
    ("guide", "101", 1),
    ("guide", "110", 1),
    ("guide", "112", 1),
    ("guide", "nha_thuoc", 1),
    ("guide", "nha_ve_sinh", 1),
    ("guide", "bhyt", 1),
    ("guide", "301", 3),        # Floor 3 -> routes to Elevator (thang_may)
    ("guide", "204", 2),        # Floor 2 -> routes to Elevator (thang_may)
    ("call_staff", "", 1),      # Routes to nearest Medical Staff
    ("emergency", "101", 1),    # Routes to nearest Medical Staff
    ("answer", "", 1),          # Verbal answer (+10 R_team)
    ("repeat", "", 1),          # Verbal repeat (+10 R_team)
]


@dataclass(frozen=True)
class WallBox2D:
    """Axis-aligned 2D obstacle rectangle (center_x, center_z, half_w, half_d)."""

    cx: float
    cz: float
    hx: float
    hz: float

    def contains(self, x: float, z: float, margin: float = 0.0) -> bool:
        return (
            abs(x - self.cx) <= (self.hx + margin)
            and abs(z - self.cz) <= (self.hz + margin)
        )

    def ray_intersect(self, ox: float, oz: float, dx: float, dz: float, max_t: float) -> float:
        """Slab intersection returning distance in [0, max_t] or max_t if no hit."""
        tmin = 0.0
        tmax = max_t
        min_x, max_x = self.cx - self.hx, self.cx + self.hx
        min_z, max_z = self.cz - self.hz, self.cz + self.hz

        if abs(dx) < 1e-8:
            if ox < min_x or ox > max_x:
                return max_t
        else:
            tx1 = (min_x - ox) / dx
            tx2 = (max_x - ox) / dx
            if tx1 > tx2:
                tx1, tx2 = tx2, tx1
            tmin = max(tmin, tx1)
            tmax = min(tmax, tx2)
            if tmin > tmax:
                return max_t

        if abs(dz) < 1e-8:
            if oz < min_z or oz > max_z:
                return max_t
        else:
            tz1 = (min_z - oz) / dz
            tz2 = (max_z - oz) / dz
            if tz1 > tz2:
                tz1, tz2 = tz2, tz1
            tmin = max(tmin, tz1)
            tmax = min(tmax, tz2)
            if tmin > tmax:
                return max_t

        return tmin if 0.0 <= tmin <= max_t else max_t


# Exact 24m x 18m hospital walls from sim_vision/godot/scenes/room.tscn
HOSPITAL_WALLS: tuple[WallBox2D, ...] = (
    # Outer shell
    WallBox2D(0.0, -9.0, 12.1, 0.15),
    WallBox2D(0.0, 9.0, 12.1, 0.15),
    WallBox2D(-12.0, 0.0, 0.15, 9.1),
    WallBox2D(12.0, 0.0, 0.15, 9.1),
    # North corridor walls (Z = -2.0) with doorways at X = -8.5, 0.0, 8.5
    WallBox2D(-10.85, -2.0, 1.15, 0.09),
    WallBox2D(-4.25, -2.0, 3.05, 0.09),
    WallBox2D(4.25, -2.0, 3.05, 0.09),
    WallBox2D(10.85, -2.0, 1.15, 0.09),
    # South corridor walls (Z = +2.0) with doorways at X = -7.5, 0.0 (wide 3m), 7.5
    WallBox2D(-10.35, 2.0, 1.65, 0.09),
    WallBox2D(-4.0, 2.0, 2.3, 0.09),
    WallBox2D(4.0, 2.0, 2.3, 0.09),
    WallBox2D(10.35, 2.0, 1.65, 0.09),
    # North room dividers (X = -3.0, X = +5.0) & South dividers (X = -4.0, X = +4.0)
    WallBox2D(-3.0, -5.5, 0.09, 3.5),
    WallBox2D(5.0, -5.5, 0.09, 3.5),
    WallBox2D(-4.0, 5.5, 0.09, 3.5),
    WallBox2D(4.0, 5.5, 0.09, 3.5),
)


def resolve_destination_key(action: str, slot_key: str, floor_num: int = 1) -> str:
    """Smart Floor-1 routing matching RLEnvBridge.resolve_destination_key."""
    act = action.strip().lower()
    key = slot_key.strip().lower()
    if act in ("call_staff", "emergency"):
        return "staff"
    if act != "guide":
        return ""
    if floor_num >= 2:
        return "thang_may"
    if len(key) == 3 and key.isdigit() and key[0] in ("2", "3", "4"):
        return "thang_may"
    if key in ("bhyt", "thu_ngan"):
        return "tiep_don"
    if key in FLOOR1_DIRECT_SLOTS:
        return key
    return "tiep_don"


def signed_angle_rad(fwd_x: float, fwd_z: float, to_x: float, to_z: float) -> float:
    """Signed angle in [-pi, pi] from robot forward vector to target vector (positive = left)."""
    norm_t = math.hypot(to_x, to_z)
    if norm_t < 1e-6:
        return 0.0
    tx, tz = to_x / norm_t, to_z / norm_t
    dot = max(-1.0, min(1.0, fwd_x * tx + fwd_z * tz))
    cross_y = fwd_z * tx - fwd_x * tz
    return math.atan2(cross_y, dot)


class HospitalShiftEnv:
    """Continuous 180s Hospital Shift RL Environment (Stage 1 & Stage 2 curriculum)."""

    def __init__(
        self,
        curriculum_stage: int = 1,
        shift_duration_s: float = 180.0,
        dt: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.curriculum_stage = int(curriculum_stage)
        self.shift_duration_s = float(shift_duration_s)
        self.dt = float(dt)
        self.max_steps = int(round(self.shift_duration_s / self.dt))
        self.rng = np.random.default_rng(seed)

        self.robot_x = 0.0
        self.robot_z = 0.0
        self.robot_yaw = 0.0
        self.robot_v = 0.0
        self.robot_omega = 0.0

        self.patient_x = 0.0
        self.patient_z = 0.0
        self.scenario_action = "guide"
        self.scenario_slot = "101"
        self.scenario_floor = 1

        self.goal_key = "101"
        self.goal_x = 0.0
        self.goal_z = 0.0
        self.task_phase = TaskPhase.SEARCH_PATIENT

        # Patrolling doctor in central corridor (Stage 2 dynamic obstacle with pardon behavior)
        self.doctor_x = -6.0
        self.doctor_z = -0.6
        self.doctor_dir = 1.0

        self.step_count = 0
        self.cumulative_team_reward = 0.0
        self.patients_helped = 0
        self.collisions = 0

    def set_stage(self, stage: int) -> None:
        self.curriculum_stage = int(stage)

    def _sample_free_point(self, lobby_or_corridor_only: bool = False) -> tuple[float, float]:
        for _ in range(200):
            if lobby_or_corridor_only:
                x = float(self.rng.uniform(-8.5, 8.5))
                z = float(self.rng.uniform(-1.3, 6.5))
            else:
                x = float(self.rng.uniform(-10.5, 10.5))
                z = float(self.rng.uniform(-7.5, 7.5))
            if all(not w.contains(x, z, margin=0.65) for w in HOSPITAL_WALLS):
                return x, z
        return 0.0, 0.0

    def _spawn_next_patient(self) -> None:
        stage1 = self.curriculum_stage <= 1
        for _ in range(100):
            px, pz = self._sample_free_point(lobby_or_corridor_only=stage1)
            if math.hypot(px - self.robot_x, pz - self.robot_z) >= 3.0:
                self.patient_x, self.patient_z = px, pz
                break
        else:
            self.patient_x, self.patient_z = 0.0, 4.0

        if stage1:
            stage1_scenarios = [
                ("guide", "tiep_don", 1),
                ("guide", "nha_thuoc", 1),
                ("guide", "thang_may", 2),
                ("call_staff", "", 1),
                ("answer", "", 1),
            ]
            idx = int(self.rng.integers(0, len(stage1_scenarios)))
            self.scenario_action, self.scenario_slot, self.scenario_floor = stage1_scenarios[idx]
        else:
            idx = int(self.rng.integers(0, len(PATIENT_SCENARIOS)))
            self.scenario_action, self.scenario_slot, self.scenario_floor = PATIENT_SCENARIOS[idx]

        # Robot seeks patient first
        self.task_phase = TaskPhase.SEARCH_PATIENT
        self.goal_key = "patient"
        self.goal_x = self.patient_x
        self.goal_z = self.patient_z

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.cumulative_team_reward = 0.0
        self.patients_helped = 0
        self.collisions = 0

        self.robot_x, self.robot_z = self._sample_free_point(
            lobby_or_corridor_only=(self.curriculum_stage <= 1)
        )
        self.robot_yaw = float(self.rng.uniform(-math.pi, math.pi))
        self.robot_v = 0.0
        self.robot_omega = 0.0

        self.doctor_x = -6.0
        self.doctor_z = -0.6
        self.doctor_dir = 1.0

        self._spawn_next_patient()
        return self._get_obs()

    def compute_lidar(self) -> np.ndarray:
        """Cast 17 horizontal rays across 200-deg FOV up to 6.0m."""
        readings = np.ones(LIDAR_NUM_RAYS, dtype=np.float32)
        half_fov = math.radians(LIDAR_FOV_DEG * 0.5)
        step_angle = math.radians(LIDAR_FOV_DEG) / float(max(LIDAR_NUM_RAYS - 1, 1))

        for i in range(LIDAR_NUM_RAYS):
            rel_angle = -half_fov + i * step_angle
            ray_yaw = self.robot_yaw - rel_angle
            dx = -math.sin(ray_yaw)
            dz = -math.cos(ray_yaw)
            best_d = LIDAR_MAX_RANGE
            for w in HOSPITAL_WALLS:
                d = w.ray_intersect(self.robot_x, self.robot_z, dx, dz, best_d)
                if d < best_d:
                    best_d = d
            if self.curriculum_stage >= 2:
                # Check patrolling doctor cylinder (radius 0.35m)
                oc_x = self.doctor_x - self.robot_x
                oc_z = self.doctor_z - self.robot_z
                proj = oc_x * dx + oc_z * dz
                if 0.0 < proj < best_d:
                    perp_sq = (oc_x * oc_x + oc_z * oc_z) - proj * proj
                    if perp_sq < 0.35 * 0.35:
                        best_d = max(0.0, proj - 0.35)
            readings[i] = float(np.clip(best_d / LIDAR_MAX_RANGE, 0.0, 1.0))
        return readings

    def _compute_yolo_det(self) -> tuple[float, float, float, float]:
        """Simulate YOLOv8 person detection in robot camera 70-deg horizontal FOV."""
        dx = self.patient_x - self.robot_x
        dz = self.patient_z - self.robot_z
        dist = math.hypot(dx, dz)
        if dist < 0.2 or dist > 12.0:
            return 0.0, 0.0, 0.0, 0.0
        fwd_x = -math.sin(self.robot_yaw)
        fwd_z = -math.cos(self.robot_yaw)
        angle = signed_angle_rad(fwd_x, fwd_z, dx, dz)
        half_cam_fov = math.radians(35.0)
        if abs(angle) > half_cam_fov:
            return 0.0, 0.0, 0.0, 0.0
        # Check line-of-sight against hospital walls
        ux, uz = dx / dist, dz / dist
        for w in HOSPITAL_WALLS:
            if w.ray_intersect(self.robot_x, self.robot_z, ux, uz, dist) < dist - 0.1:
                return 0.0, 0.0, 0.0, 0.0
        cx_offset = float(np.clip(-angle / (2.0 * half_cam_fov), -0.5, 0.5))
        h = float(np.clip(0.85 / max(dist, 0.75), 0.05, 1.0))
        conf = float(np.clip(0.95 - 0.03 * dist, 0.45, 0.98))
        return 1.0, cx_offset, h, conf

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        lidar = self.compute_lidar()
        obs[0:17] = lidar

        has_det, cx_off, h, conf = self._compute_yolo_det()
        if self.task_phase == TaskPhase.SEARCH_PATIENT and has_det > 0.5:
            self.task_phase = TaskPhase.APPROACH_PATIENT

        obs[17] = has_det
        obs[18] = cx_off
        obs[19] = h
        obs[20] = conf

        phase_idx = int(np.clip(int(self.task_phase), 0, 3))
        obs[21 + phase_idx] = 1.0

        to_gx = self.goal_x - self.robot_x
        to_gz = self.goal_z - self.robot_z
        d_goal = math.hypot(to_gx, to_gz)
        fwd_x = -math.sin(self.robot_yaw)
        fwd_z = -math.cos(self.robot_yaw)
        angle_goal = signed_angle_rad(fwd_x, fwd_z, to_gx, to_gz)

        obs[25] = float(np.clip(d_goal / 25.0, 0.0, 1.0))
        obs[26] = math.sin(angle_goal)
        obs[27] = math.cos(angle_goal)

        d_tether = (
            math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
            if self.task_phase == TaskPhase.GUIDE_ESCORT
            else 0.0
        )
        obs[28] = float(np.clip(d_tether / 5.0, 0.0, 1.0))
        obs[29] = float(np.clip(self.robot_v / V_MAX, -1.0, 1.0))
        obs[30] = float(np.clip(self.robot_omega / OMEGA_MAX, -1.0, 1.0))
        return obs

    def _route_through_corridor_goal(self, target_x: float, target_z: float) -> tuple[float, float]:
        """Provide doorway/corridor sub-goal when robot and destination are in different rooms."""
        # If robot is inside a North room (z < -2.1) and target is outside that room
        if self.robot_z < -2.1 and (target_z > -1.8 or abs(target_x - self.robot_x) > 3.5):
            if self.robot_x < -3.0:
                return -8.5, -1.2
            if self.robot_x < 5.0:
                return 0.0, -1.2
            return 8.5, -1.2
        # If robot is inside a South room (z > 2.1) and target is outside that room
        if self.robot_z > 2.1 and (target_z < 1.8 or abs(target_x - self.robot_x) > 3.5):
            if self.robot_x < -4.0:
                return -7.5, 1.2
            if self.robot_x <= 4.0:
                return 0.0, 1.2
            return 7.5, 1.2
        # If robot is in corridor (-2.1 <= z <= 2.1) and target is in a North room (z < -2.2)
        if -2.1 <= self.robot_z <= 2.1 and target_z < -2.2:
            door_x = -8.5 if target_x < -3.0 else (0.0 if target_x < 5.0 else 8.5)
            if abs(self.robot_x - door_x) > 0.9:
                return door_x, -1.0
        # If robot is in corridor (-2.1 <= z <= 2.1) and target is in a South room (z > 2.2)
        if -2.1 <= self.robot_z <= 2.1 and target_z > 2.2:
            door_x = -7.5 if target_x < -4.0 else (0.0 if target_x <= 4.0 else 7.5)
            if abs(self.robot_x - door_x) > 0.9:
                return door_x, 1.0
        return target_x, target_z

    def step(self, action: np.ndarray | list[float] | tuple[float, float]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self.step_count += 1
        a_v = float(np.clip(action[0], -1.0, 1.0))
        a_omega = float(np.clip(action[1], -1.0, 1.0))

        v = float(np.clip((a_v + 1.0) * 0.5 * V_MAX, 0.0, V_MAX))
        omega = float(np.clip(a_omega * OMEGA_MAX, -OMEGA_MAX, OMEGA_MAX))

        # Assistive navigational prior blended with policy residual so PPO learns smooth
        # obstacle avoidance + elderly tether speed regulation rapidly
        final_dest_x, final_dest_z = (
            (self.patient_x, self.patient_z)
            if self.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT)
            else DESTINATION_POSITIONS.get(self.goal_key, (0.0, 4.5))
        )
        sub_gx, sub_gz = self._route_through_corridor_goal(final_dest_x, final_dest_z)
        self.goal_x, self.goal_z = sub_gx, sub_gz

        prev_dist_final = math.hypot(final_dest_x - self.robot_x, final_dest_z - self.robot_z)

        # Integrate robot kinematics with wall collision check
        self.robot_yaw = ((self.robot_yaw + omega * self.dt + math.pi) % (2.0 * math.pi)) - math.pi
        nx = self.robot_x - math.sin(self.robot_yaw) * v * self.dt
        nz = self.robot_z - math.cos(self.robot_yaw) * v * self.dt

        hit_wall = any(w.contains(nx, nz, margin=0.32) for w in HOSPITAL_WALLS)
        if not hit_wall and abs(nx) <= 11.6 and abs(nz) <= 8.6:
            self.robot_x, self.robot_z = nx, nz
            self.robot_v = v
        else:
            self.robot_v = 0.0
        self.robot_omega = omega

        # Step patrolling doctor in Stage 2 (with pardon behavior when robot approaches)
        if self.curriculum_stage >= 2:
            if math.hypot(self.robot_x - self.doctor_x, self.robot_z - self.doctor_z) < 1.8:
                # Doctor pardons (steps aside toward z = -1.3)
                self.doctor_z += (-1.3 - self.doctor_z) * min(1.0, 2.0 * self.dt)
            else:
                self.doctor_z += (-0.6 - self.doctor_z) * min(1.0, 2.0 * self.dt)
                self.doctor_x += self.doctor_dir * 0.55 * self.dt
                if self.doctor_x > 6.0:
                    self.doctor_dir = -1.0
                elif self.doctor_x < -6.0:
                    self.doctor_dir = 1.0

        # Step escorted elderly patient when in GUIDE_ESCORT phase
        if self.task_phase == TaskPhase.GUIDE_ESCORT:
            dx_p = self.robot_x - self.patient_x
            dz_p = self.robot_z - self.patient_z
            dist_p = math.hypot(dx_p, dz_p)
            if dist_p > 1.0:
                step_p = min(ELDERLY_PATIENT_SPEED * self.dt, dist_p - 0.95)
                px_new = self.patient_x + (dx_p / dist_p) * step_p
                pz_new = self.patient_z + (dz_p / dist_p) * step_p
                if all(not w.contains(px_new, pz_new, margin=0.28) for w in HOSPITAL_WALLS):
                    self.patient_x, self.patient_z = px_new, pz_new

        curr_dist_final = math.hypot(final_dest_x - self.robot_x, final_dest_z - self.robot_z)
        d_tether = (
            math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
            if self.task_phase == TaskPhase.GUIDE_ESCORT
            else 0.0
        )
        lidar = self.compute_lidar()
        min_lidar_m = float(np.min(lidar)) * LIDAR_MAX_RANGE

        # Check task phase transitions & shared team reward events
        event = ""
        if self.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
            if curr_dist_final <= PATIENT_TALK_DIST_M:
                resolved = resolve_destination_key(
                    self.scenario_action, self.scenario_slot, self.scenario_floor
                )
                if self.scenario_action == "guide":
                    event = "engaged"
                    self.task_phase = TaskPhase.GUIDE_ESCORT
                    self.goal_key = resolved
                elif self.scenario_action in ("call_staff", "emergency"):
                    event = "engaged"
                    self.task_phase = TaskPhase.CALL_STAFF
                    self.goal_key = "staff"
                else:
                    event = "answer"
                    self.patients_helped += 1
                    self._spawn_next_patient()
        elif self.task_phase == TaskPhase.GUIDE_ESCORT:
            dest_x, dest_z = DESTINATION_POSITIONS.get(self.goal_key, (0.0, 4.5))
            d_robot_dest = math.hypot(dest_x - self.robot_x, dest_z - self.robot_z)
            d_patient_dest = math.hypot(dest_x - self.patient_x, dest_z - self.patient_z)
            if d_robot_dest <= GOAL_ARRIVE_ROBOT_M and d_patient_dest <= GOAL_ARRIVE_PATIENT_M:
                event = "guide_completed"
                self.patients_helped += 1
                self._spawn_next_patient()
        elif self.task_phase == TaskPhase.CALL_STAFF:
            dest_x, dest_z = DESTINATION_POSITIONS["staff"]
            if math.hypot(dest_x - self.robot_x, dest_z - self.robot_z) <= GOAL_ARRIVE_ROBOT_M:
                event = "call_staff"
                self.patients_helped += 1
                self._spawn_next_patient()

        # Compute step reward matching RLEnvBridge.compute_step_reward
        r_team = 0.0
        if event == "answer":
            r_team = REWARD_ANSWER
        elif event == "call_staff":
            r_team = REWARD_CALL_STAFF
        elif event == "guide_completed":
            r_team = REWARD_GUIDE_ARRIVED
        elif event == "engaged":
            r_team = REWARD_PATIENT_ENGAGED

        r_progress = SHAPING_PROGRESS_SCALE * (prev_dist_final - curr_dist_final)
        r_tether = 0.0
        if self.task_phase == TaskPhase.GUIDE_ESCORT:
            if d_tether > TETHER_MAX_M:
                r_progress = 0.0
                r_tether = PENALTY_TETHER_STRETCH
            elif 0.9 <= d_tether <= 2.2:
                r_tether = BONUS_TETHER_GOOD

        r_safety = PENALTY_STEP_TIME
        collided = hit_wall or (min_lidar_m < COLLISION_LIDAR_M)
        if collided:
            r_safety += PENALTY_COLLISION
            self.collisions += 1

        # Directional alignment bonus to encourage facing the active sub-goal
        fwd_x = -math.sin(self.robot_yaw)
        fwd_z = -math.cos(self.robot_yaw)
        angle_sub = signed_angle_rad(fwd_x, fwd_z, self.goal_x - self.robot_x, self.goal_z - self.robot_z)
        r_align = 0.04 * math.cos(angle_sub) if self.robot_v > 0.05 else 0.0

        total_reward = float(r_team + r_progress + r_tether + r_safety + r_align)
        self.cumulative_team_reward += r_team

        done = self.step_count >= self.max_steps
        info = {
            "r_team": r_team,
            "cumulative_team_reward": self.cumulative_team_reward,
            "patients_helped": self.patients_helped,
            "collisions": self.collisions,
            "task_phase": int(self.task_phase),
            "d_tether": d_tether,
            "event": event,
        }
        return self._get_obs(), total_reward, done, info

    def teacher_action(self) -> np.ndarray:
        """Compute LiDAR-aware teacher action in [-1, 1]^2 matching RLController."""
        final_dest_x, final_dest_z = (
            (self.patient_x, self.patient_z)
            if self.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT)
            else DESTINATION_POSITIONS.get(self.goal_key, (0.0, 4.5))
        )
        gx, gz = self._route_through_corridor_goal(final_dest_x, final_dest_z)
        dist = math.hypot(gx - self.robot_x, gz - self.robot_z)
        d_tether = (
            math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
            if self.task_phase == TaskPhase.GUIDE_ESCORT
            else 0.0
        )
        if self.task_phase == TaskPhase.GUIDE_ESCORT and d_tether > TETHER_MAX_M:
            return np.array([-1.0, 0.0], dtype=np.float32)

        fwd_x = -math.sin(self.robot_yaw)
        fwd_z = -math.cos(self.robot_yaw)
        angle = signed_angle_rad(fwd_x, fwd_z, gx - self.robot_x, gz - self.robot_z)

        lidar = self.compute_lidar()
        mid = LIDAR_NUM_RAYS // 2
        left_clear = float(np.sum(lidar[:mid]))
        right_clear = float(np.sum(lidar[mid + 1 :]))
        center_min = float(min(lidar[mid - 1], lidar[mid], lidar[mid + 1]))
        repel = 0.0
        if center_min < 0.35:
            repel = 0.9 if left_clear >= right_clear else -0.9

        omega = float(np.clip(1.6 * angle + repel, -OMEGA_MAX, OMEGA_MAX))
        max_v = ESCORT_WALK_SPEED if self.task_phase == TaskPhase.GUIDE_ESCORT else V_MAX
        v = float(np.clip(0.55 * dist, 0.0, max_v))
        if abs(angle) > 0.65:
            v *= 0.25

        a_v = float(np.clip(2.0 * (v / V_MAX) - 1.0, -1.0, 1.0))
        a_omega = float(np.clip(omega / OMEGA_MAX, -1.0, 1.0))
        return np.array([a_v, a_omega], dtype=np.float32)
