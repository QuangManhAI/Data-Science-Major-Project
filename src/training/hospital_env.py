"""Hospital Shift RL Environment matching Godot `RLEnvBridge` (31-D obs, 2-D action, R_team).

Mirrors the exact 24m x 18m Floor-1 geometry (`sim_vision/godot/scenes/room.tscn`),
17-ray 200-deg LiDAR fan (`robot.gd`), Smart Floor-1 + Elevator + Staff routing
(`rl_env_bridge.gd`), and coupled elderly patient escort tether (0.48 m/s, 2.5m max tether).
"""

from __future__ import annotations

import collections
import json
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
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
REWARD_ELEVATOR_TRANSIT = 10.0

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
COLLISION_LIDAR_M = 0.40
PATIENT_TALK_DIST_M = 1.35

ELDERLY_PATIENT_SPEED = 0.48
# Escort breadcrumb following (match RLEnvBridge.ESCORT_* in sim_vision/godot/scripts/rl_env_bridge.gd)
ESCORT_STOP_DIST_M = 0.9       # patient stops this close behind the robot
ESCORT_CRUMB_SPACING_M = 0.25  # robot path is sampled every 0.25 m
ESCORT_CRUMB_REACH_M = 0.3     # crumb counts as reached within 0.3 m
ESCORT_TRAIL_JUMP_M = 5.0      # robot teleported (elevator) -> restart trail
STAGNATION_PROGRESS_M = 1.0    # getting this much closer to the active goal resets the stagnation clock
ESCORT_WALK_SPEED = 0.52

LIDAR_NUM_RAYS = 17
LIDAR_FOV_DEG = 200.0
LIDAR_MAX_RANGE = 6.0

# Floor-1 Destination Anchors (matching Room.ROOMS[*].waypoint in sim_vision/godot/scripts/room.gd)
DESTINATION_POSITIONS: dict[str, tuple[float, float]] = {
    "101": (-8.0, -5.2),
    "110": (0.0, -5.2),
    "112": (8.0, -5.2),
    "nha_thuoc": (-9.0, 5.2),
    "tiep_don": (0.0, 5.2),
    "bhyt": (-1.5, 6.2),
    "thu_ngan": (1.5, 6.2),
    "cap_cuu": (-8.0, -5.2),
    "xet_nghiem": (0.0, -5.2),
    "nha_ve_sinh": (9.5, 6.5),
    "thang_may": (9.0, 5.2),
    "staff": (-1.8, 3.8),
    # Floor 2 Anchors (offset X = +50.0m)
    "201": (42.0, -5.2),
    "204": (42.0, -5.2),
    "205": (50.0, -5.2),
    "210": (58.0, -5.2),
    "220": (41.0, 5.2),
    "sanh_t2": (50.0, 5.2),
    "wc_t2": (59.5, 6.5),
    "thang_may_t2": (59.0, 4.6),
}

FLOOR2_DIRECT_SLOTS = {
    "201",
    "204",
    "205",
    "210",
    "220",
    "sanh_t2",
    "wc_t2",
    "thang_may_t2",
    "hanh_lang_t2",
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
    ("guide", "205", 2),        # Floor 2 Gastroenterology -> Elevator
    ("guide", "210", 2),        # Floor 2 ENT -> Elevator
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


def floor_of(x: float) -> int:
    """Floor 2 is laid out side by side at X + 50 m."""
    return 2 if x >= 25.0 else 1


# Walls + furniture of both floors, exported from Godot (sim_vision/godot/tools/export_nav_layout.gd).
# Re-export after editing room.tscn / room_floor2.tscn so training and Godot stay on the same map.
LAYOUT_PATH = Path(__file__).with_name("hospital_layout.json")
_LAYOUT_DATA: dict[str, Any] = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
# Full-mode NPCs (seated patients, nurse, patrolling doctor) as placed in room.tscn
HOSPITAL_NPCS: tuple[dict[str, Any], ...] = tuple(_LAYOUT_DATA.get("npcs", []))

# NPC behaviour (match HospitalNpc in sim_vision/godot/scripts/hospital_npc.gd)
NPC_PARDON_TRIGGER_DIST = 1.4
NPC_PARDON_CLEAR_DIST = 1.85
NPC_PARDON_SIDESTEP_M = 0.55
NPC_PARDON_MIN_ROBOT_V = 0.05
NPC_PARDON_LANE_HALF_W = 0.9
NPC_PARDON_CLEARANCE_M = 0.35


def _load_layout(data: dict[str, Any]) -> tuple[tuple[WallBox2D, ...], list[str]]:
    walls, names = [], []
    for r in data["rects"]:
        walls.append(WallBox2D(
            (r["x0"] + r["x1"]) / 2.0, (r["z0"] + r["z1"]) / 2.0,
            (r["x1"] - r["x0"]) / 2.0, (r["z1"] - r["z0"]) / 2.0,
        ))
        names.append(r["name"])
    return tuple(walls), names


HOSPITAL_WALLS, HOSPITAL_WALL_NAMES = _load_layout(_LAYOUT_DATA)
FLOOR1_WALLS: tuple[WallBox2D, ...] = tuple(w for w in HOSPITAL_WALLS if floor_of(w.cx) == 1)
FLOOR2_WALLS: tuple[WallBox2D, ...] = tuple(w for w in HOSPITAL_WALLS if floor_of(w.cx) == 2)


def _boxes(walls) -> np.ndarray:
    return np.array([(w.cx - w.hx, w.cx + w.hx, w.cz - w.hz, w.cz + w.hz) for w in walls], dtype=np.float64)


# Vectorised [N, 4] boxes (min_x, max_x, min_z, max_z) per floor: everything (LiDAR, motion) and
# structural walls only (camera line of sight sees over furniture)
_OBSTACLES: dict[int, np.ndarray] = {1: _boxes(FLOOR1_WALLS), 2: _boxes(FLOOR2_WALLS)}
_STRUCTURE: dict[int, np.ndarray] = {
    f: _boxes([w for w, n in zip(HOSPITAL_WALLS, HOSPITAL_WALL_NAMES)
               if floor_of(w.cx) == f and ("Shell" in n or "InteriorWalls" in n)])
    for f in (1, 2)
}


def is_blocked(x: float, z: float, margin: float) -> bool:
    """Point within `margin` of any wall / furniture box on its floor."""
    b = _OBSTACLES[floor_of(x)]
    return bool(np.any((x >= b[:, 0] - margin) & (x <= b[:, 1] + margin) & (z >= b[:, 2] - margin) & (z <= b[:, 3] + margin)))


def cast_rays(ox: float, oz: float, dxs: np.ndarray, dzs: np.ndarray, max_t: float, boxes: np.ndarray) -> np.ndarray:
    """Slab ray/box test for R rays x N boxes; nearest hit distance per ray (max_t if none)."""
    dxs = dxs[:, None]
    dzs = dzs[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx1 = (boxes[None, :, 0] - ox) / dxs
        tx2 = (boxes[None, :, 1] - ox) / dxs
        tz1 = (boxes[None, :, 2] - oz) / dzs
        tz2 = (boxes[None, :, 3] - oz) / dzs
    in_x = (ox >= boxes[None, :, 0]) & (ox <= boxes[None, :, 1])
    in_z = (oz >= boxes[None, :, 2]) & (oz <= boxes[None, :, 3])
    par_x = np.abs(dxs) < 1e-8
    par_z = np.abs(dzs) < 1e-8
    txmin = np.where(par_x, np.where(in_x, -np.inf, np.inf), np.minimum(tx1, tx2))
    txmax = np.where(par_x, np.where(in_x, np.inf, -np.inf), np.maximum(tx1, tx2))
    tzmin = np.where(par_z, np.where(in_z, -np.inf, np.inf), np.minimum(tz1, tz2))
    tzmax = np.where(par_z, np.where(in_z, np.inf, -np.inf), np.maximum(tz1, tz2))
    tmin = np.maximum(np.maximum(txmin, tzmin), 0.0)
    tmax = np.minimum(np.minimum(txmax, tzmax), max_t)
    d = np.where(tmin <= tmax, tmin, max_t)
    return d.min(axis=1) if d.shape[1] else np.full(d.shape[0], max_t)


def _derive_doors() -> dict[tuple[int, str], tuple[tuple[float, ...], tuple[float, ...]]]:
    """Doorway centres = gaps between corridor wall segments; room columns split by divider walls."""
    doors = {}
    for floor in (1, 2):
        for side, z_line in (("north", -2.0), ("south", 2.0)):
            segs = sorted(
                (w.cx - w.hx, w.cx + w.hx) for w, n in zip(HOSPITAL_WALLS, HOSPITAL_WALL_NAMES)
                if floor_of(w.cx) == floor and "InteriorWalls" in n and abs(w.cz - z_line) < 0.2 and w.hx > w.hz
            )
            centres = tuple((a[1] + b[0]) / 2.0 for a, b in zip(segs, segs[1:]) if b[0] - a[1] > 0.8)
            dividers = tuple(sorted(
                w.cx for w, n in zip(HOSPITAL_WALLS, HOSPITAL_WALL_NAMES)
                if floor_of(w.cx) == floor and "InteriorWalls" in n and w.hz > w.hx and (w.cz < 0) == (side == "north")
            ))
            assert len(centres) == len(dividers) + 1, f"floor {floor} {side}: {centres} vs dividers {dividers}"
            doors[(floor, side)] = (centres, dividers)
    return doors


_DOORS = _derive_doors()


def _door_x(x: float, side: str) -> float:
    """Doorway of the room column containing global x (north = z < -2 rooms, south = z > +2 rooms)."""
    centres, dividers = _DOORS[(floor_of(x), side)]
    return centres[sum(1 for d in dividers if d < x)]


def north_door_x(x: float) -> float:
    return _door_x(x, "north")


def south_door_x(x: float) -> float:
    return _door_x(x, "south")




ELEVATOR_POSITIONS: dict[int, tuple[float, float]] = {
    1: DESTINATION_POSITIONS["thang_may"],
    2: DESTINATION_POSITIONS["thang_may_t2"],
}


def resolve_final_target(action: str, slot_key: str, dest_floor: int = 1) -> str:
    act = action.strip().lower()
    key = slot_key.strip().lower()
    if act in ("call_staff", "emergency"):
        return "staff"
    if act != "guide":
        return ""
    if key in ("bhyt", "thu_ngan", "uu_tien"):
        return "tiep_don"
    if key in FLOOR2_DIRECT_SLOTS:
        return key
    if len(key) == 3 and key.isdigit() and key[0] == "2":
        return "205"
    if key in FLOOR1_DIRECT_SLOTS:
        return key
    if dest_floor >= 2:
        return "sanh_t2"
    return "tiep_don"


def resolve_destination_key(action: str, slot_key: str, dest_floor: int = 1, current_floor: int = 1, floor_num: int | None = None) -> str:
    if floor_num is not None:
        dest_floor = floor_num
    """Smart routing matching RLEnvBridge.resolve_destination_key."""
    act = action.strip().lower()
    key = slot_key.strip().lower()
    if act in ("call_staff", "emergency"):
        return "staff"
    if act != "guide":
        return ""
    final_target = resolve_final_target(act, key, dest_floor)
    target_floor = 2 if (final_target in FLOOR2_DIRECT_SLOTS or dest_floor >= 2 or (len(key) == 3 and key.isdigit() and key[0] in ("2", "3", "4"))) else 1

    if current_floor == 1:
        if target_floor >= 2:
            return "thang_may"
        return final_target
    else:  # current_floor == 2
        if target_floor == 1:
            return "thang_may_t2"
        return final_target


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
    """Continuous 180s Hospital Shift RL Environment (Solo Patient Focus or Full Stage 1/2 Multi-NPC)."""

    def __init__(
        self,
        curriculum_stage: int = 1,
        shift_duration_s: float = 180.0,
        dt: float = 0.2,
        seed: int = 42,
        env_mode: str = "solo",
        stagnation_timeout_s: float = 0.0,
        stagnation_penalty: float = -10.0,
    ) -> None:
        self.curriculum_stage = int(curriculum_stage)
        self.shift_duration_s = float(shift_duration_s)
        self.dt = float(dt)
        self.env_mode = str(env_mode).lower().strip()
        # "full" = solo task flow + the scene's NPCs (seated patients, nurse, patrolling doctor) as live obstacles
        self.solo_like = self.env_mode in ("solo", "full")
        self.npcs: list[dict[str, Any]] = []
        self.stagnation_timeout_s = float(stagnation_timeout_s)
        self.stagnation_penalty = float(stagnation_penalty)
        self.max_stagnation_steps = (
            int(round(self.stagnation_timeout_s / self.dt))
            if self.stagnation_timeout_s > 0.0
            else 0
        )
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

        # Patrolling doctor in central corridor (only active in Full mode Stage 2)
        self.doctor_active = not self.solo_like
        self.doctor_x = -6.0
        self.doctor_z = -0.6
        self.doctor_dir = 1.0

        self.step_count = 0
        self.steps_since_task = 0
        self.cumulative_team_reward = 0.0
        self.patients_helped = 0
        self.collisions = 0
        self.escort_trail: collections.deque[tuple[float, float]] = collections.deque(maxlen=400)
        self.best_goal_dist: float | None = None

    def set_stage(self, stage: int) -> None:
        self.curriculum_stage = int(stage)

    def _near_npc(self, x: float, z: float, clearance: float) -> bool:
        return any(abs(x - n["x"]) <= n["half_x"] + clearance and abs(z - n["z"]) <= n["half_z"] + clearance for n in self.npcs)

    def _npc_boxes(self) -> np.ndarray:
        if not self.npcs:
            return np.zeros((0, 4))
        return np.array([(n["x"] - n["half_x"], n["x"] + n["half_x"], n["z"] - n["half_z"], n["z"] + n["half_z"]) for n in self.npcs])

    def _obstacles(self) -> np.ndarray:
        """Static walls/furniture of the robot's floor plus live NPC boxes (NPCs are all on floor 1)."""
        floor = floor_of(self.robot_x)
        if floor == 1 and self.npcs:
            return np.concatenate([_OBSTACLES[1], self._npc_boxes()])
        return _OBSTACLES[floor]

    def _step_npcs(self) -> None:
        """Waypoint patrols + pardon side-step, mirroring HospitalNpc._physics_process."""
        fwd_x, fwd_z = -math.sin(self.robot_yaw), -math.cos(self.robot_yaw)
        for n in self.npcs:
            if n["seated"]:
                continue
            d_robot = math.hypot(n["x"] - self.robot_x, n["z"] - self.robot_z)
            if n["pardon_enabled"] and not n["pardoning"] and d_robot <= NPC_PARDON_TRIGGER_DIST:
                rel_x, rel_z = n["x"] - self.robot_x, n["z"] - self.robot_z
                ahead = rel_x * fwd_x + rel_z * fwd_z
                lateral = rel_x * (-fwd_z) + rel_z * fwd_x
                if abs(self.robot_v) >= NPC_PARDON_MIN_ROBOT_V and ahead > -0.2 and abs(lateral) < NPC_PARDON_LANE_HALF_W:
                    side = 1.0 if lateral >= 0.0 else -1.0
                    for sgn in (side, -side):
                        tx = n["x"] + (-fwd_z) * NPC_PARDON_SIDESTEP_M * sgn
                        tz = n["z"] + fwd_x * NPC_PARDON_SIDESTEP_M * sgn
                        if not is_blocked(tx, tz, NPC_PARDON_CLEARANCE_M):
                            n["pardoning"], n["target"] = True, (tx, tz)
                            break
            elif n["pardoning"] and d_robot >= NPC_PARDON_CLEAR_DIST:
                n["pardoning"] = False
            if n["pardoning"]:
                target = n["target"]
            elif n["walks"]:
                target = tuple(n["waypoints"][n["wp_index"]])
                if math.hypot(target[0] - n["x"], target[1] - n["z"]) <= 0.05:
                    n["wp_index"] = (n["wp_index"] + 1) % len(n["waypoints"])
                    target = tuple(n["waypoints"][n["wp_index"]])
            else:
                target = n["home"]
            dx, dz = target[0] - n["x"], target[1] - n["z"]
            d = math.hypot(dx, dz)
            step = min(n["walk_speed"] * self.dt, d)
            if d > 1e-6:
                n["x"] += dx / d * step
                n["z"] += dz / d * step

    def _sample_patient_spawn_solo(self) -> tuple[float, float]:
        """Sample free coordinates strictly in public corridors or lobby on robot's current floor."""
        x_base = 50.0 if self.robot_x >= 25.0 else 0.0
        for _ in range(250):
            if self.rng.random() < 0.60:
                # Central and connecting corridors
                x = float(self.rng.uniform(-10.0, 10.0)) + x_base
                z = float(self.rng.uniform(-1.3, 1.3))
            else:
                # Main lobby waiting & reception area
                x = float(self.rng.uniform(-3.2, 3.2)) + x_base
                z = float(self.rng.uniform(2.2, 6.5))
            if not is_blocked(x, z, 0.65) and not self._near_npc(x, z, 0.8):
                return x, z
        return 0.0 + x_base, 4.0

    def _sample_solo_scenario(self) -> None:
        """Sample scenario with 70% escort, 15% question answer, 15% call doctor."""
        r_val = float(self.rng.uniform(0.0, 1.0))
        if r_val < 0.70:
            # 70% Guide / Escort across Floor 1 & Floor 2
            dest_choices = [
                ("guide", "101", 1),
                ("guide", "110", 1),
                ("guide", "112", 1),
                ("guide", "nha_thuoc", 1),
                ("guide", "tiep_don", 1),
                ("guide", "nha_ve_sinh", 1),
                ("guide", "201", 2),
                ("guide", "205", 2),
                ("guide", "210", 2),
                ("guide", "220", 2),
                ("guide", "wc_t2", 2),
            ]
            idx = int(self.rng.integers(0, len(dest_choices)))
            self.scenario_action, self.scenario_slot, self.scenario_floor = dest_choices[idx]
        elif r_val < 0.85:
            # 15% Answer verbal question
            self.scenario_action, self.scenario_slot, self.scenario_floor = ("answer", "", 1)
        else:
            # 15% Call medical staff / doctor
            self.scenario_action, self.scenario_slot, self.scenario_floor = ("call_staff", "", 1)

    def _sample_free_point(self, lobby_or_corridor_only: bool = False) -> tuple[float, float]:
        for _ in range(200):
            if lobby_or_corridor_only:
                x = float(self.rng.uniform(-8.5, 8.5))
                z = float(self.rng.uniform(-1.3, 6.5))
            else:
                x = float(self.rng.uniform(-10.5, 10.5))
                z = float(self.rng.uniform(-7.5, 7.5))
            if not is_blocked(x, z, 0.65):
                return x, z
        return 0.0, 0.0

    def _spawn_next_patient(self) -> None:
        if self.solo_like:
            for _ in range(150):
                px, pz = self._sample_patient_spawn_solo()
                if math.hypot(px - self.robot_x, pz - self.robot_z) >= 3.0:
                    self.patient_x, self.patient_z = px, pz
                    break
            else:
                self.patient_x, self.patient_z = 0.0, 4.0
            self._sample_solo_scenario()
            self.task_phase = TaskPhase.SEARCH_PATIENT
            self.goal_key = "patient"
            self.goal_x = self.patient_x
            self.goal_z = self.patient_z
            return

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
        self.steps_since_task = 0
        self.cumulative_team_reward = 0.0
        self.patients_helped = 0
        self.collisions = 0
        self.escort_trail: collections.deque[tuple[float, float]] = collections.deque(maxlen=400)
        self.best_goal_dist: float | None = None

        self.doctor_active = not self.solo_like
        self.doctor_x = -6.0
        self.doctor_z = -0.6
        self.doctor_dir = 1.0

        if self.solo_like:
            self.npcs = [
                {**n, "home": (n["x"], n["z"]), "wp_index": 0, "pardoning": False, "target": None}
                for n in HOSPITAL_NPCS
            ] if self.env_mode == "full" else []
            # 1. Spawn patient in corridor / lobby
            self.patient_x, self.patient_z = self._sample_patient_spawn_solo()

            # 2. Spawn robot independently across the hospital corridors/lobby
            # so the robot must actively SEARCH, navigate corridors, and find the patient
            robot_placed = False
            for _ in range(150):
                rx, rz = self._sample_patient_spawn_solo()
                if math.hypot(rx - self.patient_x, rz - self.patient_z) >= 3.5:
                    self.robot_x, self.robot_z = rx, rz
                    robot_placed = True
                    break
            if not robot_placed:
                self.robot_x, self.robot_z = 0.0, 0.0

            self.robot_yaw = float(self.rng.uniform(-math.pi, math.pi))
            self.robot_v = 0.0
            self.robot_omega = 0.0

            # 3. Sample initial scenario
            self._sample_solo_scenario()
            self.task_phase = TaskPhase.SEARCH_PATIENT
            self.goal_key = "patient"
            self.goal_x = self.patient_x
            self.goal_z = self.patient_z
            return self._get_obs()

        self.robot_x, self.robot_z = self._sample_free_point(
            lobby_or_corridor_only=(self.curriculum_stage <= 1)
        )
        self.robot_yaw = float(self.rng.uniform(-math.pi, math.pi))
        self.robot_v = 0.0
        self.robot_omega = 0.0

        self._spawn_next_patient()
        return self._get_obs()

    def compute_lidar(self) -> np.ndarray:
        """Cast 17 horizontal rays across 200-deg FOV up to 6.0m."""
        half_fov = math.radians(LIDAR_FOV_DEG * 0.5)
        rel = -half_fov + np.arange(LIDAR_NUM_RAYS) * (math.radians(LIDAR_FOV_DEG) / float(max(LIDAR_NUM_RAYS - 1, 1)))
        ray_yaw = self.robot_yaw - rel
        dxs, dzs = -np.sin(ray_yaw), -np.cos(ray_yaw)
        best = cast_rays(self.robot_x, self.robot_z, dxs, dzs, LIDAR_MAX_RANGE, self._obstacles())
        if self.doctor_active and self.curriculum_stage >= 2:
            # Check patrolling doctor cylinder (radius 0.35m)
            oc_x = self.doctor_x - self.robot_x
            oc_z = self.doctor_z - self.robot_z
            proj = oc_x * dxs + oc_z * dzs
            perp_sq = (oc_x * oc_x + oc_z * oc_z) - proj * proj
            hit = (proj > 0.0) & (proj < best) & (perp_sq < 0.35 * 0.35)
            best = np.where(hit, np.maximum(0.0, proj - 0.35), best)
        readings = np.clip(best / LIDAR_MAX_RANGE, 0.0, 1.0).astype(np.float32)
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
        hit = cast_rays(self.robot_x, self.robot_z, np.array([ux]), np.array([uz]), dist, _STRUCTURE[floor_of(self.robot_x)])
        if hit[0] < dist - 0.1:
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
        rx, rz = self.robot_x, self.robot_z
        # Target on another floor -> head (through doors) for this floor's elevator
        if floor_of(target_x) != floor_of(rx):
            target_x, target_z = ELEVATOR_POSITIONS[floor_of(rx)]

        # Robot inside a North room (z < -2.1) and target outside that room
        if rz < -2.1 and (target_z > -1.8 or north_door_x(target_x) != north_door_x(rx)):
            door_x = north_door_x(rx)
            if abs(rx - door_x) > 0.6 or rz < -1.8:
                return door_x, -1.2
        # Robot inside a South room (z > 2.1) and target outside that room
        if rz > 2.1 and (target_z < 1.8 or south_door_x(target_x) != south_door_x(rx)):
            door_x = south_door_x(rx)
            if abs(rx - door_x) > 0.6 or rz > 1.8:
                return door_x, 1.2
        # Robot in corridor (-2.1 <= z <= 2.1) and target in a North room (z < -2.2)
        if -2.1 <= rz <= 2.1 and target_z < -2.2:
            door_x = north_door_x(target_x)
            if abs(rx - door_x) > 0.8:
                return door_x, 0.0  # travel along the corridor centreline (furniture lines the walls)
            if rz > -2.1:
                return door_x, -2.6
        # Robot in corridor (-2.1 <= z <= 2.1) and target in a South room (z > 2.2)
        if -2.1 <= rz <= 2.1 and target_z > 2.2:
            door_x = south_door_x(target_x)
            if abs(rx - door_x) > 0.8:
                return door_x, 0.0
            if rz < 2.1:
                return door_x, 2.6
        return target_x, target_z

    def _step_escorted_patient(self) -> None:
        """Patient walks along the robot's breadcrumb trail (not a straight line through door frames)."""
        trail = self.escort_trail
        if trail:
            d_last = math.hypot(self.robot_x - trail[-1][0], self.robot_z - trail[-1][1])
            if d_last > ESCORT_TRAIL_JUMP_M:
                trail.clear()
        if not trail or math.hypot(self.robot_x - trail[-1][0], self.robot_z - trail[-1][1]) >= ESCORT_CRUMB_SPACING_M:
            trail.append((self.robot_x, self.robot_z))

        d_robot = math.hypot(self.robot_x - self.patient_x, self.robot_z - self.patient_z)
        if d_robot <= ESCORT_STOP_DIST_M:
            return
        while len(trail) > 1 and math.hypot(trail[0][0] - self.patient_x, trail[0][1] - self.patient_z) <= ESCORT_CRUMB_REACH_M:
            trail.popleft()
        tx, tz = trail[0]
        dx, dz = tx - self.patient_x, tz - self.patient_z
        d_crumb = math.hypot(dx, dz)
        step = min(ELDERLY_PATIENT_SPEED * self.dt, d_robot - ESCORT_STOP_DIST_M, d_crumb)
        if step <= 1e-6:
            return
        sx, sz = dx / d_crumb * step, dz / d_crumb * step
        # Full step, else slide along the wall on one axis
        candidates = [(sx, sz)]
        if abs(sx) > 1e-3:
            candidates.append((math.copysign(step, sx), 0.0))
        if abs(sz) > 1e-3:
            candidates.append((0.0, math.copysign(step, sz)))
        for cx, cz in candidates:
            px_new, pz_new = self.patient_x + cx, self.patient_z + cz
            if not is_blocked(px_new, pz_new, 0.28):
                self.patient_x, self.patient_z = px_new, pz_new
                return

    def step(self, action: np.ndarray | list[float] | tuple[float, float]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self.step_count += 1
        self.steps_since_task += 1
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

        x_base = 50.0 if self.robot_x >= 25.0 else 0.0

        def free(px: float, pz: float) -> bool:
            return (
                abs(px - x_base) <= 11.6
                and abs(pz) <= 8.6
                and not is_blocked(px, pz, 0.32)
                and not self._near_npc(px, pz, 0.32)
            )

        # Blocked full step -> slide along the wall on one axis (like Godot move_and_slide)
        hit_wall = not free(nx, nz)
        moved_to = None
        if not hit_wall:
            moved_to = (nx, nz)
        elif free(nx, self.robot_z):
            moved_to = (nx, self.robot_z)
        elif free(self.robot_x, nz):
            moved_to = (self.robot_x, nz)
        if moved_to is not None:
            self.robot_v = math.hypot(moved_to[0] - self.robot_x, moved_to[1] - self.robot_z) / self.dt
            self.robot_x, self.robot_z = moved_to
        else:
            self.robot_v = 0.0
        self.robot_omega = omega

        if self.npcs:
            self._step_npcs()

        # Step patrolling doctor in Stage 2 (with pardon behavior when robot approaches)
        if self.doctor_active and self.curriculum_stage >= 2:
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
            self._step_escorted_patient()
        else:
            self.escort_trail.clear()

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
        escorting = self.task_phase == TaskPhase.GUIDE_ESCORT
        if escorting:
            active_x, _ = DESTINATION_POSITIONS.get(self.goal_key, (0.0, 4.5))
        elif self.task_phase == TaskPhase.CALL_STAFF:
            active_x, _ = DESTINATION_POSITIONS["staff"]
        else:
            active_x = self.patient_x
        robot_floor = floor_of(self.robot_x)
        elev_x, elev_z = ELEVATOR_POSITIONS[robot_floor]
        if (
            floor_of(active_x) != robot_floor
            and math.hypot(elev_x - self.robot_x, elev_z - self.robot_z) <= GOAL_ARRIVE_ROBOT_M
            and (not escorting or math.hypot(elev_x - self.patient_x, elev_z - self.patient_z) <= GOAL_ARRIVE_PATIENT_M)
        ):
            # Generic elevator ride toward the active target's floor (patient rides along when escorted)
            shift = 50.0 if robot_floor == 1 else -50.0
            self.robot_x += shift
            if escorting:
                self.patient_x += shift
            event = "elevator_transit" if escorting else "elevator_ride"
            self.steps_since_task = 0
        elif self.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
            if curr_dist_final <= PATIENT_TALK_DIST_M:
                resolved = resolve_destination_key(
                    self.scenario_action, self.scenario_slot, self.scenario_floor,
                    current_floor=floor_of(self.patient_x),
                )
                if self.scenario_action == "guide":
                    event = "engaged"
                    self.task_phase = TaskPhase.GUIDE_ESCORT
                    self.goal_key = resolved
                    self.steps_since_task = 0
                elif self.scenario_action in ("call_staff", "emergency"):
                    event = "engaged"
                    self.task_phase = TaskPhase.CALL_STAFF
                    self.goal_key = "staff"
                    self.steps_since_task = 0
                else:
                    event = "answer"
                    self.patients_helped += 1
                    self.steps_since_task = 0
                    self._spawn_next_patient()
        elif self.task_phase == TaskPhase.GUIDE_ESCORT:
            dest_x, dest_z = DESTINATION_POSITIONS.get(self.goal_key, (0.0, 4.5))
            d_robot_dest = math.hypot(dest_x - self.robot_x, dest_z - self.robot_z)
            d_patient_dest = math.hypot(dest_x - self.patient_x, dest_z - self.patient_z)
            if d_robot_dest <= GOAL_ARRIVE_ROBOT_M and d_patient_dest <= GOAL_ARRIVE_PATIENT_M:
                final_target = resolve_final_target(self.scenario_action, self.scenario_slot, self.scenario_floor)
                # Check if elevator transition is needed to continue to destination floor
                if self.goal_key in ("thang_may", "thang_may_t2") and final_target != self.goal_key:
                    event = "elevator_transit"
                    if self.goal_key == "thang_may":
                        # Floor 1 -> Floor 2 elevator teleport
                        self.robot_x += 50.0
                        self.patient_x += 50.0
                    else:
                        # Floor 2 -> Floor 1 elevator teleport
                        self.robot_x -= 50.0
                        self.patient_x -= 50.0
                    self.goal_key = final_target
                    self.steps_since_task = 0
                else:
                    event = "guide_completed"
                    self.patients_helped += 1
                    self.steps_since_task = 0
                    self._spawn_next_patient()
        elif self.task_phase == TaskPhase.CALL_STAFF:
            dest_x, dest_z = DESTINATION_POSITIONS["staff"]
            if math.hypot(dest_x - self.robot_x, dest_z - self.robot_z) <= GOAL_ARRIVE_ROBOT_M:
                event = "call_staff"
                self.patients_helped += 1
                self.steps_since_task = 0
                self._spawn_next_patient()

        # Compute step reward matching RLEnvBridge.compute_step_reward
        r_team = 0.0
        if event == "answer":
            r_team = REWARD_ANSWER
        elif event == "call_staff":
            r_team = REWARD_CALL_STAFF
        elif event == "elevator_transit":
            r_team = REWARD_ELEVATOR_TRANSIT
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

        # Stagnation = no real progress toward the active goal (long elevator/escort legs are not stagnation)
        if event:
            self.best_goal_dist = None
        prog_x, prog_z = (
            (self.patient_x, self.patient_z)
            if self.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT)
            else DESTINATION_POSITIONS.get(self.goal_key, (0.0, 4.5))
        )
        d_prog = math.hypot(prog_x - self.robot_x, prog_z - self.robot_z)
        if self.best_goal_dist is None:
            self.best_goal_dist = d_prog
        elif d_prog < self.best_goal_dist - STAGNATION_PROGRESS_M:
            self.best_goal_dist = d_prog
            self.steps_since_task = 0

        stagnated = False
        if self.max_stagnation_steps > 0 and self.steps_since_task >= self.max_stagnation_steps:
            stagnated = True
            r_team += self.stagnation_penalty

        total_reward = float(r_team + r_progress + r_tether + r_safety + r_align)
        self.cumulative_team_reward += r_team

        done = (self.step_count >= self.max_steps) or stagnated
        if stagnated:
            exit_reason = "STAGNATION_TIMEOUT"
        elif self.step_count >= self.max_steps:
            exit_reason = "COMPLETED"
        else:
            exit_reason = "RUNNING"

        info = {
            "r_team": r_team,
            "cumulative_team_reward": self.cumulative_team_reward,
            "patients_helped": self.patients_helped,
            "collisions": self.collisions,
            "task_phase": int(self.task_phase),
            "d_tether": d_tether,
            "event": event,
            "steps_since_task": self.steps_since_task,
            "exit_reason": exit_reason,
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
        speed_factor = 1.0
        if center_min < 0.65:
            turn_strength = float(np.clip((0.65 - center_min) / 0.5, 0.4, 1.2))
            repel = turn_strength if left_clear >= right_clear else -turn_strength
            speed_factor = float(np.clip(center_min / 0.65, 0.35, 1.0))

        omega = float(np.clip(1.8 * angle + repel, -OMEGA_MAX, OMEGA_MAX))
        max_v = (ESCORT_WALK_SPEED if self.task_phase == TaskPhase.GUIDE_ESCORT else V_MAX) * speed_factor
        v = float(np.clip(0.55 * dist, 0.0, max_v))
        if abs(angle) > 0.65:
            v *= 0.25

        a_v = float(np.clip(2.0 * (v / V_MAX) - 1.0, -1.0, 1.0))
        a_omega = float(np.clip(omega / OMEGA_MAX, -1.0, 1.0))
        return np.array([a_v, a_omega], dtype=np.float32)
