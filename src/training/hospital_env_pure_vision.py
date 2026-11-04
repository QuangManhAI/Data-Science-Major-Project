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
    ELEVATOR_POSITIONS,
    ESCORT_WALK_SPEED,
    GOAL_ARRIVE_PATIENT_M,
    GOAL_ARRIVE_ROBOT_M,
    HOSPITAL_NPCS,
    REWARD_ELEVATOR_TRANSIT,
    ROBOT_BOUND_X_M,
    ROBOT_BOUND_Z_M,
    ROBOT_DOCK,
    ROBOT_RADIUS_M,
    TaskPhase,
    _STRUCTURE,
    cast_rays,
    floor_of,
    is_blocked,
    nav_grid,
    resolve_destination_key,
    resolve_final_target,
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
PENALTY_DOCTOR_PROXIMITY = -0.30     # Rebalanced: gentle evasion pressure, eliminates -1300 reward cliff
PENALTY_FREE_SEARCH_TIME = -0.04
REWARD_VISUAL_SPOTTED = 5.0
REWARD_PATIENT_ENGAGED_VIS = 50.0
REWARD_GUIDE_ARRIVED_VIS = 150.0     # Emphasized completion reward (was 80.0)
WEIGHT_VISUAL_OBSTACLE = 0.50
WEIGHT_PATROL_PROGRESS = 0.25        # Active incentive for corridor exploration in SEARCH phase

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


# Nominal physical heights (in meters) for hospital object classes to normalize visual proximity
OBSTACLE_NOMINAL_HEIGHTS: dict[str, float] = {
    "medical_waste_bin": 0.40,
    "hospital_chair": 0.50,
    "pccc_station": 0.60,
    "wheelchair": 0.85,
    "medical_trolley": 0.85,
    "stretcher_cart": 0.85,
    "hospital_bed": 0.85,
    "water_dispenser": 1.00,
    "sanitizer_stand": 1.10,
    "vital_monitor": 1.20,
    "queue_kiosk": 1.40,
    "reception_counter": 1.10,
}
HUMAN_REFERENCE_HEIGHT: float = 1.70


def compute_5_sector_depth(
    robot_x: float,
    robot_z: float,
    robot_yaw: float,
    fl: int = 1,
    structure_walls: dict[int, np.ndarray] | None = None,
    obstacle_boxes: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Compute 5-sector visual obstacle proximity [0..1]^5.

    Strict Sim-to-Real:
    Zero LiDAR, Zero map wall raycasting.
    Populated EXCLUSIVELY by mapping 2D obstacle bounding boxes into the 5 camera sectors.
    Uses class-aware height scaling, ground contact y2, and box width coverage across sectors.
    If no obstacles are detected (or YOLO is disconnected), all sectors are 0.0.
    """
    sectors = np.zeros(5, dtype=np.float32)

    if obstacle_boxes:
        for b in obstacle_boxes:
            x1 = float(b.get("x1", 0.0))
            x2 = float(b.get("x2", 1.0))
            y1 = float(b.get("y1", 0.0))
            y2 = float(b.get("y2", 1.0))
            cx_val = b.get("cx", (x1 + x2) / 2.0 - 0.5)
            h_val = b.get("h", y2 - y1)
            cx = float(cx_val) if math.isfinite(cx_val) else 0.0
            h = float(h_val) if math.isfinite(h_val) else 0.0

            cname = str(b.get("class_name", "")).lower()
            nom_h = OBSTACLE_NOMINAL_HEIGHTS.get(cname, 0.60)
            scale = min(4.0, max(1.0, HUMAN_REFERENCE_HEIGHT / nom_h))

            # Ground contact proximity: for an object on the floor, y2 approaching 1.0 means it is directly ahead
            prox_floor = (y2 - 0.5) / 0.45 if y2 > 0.5 else 0.0
            prox = max(h * scale, prox_floor)

            # Determine sector span
            if "x1" in b and "x2" in b and x2 > x1:
                sec_start = int(np.clip(math.floor(x1 * 5.0), 0, 4))
                sec_end = int(np.clip(math.floor(x2 * 5.0), 0, 4))
            else:
                offset = cx if abs(cx) <= 0.5 else cx * 0.5
                sec_start = int(np.clip(math.floor((offset + 0.5) * 5.0), 0, 4))
                sec_end = sec_start

            for s_idx in range(sec_start, sec_end + 1):
                sectors[s_idx] = max(sectors[s_idx], float(np.clip(prox, 0.0, 1.0)))

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
        shift_duration_s: float = 700.0,
        dt: float = 0.2,
        seed: int = 42,
        env_mode: str = "full",
        stagnation_timeout_s: float = 90.0,
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
        self.pure_vision = True
        self.patients: list[dict[str, Any]] = []
        self.active_patient_idx: int | None = None
        self.returning_to_lobby: bool = False
        self.last_doctor_det: dict[str, float] = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
        self.last_sectors: np.ndarray = np.zeros(5, dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 1. Start strictly at Floor 1 Main Entrance / Dock facing inward
        self.robot_x, self.robot_z, self.robot_yaw = ROBOT_DOCK
        self.robot_v = 0.0
        self.robot_omega = 0.0

        # 2. Populate 6 Floor 1 Patients from Scene Layout with diverse destinations
        dest_pool = [
            ("guide", "101", 1),
            ("guide", "nha_thuoc", 1),
            ("guide", "205", 2),
            ("guide", "210", 2),
            ("guide", "301", 3),
            ("guide", "303", 3),
        ]
        if hasattr(self, "rng"):
            self.rng.shuffle(dest_pool)

        self.patients = []
        raw_patients = [n for n in HOSPITAL_NPCS if n.get("role") == "patient"]
        for i, p in enumerate(raw_patients):
            pert_x = float(self.rng.uniform(-0.3, 0.3)) if self.domain_randomization else 0.0
            pert_z = float(self.rng.uniform(-0.3, 0.3)) if self.domain_randomization else 0.0
            cx = float(p["x"]) + pert_x
            cz = float(p["z"]) + pert_z
            if is_blocked(cx, cz, 0.30):
                cx, cz = float(p["x"]), float(p["z"])
            self.patients.append({
                "id": p.get("name", f"Patient_{i+1:02d}"),
                "home_x": cx,
                "home_z": cz,
                "x": cx,
                "z": cz,
                "seated": bool(p.get("seated", False)),
                "served": False,
                "destination": dest_pool[i % len(dest_pool)],
            })

        self.active_patient_idx = None
        self.returning_to_lobby = False
        self.task_phase = TaskPhase.SEARCH_PATIENT
        self.goal_key = "patient"
        self.patient_x = -999.0
        self.patient_z = -999.0
        self.patient_seated = False
        self.patrol_wp_idx = 0
        self.patients_helped = 0
        self.step_count = 0
        self.steps_since_task = 0
        self.cumulative_team_reward = 0.0
        self.cumulative_pure_vision_reward = 0.0
        self.escort_trail.clear()
        self.last_doctor_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
        self.last_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
        self.last_sectors = np.zeros(5, dtype=np.float32)

        # Setup NPCs as obstacles
        self.npcs = [
            {**n, "home": (n["x"], n["z"]), "wp_index": 0, "pardoning": False, "target": None, "pardon_t": 0.0, "cooldown": 0.0, "blocked_t": 0.0}
            for n in HOSPITAL_NPCS
        ] if self.env_mode == "full" else []

        self.doctor_active = True
        self.doctor_x = -6.0
        self.doctor_z = -0.6
        self.doctor_dir = 1.0

        return self._get_obs()

    def spawn_escort_curriculum(self) -> np.ndarray:
        """Deprecated: All episodes start at Floor 1 front entrance/dock."""
        return self.reset()

    def _compute_yolo_det(self) -> tuple[float, float, float, float]:
        """Compute pinhole detection across multiple unserved patients or currently escorted patient."""
        fl = floor_of(self.robot_x)

        # 1. During GUIDE_ESCORT or CALL_STAFF: track active escorted patient
        if self.task_phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF):
            if self.active_patient_idx is not None and self.active_patient_idx < len(self.patients):
                p = self.patients[self.active_patient_idx]
                target_x, target_z, is_seated = p["x"], p["z"], p["seated"]
            else:
                target_x, target_z, is_seated = self.patient_x, self.patient_z, getattr(self, "patient_seated", False)

            if floor_of(target_x) != fl:
                self.last_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
                return 0.0, 0.0, 0.0, 0.0

            det = compute_pinhole_detection(
                robot_x=self.robot_x,
                robot_z=self.robot_z,
                robot_yaw=self.robot_yaw,
                target_x=target_x,
                target_z=target_z,
                target_seated=is_seated,
                is_blocked_fn=self._check_los_occluded,
                domain_randomization=self.domain_randomization,
                rng=self.rng,
            )
            self.last_det = det
            conf = float(np.clip(0.95 - 0.04 * math.hypot(target_x - self.robot_x, target_z - self.robot_z), 0.45, 0.98)) if det["seen"] > 0.5 else 0.0
            return det["seen"], det["cx"], det["h"], conf

        # 2. During APPROACH_PATIENT: track currently locked unserved patient
        if self.task_phase == TaskPhase.APPROACH_PATIENT:
            if self.active_patient_idx is not None and self.active_patient_idx < len(self.patients):
                p = self.patients[self.active_patient_idx]
                target_x, target_z, is_seated = p["x"], p["z"], p["seated"]
            else:
                target_x, target_z, is_seated = self.patient_x, self.patient_z, getattr(self, "patient_seated", False)

            if floor_of(target_x) != fl:
                self.last_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
                return 0.0, 0.0, 0.0, 0.0

            det = compute_pinhole_detection(
                robot_x=self.robot_x,
                robot_z=self.robot_z,
                robot_yaw=self.robot_yaw,
                target_x=target_x,
                target_z=target_z,
                target_seated=is_seated,
                is_blocked_fn=self._check_los_occluded,
                domain_randomization=self.domain_randomization,
                rng=self.rng,
            )
            self.last_det = det
            conf = float(np.clip(0.95 - 0.04 * math.hypot(target_x - self.robot_x, target_z - self.robot_z), 0.45, 0.98)) if det["seen"] > 0.5 else 0.0
            return det["seen"], det["cx"], det["h"], conf

        # 3. During SEARCH_PATIENT: scan all unserved patients on current floor
        if getattr(self, "returning_to_lobby", False):
            # Robot is returning to Floor 1 via elevator; don't acquire patients on Floor 2/3
            self.last_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
            return 0.0, 0.0, 0.0, 0.0

        best_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
        best_h = 0.0
        best_idx = None

        for idx, p in enumerate(self.patients):
            if p.get("served", False):
                continue  # Never speak to served patients again!
            if floor_of(p["x"]) != fl:
                continue

            det = compute_pinhole_detection(
                robot_x=self.robot_x,
                robot_z=self.robot_z,
                robot_yaw=self.robot_yaw,
                target_x=p["x"],
                target_z=p["z"],
                target_seated=p["seated"],
                is_blocked_fn=self._check_los_occluded,
                domain_randomization=self.domain_randomization,
                rng=self.rng,
            )
            if det["seen"] > 0.5 and det["h"] > best_h:
                best_h = det["h"]
                best_det = det
                best_idx = idx

        if best_idx is not None:
            # Dynamically acquire this unserved patient!
            self.active_patient_idx = best_idx
            p = self.patients[best_idx]
            self.patient_x = p["x"]
            self.patient_z = p["z"]
            self.patient_seated = p["seated"]
            self.scenario_action, self.scenario_slot, self.scenario_floor = p["destination"]
            self.task_phase = TaskPhase.APPROACH_PATIENT
            self.patient_spotted_once = True
            self.steps_since_task = 0
            self.last_det = best_det
            conf = float(np.clip(0.95 - 0.04 * math.hypot(p["x"] - self.robot_x, p["z"] - self.robot_z), 0.45, 0.98))
            return best_det["seen"], best_det["cx"], best_det["h"], conf

        self.last_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
        return 0.0, 0.0, 0.0, 0.0

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

    def _compute_door_det(self) -> dict[str, float]:
        """Compute visual pinhole detection for doors in camera FOV."""
        fl = floor_of(self.robot_x)
        best_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
        best_dist = 999.0

        for (floor, side), (centres, _) in _DOORS.items():
            if floor != fl:
                continue
            z_line = -2.0 if side == "north" else 2.0
            for c in centres:
                d = math.hypot(self.robot_x - c, self.robot_z - z_line)
                if d > 8.0:
                    continue
                det = compute_pinhole_detection(
                    robot_x=self.robot_x,
                    robot_z=self.robot_z,
                    robot_yaw=self.robot_yaw,
                    target_x=c,
                    target_z=z_line,
                    target_seated=False,
                    is_blocked_fn=self._check_los_occluded,
                    domain_randomization=self.domain_randomization,
                    rng=self.rng,
                )
                if det["seen"] > 0.5 and d < best_dist:
                    best_dist = d
                    best_det = det

        return best_det

    def _step_escorted_patient(self) -> None:
        """Patient walks in peripheral escort formation (side-by-side) within camera FOV."""
        if self.active_patient_idx is None or self.active_patient_idx >= len(self.patients):
            return

        p = self.patients[self.active_patient_idx]
        cur_px, cur_pz = p["x"], p["z"]

        # If patient starts or is currently inside furniture/wall (e.g. seated), snap out to free floor
        if is_blocked(cur_px, cur_pz, 0.28):
            fl_g = nav_grid(floor_of(cur_px))
            si, sj = fl_g.snap(cur_px, cur_pz)
            cur_px, cur_pz = float(fl_g.xs[si]), float(fl_g.zs[sj])
            p["x"], p["z"] = cur_px, cur_pz
            self.patient_x, self.patient_z = cur_px, cur_pz

        fwd_x = -math.sin(self.robot_yaw)
        fwd_z = -math.cos(self.robot_yaw)
        right_x = math.cos(self.robot_yaw)
        right_z = -math.sin(self.robot_yaw)

        # Companion waypoint: 1.0m forward, 0.55m to the right (bearing ~29 deg, inside 75 deg FOV)
        target_x = self.robot_x + fwd_x * 1.0 + right_x * 0.55
        target_z = self.robot_z + fwd_z * 1.0 + right_z * 0.55

        # If blocked by wall, fall back to narrower offset
        if is_blocked(target_x, target_z, 0.28):
            target_x = self.robot_x + fwd_x * 0.50 + right_x * 0.35
            if is_blocked(target_x, target_z, 0.28):
                target_x = self.robot_x - fwd_x * 0.8
                target_z = self.robot_z - fwd_z * 0.8

        dx = target_x - cur_px
        dz = target_z - cur_pz
        dist = math.hypot(dx, dz)
        if dist <= 0.20:
            return

        step = min(ESCORT_WALK_SPEED * self.dt, dist)
        sx = (dx / dist) * step
        sz = (dz / dist) * step

        candidates = [(sx, sz)]
        if abs(sx) > 1e-3:
            candidates.append((math.copysign(step, sx), 0.0))
        if abs(sz) > 1e-3:
            candidates.append((0.0, math.copysign(step, sz)))

        # Fallback candidate angles if straight and axis-aligned moves are obstructed by a corner
        for ang in [-math.pi / 4, math.pi / 4, -math.pi / 2, math.pi / 2]:
            cs, sn = math.cos(ang), math.sin(ang)
            candidates.append((sx * cs - sz * sn, sx * sn + sz * cs))

        for cx, cz in candidates:
            px_new, pz_new = cur_px + cx, cur_pz + cz
            if not is_blocked(px_new, pz_new, 0.28):
                p["x"], p["z"] = px_new, pz_new
                self.patient_x, self.patient_z = px_new, pz_new
                return

    def _get_obs(self) -> np.ndarray:
        """Build 29-D pure-vision observation vector. ZERO LiDAR, ZERO Map Raycasts."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        fl = floor_of(self.robot_x)

        # 1. Compute 5-Sector Visual Depth Map [0..4] (Strictly from detected obstacles)
        doc_seen, doc_cx, doc_h, doc_conf = self._compute_doctor_det()
        obs_boxes = []
        if doc_seen > 0.5:
            obs_boxes.append({"cx": doc_cx, "h": doc_h})

        sectors = compute_5_sector_depth(
            robot_x=self.robot_x,
            robot_z=self.robot_z,
            robot_yaw=self.robot_yaw,
            fl=fl,
            structure_walls=None,
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
        if getattr(self, "returning_to_lobby", False):
            # Navigating back to elevator on current floor to return down to Floor 1
            elev_key = f"thang_may_t{fl}" if fl > 1 else "thang_may"
            self.set_nav_target(*DESTINATION_POSITIONS.get(elev_key, (0.0, 4.5)))
            to_gx = self.goal_x - self.robot_x
            to_gz = self.goal_z - self.robot_z
            fwd_x = -math.sin(self.robot_yaw)
            fwd_z = -math.cos(self.robot_yaw)
            angle_goal = signed_angle_rad(fwd_x, fwd_z, to_gx, to_gz)
            obs[17] = float(np.clip(self.nav_remaining / 25.0, 0.0, 1.0))
            obs[18] = math.sin(angle_goal)
            obs[19] = math.cos(angle_goal)
        elif self.task_phase == TaskPhase.APPROACH_PATIENT and pat_seen > 0.5:
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
            # During GUIDE_ESCORT or CALL_STAFF, landmark goal navigation is active
            if getattr(self, "custom_nav_target", None) is not None:
                self.set_nav_target(*self.custom_nav_target)
            else:
                self.set_nav_target(*self._nav_final_target())
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

        # 7. Patient Tether Status [22..24] (Visual Tether from Peripheral Companion Camera Lock)
        if self.task_phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF):
            if pat_seen > 0.5:
                d_vis_tether = float(np.clip(1.20 / max(pat_h, 0.05), 0.5, 5.0))
                self.last_known_tether = d_vis_tether
                self.lost_sight_steps = 0
                obs[22] = float(np.clip(d_vis_tether / 5.0, 0.0, 1.0))
                obs[23] = 1.0 if d_vis_tether > 2.0 else 0.0
                obs[24] = 1.0 if d_vis_tether > 3.5 else 0.0
            else:
                # Occlusion / turn persistence buffer: decay last known tether distance rather than instant panic
                self.lost_sight_steps = getattr(self, "lost_sight_steps", 0) + 1
                prev_tether = getattr(self, "last_known_tether", 1.2)
                if self.lost_sight_steps > 15:  # 3.0 seconds continuous occlusion
                    obs[22] = 1.0
                    obs[23] = 1.0
                    obs[24] = 1.0
                else:
                    obs[22] = float(np.clip(prev_tether / 5.0, 0.0, 1.0))
                    obs[23] = 1.0 if prev_tether > 2.0 else 0.0
                    obs[24] = 0.0
        else:
            self.lost_sight_steps = 0
            obs[22] = 0.0
            obs[23] = 0.0
            obs[24] = 0.0

        # 8. Doorway Geometry [25..28] (Visual Door Detection, Zero Oracle Lookup)
        door_det = self._compute_door_det()
        if door_det["seen"] > 0.5:
            door_cx = door_det["cx"]
            door_h = door_det["h"]
            d_vis_door = float(np.clip(1.20 / max(door_h, 0.05), 0.5, 10.0))
            angle_door = float(-math.atan(door_cx * math.tan(CAMERA_HFOV_RAD / 2.0)))
            obs[25] = float(np.clip(d_vis_door / 10.0, 0.0, 1.0))
            obs[26] = math.sin(angle_door)
            obs[27] = math.cos(angle_door)
            obs[28] = 1.0 if d_vis_door < 2.0 else 0.0
        else:
            obs[25:29] = 0.0

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        # Free search step penalty
        free_search_penalty = (
            PENALTY_FREE_SEARCH_TIME if self.task_phase == TaskPhase.SEARCH_PATIENT else 0.0
        )
        prev_nav = getattr(self, "nav_remaining", 0.0)
        prev_fl = floor_of(self.robot_x)
        obs, reward, done, info = super().step(action)
        reward += free_search_penalty

        # Active corridor patrol waypoint progress incentive during SEARCH_PATIENT
        if self.task_phase == TaskPhase.SEARCH_PATIENT:
            curr_nav = getattr(self, "nav_remaining", 0.0)
            if prev_nav > 0.0 and curr_nav > 0.0:
                patrol_prog = float(np.clip(prev_nav - curr_nav, -0.5, 0.5))
                reward += WEIGHT_PATROL_PROGRESS * patrol_prog
                if patrol_prog > 0.02:
                    self.steps_since_task = max(0, self.steps_since_task - 1)

        # 1. Multi-Patient Phase Management & Transitions
        cur_fl = floor_of(self.robot_x)

        # 1a. Return to Floor 1 Lobby Elevator Transit
        if getattr(self, "returning_to_lobby", False):
            if (prev_fl > 1 and cur_fl == 1) or (cur_fl > 1 and math.hypot(ELEVATOR_POSITIONS[cur_fl][0] - self.robot_x, ELEVATOR_POSITIONS[cur_fl][1] - self.robot_z) <= GOAL_ARRIVE_ROBOT_M):
                if cur_fl > 1:
                    shift = float(1 - cur_fl) * 50.0
                    self.robot_x += shift
                self.returning_to_lobby = False
                self.goal_key = "patient"
                self.steps_since_task = 0
                self.visited_cells.clear()
                self.visited_cells.add((int(math.floor(self.robot_x)), int(math.floor(self.robot_z)), 1))
                reward += REWARD_ELEVATOR_TRANSIT
                self.cumulative_team_reward += REWARD_ELEVATOR_TRANSIT
                info["return_transit"] = True

        # 1b. APPROACH_PATIENT Arrival Check (Dialogue Trigger)
        if self.task_phase == TaskPhase.APPROACH_PATIENT and self.active_patient_idx is not None:
            pat_seen, pat_cx, pat_h, _ = self._compute_yolo_det()
            d_phys = math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
            if (
                pat_h >= TARGET_BOX_H
                or d_phys <= PATIENT_TALK_DIST_M_VIS
                or (pat_seen > 0.5 and pat_h >= 0.38 and d_phys <= 2.2)
            ):
                resolved = resolve_destination_key(
                    self.scenario_action,
                    self.scenario_slot,
                    self.scenario_floor,
                    current_floor=floor_of(self.patient_x),
                )
                self.task_phase = TaskPhase.GUIDE_ESCORT
                self.goal_key = resolved
                self.steps_since_task = 0
                reward += REWARD_PATIENT_ENGAGED_VIS
                info["patient_engaged"] = True

                # Stand up patient from chair and snap onto legal, navigable floor
                p = self.patients[self.active_patient_idx]
                p["seated"] = False
                fl_grid = nav_grid(floor_of(p["x"]))
                si, sj = fl_grid.snap(p["x"], p["z"])
                p["x"], p["z"] = float(fl_grid.xs[si]), float(fl_grid.zs[sj])
                self.patient_x, self.patient_z = p["x"], p["z"]
                self.escort_trail.clear()
                self.last_known_tether = 1.0
                self.lost_sight_steps = 0

        # 1c. GUIDE_ESCORT Elevator Transit or Room Arrival
        if self.task_phase == TaskPhase.GUIDE_ESCORT and self.active_patient_idx is not None:
            dest_x, dest_z = DESTINATION_POSITIONS.get(self.goal_key, (0.0, 4.5))
            d_r = math.hypot(dest_x - self.robot_x, dest_z - self.robot_z)
            d_p = math.hypot(dest_x - self.patient_x, dest_z - self.patient_z)
            if d_r <= GOAL_ARRIVE_ROBOT_M and d_p <= GOAL_ARRIVE_PATIENT_M:
                final_target = resolve_final_target(self.scenario_action, self.scenario_slot, self.scenario_floor)
                # Check if this arrival is at an elevator for cross-floor transit
                if self.goal_key in ("thang_may", "thang_may_t2", "thang_may_t3") and final_target != self.goal_key:
                    # Ride elevator with patient to destination floor
                    dest_fl = floor_of(DESTINATION_POSITIONS.get(final_target, (0.0, 4.5))[0])
                    shift = float(dest_fl - cur_fl) * 50.0
                    self.robot_x += shift
                    self.patient_x += shift
                    self.patients[self.active_patient_idx]["x"] += shift
                    self.goal_key = final_target
                    self.steps_since_task = 0
                    self.escort_trail.clear()
                    reward += REWARD_ELEVATOR_TRANSIT
                    info["elevator_transit"] = True
                else:
                    # Successfully arrived at final destination room! Mark SERVED
                    self.patients[self.active_patient_idx]["served"] = True
                    # Progressive reward escalation: +150 * (1.0 + 0.20 * k)
                    reward_delivery = float(REWARD_GUIDE_ARRIVED_VIS * (1.0 + 0.20 * self.patients_helped))
                    self.patients_helped += 1
                    self.steps_since_task = 0
                    reward += reward_delivery
                    info["guide_completed"] = True

                    # Check if all 6 patients on Floor 1 are served
                    if all(p.get("served", False) for p in self.patients):
                        done = True
                        info["exit_reason"] = "ALL_PATIENTS_SERVED"
                        reward += 250.0  # Full shift bonus!
                    else:
                        new_fl = floor_of(self.robot_x)
                        if new_fl > 1:
                            self.returning_to_lobby = True
                            self.task_phase = TaskPhase.SEARCH_PATIENT
                            self.goal_key = f"thang_may_t{new_fl}"
                            self.active_patient_idx = None
                            self.patient_x = -999.0
                            self.patient_z = -999.0
                        else:
                            self.returning_to_lobby = False
                            self.task_phase = TaskPhase.SEARCH_PATIENT
                            self.goal_key = "patient"
                            self.active_patient_idx = None
                            self.patient_x = -999.0
                            self.patient_z = -999.0
                            self.visited_cells.clear()
                            self.visited_cells.add((int(math.floor(self.robot_x)), int(math.floor(self.robot_z)), 1))

        # 2. Continuous Visual Obstacle Proximity Repulsion
        obs_sectors = obs[0:5]
        for s_val in obs_sectors:
            if s_val > 0.50:
                reward += -WEIGHT_VISUAL_OBSTACLE * (s_val - 0.50)

        # 3. Strict Doctor Avoidance Penalty
        doc_seen = obs[9]
        doc_h = obs[11]
        if doc_seen > 0.5 and doc_h > 0.35:
            reward += PENALTY_DOCTOR_PROXIMITY * (doc_h / 0.50)

        if not math.isfinite(reward):
            reward = 0.0
        reward = float(np.clip(reward, -50.0, 100.0))

        # Synchronize cumulative reward strictly to sum of step rewards (zero phantom superclass spikes)
        self.cumulative_pure_vision_reward = getattr(self, "cumulative_pure_vision_reward", 0.0) + reward
        self.cumulative_team_reward = self.cumulative_pure_vision_reward

        info["cumulative_team_reward"] = self.cumulative_team_reward
        info["visual_sectors"] = obs_sectors.tolist()
        info["patients_helped"] = self.patients_helped
        info["total_patients"] = len(self.patients)
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

        # 3. Escort or Corridor Navigation: smooth visual sector potential field avoidance
        goal_angle = math.atan2(obs[18], obs[19])
        v_base = ESCORT_WALK_SPEED if self.task_phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF) else V_MAX

        # In GUIDE_ESCORT: adapt forward speed according to visual patient tether
        if self.task_phase == TaskPhase.GUIDE_ESCORT:
            if obs[24] > 0.5:  # Tether broken or prolonged occlusion
                v_base = 0.08  # Creep forward gently to maintain steering / turning authority while patient catches up
            elif obs[23] > 0.5:  # Tether stretched (> 2.0m)
                v_base = 0.20  # Slow down to allow patient to close distance

        # Repulsion from side sectors:
        # Left sectors (0, 1) push clockwise (+yaw rate / right turn)
        # Right sectors (3, 4) push counter-clockwise (-yaw rate / left turn)
        repulsion = (
            + 0.8 * (sectors[0] + 0.5 * sectors[1])
            - 0.8 * (sectors[4] + 0.5 * sectors[3])
        )

        # Center sector controls forward speed factor
        c_obs = sectors[2]
        speed_factor = float(np.clip(1.0 - max(0.0, c_obs - 0.35) / 0.45, 0.0, 1.0))

        v_des = v_base * speed_factor
        omega_des = float(np.clip(goal_angle * 1.5 + repulsion, -OMEGA_MAX, OMEGA_MAX))

        # If completely stopped by frontal wall or obstacle, rotate in-place towards clearer side
        if speed_factor < 0.15:
            v_des = 0.0
            omega_des = 1.0 if repulsion >= 0.0 else -1.0

        a_v = (v_des / (0.5 * V_MAX)) - 1.0
        a_omega = omega_des / OMEGA_MAX
        return np.array([a_v, a_omega], dtype=np.float32)
