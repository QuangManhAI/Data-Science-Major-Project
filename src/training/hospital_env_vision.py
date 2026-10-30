"""Pure Visual-Search RL Environment for Hospital Robot Navigation.

Eliminates the oracle coordinate cheat during the patient search phase.
The robot receives ZERO coordinates for the patient (obs[25..27] = 0).
The robot must explore public areas using 17 LiDAR rays and visually acquire
patients via simulated Pinhole Perspective Camera / YOLOv8 detections.

Once a patient is acquired and dialogue concludes, fixed landmark navigation
(clinic rooms, elevators) activates to guide the escorted patient safely.
"""

from __future__ import annotations

import collections
import math
from typing import Any

import numpy as np

from src.training.camera_vision_model import compute_pinhole_detection, CAMERA_HFOV_RAD
from src.training.hospital_env import (
    BONUS_TETHER_GOOD,
    COLLISION_LIDAR_M,
    DESTINATION_POSITIONS,
    DOCK_START_PROB,
    ESCORT_CRUMB_REACH_M,
    ESCORT_CRUMB_SPACING_M,
    ESCORT_STOP_DIST_M,
    ESCORT_TRAIL_JUMP_M,
    ESCORT_WALK_SPEED,
    GOAL_ARRIVE_PATIENT_M,
    GOAL_ARRIVE_ROBOT_M,
    HospitalShiftEnv,
    NavGrid,
    OBS_DIM,
    OMEGA_MAX,
    PATIENT_TALK_DIST_M,
    PENALTY_COLLISION,
    PENALTY_PROXIMITY,
    PENALTY_STEP_TIME,
    PENALTY_TETHER_STRETCH,
    PROXIMITY_LIDAR_M,
    REWARD_ANSWER,
    REWARD_CALL_STAFF,
    REWARD_ELEVATOR_TRANSIT,
    REWARD_GUIDE_ARRIVED,
    REWARD_PATIENT_ENGAGED,
    ROBOT_BOUND_X_M,
    ROBOT_BOUND_Z_M,
    ROBOT_RADIUS_M,
    SHAPING_PROGRESS_SCALE,
    TaskPhase,
    TETHER_MAX_M,
    V_MAX,
    _STRUCTURE,
    cast_rays,
    floor_of,
    is_blocked,
    nav_grid,
    resolve_destination_key,
    resolve_final_target,
    signed_angle_rad,
)

# Visual-specific reward shaping constants
REWARD_VISUAL_SPOTTED = 2.0         # One-time bonus when patient first enters camera FOV
REWARD_CELL_EXPLORATION = 0.05      # Curiosity reward per newly visited 1m x 1m grid cell
WEIGHT_CENTERING_PENALTY = 0.3      # -0.3 * |cx| to align robot's camera heading to patient
WEIGHT_APPROACH_EXPANSION = 1.5     # +1.5 * delta_h to reward driving towards patient
TARGET_BOX_H = 0.85                 # Config.H_TARGET: corresponds to distance <= 1.35m

# Public corridor patrol waypoints per floor (static hospital map coordinates, NO patient GPS)
PUBLIC_PATROL_WAYPOINTS: dict[int, list[tuple[float, float]]] = {
    1: [(0.0, 4.5), (7.0, 0.0), (-7.0, 0.0), (0.0, 0.0)],
    2: [(50.0, 4.5), (57.0, 0.0), (43.0, 0.0), (50.0, 0.0)],
    3: [(100.0, 4.5), (107.0, 0.0), (93.0, 0.0), (100.0, 0.0)],
}


class HospitalVisualShiftEnv(HospitalShiftEnv):
    """Pure Visual-Search & LiDAR-Assisted Hospital Environment.

    Zero-oracle search: the robot NEVER receives ground-truth coordinates of the patient.
    - When searching (has_det == 0), it patrols public hospital corridors/lobby.
    - When a person is detected (has_det == 1), range and bearing are computed
      STRICTLY FROM THE 2D CAMERA BOUNDING BOX (cx, h) via optical pinhole projection.
    """

    def __init__(
        self,
        curriculum_stage: int = 1,
        shift_duration_s: float = 180.0,
        dt: float = 0.2,
        seed: int = 42,
        env_mode: str = "solo",
        stagnation_timeout_s: float = 45.0,
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
        )
        self.domain_randomization = bool(domain_randomization)
        self.visited_cells: set[tuple[int, int, int]] = set()
        self.prev_has_det: float = 0.0
        self.prev_vision_h: float = 0.0
        self.best_vision_h: float = 0.0
        self.patrol_wp_idx: int = 0
        self.last_det: dict[str, float] = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = super().reset(seed=seed)
        self.visited_cells.clear()
        cell = (int(math.floor(self.robot_x)), int(math.floor(self.robot_z)), floor_of(self.robot_x))
        self.visited_cells.add(cell)
        self.prev_has_det = 0.0
        self.prev_vision_h = 0.0
        self.best_vision_h = 0.0
        self.patrol_wp_idx = int(self.rng.integers(0, 4))
        self.last_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
        return self._get_obs()

    def _check_los_occluded(self, x1: float, z1: float, x2: float, z2: float) -> bool:
        """Raycast check against hospital walls between camera (x1, z1) and target (x2, z2)."""
        fl = floor_of(x1)
        if fl not in _STRUCTURE:
            return False
        dx = x2 - x1
        dz = z2 - z1
        dist = math.hypot(dx, dz)
        if dist < 1e-4:
            return False
        ux, uz = dx / dist, dz / dist
        hit = cast_rays(x1, z1, np.array([ux], dtype=np.float32), np.array([uz], dtype=np.float32), dist, _STRUCTURE[fl])
        return bool(hit[0] < dist - 0.12)

    def _compute_yolo_det(self) -> tuple[float, float, float, float]:
        """Compute visual detection using Pinhole Camera model with LOS raycasting."""
        # Patient is only visible if on the same floor as robot
        if floor_of(self.patient_x) != floor_of(self.robot_x):
            self.last_det = {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
            return 0.0, 0.0, 0.0, 0.0

        is_seated = bool(self.patient_seated) if hasattr(self, "patient_seated") else False
        det = compute_pinhole_detection(
            robot_x=self.robot_x,
            robot_z=self.robot_z,
            robot_yaw=self.robot_yaw,
            target_x=self.patient_x,
            target_z=self.patient_z,
            target_seated=is_seated,
            is_blocked_fn=self._check_los_occluded,
            domain_randomization=self.domain_randomization,
            rng=self.rng,
        )
        self.last_det = det
        conf = float(np.clip(0.95 - 0.04 * math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z), 0.45, 0.98)) if det["seen"] > 0.5 else 0.0
        return det["seen"], det["cx"], det["h"], conf

    def _get_obs(self) -> np.ndarray:
        """Build observation vector. Enforces ZERO-ORACLE during patient search."""
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

        # Navigation goal representation:
        # 1. During APPROACH_PATIENT with person in view: range and bearing derived strictly from 2D camera bounding box
        # 2. During SEARCH_PATIENT: navigate between public corridor exploration waypoints (NO patient GPS)
        # 3. During GUIDE_ESCORT or CALL_STAFF: navigate to clinic room or elevator landmark
        if self.task_phase == TaskPhase.APPROACH_PATIENT and has_det > 0.5:
            d_vis = float(np.clip(1.20 / max(h, 0.05), 0.8, 15.0))
            angle_vis = float(-math.atan(cx_off * math.tan(CAMERA_HFOV_RAD / 2.0)))
            obs[25] = float(np.clip(d_vis / 25.0, 0.0, 1.0))
            obs[26] = math.sin(angle_vis)
            obs[27] = math.cos(angle_vis)
        elif self.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
            fl = floor_of(self.robot_x)
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

            obs[25] = float(np.clip(self.nav_remaining / 25.0, 0.0, 1.0))
            obs[26] = math.sin(angle_goal)
            obs[27] = math.cos(angle_goal)
        else:
            # During GUIDE_ESCORT or CALL_STAFF, landmark goal navigation is active
            self.set_nav_target(*self._nav_final_target())
            to_gx = self.goal_x - self.robot_x
            to_gz = self.goal_z - self.robot_z
            fwd_x = -math.sin(self.robot_yaw)
            fwd_z = -math.cos(self.robot_yaw)
            angle_goal = signed_angle_rad(fwd_x, fwd_z, to_gx, to_gz)

            obs[25] = float(np.clip(self.nav_remaining / 25.0, 0.0, 1.0))
            obs[26] = math.sin(angle_goal)
            obs[27] = math.cos(angle_goal)

        d_tether = (
            math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
            if self.task_phase == TaskPhase.GUIDE_ESCORT
            else 0.0
        )
        obs[28] = float(np.clip(d_tether / 5.0, 0.0, 1.0))
        obs[29] = float(np.clip(self.robot_v / V_MAX, 0.0, 1.0))
        obs[30] = float(np.clip(self.robot_omega / OMEGA_MAX, -1.0, 1.0))
        return obs

    def step(self, action: np.ndarray | list[float] | tuple[float, float]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self.step_count += 1
        self.steps_since_task += 1
        a_v = float(np.clip(action[0], -1.0, 1.0))
        a_omega = float(np.clip(action[1], -1.0, 1.0))

        v_limit = ESCORT_WALK_SPEED if self.task_phase == TaskPhase.GUIDE_ESCORT else V_MAX
        v = float(np.clip((a_v + 1.0) * 0.5 * v_limit, 0.0, v_limit))
        omega = float(np.clip(a_omega * OMEGA_MAX, -OMEGA_MAX, OMEGA_MAX))

        # Measure path progress only when destination landmark is active
        prev_dist_final = 0.0
        if self.task_phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF):
            nav, nav_fld, _, _ = self._nav_field(*self._nav_final_target())
            prev_dist_final = nav.distance(nav_fld, self.robot_x, self.robot_z)

        # Integrate robot kinematics with wall collision check
        self.robot_yaw = ((self.robot_yaw + omega * self.dt + math.pi) % (2.0 * math.pi)) - math.pi
        nx = self.robot_x - math.sin(self.robot_yaw) * v * self.dt
        nz = self.robot_z - math.cos(self.robot_yaw) * v * self.dt

        x_base = 100.0 if self.robot_x >= 75.0 else (50.0 if self.robot_x >= 25.0 else 0.0)

        def free(px: float, pz: float) -> bool:
            return (
                abs(px - x_base) <= ROBOT_BOUND_X_M
                and abs(pz) <= ROBOT_BOUND_Z_M
                and not is_blocked(px, pz, ROBOT_RADIUS_M)
                and not self._npc_blocks_move(px, pz)
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

        # Step escorted patient trail
        if self.task_phase == TaskPhase.GUIDE_ESCORT:
            self._step_escorted_patient()

        d_tether = math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
        min_lidar_m = float(np.min(self.compute_lidar())) * 5.0

        # Update visual detection state
        has_det, cx_off, h, conf = self._compute_yolo_det()
        det = self.last_det

        event = ""
        r_exploration = 0.0
        r_visual = 0.0

        # Visual exploration and acquisition reward shaping
        if self.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
            # 1. Grid cell coverage reward (curiosity exploration)
            cur_cell = (int(math.floor(self.robot_x)), int(math.floor(self.robot_z)), floor_of(self.robot_x))
            if cur_cell not in self.visited_cells:
                self.visited_cells.add(cur_cell)
                r_exploration += REWARD_CELL_EXPLORATION
                self.steps_since_task = 0  # Active spatial exploration resets stagnation

            # 2. Visual acquisition reward
            if det["seen"] > 0.5:
                if self.prev_has_det < 0.5:
                    r_visual += REWARD_VISUAL_SPOTTED  # First spotting bonus!
                    self.steps_since_task = 0          # Visual target acquired!
                # Centering reward (align camera heading to person)
                r_visual -= WEIGHT_CENTERING_PENALTY * abs(det["cx"])
                # Approach sizing reward (getting closer enlarges bounding box)
                r_visual += WEIGHT_APPROACH_EXPANSION * max(0.0, det["h"] - self.prev_vision_h)
                if det["h"] > self.best_vision_h + 0.04:
                    self.best_vision_h = det["h"]
                    self.steps_since_task = 0          # Closing distance to visual target!

            self.prev_has_det = det["seen"]
            self.prev_vision_h = det["h"]

            # 3. Check dialogue engagement: bounding box height >= 0.85 or physical dist <= 1.35m
            dist_to_pat = math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
            if (det["seen"] > 0.5 and det["h"] >= TARGET_BOX_H) or (dist_to_pat <= PATIENT_TALK_DIST_M and det["seen"] > 0.5):
                resolved = resolve_destination_key(
                    self.scenario_action, self.scenario_slot, self.scenario_floor,
                    current_floor=floor_of(self.patient_x),
                )
                event = "engaged"
                self.task_phase = TaskPhase.GUIDE_ESCORT if self.scenario_action == "guide" else TaskPhase.CALL_STAFF
                self.goal_key = resolved if self.scenario_action == "guide" else "staff"
                if self.scenario_action in ("call_staff", "emergency"):
                    self.staff_goal = self._nearest_staff_position()
                self.steps_since_task = 0
                self.best_goal_dist = None
                self.best_vision_h = 0.0

        elif self.task_phase == TaskPhase.GUIDE_ESCORT:
            dest_x, dest_z = DESTINATION_POSITIONS.get(self.goal_key, (0.0, 4.5))
            d_robot_dest = math.hypot(dest_x - self.robot_x, dest_z - self.robot_z)
            d_patient_dest = math.hypot(dest_x - self.patient_x, dest_z - self.patient_z)
            if d_robot_dest <= GOAL_ARRIVE_ROBOT_M and d_patient_dest <= GOAL_ARRIVE_PATIENT_M:
                final_target = resolve_final_target(self.scenario_action, self.scenario_slot, self.scenario_floor)
                if self.goal_key in ("thang_may", "thang_may_t2", "thang_may_t3") and final_target != self.goal_key:
                    event = "elevator_transit"
                    cur_fl = floor_of(self.robot_x)
                    dest_fl = floor_of(DESTINATION_POSITIONS.get(final_target, (0.0, 4.5))[0])
                    shift = float(dest_fl - cur_fl) * 50.0
                    self.robot_x += shift
                    self.patient_x += shift
                    self.goal_key = final_target
                    self.steps_since_task = 0
                    self.best_goal_dist = None
                    self.best_vision_h = 0.0
                else:
                    event = "guide_completed"
                    self.patients_helped += 1
                    self.steps_since_task = 0
                    self.best_goal_dist = None
                    self.best_vision_h = 0.0
                    self._spawn_next_patient()

        elif self.task_phase == TaskPhase.CALL_STAFF:
            dest_x, dest_z = self.staff_goal
            if math.hypot(dest_x - self.robot_x, dest_z - self.robot_z) <= GOAL_ARRIVE_ROBOT_M:
                event = "call_staff"
                self.patients_helped += 1
                self.steps_since_task = 0
                self.best_goal_dist = None
                self.best_vision_h = 0.0
                self._spawn_next_patient()

        # Compute rewards
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

        # Landmark path progress shaping (only during escort or staff calling)
        r_progress = 0.0
        if self.task_phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF):
            nav, nav_fld, _, _ = self._nav_field(*self._nav_final_target())
            curr_dist_final = nav.distance(nav_fld, self.robot_x, self.robot_z)
            r_progress = SHAPING_PROGRESS_SCALE * (prev_dist_final - curr_dist_final)
            if self.best_goal_dist is None:
                self.best_goal_dist = curr_dist_final
            elif curr_dist_final < self.best_goal_dist - 1.0:
                self.best_goal_dist = curr_dist_final
                self.steps_since_task = 0  # Escort progress along shortest path resets stagnation!

        # Tether shaping
        r_tether = 0.0
        if self.task_phase == TaskPhase.GUIDE_ESCORT:
            if d_tether > TETHER_MAX_M:
                r_progress = 0.0
                r_tether = PENALTY_TETHER_STRETCH
            elif 0.9 <= d_tether <= 2.2:
                r_tether = BONUS_TETHER_GOOD

        # Safety & step penalty
        r_safety = PENALTY_STEP_TIME
        collided = hit_wall or (min_lidar_m < COLLISION_LIDAR_M)
        if collided:
            r_safety += PENALTY_COLLISION
            self.collisions += 1
        elif min_lidar_m < PROXIMITY_LIDAR_M:
            r_safety += PENALTY_PROXIMITY * (PROXIMITY_LIDAR_M - min_lidar_m) / (PROXIMITY_LIDAR_M - COLLISION_LIDAR_M)

        total_reward = r_team + r_progress + r_tether + r_safety + r_exploration + r_visual
        self.cumulative_team_reward += total_reward

        done = self.step_count >= int(self.shift_duration_s / self.dt)
        exit_reason = ""
        if self.stagnation_timeout_s > 0.0 and self.steps_since_task * self.dt >= self.stagnation_timeout_s:
            total_reward += self.stagnation_penalty
            done = True
            exit_reason = "STAGNATION_TIMEOUT"
        elif done:
            exit_reason = "SHIFT_COMPLETED"

        info = {
            "r_team": r_team,
            "cumulative_team_reward": self.cumulative_team_reward,
            "r_progress": r_progress,
            "r_tether": r_tether,
            "r_safety": r_safety,
            "r_exploration": r_exploration,
            "r_visual": r_visual,
            "event": event,
            "patients_helped": self.patients_helped,
            "collisions": self.collisions,
            "collided": collided,
            "task_phase": self.task_phase.name,
            "has_det": det["seen"],
            "box_h": det["h"],
            "box_cx": det["cx"],
            "steps_since_task": self.steps_since_task,
            "exit_reason": exit_reason,
        }
        return self._get_obs(), total_reward, done, info

    def teacher_action(self) -> np.ndarray:
        """Visual-aware and landmark-aware teacher action. Zero-oracle during search."""
        lidar = self.compute_lidar()
        mid = len(lidar) // 2
        left_clear = float(np.sum(lidar[:mid]))
        right_clear = float(np.sum(lidar[mid + 1 :]))
        center_min = float(min(lidar[mid - 1], lidar[mid], lidar[mid + 1]))

        repel = 0.0
        speed_factor = 1.0
        if center_min < 0.65:
            turn_strength = float(np.clip((0.65 - center_min) / 0.5, 0.4, 1.2))
            repel = turn_strength if left_clear >= right_clear else -turn_strength
            speed_factor = float(np.clip(center_min / 0.65, 0.35, 1.0))

        if self.task_phase == TaskPhase.APPROACH_PATIENT:
            det = self.last_det
            if det["seen"] > 0.5:
                # Steer towards bounding box center cx
                turn_cmd = float(np.clip(-2.2 * det["cx"] + repel, -OMEGA_MAX, OMEGA_MAX))
                v_target = V_MAX * speed_factor * max(0.25, 1.0 - abs(det["cx"]))
                if det["h"] > 0.7:
                    v_target *= 0.4
                a_v = float(np.clip(2.0 * (v_target / V_MAX) - 1.0, -1.0, 1.0))
                a_omega = float(np.clip(turn_cmd / OMEGA_MAX, -1.0, 1.0))
                return np.array([a_v, a_omega], dtype=np.float32)

        if self.task_phase == TaskPhase.SEARCH_PATIENT:
            fl = floor_of(self.robot_x)
            waypoints = PUBLIC_PATROL_WAYPOINTS.get(fl, PUBLIC_PATROL_WAYPOINTS[1])
            pw_x, pw_z = waypoints[self.patrol_wp_idx % len(waypoints)]
            self.set_nav_target(pw_x, pw_z)
            gx, gz = self.goal_x, self.goal_z
            fwd_x = -math.sin(self.robot_yaw)
            fwd_z = -math.cos(self.robot_yaw)
            angle = signed_angle_rad(fwd_x, fwd_z, gx - self.robot_x, gz - self.robot_z)
            omega = float(np.clip(1.8 * angle + repel, -OMEGA_MAX, OMEGA_MAX))
            v = float(np.clip(0.6 * self.nav_remaining, 0.0, V_MAX * speed_factor))
            if abs(angle) > 0.65:
                v *= 0.25
            a_v = float(np.clip(2.0 * (v / V_MAX) - 1.0, -1.0, 1.0))
            a_omega = float(np.clip(omega / OMEGA_MAX, -1.0, 1.0))
            return np.array([a_v, a_omega], dtype=np.float32)

        return super().teacher_action()
