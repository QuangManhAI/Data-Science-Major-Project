"""Pure-Vision V2 hospital environment: camera depth sectors + redesigned reward.

Why V2 (decision D-074 in docs/AI-Human-logs/Manh.md):
- V1 obstacle sectors only ever contained the doctor, so the policy never saw walls/furniture and the
  Godot side needed a hard-coded "back up and spin" unstick maneuver. V2 feeds a camera *depth* reading
  (RGB-D camera / monocular depth on the real robot, 15 physics rays across the camera HFOV in Godot,
  15 analytic rays here) so walls, furniture, people and the doctor are all visible geometrically.
- With geometric depth, the V1 doctor channel and door channel are redundant and are removed.
- V1 reward paid +0.15/step for keeping the patient close (slower escort = more reward), was clipped at
  +100 (delivery escalation / shift bonus never reached the agent) and was discounted away at gamma=0.99.
  V2 uses a small per-step time cost, potential-based path shaping, penalty-only tether shaping and
  unclipped sparse task rewards, so the fastest safe escort is the highest-return behaviour.

Observation (21-D):
  [0..4]   depth proximity per camera sector (far-left .. far-right), 0 = free to 4 m, 1 = touching
  [5..8]   target patient box [seen, cx, h, conf]
  [9..12]  task phase one-hot [SEARCH, APPROACH, GUIDE_ESCORT, CALL_STAFF]
  [13..15] nav goal [path_dist/25, sin, cos]
  [16..17] kinematics [v/V_MAX, omega/OMEGA_MAX]
  [18..20] tether [last_seen_dist/5, patient_visible, time_since_seen/5s]
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.models.ppo_agent import OMEGA_MAX, V_MAX
from src.training.camera_vision_model import CAMERA_HFOV_RAD
from src.training.hospital_env import (
    DESTINATION_POSITIONS,
    ELEVATOR_POSITIONS,
    ESCORT_WALK_SPEED,
    ROBOT_RADIUS_M,
    TaskPhase,
    cast_rays,
    is_blocked,
    nav_grid,
    floor_of,
)
from src.training.hospital_env_pure_vision import HospitalPureVisionEnv
from src.training.hospital_env_vision import PUBLIC_PATROL_WAYPOINTS, TARGET_BOX_H

OBS_DIM_V2 = 21

# --- Depth camera model (must match robot.gd get_depth_readings) ---
DEPTH_NUM_RAYS = 15                 # 3 rays per sector
DEPTH_HFOV_RAD = CAMERA_HFOV_RAD    # ~86.4 deg
DEPTH_MAX_RANGE_M = 4.0
PERSON_RADIUS_M = 0.25

# --- Escort companion ---
PATIENT_ESCORT_SPEED = 0.70         # patient must be able to out-walk the robot's 0.52 m/s escort cap
COMPANION_OFFSETS = ((1.2, 0.45), (1.1, 0.25), (1.0, 0.0))  # (forward, right) m; ~20 deg, inside the 86 deg camera FOV
PATIENT_ROBOT_GAP_M = 0.45

# --- Reward V2 ---
R_STEP = -0.005
K_PATH = 0.20                       # potential shaping per metre of path (approach / escort / return)
K_PATH_SEARCH = 0.05                # weaker on patrol waypoints (cyclic target, not a true potential)
R_SPOTTED = 0.5
R_ENGAGED = 2.0
R_TRANSIT = 0.5
R_DELIVERED = 10.0
R_ALL_SERVED = 5.0
R_COLLISION = -1.0
R_CONTACT = -0.1                    # per step while still pressing against an obstacle
R_NEAR = -0.05                      # scaled by closeness when a front ray is under NEAR_M
NEAR_M = 0.50
R_TETHER = -0.05                    # per metre beyond TETHER_FREE_M, per step
TETHER_FREE_M = 2.0
TETHER_SHAPING_CUTOFF_M = 3.0       # no positive path credit while patient is this far behind
R_STAGNATION = -2.0
GAMMA_V2 = 0.995


def depth_rays_to_sectors(dist_m: np.ndarray) -> np.ndarray:
    """15 ray distances (m, left -> right) -> 5 sector proximities in [0, 1]."""
    d = np.asarray(dist_m, dtype=np.float32).reshape(5, -1).min(axis=1)
    prox = 1.0 - (d - ROBOT_RADIUS_M) / (DEPTH_MAX_RANGE_M - ROBOT_RADIUS_M)
    return np.clip(prox, 0.0, 1.0).astype(np.float32)


def tether_features(last_d: float, visible: bool, unseen_s: float) -> np.ndarray:
    return np.array([np.clip(last_d / 5.0, 0.0, 1.0), 1.0 if visible else 0.0, np.clip(unseen_s / 5.0, 0.0, 1.0)], dtype=np.float32)


def remap_v1_obs(obs29: np.ndarray, sectors: np.ndarray, tether: np.ndarray) -> np.ndarray:
    """Build the 21-D V2 observation from the shared V1 29-D pipeline (patient/phase/goal/kinematics)."""
    o = np.zeros(OBS_DIM_V2, dtype=np.float32)
    o[0:5] = sectors
    o[5:9] = obs29[5:9]
    o[9:13] = obs29[13:17]
    o[13:16] = obs29[17:20]
    o[16:18] = obs29[20:22]
    o[18:21] = tether
    return np.nan_to_num(o, nan=0.0, posinf=1.0, neginf=-1.0)


class HospitalPureVisionV2Env(HospitalPureVisionEnv):
    """V1 dynamics / task flow, V2 observation and reward."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.patient_escort_speed = PATIENT_ESCORT_SPEED
        self.gamma = GAMMA_V2

    # ------------------------------------------------------------------ geometry
    def reset(self, seed: int | None = None) -> np.ndarray:
        self._v2_ready = False
        super().reset(seed)
        # Patients are tracked in self.patients (they move when escorted); V1 also kept a frozen copy
        # of them in self.npcs, which left phantom obstacles at their home seats.
        self.npcs = [n for n in self.npcs if n.get("role") != "patient"]
        self.tether_last_d = 1.0
        self.tether_unseen_s = 0.0
        self.spotted_ids: set[int] = set()
        self.escort_dist_m = 0.0
        self.escort_steps = 0
        self.v2_trail: list[tuple[float, float]] = []
        self.collisions = 0
        self.cumulative_v2_reward = 0.0
        self._v2_ready = True
        self.last_obs_v2 = self._get_obs()
        return self.last_obs_v2

    def _patient_discs(self, skip_target_in: tuple[TaskPhase, ...] = ()) -> list[tuple[float, float]]:
        """People on the robot's floor; the tracked patient is skipped while the phase is in `skip_target_in`."""
        fl = floor_of(self.robot_x)
        out = []
        for i, p in enumerate(getattr(self, "patients", [])):
            if floor_of(p["x"]) != fl:
                continue
            if i == self.active_patient_idx and self.task_phase in skip_target_in:
                continue
            out.append((p["x"], p["z"]))
        if getattr(self, "doctor_active", False) and floor_of(self.doctor_x) == fl:
            out.append((self.doctor_x, self.doctor_z))
        return out

    def _npc_blocks_move(self, x: float, z: float) -> bool:
        if super()._npc_blocks_move(x, z):
            return True
        r = PERSON_RADIUS_M + ROBOT_RADIUS_M
        for px, pz in self._patient_discs(skip_target_in=(TaskPhase.GUIDE_ESCORT,)):
            if math.hypot(x - px, z - pz) < r and math.hypot(x - px, z - pz) < math.hypot(self.robot_x - px, self.robot_z - pz):
                return True
        return False

    # ------------------------------------------------------------ escort follow
    @staticmethod
    def _segment_clear(x0: float, z0: float, x1: float, z1: float, margin: float = 0.28) -> bool:
        n = max(2, int(math.hypot(x1 - x0, z1 - z0) / 0.15) + 1)
        return not any(is_blocked(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t, margin) for t in np.linspace(0.0, 1.0, n)[1:])

    def _step_escorted_patient(self) -> None:
        """Companion point (ahead-right, in camera view) when reachable in a straight line, otherwise walk
        the robot's own breadcrumb trail (always drivable) instead of sliding against furniture."""
        if self.active_patient_idx is None or self.active_patient_idx >= len(self.patients):
            return
        p = self.patients[self.active_patient_idx]
        trail = self.v2_trail
        if not trail or math.hypot(self.robot_x - trail[-1][0], self.robot_z - trail[-1][1]) >= 0.25:
            trail.append((self.robot_x, self.robot_z))
            del trail[:-200]
        if trail and abs(trail[-1][0] - p["x"]) > 25.0:  # elevator teleport: restart trail
            trail.clear()
        if is_blocked(p["x"], p["z"], 0.28):  # stood up inside furniture -> nearest free cell
            g = nav_grid(floor_of(p["x"]))
            si, sj = g.snap(p["x"], p["z"])
            p["x"], p["z"] = float(g.xs[si]), float(g.zs[sj])

        fx, fz = -math.sin(self.robot_yaw), -math.cos(self.robot_yaw)
        rx, rz = math.cos(self.robot_yaw), -math.sin(self.robot_yaw)
        target = None
        for fwd, side in COMPANION_OFFSETS:
            cx, cz = self.robot_x + fx * fwd + rx * side, self.robot_z + fz * fwd + rz * side
            if not is_blocked(cx, cz, 0.28) and self._segment_clear(p["x"], p["z"], cx, cz):
                target = (cx, cz)
                break
        if target is None:
            for tx, tz in reversed(trail):  # newest crumb the patient can walk to directly
                if self._segment_clear(p["x"], p["z"], tx, tz):
                    target = (tx, tz)
                    break
        if target is None and trail:
            target = min(trail, key=lambda c: math.hypot(c[0] - p["x"], c[1] - p["z"]))
        if target is None:
            return
        dx, dz = target[0] - p["x"], target[1] - p["z"]
        dist = math.hypot(dx, dz)
        if dist <= 0.15:
            return
        step = min(self.patient_escort_speed * self.dt, dist)
        nx, nz = p["x"] + dx / dist * step, p["z"] + dz / dist * step
        gx, gz = nx - self.robot_x, nz - self.robot_z
        g = math.hypot(gx, gz)
        if g < PATIENT_ROBOT_GAP_M:  # slide around the robot body instead of stopping behind it
            if g < 1e-6:
                gx, gz, g = rx, rz, 1.0
            nx, nz = self.robot_x + gx / g * PATIENT_ROBOT_GAP_M, self.robot_z + gz / g * PATIENT_ROBOT_GAP_M
        if not is_blocked(nx, nz, 0.28):
            p["x"], p["z"] = nx, nz
            self.patient_stuck_s = 0.0
        else:
            self.patient_stuck_s = getattr(self, "patient_stuck_s", 0.0) + self.dt
            if self.patient_stuck_s >= 2.0 and trail:  # a real person finds their way back onto the robot's path
                cx, cz = min(trail, key=lambda c: math.hypot(c[0] - p["x"], c[1] - p["z"]))
                p["x"], p["z"] = cx, cz
                self.patient_stuck_s = 0.0
        self.patient_x, self.patient_z = p["x"], p["z"]

    def depth_rays(self) -> np.ndarray:
        """True ray distances (m), left -> right, across the camera HFOV."""
        half = DEPTH_HFOV_RAD / 2.0
        rel = np.linspace(-half, half, DEPTH_NUM_RAYS)
        ray_yaw = self.robot_yaw - rel  # +rel = right
        dxs, dzs = -np.sin(ray_yaw), -np.cos(ray_yaw)
        best = cast_rays(self.robot_x, self.robot_z, dxs, dzs, DEPTH_MAX_RANGE_M, self._obstacles())
        # Depth pixels inside the tracked patient's YOLO box are masked out (they are the goal, not an obstacle)
        for px, pz in self._patient_discs(skip_target_in=(TaskPhase.APPROACH_PATIENT, TaskPhase.GUIDE_ESCORT)):
            ox, oz = px - self.robot_x, pz - self.robot_z
            proj = ox * dxs + oz * dzs
            perp_sq = ox * ox + oz * oz - proj * proj
            hit = (proj > 0.0) & (perp_sq < PERSON_RADIUS_M ** 2)
            t = proj - np.sqrt(np.maximum(PERSON_RADIUS_M ** 2 - perp_sq, 0.0))
            best = np.where(hit & (t < best), np.maximum(t, 0.0), best)
        return best.astype(np.float32)

    def _noisy_depth(self, d: np.ndarray) -> np.ndarray:
        if not self.domain_randomization:
            return d
        d = d * (1.0 + self.rng.normal(0.0, 0.03, size=d.shape)) + self.rng.normal(0.0, 0.02, size=d.shape)
        drop = self.rng.random(d.shape) < 0.02  # missing depth pixels read as max range
        d = np.where(drop, DEPTH_MAX_RANGE_M, d)
        return np.clip(d, 0.0, DEPTH_MAX_RANGE_M).astype(np.float32)

    # --------------------------------------------------------------- observation
    def _get_obs(self) -> np.ndarray:
        obs29 = super()._get_obs()
        if not getattr(self, "_v2_ready", False):
            return obs29
        self._true_depth = self.depth_rays()
        sectors = depth_rays_to_sectors(self._noisy_depth(self._true_depth))
        if self.task_phase == TaskPhase.GUIDE_ESCORT:
            seen = obs29[5] > 0.5
            if seen:
                self.tether_last_d = float(np.clip(1.20 / max(float(obs29[7]), 0.05), 0.5, 5.0))
                self.tether_unseen_s = 0.0
            tether = tether_features(self.tether_last_d, seen, self.tether_unseen_s)
        else:
            tether = np.zeros(3, dtype=np.float32)
        return remap_v1_obs(obs29, sectors, tether)

    # -------------------------------------------------------------------- reward
    def _potential_target(self) -> tuple[tuple, tuple[float, float], float] | None:
        """(identity key, target xy, shaping gain) for the current sub-task, or None."""
        fl = floor_of(self.robot_x)
        if getattr(self, "returning_to_lobby", False):
            return ("ret", fl), ELEVATOR_POSITIONS[fl], K_PATH
        if self.task_phase == TaskPhase.SEARCH_PATIENT:
            wps = PUBLIC_PATROL_WAYPOINTS.get(fl, PUBLIC_PATROL_WAYPOINTS[1])
            i = self.patrol_wp_idx % len(wps)
            return ("pat", fl, i), wps[i], K_PATH_SEARCH
        if self.task_phase == TaskPhase.APPROACH_PATIENT and self.active_patient_idx is not None:
            return ("app", fl, self.active_patient_idx), (self.patient_x, self.patient_z), K_PATH
        if self.task_phase == TaskPhase.GUIDE_ESCORT:
            return ("esc", fl, self.active_patient_idx, self.goal_key), DESTINATION_POSITIONS.get(self.goal_key, (0.0, 4.5)), K_PATH
        return None

    def _path_dist(self, target: tuple[float, float]) -> float:
        grid, fld, _, _ = self._nav_field(*target)
        return float(min(grid.distance(fld, self.robot_x, self.robot_z), 60.0))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        pt0 = self._potential_target()
        d0 = self._path_dist(pt0[1]) if pt0 else 0.0
        col0 = self.collisions
        was_escort = self.task_phase == TaskPhase.GUIDE_ESCORT
        x0, z0 = self.robot_x, self.robot_z

        _, _, done, info = super().step(action)   # V1 dynamics + task transitions; V1 reward discarded
        if info.get("patient_engaged"):
            self.v2_trail.clear()
        obs = self._get_obs()                     # fresh obs after transitions (V1 returned a stale one)
        self.last_obs_v2 = obs

        r = R_STEP
        # potential-based path shaping, only while the sub-task target is unchanged
        pt1 = self._potential_target()
        if pt0 and pt1 and pt0[0] == pt1[0]:
            d1 = self._path_dist(pt1[1])
            # undiscounted difference: gamma*Phi(s')-Phi(s) with Phi<0 would pay (1-gamma)*K*d for standing still
            shaping = pt0[2] * (d0 - d1)
            d_tether = math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
            if pt1[0][0] == "esc" and d_tether > TETHER_SHAPING_CUTOFF_M:
                shaping = min(shaping, 0.0)
            r += shaping

        # task events
        if self.active_patient_idx is not None and self.task_phase == TaskPhase.APPROACH_PATIENT and self.active_patient_idx not in self.spotted_ids:
            self.spotted_ids.add(self.active_patient_idx)
            r += R_SPOTTED
        if info.get("patient_engaged"):
            r += R_ENGAGED
        if info.get("elevator_transit") or info.get("return_transit"):
            r += R_TRANSIT
        if info.get("guide_completed"):
            r += R_DELIVERED
        if info.get("exit_reason") == "ALL_PATIENTS_SERVED":
            r += R_ALL_SERVED
        if info.get("exit_reason") == "STAGNATION_TIMEOUT":
            r += R_STAGNATION

        # safety
        if self.collisions > col0:
            r += R_COLLISION
        elif info.get("collided"):
            r += R_CONTACT
        dmin = float(np.min(self._true_depth[3:12]))  # central ~50 deg
        if dmin < NEAR_M:
            r += R_NEAR * float(np.clip((NEAR_M - dmin) / (NEAR_M - ROBOT_RADIUS_M), 0.0, 1.0))

        # escort: penalty-only tether + speed bookkeeping
        if self.task_phase == TaskPhase.GUIDE_ESCORT:
            d_tether = math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
            r += R_TETHER * float(np.clip(d_tether - TETHER_FREE_M, 0.0, 3.0))
            if obs[19] < 0.5:
                self.tether_unseen_s += self.dt
        if was_escort and abs(self.robot_x - x0) < 5.0:  # skip elevator teleports
            self.escort_dist_m += math.hypot(self.robot_x - x0, self.robot_z - z0)
            self.escort_steps += 1

        if not math.isfinite(r):
            r = 0.0
        r = float(np.clip(r, -20.0, 20.0))
        self.cumulative_v2_reward = getattr(self, "cumulative_v2_reward", 0.0) + r
        info["cumulative_team_reward"] = self.cumulative_v2_reward
        info["escort_speed"] = self.escort_dist_m / max(1, self.escort_steps) / self.dt
        info["truncated"] = info.get("exit_reason") == "SHIFT_COMPLETED"
        return obs, r, done, info

    # ------------------------------------------------------------------- teacher
    def teacher_action(self) -> np.ndarray:
        """Rule-based controller on the V2 observation (no side effects). Also the classical baseline."""
        o = self.last_obs_v2
        s = o[0:5]
        phase = int(np.argmax(o[9:13]))
        pat_seen, pat_cx, pat_h = o[5] > 0.5, float(o[6]), float(o[7])

        # +omega = turn LEFT. Obstacles on the left push right (negative), on the right push left.
        repulsion = -0.9 * (s[0] + 0.6 * s[1]) + 0.9 * (s[4] + 0.6 * s[3])
        front = float(max(s[2], 0.9 * max(s[1], s[3])))  # door frames at +-15 deg weigh less than dead ahead

        if phase == TaskPhase.APPROACH_PATIENT and pat_seen:
            omega = -2.0 * pat_cx * OMEGA_MAX
            close = pat_h >= 0.8 * TARGET_BOX_H
            if not close:
                omega += 0.5 * repulsion
            v = 0.0 if pat_h >= TARGET_BOX_H else V_MAX * max(0.2, 1.0 - abs(pat_cx) / 0.3)
            if not close:
                v *= float(np.clip(1.0 - max(0.0, s[2] - 0.75) / 0.2, 0.0, 1.0))
            v_lim = V_MAX
        else:
            goal_angle = math.atan2(float(o[14]), float(o[15]))
            v_lim = ESCORT_WALK_SPEED if phase == TaskPhase.GUIDE_ESCORT else V_MAX
            omega = 1.6 * goal_angle + repulsion
            v = v_lim * float(np.clip(1.0 - max(0.0, front - 0.80) / 0.13, 0.0, 1.0))
            if abs(goal_angle) > 0.9:
                v *= 0.3
            if phase == TaskPhase.GUIDE_ESCORT and (o[20] > 0.4 or o[18] > 0.45):
                v = min(v, 0.15)  # patient lost > 2 s or > 2.25 m behind: wait for them
            # Blocked ahead: rotate in place, committing to one direction until the front clears
            turn = getattr(self, "_teacher_turn", 0.0)
            if front > 0.92:
                if turn == 0.0:
                    left_blk, right_blk = s[0] + s[1], s[3] + s[4]
                    turn = 1.0 if goal_angle >= 0.0 else -1.0          # prefer the goal side...
                    if (turn > 0 and left_blk > right_blk + 0.4) or (turn < 0 and right_blk > left_blk + 0.4):
                        turn = -turn                                   # ...unless it is clearly more blocked
                v, omega = 0.0, turn * OMEGA_MAX
            elif front < 0.80:
                turn = 0.0
            elif turn != 0.0:
                omega = turn * OMEGA_MAX * 0.6
            self._teacher_turn = turn
        a_v = float(np.clip(2.0 * v / v_lim - 1.0, -1.0, 1.0))
        a_w = float(np.clip(omega / OMEGA_MAX, -1.0, 1.0))
        return np.array([a_v, a_w], dtype=np.float32)
