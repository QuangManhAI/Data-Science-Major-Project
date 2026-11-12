"""Engagement V3 hospital environment: robot learns to face, ask and remember the person it helps.

Design: docs/RL_V3_DESIGN.md. Builds on HospitalPureVisionV2Env (navigation / escort / reward V2 unchanged) and adds:
- people who need help or not, who call the robot (voice + direction) or wait to be asked, who face the robot when
  talking to it;
- a head (camera) pitch the robot controls, so a standing person's face very close to the robot is only in view
  when the robot looks up;
- an "ask" action ("Cần tôi giúp gì không ạ?", reaches the person nearest the image centre within 2.5 m, face
  not needed: they turn round) and a dialogue phase in which the robot must stand still and keep
  the speaker's face in view long enough to remember them (face enrolment) before the escort starts.

Observation (31-D) = V2 21-D +
  [21]     voice active (someone talking / calling)
  [22..23] voice direction relative to the robot heading (sin, cos); (0, 0) when silent
  [24]     face seen
  [25..26] face centre in the image (cx right +, cy down +), each in [-0.5, 0.5]
  [27]     face size (face box height / image height)
  [28]     head pitch, normalised (-1 = 15 deg down, 0 = level, 1 = 30 deg up)
  [29]     time since the last ask / 5 s (1 = long ago / never)
  [30]     in dialogue
Action (4-D, each in [-1, 1]): [v, omega] as V2, [2] head pitch target, [3] ask when > 0.5.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.models.ppo_agent import OMEGA_MAX
from src.training.camera_vision_model import CAMERA_HEIGHT_M, CAMERA_HFOV_RAD, CAMERA_MOUNT_FORWARD_M, CAMERA_VFOV_RAD
from src.training.hospital_env import TaskPhase, floor_of
from src.training.hospital_env_pure_vision_v2 import HospitalPureVisionV2Env, OBS_DIM_V2
from src.training.hospital_env_vision import TARGET_BOX_H

OBS_DIM_V3 = 35
DOA_HOLDS_MEMORY = False         # round 8 tried o[22:24] = remembered direction while silent: worse, off
VOICE_MEMORY_S = 10.0             # remembered voice direction (odometry-rotated) is dropped after this
ACT_DIM_V3 = 4

# --- head / camera pitch ---
TILT_DOWN_RAD = math.radians(15.0)
TILT_UP_RAD = math.radians(30.0)
TILT_RATE_RAD_S = math.radians(60.0)

# --- faces (YuNet-like) ---
FACE_H_SEATED_M = 1.15
FACE_H_STANDING_M = 1.60
FACE_SIZE_M = 0.22
FACE_MAX_RANGE_M = 4.5
FACE_SURE_RANGE_M = 3.0
FACE_FRONTAL_COS = math.cos(math.radians(60.0))
FACE_VFOV_MARGIN_RAD = math.radians(3.0)
FACE_HFOV_FRAC = 0.95

# --- people ---
NEEDS_HELP_PROB = 0.8
CALLER_PROB = 0.5                  # of the people who need help
CALL_RANGE_M = 4.0
CALL_UTTER_S = 2.5
CALL_PAUSE_S = 4.0
CALL_MAX = 3
FACE_TURN_RAD_S = math.radians(90.0)
DECLINE_UTTER_S = 1.5
THANKS_UTTER_S = 2.0               # delivered patient turns to the robot and thanks it (as in Godot)

# --- ask / dialogue ---
ASK_THRESHOLD = 0.5
ASK_COOLDOWN_S = 3.0
ASK_RANGE_M = 2.5
ASK_CENTRE_CX = 0.3                # face must be this close to the image centre to be the one asked
DIALOGUE_S = (3.0, 5.0)
ENROLL_NEED_S = 1.0
DIALOGUE_EXTRA_S = 4.0
DIALOGUE_LEAVE_M = 1.2             # robot drives this far from where the dialogue started -> the person gives up
# Walk-up talker (as Godot npc.gd: an NPC starts talking once the robot stops / slows within TALK_TRIGGER_DIST, whether
# or not the robot is looking at them) -> the policy must turn to the voice during the dialogue to see the face.
WALKUP_RANGE_M = 2.0
WALKUP_MAX_V = 0.35
WALKUP_RATE_HZ = 0.4               # per second while the conditions hold
# Godot NPCs often stand up and step right up to the robot before talking (face high, big box, maybe off to the side)
WALKUP_STEP_IN_PROB = 0.5
WALKUP_STEP_IN_M = (0.6, 1.3)
WALKUP_STEP_IN_BEARING_RAD = math.radians(100.0)

# --- reward V3 (added to V2) ---
R_ENROLLED = 1.0
R_FACE_DIALOGUE = 0.03
R_NOT_ENROLLED = -1.0
R_MOVE_DIALOGUE = -0.05
R_DIALOGUE_ABORT = -1.0
R_ASK_EMPTY = -0.5
R_ASK_REPEAT = -0.3
R_ASK_DECLINED = -0.1
R_CALL_IGNORED = -0.01
R_CALLER_GAVE_UP = -1.0
R_TILT_MOVING = -0.003
# Potential-based shaping during the dialogue: reward per radian the speaker's bearing gets closer to straight ahead
# (Φ = −K·|bearing|) -> gradient on the sign of the voice direction; does not change the optimal policy.
R_TURN_TO_SPEAKER = 0.3


class HospitalEngageV3Env(HospitalPureVisionV2Env):
    """V2 dynamics + people who call / wait to be asked, head pitch, ask action, dialogue with face enrolment."""

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None) -> np.ndarray:
        self._v3_ready = False
        super().reset(seed)
        for p in self.patients:
            needs = bool(self.rng.random() < NEEDS_HELP_PROB) if self.domain_randomization else True
            p.update({
                "needs_help": needs,
                "caller": bool(needs and self.rng.random() < CALLER_PROB),
                "face_yaw": float(self.rng.uniform(-math.pi, math.pi)),
                "declined": False,
                "calls_done": 0,
                "call_t": 0.0,            # > 0: calling out (utterance time left)
                "pause_t": 0.0,           # > 0: pause between calls
                "gave_up": False,
                "engaged": False,         # finished a dialogue with the robot (being / about to be escorted)
                "speak_t": 0.0,           # other speech (decline / dialogue) time left
            })
        self.tilt = 0.0
        self.tilt_target = 0.0
        self.since_ask_s = 99.0
        self.dialogue: dict[str, Any] | None = None
        self.dialogue_done = False
        self.v3_stats = {"asks": 0, "walkup": 0, "walkup_enrolled": 0, "asks_empty": 0, "asks_repeat": 0, "asks_declined": 0, "dialogues": 0,
                         "enrolled": 0, "not_enrolled": 0, "aborted": 0, "callers_gave_up": 0, "face_steps": 0,
                         "dialogue_steps": 0}
        self.last_face: dict[str, float] | None = None
        self.last_voice: tuple[float, float] | None = None
        self.voice_mem: tuple[float, float, float] | None = None   # world direction (x, z) + time heard (s)
        self._v3_ready = True
        self.last_obs_v2 = self._get_obs()
        return self.last_obs_v2

    # ------------------------------------------------------------------ geometry
    def _rel_angle(self, x: float, z: float) -> tuple[float, float]:
        """(bearing relative to the robot heading, + = right; horizontal distance) from the camera mount."""
        cam_x = self.robot_x - math.sin(self.robot_yaw) * CAMERA_MOUNT_FORWARD_M
        cam_z = self.robot_z - math.cos(self.robot_yaw) * CAMERA_MOUNT_FORWARD_M
        dx, dz = x - cam_x, z - cam_z
        fwd = dx * -math.sin(self.robot_yaw) + dz * -math.cos(self.robot_yaw)
        right = dx * math.cos(self.robot_yaw) + dz * -math.sin(self.robot_yaw)
        return math.atan2(right, fwd), math.hypot(dx, dz)

    def _rel_to_world_dir(self, rel: float) -> tuple[float, float]:
        fwd = (-math.sin(self.robot_yaw), -math.cos(self.robot_yaw))
        right = (math.cos(self.robot_yaw), -math.sin(self.robot_yaw))
        return fwd[0] * math.cos(rel) + right[0] * math.sin(rel), fwd[1] * math.cos(rel) + right[1] * math.sin(rel)

    def _world_dir_to_rel(self, dx: float, dz: float) -> float:
        fwd = dx * -math.sin(self.robot_yaw) + dz * -math.cos(self.robot_yaw)
        right = dx * math.cos(self.robot_yaw) + dz * -math.sin(self.robot_yaw)
        return math.atan2(right, fwd)

    def _face_of(self, i: int) -> dict[str, float] | None:
        """YuNet-like face detection of patient i, or None."""
        p = self.patients[i]
        if floor_of(p["x"]) != floor_of(self.robot_x):
            return None
        a, d = self._rel_angle(p["x"], p["z"])
        if d < 0.25 or d > FACE_MAX_RANGE_M or abs(a) > CAMERA_HFOV_RAD * 0.5 * FACE_HFOV_FRAC:
            return None
        # the person must face the robot (frontal face)
        fdx, fdz = -math.sin(p["face_yaw"]), -math.cos(p["face_yaw"])
        tx, tz = (self.robot_x - p["x"]) / max(d, 1e-6), (self.robot_z - p["z"]) / max(d, 1e-6)
        if fdx * tx + fdz * tz < FACE_FRONTAL_COS:
            return None
        h_face = FACE_H_SEATED_M if p["seated"] else FACE_H_STANDING_M
        rel_elev = math.atan2(h_face - CAMERA_HEIGHT_M, d) - self.tilt
        if abs(rel_elev) > CAMERA_VFOV_RAD * 0.5 - FACE_VFOV_MARGIN_RAD:
            return None
        if self._check_los_occluded(self.robot_x, self.robot_z, p["x"], p["z"]):
            return None
        if d > FACE_SURE_RANGE_M and self.domain_randomization:
            p_det = 1.0 - 0.5 * (d - FACE_SURE_RANGE_M) / (FACE_MAX_RANGE_M - FACE_SURE_RANGE_M)
            if self.rng.random() > p_det:
                return None
        cx = float(np.clip(math.tan(a) / (2.0 * math.tan(CAMERA_HFOV_RAD * 0.5)), -0.5, 0.5))
        cy = float(np.clip(-math.tan(rel_elev) / (2.0 * math.tan(CAMERA_VFOV_RAD * 0.5)), -0.5, 0.5))
        size = float(np.clip(FACE_SIZE_M / (d * 2.0 * math.tan(CAMERA_VFOV_RAD * 0.5)), 0.0, 1.0))
        return {"idx": i, "cx": cx, "cy": cy, "size": size, "dist": d}

    def _visible_faces(self) -> list[dict[str, float]]:
        out = []
        for i in range(len(self.patients)):
            f = self._face_of(i)
            if f is not None:
                out.append(f)
        return out

    def _speaking(self, p: dict[str, Any]) -> bool:
        return p["call_t"] > 0.0 or p["speak_t"] > 0.0

    # ------------------------------------------------------------------ people
    def _turn_face_towards_robot(self, p: dict[str, Any]) -> None:
        want = math.atan2(-(self.robot_x - p["x"]), -(self.robot_z - p["z"]))   # yaw whose forward points at the robot
        err = (want - p["face_yaw"] + math.pi) % (2.0 * math.pi) - math.pi
        step = FACE_TURN_RAD_S * self.dt
        p["face_yaw"] += float(np.clip(err, -step, step))

    def _busy_escorting(self) -> bool:
        return self.task_phase in (TaskPhase.GUIDE_ESCORT, TaskPhase.CALL_STAFF) or getattr(self, "returning_to_lobby", False)

    def _calling_episode(self, p: dict[str, Any]) -> bool:
        """Between the first call and giving up (calling or pausing between calls)."""
        return p["calls_done"] > 0 and not p["gave_up"] and (p["call_t"] > 0.0 or p["pause_t"] > 0.0)

    def _step_people(self) -> float:
        """Calls, face turning, speech timers. Returns the reward for ignored / given-up callers."""
        r = 0.0
        partner = self.dialogue["idx"] if self.dialogue else None
        busy = self._busy_escorting()
        for i, p in enumerate(self.patients):
            p["speak_t"] = max(0.0, p["speak_t"] - self.dt)
            d = math.hypot(p["x"] - self.robot_x, p["z"] - self.robot_z)
            may_call = (p["caller"] and not p["served"] and not p["gave_up"] and not p["engaged"] and i != partner and not busy
                        and floor_of(p["x"]) == floor_of(self.robot_x) and d <= CALL_RANGE_M)
            if may_call:
                if p["call_t"] > 0.0:
                    p["call_t"] -= self.dt
                    if p["call_t"] <= 0.0:
                        p["call_t"] = 0.0
                        if p["calls_done"] >= CALL_MAX:
                            p["gave_up"] = True
                            self.v3_stats["callers_gave_up"] += 1
                            r += R_CALLER_GAVE_UP
                        else:
                            p["pause_t"] = CALL_PAUSE_S
                elif p["pause_t"] > 0.0:
                    p["pause_t"] = max(0.0, p["pause_t"] - self.dt)
                elif p["calls_done"] < CALL_MAX:
                    p["call_t"] = CALL_UTTER_S
                    p["calls_done"] += 1
                if self._calling_episode(p) and self._face_of(i) is None:
                    r += R_CALL_IGNORED
            else:
                p["call_t"] = 0.0            # out of range / robot busy: stops calling (keeps its count)
            if i == partner or p["call_t"] > 0.0 or p["speak_t"] > 0.0:
                self._turn_face_towards_robot(p)
        return r

    # ------------------------------------------------------------------ dialogue
    def _start_dialogue(self, i: int) -> None:
        p = self.patients[i]
        p["call_t"] = p["pause_t"] = 0.0
        dur = float(self.rng.uniform(*DIALOGUE_S)) if self.domain_randomization else float(np.mean(DIALOGUE_S))
        p["speak_t"] = dur
        self.dialogue = {"idx": i, "t": 0.0, "dur": dur, "face_s": 0.0, "enrolled": False,
                         "x": self.robot_x, "z": self.robot_z,
                         "abs_bearing": abs(self._rel_angle(p["x"], p["z"])[0])}
        self.v3_stats["dialogues"] += 1
        # make this person the patient being handled (V1 bookkeeping as on acquisition)
        if self.active_patient_idx != i:
            self.active_patient_idx = i
            self.patient_x, self.patient_z = p["x"], p["z"]
            self.patient_seated = p["seated"]
            self.scenario_action, self.scenario_slot, self.scenario_floor = p["destination"]
            self.steps_since_task = 0
        if self.task_phase == TaskPhase.SEARCH_PATIENT:
            self.task_phase = TaskPhase.APPROACH_PATIENT

    def _step_in(self, p: dict[str, Any]) -> None:
        """The person stands up and steps close to the robot, at some bearing (not through walls)."""
        r = float(self.rng.uniform(*WALKUP_STEP_IN_M))
        for _ in range(6):
            b = self.robot_yaw + float(self.rng.uniform(-WALKUP_STEP_IN_BEARING_RAD, WALKUP_STEP_IN_BEARING_RAD))
            nx, nz = self.robot_x - math.sin(b) * r, self.robot_z - math.cos(b) * r     # forward = (−sin, −cos)
            fx, fz = self.robot_x - math.sin(b) * (r + 0.4), self.robot_z - math.cos(b) * (r + 0.4)
            if floor_of(nx) == floor_of(self.robot_x) and not self._check_los_occluded(self.robot_x, self.robot_z, fx, fz):
                p["x"], p["z"] = nx, nz
                p["seated"] = False
                if self.active_patient_idx is not None and self.patients[self.active_patient_idx] is p:
                    self.patient_x, self.patient_z, self.patient_seated = nx, nz, False
                return

    def setup_walkup_drill(self, d_range: tuple[float, float] = (0.8, 1.9)) -> bool:
        """Short drill episode: right after reset, one person needing help (not a caller) stands next to the stopped
        robot at any bearing (±180°), looking anywhere. Walk-up talkers are rare in full shifts (~0.5 per 700 s), so
        drills give the policy many more chances to learn turning to the voice. Returns False if no spot was found."""
        cands = [i for i, p in enumerate(self.patients) if floor_of(p["x"]) == floor_of(self.robot_x)]
        if not cands:
            return False
        p = self.patients[int(self.rng.choice(cands))]
        for _ in range(12):
            b = float(self.rng.uniform(-math.pi, math.pi))
            r = float(self.rng.uniform(*d_range))
            nx, nz = self.robot_x - math.sin(b) * r, self.robot_z - math.cos(b) * r
            fx, fz = self.robot_x - math.sin(b) * (r + 0.4), self.robot_z - math.cos(b) * (r + 0.4)
            if floor_of(nx) == floor_of(self.robot_x) and not self._check_los_occluded(self.robot_x, self.robot_z, fx, fz):
                p.update(x=nx, z=nz, seated=False, needs_help=True, caller=False,
                         face_yaw=float(self.rng.uniform(-math.pi, math.pi)))
                if self.active_patient_idx is not None and self.patients[self.active_patient_idx] is p:
                    self.patient_x, self.patient_z, self.patient_seated = nx, nz, False
                self.last_obs_v2 = self._get_obs()
                return True
        return False

    def _ask_candidates(self) -> list[tuple[int, float, float]]:
        """People the question reaches: body in the camera view (YOLO box, face not needed: someone facing away hears
        the robot and turns round), within ASK_RANGE_M, near the image centre. (idx, |cx|, dist)."""
        out = []
        fl = floor_of(self.robot_x)
        for i, p in enumerate(self.patients):
            if floor_of(p["x"]) != fl:
                continue
            ang, d = self._rel_angle(p["x"], p["z"])
            if d > ASK_RANGE_M or abs(ang) > CAMERA_HFOV_RAD * 0.5:
                continue
            cx = math.tan(ang) / (2.0 * math.tan(CAMERA_HFOV_RAD * 0.5))
            if abs(cx) > ASK_CENTRE_CX or self._check_los_occluded(self.robot_x, self.robot_z, p["x"], p["z"]):
                continue
            out.append((i, abs(cx), d))
        return out

    def _ask(self) -> float:
        self.v3_stats["asks"] += 1
        if self.since_ask_s < ASK_COOLDOWN_S:
            self.v3_stats["asks_repeat"] += 1
            return R_ASK_REPEAT
        self.since_ask_s = 0.0
        cands = self._ask_candidates()
        if not cands:
            self.v3_stats["asks_empty"] += 1
            return R_ASK_EMPTY
        i = min(cands, key=lambda c: c[1])[0]
        p = self.patients[i]
        if p["served"] or p["declined"]:
            self.v3_stats["asks_repeat"] += 1
            return R_ASK_REPEAT
        if not p["needs_help"]:
            p["declined"] = True
            p["served"] = True            # never acquired / approached again
            p["speak_t"] = DECLINE_UTTER_S
            self.v3_stats["asks_declined"] += 1
            if self.active_patient_idx == i and self.task_phase == TaskPhase.APPROACH_PATIENT:
                self.task_phase = TaskPhase.SEARCH_PATIENT
                self.goal_key = "patient"
                self.active_patient_idx = None
                self.patient_x = self.patient_z = -999.0
            return R_ASK_DECLINED
        self._start_dialogue(i)           # they turn to the robot and start talking
        return 0.0

    def _step_dialogue(self, faces: list[dict[str, float]]) -> float:
        d = self.dialogue
        if d is None:
            return 0.0
        r = 0.0
        d["t"] += self.dt
        self.v3_stats["dialogue_steps"] += 1
        p = self.patients[d["idx"]]
        if any(f["idx"] == d["idx"] for f in faces):
            d["face_s"] += self.dt
            self.v3_stats["face_steps"] += 1
            r += R_FACE_DIALOGUE
            if not d["enrolled"] and d["face_s"] >= ENROLL_NEED_S:
                d["enrolled"] = True
                self.v3_stats["enrolled"] += 1
                if d.get("walkup"):
                    self.v3_stats["walkup_enrolled"] += 1
                r += R_ENROLLED
        else:
            r -= R_FACE_DIALOGUE
        abs_b = abs(self._rel_angle(p["x"], p["z"])[0])
        r += R_TURN_TO_SPEAKER * (d["abs_bearing"] - abs_b)
        d["abs_bearing"] = abs_b
        if self.robot_v > 0.05:
            r += R_MOVE_DIALOGUE
        if math.hypot(self.robot_x - d["x"], self.robot_z - d["z"]) > DIALOGUE_LEAVE_M:
            self.v3_stats["aborted"] += 1
            p["speak_t"] = 0.0
            self.dialogue = None
            return r + R_DIALOGUE_ABORT
        finished = d["t"] >= d["dur"] and (d["enrolled"] or d["t"] >= d["dur"] + DIALOGUE_EXTRA_S)
        if finished:
            if not d["enrolled"]:
                self.v3_stats["not_enrolled"] += 1
                r += R_NOT_ENROLLED
            p["speak_t"] = 0.0
            p["engaged"] = True
            self.dialogue = None
            self.dialogue_done = True     # consumed by _engagement_ready on the next V1 step
        return r

    def _engagement_ready(self, pat_seen: float, pat_h: float, d_phys: float) -> bool:
        if self.dialogue_done:
            self.dialogue_done = False
            return True
        return False

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        a = np.zeros(ACT_DIM_V3, dtype=np.float32)
        a_in = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        a[: min(ACT_DIM_V3, a_in.shape[0])] = a_in[:ACT_DIM_V3]
        if a_in.shape[0] < 4:
            a[3] = -1.0                                   # 2-D action (V2 policy / tests): never ask
        self.tilt_target = float(a[2]) * (TILT_UP_RAD if a[2] > 0.0 else TILT_DOWN_RAD)
        dt_tilt = float(np.clip(self.tilt_target - self.tilt, -TILT_RATE_RAD_S * self.dt, TILT_RATE_RAD_S * self.dt))
        self.tilt = float(np.clip(self.tilt + dt_tilt, -TILT_DOWN_RAD, TILT_UP_RAD))
        self.since_ask_s += self.dt

        r_v3 = 0.0
        if float(a[3]) > ASK_THRESHOLD and self.dialogue is None and not self._busy_escorting():
            r_v3 += self._ask()
        r_v3 += self._step_people()

        idx_before = self.active_patient_idx
        obs, r, done, info = super().step(a[:2])
        if info.get("guide_completed") and idx_before is not None:
            self.patients[idx_before]["speak_t"] = THANKS_UTTER_S      # turns to the robot, says thanks

        faces = self._visible_faces()
        r_v3 += self._step_dialogue(faces)
        # a caller the robot now faces up close starts talking without being asked
        if self.dialogue is None and not self._busy_escorting():
            for f in faces:
                p = self.patients[f["idx"]]
                if self._calling_episode(p) and f["dist"] <= ASK_RANGE_M and not p["served"]:
                    self._start_dialogue(f["idx"])
                    break
        if self.dialogue is None and not self._busy_escorting() and abs(self.robot_v) < WALKUP_MAX_V:
            fl = floor_of(self.robot_x)
            for i, p in enumerate(self.patients):
                if (not p["needs_help"] or p["served"] or p["engaged"] or p["declined"] or p["gave_up"]
                        or floor_of(p["x"]) != fl):
                    continue
                if math.hypot(p["x"] - self.robot_x, p["z"] - self.robot_z) > WALKUP_RANGE_M:
                    continue
                if self.rng.random() < WALKUP_RATE_HZ * self.dt and not self._check_los_occluded(
                        self.robot_x, self.robot_z, p["x"], p["z"]):
                    self.v3_stats["walkup"] += 1
                    if self.domain_randomization and self.rng.random() < WALKUP_STEP_IN_PROB:
                        self._step_in(p)
                    self._start_dialogue(i)
                    self.dialogue["walkup"] = True
                    break
        if self.robot_v > 0.2:
            r_v3 += R_TILT_MOVING * abs(self._tilt_norm())

        obs = self._get_obs(faces)
        self.last_obs_v2 = obs
        r = float(np.clip(r + r_v3, -20.0, 20.0))
        self.cumulative_v2_reward += r_v3
        info["cumulative_team_reward"] = self.cumulative_v2_reward
        info["v3"] = dict(self.v3_stats)
        return obs, r, done, info

    # ------------------------------------------------------------------ observation
    def _tilt_norm(self) -> float:
        return self.tilt / TILT_UP_RAD if self.tilt >= 0.0 else self.tilt / TILT_DOWN_RAD

    def _get_obs(self, faces: list[dict[str, float]] | None = None) -> np.ndarray:
        o21 = super()._get_obs()
        if not getattr(self, "_v3_ready", False) or o21.shape[0] != OBS_DIM_V2:
            return o21
        if faces is None:
            faces = self._visible_faces()
        o = np.zeros(OBS_DIM_V3, dtype=np.float32)
        o[:OBS_DIM_V2] = o21
        # voice: loudest (closest) speaking person on this floor
        best = None
        for p in self.patients:
            if self._speaking(p) and floor_of(p["x"]) == floor_of(self.robot_x):
                ang, d = self._rel_angle(p["x"], p["z"])
                if best is None or d < best[1]:
                    best = (ang, d)
        self.last_voice = best
        now = self.step_count * self.dt
        if best is not None:
            o[21] = 1.0
            o[22], o[23] = math.sin(best[0]), math.cos(best[0])
            self.voice_mem = (*self._rel_to_world_dir(best[0]), now)
        # remembered voice direction: last heard bearing kept in the world frame (robot: DOA + odometry), so it stays
        # valid while the robot turns after the person stopped talking
        o[32], o[33], o[34] = 0.0, 0.0, 1.0
        if self.voice_mem is not None and now - self.voice_mem[2] <= VOICE_MEMORY_S:
            rel = self._world_dir_to_rel(self.voice_mem[0], self.voice_mem[1])
            o[32], o[33], o[34] = math.sin(rel), math.cos(rel), (now - self.voice_mem[2]) / VOICE_MEMORY_S
            if best is None and DOA_HOLDS_MEMORY:
                # DOA slots hold the remembered direction once the person stops talking (o[21] still says "silent"):
                # the turn-to-voice response learned on o[22:24] keeps working instead of seeing (0, 0) = "nobody"
                o[22], o[23] = o[32], o[33]
        # face: dialogue partner, else the most central one
        face = None
        if self.dialogue is not None:
            face = next((f for f in faces if f["idx"] == self.dialogue["idx"]), None)
        elif faces:
            face = min(faces, key=lambda f: abs(f["cx"]))
        self.last_face = face
        if face is not None:
            o[24], o[25], o[26], o[27] = 1.0, face["cx"], face["cy"], face["size"]
        o[28] = self._tilt_norm()
        o[29] = min(self.since_ask_s / 5.0, 1.0)
        o[30] = 1.0 if self.dialogue is not None else 0.0
        # Face already met (helped or declined): Godot / real robot -> ReID / face gallery of past people
        if face is not None:
            q = self.patients[face["idx"]]
            o[31] = 1.0 if (q["served"] or q["declined"]) else 0.0
        return np.nan_to_num(o, nan=0.0, posinf=1.0, neginf=-1.0)

    # ------------------------------------------------------------------ teacher
    def teacher_action(self) -> np.ndarray:
        """Rule-based baseline (= the current Godot rules): turn to a voice, look up / down to centre the face, ask a
        locked-on patient when close, stand still and keep the face in view while talking."""
        o = self.last_obs_v2
        base = super().teacher_action()
        a_v, a_w = float(base[0]), float(base[1])
        a_tilt = 0.0
        a_ask = -1.0
        face_seen, fcx, fcy, fsize = o[24] > 0.5, float(o[25]), float(o[26]), float(o[27])
        phase = int(np.argmax(o[9:13]))
        if face_seen:
            # move the face to the vertical centre: cy > 0 (face low in the image) -> look down
            want = self.tilt - math.atan(2.0 * fcy * math.tan(CAMERA_VFOV_RAD * 0.5))
            a_tilt = float(np.clip(want / (TILT_UP_RAD if want >= 0.0 else TILT_DOWN_RAD), -1.0, 1.0))
        if o[30] > 0.5:                                         # in dialogue: stand still, face the person
            a_v = -1.0
            if face_seen:
                a_w = float(np.clip(-2.0 * fcx, -1.0, 1.0))
            elif o[21] > 0.5:
                a_w = float(np.clip(-1.5 * math.atan2(o[22], o[23]), -1.0, 1.0))
            elif o[34] < 1.0:                                   # person stopped talking: remembered direction
                a_w = float(np.clip(-1.5 * math.atan2(o[32], o[33]), -1.0, 1.0))
        elif o[21] > 0.5 and phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT):
            ang = math.atan2(float(o[22]), float(o[23]))         # turn to the voice, walk to it once facing it
            s = o[0:5]
            repulsion = -0.9 * (s[0] + 0.6 * s[1]) + 0.9 * (s[4] + 0.6 * s[3])   # same obstacle push as V2 teacher
            front = float(max(s[2], 0.9 * max(s[1], s[3])))
            a_w = float(np.clip(-1.5 * ang + 0.5 * repulsion / OMEGA_MAX, -1.0, 1.0))
            near = face_seen and fsize >= FACE_SIZE_M / (ASK_RANGE_M * 0.9 * 2.0 * math.tan(CAMERA_VFOV_RAD * 0.5))
            a_v = -1.0 if (abs(ang) > 0.5 or near or front > 0.8) else -0.4
        if o[30] < 0.5 and (self.robot_v > 0.2 or not face_seen):
            a_tilt = 0.0                                        # keep the head level while driving
        # ask the locked-on patient once the robot stops in front of them (V2 teacher stops at TARGET_BOX_H ~ 1.85 m)
        pat_seen, pat_cx, pat_h = o[5] > 0.5, float(o[6]), float(o[7])
        if (o[30] < 0.5 and o[31] < 0.5 and phase == TaskPhase.APPROACH_PATIENT and pat_seen and abs(pat_cx) <= 0.15
                and pat_h >= 0.9 * TARGET_BOX_H and o[29] >= ASK_COOLDOWN_S / 5.0):
            a_ask, a_v = 1.0, -1.0
        return np.array([a_v, a_w, a_tilt, a_ask], dtype=np.float32)
