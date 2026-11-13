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
  [31]     the face in view belongs to someone already met (helped / declined)
  [32..34] remembered voice direction (sin, cos) and its age / 10 s
  [35..40] obstacle memory: proximity in 6 sectors outside the camera view (left side .. behind .. right side),
           from the depth hits of the last 8 s + odometry (local_memory.py)
  [41]     recovery cue: 0, or +-(time left of the 2.2 s recovery window that starts when the robot has commanded
           forward speed for 0.4 s without moving); the sign is the side to turn to (+ = left)
  [42..44] largest person box in view, whoever it is [seen, cx (-0.5..0.5), h]: a wall and a person both fill the depth
           sectors, but only a person can be asked to make way
The ask action (a[3]) doubles as "Xin lỗi, cho tôi đi nhờ ạ": used on someone already met, or while the robot is busy
(escorting / returning), it asks the person standing right ahead to make way; most people then step aside.
While talking, the nav-goal direction [14..15] is the (remembered) voice direction: turning to the speaker uses the same
"turn to the goal" skill as driving. During the escort the tether [18..20] also counts the rear camera (the patient
usually walks behind the robot).
Action (4-D, each in [-1, 1]): [v, omega] as V2, [2] head pitch target, [3] ask when > 0.5.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.models.ppo_agent import OMEGA_MAX, V_MAX
from src.training.camera_vision_model import (
    CAMERA_HEIGHT_M, CAMERA_HFOV_RAD, CAMERA_MOUNT_FORWARD_M, CAMERA_VFOV_RAD, compute_pinhole_detection,
)
from src.training.hospital_env import (
    _DOORS, DESTINATION_POSITIONS, ESCORT_WALK_SPEED, ROBOT_RADIUS_M, TaskPhase, floor_of, is_blocked,
)
from src.training.hospital_env_pure_vision_v2 import (
    DEPTH_HFOV_RAD, DEPTH_MAX_RANGE_M, HospitalPureVisionV2Env, OBS_DIM_V2, PERSON_RADIUS_M, tether_features,
)
from src.training.hospital_env_vision import TARGET_BOX_H
from src.training.local_memory import (
    MEM_SECTORS, PERSON_EXTRA_DIMS, RECOVERY_TURN_S, STILL_NORM_S, TRACK_DIMS, ObstacleMemory, PatientTrack, RecoveryCue,
    flank_proximity, person_slots,
)

OBS_DIM_V3 = 59                   # round 11: + people slots with "already met" flags, escorted-patient track
OBS_MEM0 = 35                     # [35..40] obstacle memory sectors
OBS_STALL = 41
OBS_PERSON0 = 42                  # [42..44] largest person box [seen, cx, h] (= person slot 0)
OBS_PERSON_X0 = 45                # [45..53] known0, slots 1-2 [seen, cx, h, known] (local_memory.person_slots)
OBS_TRACK0 = 54                   # [54..58] escorted patient: [valid, sin, cos (bearing, + = right), dist / 5, still / 20 s]
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

# --- escorted / delivered people (as seen in Godot: the patient mostly walks BEHIND the robot, stops now and then,
#     and after the delivery stands where they are, often between the robot and the door) ---
FOLLOW_BEHIND_PROB = 0.85                # Godot (round 10): every escorted patient walks behind; some front-right kept
FOLLOW_BEHIND_GAP_M = (0.8, 1.3)
LAG_PROB = 0.7                     # the escorted patient stops once during the escort ...
LAG_AT_S = (5.0, 40.0)
LAG_S = (5.0, 18.0)                # ... for this long (Godot: run_simulation.py --lag-prob)
DELIVERED_LEAVE_PROB = 0.5         # the delivered person walks on into the room after a while; the rest stay put
DELIVERED_LEAVE_DELAY_S = (3.0, 20.0)
DELIVERED_WALK_SPEED = 0.48
NPC_PARDON_PROB = 0.5              # share of staff NPCs that step aside for the robot (people are not that predictable)
PUSH_RANGE_M = PERSON_RADIUS_M + ROBOT_RADIUS_M + 0.06
PUSH_CMD_V = 0.15
# "excuse me": the ask action aimed at someone standing in the way
EXCUSE_RANGE_M = 1.5
EXCUSE_HALF_ANGLE_RAD = math.radians(40.0)
EXCUSE_YIELD_PROB = 0.85           # some people do not react
EXCUSE_DELAY_S = (0.4, 1.5)
EXCUSE_SIDE_M = 0.95               # they step this far off the robot's line
EXCUSE_WALK_SPEED = 0.6
# exit drill: the robot has just delivered someone into a room and must get back out past them
EXIT_ROOMS = ("101", "110", "112", "nha_thuoc", "201", "205", "210", "220", "301", "302", "303", "315")
EXIT_DONE_Z_M = 1.0                # |z| below this = back in the corridor

# --- reward V3 (added to V2) ---
R_ENROLLED = 1.0
R_FACE_DIALOGUE = 0.03
R_NOT_ENROLLED = -1.0
R_MOVE_DIALOGUE = -0.05
R_DIALOGUE_ABORT = -1.0
R_ASK_EMPTY = -1.5                 # round 11b: was -0.5 -> the policy learned to "ask just in case" (5 wasted / shift)
R_ASK_REPEAT = -1.0
R_ASK_DECLINED = -0.1
R_CALL_IGNORED = -0.01
R_CALLER_GAVE_UP = -1.0
R_TILT_MOVING = -0.003
# Potential-based shaping during the dialogue: reward per radian the speaker's bearing gets closer to straight ahead
# (Φ = −K·|bearing|) -> gradient on the sign of the voice direction; does not change the optimal policy.
R_TURN_TO_SPEAKER = 0.3
R_FACE_CENTRE = 0.02               # per step while talking, scaled by how centred the speaker's face is (eye contact)
R_PERSON_PUSH = -0.2               # per step driving into a person standing in the way
R_EXIT_DRILL = 3.0                 # exit drill: reached the corridor
# Round 10: the robot itself waits for a patient who stops (Godot no longer stops it by rule when they fall behind).
R_LEAVE_BEHIND = -0.15             # per step driving on away from a patient more than LEAVE_BEHIND_M behind
LEAVE_BEHIND_M = 3.0
LAG_DRILL_S = 60.0                 # lag drill: an escort in which the patient stops after LAG_DRILL_AT_S
LAG_DRILL_AT_S = (2.0, 6.0)
LAG_DRILL_LAG_S = (8.0, 18.0)
LAG_FAR_M = 2.5                    # lag drill bookkeeping: robot driving on while the stopped patient is this far


def patient_left_behind(o: np.ndarray) -> bool:
    """Evidence that the escorted patient has fallen behind: seen more than 2.5 m away; lost for 4 s after last being
    seen beyond 1.8 m; or lost for 2 s while the robot is already 2.5 m from where it last saw them (round 11: someone
    who stops close to the robot — a follower would have shown up in the rear camera by then). NOT "just not seen": a
    patient walking close behind-beside is in the blind spot between the front and the rear camera (|bearing|
    90-137 deg) — waiting there made the robot and the patient wait for each other (round 10, first attempt)."""
    seen_far = o[19] > 0.5 and o[18] > 0.5
    lost_far = o[19] < 0.5 and o[20] >= 0.8 and o[18] > 0.36
    track = len(o) > OBS_TRACK0 + 4 and o[OBS_TRACK0] > 0.5
    left_spot = track and o[19] < 0.5 and o[20] >= 0.4 and float(o[OBS_TRACK0 + 3]) * 5.0 > 2.5
    return bool(seen_far or lost_far or left_spot)


# Round 10: the tether the robot sees is a ReID estimate (Godot: box matched to the remembered person), not the truth:
# it sometimes loses them for a moment, sometimes latches onto someone else, and the distance from the box is noisy.
# Round 11: half the stops are "waits for the robot": the patient stays until the robot comes back, faces them and asks
# "Anh/chị đi tiếp được không ạ?" (the ask action while escorting). The robot must go back, not only wait.
LAG_WAIT_PROB = 0.5
LAG_WAIT_MAX_S = 60.0
REENGAGE_M = 2.5
REENGAGE_HEAR_M = 2.0              # out of sight (corner, doorway) but this close: they hear the question
R_REENGAGE = 3.0
R_PATIENT_WAITING = -0.03           # per step a patient stands waiting for the robot to come back (round 11c: the
                                   # policy came within 1.2-1.7 m, saw them and did not ask in 11 of 24 drills)
RETURN_AFTER_S = 6.0               # teacher: the patient has not moved for this long while left behind -> go back
TRACK_POS_NOISE_M = 0.15
REID_MISS_HZ = 0.04
REID_MISS_S = (0.5, 3.0)
REID_SWAP_HZ = 0.015
REID_SWAP_S = (0.5, 3.0)
REID_D_NOISE = 0.10                # relative


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
                "follows_behind": bool(self.domain_randomization and self.rng.random() < FOLLOW_BEHIND_PROB),
                "follow_gap": float(self.rng.uniform(*FOLLOW_BEHIND_GAP_M)),
                "lag_at": -1.0,           # escort time at which they stop for a while (-1 = never)
                "lag_left": 0.0,
                "leave_t": -1.0,          # delivered: seconds until they walk on (-1 = stay)
                "leave_to": None,
                "yield_t": -1.0,          # asked to make way: seconds until they step aside (-1 = not asked)
                "yield_to": None,
            })
        if self.domain_randomization:
            for n in self.npcs:
                n["pardon_enabled"] = bool(n.get("pardon_enabled", True) and self.rng.random() < NPC_PARDON_PROB)
        self.mem = ObstacleMemory(DEPTH_HFOV_RAD, DEPTH_MAX_RANGE_M, ROBOT_RADIUS_M)
        self.stall = RecoveryCue()
        self.escort_t = 0.0
        self.drill: str | None = None
        self._cue_pending: float | None = None
        self._teacher_turn = 0.0
        self.tilt = 0.0
        self.tilt_target = 0.0
        self.since_ask_s = 99.0
        self.dialogue: dict[str, Any] | None = None
        self.dialogue_done = False
        self.v3_stats = {"asks": 0, "walkup": 0, "walkup_enrolled": 0, "asks_empty": 0, "asks_repeat": 0, "asks_declined": 0, "dialogues": 0,
                         "enrolled": 0, "not_enrolled": 0, "aborted": 0, "callers_gave_up": 0, "face_steps": 0,
                         "dialogue_steps": 0, "push_steps": 0, "exit_ok": 0, "centre_sum": 0.0, "excuse": 0,
                         "leave_behind_steps": 0, "lag_steps": 0, "lag_far_m": 0.0, "lag_max_tether": 0.0,
                         "lag_after_m": 0.0, "lag_done": 0, "reid_miss_steps": 0, "reid_swap_steps": 0,
                         "lag_wait": 0, "reengaged": 0, "asks_continue_wasted": 0}
        self.last_face: dict[str, float] | None = None
        self.last_voice: tuple[float, float] | None = None
        self.voice_mem: tuple[float, float, float] | None = None   # world direction (x, z) + time heard (s)
        self._reid_miss_left = 0.0
        self._reid_swap_left = 0.0
        self._reid_swap_idx: int | None = None
        self._reid_last_d = 1.0
        self._reid_unseen_s = 0.0
        self._reid_step = -1
        self._lag_seen = False
        self._lag_drill_s = 0.0
        self._lag_drill_wait: bool | None = None
        self.track = PatientTrack()
        self._track_phase = None
        self._returning = False
        self._v3_ready = True
        self._remember_depth()
        self.last_obs_v2 = self._get_obs()
        return self.last_obs_v2

    def _remember_depth(self) -> None:
        self.mem.update(self.robot_x, self.robot_z, self.robot_yaw, self._noisy_depth(self.depth_rays()),
                        self.step_count * self.dt)

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
            if p["yield_t"] >= 0.0:
                self._step_yield(p)
            elif p["leave_t"] >= 0.0 and p["speak_t"] <= 0.0:
                self._step_delivered(p, d)
        return r

    def _step_delivered(self, p: dict[str, Any], d_robot: float) -> None:
        """A delivered person who walks on into the room after a while (never towards / through the robot)."""
        if p["leave_t"] > 0.0:
            p["leave_t"] = max(0.0, p["leave_t"] - self.dt)
            return
        tx, tz = p["leave_to"]
        dx, dz = tx - p["x"], tz - p["z"]
        dist = math.hypot(dx, dz)
        if dist < 0.1:
            p["leave_t"] = -1.0
            return
        step = min(DELIVERED_WALK_SPEED * self.dt, dist)
        nx, nz = p["x"] + dx / dist * step, p["z"] + dz / dist * step
        d_next = math.hypot(nx - self.robot_x, nz - self.robot_z)
        if is_blocked(nx, nz, 0.28) or (d_next < 0.9 and d_next < d_robot):
            return                                   # waits (furniture / would walk into the robot)
        p["x"], p["z"] = nx, nz

    def _step_yield(self, p: dict[str, Any]) -> None:
        if p["yield_t"] > 0.0:
            p["yield_t"] = max(0.0, p["yield_t"] - self.dt)
            return
        tx, tz = p["yield_to"]
        dx, dz = tx - p["x"], tz - p["z"]
        dist = math.hypot(dx, dz)
        if dist < 0.08:
            p["yield_t"] = -1.0
            return
        step = min(EXCUSE_WALK_SPEED * self.dt, dist)
        p["x"], p["z"] = p["x"] + dx / dist * step, p["z"] + dz / dist * step

    def _blocking_person(self) -> int | None:
        """Someone standing right ahead of the robot who is not the person it approaches / escorts."""
        best, best_d = None, EXCUSE_RANGE_M
        fl = floor_of(self.robot_x)
        for i, p in enumerate(self.patients):
            if p["seated"] or floor_of(p["x"]) != fl:
                continue
            if i == self.active_patient_idx and self.task_phase in (TaskPhase.APPROACH_PATIENT, TaskPhase.GUIDE_ESCORT):
                continue
            ang, d = self._rel_angle(p["x"], p["z"])
            if d <= best_d and abs(ang) <= EXCUSE_HALF_ANGLE_RAD and not self._check_los_occluded(self.robot_x, self.robot_z, p["x"], p["z"]):
                best, best_d = i, d
        return best

    def _yield_spot(self, p: dict[str, Any]) -> tuple[float, float] | None:
        """Where a person asked to make way steps to: off the robot's line (their own side first); out of a doorway
        they go through to the side the robot is not on."""
        fx, fz = -math.sin(self.robot_yaw), -math.cos(self.robot_yaw)
        rx, rz = math.cos(self.robot_yaw), -math.sin(self.robot_yaw)
        ox, oz = p["x"] - self.robot_x, p["z"] - self.robot_z
        lat = ox * rx + oz * rz
        side = 1.0 if lat >= 0.0 else -1.0
        cands = [(p["x"] + rx * (sg * EXCUSE_SIDE_M - lat), p["z"] + rz * (sg * EXCUSE_SIDE_M - lat)) for sg in (side, -side)]
        fl = floor_of(p["x"])
        for (f, dside), (centres, _) in _DOORS.items():          # in / near a doorway: step through and aside
            if f != fl:
                continue
            z_line = -2.0 if dside == "north" else 2.0
            for c in centres:
                if math.hypot(p["x"] - c, p["z"] - z_line) < 1.2:
                    away = 1.0 if (z_line - self.robot_z) >= 0.0 else -1.0
                    cands = [(c + sg * 1.1, z_line + away * 1.1) for sg in (side, -side)] + cands
        for cx, cz in cands:
            if (floor_of(cx) == fl and not is_blocked(cx, cz, 0.3) and self._segment_clear(p["x"], p["z"], cx, cz)
                    and not any(math.hypot(cx - c, cz - (-2.0 if ds == "north" else 2.0)) < 0.9
                                for (f, ds), (cs, _) in _DOORS.items() if f == fl for c in cs)):
                return cx, cz
        return None

    def _excuse(self) -> float | None:
        """"Xin lỗi, cho tôi đi nhờ ạ" to the person right ahead. None if nobody is there."""
        i = self._blocking_person()
        if i is None:
            return None
        p = self.patients[i]
        self.v3_stats["excuse"] += 1
        p["speak_t"] = max(p["speak_t"], 0.6)                    # they look round at the robot
        if p["yield_t"] < 0.0 and (not self.domain_randomization or self.rng.random() < EXCUSE_YIELD_PROB):
            spot = self._yield_spot(p)
            if spot is not None:
                p["yield_to"] = spot
                p["yield_t"] = float(self.rng.uniform(*EXCUSE_DELAY_S)) if self.domain_randomization else 0.5
                p["leave_t"] = -1.0
        return 0.0

    def _on_delivered(self, p: dict[str, Any], goal_key: str) -> None:
        p["speak_t"] = THANKS_UTTER_S                # turns to the robot, says thanks
        if not self.domain_randomization or self.rng.random() >= DELIVERED_LEAVE_PROB:
            return
        gx, gz = DESTINATION_POSITIONS.get(goal_key, (p["x"], p["z"]))
        for _ in range(12):
            tx, tz = gx + float(self.rng.uniform(-2.0, 2.0)), gz + float(self.rng.uniform(-1.5, 1.5))
            if floor_of(tx) == floor_of(p["x"]) and not is_blocked(tx, tz, 0.4) and self._segment_clear(p["x"], p["z"], tx, tz):
                p["leave_t"] = float(self.rng.uniform(*DELIVERED_LEAVE_DELAY_S))
                p["leave_to"] = (tx, tz)
                return

    # ------------------------------------------------------------------ escort follow
    def _step_escorted_patient(self) -> None:
        """Godot's patients mostly end up walking behind the robot (and stop now and then): the robot must cope with a
        follower it only sees in the rear camera, who is still behind it — between it and the door — on arrival."""
        if self.active_patient_idx is None or self.active_patient_idx >= len(self.patients):
            return
        p = self.patients[self.active_patient_idx]
        self.escort_t += self.dt
        if p["lag_at"] >= 0.0 and self.escort_t >= p["lag_at"]:
            p["lag_at"] = -1.0
            wait = self._lag_drill_wait if self._lag_drill_wait is not None else bool(self.rng.random() < LAG_WAIT_PROB)
            p["lag_wait"] = wait
            if wait:
                p["lag_left"] = LAG_WAIT_MAX_S                   # until the robot comes back and asks (or a minute)
                self.v3_stats["lag_wait"] += 1
            else:
                p["lag_left"] = float(getattr(self, "_lag_drill_s", 0.0) or self.rng.uniform(*LAG_S))
            self._lag_drill_s = 0.0
            self._lag_drill_wait = None
        lagging = p["lag_left"] > 0.0
        if not lagging and not p["follows_behind"]:
            super()._step_escorted_patient()
            return
        trail = self.v2_trail
        if not trail or math.hypot(self.robot_x - trail[-1][0], self.robot_z - trail[-1][1]) >= 0.25:
            trail.append((self.robot_x, self.robot_z))
            del trail[:-200]
        if trail and abs(trail[-1][0] - p["x"]) > 25.0:      # elevator: they step out next to the robot
            trail.clear()
            p["lag_left"] = 0.0
            return
        if lagging:
            p["lag_left"] = max(0.0, p["lag_left"] - self.dt)
            return
        if is_blocked(p["x"], p["z"], 0.28):                # stood up inside furniture -> nearest free cell
            super()._step_escorted_patient()
            return
        target = None
        for tx, tz in reversed(trail):                       # newest crumb far enough behind the robot, in a clear line
            if math.hypot(tx - self.robot_x, tz - self.robot_z) >= p["follow_gap"] and self._segment_clear(p["x"], p["z"], tx, tz):
                target = (tx, tz)
                break
        if target is None:
            if math.hypot(p["x"] - self.robot_x, p["z"] - self.robot_z) > p["follow_gap"] + 0.5:
                super()._step_escorted_patient()             # lost the trail (e.g. after a pause): catch up as V2
            return
        dx, dz = target[0] - p["x"], target[1] - p["z"]
        dist = math.hypot(dx, dz)
        if dist <= 0.1:
            return
        step = min(self.patient_escort_speed * self.dt, dist)
        nx, nz = p["x"] + dx / dist * step, p["z"] + dz / dist * step
        if math.hypot(nx - self.robot_x, nz - self.robot_z) < PERSON_RADIUS_M + ROBOT_RADIUS_M:
            pass                                             # right behind the robot: waits
        elif not is_blocked(nx, nz, 0.28):
            p["x"], p["z"] = nx, nz
            self.patient_stuck_s = 0.0
        else:
            # wedged against a corner they cut: a real person steps back onto the robot's path (as in V2)
            self.patient_stuck_s = getattr(self, "patient_stuck_s", 0.0) + self.dt
            if self.patient_stuck_s >= 2.0:
                p["x"], p["z"] = min(trail, key=lambda c: math.hypot(c[0] - p["x"], c[1] - p["z"]))
                self.patient_stuck_s = 0.0
        self.patient_x, self.patient_z = p["x"], p["z"]

    def _escorted(self) -> dict[str, Any] | None:
        if self.task_phase != TaskPhase.GUIDE_ESCORT or self.active_patient_idx is None \
                or self.active_patient_idx >= len(self.patients):
            return None
        return self.patients[self.active_patient_idx]

    def _step_lag_bookkeeping(self, x0: float, z0: float) -> float:
        """Round 10: the robot must wait for a patient who stops. Penalty for driving on away from someone left more
        than LEAVE_BEHIND_M behind (true distance), plus the lag-drill statistics."""
        p = self._escorted()
        if p is None or abs(self.robot_x - x0) > 5.0:            # not escorting / elevator ride
            return 0.0
        r = 0.0
        d = math.hypot(p["x"] - self.robot_x, p["z"] - self.robot_z)
        step = math.hypot(self.robot_x - x0, self.robot_z - z0)
        if d > LEAVE_BEHIND_M and step > 0.02:
            away = ((self.robot_x - x0) * (self.robot_x - p["x"]) + (self.robot_z - z0) * (self.robot_z - p["z"])) / max(d, 1e-6)
            if away > 0.01:
                r += R_LEAVE_BEHIND
                self.v3_stats["leave_behind_steps"] += 1
        if p["lag_left"] > 0.0 and p.get("lag_wait"):
            r += R_PATIENT_WAITING
        if p["lag_left"] > 0.0:
            self.v3_stats["lag_steps"] += 1
            self.v3_stats["lag_max_tether"] = max(self.v3_stats["lag_max_tether"], d)
            if d > LAG_FAR_M:
                self.v3_stats["lag_far_m"] += step
            self._lag_seen = True
        elif getattr(self, "_lag_seen", False):
            self.v3_stats["lag_done"] = 1
            self.v3_stats["lag_after_m"] += step
        return r

    def _reid_tether(self, t: np.ndarray) -> np.ndarray:
        """The tether as a ReID tracker reports it: brief misses, brief swaps onto another visible person, noisy
        distance. Timers advance once per env step (the observation may be built more than once per step)."""
        new_step = self.step_count != self._reid_step
        self._reid_step = self.step_count
        if new_step:
            self._reid_miss_left = max(0.0, self._reid_miss_left - self.dt)
            self._reid_swap_left = max(0.0, self._reid_swap_left - self.dt)
            if self._reid_miss_left <= 0.0 and self._reid_swap_left <= 0.0:
                if self.rng.random() < REID_MISS_HZ * self.dt:
                    self._reid_miss_left = float(self.rng.uniform(*REID_MISS_S))
                elif self.rng.random() < REID_SWAP_HZ * self.dt:
                    self._reid_swap_idx = self._other_visible_person()
                    if self._reid_swap_idx is not None:
                        self._reid_swap_left = float(self.rng.uniform(*REID_SWAP_S))
        out = np.array(t, dtype=np.float32)
        if self._reid_swap_left > 0.0 and self._reid_swap_idx is not None:
            q = self.patients[self._reid_swap_idx]
            d = self._box_distance(q)
            if d is not None:
                self.v3_stats["reid_swap_steps"] += int(new_step)
                self._reid_last_d, self._reid_unseen_s = d, 0.0
                return tether_features(d, True, 0.0)
            self._reid_swap_left = 0.0
        if self._reid_miss_left > 0.0 or out[1] < 0.5:
            if new_step:
                self._reid_unseen_s += self.dt
            self.v3_stats["reid_miss_steps"] += int(new_step and out[1] >= 0.5)
            return tether_features(self._reid_last_d, False, self._reid_unseen_s)
        d = float(out[0]) * 5.0
        if new_step:
            d *= float(1.0 + self.rng.normal(0.0, REID_D_NOISE))
            self._reid_last_d = float(np.clip(d, 0.5, 5.0))
        self._reid_unseen_s = 0.0
        return tether_features(self._reid_last_d, True, 0.0)

    def _box_distance(self, q: dict[str, Any]) -> float | None:
        """Distance estimate of person q from their box in the front or the rear camera (None = in neither)."""
        if floor_of(q["x"]) != floor_of(self.robot_x):
            return None
        for yaw in (self.robot_yaw, self.robot_yaw + math.pi):
            det = compute_pinhole_detection(
                robot_x=self.robot_x, robot_z=self.robot_z, robot_yaw=yaw, target_x=q["x"], target_z=q["z"],
                target_seated=q["seated"], is_blocked_fn=self._check_los_occluded, domain_randomization=False, rng=self.rng)
            if det["seen"] >= 0.5:
                return float(np.clip(1.20 / max(float(det["h"]), 0.05), 0.5, 5.0))
        return None

    def _other_visible_person(self) -> int | None:
        cands = [i for i, q in enumerate(self.patients) if i != self.active_patient_idx and self._box_distance(q) is not None]
        return int(self.rng.choice(cands)) if cands else None

    def setup_lag_drill(self, max_prelude_steps: int = 1500) -> bool:
        """Drill: an escort in which the patient stops after a few seconds for 8-18 s. The prelude (find someone,
        talk, start the escort) is driven by the teacher; the drill then lasts LAG_DRILL_S. False if no escort began."""
        for _ in range(max_prelude_steps):
            if self.task_phase == TaskPhase.GUIDE_ESCORT and self._escorted() is not None:
                break
            _, _, done, _ = self.step(self.teacher_action())
            if done:
                return False
        p = self._escorted()
        if p is None:
            return False
        p["follows_behind"] = True
        p["lag_at"] = self.escort_t + float(self.rng.uniform(*LAG_DRILL_AT_S))
        p["lag_left"] = 0.0
        self._lag_drill_s = float(self.rng.uniform(*LAG_DRILL_LAG_S))
        self._lag_drill_wait = bool(self.rng.random() < LAG_WAIT_PROB)
        self.drill = "lag"
        self.max_steps = self.step_count + int(round(LAG_DRILL_S / self.dt))
        self.shift_duration_s = self.max_steps * self.dt           # hospital_env_vision ends the episode on this
        for k in ("leave_behind_steps", "lag_steps", "lag_far_m", "lag_max_tether", "lag_after_m", "lag_done",
                  "lag_wait", "reengaged"):
            self.v3_stats[k] = 0 if isinstance(self.v3_stats[k], int) else 0.0
        self._lag_seen = False
        self.last_obs_v2 = self._get_obs()
        return True

    def _largest_person_box(self) -> tuple[float, float, float]:
        """[seen, cx (-0.5..0.5), h] of the biggest person box in the front camera (YOLO patient / doctor classes)."""
        fl = floor_of(self.robot_x)
        people = [(p["x"], p["z"], p["seated"]) for p in self.patients if floor_of(p["x"]) == fl]
        people += [(n["x"], n["z"], bool(n.get("seated", False))) for n in self.npcs if floor_of(n["x"]) == fl]
        best = (0.0, 0.0, 0.0)
        for x, z, seated in people:
            if math.hypot(x - self.robot_x, z - self.robot_z) > 6.0:
                continue
            det = compute_pinhole_detection(
                robot_x=self.robot_x, robot_z=self.robot_z, robot_yaw=self.robot_yaw, target_x=x, target_z=z,
                target_seated=seated, is_blocked_fn=self._check_los_occluded,
                domain_randomization=self.domain_randomization, rng=self.rng)
            if det["seen"] > 0.5 and det["h"] > best[2]:
                best = (1.0, float(det["cx"]) * 0.5, float(det["h"]))
        return best

    def _person_items(self) -> list[tuple[float, float, bool]]:
        """People in the front camera (YOLO boxes): (cx -0.5..0.5, h, known). known = already met (helped, declined,
        engaged / escorted) or staff — what the robot's face / ReID gallery tells it (Godot: helped meta, staff role)."""
        fl = floor_of(self.robot_x)
        people = [(p["x"], p["z"], p["seated"], bool(p["served"] or p["declined"] or p["engaged"]))
                  for p in self.patients if floor_of(p["x"]) == fl]
        people += [(n["x"], n["z"], bool(n.get("seated", False)), True) for n in self.npcs if floor_of(n["x"]) == fl]
        out = []
        for x, z, seated, known in people:
            if math.hypot(x - self.robot_x, z - self.robot_z) > 6.0:
                continue
            det = compute_pinhole_detection(
                robot_x=self.robot_x, robot_z=self.robot_z, robot_yaw=self.robot_yaw, target_x=x, target_z=z,
                target_seated=seated, is_blocked_fn=self._check_los_occluded,
                domain_randomization=self.domain_randomization, rng=self.rng)
            if det["seen"] > 0.5:
                out.append((float(det["cx"]) * 0.5, float(det["h"]), known))
        return out

    def _update_track(self) -> None:
        """Escorted patient's position as the robot perceives it (box in the front / rear camera, not during a ReID miss;
        during a swap: the other person's position), noisy. Godot: ReID position estimate."""
        p = self._escorted()
        if p is None or floor_of(p["x"]) != floor_of(self.robot_x):
            if self.task_phase != TaskPhase.GUIDE_ESCORT:
                self.track.reset()
                self._returning = False
            return
        if self._track_phase != self.active_patient_idx:
            self.track.reset()
            self._track_phase = self.active_patient_idx
        q = p
        if self.domain_randomization and self._reid_swap_left > 0.0 and self._reid_swap_idx is not None:
            q = self.patients[self._reid_swap_idx]
        elif self.domain_randomization and self._reid_miss_left > 0.0:
            return
        if self._box_distance(q) is None:
            return
        n = TRACK_POS_NOISE_M if self.domain_randomization else 0.0
        self.track.update((q["x"] + float(self.rng.normal(0.0, n)), q["z"] + float(self.rng.normal(0.0, n))),
                          self.step_count * self.dt)

    def _rear_tether(self, p: dict[str, Any]) -> float | None:
        """Distance estimate of the escorted patient from the rear camera box (None = not in the rear view)."""
        det = compute_pinhole_detection(
            robot_x=self.robot_x, robot_z=self.robot_z, robot_yaw=self.robot_yaw + math.pi,
            target_x=p["x"], target_z=p["z"], target_seated=p["seated"], is_blocked_fn=self._check_los_occluded,
            domain_randomization=self.domain_randomization, rng=self.rng)
        if det["seen"] < 0.5:
            return None
        return float(np.clip(1.20 / max(float(det["h"]), 0.05), 0.5, 5.0))

    # ------------------------------------------------------------------ drills
    def setup_exit_drill(self) -> bool:
        """Short drill: the robot stands in a room it has just led someone to; that person stands between it and the
        door (sometimes right in the doorway). It has to get back to the corridor without pushing them. Returns False
        if no valid placement was found."""
        for _ in range(20):
            key = str(self.rng.choice(EXIT_ROOMS))
            gx, gz = DESTINATION_POSITIONS[key]
            fl = floor_of(gx)
            side = "north" if gz < 0.0 else "south"
            z_line = -2.0 if gz < 0.0 else 2.0
            door_x = min(_DOORS[(fl, side)][0], key=lambda c: abs(c - gx))
            sgn = 1.0 if z_line > gz else -1.0                   # room -> door along z
            rx = gx + float(self.rng.uniform(-0.6, 0.6))
            rz = gz + sgn * float(self.rng.uniform(0.9, 1.7))
            if is_blocked(rx, rz, ROBOT_RADIUS_M + 0.08):
                continue
            if self.rng.random() < 0.25:                         # right in the doorway
                px, pz = door_x + float(self.rng.uniform(-0.35, 0.35)), z_line - sgn * float(self.rng.uniform(0.0, 0.5))
            else:                                                # on the way to the door
                ux, uz = door_x - rx, z_line - rz
                n = math.hypot(ux, uz)
                ux, uz = ux / n, uz / n
                gap, lat = float(self.rng.uniform(0.6, min(1.6, n))), float(self.rng.uniform(-0.5, 0.5))
                px, pz = rx + ux * gap - uz * lat, rz + uz * gap + ux * lat
            if is_blocked(px, pz, 0.28) or math.hypot(px - rx, pz - rz) < PERSON_RADIUS_M + ROBOT_RADIUS_M + 0.03:
                continue
            to_p = math.atan2(-(px - rx), -(pz - rz))            # yaw looking at the person
            yaw = to_p + float(self.rng.uniform(-0.6, 0.6)) if self.rng.random() < 0.7 else float(self.rng.uniform(-math.pi, math.pi))
            self.robot_x, self.robot_z, self.robot_yaw = rx, rz, ((yaw + math.pi) % (2.0 * math.pi)) - math.pi
            self.robot_v = self.robot_omega = 0.0
            p = self.patients[0]
            p.update(x=px, z=pz, seated=False, served=True, engaged=True, needs_help=True, caller=False,
                     face_yaw=math.atan2(-(rx - px), -(rz - pz)))
            self._on_delivered(p, key)
            self.active_patient_idx = None
            self.patient_x = self.patient_z = -999.0
            self.task_phase = TaskPhase.SEARCH_PATIENT
            self.returning_to_lobby = fl > 1
            self.goal_key = f"thang_may_t{fl}" if fl > 1 else "patient"
            self.steps_since_task = 0
            self.drill = "exit"
            self.mem.reset()
            self.stall.reset()
            self._remember_depth()
            self.last_obs_v2 = self._get_obs()
            return True
        return False

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
                self.drill = "walkup"
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
        p_esc = self._escorted()
        if p_esc is not None:
            # "Anh/chị đi tiếp được không ạ?": to the escorted patient who stopped, close and in front of the robot
            ang, d = self._rel_angle(p_esc["x"], p_esc["z"])
            in_view = abs(ang) <= CAMERA_HFOV_RAD * 0.5 and not self._check_los_occluded(
                self.robot_x, self.robot_z, p_esc["x"], p_esc["z"])
            # in view within REENGAGE_M, or heard round a corner / through a doorway when this close
            if p_esc["lag_left"] > 0.0 and ((in_view and d <= REENGAGE_M) or d <= REENGAGE_HEAR_M):
                p_esc["lag_left"] = 0.0
                p_esc["lag_wait"] = False
                p_esc["speak_t"] = max(p_esc["speak_t"], 1.5)          # "ừ, đi tiếp"
                self.v3_stats["reengaged"] += 1
                return R_REENGAGE
        cands = [] if self._busy_escorting() else self._ask_candidates()
        if not cands:
            r_exc = self._excuse()
            if r_exc is not None:
                return r_exc
            self.v3_stats["asks_empty"] += 1
            return R_ASK_EMPTY
        i = min(cands, key=lambda c: c[1])[0]
        p = self.patients[i]
        if p["served"] or p["declined"]:
            r_exc = self._excuse()
            if r_exc is not None:
                return r_exc
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
        mine = next((f for f in faces if f["idx"] == d["idx"]), None)
        if mine is not None:
            d["face_s"] += self.dt
            self.v3_stats["face_steps"] += 1
            centred = 1.0 - min(1.0, (abs(mine["cx"]) + abs(mine["cy"])) / 0.5)
            self.v3_stats["centre_sum"] += centred
            r += R_FACE_DIALOGUE + R_FACE_CENTRE * centred
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
        if float(a[3]) > ASK_THRESHOLD and self.dialogue is None:
            r_v3 += self._ask()
        r_v3 += self._step_people()

        idx_before, goal_before = self.active_patient_idx, self.goal_key
        x_before, z_before = self.robot_x, self.robot_z
        cmd_v = (float(a[0]) + 1.0) * 0.5 * (ESCORT_WALK_SPEED if self.task_phase == TaskPhase.GUIDE_ESCORT else V_MAX)
        obs, r, done, info = super().step(a[:2])
        if info.get("patient_engaged") and self.active_patient_idx is not None:
            self.escort_t = 0.0
            q = self.patients[self.active_patient_idx]
            q["lag_at"] = float(self.rng.uniform(*LAG_AT_S)) if (self.domain_randomization and self.rng.random() < LAG_PROB) else -1.0
        if info.get("guide_completed") and idx_before is not None:
            self._on_delivered(self.patients[idx_before], goal_before)
        self._remember_depth()
        self._cue_pending = cmd_v
        # driving into someone who stands in the way (not the person being approached / escorted)
        if cmd_v > PUSH_CMD_V:
            fx, fz = -math.sin(self.robot_yaw), -math.cos(self.robot_yaw)
            for px, pz in self._patient_discs(skip_target_in=(TaskPhase.APPROACH_PATIENT, TaskPhase.GUIDE_ESCORT)):
                ox, oz = px - self.robot_x, pz - self.robot_z
                if math.hypot(ox, oz) <= PUSH_RANGE_M and ox * fx + oz * fz > 0.0:
                    r_v3 += R_PERSON_PUSH
                    self.v3_stats["push_steps"] += 1
                    break
        r_v3 += self._step_lag_bookkeeping(x_before, z_before)
        self._update_track()
        if self.drill == "exit" and not done and abs(self.robot_z) < EXIT_DONE_Z_M:
            r_v3 += R_EXIT_DRILL
            self.v3_stats["exit_ok"] += 1
            done = True
            info["exit_reason"] = "DRILL_DONE"
            info["truncated"] = False

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
        if self._cue_pending is not None:
            left, right = flank_proximity(obs[0:5], obs[OBS_MEM0:OBS_MEM0 + MEM_SECTORS])
            obs[OBS_STALL] = self.stall.update(self._cue_pending, self.robot_v, self.dt, self.dialogue is None, left, right,
                                               math.atan2(float(obs[14]), float(obs[15])))
            self._cue_pending = None
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
        # escort tether: the patient usually walks behind -> the rear camera counts too
        if (self.task_phase == TaskPhase.GUIDE_ESCORT and o[19] < 0.5 and self.active_patient_idx is not None
                and self.active_patient_idx < len(self.patients)):
            d_rear = self._rear_tether(self.patients[self.active_patient_idx])
            if d_rear is not None:
                self.tether_last_d, self.tether_unseen_s = d_rear, 0.0
                o[18:21] = tether_features(d_rear, True, 0.0)
        if self.task_phase == TaskPhase.GUIDE_ESCORT and self.domain_randomization:
            o[18:21] = self._reid_tether(o[18:21])
        o[OBS_MEM0:OBS_MEM0 + MEM_SECTORS] = self.mem.sectors(self.robot_x, self.robot_z, self.robot_yaw)
        o[OBS_STALL] = self.stall.obs()
        slot0, extra = person_slots(self._person_items())
        o[OBS_PERSON0:OBS_PERSON0 + 3] = slot0
        o[OBS_PERSON_X0:OBS_PERSON_X0 + PERSON_EXTRA_DIMS] = extra
        if self.task_phase == TaskPhase.GUIDE_ESCORT:
            o[OBS_TRACK0:OBS_TRACK0 + TRACK_DIMS] = self.track.obs(self.robot_x, self.robot_z, self.robot_yaw,
                                                                  self.step_count * self.dt)
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
        if self.dialogue is not None and o[34] < 1.0:
            # talking: the "goal" is the speaker (remembered voice direction; goal angle is + = left)
            o[13], o[14], o[15] = 0.0, -o[32], o[33]
        o[28] = self._tilt_norm()
        o[29] = min(self.since_ask_s / 5.0, 1.0)
        o[30] = 1.0 if self.dialogue is not None else 0.0
        # Face already met (helped or declined): Godot / real robot -> ReID / face gallery of past people
        if face is not None:
            q = self.patients[face["idx"]]
            o[31] = 1.0 if (q["served"] or q["declined"]) else 0.0
        return np.nan_to_num(o, nan=0.0, posinf=1.0, neginf=-1.0)

    # ------------------------------------------------------------------ teacher
    def _teacher_escape(self, o: np.ndarray, a_v: float, a_w: float) -> tuple[float, float]:
        """Getting past something the camera no longer sees (a person at the robot's flank): keep clear of remembered
        obstacles at the sides and, when stalled, stop pushing: turn away from the blocked side, then move on.
        A function of the observation only, so the policy can imitate it exactly."""
        s, m, cue = o[0:5], o[OBS_MEM0:OBS_MEM0 + MEM_SECTORS], float(o[OBS_STALL])
        person_ahead = o[OBS_PERSON0] > 0.5 and abs(float(o[OBS_PERSON0 + 1])) < 0.25 and float(o[OBS_PERSON0 + 2]) >= 0.55
        escort_ahead = (int(np.argmax(o[9:13])) == TaskPhase.GUIDE_ESCORT and o[19] > 0.5 and o[18] * 5.0 <= 1.6
                        and o[OBS_TRACK0] > 0.5 and o[OBS_TRACK0 + 2] > 0.5)   # it is the patient being escorted
        if person_ahead and s[2] > 0.8 and o[29] >= ASK_COOLDOWN_S / 5.0 and not escort_ahead:
            self._want_excuse = True                            # someone stands right ahead: ask them to make way
        if cue != 0.0:
            # recovery window (local_memory.RecoveryCue): turn away on the spot first, then drive clear
            front = float(max(s[2], 0.9 * max(s[1], s[3])))
            if RecoveryCue.elapsed_s(cue) <= RECOVERY_TURN_S or front > 0.85:
                a_v, a_w = -1.0, (1.0 if cue > 0.0 else -1.0)
            else:
                a_v, a_w = 0.2, 0.0
        else:
            push = 0.0                                          # remembered obstacle close at a flank: steer off it
            if m[0] > 0.7:
                push -= 0.6 * float(m[0])
            if m[MEM_SECTORS - 1] > 0.7:
                push += 0.6 * float(m[MEM_SECTORS - 1])
            a_w = float(np.clip(a_w + push, -1.0, 1.0))
        return a_v, a_w

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
            if not face_seen and o[34] < 1.0 and abs(math.atan2(o[32], o[33])) < 0.35:
                a_tilt = 0.8                                    # facing the voice, no face: they stand close -> look up
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
        if o[30] < 0.5 and phase == TaskPhase.GUIDE_ESCORT and patient_left_behind(o):
            a_v = -1.0                                          # patient seen far behind / lost after being far: wait
        still_s = float(o[OBS_TRACK0 + 4]) * STILL_NORM_S if o[OBS_TRACK0] > 0.5 else 0.0
        if phase != TaskPhase.GUIDE_ESCORT or o[OBS_TRACK0] < 0.5 or still_s < 1.0:
            self._returning = False
        elif patient_left_behind(o) and still_s >= RETURN_AFTER_S:
            self._returning = True                              # they are not coming: go back to them
        if o[30] < 0.5 and self._returning:
            b = math.atan2(float(o[OBS_TRACK0 + 1]), float(o[OBS_TRACK0 + 2]))   # + = right
            d_track = float(o[OBS_TRACK0 + 3]) * 5.0
            if o[19] > 0.5 and float(o[18]) * 5.0 <= REENGAGE_M - 0.3:
                # seen, close: stop, face them, ask
                a_v, a_w = -1.0, float(np.clip(-1.5 * b, -1.0, 1.0))
                if abs(b) < 0.35 and o[29] >= ASK_COOLDOWN_S / 5.0:
                    a_ask = 1.0                                  # "Anh/chị đi tiếp được không ạ?"
            elif d_track <= REENGAGE_HEAR_M - 0.4:
                # where they were last seen is this close but they are out of sight (wall / doorway): call them —
                # once, then again only after a pause (the teacher used to call every 3 s: 4 wasted asks / shift)
                a_v, a_w = -1.0, float(np.clip(-1.5 * b, -1.0, 1.0))
                now = self.step_count * self.dt
                if o[29] >= ASK_COOLDOWN_S / 5.0 and now - getattr(self, "_teacher_call_t", -1e9) >= 8.0:
                    a_ask = 1.0
                    self._teacher_call_t = now
            elif d_track > 0.8:
                # not seen yet (round a corner / out of view): head for where they were last seen
                s5 = o[0:5]
                repulsion = -0.9 * (s5[0] + 0.6 * s5[1]) + 0.9 * (s5[4] + 0.6 * s5[3])
                a_w = float(np.clip(-1.5 * b + 0.5 * repulsion / OMEGA_MAX, -1.0, 1.0))
                a_v = 0.2 if abs(b) < 0.6 else -1.0
            else:
                a_v, a_w = -1.0, float(np.clip(-1.5 * b, -1.0, 1.0))
        if o[30] < 0.5:
            a_v, a_w = self._teacher_escape(o, a_v, a_w)
        # ask the locked-on patient once the robot stops in front of them (V2 teacher stops at TARGET_BOX_H ~ 1.85 m)
        pat_seen, pat_cx, pat_h = o[5] > 0.5, float(o[6]), float(o[7])
        if (o[30] < 0.5 and o[31] < 0.5 and phase == TaskPhase.APPROACH_PATIENT and pat_seen and abs(pat_cx) <= 0.15
                and pat_h >= 0.9 * TARGET_BOX_H and o[29] >= ASK_COOLDOWN_S / 5.0):
            a_ask, a_v = 1.0, -1.0
        if getattr(self, "_want_excuse", False):
            self._want_excuse = False
            if o[30] < 0.5:
                a_ask, a_v = 1.0, -1.0
        return np.array([a_v, a_w, a_tilt, a_ask], dtype=np.float32)
