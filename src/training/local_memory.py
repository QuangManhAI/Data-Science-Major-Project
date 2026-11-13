"""What the robot still knows about its surroundings outside the camera view (shared by training and Godot serving).

The depth camera covers ~86 deg ahead. A person the robot has just turned away from (e.g. the patient it delivered,
standing between it and the door) leaves the view but is still there; without memory the policy drives into them.
A real robot keeps a local obstacle map from depth + odometry; this is the smallest version of it:

- ObstacleMemory: depth hits of the last few seconds, kept in the world frame, reported as proximity in 6 sectors
  around the robot outside the camera view (left side .. behind .. right side).
- RecoveryCue: the robot commands forward speed without moving (wheel odometry vs command) -> a recovery window with
  a fixed turn side, visible to the policy.

The same classes build the observation in hospital_env_engage_v3.py and in sim_vision/python/rl_policy_server.py,
so training and Godot cannot drift apart.
"""

from __future__ import annotations

import math

import numpy as np

MEM_KEEP_S = 8.0                 # hits older than this are forgotten (people move on)
MEM_RANGE_M = 2.0                # only the near field matters for manoeuvring
MEM_SECTORS = 6
MEM_JUMP_M = 3.0                 # pose jump (elevator) -> forget everything
MEM_FOV_MARGIN_RAD = math.radians(3.0)
STALL_CMD_V = 0.15               # commanded speed above this ...
STALL_REAL_V = 0.05              # ... while really moving slower than this = stalled
STALL_TRIGGER_S = 0.4            # stalled this long -> recovery window
RECOVERY_S = 2.2                 # turn away (first RECOVERY_TURN_S), then drive clear
RECOVERY_TURN_S = 0.8

def memory_sector_edges(hfov_rad: float) -> np.ndarray:
    """Bearing edges (rad, + = right) of the 6 sectors outside the camera view, going left side -> behind -> right side:
    [-hfov/2 .. -90], [-90 .. -135], [-135 .. -180] and mirrored. Returned as 3 edges of the absolute bearing."""
    return np.array([hfov_rad * 0.5, math.radians(90.0), math.radians(135.0), math.pi + 1e-6])


class ObstacleMemory:
    def __init__(self, hfov_rad: float, max_range_m: float, robot_radius_m: float) -> None:
        self.hfov = float(hfov_rad)
        self.max_range = float(max_range_m)
        self.radius = float(robot_radius_m)
        self.edges = memory_sector_edges(self.hfov)
        self.reset()

    def reset(self) -> None:
        self.px = np.zeros(0, dtype=np.float64)
        self.pz = np.zeros(0, dtype=np.float64)
        self.pt = np.zeros(0, dtype=np.float64)
        self._last_pose: tuple[float, float] | None = None

    def _bearing_dist(self, x: float, z: float, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        dx, dz = self.px - x, self.pz - z
        fwd = dx * -math.sin(yaw) + dz * -math.cos(yaw)
        right = dx * math.cos(yaw) + dz * -math.sin(yaw)
        return np.arctan2(right, fwd), np.hypot(dx, dz)

    def update(self, x: float, z: float, yaw: float, depth_m: np.ndarray, t: float) -> None:
        """Add this frame's depth hits (ray distances in m, left -> right across the HFOV)."""
        if self._last_pose is not None and math.hypot(x - self._last_pose[0], z - self._last_pose[1]) > MEM_JUMP_M:
            self.reset()
        self._last_pose = (x, z)
        if self.px.size:
            b, d = self._bearing_dist(x, z, yaw)
            # forget: too old, far away, or inside the current view (the fresh depth frame replaces them)
            keep = (t - self.pt <= MEM_KEEP_S) & (d <= self.max_range + 1.0) \
                & (np.abs(b) > self.hfov * 0.5 - MEM_FOV_MARGIN_RAD)
            self.px, self.pz, self.pt = self.px[keep], self.pz[keep], self.pt[keep]
        d = np.asarray(depth_m, dtype=np.float64).reshape(-1)
        rel = np.linspace(-self.hfov * 0.5, self.hfov * 0.5, d.size)
        hit = d < self.max_range - 0.05
        if hit.any():
            ray_yaw = yaw - rel[hit]                               # +rel = right
            self.px = np.concatenate([self.px, x - np.sin(ray_yaw) * d[hit]])
            self.pz = np.concatenate([self.pz, z - np.cos(ray_yaw) * d[hit]])
            self.pt = np.concatenate([self.pt, np.full(int(hit.sum()), t)])

    def sectors(self, x: float, z: float, yaw: float) -> np.ndarray:
        """Proximity (0 = nothing within MEM_RANGE_M, 1 = touching) per sector:
        [left side, left rear, left back, right back, right rear, right side]."""
        out = np.zeros(MEM_SECTORS, dtype=np.float32)
        if not self.px.size:
            return out
        b, d = self._bearing_dist(x, z, yaw)
        ab = np.abs(b)
        for k in range(3):
            in_ring = (ab >= self.edges[k]) & (ab < self.edges[k + 1])
            for left in (True, False):
                m = in_ring & ((b < 0.0) if left else (b >= 0.0))
                if m.any():
                    prox = 1.0 - (float(d[m].min()) - self.radius) / (MEM_RANGE_M - self.radius)
                    out[k if left else MEM_SECTORS - 1 - k] = float(np.clip(prox, 0.0, 1.0))
        return out


def flank_proximity(sectors5: np.ndarray, mem6: np.ndarray) -> tuple[float, float]:
    """How close something is on the (left, right): edge of the camera view or remembered just outside it."""
    left = float(max(sectors5[0], 0.8 * sectors5[1], mem6[0]))
    right = float(max(sectors5[4], 0.8 * sectors5[3], mem6[MEM_SECTORS - 1]))
    return left, right


class RecoveryCue:
    """Stall detector with a recovery reflex the policy can see.

    The robot is *stalled* when it commands forward speed without moving (wheel odometry vs command) for
    STALL_TRIGGER_S. That starts a recovery window of RECOVERY_S; the cue is +-(time left / RECOVERY_S), the sign
    being the side to turn to (+ = left): away from the closer flank, else towards the goal. It is fixed for the whole
    window, so a policy without memory does not dither between left and right, and it knows how far into the
    manoeuvre it is (turn away first, then drive clear). The policy stays free to do something else."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stall_s = 0.0
        self.left_s = 0.0
        self.dir = 0.0

    def update(self, cmd_v: float, real_v: float, dt: float, active: bool, left: float, right: float, goal_angle: float) -> float:
        if active and cmd_v > STALL_CMD_V and abs(real_v) < STALL_REAL_V:
            self.stall_s += dt
        else:
            self.stall_s = 0.0
        if self.left_s > 0.0:
            self.left_s = max(0.0, self.left_s - dt)
        elif self.stall_s >= STALL_TRIGGER_S - 1e-9:
            if abs(left - right) > 0.1:
                self.dir = -1.0 if left > right else 1.0        # + = turn left: blocked on the left -> turn right
            else:
                self.dir = 1.0 if goal_angle >= 0.0 else -1.0
            self.left_s = RECOVERY_S
        return self.obs()

    def obs(self) -> float:
        return float(self.dir * self.left_s / RECOVERY_S) if self.left_s > 0.0 else 0.0

    @staticmethod
    def elapsed_s(cue: float) -> float:
        """Seconds since the recovery window started, from the cue value."""
        return (1.0 - abs(float(cue))) * RECOVERY_S
