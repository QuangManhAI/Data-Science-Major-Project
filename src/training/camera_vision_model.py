"""Pinhole Perspective Camera Model for Fast Python Reinforcement Learning.

Accurately simulates the optical perspective projection of Godot 3D's Camera3D
and YOLOv8 human detection bounding boxes without requiring 3D rendering.

Mathematical formulation:
- Vertical FOV = 70.0 degrees (matches robot.tscn RobotCamera fov=70)
- Aspect Ratio = 640 / 480 (4:3)
- Horizontal FOV = 2 * atan( (4/3) * tan(35 deg) ) approx 86.4 degrees
- Line of sight (LOS) checking against hospital obstacle layout
- Perspective scaling: bounding box height h = f_y * H_human / D
- Normalized outputs matching YOLOv8 inference:
    seen: float (0.0 or 1.0)
    cx: float in [-1.0, 1.0] (0.0 = center of camera)
    cy: float in [-1.0, 1.0]
    w: float in [0.0, 1.0]
    h: float in [0.0, 1.0] (h >= 0.85 when distance <= 1.35m)
"""

from __future__ import annotations
import math
from typing import Sequence, Any, Optional

# Camera optical constants derived from robot.tscn
CAMERA_HEIGHT_M: float = 1.20
CAMERA_MOUNT_FORWARD_M: float = 0.16
CAMERA_VFOV_RAD: float = math.radians(70.0)  # 70 degrees
CAMERA_ASPECT: float = 640.0 / 480.0         # 4:3
CAMERA_HFOV_RAD: float = 2.0 * math.atan(CAMERA_ASPECT * math.tan(CAMERA_VFOV_RAD / 2.0))  # ~86.4 deg
CAMERA_MAX_RANGE_M: float = 8.0              # Max visual range for reliable YOLO detection
CAMERA_MIN_RANGE_M: float = 0.35

# Human standard physical dimensions
HUMAN_HEIGHT_M: float = 1.70
HUMAN_WIDTH_M: float = 0.48
SEATED_HEIGHT_M: float = 1.25

# Focal length in normalized screen height units
F_Y: float = 1.0 / (2.0 * math.tan(CAMERA_VFOV_RAD / 2.0))
F_X: float = 1.0 / (2.0 * math.tan(CAMERA_HFOV_RAD / 2.0))


def compute_pinhole_detection(
    robot_x: float,
    robot_z: float,
    robot_yaw: float,
    target_x: float,
    target_z: float,
    target_seated: bool = False,
    is_blocked_fn: Optional[Any] = None,
    domain_randomization: bool = False,
    rng: Optional[Any] = None,
) -> dict[str, float]:
    """Compute normalized YOLO-equivalent bounding box for a target person.

    Args:
        robot_x, robot_z: Robot world position.
        robot_yaw: Robot yaw in radians (Godot: facing -Z when yaw=0, left turns +yaw).
        target_x, target_z: Target world position.
        target_seated: Whether the target is seated (affects apparent height).
        is_blocked_fn: Optional callable (x1, z1, x2, z2) -> bool returning True if LOS is occluded.
        domain_randomization: Add realistic jitter (+/-5%) and frame drops (2%).
        rng: Random number generator instance.

    Returns:
        dict with keys: 'seen', 'cx', 'cy', 'w', 'h'
    """
    # Camera mount position offset
    cam_x = robot_x - math.sin(robot_yaw) * CAMERA_MOUNT_FORWARD_M
    cam_z = robot_z - math.cos(robot_yaw) * CAMERA_MOUNT_FORWARD_M

    dx = target_x - cam_x
    dz = target_z - cam_z
    dist = math.hypot(dx, dz)

    if dist < CAMERA_MIN_RANGE_M or dist > CAMERA_MAX_RANGE_M:
        return {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}

    # Robot forward vector is (-sin(yaw), -cos(yaw))
    # Robot right vector is (cos(yaw), -sin(yaw))
    fwd_x = -math.sin(robot_yaw)
    fwd_z = -math.cos(robot_yaw)
    right_x = math.cos(robot_yaw)
    right_z = -math.sin(robot_yaw)

    # Project target into camera local coordinate frame
    longitudinal = dx * fwd_x + dz * fwd_z   # forward distance (> 0 in front)
    lateral = dx * right_x + dz * right_z    # lateral offset (+ right, - left)

    if longitudinal <= 0.1:
        # Target is behind or parallel to camera plane
        return {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}

    # Normalized horizontal position on screen [-1.0, 1.0]
    # Screen center is 0.0. Right is > 0, left is < 0.
    norm_cx = (lateral / longitudinal) / math.tan(CAMERA_HFOV_RAD / 2.0)
    if abs(norm_cx) > 1.0:
        # Out of horizontal field of view
        return {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}

    # Check Line of Sight (LOS) occlusion if check function is provided
    if is_blocked_fn is not None:
        if is_blocked_fn(cam_x, cam_z, target_x, target_z):
            return {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}

    # Compute bounding box height and width using perspective projection
    actual_height = SEATED_HEIGHT_M if target_seated else HUMAN_HEIGHT_M
    norm_h = min(1.0, max(0.02, (actual_height / longitudinal) * F_Y))
    norm_w = min(1.0, max(0.01, (HUMAN_WIDTH_M / longitudinal) * F_X))

    # Center Y (camera is at 1.2m, human center is at 0.85m standing or 0.6m seated)
    target_center_y = 0.65 if target_seated else 0.85
    dy = target_center_y - CAMERA_HEIGHT_M
    norm_cy = max(-1.0, min(1.0, (dy / longitudinal) / math.tan(CAMERA_VFOV_RAD / 2.0)))

    # Apply Domain Randomization to simulate real-world YOLOv8 detection fluctuations
    if domain_randomization and rng is not None:
        # 2% chance of frame dropout (simulating temporary occlusion or YOLO miss)
        if float(rng.random()) < 0.02:
            return {"seen": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
        # +/- 4% bounding box jitter
        norm_cx = max(-1.0, min(1.0, norm_cx + float(rng.uniform(-0.04, 0.04))))
        norm_cy = max(-1.0, min(1.0, norm_cy + float(rng.uniform(-0.03, 0.03))))
        norm_h = max(0.02, min(1.0, norm_h * float(rng.uniform(0.95, 1.05))))
        norm_w = max(0.01, min(1.0, norm_w * float(rng.uniform(0.95, 1.05))))

    return {
        "seen": 1.0,
        "cx": float(norm_cx),
        "cy": float(norm_cy),
        "w": float(norm_w),
        "h": float(norm_h),
    }
