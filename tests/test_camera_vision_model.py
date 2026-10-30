"""Unit tests for Pinhole Perspective Camera Model for Visual RL."""

import math
import pytest
from src.training.camera_vision_model import compute_pinhole_detection, CAMERA_HFOV_RAD


def test_camera_detection_directly_in_front():
    # Robot at (0, 0) facing -Z (yaw = 0.0)
    # Human at (0, -1.4) directly in front at 1.4m
    res = compute_pinhole_detection(
        robot_x=0.0, robot_z=0.0, robot_yaw=0.0,
        target_x=0.0, target_z=-1.4, target_seated=False
    )
    assert res["seen"] == 1.0
    assert abs(res["cx"]) < 0.05, f"Expected centered cx, got {res['cx']}"
    # At 1.4m, bounding box height should be around 0.85 (Config.H_TARGET)
    assert 0.75 <= res["h"] <= 0.98, f"Expected h approx 0.85, got {res['h']}"


def test_camera_detection_behind_robot():
    # Human at (0, +2.0) behind the robot
    res = compute_pinhole_detection(
        robot_x=0.0, robot_z=0.0, robot_yaw=0.0,
        target_x=0.0, target_z=2.0, target_seated=False
    )
    assert res["seen"] == 0.0
    assert res["cx"] == 0.0
    assert res["h"] == 0.0


def test_camera_detection_outside_fov():
    # Robot facing -Z. Target is 90 degrees to the right (X = 3.0, Z = 0)
    res = compute_pinhole_detection(
        robot_x=0.0, robot_z=0.0, robot_yaw=0.0,
        target_x=3.0, target_z=0.0, target_seated=False
    )
    assert res["seen"] == 0.0


def test_camera_detection_lateral_offset():
    # Target in front at Z = -2.0, slightly to the right at X = 0.5
    res = compute_pinhole_detection(
        robot_x=0.0, robot_z=0.0, robot_yaw=0.0,
        target_x=0.5, target_z=-2.0, target_seated=False
    )
    assert res["seen"] == 1.0
    assert res["cx"] > 0.15, f"Expected positive cx (on right), got {res['cx']}"
    assert res["cx"] <= 1.0


def test_camera_detection_occlusion():
    # Target in front, but blocked by a wall function
    def always_blocked(x1, z1, x2, z2):
        return True

    res = compute_pinhole_detection(
        robot_x=0.0, robot_z=0.0, robot_yaw=0.0,
        target_x=0.0, target_z=-2.0,
        is_blocked_fn=always_blocked
    )
    assert res["seen"] == 0.0


def test_camera_detection_seated_shorter_height():
    standing = compute_pinhole_detection(0.0, 0.0, 0.0, 0.0, -2.5, target_seated=False)
    seated = compute_pinhole_detection(0.0, 0.0, 0.0, 0.0, -2.5, target_seated=True)
    assert standing["seen"] == 1.0 and seated["seen"] == 1.0
    assert seated["h"] < standing["h"], "Seated person should have smaller bounding box height"
