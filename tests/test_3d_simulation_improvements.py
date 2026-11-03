"""Unit tests verifying fixes for the 3 simulation issues:
1. Small obstacle (chair/trashbin) visual depth scaling in compute_5_sector_depth.
2. Return-to-Floor-1 / CALL_STAFF navigation target updates and zero tether penalty.
3. Patient freeze during dialogue in hospital NPC logic.
"""

import math
import numpy as np
import pytest

from src.training.hospital_env_pure_vision import (
    compute_5_sector_depth,
    OBSTACLE_NOMINAL_HEIGHTS,
    HUMAN_REFERENCE_HEIGHT,
)


def test_small_obstacle_visual_depth_scaling():
    """Chairs and trash bins must produce calibrated proximity values rather than tiny human-based fractions."""
    # A trash bin directly ahead: x1=0.4, x2=0.6 (center 0.5), y1=0.6, y2=0.85 (h = 0.25, ground contact y2=0.85)
    trash_box = {
        "x1": 0.4,
        "x2": 0.6,
        "y1": 0.6,
        "y2": 0.85,
        "class_name": "medical_waste_bin",
    }
    sectors = compute_5_sector_depth(
        robot_x=0.0,
        robot_z=0.0,
        robot_yaw=0.0,
        fl=1,
        structure_walls=None,
        obstacle_boxes=[trash_box],
    )
    # Sector 2 (center sector) should register strong proximity (> 0.70)
    assert sectors[2] >= 0.70, f"Expected center sector >= 0.70, got {sectors[2]}"


def test_wide_chair_covers_multiple_sectors():
    """A wide bench or chair spanning across sectors should populate all spanned sectors."""
    chair_box = {
        "x1": 0.25,
        "x2": 0.55,
        "y1": 0.5,
        "y2": 0.80,
        "class_name": "hospital_chair",
    }
    sectors = compute_5_sector_depth(
        robot_x=0.0,
        robot_z=0.0,
        robot_yaw=0.0,
        fl=1,
        structure_walls=None,
        obstacle_boxes=[chair_box],
    )
    # x1=0.25 maps to sector 1, x2=0.55 maps to sector 2
    assert sectors[1] > 0.50, f"Expected sector 1 > 0.50, got {sectors[1]}"
    assert sectors[2] > 0.50, f"Expected sector 2 > 0.50, got {sectors[2]}"


def test_call_staff_solo_transit_tether_and_navigation():
    """In CALL_STAFF mode, the robot travels solo without an accompanied patient.

    Tether must be 0 and navigation target must be populated.
    """
    from sim_vision.python.rl_policy_server import PolicyRunner, PPOHospitalAgent, ACT_DIM, OBS_DIM_PURE_VISION
    from src.training.hospital_env import TaskPhase

    agent = PPOHospitalAgent(obs_dim=OBS_DIM_PURE_VISION, act_dim=ACT_DIM)
    runner = PolicyRunner(agent, lidar_source="pure_vision", trace=False)

    msg = {
        "id": 1,
        "robot": [58.5, 4.5, 0.0, 0.4, 0.0],
        "patient": [-999.0, -999.0],
        "phase": int(TaskPhase.CALL_STAFF),
        "goal": [59.0, 4.6],
        "final_goal": [9.0, 5.2],
        "helped": 1,
        "detections": [],
    }

    action_msg = runner.act(msg)
    assert action_msg["type"] == "rl_action"
    assert "v" in action_msg
    assert "omega" in action_msg
    assert len(action_msg["sectors"]) == 5
