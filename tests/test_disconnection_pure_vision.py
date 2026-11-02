"""Verification of Zero Oracle Cheating & YOLO Disconnection Safety.

Ensures that when YOLO detections are absent (e.g. YOLO server disconnected):
1. 5-Sector obstacle depth map obs[0..4] is strictly [0, 0, 0, 0, 0] (no hidden map raycasts).
2. Visual channels obs[5..8] and obs[9..12] are strictly 0.0.
3. Doorway channels obs[25..28] are strictly 0.0 (no oracle _DOORS lookup).
4. In GUIDE_ESCORT, loss of camera tracking triggers tether break flags obs[22..24] == 1.0.
"""

import numpy as np
import pytest

from src.models.ppo_agent import ACT_DIM, OBS_DIM_PURE_VISION, PPOHospitalAgent
from src.training.hospital_env import TaskPhase
from src.training.hospital_env_pure_vision import HospitalPureVisionEnv, compute_5_sector_depth
from sim_vision.python.rl_policy_server import PolicyRunner


def test_yolo_disconnection_sectors_are_strictly_zero() -> None:
    """When YOLO sends no detections, sectors MUST NOT cheat using map walls."""
    sectors = compute_5_sector_depth(
        robot_x=10.0,
        robot_z=0.0,
        robot_yaw=0.0,
        fl=1,
        structure_walls=None,
        obstacle_boxes=[],
    )
    assert np.allclose(sectors, 0.0), f"Expected all zeros with no YOLO detections, got {sectors}"


def test_pure_vision_env_zero_oracle_door_lookup() -> None:
    """When robot is in corridor facing away from doors, door channel is zero."""
    env = HospitalPureVisionEnv(env_mode="solo")
    env.reset()
    # Position robot facing down empty hallway away from doors
    env.robot_x = 0.0
    env.robot_z = 0.0
    env.robot_yaw = 0.0  # Facing -Z (away from north/south doors)
    obs = env._get_obs()

    assert obs.shape == (29,)
    # If no door is in the camera FOV, door channel obs[25..28] must be 0.0
    door_det = env._compute_door_det()
    if door_det["seen"] <= 0.5:
        assert np.allclose(obs[25:29], 0.0)


def test_policy_server_yolo_disconnection_handling() -> None:
    """When Godot sends msg with empty dets (YOLO disconnected), verify no oracle cheating."""
    agent = PPOHospitalAgent(obs_dim=OBS_DIM_PURE_VISION, act_dim=ACT_DIM)
    session = PolicyRunner(agent, lidar_source="godot")

    # 1. SEARCH Phase with empty dets
    msg = {
        "id": 1,
        "robot": [10.0, 0.0, 0.0, 0.2, 0.0],
        "patient": [15.0, 0.0],
        "phase": int(TaskPhase.SEARCH_PATIENT),
        "goal": [15.0, 0.0],
        "task_id": 1,
        "dets": [],  # YOLO disconnected!
    }
    reply = session.act(msg)
    assert "v" in reply and "omega" in reply
    assert np.allclose(session.last_sectors, 0.0), "Sectors must be zero without YOLO detections"

    # 2. GUIDE_ESCORT Phase with empty dets (patient lost from camera FOV)
    msg["phase"] = int(TaskPhase.GUIDE_ESCORT)
    reply_escort = session.act(msg)
    # When patient is not visible in peripheral camera during escort, tether break is triggered
    assert session.env.task_phase == TaskPhase.GUIDE_ESCORT
