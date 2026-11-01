"""Unit tests for Pure-Vision (Zero LiDAR) RL Environment and Observation Space."""

import math
import numpy as np
import pytest
import torch

from src.models.ppo_agent import ACT_DIM, OBS_DIM_PURE_VISION, PPOHospitalAgent
from src.training.hospital_env import TaskPhase
from src.training.hospital_env_pure_vision import (
    HospitalPureVisionEnv,
    compute_5_sector_depth,
)


def test_pure_vision_obs_dimension():
    env = HospitalPureVisionEnv(env_mode="solo")
    obs = env.reset()
    assert obs.shape == (29,), f"Expected 29-D observation, got {obs.shape}"
    assert obs.dtype == np.float32


def test_pure_vision_agent_forward():
    agent = PPOHospitalAgent(obs_dim=OBS_DIM_PURE_VISION, act_dim=ACT_DIM)
    dummy_obs = torch.zeros((4, 29), dtype=torch.float32)
    action, logp, entropy, val = agent.get_action_and_value(dummy_obs)
    assert action.shape == (4, 2)
    assert val.shape == (4,)


def test_5_sector_depth_mapping():
    walls = {1: np.array([[12.0, -9.0, 12.0, 9.0]], dtype=np.float32)}
    # Obstacle bounding box right in the center (cx = 0.0, h = 0.8)
    boxes = [{"cx": 0.0, "h": 0.8}]
    sectors = compute_5_sector_depth(
        robot_x=0.0,
        robot_z=0.0,
        robot_yaw=0.0,
        fl=1,
        structure_walls=walls,
        obstacle_boxes=boxes,
    )
    assert sectors.shape == (5,)
    # Center sector (index 2) should have high obstacle proximity >= 0.8
    assert sectors[2] >= 0.8, f"Expected Center sector >= 0.8, got {sectors[2]}"


def test_doctor_channel_and_avoidance():
    env = HospitalPureVisionEnv(env_mode="full")
    env.doctor_active = True
    env.doctor_x = env.robot_x
    env.doctor_z = env.robot_z - 1.2  # 1.2m directly in front of robot
    env.robot_yaw = 0.0  # Facing north (-Z)
    obs = env._get_obs()

    # Doctor channel is obs[9..12]
    doc_seen = obs[9]
    doc_h = obs[11]
    assert doc_seen > 0.5, "Doctor should be visually detected in front of camera"
    assert doc_h > 0.35, f"Doctor height should be significant at 1.2m, got {doc_h}"

    # Step should trigger doctor avoidance penalty
    act = np.array([0.0, 0.0], dtype=np.float32)
    _, r, _, _ = env.step(act)
    assert r < -2.0, f"Expected doctor avoidance penalty, got reward {r}"
