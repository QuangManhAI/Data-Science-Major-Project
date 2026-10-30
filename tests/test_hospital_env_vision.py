"""Unit tests for Pure Visual-Search RL Environment."""

import numpy as np
import pytest
from src.training.hospital_env import TaskPhase, OBS_DIM
from src.training.hospital_env_vision import HospitalVisualShiftEnv


def test_visual_env_zero_oracle_during_search():
    env = HospitalVisualShiftEnv(seed=123, domain_randomization=False)
    obs = env.reset(seed=123)

    assert obs.shape == (OBS_DIM,), f"Expected obs shape ({OBS_DIM},), got {obs.shape}"
    assert env.task_phase in (TaskPhase.SEARCH_PATIENT, TaskPhase.APPROACH_PATIENT)

    # In zero-oracle mode, goal polar vector obs[25..27] MUST be 0.0 during search!
    assert obs[25] == 0.0, f"Expected obs[25] (nav_remaining) == 0.0, got {obs[25]}"
    assert obs[26] == 0.0, f"Expected obs[26] (sin angle) == 0.0, got {obs[26]}"
    assert obs[27] == 0.0, f"Expected obs[27] (cos angle) == 0.0, got {obs[27]}"


def test_visual_detection_activation_when_facing_patient():
    env = HospitalVisualShiftEnv(seed=42, domain_randomization=False)
    env.reset(seed=42)

    # Place robot at (0.0, 0.0) facing -Z (yaw = 0.0)
    env.robot_x = 0.0
    env.robot_z = 0.0
    env.robot_yaw = 0.0
    # Place patient directly in front at (0.0, -2.0)
    env.patient_x = 0.0
    env.patient_z = -2.0

    obs, reward, done, info = env.step((0.0, 0.0))
    # obs[17] is has_det
    assert obs[17] == 1.0, f"Expected has_det == 1.0, got {obs[17]}"
    assert abs(obs[18]) < 0.1, f"Expected centered cx, got {obs[18]}"
    assert obs[19] > 0.3, f"Expected positive h, got {obs[19]}"
    assert info["has_det"] == 1.0
    assert info["r_visual"] > 0.0, "Expected visual reward when spotting patient"


def test_visual_engagement_switches_to_escort_and_activates_goal():
    env = HospitalVisualShiftEnv(seed=42, domain_randomization=False)
    env.reset(seed=42)

    # Place robot right in front of patient within talk distance (1.2m)
    env.robot_x = 0.0
    env.robot_z = 0.0
    env.robot_yaw = 0.0
    env.patient_x = 0.0
    env.patient_z = -1.2

    obs, reward, done, info = env.step((0.0, 0.0))

    # Engagement must trigger!
    assert info["event"] == "engaged", f"Expected engaged event, got {info['event']}"
    assert env.task_phase == TaskPhase.GUIDE_ESCORT
    assert env.goal_key != "", "Expected resolved destination goal key"

    # Now that we are in GUIDE_ESCORT, goal navigation is ACTIVE (obs[25..27] non-zero)
    obs_next = env._get_obs()
    assert obs_next[25] > 0.0, f"Expected nav_remaining > 0 in escort, got {obs_next[25]}"
    assert (obs_next[26] ** 2 + obs_next[27] ** 2) > 0.5, "Expected active goal direction vector"


def test_exploration_cell_coverage_reward():
    env = HospitalVisualShiftEnv(seed=99, domain_randomization=False)
    env.reset(seed=99)

    # Initial cell is already visited
    initial_visited = len(env.visited_cells)
    assert initial_visited == 1

    # Drive forward to a new cell
    # Step forward with v = 0.8 for multiple steps
    total_r_exp = 0.0
    for _ in range(15):
        _, _, _, info = env.step((1.0, 0.0))
        total_r_exp += info["r_exploration"]

    assert len(env.visited_cells) > initial_visited, "Expected new cells visited"
    assert total_r_exp > 0.0, f"Expected exploration rewards, got {total_r_exp}"
