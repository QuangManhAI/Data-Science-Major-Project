"""Unit tests for Pure Visual-Search RL Environment."""

import math
import numpy as np
import pytest
from src.training.hospital_env import TaskPhase, OBS_DIM
from src.training.hospital_env_vision import HospitalVisualShiftEnv, PUBLIC_PATROL_WAYPOINTS
from src.training.hospital_env import floor_of
from src.training.camera_vision_model import CAMERA_HFOV_RAD


def test_visual_env_zero_oracle_during_search():
    """Verify that during search, the robot receives corridor patrol waypoints,

    NEVER ground-truth patient coordinates.
    """
    env = HospitalVisualShiftEnv(seed=123, domain_randomization=False)
    env.reset(seed=123)

    # Ensure patient is out of sight so env is in SEARCH_PATIENT phase
    env.task_phase = TaskPhase.SEARCH_PATIENT
    env.patient_x = 100.0  # Far away / occluded
    env.patient_z = 100.0
    obs = env._get_obs()

    assert obs.shape == (OBS_DIM,), f"Expected obs shape ({OBS_DIM},), got {obs.shape}"
    assert env.task_phase == TaskPhase.SEARCH_PATIENT

    # Check that obs[25..27] is pointing to a patrol waypoint, NOT patient position
    fl = floor_of(env.robot_x)
    waypoints = PUBLIC_PATROL_WAYPOINTS[fl]
    patrol_wp = waypoints[env.patrol_wp_idx % len(waypoints)]
    dx_wp = patrol_wp[0] - env.robot_x
    dz_wp = patrol_wp[1] - env.robot_z
    dist_wp = math.hypot(dx_wp, dz_wp)
    expected_nav = float(np.clip(dist_wp / 25.0, 0.0, 1.0))

    assert abs(obs[25] - expected_nav) < 1e-3, (
        f"Expected obs[25] to match patrol waypoint distance {expected_nav}, got {obs[25]}"
    )

    # Verify zero-oracle: obs[25] does not encode the 100m patient distance
    assert abs(obs[25] - 1.0) > 0.05, "obs[25] must NOT be the patient ground-truth coordinate!"


def test_vision_derived_perception_vector_when_patient_visible():
    """When a patient is spotted in camera FOV, range and bearing must be

    derived strictly from 2D bounding box (cx, h), not ground truth.
    """
    env = HospitalVisualShiftEnv(seed=42, domain_randomization=False)
    env.reset(seed=42)

    # Place robot at (0.0, 0.0) facing -Z (yaw = 0.0)
    env.robot_x = 0.0
    env.robot_z = 0.0
    env.robot_yaw = 0.0
    # Place patient at (0.5, -2.5) -> visible in camera FOV
    env.patient_x = 0.5
    env.patient_z = -2.5

    obs, reward, done, info = env.step((0.0, 0.0))

    assert env.task_phase == TaskPhase.APPROACH_PATIENT
    assert obs[17] == 1.0, f"Expected has_det == 1.0, got {obs[17]}"

    cx = obs[18]
    h = obs[19]
    assert h > 0.1, f"Expected bounding box height > 0.1, got {h}"

    # Pinhole optics formula verification:
    # d_vis = 1.20 / h, theta_vis = -atan(cx * tan(HFOV / 2))
    expected_d_vis = float(np.clip(1.20 / max(h, 0.05), 0.8, 15.0))
    expected_nav_vis = float(np.clip(expected_d_vis / 25.0, 0.0, 1.0))
    expected_theta_vis = float(-math.atan(cx * math.tan(CAMERA_HFOV_RAD / 2.0)))

    assert abs(obs[25] - expected_nav_vis) < 1e-3, (
        f"Expected vision-derived nav {expected_nav_vis}, got {obs[25]}"
    )
    assert abs(obs[26] - math.sin(expected_theta_vis)) < 1e-3
    assert abs(obs[27] - math.cos(expected_theta_vis)) < 1e-3


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


def test_stagnation_resets_on_exploration_and_visual_progress():
    """Verify that exploration and visual approach progress reset stagnation counter."""
    env = HospitalVisualShiftEnv(seed=99, domain_randomization=False)
    env.reset(seed=99)

    # Initial cell visited
    assert env.steps_since_task == 0

    # Step forward with v = 0.8
    _, _, _, info = env.step((1.0, 0.0))
    # Stagnation timer should stay low due to movement/exploration
    assert env.steps_since_task <= 1

    # Place patient ahead to test visual progress reset
    env.patient_x = env.robot_x
    env.patient_z = env.robot_z - 3.0
    env.robot_yaw = 0.0  # Facing -Z toward patient

    env.steps_since_task = 50  # Artificially age stagnation
    # Step forward closing distance
    env.step((1.0, 0.0))
    # Since bbox height increased, stagnation counter must reset to 0
    assert env.steps_since_task == 0, (
        f"Expected steps_since_task to reset on visual progress, got {env.steps_since_task}"
    )


def test_exploration_cell_coverage_reward():
    env = HospitalVisualShiftEnv(seed=99, domain_randomization=False)
    env.reset(seed=99)

    initial_visited = len(env.visited_cells)
    assert initial_visited == 1

    # Drive forward to new cells
    total_r_exp = 0.0
    for _ in range(15):
        _, _, _, info = env.step((1.0, 0.0))
        total_r_exp += info["r_exploration"]

    assert len(env.visited_cells) > initial_visited, "Expected new cells visited"
    assert total_r_exp > 0.0, f"Expected exploration rewards, got {total_r_exp}"
