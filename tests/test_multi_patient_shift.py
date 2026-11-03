"""Unit and Integration Tests for Continuous Multi-Patient Hospital Shift Environment."""

import math
import numpy as np
import pytest

from src.training.hospital_env import TaskPhase, floor_of
from src.training.hospital_env_pure_vision import HospitalPureVisionEnv


def test_entrance_spawn_and_multi_patient_population():
    env = HospitalPureVisionEnv(shift_duration_s=300.0, seed=123)
    obs = env.reset()

    # 1. Robot must start strictly at Floor 1 entrance/dock
    assert math.isclose(env.robot_x, -10.6, abs_tol=0.2)
    assert math.isclose(env.robot_z, 0.0, abs_tol=0.2)
    assert floor_of(env.robot_x) == 1

    # 2. Floor 1 must be populated with 6 patients, all unserved
    assert len(env.patients) == 6
    for p in env.patients:
        assert p["served"] is False
        assert floor_of(p["x"]) == 1
        assert "destination" in p

    # 3. Initially in SEARCH_PATIENT with no locked patient
    assert env.task_phase == TaskPhase.SEARCH_PATIENT
    assert env.active_patient_idx is None
    assert obs.shape == (29,)


def test_visual_patient_acquisition_and_approach():
    env = HospitalPureVisionEnv(shift_duration_s=300.0, seed=456)
    env.reset()

    # Place robot in front of Patient_01 from the open walkway
    p0 = env.patients[0]
    p0_x, p0_z = p0["x"], p0["z"]

    # Position robot 2.0m in front (+X direction), facing -X directly at patient
    env.robot_x = p0_x + 2.0
    env.robot_z = p0_z
    env.robot_yaw = math.pi / 2.0

    obs = env._get_obs()
    # Should detect patient in camera FOV and automatically transition to APPROACH_PATIENT
    assert env.task_phase == TaskPhase.APPROACH_PATIENT
    assert env.active_patient_idx == 0
    assert obs[5] > 0.5  # pat_seen == 1.0


def test_served_patient_anti_reward_farming():
    env = HospitalPureVisionEnv(shift_duration_s=300.0, seed=789)
    env.reset()

    # Mark all patients as served
    for p in env.patients:
        p["served"] = True
    p0 = env.patients[0]

    # Face robot directly at served patient
    env.robot_x = p0["x"] + 2.0
    env.robot_z = p0["z"]
    env.robot_yaw = math.pi / 2.0

    # In SEARCH_PATIENT, served patient must be ignored
    env.task_phase = TaskPhase.SEARCH_PATIENT
    env.active_patient_idx = None

    obs = env._get_obs()
    # pat_seen for unserved patients must be 0 (all are served)
    assert env.active_patient_idx is None
    assert env.task_phase == TaskPhase.SEARCH_PATIENT


def test_elevator_return_transit():
    env = HospitalPureVisionEnv(shift_duration_s=300.0, seed=999)
    env.reset()

    # Simulate delivery on Floor 2
    env.robot_x = 50.0  # Floor 2
    env.robot_z = -5.2
    env.returning_to_lobby = True
    env.task_phase = TaskPhase.SEARCH_PATIENT
    env.goal_key = "thang_may_t2"

    # Step robot to the Floor 2 elevator
    env.robot_x = 62.8
    env.robot_z = 4.6

    obs, reward, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    # Robot must have transitioned down to Floor 1!
    assert floor_of(env.robot_x) == 1
    assert env.returning_to_lobby is False
    assert info.get("return_transit", False) is True
    assert reward >= 4.0  # Got elevator return transit bonus


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
