"""Unit and Integration Tests for Continuous Multi-Patient Hospital Shift Environment."""

import math
import numpy as np
import pytest

from src.training.hospital_env import DESTINATION_POSITIONS, ELEVATOR_POSITIONS, TaskPhase, floor_of, is_blocked
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


def test_no_phantom_reward_spike_on_acquisition():
    """Verify that acquiring a patient from search does NOT leak Euclidean distance (+3500 reward spike)."""
    env = HospitalPureVisionEnv(seed=42)
    obs = env.reset()
    assert env.task_phase == TaskPhase.SEARCH_PATIENT
    assert env.patient_x == -999.0

    # Step forward towards patient 0
    p0 = env.patients[0]
    env.robot_x = p0["x"] + 2.5
    env.robot_z = p0["z"]
    env.robot_yaw = math.pi / 2  # Facing -X towards patient

    obs, r, done, info = env.step(np.array([0.0, 0.0]))
    # Reward should be bounded normal step reward, NEVER > +20.0
    assert r < 20.0, f"Detected phantom reward spike on acquisition: r = {r}"
    assert env.cumulative_team_reward < 20.0, f"Cumulative reward inflated on acquisition: {env.cumulative_team_reward}"


def test_seated_patient_standup_and_escort_movement():
    """Verify that seated patients properly snap to legal floor on engagement and escort smoothly."""
    env = HospitalPureVisionEnv(seed=42)
    env.reset()

    # Patient 2 is seated in lobby chair (is_blocked was initially True)
    p2 = env.patients[2]
    assert p2["seated"] is True

    # Place robot close to trigger dialogue engagement
    env.active_patient_idx = 2
    env.task_phase = TaskPhase.APPROACH_PATIENT
    env.patient_x = p2["x"]
    env.patient_z = p2["z"]
    env.robot_x = p2["x"] - 0.5
    env.robot_z = 2.5  # In open corridor
    env.robot_yaw = 0.0

    obs, r, done, info = env.step(np.array([0.0, 0.0]))
    assert env.task_phase == TaskPhase.GUIDE_ESCORT, f"Expected GUIDE_ESCORT but got {env.task_phase}"
    assert p2["seated"] is False, "Patient should stand up when engaged"
    assert not is_blocked(p2["x"], p2["z"], 0.28), f"Patient must be on legal unblocked floor, got ({p2['x']}, {p2['z']})"

    # Robot moves forward, patient must step along companion formation
    prev_x, prev_z = p2["x"], p2["z"]
    for _ in range(5):
        env.step(np.array([0.5, 0.0]))
    # Patient position must have moved
    moved_dist = math.hypot(p2["x"] - prev_x, p2["z"] - prev_z)
    assert moved_dist > 0.05, f"Patient failed to move in escort formation: moved only {moved_dist}m"


def test_cross_floor_elevator_transit_and_room_delivery():
    """Verify that cross-floor patients are escorted into the elevator, ride to destination floor,

    delivered to their final room, and the robot returns to floor 1 lobby.
    """
    env = HospitalPureVisionEnv(seed=42)
    env.reset()

    # Patient 0 heading to Room 205 on Floor 2
    p0 = env.patients[0]
    p0["destination"] = ("guide", "205", 2)
    env.active_patient_idx = 0
    env.scenario_action, env.scenario_slot, env.scenario_floor = p0["destination"]
    env.task_phase = TaskPhase.APPROACH_PATIENT
    env.patient_x, env.patient_z = p0["x"], p0["z"]

    # 1. Engage patient
    env.robot_x, env.robot_z = p0["x"] - 0.5, 2.5
    env.step(np.array([0.0, 0.0]))
    assert env.task_phase == TaskPhase.GUIDE_ESCORT
    assert env.goal_key == "thang_may"

    # 2. Arrive at Floor 1 elevator
    elev1_x, elev1_z = ELEVATOR_POSITIONS[1]
    env.robot_x, env.robot_z = elev1_x, elev1_z
    p0["x"], p0["z"] = elev1_x + 0.5, elev1_z
    env.patient_x, env.patient_z = p0["x"], p0["z"]
    _, _, _, info_elev = env.step(np.array([0.0, 0.0]))
    assert floor_of(env.robot_x) == 2, "Robot must be on floor 2"
    assert floor_of(p0["x"]) == 2, "Patient must be on floor 2"
    assert env.goal_key == "205", "Goal key must update to final room on destination floor"
    assert p0["served"] is False, "Patient must not be marked served at the elevator!"
    assert info_elev.get("elevator_transit") is True

    # 3. Arrive at Room 205 on Floor 2
    r205_x, r205_z = DESTINATION_POSITIONS["205"]
    env.robot_x, env.robot_z = r205_x, r205_z
    p0["x"], p0["z"] = r205_x + 0.5, r205_z
    env.patient_x, env.patient_z = p0["x"], p0["z"]
    _, _, _, info_room = env.step(np.array([0.0, 0.0]))
    assert p0["served"] is True, "Patient must be marked served inside room 205"
    assert env.patients_helped == 1
    assert env.returning_to_lobby is True
    assert env.goal_key == "thang_may_t2"
    assert info_room.get("guide_completed") is True

    # 4. Return to Floor 1 via elevator
    elev2_x, elev2_z = ELEVATOR_POSITIONS[2]
    env.robot_x, env.robot_z = elev2_x, elev2_z
    _, _, _, info_return = env.step(np.array([0.0, 0.0]))
    assert floor_of(env.robot_x) == 1, "Robot must return to floor 1"
    assert env.returning_to_lobby is False
    assert env.goal_key == "patient"
    assert info_return.get("return_transit") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
