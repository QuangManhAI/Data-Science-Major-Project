"""Godot-exported nav layout, doorway routing, elevator rides and escort following in HospitalShiftEnv."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from src.training.hospital_env import (
    _DOORS,
    DESTINATION_POSITIONS,
    ELEVATOR_POSITIONS,
    HospitalShiftEnv,
    TaskPhase,
    floor_of,
    is_blocked,
)

RL_CONTROLLER_GD = Path(__file__).resolve().parents[1] / "sim_vision" / "godot" / "scripts" / "rl_controller.gd"


def test_godot_routing_doors_match_layout():
    """rl_controller.gd DOORS must equal the doors derived from hospital_layout.json."""
    text = RL_CONTROLLER_GD.read_text(encoding="utf-8")
    for (floor, side), (centres, dividers) in _DOORS.items():
        m = re.search(rf'"{floor}_{side}": \[\[([^\]]+)\], \[([^\]]+)\]\]', text)
        assert m, f"missing {floor}_{side} in rl_controller.gd"
        gd_centres = [float(v) for v in m.group(1).split(",")]
        gd_dividers = [float(v) for v in m.group(2).split(",")]
        assert np.allclose(gd_centres, centres, atol=1e-3)
        assert np.allclose(gd_dividers, dividers, atol=1e-3)


def test_destinations_are_reachable_free_space():
    for key, (x, z) in DESTINATION_POSITIONS.items():
        assert not is_blocked(x, z, 0.32), key


def test_robot_slides_along_wall_instead_of_freezing():
    env = HospitalShiftEnv(env_mode="solo")
    env.reset(seed=1)
    # Corridor, just below the north corridor wall (z = -2), heading diagonally into it
    env.robot_x, env.robot_z = -2.5, -1.4
    assert not is_blocked(env.robot_x, env.robot_z, 0.32)
    env.robot_yaw = math.radians(-45.0)  # forward = (-sin, -cos) -> (+x, -z) diagonal into the wall
    x0 = env.robot_x
    for _ in range(5):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.robot_x > x0 + 0.2
    assert not is_blocked(env.robot_x, env.robot_z, 0.32)


def test_search_takes_elevator_back_to_patient_floor():
    env = HospitalShiftEnv(env_mode="solo")
    env.reset(seed=2)
    env.task_phase = TaskPhase.SEARCH_PATIENT
    env.patient_x, env.patient_z = 0.0, 0.0  # floor 1
    env.robot_x, env.robot_z = ELEVATOR_POSITIONS[2]  # robot left on floor 2
    _, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert info["event"] == "elevator_ride"
    assert floor_of(env.robot_x) == 1


def test_escorted_patient_follows_trail_through_door_without_clipping():
    env = HospitalShiftEnv(env_mode="solo")
    env.reset(seed=3)
    env.task_phase = TaskPhase.GUIDE_ESCORT
    env.goal_key = "110"
    # Robot walks from the corridor through the north door at x=0 into room 110
    env.robot_x, env.robot_z, env.robot_yaw = 2.5, 0.0, math.pi / 2  # facing -x
    env.patient_x, env.patient_z = 3.5, 0.0
    path = [(1.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, -1.5), (0.0, -2.5), (0.0, -3.5), (0.0, -4.0)]
    for tx, tz in path:
        for _ in range(15):
            env.robot_x += float(np.clip(tx - env.robot_x, -0.08, 0.08))
            env.robot_z += float(np.clip(tz - env.robot_z, -0.08, 0.08))
            env._step_escorted_patient()
            assert not is_blocked(env.patient_x, env.patient_z, 0.0), (env.patient_x, env.patient_z)
    for _ in range(60):
        env._step_escorted_patient()
    assert env.patient_z < -2.1, "patient followed the robot through the doorway into the room"
    assert math.hypot(env.robot_x - env.patient_x, env.robot_z - env.patient_z) <= 1.0
