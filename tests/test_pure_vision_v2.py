"""Pure-Vision V2: depth observation, reward design, teacher sign conventions, policy server wiring."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim_vision" / "python"))

from src.models.ppo_agent import OBS_DIM_PURE_VISION_V2, PPOHospitalAgent  # noqa: E402
from src.training.hospital_env import TaskPhase  # noqa: E402
from src.training.hospital_env_pure_vision_v2 import (  # noqa: E402
    DEPTH_MAX_RANGE_M,
    OBS_DIM_V2,
    R_STEP,
    HospitalPureVisionV2Env,
    depth_rays_to_sectors,
)


def make_env(seed: int = 3) -> HospitalPureVisionV2Env:
    return HospitalPureVisionV2Env(seed=seed, domain_randomization=False)


def run_until_escort(env: HospitalPureVisionV2Env, max_steps: int = 1500) -> np.ndarray:
    obs = env.reset()
    for _ in range(max_steps):
        obs, _, done, _ = env.step(env.teacher_action())
        if env.task_phase == TaskPhase.GUIDE_ESCORT or done:
            break
    assert env.task_phase == TaskPhase.GUIDE_ESCORT, "teacher should engage a patient within the budget"
    return obs


def test_obs_dims_and_ranges():
    assert OBS_DIM_V2 == OBS_DIM_PURE_VISION_V2 == 21
    obs = make_env().reset()
    assert obs.shape == (21,) and np.all(np.isfinite(obs))
    assert np.all((obs[0:5] >= 0.0) & (obs[0:5] <= 1.0))
    assert obs[9:13].sum() == pytest.approx(1.0)


def test_depth_sees_walls_not_only_doctor():
    """V1 sectors only ever contained the doctor; V2 depth must register static walls/furniture."""
    env = make_env()
    env.reset()
    env.doctor_active = False
    hits = 0
    for yaw in np.linspace(-math.pi, math.pi, 16, endpoint=False):
        env.robot_yaw = float(yaw)
        hits += int(np.any(env.depth_rays() < DEPTH_MAX_RANGE_M))
    assert hits > 0


def test_depth_sector_mapping():
    d = np.full(15, DEPTH_MAX_RANGE_M, dtype=np.float32)
    d[0] = 0.32  # touching on the far left
    s = depth_rays_to_sectors(d)
    assert s[0] == pytest.approx(1.0) and np.all(s[1:] == 0.0)


def test_teacher_turns_away_from_obstacles():
    """+omega = turn left. Obstacle on the left must give omega < 0 (V1 teacher had this sign inverted)."""
    env = make_env()
    env.reset()
    o = np.zeros(21, dtype=np.float32)
    o[9] = 1.0          # SEARCH
    o[15] = 1.0         # goal straight ahead (sin 0, cos 1)
    o[0:2] = [0.8, 0.6]
    env.last_obs_v2 = o
    assert env.teacher_action()[1] < 0.0
    o[0:2] = 0.0
    o[3:5] = [0.6, 0.8]
    env.last_obs_v2 = o
    assert env.teacher_action()[1] > 0.0


def test_standing_still_during_escort_is_never_rewarded():
    """V1 paid +0.15/step for keeping the patient close, so slow escorts earned more. V2 must not."""
    env = make_env()
    run_until_escort(env)
    for _ in range(30):
        _, r, done, _ = env.step(np.array([-1.0, 0.0], dtype=np.float32))  # v = 0
        assert r <= 1e-6
        if done:
            break


def test_moving_along_path_beats_waiting():
    env_a, env_b = make_env(), make_env()
    run_until_escort(env_a)
    run_until_escort(env_b)
    ra = sum(env_a.step(env_a.teacher_action())[1] for _ in range(40))
    rb = sum(env_b.step(np.array([-1.0, 0.0], dtype=np.float32))[1] for _ in range(40))
    assert ra > rb
    assert rb == pytest.approx(40 * R_STEP, abs=0.05)


def test_tracked_patient_is_masked_from_depth():
    env = make_env()
    run_until_escort(env)
    p = env.patients[env.active_patient_idx]
    fx, fz = -math.sin(env.robot_yaw), -math.cos(env.robot_yaw)
    p["x"], p["z"] = env.robot_x + 1.0 * fx, env.robot_z + 1.0 * fz  # put the patient dead ahead
    env.patient_x, env.patient_z = p["x"], p["z"]
    masked = env.depth_rays()[7]
    env.task_phase = TaskPhase.SEARCH_PATIENT  # no longer tracked -> must show up as an obstacle
    assert env.depth_rays()[7] <= min(masked, 1.0)


def test_v1_doctor_and_door_channels_removed():
    env = make_env()
    env.reset()
    assert env._get_obs().shape[0] == 21  # 29 -> 21: doctor (4) + door (4) channels dropped


def test_policy_server_v2_roundtrip():
    from rl_policy_server import PolicyRunner, depth_from_msg

    agent = PPOHospitalAgent(obs_dim=21, act_dim=2)
    runner = PolicyRunner(agent, lidar_source="godot")
    msg = {"type": "rl_state", "id": 1, "robot": [-8.0, 0.0, 0.0, 0.0, 0.0], "lidar": [1.0] * 17,
           "patient": [-4.0, 4.0], "phase": 0, "goal": [-4.0, 4.0], "task_id": 0,
           "depth": [0.4] * 3 + [4.0] * 12}
    out = runner.act(msg)
    assert out["type"] == "rl_action" and math.isfinite(out["v"]) and math.isfinite(out["omega"])
    assert out["sectors"][0] > 0.9 and out["sectors"][4] == 0.0
    no_depth = dict(msg)
    no_depth.pop("depth")
    assert depth_from_msg(no_depth).shape == (15,)
    with torch.no_grad():
        assert runner.act(no_depth)["type"] == "rl_action"
