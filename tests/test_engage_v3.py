"""Engagement V3 environment (docs/RL_V3_DESIGN.md): faces, head pitch, ask, callers, dialogue / enrolment."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.models.ppo_agent import PPOHospitalAgent
from src.training.hospital_env import TaskPhase
from src.training.hospital_env_engage_v3 import (
    ACT_DIM_V3, ASK_COOLDOWN_S, ENROLL_NEED_S, OBS_DIM_V3, R_ASK_EMPTY, HospitalEngageV3Env,
)
from src.training.hospital_env_pure_vision_v2 import OBS_DIM_V2
from src.training.train_engage_v3 import V2_CKPT, policy_mean, warm_start_from_v2


def make(seed: int = 1, dr: bool = False) -> HospitalEngageV3Env:
    env = HospitalEngageV3Env(curriculum_stage=1, shift_duration_s=120.0, dt=0.2, seed=seed, env_mode="solo",
                              stagnation_timeout_s=90.0, stagnation_penalty=0.0, escort_only=True,
                              domain_randomization=dr)
    env.reset(seed=seed)
    return env


def place_robot_facing(env: HospitalEngageV3Env, i: int, dist: float) -> None:
    """Robot `dist` m from patient i, looking at them; the patient faces the robot."""
    p = env.patients[i]
    env.robot_yaw = 0.0                       # forward = −Z
    env.robot_x, env.robot_z = p["x"], p["z"] + dist
    p["face_yaw"] = math.pi                   # forward = +Z (towards the robot)


def test_obs_and_action_dims() -> None:
    env = make()
    assert env.reset().shape == (OBS_DIM_V3,)
    obs, r, done, info = env.step(np.zeros(ACT_DIM_V3, dtype=np.float32))
    assert obs.shape == (OBS_DIM_V3,) and "v3" in info
    obs, *_ = env.step(np.zeros(2, dtype=np.float32))        # V2-style 2-D action still works (never asks)
    assert obs.shape == (OBS_DIM_V3,)


def test_face_seen_only_when_person_faces_robot() -> None:
    env = make()
    place_robot_facing(env, 0, 1.8)
    assert env._face_of(0) is not None
    env.patients[0]["face_yaw"] = 0.0         # looking away
    assert env._face_of(0) is None


def test_standing_person_very_close_needs_head_up() -> None:
    env = make()
    place_robot_facing(env, 0, 0.55)
    env.patients[0]["seated"] = False
    env.tilt = 0.0
    assert env._face_of(0) is None, "camera ngang không thấy mặt người đứng sát"
    env.tilt = math.radians(25.0)
    assert env._face_of(0) is not None, "ngẩng đầu thì thấy"


def test_ask_with_nobody_is_penalised() -> None:
    env = make()
    env.robot_x, env.robot_z, env.robot_yaw = 0.0, -1.3, math.pi / 2    # corridor, facing a wall
    for p in env.patients:
        p["x"] += 30.0
    r = env._ask()
    assert r == R_ASK_EMPTY and env.v3_stats["asks_empty"] == 1


def test_ask_starts_dialogue_and_enrolment_needs_face() -> None:
    env = make()
    i = 0
    env.patients[i].update(needs_help=True, caller=False)
    place_robot_facing(env, i, 1.8)
    env.patients[i]["face_yaw"] = 0.0         # facing away: still hears the question
    assert env._ask() == 0.0
    assert env.dialogue is not None and env.dialogue["idx"] == i
    assert env.since_ask_s == 0.0
    stand = np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32)
    for _ in range(int(ENROLL_NEED_S / env.dt) + 15):
        _, _, done, info = env.step(stand)
        if done:
            break
    assert info["v3"]["enrolled"] == 1, "người được hỏi quay lại ➔ robot đứng yên nhìn ➔ ghi nhớ được"


def test_ask_cooldown_counts_as_repeat() -> None:
    env = make()
    place_robot_facing(env, 0, 1.8)
    env.patients[0].update(needs_help=False, caller=False)
    env._ask()
    assert env.patients[0]["declined"]
    env.since_ask_s = ASK_COOLDOWN_S + 1.0
    env._ask()
    assert env.v3_stats["asks_repeat"] == 1, "hỏi lại người đã từ chối"


def test_dialogue_ends_with_escort_after_enrolment() -> None:
    env = make()
    env.patients[0].update(needs_help=True, caller=False)
    place_robot_facing(env, 0, 1.8)
    env._ask()
    stand = np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32)
    for _ in range(60):
        env.step(stand)
        if env.task_phase == TaskPhase.GUIDE_ESCORT:
            break
    assert env.task_phase == TaskPhase.GUIDE_ESCORT
    assert env.active_patient_idx == 0


def test_caller_calls_and_gives_up_when_ignored() -> None:
    env = make()
    p = env.patients[0]
    p.update(needs_help=True, caller=True)
    for q in env.patients[1:]:
        q["x"] += 30.0                                                      # nobody else nearby
    env.robot_x, env.robot_z, env.robot_yaw = p["x"], p["z"] + 3.0, math.pi   # 3 m away (beyond walk-up), back turned
    stand = np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32)
    heard = False
    for _ in range(150):
        obs, _, done, info = env.step(stand)
        heard |= obs[21] > 0.5
        if done:
            break
    assert heard, "robot nghe thấy tiếng gọi"
    assert p["gave_up"] and info["v3"]["callers_gave_up"] == 1


def test_voice_direction_points_to_caller() -> None:
    env = make()
    p = env.patients[0]
    p.update(needs_help=True, caller=True, call_t=1.0, calls_done=1)
    env.robot_x, env.robot_z, env.robot_yaw = p["x"] + 2.0, p["z"], 0.0      # caller on the robot's left
    obs = env._get_obs()
    assert obs[21] == 1.0
    assert math.atan2(obs[22], obs[23]) < -1.0, "giọng nói từ bên trái"


@pytest.mark.skipif(not V2_CKPT.exists(), reason="no V2 checkpoint")
def test_warm_start_reproduces_v2_driving() -> None:
    v2 = PPOHospitalAgent(obs_dim=OBS_DIM_V2, act_dim=2)
    v2.load_state_dict(torch.load(V2_CKPT, map_location="cpu", weights_only=True)["model_state_dict"])
    v3 = PPOHospitalAgent(obs_dim=OBS_DIM_V3, act_dim=ACT_DIM_V3)
    warm_start_from_v2(v3, V2_CKPT)
    o = torch.randn(5, OBS_DIM_V3)
    a3 = policy_mean(v3, o)
    a2 = policy_mean(v2, o[:, :OBS_DIM_V2])
    assert torch.allclose(a3[:, :2], a2, atol=1e-5), "lái giống hệt V2 lúc bắt đầu"
    assert torch.all(a3[:, 3] < -0.9), "chưa hỏi gì lúc bắt đầu"
    assert torch.allclose(v3.critic(o), v2.critic(o[:, :OBS_DIM_V2]), atol=1e-5)


def test_walkup_talker_starts_dialogue_from_behind_and_teacher_turns_to_enrol() -> None:
    """As in Godot (npc.gd): a person needing help talks once the robot stops within 2 m, even behind the robot."""
    from src.training.hospital_env_engage_v3 import WALKUP_RANGE_M
    env = make()
    p = env.patients[0]
    p.update(needs_help=True, caller=False)
    env.robot_yaw = 0.0                                   # forward = −Z; person behind (+Z)
    env.robot_x, env.robot_z = p["x"], p["z"] - (WALKUP_RANGE_M - 0.5)
    p["face_yaw"] = 0.0
    started = False
    info = {}
    for _ in range(200):
        _, _, done, info = env.step(env.teacher_action() if env.dialogue is not None else
                                    np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32))
        started |= env.dialogue is not None and env.dialogue.get("walkup", False)
        if done or env.task_phase == TaskPhase.GUIDE_ESCORT:
            break
    assert started, "người cần giúp tự nói khi robot đứng gần"
    assert info["v3"]["walkup_enrolled"] >= 1, "robot quay theo hướng giọng nói ➔ thấy mặt ➔ ghi nhớ"


def test_face_known_flag_and_thanks_after_delivery() -> None:
    env = make()
    place_robot_facing(env, 0, 1.8)
    assert env._get_obs()[31] == 0.0
    env.patients[0]["served"] = True
    assert env._get_obs()[31] == 1.0, "mặt người đã giúp ➔ cờ 'đã gặp'"


def test_widening_loader_keeps_old_policy(tmp_path) -> None:
    from src.training.train_engage_v3 import load_widening
    old = PPOHospitalAgent(obs_dim=31, act_dim=ACT_DIM_V3)
    p = tmp_path / "old.pt"
    torch.save({"model_state_dict": old.state_dict()}, p)
    new = PPOHospitalAgent(obs_dim=OBS_DIM_V3, act_dim=ACT_DIM_V3)
    load_widening(new, p)
    o = torch.randn(4, OBS_DIM_V3)
    assert torch.allclose(policy_mean(new, o), policy_mean(old, o[:, :31]), atol=1e-5)


def test_selection_score_prefers_walkup_enrolment_and_few_wasted_asks() -> None:
    from src.training.train_engage_v3 import selection_score
    a = {"return": 108.0, "walkup_enroll_rate": 0.87, "wasted_asks_per_ep": 0.4}
    b = {"return": 106.0, "walkup_enroll_rate": 1.0, "wasted_asks_per_ep": 0.5}
    assert selection_score(a) > selection_score(b), "mặc định: chỉ theo return như trước"
    assert selection_score(b, 40.0, 2.0) > selection_score(a, 40.0, 2.0)


def test_walkup_drill_puts_needy_person_next_to_robot() -> None:
    from src.training.train_engage_v3 import make_env
    for seed in range(5):
        env = make_env(seed, 30.0, dr=True)
        env.reset(seed=seed)
        assert env.setup_walkup_drill()
        near = [p for p in env.patients
                if p["needs_help"] and not p["caller"] and not p["seated"]
                and 0.75 <= math.hypot(p["x"] - env.robot_x, p["z"] - env.robot_z) <= 1.95]
        assert near, "có một người cần giúp đứng sát robot"
        assert env.last_obs_v2.shape == (OBS_DIM_V3,)


def test_walkup_probe_runs_and_scores_selection() -> None:
    from src.training.train_engage_v3 import selection_score, walkup_probe
    agent = PPOHospitalAgent(obs_dim=OBS_DIM_V3, act_dim=ACT_DIM_V3)
    pr = walkup_probe(agent, n=3)
    assert 0.0 <= pr["probe_walkup"] <= 1.0 and "probe_walkup_side" in pr
    ev = {"return": 100.0, "walkup_enroll_rate": 1.0, "wasted_asks_per_ep": 0.0, "probe_walkup": 0.5}
    assert selection_score(ev, 40.0) == 120.0, "có probe thì dùng probe, không dùng walkup_enroll_rate"


def test_voice_memory_follows_robot_rotation_after_voice_stops() -> None:
    from src.training.hospital_env_engage_v3 import VOICE_MEMORY_S
    env = make()
    p = env.patients[0]
    for q in env.patients[1:]:
        q["x"] += 30.0
    env.robot_x, env.robot_z, env.robot_yaw = p["x"] - 2.0, p["z"], 0.0   # person on the robot's right
    p["speak_t"] = 1.0
    o = env._get_obs()
    assert o[21] == 1.0 and o[34] == 0.0
    assert math.atan2(o[32], o[33]) == pytest.approx(math.atan2(o[22], o[23]), abs=1e-5)
    p["speak_t"] = 0.0
    before = math.atan2(o[32], o[33])
    env.robot_yaw -= math.radians(30.0)                  # robot turns right by 30° → person 30° less to the right
    env.step_count += 5
    o = env._get_obs()
    assert o[21] == 0.0, "đã ngừng nói"
    assert math.atan2(o[32], o[33]) == pytest.approx(before - math.radians(30.0), abs=0.05), "nhớ hướng, xoay theo robot"
    assert 0.0 < o[34] < 1.0
    assert (o[22], o[23]) == (0.0, 0.0), "im lặng ➔ ô DOA = 0 (vòng 8 thử giữ hướng, kém hơn)"
    env.step_count += int(VOICE_MEMORY_S / env.dt) + 1
    o = env._get_obs()
    assert (o[32], o[33], o[34]) == (0.0, 0.0, 1.0), "quên sau VOICE_MEMORY_S"
    assert (o[22], o[23]) == (0.0, 0.0)


# ---------- vòng 9: người chắn lối, xin đường, trí nhớ vật cản, tín hiệu gỡ kẹt ----------

def _full(seed: int = 5, dr: bool = False) -> HospitalEngageV3Env:
    env = HospitalEngageV3Env(curriculum_stage=1, shift_duration_s=40.0, dt=0.2, seed=seed, env_mode="full",
                              stagnation_timeout_s=90.0, stagnation_penalty=0.0, escort_only=True,
                              domain_randomization=dr)
    env.reset(seed=seed)
    return env


def test_exit_drill_places_person_between_robot_and_door() -> None:
    env = _full()
    assert env.setup_exit_drill()
    p = env.patients[0]
    assert p["served"] and not p["seated"] and env.active_patient_idx is None
    assert abs(env.robot_z) > 2.0, "robot ở trong phòng"
    assert abs(p["z"]) < abs(env.robot_z) + 0.6, "người đứng về phía cửa"
    assert env.last_obs_v2.shape == (OBS_DIM_V3,)


def test_delivered_person_blocks_and_is_seen_in_depth() -> None:
    env = _full()
    p = env.patients[0]
    env.robot_x, env.robot_z, env.robot_yaw = 100.0, -4.0, math.pi     # nhìn +Z (ra cửa phòng 302)
    p.update(x=100.0, z=-3.4, served=True, seated=False)
    env.returning_to_lobby = True
    o = env._get_obs()
    assert o[2] > 0.8, "người đứng ngay trước mặt hiện trong depth (không bị che)"
    assert o[42] == 1.0 and o[44] > 0.5, "và có box người to ở giữa khung"
    z0 = env.robot_z
    for _ in range(10):
        _, _, _, info = env.step(np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32))
    assert env.robot_z < z0 + 0.25, "không đi xuyên qua người"
    assert info["v3"]["push_steps"] > 0, "đẩy vào người bị tính (phạt)"
    assert env.last_obs_v2[41] != 0.0, "kẹt ➔ có tín hiệu gỡ kẹt"


def test_excuse_me_makes_blocking_person_step_aside() -> None:
    env = _full()
    p = env.patients[0]
    env.robot_x, env.robot_z, env.robot_yaw = 100.0, -4.0, math.pi
    p.update(x=100.0, z=-3.2, served=True, seated=False)
    env.returning_to_lobby = True                                       # đang bận quay về: "hỏi" = xin đường
    env.since_ask_s = 99.0
    _, r, _, info = env.step(np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert info["v3"]["excuse"] == 1 and info["v3"]["asks_empty"] == 0
    for _ in range(25):
        env.step(np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32))
    assert abs(p["x"] - 100.0) > 0.7, "người được xin đã bước sang bên"
    for _ in range(40):
        env.step(np.array([0.6, 0.0, 0.0, -1.0], dtype=np.float32))
    assert env.robot_z > -2.6, "robot đi tiếp ra cửa được"


def test_ask_when_busy_and_nobody_ahead_is_wasted() -> None:
    env = _full()
    for q in env.patients:
        q["x"] += 30.0
    env.returning_to_lobby = True
    env.since_ask_s = 99.0
    _, _, _, info = env.step(np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert info["v3"]["asks_empty"] == 1 and info["v3"]["excuse"] == 0


def test_obstacle_memory_keeps_person_after_turning_away() -> None:
    def right_memory(with_person: bool) -> float:
        env = _full()
        p = env.patients[0]
        env.robot_x, env.robot_z, env.robot_yaw = 100.0, -5.0, math.pi
        p.update(x=100.0 if with_person else 130.0, z=-4.1, served=True, seated=False)
        env.returning_to_lobby = True
        env.mem.reset()
        for _ in range(8):                                              # quay trái ~110°: người ra khỏi khung, ở bên phải
            env.step(np.array([-1.0, 1.0, 0.0, -1.0], dtype=np.float32))
        ang, _ = env._rel_angle(100.0, -4.1)
        assert abs(ang) > math.radians(50.0), "người đã ra ngoài khung nhìn"
        return float(env.last_obs_v2[38:41].max())

    assert right_memory(True) > right_memory(False) + 0.3, "người vừa ra khỏi khung vẫn còn trong trí nhớ phía bên phải"


def test_dialogue_goal_points_at_speaker() -> None:
    env = make()
    p = env.patients[0]
    p.update(needs_help=True)
    env.robot_x, env.robot_z, env.robot_yaw = p["x"] - 1.2, p["z"], 0.0   # người ở bên phải robot
    env._start_dialogue(0)
    o = env._get_obs()
    assert o[30] == 1.0 and o[13] == 0.0
    assert math.atan2(float(o[14]), float(o[15])) == pytest.approx(-math.pi / 2, abs=0.3), "đích = hướng giọng nói (âm = phải)"


def test_follower_behind_is_tracked_by_rear_camera() -> None:
    env = _full()
    p = env.patients[0]
    env.robot_x, env.robot_z, env.robot_yaw = 0.0, 0.0, -math.pi / 2     # đi về +X
    p.update(x=-1.2, z=0.0, seated=False, follows_behind=True)
    env.active_patient_idx, env.task_phase, env.goal_key = 0, TaskPhase.GUIDE_ESCORT, "112"
    env.patient_x, env.patient_z = p["x"], p["z"]
    o = env._get_obs()
    assert o[5] == 0.0, "camera trước không thấy người đi sau"
    assert o[19] == 1.0 and o[18] == pytest.approx(1.2 / 5.0, abs=0.08), "camera sau thấy ➔ tether vẫn biết họ ở gần"
    for _ in range(30):
        env.step(np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32))
    d = math.hypot(p["x"] - env.robot_x, p["z"] - env.robot_z)
    assert 0.6 < d < 2.0 and p["x"] < env.robot_x, "người vẫn theo sau lưng robot"


def test_recovery_cue_keeps_side_and_counts_down() -> None:
    from src.training.local_memory import RECOVERY_S, RecoveryCue
    c = RecoveryCue()
    assert c.update(0.5, 0.0, 0.2, True, 0.9, 0.1, 0.0) == 0.0
    v = c.update(0.5, 0.0, 0.2, True, 0.9, 0.1, 0.0)
    assert v == pytest.approx(-1.0), "kẹt 0.4 s, bên trái bị chắn ➔ quay phải"
    v2 = c.update(0.0, 0.0, 0.2, True, 0.1, 0.9, 1.0)
    assert -1.0 < v2 < 0.0, "giữ nguyên phía quay dù quan sát đổi, đếm lùi"
    for _ in range(int(RECOVERY_S / 0.2) + 1):
        v2 = c.update(0.0, 0.0, 0.2, True, 0.0, 0.0, 0.0)
    assert v2 == 0.0
    assert c.update(0.5, 0.0, 0.2, False, 0.9, 0.1, 0.0) == 0.0 and c.stall_s == 0.0, "đang nói chuyện ➔ không tính kẹt"
