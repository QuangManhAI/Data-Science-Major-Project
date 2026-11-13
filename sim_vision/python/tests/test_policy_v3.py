"""rl_policy_server với checkpoint V3 (31-D vào, 4-D ra): trả thêm tilt / ask (docs/RL_V3_DESIGN.md)."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.models.ppo_agent import PPOHospitalAgent  # noqa: E402
from src.training.hospital_env_engage_v3 import ACT_DIM_V3, OBS_DIM_V3  # noqa: E402
from src.training.train_engage_v3 import V2_CKPT, warm_start_from_v2  # noqa: E402

import rl_policy_server as rps  # noqa: E402


@pytest.fixture()
def v3_ckpt(tmp_path: Path) -> Path:
    agent = PPOHospitalAgent(obs_dim=OBS_DIM_V3, act_dim=ACT_DIM_V3)
    if V2_CKPT.exists():
        warm_start_from_v2(agent, V2_CKPT)
    p = tmp_path / "v3.pt"
    torch.save({"model_state_dict": agent.state_dict(), "episode": 0}, p)
    return p


def _msg(v3: dict | None = None) -> dict:
    m = {"type": "rl_state", "id": 1, "robot": [0.0, 2.0, 0.0, 0.0, 0.0], "lidar": [1.0] * 17,
         "patient": [0.0, 0.0], "phase": 0, "goal": [0.0, 0.0], "task_id": 0, "depth": [4.0] * 15, "dets": []}
    if v3 is not None:
        m["v3"] = v3
    return m


def test_v3_checkpoint_loads_with_4_actions(v3_ckpt: Path) -> None:
    agent = rps.load_agent(v3_ckpt)
    assert agent.obs_dim == OBS_DIM_V3 and agent.act_dim == ACT_DIM_V3


def test_v3_reply_has_tilt_and_ask(v3_ckpt: Path) -> None:
    runner = rps.PolicyRunner(rps.load_agent(v3_ckpt), "camera")
    reply = runner.act(_msg({"voice": [1.0, 0.5], "face": [1.0, 0.1, -0.1, 0.06], "tilt": 0.0,
                             "since_ask_s": 10.0, "dialogue": False}))
    assert -1.0 <= reply["tilt"] <= 1.0 and isinstance(reply["ask"], bool)
    assert len(reply["a"]) == 4
    reply2 = runner.act(_msg())                       # Godot cũ không gửi v3 ➔ vẫn chạy
    assert "tilt" in reply2


def test_v2_checkpoint_reply_unchanged() -> None:
    if not V2_CKPT.exists():
        pytest.skip("no V2 checkpoint")
    runner = rps.PolicyRunner(rps.load_agent(V2_CKPT), "camera")
    reply = runner.act(_msg({"voice": [1.0, 0.5]}))
    assert "tilt" not in reply and len(reply["a"]) == 2


def test_v3_features_face_known_flag() -> None:
    f = rps.v3_features({"face": [1.0, 0.0, 0.0, 0.1, 1.0]})
    assert f.shape == (14,) and f[10] == 1.0
    assert rps.v3_features({"face": [1.0, 0.0, 0.0, 0.1]})[10] == 0.0, "Godot cũ không gửi cờ ➔ 0"


def test_old_31d_v3_checkpoint_still_served(tmp_path: Path) -> None:
    agent = PPOHospitalAgent(obs_dim=31, act_dim=ACT_DIM_V3)
    p = tmp_path / "v3_31.pt"
    torch.save({"model_state_dict": agent.state_dict(), "episode": 0}, p)
    runner = rps.PolicyRunner(rps.load_agent(p), "camera")
    assert "tilt" in runner.act(_msg({"face": [1.0, 0.0, 0.0, 0.1, 1.0], "dialogue": False}))


def test_v3_features_voice_memory() -> None:
    import math
    f = rps.v3_features({"voice_mem": [1.0, 0.8, 2.0]})
    assert f[0] == 0.0, "không có tiếng lúc này"
    assert tuple(f[1:3]) == (0.0, 0.0), "im lặng ➔ ô DOA = 0, như lúc train"
    held = rps.v3_features({"voice_mem": [1.0, 0.8, 2.0]}, hold_doa=True)
    assert math.atan2(held[1], held[2]) == pytest.approx(0.8)
    assert math.atan2(f[11], f[12]) == pytest.approx(0.8) and f[13] == pytest.approx(2.0 / rps.VOICE_MEMORY_S)
    old = rps.v3_features({"voice_mem": [1.0, 0.8, rps.VOICE_MEMORY_S + 1.0]})
    assert tuple(old[11:]) == (0.0, 0.0, 1.0), "quá lâu ➔ quên"
    assert tuple(rps.v3_features({})[11:]) == (0.0, 0.0, 1.0), "Godot cũ không gửi ➔ như chưa nghe"


def test_old_32d_v3_checkpoint_still_served(tmp_path: Path) -> None:
    agent = PPOHospitalAgent(obs_dim=32, act_dim=ACT_DIM_V3)
    p = tmp_path / "v3_32.pt"
    torch.save({"model_state_dict": agent.state_dict(), "episode": 0}, p)
    runner = rps.PolicyRunner(rps.load_agent(p), "camera")
    assert "tilt" in runner.act(_msg({"voice_mem": [1.0, 0.3, 1.0], "dialogue": False}))


# ---------- vòng 9 (45-D): server dựng quan sát giống môi trường train ----------

def _env_msg(env, phase: int | None = None) -> dict:
    """Thông điệp Godot sẽ gửi, dựng từ trạng thái thật của môi trường train (depth không nhiễu)."""
    import math

    import numpy as np

    from src.training.camera_vision_model import compute_pinhole_detection
    from src.training.hospital_env import DESTINATION_POSITIONS, floor_of

    dets = []
    people = [(p["x"], p["z"], p["seated"]) for p in env.patients if floor_of(p["x"]) == floor_of(env.robot_x)]
    people += [(n["x"], n["z"], bool(n.get("seated", False))) for n in env.npcs if floor_of(n["x"]) == floor_of(env.robot_x)]
    for x, z, seated in people:
        if math.hypot(x - env.robot_x, z - env.robot_z) > 6.0:
            continue
        d = compute_pinhole_detection(robot_x=env.robot_x, robot_z=env.robot_z, robot_yaw=env.robot_yaw, target_x=x,
                                      target_z=z, target_seated=seated, is_blocked_fn=env._check_los_occluded)
        if d["seen"] > 0.5:
            cx = 0.5 + 0.5 * d["cx"]
            dets.append({"class_id": 0, "class_name": "patient", "conf": 0.9, "x1": cx - d["w"] / 2, "x2": cx + d["w"] / 2,
                         "y1": 0.5 - d["h"] / 2, "y2": 0.5 + d["h"] / 2})
    o = env.last_obs_v2
    v3 = {"tilt": float(o[28]), "since_ask_s": env.since_ask_s, "dialogue": env.dialogue is not None}
    if hasattr(env, "_person_items"):       # vòng 11: Godot gửi người trong khung kèm cờ "đã gặp"
        v3["persons"] = [[cx, h, 1.0 if known else 0.0] for cx, h, known in env._person_items()]
    if o[21] > 0.5:
        v3["voice"] = [1.0, math.atan2(float(o[22]), float(o[23]))]
    if o[34] < 1.0:
        v3["voice_mem"] = [1.0, math.atan2(float(o[32]), float(o[33])), float(o[34]) * rps.VOICE_MEMORY_S]
    gx, gz = DESTINATION_POSITIONS.get(env.goal_key, (env.robot_x, env.robot_z))
    return {"type": "rl_state", "id": 1, "robot": [env.robot_x, env.robot_z, env.robot_yaw, env.robot_v, env.robot_omega],
            "lidar": [1.0] * 17, "patient": [env.robot_x, env.robot_z], "phase": int(env.task_phase) if phase is None else phase,
            "goal": [gx, gz], "task_id": 0, "depth": [float(t) for t in env.depth_rays()], "dets": dets, "v3": v3}


def test_round9_server_obs_matches_training_env(v3_ckpt: Path) -> None:
    """Thoát khỏi phòng sau khi giao bệnh nhân: trí nhớ vật cản, cờ kẹt, box người, pha "quay về" phải giống lúc train."""
    import numpy as np

    from src.training.hospital_env import TaskPhase
    from src.training.hospital_env_engage_v3 import OBS_MEM0, OBS_PERSON0, OBS_STALL, HospitalEngageV3Env

    env = HospitalEngageV3Env(curriculum_stage=1, shift_duration_s=40.0, dt=0.2, seed=3, env_mode="full",
                              stagnation_timeout_s=90.0, stagnation_penalty=0.0, escort_only=True, domain_randomization=False)
    env.reset(seed=3)
    for _ in range(40):
        if env.setup_exit_drill() and env.returning_to_lobby:
            break
    assert env.returning_to_lobby, "cần một bài thoát phòng ở tầng 2 / 3"
    runner = rps.PolicyRunner(rps.load_agent(v3_ckpt), "camera")
    env.patients[0]["yield_t"] = env.patients[0]["leave_t"] = -1.0
    env._yield_spot = lambda p: None                             # người chắn lối không nhường ➔ chắc chắn có lúc robot kẹt
    seen_mem = seen_stall = seen_person = False
    done = False
    for k in range(120):
        if done:
            break
        # robot đang quay về: Godot gửi pha CALL_STAFF + đích = thang máy
        runner.act(_env_msg(env, phase=int(TaskPhase.CALL_STAFF)))
        got, want = runner.last_obs, env.last_obs_v2
        assert got.shape == want.shape == (OBS_DIM_V3,)
        assert np.allclose(got[0:5], want[0:5], atol=1e-4), f"depth step {k}"
        assert np.allclose(got[9:13], want[9:13]), f"pha step {k}: {got[9:13]} vs {want[9:13]}"
        assert np.allclose(got[13:16], want[13:16], atol=1e-3), f"đích step {k}: {got[13:16]} vs {want[13:16]}"
        assert np.allclose(got[OBS_MEM0:OBS_STALL], want[OBS_MEM0:OBS_STALL], atol=1e-4), f"trí nhớ vật cản step {k}"
        assert got[OBS_PERSON0] == want[OBS_PERSON0] and np.allclose(got[OBS_PERSON0:], want[OBS_PERSON0:], atol=1e-3), f"box người step {k}"
        seen_mem |= bool(want[OBS_MEM0:OBS_STALL].max() > 0.0)
        seen_person |= bool(want[OBS_PERSON0] > 0.5)
        if k > 0:
            assert got[OBS_STALL] == pytest.approx(float(want[OBS_STALL]), abs=1e-5), f"tín hiệu gỡ kẹt step {k}"
        seen_stall |= bool(want[OBS_STALL] != 0.0)
        a = env.teacher_action()
        runner.last_cmd_v = (float(a[0]) + 1.0) * 0.5 * 0.8      # lệnh thật sự được thực hiện (thầy lái, không phải policy)
        _, _, done, _ = env.step(a)
    assert seen_person, "bài tập phải có lúc thấy box người chắn lối"
    assert seen_mem, "bài tập phải có lúc vật cản ra khỏi khung nhìn nhưng còn trong trí nhớ"
    assert seen_stall, "bài tập phải có lúc robot kẹt (tín hiệu gỡ kẹt khác 0)"


def test_round9_patient_box_centre_scaled_like_training(v3_ckpt: Path) -> None:
    runner = rps.PolicyRunner(rps.load_agent(v3_ckpt), "camera")
    m = _msg({"dialogue": False})
    m["dets"] = [{"class_id": 0, "class_name": "patient", "conf": 0.9, "x1": 0.7, "x2": 0.9, "y1": 0.2, "y2": 0.8}]
    runner.act(m)
    assert runner.last_obs[6] == pytest.approx(0.6), "tâm box 0.8 ➔ lệch 0.3 khung ➔ 0.6 theo thang -1..1 lúc train"
    assert runner.last_obs[42] == 1.0 and runner.last_obs[43] == pytest.approx(0.3) and runner.last_obs[44] == pytest.approx(0.6)


def test_round9_dialogue_goal_is_voice_direction(v3_ckpt: Path) -> None:
    import math
    runner = rps.PolicyRunner(rps.load_agent(v3_ckpt), "camera")
    runner.act(_msg({"dialogue": True, "voice": [1.0, 1.0], "voice_mem": [1.0, 1.0, 0.0]}))
    o = runner.last_obs
    assert o[13] == 0.0 and math.atan2(float(o[14]), float(o[15])) == pytest.approx(-1.0), "người nói bên phải ➔ góc đích âm (quay phải)"


def test_round8_35d_checkpoint_still_served(tmp_path: Path) -> None:
    agent = PPOHospitalAgent(obs_dim=35, act_dim=ACT_DIM_V3)
    p = tmp_path / "v3_35.pt"
    torch.save({"model_state_dict": agent.state_dict(), "episode": 0}, p)
    runner = rps.PolicyRunner(rps.load_agent(p), "camera")
    assert "tilt" in runner.act(_msg({"face": [1.0, 0.0, 0.0, 0.1, 1.0], "dialogue": False}))
    assert runner.last_obs.shape == (35,)
