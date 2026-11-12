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
