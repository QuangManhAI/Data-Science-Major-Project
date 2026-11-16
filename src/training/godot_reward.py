"""Phần thưởng khi train policy V3 ngay trong Godot (src/training/train_godot_v3.py).

Godot (sim_vision/godot/scripts/train_env.gd) gửi kèm mỗi bước policy:
  train = {"ep", "t", "done"?, "bump": số physics frame cọ tường / đồ đạc, "ev": {sự kiện: số lần},
           "gt": trạng thái thật (robot, tầng, pha, người cần giúp, đích, bệnh nhân đang dẫn, người chắn lối...)}
Policy không thấy phần này; nó chỉ dùng để tính thưởng. Hệ số giữ như HospitalEngageV3Env (env Python) để
policy 11c (train trong env Python) tinh chỉnh tiếp mà không phải học lại thang thưởng.

Khác env Python: mọi sự kiện là của Godot thật (hội thoại STT/NLU thật, NPC thật, va chạm vật lý thật), và
"chủ động đi tìm" được thưởng trực tiếp: shaping theo đường đi tới người cần giúp gần nhất + thưởng lần đầu
thấy một người cần giúp trong khung camera.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from src.training.hospital_env import ELEVATOR_POSITIONS, floor_of, nav_grid

R_STEP = -0.005
K_PATH = 0.20                 # shaping / m đường đi: đích dẫn, đường về sảnh
K_PATH_SEEK = 0.08            # shaping / m tới người cần giúp gần nhất (thật) khi rảnh
R_SPOTTED = 0.5               # lần đầu một người cần giúp lọt vào khung camera trước (box YOLO của đúng người đó)
R_TASK_STARTED = 2.0          # hội thoại xong, nhận việc dẫn / gọi nhân viên
R_ANSWERED = 2.0
R_DELIVERED = 10.0
R_STAFF_CALLED = 10.0
R_LOBBY_RETURN = 0.5
R_COLLISION = -1.0
R_CONTACT = -0.1
R_NEAR = -0.05
NEAR_M = 0.50
ROBOT_RADIUS_M = 0.30
R_TETHER = -0.05              # / m quá TETHER_FREE_M, khi bệnh nhân đang đi theo
TETHER_FREE_M = 2.0
TETHER_SHAPING_CUTOFF_M = 3.0
R_LEAVE_BEHIND = -0.15        # bệnh nhân đang dừng mà robot vẫn đi xa họ
LEAVE_BEHIND_M = 2.5
R_PATIENT_WAITING = -0.03
R_REENGAGE = 3.0
R_ASK_WASTED = -1.5
R_EXCUSE = -0.05
R_DIALOGUE_ABORT = -1.0
R_MOVE_DIALOGUE = -0.05
R_PERSON_PUSH = -0.2
PUSH_M = 0.75
R_STAGNATION = -2.0
R_GIVEN_UP = -3.0
R_ARRIVED_TIMEOUT = -0.5

EVENT_R = {
    "spotted": R_SPOTTED, "task_started": R_TASK_STARTED, "answered": R_ANSWERED, "delivered": R_DELIVERED,
    "staff_called": R_STAFF_CALLED, "lobby_return": R_LOBBY_RETURN, "reengaged": R_REENGAGE,
    "ask_wasted": R_ASK_WASTED, "excuse": R_EXCUSE, "dialogue_abort": R_DIALOGUE_ABORT,
    "stagnation": R_STAGNATION, "escort_given_up": R_GIVEN_UP, "arrived_timeout": R_ARRIVED_TIMEOUT,
}

_FIELDS: OrderedDict = OrderedDict()


def path_dist(rx: float, rz: float, tx: float, tz: float) -> float:
    """Đường đi ngắn nhất (m) trên lưới tầng robot; đích ở tầng khác ➔ tới thang máy tầng này."""
    fl = floor_of(rx)
    if floor_of(tx) != fl:
        tx, tz = ELEVATOR_POSITIONS[fl]
    key = (fl, round(tx * 4) / 4, round(tz * 4) / 4)       # lưới 0.25 m: NPC đi chậm ➔ dùng lại field
    fld = _FIELDS.get(key)
    grid = nav_grid(fl)
    if fld is None:
        fld = grid.field(key[1], key[2])
        _FIELDS[key] = fld
        if len(_FIELDS) > 512:
            _FIELDS.popitem(last=False)
    else:
        _FIELDS.move_to_end(key)
    return float(min(grid.distance(fld, rx, rz), 60.0))


class GodotReward:
    """Tính thưởng cho chuyển trạng thái (train của bước t ➔ train của bước t+1) của một ca Godot."""

    def __init__(self) -> None:
        self.prev_gt: dict | None = None
        self.stats: dict[str, float] = {}
        self.ret = 0.0

    def _potential(self, gt: dict) -> tuple[tuple, float, float] | None:
        """(khoá đích, khoảng cách đường đi, hệ số) — khoá đổi thì bước đó không có shaping."""
        rx, rz = float(gt["robot"][0]), float(gt["robot"][1])
        if gt.get("dialogue"):
            return None
        if gt.get("busy") and "goal" in gt:
            g = gt["goal"]
            return ("goal", str(g[2]), floor_of(rx)), path_dist(rx, rz, float(g[0]), float(g[1])), K_PATH
        needy = gt.get("needy") or []
        if not needy:
            return None
        cand = sorted(needy, key=lambda p: math.hypot(p[1] - rx, p[2] - rz))[:3]
        best = min(cand, key=lambda p: path_dist(rx, rz, p[1], p[2]))
        return ("seek", int(best[0])), path_dist(rx, rz, best[1], best[2]), K_PATH_SEEK

    def step(self, train: dict, depth: list | None) -> float:
        gt = train.get("gt") or {}
        ev = train.get("ev") or {}
        r = R_STEP
        for k, n in ev.items():
            self.stats[k] = self.stats.get(k, 0.0) + float(n)
            r += EVENT_R.get(k, 0.0) * float(n)

        pg = self.prev_gt
        if pg is not None and gt:
            p0, p1 = self._potential(pg), self._potential(gt)
            if p0 and p1 and p0[0] == p1[0] and abs(float(gt["robot"][0]) - float(pg["robot"][0])) < 5.0:
                shaping = p0[2] * (p0[1] - p1[1])
                esc = gt.get("escort")
                if p1[0][0] == "goal" and esc is not None and float(esc[2]) > TETHER_SHAPING_CUTOFF_M:
                    shaping = min(shaping, 0.0)
                r += float(np.clip(shaping, -1.0, 1.0))

        # an toàn
        bump = int(train.get("bump", 0))
        if bump > 0:
            if self.stats.get("_in_contact", 0.0) == 0.0:
                r += R_COLLISION
                self.stats["collisions"] = self.stats.get("collisions", 0.0) + 1
            else:
                r += R_CONTACT
            self.stats["_in_contact"] = 1.0
        else:
            self.stats["_in_contact"] = 0.0
        if depth is not None and len(depth) >= 12:
            dmin = float(np.min(np.asarray(depth, dtype=np.float32)[3:12]))
            if dmin < NEAR_M:
                r += R_NEAR * float(np.clip((NEAR_M - dmin) / (NEAR_M - ROBOT_RADIUS_M), 0.0, 1.0))
        v = float(gt.get("v", 0.0))
        if gt.get("blocker_m") is not None and float(gt["blocker_m"]) < PUSH_M and v > 0.1:
            r += R_PERSON_PUSH
            self.stats["push_steps"] = self.stats.get("push_steps", 0.0) + 1
        if gt.get("dialogue") and v > 0.05:
            r += R_MOVE_DIALOGUE

        # dẫn đường
        esc = gt.get("escort")
        if esc is not None:
            d, hold = float(esc[2]), float(esc[3])
            if hold > 0.0:
                r += R_PATIENT_WAITING
                if pg is not None and pg.get("escort") is not None and d > LEAVE_BEHIND_M and d > float(pg["escort"][2]) + 0.02:
                    r += R_LEAVE_BEHIND
                    self.stats["leave_steps"] = self.stats.get("leave_steps", 0.0) + 1
            else:
                r += R_TETHER * float(np.clip(d - TETHER_FREE_M, 0.0, 3.0))

        self.prev_gt = gt if gt else self.prev_gt
        r = float(np.clip(r, -20.0, 20.0)) if math.isfinite(r) else 0.0
        self.ret += r
        return r

    def summary(self) -> dict[str, float]:
        s = {k: v for k, v in self.stats.items() if not k.startswith("_")}
        s["return"] = self.ret
        s["helped"] = s.get("delivered", 0.0) + s.get("answered", 0.0) + s.get("staff_called", 0.0)
        return s


def episode_score(s: dict[str, float]) -> float:
    """Điểm chọn checkpoint theo Godot (cao = tốt): giúp được người, ít hỏi thừa / va chạm / bỏ dở."""
    return (10.0 * s.get("helped", 0.0) + 1.0 * s.get("task_started", 0.0) + 3.0 * s.get("reengaged", 0.0)
            - 2.0 * s.get("ask_wasted", 0.0) - 1.0 * s.get("collisions", 0.0) - 3.0 * s.get("escort_given_up", 0.0)
            - 2.0 * s.get("stagnation", 0.0) - 1.0 * s.get("dialogue_abort", 0.0) - 0.05 * s.get("push_steps", 0.0))
