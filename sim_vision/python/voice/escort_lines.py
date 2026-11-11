"""Câu robot tự nói khi dẫn đường (docs/UPGRADE_HRI_PLAN.md mục B).

Nguồn duy nhất: sim_vision/godot/data/escort_lines.json (Godot đọc trong robot_announcer.gd). Module này
mở rộng mọi biến thể {place} / {floor} để tts.pregen sinh sẵn giọng, giống all_responses() của NLU.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GODOT_DIR: Path = Path(__file__).resolve().parents[2] / "godot"
LINES_PATH: Path = GODOT_DIR / "data" / "escort_lines.json"
# Mã phòng số có thể làm đích: lấy từ chính code Godot để không lệch nhau.
_ROOM_SOURCES: tuple[Path, ...] = (GODOT_DIR / "scripts" / "room.gd", GODOT_DIR / "scripts" / "rl_env_bridge.gd")


def load_lines(path: Path = LINES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def room_numbers(sources: tuple[Path, ...] = _ROOM_SOURCES) -> list[str]:
    nums: set[str] = set()
    for p in sources:
        nums.update(re.findall(r'"(\d{3})"', p.read_text(encoding="utf-8")))
    return sorted(nums)


def place_name(key: str, places: dict[str, str]) -> str:
    """Giống RobotAnnouncer.place_name_of: số ➔ "Phòng 205", mã khác ➔ bảng places."""
    return f"Phòng {key}" if key.isdigit() else places.get(key, "")


def escort_texts(data: dict | None = None, rooms: list[str] | None = None) -> list[str]:
    d = load_lines() if data is None else data
    lines: dict[str, str] = d["lines"]
    names = {place_name(k, d["places"]) for k in d["places"]}
    names |= {place_name(n, {}) for n in (room_numbers() if rooms is None else rooms)}
    floors = [str(f) for f in d.get("floors", [1, 2, 3])]
    out: set[str] = set()
    for text in lines.values():
        places = sorted(names) if "{place}" in text else [None]
        for place in places:
            for fl in floors if "{floor}" in text else [None]:
                t = text
                if place is not None:
                    t = t.replace("{place}", place)
                if fl is not None:
                    t = t.replace("{floor}", fl)
                out.add(t)
    return sorted(out)
