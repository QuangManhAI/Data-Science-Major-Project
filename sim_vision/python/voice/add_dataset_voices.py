"""Thêm giọng người lớn thật từ bộ thu âm Major_BE_FE vào clip của sim: mỗi NPC một giọng riêng, đúng giới tính.

Nguồn: data/processed/voice_dataset/manifest.csv (scripts/prepare_voice_dataset.py, đã chuẩn hóa 16 kHz, −23 dBFS).
Chọn: không phải trẻ em (age ≠ 10), usable, take r / n, người nói có đủ câu dẫn đường + cảm ơn.
Mỗi người nói thành một "voice": nữ B01, B02, … (speaker "B"), nam M01, M02, … (speaker "A", cạnh giọng "A" gốc).
Godot (Npc._ensure_voice) cấp cho mỗi NPC một voice chưa ai dùng, giữ cố định suốt các lượt.

Ghi vào data/processed/manifest.csv + data/processed/wav/ (STT server đọc), rồi chạy `python -m voice.sync_voices`
để copy sang Godot. Chạy lại an toàn: xóa các dòng source=dataset cũ rồi thêm lại. Cũng tính cột escort_ok cho
mọi clip (câu dẫn đường có đích mà policy G20 dẫn được, không phải câu hỏi triệu chứng kiểu "nên khám khoa nào").

Chạy: python -m voice.add_dataset_voices
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from nlu import policy, slots
from nlu.nlu import IntentResult
from voice import config

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DATASET_DIR: Path = REPO_ROOT / "data" / "processed" / "voice_dataset"
ESCORT_INTENTS: tuple[str, ...] = ("hoi_phong", "hoi_khoa", "dan_duong")
MIN_ESCORT_CLIPS: int = 10
EXTRA_COLS: tuple[str, ...] = ("voice", "escort_ok", "source")


def escort_ok(intent: str, text: str) -> bool:
    """Câu dẫn đường mà policy thật sự dẫn được (có khoa / phòng trong bản đồ)."""
    if intent not in ESCORT_INTENTS:
        return False
    return policy.decide(IntentResult(intent, 1.0, slots.extract(text), "rule", []), 0, None).action == "guide"


def select_speakers(rows: list[dict[str, str]]) -> list[str]:
    """Người nói nữ người lớn có ≥ MIN_ESCORT_CLIPS câu dẫn đường và ≥ 1 câu cảm ơn."""
    stats: dict[str, list[int]] = {}
    for r in rows:
        s = stats.setdefault(r["speaker_id"], [0, 0])
        s[0] += r["intent"] in ESCORT_INTENTS
        s[1] += r["intent"] == "cam_on"
    return sorted(spk for spk, (esc, thanks) in stats.items() if esc >= MIN_ESCORT_CLIPS and thanks >= 1)


# giới tính trong bộ dữ liệu ➔ (speaker của sim, gender, tiền tố voice)
GROUPS: dict[str, tuple[str, str, str]] = {"Nữ": ("B", "female", "B"), "Nam": ("A", "male", "M")}


def load_candidates(dataset_dir: Path, gender: str) -> list[dict[str, str]]:
    with (dataset_dir / "manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["gender"] == gender and r["age"] != "10" and r["usable"] == "True"
            and r["mode"] in ("r", "n")]


def build_rows(cands: list[dict[str, str]], speakers: list[str], gender: str) -> list[dict[str, str]]:
    sim_spk, sim_gender, prefix = GROUPS[gender]
    voice_of = {spk: f"{prefix}{i + 1:02d}" for i, spk in enumerate(speakers)}
    out = []
    for r in cands:
        if r["speaker_id"] not in voice_of:
            continue
        sid = r["code"].replace("_", "")
        vid = voice_of[r["speaker_id"]]
        out.append({
            "clip_id": f"{vid}_{sid}_{r['mode']}", "speaker": sim_spk, "gender": sim_gender, "sentence_id": sid,
            "take": r["mode"], "text_ref": r["text"], "intent": r["intent"], "ref_verified": str(r["mode"] == "r"),
            "duration_s": f"{float(r['duration_s']):.3f}", "rms_dbfs": f"{float(r['speech_rms_dbfs']):.2f}",
            "clipping": "False", "src_path": f"voice_dataset/{r['audio_path']}",
            "voice": vid, "escort_ok": str(escort_ok(r["intent"], r["text"])), "source": "dataset",
        })
    return sorted(out, key=lambda r: r["clip_id"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    man_path = config.manifest_path(args.data_dir)
    wav_dir = config.wav_dir(args.data_dir)
    with man_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    old = [r for r in rows if r.get("source") == "dataset"]
    for r in old:
        (wav_dir / f"{r['clip_id']}.wav").unlink(missing_ok=True)
    base = [r for r in rows if r.get("source") != "dataset"]
    for r in base:  # clip gốc (giọng A): voice = speaker, cờ dẫn đường theo cùng tiêu chí
        r.setdefault("voice", r["speaker"])
        r["voice"] = r["voice"] or r["speaker"]
        r["escort_ok"] = str(escort_ok(r["intent"], r["text_ref"]))
        r["source"] = r.get("source") or "real_voice"

    new: list[dict[str, str]] = []
    summary: list[str] = []
    for gender in GROUPS:
        cands = load_candidates(args.dataset_dir, gender)
        speakers = select_speakers(cands)
        rows_g = build_rows(cands, speakers, gender)
        new += rows_g
        summary.append(f"{len(speakers)} giọng {gender.lower()} ({len(rows_g)} clip)")
    for r in new:
        shutil.copyfile(args.dataset_dir / r["src_path"].removeprefix("voice_dataset/"), wav_dir / f"{r['clip_id']}.wav")

    for c in EXTRA_COLS:
        if c not in cols:
            cols.append(c)
    with man_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(base + new)
    n_esc = sum(r["escort_ok"] == "True" for r in new)
    print(f"{', '.join(summary)} ➔ {len(new)} clip ({n_esc} câu dẫn đường được), bỏ {len(old)} clip cũ ➔ {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
