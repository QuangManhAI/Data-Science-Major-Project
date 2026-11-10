"""Build the STT fine-tuning dataset from every real-voice recording we hold.

Sources
  * voice_finetune/            raw recordings (Lê Hà zips + Phương Uyên, Hoàng Tuấn,
                               Female 30, Male 30 folders from the HF dataset)
  * data/raw/minio_app_voice/  recordings uploaded by the web recorder app
                               (copied from MinIO major-be-fe/voice/)

Output  data/processed/voice_dataset/
  wav/<speaker_id>/<utt_id>.wav   16 kHz, mono, PCM_16 (Whisper / PhoASR input spec)
  manifest.csv                    one row per recording: speaker metadata, transcript,
                                  intent, audio-quality metrics, exclusion flags, split
  train / val / test / test_newtext .jsonl   {"audio_filepath", "text", "duration", ...}

Transcripts come from the recorder app's prompt list (configs/voice_scripts.json,
exported from Major_BE_FE/backend/src/voice/voice.scripts.ts). They are the *script*
text: exact for `_r` (read) takes, approximate for `_n` (natural) takes.

Audio is loudness-normalised (speech RMS -23 dBFS, peak <= -1 dBFS); the quality
columns in the manifest are measured before that gain so they describe the recording.

Usage: python scripts/prepare_voice_dataset.py [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RAW = ROOT / "voice_finetune"
APP_RAW = ROOT / "data" / "raw" / "minio_app_voice"
OUT = ROOT / "data" / "processed" / "voice_dataset"
SCRIPTS = json.loads((ROOT / "configs" / "voice_scripts.json").read_text(encoding="utf8"))

SR = 16000
AUDIO_EXT = {".wav", ".m4a", ".aac", ".webm", ".mp3", ".ogg"}
MIN_DUR, MAX_DUR = 0.5, 30.0  # Whisper encoder window is 30 s
TARGET_SPEECH_RMS, PEAK_LIMIT = -23.0, -1.0  # dBFS, loudness normalisation

# Two-way held-out design (see docs/VOICE_DATASET_EDA.md §11).
# 1) Speakers. Test = 6 complete speakers, 3 male / 3 female, one per age x gender
#    cell (child, 20, 30), every region we have (Bắc, Trung, Nam), every source
#    (Lê Hà, Phương Uyên, web app), both codecs and both a noisy and clean voice.
#    Speakers with heavy end-truncation (LH_F30_04, PU_F20_A) are kept out of test
#    so label errors do not inflate WER. Val = one speaker per adult cell.
TEST_SPEAKERS = {"PU_M10", "LH_A3", "LH_B2", "APP_1603b1", "LH_M30", "LH_F30_01"}
VAL_SPEAKERS = {"LH_A", "LH_B7", "LH_F30_03"}
# 2) Sentences. 16 of 110 prompts (~15% per script) never appear in train. Each is
#    taken from an intent that keeps >=1 other prompt in train, so every intent is
#    still trained. Their recordings by train speakers form `test_newtext`.
HELDOUT_CODES = {
    "C02", "C04",                          # hoi_khoa, hoi_phong
    "A_06", "A_08", "A_14", "A_18",        # hoi_khoa, hoi_phong, thanh_toan, tien_ich
    "B_02", "B_07", "B_15", "B_29",        # dang_ky_kham, hoi_phong, tien_ich, nhac_lai
    "T08", "T10", "T11", "T17",            # dang_ky_kham, hoi_phong, hoi_khoa, tien_ich
    "K05", "K06",                          # gap_nhan_vien, tien_ich
}
SPLITS = ("train", "val", "test", "test_newtext")

LOOKUP = {
    key: {item["code"]: item for item in items} for key, items in SCRIPTS["scripts"].items()
}


# --------------------------------------------------------------------------- speakers
def speaker_for(path: Path) -> dict:
    """Map a raw file to its speaker record (id, collector, demographics, script)."""
    if path.is_relative_to(APP_RAW):
        group, session = path.relative_to(APP_RAW).parts[:2]
        subj = SCRIPTS["subjects"][group]
        return dict(
            speaker_id=f"APP_{session.split('_')[-1]}", collector="web_app", source="minio_app",
            group=group, session=session, gender=subj["gender"], age=subj["age"],
            region="", province="", script=subj["script"],
        )

    rel = path.relative_to(LOCAL_RAW).parts
    base = dict(collector=rel[0], source="local", session="", region="", province="")
    if rel[0] == "Lê Hà":
        folder = rel[2]
        meta = json.loads((LOCAL_RAW / Path(*rel[:3]) / "metadata.json").read_text(encoding="utf8"))
        s = meta["subject"]
        folder = "M30" if "M30" in folder else folder
        script = {"A": "A", "B": "B", "M30": "T", "F30": "T"}[s["id"]]
        return dict(
            base, speaker_id=f"LH_{folder}", group=s["id"], gender=s["gender"],
            age=int(s["realAge"]), region=s.get("region", ""), province=s.get("province", ""),
            script=script,
        )
    if rel[0] == "Female 30":
        return dict(base, collector="unknown", speaker_id="X_F30", group="F30", gender="Nữ", age=30, script="T")
    if rel[0] == "Male 30":
        return dict(base, collector="unknown", speaker_id="X_M30", group="M30", gender="Nam", age=30, script="T")
    if rel[0] == "Hoàng Tuấn":
        male = rel[1].startswith("Nam")
        return dict(base, speaker_id="HT_M20" if male else "HT_F20", group="A" if male else "B",
                    gender="Nam" if male else "Nữ", age=20, script="A" if male else "B")
    if rel[0] == "Phương Uyên":
        if rel[1] == "Trẻ con":
            male = rel[2].startswith("Nam")
            return dict(base, speaker_id="PU_M10" if male else "PU_F10", group="M10" if male else "F10",
                        gender="Nam" if male else "Nữ", age=10, script="K")
        # Nữ 20 folder holds two prompt sets (A_ and B_); treated as two speakers
        # until confirmed otherwise (see EDA §3 pitch check).
        setname = "A" if path.name.startswith("A_") else "B"
        return dict(base, speaker_id=f"PU_F20_{setname}", group="B", gender="Nữ", age=20, script=setname)
    raise ValueError(f"unmapped path {path}")


CODE_RE = re.compile(r"(?<![A-Z0-9])([ABTKNG])_?(\d{2})(?!\d)|(?<![A-Z0-9])(C)(\d{2})(?!\d)")


def prompt_for(path: Path, spk: dict) -> dict:
    stem = re.sub(r"\s*\(\d+\)$", "", path.stem)
    note = ""
    if "wrong-order" in stem:
        note = "filename marked wrong-order"
    mode_m = re.search(r"_([rn])(?:_[a-z-]+)?$", stem)
    mode = mode_m.group(1) if mode_m else "u"  # u = single take, mode not labelled
    matches = list(CODE_RE.finditer(stem))
    if not matches:
        return dict(code="", mode=mode, text="", intent="", prompt_note=note or "no prompt code in filename")
    m = matches[-1]
    if m.group(3):
        key, code = "COMMON", f"C{m.group(4)}"
    else:
        letter, num = m.group(1), m.group(2)
        key = spk["script"]
        if key == "K":  # children's files are named A_01..A_15 but read script K
            code = f"K{num}"
        elif key in ("A", "B"):
            code = f"{key}_{num}"
        else:
            code = f"{key}{num}"
        if key in ("A", "B") and letter != key:
            note = (note + "; " if note else "") + f"prefix {letter} vs script {key}"
    item = LOOKUP.get(key, {}).get(code)
    if not item:
        return dict(code=code, mode=mode, text="", intent="", prompt_note="code not in script")
    return dict(code=code, mode=mode, text=item["text"], intent=item["intent"], prompt_note=note)


# --------------------------------------------------------------------------- audio
def probe(path: Path) -> tuple[str, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip().split(",")
    return (out[0], int(out[1])) if len(out) >= 2 and out[1].isdigit() else ("unknown", 0)


def convert(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR),
         "-sample_fmt", "s16", "-map_metadata", "-1", str(dst)],
        check=True,
    )


def f0_track(x: np.ndarray, frames: np.ndarray) -> float:
    """Median F0 (Hz) over loud frames via normalised autocorrelation, 70-500 Hz."""
    win = int(0.04 * SR)
    lo, hi = SR // 500, SR // 70
    vals = []
    for start in frames[:: max(1, len(frames) // 60)]:
        seg = x[start : start + win]
        if len(seg) < win:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[win - 1 :]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.5:
            vals.append(SR / lag)
    return float(np.median(vals)) if vals else float("nan")


def measure(path: Path) -> dict:
    sr, x = wavfile.read(path)
    x = x.astype(np.float32) / 32768.0
    dur = len(x) / sr
    if len(x) < int(0.05 * sr):
        return dict(duration_s=round(dur, 3), rms_dbfs=-120, speech_rms_dbfs=-120, peak_dbfs=-120,
                    clip_ratio=0, snr_db=0, lead_sil_s=0, trail_sil_s=0, speech_s=0,
                    f0_hz=float("nan"), cutoff_hz=0)
    hop, win = int(0.010 * sr), int(0.025 * sr)
    n = 1 + max(0, (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx]
    db = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-10)
    p10, p95 = np.percentile(db, 10), np.percentile(db, 95)
    speech = db > max(p95 - 25, p10 + 6)
    on = np.flatnonzero(speech)
    first, last = (on[0], on[-1]) if len(on) else (0, n - 1)
    # effective bandwidth: highest 20 Hz band whose mean power is within 60 dB of the peak band
    spec = np.abs(np.fft.rfft(x[: sr * 30])) ** 2
    k = len(spec) // 400
    bands = 10 * np.log10(spec[: k * 400].reshape(400, -1).mean(axis=1) + 1e-12)
    cutoff = float((np.flatnonzero(bands > bands.max() - 60).max() + 1) * (sr / 2) / 400)
    return dict(
        duration_s=round(dur, 3),
        rms_dbfs=round(float(10 * np.log10(np.mean(x**2) + 1e-10)), 2),
        speech_rms_dbfs=round(float(10 * np.log10(np.mean(fr[speech] ** 2) + 1e-10)) if speech.any() else -120.0, 2),
        peak_dbfs=round(float(20 * np.log10(np.max(np.abs(x)) + 1e-10)), 2),
        clip_ratio=round(float(np.mean(np.abs(x) >= 0.999)), 5),
        snr_db=round(float(p95 - p10), 2),
        lead_sil_s=round(first * hop / sr, 3),
        trail_sil_s=round(max(0.0, dur - (last * hop + win) / sr), 3),
        speech_s=round((last - first) * hop / sr + win / sr, 3),
        f0_hz=round(f0_track(x, idx[speech, 0]), 1),
        cutoff_hz=round(cutoff),
    )


# --------------------------------------------------------------------------- main
def normalise(path: Path, m: dict) -> float:
    """Scale speech to TARGET_SPEECH_RMS without letting the peak exceed PEAK_LIMIT."""
    if m["peak_dbfs"] < -45:  # no signal: leave as is, it is excluded anyway
        return 0.0
    gain = min(TARGET_SPEECH_RMS - m["speech_rms_dbfs"], PEAK_LIMIT - m["peak_dbfs"])
    sr, x = wavfile.read(path)
    y = np.clip(np.round(x.astype(np.float64) * 10 ** (gain / 20)), -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, y)
    return round(float(gain), 2)


def build_row(src: Path) -> dict:
    spk = speaker_for(src)
    pr = prompt_for(src, spk)
    codec, src_sr = probe(src)
    tag = pr["code"] + (f"_{pr['mode']}" if pr["mode"] != "u" else "") if pr["code"] else re.sub(r"\W+", "_", src.stem)
    utt = f"{spk['speaker_id']}__{tag}"
    dst = OUT / "wav" / spk["speaker_id"] / f"{utt}.wav"
    convert(src, dst)
    m = measure(dst)  # metrics describe the recording as captured (before gain)
    return dict(utt_id=utt, audio_path=str(dst.relative_to(OUT)), raw_path=str(src.relative_to(ROOT)),
                src_codec=codec, src_sr=src_sr, **spk, **pr, **m, gain_db=normalise(dst, m))


def exclusion(r: dict, dup_utts: set) -> str:
    reasons = []
    if r["speaker_id"] == "LH_F30_05":
        reasons.append("withdrawn_subject")
    if r["group"] == "M30" and r["source"] == "minio_app":
        reasons.append("app_test_session")
    if not r["text"]:
        reasons.append("no_transcript")
    if r["duration_s"] < MIN_DUR:
        reasons.append("too_short")
    if r["duration_s"] > MAX_DUR:
        reasons.append("too_long")
    if r["peak_dbfs"] < -45:
        reasons.append("no_signal")
    if r["utt_id"] in dup_utts:
        reasons.append("duplicate_take")
    return ";".join(reasons)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    srcs = sorted(
        p for base in (LOCAL_RAW, APP_RAW) for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXT and not p.name.startswith(".")
    )
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(build_row, srcs))

    # same speaker + prompt + mode recorded twice: keep the first, flag the rest
    seen, dup = set(), set()
    for r in rows:
        if r["utt_id"] in seen:
            dup.add(r["utt_id"])
        seen.add(r["utt_id"])
    if dup:
        counts: dict[str, int] = {}
        for r in rows:
            if r["utt_id"] in dup:
                counts[r["utt_id"]] = counts.get(r["utt_id"], 0) + 1
                if counts[r["utt_id"]] > 1:
                    r["utt_id"] += f"__dup{counts[r['utt_id']] - 1}"
    dup_ids = {r["utt_id"] for r in rows if "__dup" in r["utt_id"]}
    for r in rows:
        r["exclude_reason"] = exclusion(r, dup_ids)
        r["usable"] = not r["exclude_reason"]
        r["heldout_text"] = r["code"] in HELDOUT_CODES
        if not r["usable"]:
            r["split"] = ""
        elif r["speaker_id"] in TEST_SPEAKERS:
            r["split"] = "test"
        elif r["speaker_id"] in VAL_SPEAKERS:
            r["split"] = "val"
        else:
            r["split"] = "test_newtext" if r["heldout_text"] else "train"

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for split in SPLITS:
        with open(OUT / f"{split}.jsonl", "w", encoding="utf8") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps(dict(audio_filepath=r["audio_path"], text=r["text"],
                                            duration=r["duration_s"], speaker_id=r["speaker_id"],
                                            mode=r["mode"], intent=r["intent"],
                                            heldout_text=r["heldout_text"]), ensure_ascii=False) + "\n")

    usable = [r for r in rows if r["usable"]]
    print(f"recordings: {len(rows)}  usable: {len(usable)}  "
          f"hours usable: {sum(r['duration_s'] for r in usable) / 3600:.2f}")
    for split in SPLITS:
        s = [r for r in usable if r["split"] == split]
        print(f"  {split:12s} {len(s):5d} utts  {sum(r['duration_s'] for r in s) / 60:6.1f} min  "
              f"{len({r['speaker_id'] for r in s})} speakers")


if __name__ == "__main__":
    main()
