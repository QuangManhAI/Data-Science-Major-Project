"""Build voice_dataset_v2: synthetic + Bud500 for train/val, ALL real recordings for test.

  train/val  OmniVoice hospital sentences (QC-passed)  +  Bud500 general speech
  test       every usable real recording (1726 utts, 29 speakers) from voice_dataset
  test_newtext  real utterances of the 16 held-out prompt sentences (subset of test)

Synthetic clips get an offline *channel* degradation (p=--p-codec) that the online
training augmentation cannot produce: lossy codecs like the recorder app (Opus/WebM,
AAC/MP4, MP3) and a telephone band. Hospital noise, gain, speed and low-pass stay
online in the trainer, so every epoch sees a different noisy copy.

Layout matches voice_dataset, so scripts/train_asr_finetune.py --data voice_dataset_v2
and the Colab notebook work unchanged:
  wav/{synth,bud500}/*.flac  wav/real/<spk>/*.wav  noise/  manifest.csv   {train,val,test,test_newtext}.jsonl

  sim_vision/python/.venv/bin/python scripts/build_dataset_v2.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "processed" / "synth"
REAL = ROOT / "data" / "processed" / "voice_dataset"
OUT = ROOT / "data" / "processed" / "voice_dataset_v2"
SR = 16000
AGE = {"child": 10, "young": 22, "adult": 35, "elder": 70}
CODECS = [  # (name, ffmpeg encode args, container ext)
    ("opus16k", ["-c:a", "libopus", "-b:a", "16k"], "webm"),
    ("opus24k", ["-c:a", "libopus", "-b:a", "24k"], "webm"),
    ("aac32k", ["-c:a", "aac", "-b:a", "32k"], "m4a"),
    ("mp3_32k", ["-c:a", "libmp3lame", "-b:a", "32k"], "mp3"),
    ("phone", ["-af", "highpass=f=300,lowpass=f=3400", "-c:a", "pcm_s16le"], "wav"),
]


def speech_rms_db(x: np.ndarray) -> float:
    hop, win = 160, 400
    n = 1 + max(0, (len(x) - win) // hop)
    fr = x[np.arange(win)[None, :] + hop * np.arange(n)[:, None]]
    db = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-10)
    loud = db > np.percentile(db, 95) - 25
    return float(10 * np.log10(np.mean(fr[loud] ** 2) + 1e-10))


def level(x: np.ndarray, target_db: float = -23.0, peak_db: float = -1.0) -> np.ndarray:
    """Same loudness rule as prepare_voice_dataset: speech RMS -23 dBFS, peak <= -1 dBFS."""
    g = min(target_db - speech_rms_db(x), peak_db - 20 * np.log10(np.abs(x).max() + 1e-9))
    return np.clip(x * 10 ** (g / 20), -1, 1)


def degrade(src: Path, dst: Path, codec: tuple, tmp: Path) -> None:
    name, enc, ext = codec
    mid = tmp / f"{dst.stem}.{ext}"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), *enc, str(mid)], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(mid), "-ac", "1", "-ar", str(SR),
                    "-sample_fmt", "s16", str(dst)], check=True)
    mid.unlink()
    # dst is .flac: lossless re-pack of the decoded (degraded) signal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", default=str(SYN / "tts"))
    ap.add_argument("--p-codec", type=float, default=0.5)
    ap.add_argument("--max-bud500", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tts = Path(args.tts)

    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []

    # ---------------------------------------------------------------- synthetic
    syn = pd.DataFrame([json.loads(l) for l in (tts / "manifest.jsonl").open(encoding="utf8")])
    qc = pd.read_csv(tts / "qc.csv")[["utt_id", "cer", "keep"]]
    syn = syn.merge(qc, on="utt_id", how="inner")
    print(f"synthetic: {len(syn)} QC'd, keep {int(syn.keep.sum())} (CER<=thr)")
    syn = syn[syn.keep]
    (OUT / "wav" / "synth").mkdir(parents=True)
    with tempfile.TemporaryDirectory() as td:
        for i, r in enumerate(syn.itertuples()):
            src, dst = ROOT / r.audio_path, OUT / "wav" / "synth" / f"{r.utt_id}.flac"
            codec = "clean"
            if rng.random() < args.p_codec:
                c = rng.choice(CODECS); codec = c[0]
                degrade(src, dst, c, Path(td))
                x, _ = sf.read(dst, dtype="float32")
            else:
                x, _ = sf.read(src, dtype="float32")
            # random lead/trail silence (real takes have 0-1 s, TTS output is trimmed tight)
            lead, trail = (np.zeros(int(SR * rng.uniform(0.05, 0.6)), np.float32),
                           np.zeros(int(SR * rng.uniform(0.05, 0.5)), np.float32))
            x = level(np.concatenate([lead, x, trail]))
            sf.write(dst, x, SR, subtype="PCM_16", format="FLAC")
            rows.append(dict(utt_id=r.utt_id, audio_path=f"wav/synth/{r.utt_id}.flac", source=f"synth_{r.mode}",
                             speaker_id=r.voice, gender=r.gender, age=AGE[r.register], register=r.register,
                             mode="r", text=r.text, intent=r.intent, duration_s=round(len(x) / SR, 3),
                             snr_db=60.0, codec=codec, tts_cer=r.cer, heldout_text=False, split=r.split))
            if i % 1000 == 0:
                print(f"  synth {i}/{len(syn)}", flush=True)

    # ---------------------------------------------------------------- Bud500 general speech
    bud = [json.loads(l) for l in (SYN / "bud500_mix.jsonl").open(encoding="utf8")][: args.max_bud500]
    (OUT / "wav" / "bud500").mkdir(parents=True)
    for r in bud:
        x, _ = sf.read(ROOT / r["audio_path"], dtype="float32")
        x = level(x)
        sf.write(OUT / "wav" / "bud500" / f"{r['utt_id']}.flac", x, SR, subtype="PCM_16", format="FLAC")
        rows.append(dict(utt_id=r["utt_id"], audio_path=f"wav/bud500/{r['utt_id']}.flac", source="bud500",
                         speaker_id="bud500", gender="", age=-1, register="", mode="u", text=r["text"],
                         intent="", duration_s=r["duration_s"], snr_db=50.0, codec="src", tts_cer=np.nan,
                         heldout_text=False, split=r["split"]))

    # ---------------------------------------------------------------- real -> test
    real = pd.read_csv(REAL / "manifest.csv")
    real = real[real.usable.astype(str).str.lower().isin(["true", "1"])]
    for r in real.itertuples():
        dst = OUT / "wav" / "real" / r.audio_path.removeprefix("wav/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REAL / r.audio_path, dst)
        rows.append(dict(utt_id=r.utt_id, audio_path=str(dst.relative_to(OUT)), source="real",
                         speaker_id=r.speaker_id, gender=r.gender, age=r.age, register="", mode=r.mode, text=r.text,
                         intent=r.intent, duration_s=r.duration_s, snr_db=r.snr_db, codec=r.src_codec, tts_cer=np.nan,
                         heldout_text=bool(r.heldout_text), split="test", old_split=r.split))

    # recorded ambience for the trainer's HospitalNoise (same files as the v1 Colab zip)
    (OUT / "noise").mkdir()
    for f in sorted((ROOT / "sim_vision" / "godot" / "assets" / "sounds").glob("elevator_*.wav")):
        shutil.copy2(f, OUT / "noise" / f.name)

    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)
    splits = {"train": man.split == "train", "val": man.split == "val", "test": man.split == "test",
              "test_newtext": (man.split == "test") & man.heldout_text.astype(bool)}
    for name, sel in splits.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf8") as f:
            for r in man[sel].itertuples():
                f.write(json.dumps(dict(audio_filepath=r.audio_path, text=r.text, duration=r.duration_s,
                                        speaker_id=r.speaker_id, mode=r.mode, intent=r.intent,
                                        heldout_text=bool(r.heldout_text), source=r.source), ensure_ascii=False) + "\n")
    summ = man.assign(split=man.split.where(~(splits["test_newtext"]), "test")).groupby(["split", "source"]).agg(
        utts=("utt_id", "size"), hours=("duration_s", lambda s: round(s.sum() / 3600, 2)))
    print(summ)
    print(man[man.split != "test"].groupby(["split", "register"]).duration_s.sum().div(60).round(1))
    print(f"test_newtext: {int(splits['test_newtext'].sum())} utts  -> {OUT}")


if __name__ == "__main__":
    main()
