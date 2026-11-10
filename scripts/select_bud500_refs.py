"""Pick clean, diverse reference clips from Bud500 shards for zero-shot voice cloning,
and a separate pool of Bud500 utterances to mix into ASR training as real general
Vietnamese speech (keeps the model from over-fitting to TTS timbre/hospital text).

Bud500 has no speaker ids. Consecutive rows usually come from the same source
recording, so references are taken at most one per BLOCK consecutive rows, which
spreads them over many sources/speakers. Accent is unlabeled (Bắc/Trung/Nam mixed).

Outputs (data/processed/synth/):
  refs/<ref_id>.wav       16 kHz mono reference clips, 3-10 s
  refs.jsonl              {ref_id, path, text, duration_s, snr_db, f0_hz, gender_guess, split}
  bud500_mix/<id>.wav     general-speech clips for the ASR training mix
  bud500_mix.jsonl        {utt_id, audio_path, text, duration_s, split}

Run with tools/tts/.venv/bin/python (needs pyarrow + soundfile).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "bud500" / "data"
OUT = ROOT / "data" / "processed" / "synth"
SR = 16000
BLOCK = 20


def metrics(x: np.ndarray, sr: int) -> dict:
    hop, win = int(0.010 * sr), int(0.025 * sr)
    n = 1 + max(0, (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    db = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-10)
    p10, p95 = np.percentile(db, 10), np.percentile(db, 95)
    speech = db > max(p95 - 25, p10 + 6)
    # F0 by normalised autocorrelation on loud frames (same method as prepare_voice_dataset)
    w = int(0.04 * sr); lo, hi = sr // 500, sr // 70; f0 = []
    starts = idx[speech, 0]
    for s in starts[:: max(1, len(starts) // 40)]:
        seg = x[s : s + w]
        if len(seg) < w:
            continue
        seg = seg - seg.mean(); ac = np.correlate(seg, seg, "full")[w - 1 :]
        if ac[0] <= 0:
            continue
        ac /= ac[0]; lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.5:
            f0.append(sr / lag)
    spec = np.abs(np.fft.rfft(x)) ** 2; k = len(spec) // 200
    bands = 10 * np.log10(spec[: k * 200].reshape(200, -1).mean(axis=1) + 1e-12)
    cutoff = (np.flatnonzero(bands > bands.max() - 60).max() + 1) * (sr / 2) / 200
    return dict(snr_db=float(p95 - p10), f0_hz=float(np.median(f0)) if f0 else float("nan"),
                clip=float(np.mean(np.abs(x) >= 0.999)), cutoff_hz=float(cutoff),
                speech_ratio=float(speech.mean()))


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(t))).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-refs", type=int, default=240)
    ap.add_argument("--n-val-refs", type=int, default=24)
    ap.add_argument("--n-mix", type=int, default=2500, help="Bud500 utterances for the ASR training mix")
    args = ap.parse_args()

    shards = sorted(SRC.glob("train-*.parquet"))
    (OUT / "refs").mkdir(parents=True, exist_ok=True)
    (OUT / "bud500_mix").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    cands, mix_pool = [], []
    for si, shard in enumerate(shards):
        tab = pq.read_table(shard).to_pandas()
        for b0 in range(0, len(tab), BLOCK):
            block = tab.iloc[b0 : b0 + BLOCK]
            best = None
            for i, r in block.iterrows():
                x, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                d, text = len(x) / sr, clean_text(r.transcription)
                key = (si, int(i))
                if 1.0 <= d <= 12 and text:
                    mix_pool.append((key, d))
                if not (3 <= d <= 10) or len(text.split()) < 6:
                    continue
                m = metrics(x, sr)
                if m["clip"] > 1e-4 or m["cutoff_hz"] < 6000 or m["speech_ratio"] < 0.55 or np.isnan(m["f0_hz"]):
                    continue
                if best is None or m["snr_db"] > best[1]["snr_db"]:
                    best = (key, m, d, text)
            if best and best[1]["snr_db"] >= 35:
                cands.append(best)
        print(f"{shard.name}: {len(cands)} ref candidates so far, mix pool {len(mix_pool)}", flush=True)

    # gender balance by F0 (guess only: 165 Hz split), then spread evenly
    male = [c for c in cands if c[1]["f0_hz"] < 165]
    female = [c for c in cands if c[1]["f0_hz"] >= 165]
    half = args.n_refs // 2
    pick = [male[i] for i in rng.permutation(len(male))[:half]] + [female[i] for i in rng.permutation(len(female))[: args.n_refs - min(half, len(male))]]
    pick = pick[: args.n_refs]
    val_ids = set(rng.permutation(len(pick))[: args.n_val_refs].tolist())
    ref_keys = {c[0] for c in pick}

    tabs = {si: pq.read_table(s).to_pandas() for si, s in enumerate(shards)}
    with (OUT / "refs.jsonl").open("w", encoding="utf8") as f:
        for j, (key, m, d, text) in enumerate(pick):
            x, sr = sf.read(io.BytesIO(tabs[key[0]].iloc[key[1]].audio["bytes"]), dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            rid = f"bud_{key[0]}_{key[1]:05d}"
            p = OUT / "refs" / f"{rid}.wav"
            sf.write(p, x / max(1e-3, np.abs(x).max()) * 0.9, sr, subtype="PCM_16")
            f.write(json.dumps(dict(ref_id=rid, path=str(p.relative_to(ROOT)), text=text, duration_s=round(d, 2),
                                    snr_db=round(m["snr_db"], 1), f0_hz=round(m["f0_hz"], 1),
                                    gender_guess="male" if m["f0_hz"] < 165 else "female",
                                    split="val" if j in val_ids else "train"), ensure_ascii=False) + "\n")

    # general-speech mix: not a reference clip, not from the same block as a val reference
    val_blocks = {(c[0][0], c[0][1] // BLOCK) for j, c in enumerate(pick) if j in val_ids}
    pool = [k for k, d in mix_pool if k not in ref_keys]
    pool = [pool[i] for i in rng.permutation(len(pool))[: args.n_mix]]
    with (OUT / "bud500_mix.jsonl").open("w", encoding="utf8") as f:
        for key in pool:
            r = tabs[key[0]].iloc[key[1]]
            x, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            uid = f"bud500_{key[0]}_{key[1]:05d}"
            p = OUT / "bud500_mix" / f"{uid}.wav"
            sf.write(p, x, sr, subtype="PCM_16")
            split = "val" if (key[0], key[1] // BLOCK) in val_blocks else "train"
            f.write(json.dumps(dict(utt_id=uid, audio_path=str(p.relative_to(ROOT)), text=clean_text(r.transcription),
                                    duration_s=round(len(x) / sr, 2), split=split), ensure_ascii=False) + "\n")
    print(f"refs: {len(pick)} ({len(male)} male / {len(female)} female candidates), val refs {len(val_ids)}")
    print(f"bud500 mix: {len(pool)} utterances, {sum(d for k, d in mix_pool if k in set(pool)) / 3600:.2f} h")


if __name__ == "__main__":
    main()
