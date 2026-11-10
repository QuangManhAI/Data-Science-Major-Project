"""Evaluate an ASR checkpoint on a split of data/processed/voice_dataset.

Default: the pretrained PhoASR-whisper-small (no fine-tuning) on the `test` split,
i.e. the baseline every fine-tuned checkpoint must beat.

Outputs (experiments/results/<run-name>/):
  predictions.csv   one row per utterance: reference, hypothesis, WER/CER, metadata
  summary.json      corpus WER/CER overall and broken down by speaker, group,
                    mode (_r/_n), held-out sentence, SNR band, digits in prompt

Usage:
  sim_vision/python/.venv/bin/python scripts/eval_asr_testset.py
  ... --model models_finetuned/<ckpt> --split test --run-name ft_v1_test
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "voice_dataset"


DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def num_to_words(n: int) -> str:
    """Vietnamese reading of 0-999 (prompts contain room numbers like 304, 410)."""
    if n < 10:
        return DIGITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        head = "mười" if tens == 1 else f"{DIGITS[tens]} mươi"
        tail = {0: "", 1: " mốt" if tens > 1 else " một", 4: " tư" if tens > 1 else " bốn",
                5: " lăm"}.get(unit, f" {DIGITS[unit]}")
        return head + tail
    if n < 1000:
        hund, rest = divmod(n, 100)
        if rest == 0:
            return f"{DIGITS[hund]} trăm"
        return f"{DIGITS[hund]} trăm " + (f"linh {DIGITS[rest]}" if rest < 10 else num_to_words(rest))
    return " ".join(DIGITS[int(c)] for c in str(n))


def normalize(text: str) -> str:
    """Spoken-form text for scoring: NFC, lowercase, numbers read out as words,
    `lẻ` == `linh`, punctuation dropped. Applied identically to reference and
    hypothesis, so "410" and "bốn trăm mười" score as a match."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"\d+", lambda m: f" {num_to_words(int(m.group()))} ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\blẻ\b", "linh", text)
    return re.sub(r"\s+", " ", text).strip()


def corpus(df: pd.DataFrame) -> dict:
    refs, hyps = df.ref_norm.tolist(), df.hyp_norm.tolist()
    m = jiwer.process_words(refs, hyps)
    return dict(
        n=len(df), minutes=round(df.duration_s.sum() / 60, 1),
        wer=round(m.wer, 4), cer=round(jiwer.cer(refs, hyps), 4),
        sub=m.substitutions, dele=m.deletions, ins=m.insertions, ref_words=sum(len(r.split()) for r in refs),
        sent_acc=round(float((df.ref_norm == df.hyp_norm).mean()), 4),
    )


def breakdown(df: pd.DataFrame, col: str) -> dict:
    return {str(k): corpus(g) for k, g in df.groupby(col, dropna=False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "PhoASR-whisper-small"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--run-name", default="baseline_phoasr_small_test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-beams", type=int, default=1)
    args = ap.parse_args()

    man = pd.read_csv(DATA / "manifest.csv")
    df = man[man.split == args.split].reset_index(drop=True)
    out = ROOT / "experiments" / "results" / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, local_files_only=True).to(device).eval()
    print(f"model={args.model}  device={device}  split={args.split}  utts={len(df)}")

    hyps, infer_s = [], 0.0
    for i in range(0, len(df), args.batch_size):
        batch = df.iloc[i : i + args.batch_size]
        audio = [sf.read(DATA / p, dtype="float32")[0] for p in batch.audio_path]
        feats = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            seq = model.generate(
                input_features=feats.input_features.to(device),
                attention_mask=feats.attention_mask.to(device),
                language="vi", task="transcribe", num_beams=args.num_beams, use_cache=True,
            )
        if device == "mps":
            torch.mps.synchronize()
        infer_s += time.perf_counter() - t0
        hyps += [h.strip() for h in processor.batch_decode(seq, skip_special_tokens=True)]
        print(f"  {min(i + args.batch_size, len(df))}/{len(df)}", flush=True)

    df["hyp"] = hyps
    df["ref_norm"] = df.text.map(normalize)
    df["hyp_norm"] = df.hyp.map(normalize)
    df["wer"] = [jiwer.wer(r, h) for r, h in zip(df.ref_norm, df.hyp_norm)]
    df["cer"] = [jiwer.cer(r, h) for r, h in zip(df.ref_norm, df.hyp_norm)]
    df["group"] = df.apply(lambda r: ("trẻ em" if r.age <= 12 else f"{r.gender} {20 if r.age < 25 else 30}"), axis=1)
    df["snr_band"] = pd.cut(df.snr_db, [0, 30, 45, 60, 200], labels=["<30", "30-45", "45-60", ">60"])
    df["has_digit"] = df.text.str.contains(r"\d")

    summary = dict(
        model=args.model, split=args.split, device=device, num_beams=args.num_beams,
        rtf=round(infer_s / df.duration_s.sum(), 4),
        overall=corpus(df),
        macro_wer_by_speaker=round(float(np.mean([corpus(g)["wer"] for _, g in df.groupby("speaker_id")])), 4),
        macro_cer_by_speaker=round(float(np.mean([corpus(g)["cer"] for _, g in df.groupby("speaker_id")])), 4),
        by_speaker=breakdown(df, "speaker_id"), by_group=breakdown(df, "group"),
        by_mode=breakdown(df, "mode"), by_heldout_text=breakdown(df, "heldout_text"),
        by_snr=breakdown(df, "snr_band"), by_has_digit=breakdown(df, "has_digit"),
        by_intent=breakdown(df, "intent"),
        r_only_by_speaker=breakdown(df[df["mode"] == "r"], "speaker_id"),
    )
    cols = ["utt_id", "speaker_id", "group", "mode", "code", "intent", "heldout_text", "duration_s",
            "snr_db", "text", "hyp", "ref_norm", "hyp_norm", "wer", "cer"]
    df[cols].to_csv(out / "predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf8")
    o = summary["overall"]
    print(f"\nmacro WER (speaker avg) {summary['macro_wer_by_speaker']:.2%}")
    print(f"WER {o['wer']:.2%}  CER {o['cer']:.2%}  sent_acc {o['sent_acc']:.2%}  RTF {summary['rtf']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
