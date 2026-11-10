"""Round-trip QC for synthetic TTS audio: transcribe with the *base* PhoASR model and
score CER against the text the TTS was asked to say.

High CER means the TTS mispronounced, skipped or hallucinated words, so the label no
longer matches the audio. Those clips are dropped. The threshold is deliberately loose
(default CER <= 0.35): the base model itself makes ~13% CER on clean real speech, so a
tight threshold would also throw away the hard-but-correct clips we want to train on.

  sim_vision/python/.venv/bin/python scripts/qc_synth_asr.py [--manifest ...] [--wav-dir ...]
Writes <manifest dir>/qc.csv and prints the CER distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jiwer
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_asr_testset import normalize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/processed/synth/tts/manifest.jsonl"))
    ap.add_argument("--wav-dir", default="", help="QC every *.wav here instead (text from <name>.txt or none)")
    ap.add_argument("--model", default=str(ROOT / "models" / "PhoASR-whisper-small"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-cer", type=float, default=0.35)
    args = ap.parse_args()

    man = Path(args.manifest)
    df = pd.DataFrame([json.loads(l) for l in man.open(encoding="utf8")])
    out = man.parent / "qc.csv"
    if out.exists():  # resume: only score new clips
        old = pd.read_csv(out)
        todo = df[~df.utt_id.isin(old.utt_id)]
    else:
        old, todo = None, df
    print(f"{len(todo)} clips to score ({len(df)} in manifest)")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    local = Path(args.model).is_dir()  # local folder, or a HF repo id such as Qualcomm-AI-Research/PhoASR-whisper-small
    proc = AutoProcessor.from_pretrained(args.model, local_files_only=local)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, local_files_only=local).to(device).eval()
    hyps = []
    for i in range(0, len(todo), args.batch_size):
        b = todo.iloc[i : i + args.batch_size]
        audio = [sf.read(ROOT / p, dtype="float32")[0] for p in b.audio_path]
        f = proc(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            seq = model.generate(input_features=f.input_features.to(device), attention_mask=f.attention_mask.to(device),
                                 language="vi", task="transcribe", use_cache=True)
        hyps += [h.strip() for h in proc.batch_decode(seq, skip_special_tokens=True)]
        if (i // args.batch_size) % 20 == 0:
            print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)
    todo = todo.assign(hyp=hyps)
    todo["cer"] = [jiwer.cer(normalize(r), normalize(h)) if normalize(r) else 1.0 for r, h in zip(todo.text, todo.hyp)]
    todo["wer"] = [jiwer.wer(normalize(r), normalize(h)) if normalize(r) else 1.0 for r, h in zip(todo.text, todo.hyp)]
    cols = ["utt_id", "mode", "voice", "register", "split", "duration_s", "text", "hyp", "cer", "wer"]
    res = pd.concat([old, todo[cols]]) if old is not None else todo[cols]
    res["keep"] = res.cer <= args.max_cer
    res.to_csv(out, index=False)
    print(res.cer.describe(percentiles=[.5, .75, .9, .95]).round(3).to_string())
    print(f"keep {res.keep.mean():.1%} at CER<={args.max_cer}")
    print(res.groupby(["mode", "register"]).agg(n=("cer", "size"), cer_med=("cer", "median"), keep=("keep", "mean")).round(3))
    print(res.assign(v=res.voice.str.replace("design:", "")).query("mode=='design'")
          .groupby("v").agg(n=("cer", "size"), cer_med=("cer", "median"), keep=("keep", "mean")).round(3))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
