"""Synthesize hospital-domain training audio with OmniVoice (zero-shot clone + voice design).

Each sentence of data/processed/synth/texts.jsonl is rendered once (or --renders times)
by a voice matched to its register:
  child  -> voice design: child / teenager, high pitch            (no child refs exist)
  elder  -> voice design: elderly / middle-aged (70%) or a Bud500 clone slowed to 0.85-0.95x
  young/adult -> Bud500 clone (70%) or voice design young adult / middle-aged
Voice split follows the text split: val sentences only use val Bud500 refs, so val
measures unseen voices + unseen text; voice-design voices are fresh per utterance.

Audio is written at 16 kHz PCM16. Resumable: existing wavs are skipped and the
manifest is appended, so the run can be stopped and restarted.

  tools/tts/.venv/bin/python scripts/gen_synth_tts.py --limit 40      # quick look
  tools/tts/.venv/bin/python scripts/gen_synth_tts.py                 # full run
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from omnivoice import OmniVoice

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "processed" / "synth"
SR = 16000

DESIGN = {
    "child": ["male, child, high pitch", "female, child, high pitch", "female, child, very high pitch",
              "male, child, moderate pitch", "male, teenager, moderate pitch", "female, teenager, high pitch"],
    "elder": ["male, elderly, low pitch", "male, elderly, moderate pitch", "female, elderly, moderate pitch",
              "female, elderly, low pitch", "male, middle-aged, low pitch", "female, middle-aged, moderate pitch",
              "male, elderly, very low pitch", "female, elderly, low pitch"],
    "adult": ["male, middle-aged, moderate pitch", "female, middle-aged, moderate pitch", "male, young adult, low pitch",
              "female, young adult, moderate pitch", "male, middle-aged, low pitch", "female, young adult, high pitch"],
}
DESIGN["young"] = ["male, young adult, moderate pitch", "female, young adult, high pitch",
                   "female, young adult, moderate pitch", "male, young adult, low pitch", "male, teenager, moderate pitch"]
P_CLONE = {"child": 0.0, "elder": 0.3, "adult": 0.7, "young": 0.7}


def plan(texts: list[dict], refs: list[dict], renders: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_split = {s: [r for r in refs if r["split"] == s] for s in ("train", "val")}
    jobs = []
    for t in texts:
        for k in range(renders):
            j = dict(t, utt_id=f"syn_{t['text_id']}_{k}", render=k)
            if rng.random() < P_CLONE[t["register"]]:
                ref = rng.choice(by_split[t["split"]])
                j.update(mode="clone", ref_id=ref["ref_id"], ref_path=ref["path"], ref_text=ref["text"],
                         voice=ref["ref_id"], gender=ref["gender_guess"],
                         speed=round(rng.uniform(0.85, 0.95), 2) if t["register"] == "elder" else round(rng.uniform(0.92, 1.12), 2))
            else:
                ins = rng.choice(DESIGN[t["register"]])
                j.update(mode="design", instruct=ins, voice=f"design:{ins}", gender=ins.split(",")[0],
                         speed=round(rng.uniform(0.82, 0.95), 2) if "elderly" in ins else round(rng.uniform(0.92, 1.1), 2))
            jobs.append(j)
    rng.shuffle(jobs)
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="g-group-ai-lab/g-omnivoice",
                    help="Vietnamese fine-tune of OmniVoice (gated: accept terms on HF first); k2-fsa/OmniVoice = multilingual base")
    ap.add_argument("--renders", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-step", type=int, default=16)  # 32 -> 16: 2x faster, CER 2.1% -> 2.7%
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default=str(SYN / "tts"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.device == "mps" and args.batch > 1:
        # OmniVoice on MPS corrupts every padded item of a batch (only the longest text
        # comes out right; the others decode to "Ê."/"A.A."). Batch 1 is just as fast there.
        print("MPS: forcing --batch 1 (batched generation is broken on MPS)")
        args.batch = 1

    texts = [json.loads(l) for l in (SYN / "texts.jsonl").open(encoding="utf8")]
    refs = [json.loads(l) for l in (SYN / "refs.jsonl").open(encoding="utf8")]
    jobs = plan(texts, refs, args.renders, args.seed)
    if args.limit:
        jobs = jobs[: args.limit]
    out = Path(args.out).resolve(); (out / "wav").mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.jsonl"
    done = {json.loads(l)["utt_id"] for l in man_path.open(encoding="utf8")} if man_path.exists() else set()
    jobs = [j for j in jobs if j["utt_id"] not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} done) model={args.model} device={args.device}", flush=True)
    if not jobs:
        return

    model = OmniVoice.from_pretrained(args.model, device_map=args.device,
                                      dtype=torch.float16 if args.device != "cpu" else torch.float32)
    tts_sr = getattr(model, "sampling_rate", 24000)
    ref_cache: dict[str, tuple] = {}

    def ref_audio(p: str):
        if p not in ref_cache:
            x, sr = sf.read(ROOT / p, dtype="float32")
            ref_cache[p] = (torch.from_numpy(x)[None], sr)
        return ref_cache[p]

    t_start, audio_s = time.time(), 0.0
    with man_path.open("a", encoding="utf8") as man:
        # group by mode so each generate() call is homogeneous
        for mode in ("clone", "design"):
            mj = [j for j in jobs if j["mode"] == mode]
            if args.batch > 1:  # similar lengths per batch -> less padding, faster
                mj.sort(key=lambda j: len(j["tts_text"]))
            for b in range(0, len(mj), args.batch):
                batch = mj[b : b + args.batch]
                kw = dict(text=[j["tts_text"] for j in batch], language="vi", speed=[j["speed"] for j in batch],
                          num_step=args.num_step)
                if mode == "clone":
                    kw.update(ref_audio=[ref_audio(j["ref_path"]) for j in batch], ref_text=[j["ref_text"] for j in batch])
                else:
                    kw.update(instruct=[j["instruct"] for j in batch])
                try:
                    wavs = model.generate(**kw)
                except Exception as e:  # one bad item should not kill a multi-hour run
                    print(f"  batch failed ({e!r}); retrying one by one", flush=True)
                    wavs = []
                    for i in range(len(batch)):
                        one = {k: (v[i : i + 1] if isinstance(v, list) else v) for k, v in kw.items()}
                        try:
                            wavs += model.generate(**one)
                        except Exception as e2:
                            print(f"  skip {batch[i]['utt_id']}: {e2!r}", flush=True)
                            wavs.append(None)
                for j, w in zip(batch, wavs):
                    if w is None or len(w) < tts_sr * 0.3:
                        continue
                    y = AF.resample(torch.from_numpy(np.asarray(w, dtype=np.float32)), tts_sr, SR).numpy()
                    y = y / max(1e-4, float(np.abs(y).max())) * 0.89
                    p = out / "wav" / f"{j['utt_id']}.wav"
                    sf.write(p, y, SR, subtype="PCM_16")
                    d = len(y) / SR; audio_s += d
                    keep = {k: j[k] for k in ("utt_id", "text_id", "text", "tts_text", "intent", "register", "split",
                                              "mode", "voice", "gender", "speed") if k in j}
                    rel = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)  # --out may be on Drive
                    man.write(json.dumps(dict(keep, audio_path=rel, duration_s=round(d, 3)),
                                         ensure_ascii=False) + "\n")
                man.flush()
                el = time.time() - t_start
                n_done = b + len(batch) + (len(jobs) - len(mj) if mode == "design" else 0)
                print(f"  [{mode}] {n_done}/{len(jobs)}  audio {audio_s/60:.1f} min  RTF {el/max(audio_s,1e-6):.2f}  "
                      f"ETA {(len(jobs)-n_done) * el / max(n_done,1) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
