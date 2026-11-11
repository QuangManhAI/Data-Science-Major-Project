"""Fine-tune PhoASR-whisper-small on the hospital voice dataset, with heavy
hospital-noise augmentation and light everything-else augmentation.

Everything a run produces goes into one timestamped folder:

  experiments/runs/<YYYYmmdd_HHMMSS>_ASR-<run-name>/
    config.json            all arguments + dataset sizes + git commit
    train.log              full console log
    metrics_steps.csv      step, epoch, lr, train loss, grad norm, sec/step
    metrics_epochs.csv     epoch, train loss, val loss, val WER/CER (all, _r, _n)
    plots/                 loss_curve.png, val_wer.png, lr.png, test_by_speaker.png,
                           test_by_snr.png
    checkpoints/best/      best val WER(_r) model + processor (save_pretrained)
    checkpoints/last/      model + processor at the end of training
    eval/                  final test / test_newtext / noisy-test predictions + summary.json
    aug_samples/           a few augmented training clips to listen to

Data: data/processed/voice_dataset/{train,val,test,test_newtext}.jsonl
      (built by scripts/prepare_voice_dataset.py). See docs/VOICE_DATASET_EDA.md.

Augmentation (training only)
  hospital noise  p=0.85, SNR U(0, 20) dB, 1-2 layers mixed from:
                  babble (3-6 other train speakers), HVAC hum (filtered pink/brown
                  noise + mains hum), monitor beeps, trolley rumble, elevator
                  chime/hum (sim_vision/godot/assets/sounds), + any wav in --noise-dir
  light           speed 0.9-1.1x (p=0.3), gain +-6 dB (p=0.5), low-pass 3.5-7 kHz (p=0.15),
                  SpecAugment (time/feature masks, p=0.05)
  `_n` takes use the script text as label (approximate), so their loss is scaled by
  --n-weight (default 0.5).

Usage
  sim_vision/python/.venv/bin/python scripts/train_asr_finetune.py
  ... --epochs 8 --lr 1e-5 --batch-size 4 --grad-accum 4 --run-name noisy-v1
  ... --max-steps 10 --eval-limit 16           # smoke test
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import logging
import math
import random
import re
import subprocess
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import jiwer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, resample_poly, sosfilt
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "voice_dataset"
SR = 16000
log = logging.getLogger("asr_ft")

# ============================================================================ text
DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def num_to_words(n: int) -> str:
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
    """Scoring normaliser, identical to scripts/eval_asr_testset.py."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"\d+", lambda m: f" {num_to_words(int(m.group()))} ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\blẻ\b", "linh", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================ audio / noise
def load_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def fit_length(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Loop or crop a noise clip to exactly n samples, random offset."""
    if len(x) == 0:
        return np.zeros(n, np.float32)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))) + 1)
    start = rng.integers(0, len(x) - n + 1)
    return x[start : start + n]


def colored_noise(n: int, rng: np.random.Generator, beta: float) -> np.ndarray:
    """1/f^beta noise (beta=1 pink, 2 brown)."""
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    x = np.fft.irfft(spec / f ** (beta / 2), n)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def lowpass(x: np.ndarray, cutoff: float, order: int = 6) -> np.ndarray:
    sos = butter(order, cutoff, btype="low", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


class HospitalNoise:
    """Generates hospital-like background noise for a given length."""

    def __init__(self, babble_pool: list[np.ndarray], files: list[np.ndarray]):
        self.babble_pool = babble_pool
        self.files = files
        self.kinds = ["babble", "hvac", "beeps", "trolley"] + (["file"] if files else [])
        # babble and recorded ambience are the most realistic -> weighted up
        w = {"babble": 3.0, "hvac": 2.0, "beeps": 1.0, "trolley": 1.0, "file": 2.0}
        self.p = np.array([w[k] for k in self.kinds]) / sum(w[k] for k in self.kinds)

    def babble(self, n, rng):
        k = rng.integers(3, 7)
        idx = rng.choice(len(self.babble_pool), size=k, replace=False)
        mix = sum(fit_length(self.babble_pool[i], n, rng) / (rms(self.babble_pool[i]) + 1e-6) for i in idx)
        return lowpass(mix, rng.uniform(3000, 6000), 4)  # distant speech: dull, reverberant

    def hvac(self, n, rng):
        x = colored_noise(n, rng, rng.choice([1.0, 2.0]))
        x = lowpass(x, rng.uniform(300, 1500), 4)
        t = np.arange(n) / SR
        hum = sum(np.sin(2 * np.pi * h * 50 * t + rng.uniform(0, 6.3)) / h for h in (1, 2, 3))
        return (x / (rms(x) + 1e-9) + 0.3 * hum / (rms(hum) + 1e-9)).astype(np.float32)

    def beeps(self, n, rng):
        t = np.arange(n) / SR
        f0, period, dur = rng.uniform(800, 2000), rng.uniform(0.6, 1.5), rng.uniform(0.08, 0.2)
        gate = ((t + rng.uniform(0, period)) % period) < dur
        tone = np.sin(2 * np.pi * f0 * t) * gate
        floor = 0.05 * colored_noise(n, rng, 1.0)
        return (tone + floor).astype(np.float32)

    def trolley(self, n, rng):
        x = lowpass(colored_noise(n, rng, 2.0), rng.uniform(200, 800), 4)
        t = np.arange(n) / SR
        rattle = 1 + 0.6 * np.sin(2 * np.pi * rng.uniform(4, 12) * t) ** 2
        return (x * rattle).astype(np.float32)

    def file(self, n, rng):
        return fit_length(self.files[rng.integers(len(self.files))], n, rng)

    def __call__(self, n: int, rng: np.random.Generator, n_layers: int | None = None) -> np.ndarray:
        layers = n_layers or rng.integers(1, 3)
        out = np.zeros(n, np.float32)
        for kind in rng.choice(self.kinds, size=layers, replace=False, p=self.p):
            x = getattr(self, kind)(n, rng)
            out += x / (rms(x) + 1e-9) * rng.uniform(0.5, 1.0)
        return out


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    gain = rms(x) / (rms(noise) * 10 ** (snr_db / 20) + 1e-12)
    y = x + noise * gain
    peak = np.abs(y).max()
    return (y / peak * 0.89 if peak > 0.89 else y).astype(np.float32)  # keep <= -1 dBFS


class Augmenter:
    def __init__(self, noise: HospitalNoise, args):
        self.noise, self.a = noise, args

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        a = self.a
        if rng.random() < a.p_speed:  # Kaldi-style speed perturbation (tempo + pitch)
            f = rng.uniform(0.9, 1.1)
            up = 100
            x = resample_poly(x, up, int(round(up * f))).astype(np.float32)
        if rng.random() < a.p_gain:
            x = x * 10 ** (rng.uniform(-6, 6) / 20)
        if rng.random() < a.p_lowpass:
            x = lowpass(x, rng.uniform(3500, 7000))
        if rng.random() < a.p_noise:
            x = add_noise(x, self.noise(len(x), rng), rng.uniform(a.snr_min, a.snr_max))
        return np.clip(x, -1, 1).astype(np.float32)


# ============================================================================ data
def read_split(name: str) -> list[dict]:
    with open(DATA / f"{name}.jsonl", encoding="utf8") as f:
        rows = [json.loads(line) for line in f]
    man = {r["audio_path"]: r for r in csv.DictReader(open(DATA / "manifest.csv", encoding="utf8"))}
    for r in rows:
        m = man[r["audio_filepath"]]
        r.update(utt_id=m["utt_id"], snr_db=float(m["snr_db"]), age=int(m["age"]), gender=m["gender"])
    return rows


class SpeechDataset(Dataset):
    def __init__(self, rows, audio, augment: Augmenter | None, seed: int):
        self.rows, self.audio, self.augment, self.seed = rows, audio, augment, seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.audio[r["audio_filepath"]]
        if self.augment:
            x = self.augment(x, np.random.default_rng((self.seed, self.epoch, i)))
        return dict(audio=x, text=r["text"], mode=r["mode"], idx=i)


class Collator:
    def __init__(self, processor: WhisperProcessor, n_weight: float):
        self.p, self.n_weight = processor, n_weight
        self.sot = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, batch):
        feats = self.p.feature_extractor([b["audio"] for b in batch], sampling_rate=SR, return_tensors="pt")
        lab = self.p.tokenizer([b["text"] for b in batch], padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.sot).all():  # model prepends decoder_start itself
            labels = labels[:, 1:]
        weights = torch.tensor([self.n_weight if b["mode"] == "n" else 1.0 for b in batch])
        return dict(input_features=feats.input_features, labels=labels, weights=weights,
                    idx=torch.tensor([b["idx"] for b in batch]))


def weighted_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    tok = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
    mask = labels.ne(-100).float()
    per_utt = (tok * mask).sum(1) / mask.sum(1).clamp(min=1)
    return (per_utt * weights).sum() / weights.sum()


# ============================================================================ eval
@torch.no_grad()
def transcribe(model, processor, waves: list[np.ndarray], device, bs: int) -> list[str]:
    model.eval()
    out = []
    for i in range(0, len(waves), bs):
        feats = processor.feature_extractor(waves[i : i + bs], sampling_rate=SR, return_tensors="pt")
        seq = model.generate(input_features=feats.input_features.to(device), language="vi",
                             task="transcribe", num_beams=1)
        out += [s.strip() for s in processor.batch_decode(seq, skip_special_tokens=True)]
    return out


def score(rows, hyps) -> dict:
    refs = [normalize(r["text"]) for r in rows]
    hn = [normalize(h) for h in hyps]
    res = {}
    for name, sel in [("all", lambda r: True), ("r", lambda r: r["mode"] == "r"), ("n", lambda r: r["mode"] == "n")]:
        ix = [i for i, r in enumerate(rows) if sel(r)]
        if ix:
            rr, hh = [refs[i] for i in ix], [hn[i] for i in ix]
            res[f"wer_{name}"] = round(jiwer.wer(rr, hh), 4)
            res[f"cer_{name}"] = round(jiwer.cer(rr, hh), 4)
    res["sent_acc"] = round(float(np.mean([a == b for a, b in zip(refs, hn)])), 4)
    return res


@torch.no_grad()
def val_loss(model, loader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        out = model(input_features=b["input_features"].to(device), labels=b["labels"].to(device))
        tot += weighted_loss(out.logits, b["labels"].to(device), torch.ones(len(b["labels"]), device=device)).item() * len(b["labels"])
        n += len(b["labels"])
    return tot / max(n, 1)


# ============================================================================ plots
C1, C2, C3, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "legend.frameon": False, "axes.titleweight": "bold", "font.family": "DejaVu Sans"})


def plot_curves(run: Path, steps: list[dict], epochs: list[dict]) -> None:
    pdir = run / "plots"
    pdir.mkdir(exist_ok=True)
    if steps:
        s = np.array([r["step"] for r in steps])
        loss = np.array([r["train_loss"] for r in steps])
        k = max(1, len(loss) // 30)
        smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(s, loss, color=C1, alpha=0.25, linewidth=1)
        ax.plot(s[k - 1 :], smooth, color=C1, linewidth=2, label="train loss (làm mượt)")
        if epochs:
            ax.plot([e["step"] for e in epochs], [e["val_loss"] for e in epochs], color=C2,
                    linewidth=2, marker="o", markersize=6, label="val loss")
        ax.set(title="Loss", xlabel="step", ylabel="cross-entropy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(pdir / "loss_curve.png", dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(s, [r["lr"] for r in steps], color=C1, linewidth=2)
        ax.set(title="Learning rate", xlabel="step")
        fig.tight_layout()
        fig.savefig(pdir / "lr.png", dpi=130)
        plt.close(fig)
    if epochs:
        e = [r["epoch"] for r in epochs]
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, c, lab in [("val_wer_all", C1, "val WER"), ("val_wer_r", C2, "val WER _r"),
                            ("val_wer_n", C3, "val WER _n")]:
            ax.plot(e, [100 * r[key] for r in epochs], color=c, linewidth=2, marker="o", markersize=6, label=lab)
        ax.set(title="WER trên val theo epoch (epoch 0 = model gốc)", xlabel="epoch", ylabel="WER %")
        ax.legend()
        fig.tight_layout()
        fig.savefig(pdir / "val_wer.png", dpi=130)
        plt.close(fig)


def plot_final(run: Path, final: dict) -> None:
    pdir = run / "plots"
    spk = final["test_by_speaker"]
    names = list(spk)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (key, c, lab) in enumerate([("base", C1, "model gốc"), ("ft", C2, "fine-tuned")]):
        vals = [100 * spk[n][key] for n in names]
        ax.bar(x + (j - 0.5) * 0.38, vals, width=0.36, color=c, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 0.5) * 0.38, v + 0.8, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, names, fontsize=8)
    ax.set(title="WER trên test theo người nói", ylabel="WER %")
    ax.grid(axis="x", visible=False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "test_by_speaker.png", dpi=130)
    plt.close(fig)

    cond = final["test_by_condition"]
    names = list(cond)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (key, c, lab) in enumerate([("base", C1, "model gốc"), ("ft", C2, "fine-tuned")]):
        vals = [100 * cond[n][key] for n in names]
        ax.bar(x + (j - 0.5) * 0.38, vals, width=0.36, color=c, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 0.5) * 0.38, v + 0.8, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, names)
    ax.set(title="WER trên test: sạch và có thêm ồn bệnh viện", ylabel="WER %")
    ax.grid(axis="x", visible=False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "test_by_snr.png", dpi=130)
    plt.close(fig)


# ============================================================================ main
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return ""


def setup_run(args) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = ROOT / "experiments" / "runs" / f"{stamp}_ASR-{args.run_name}"
    for sub in ("plots", "checkpoints", "eval", "aug_samples"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(run / "train.log", encoding="utf8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    return run


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with open(path, "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "PhoASR-whisper-small"))
    ap.add_argument("--data", default="voice_dataset",
                    help="dataset folder under data/processed/ (voice_dataset_v2 = synthetic train, all real test)")
    ap.add_argument("--run-name", default="PhoASR-small-hospitalnoise")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=3, help="epochs without val WER_r improvement")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--no-grad-checkpointing", action="store_true",
                    help="faster but needs ~2x memory (OOM at batch 4 on a 16 GB Mac)")
    ap.add_argument("--modes", default="r,n", help="which takes to train on, e.g. 'r' or 'r,n'")
    ap.add_argument("--n-weight", type=float, default=0.5, help="loss weight for _n takes")
    ap.add_argument("--p-noise", type=float, default=0.85)
    ap.add_argument("--snr-min", type=float, default=0.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--p-speed", type=float, default=0.3)
    ap.add_argument("--p-gain", type=float, default=0.5)
    ap.add_argument("--p-lowpass", type=float, default=0.15)
    ap.add_argument("--spec-augment", type=float, default=0.05, help="mask_time_prob / mask_feature_prob")
    ap.add_argument("--noise-dir", default="", help="extra folder of real ambience wavs (any sr)")
    ap.add_argument("--test-snrs", default="10,5", help="SNRs for the noisy-test evaluation")
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--eval-limit", type=int, default=0, help="cap eval utterances (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    global DATA
    data_p = Path(args.data)
    DATA = data_p if data_p.is_dir() else (ROOT / "data" / "processed" / args.data)

    run = setup_run(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"run dir: {run}")
    log.info(f"device: {device}")

    # ---------------------------------------------------------------- data
    modes = set(args.modes.split(","))
    train = [r for r in read_split("train") if r["mode"] in modes or r["mode"] == "u"]
    val, test, test_new = read_split("val"), read_split("test"), read_split("test_newtext")
    if args.eval_limit:
        val, test, test_new = val[: args.eval_limit], test[: args.eval_limit], test_new[: args.eval_limit]
    audio = {r["audio_filepath"]: load_wav(DATA / r["audio_filepath"]) for r in train + val + test + test_new}
    log.info(f"train {len(train)} utts ({sum(r['duration'] for r in train) / 60:.1f} min) | val {len(val)} | "
             f"test {len(test)} | test_newtext {len(test_new)}")

    files = []
    for f in sorted((ROOT / "sim_vision" / "godot" / "assets" / "sounds").glob("*.wav")):
        files.append(load_wav(f))
    if args.noise_dir:
        files += [load_wav(f) for f in sorted(Path(args.noise_dir).rglob("*.wav"))]
    noise = HospitalNoise([audio[r["audio_filepath"]] for r in train], files)
    log.info(f"noise kinds: {noise.kinds} (recorded ambience files: {len(files)})")
    augment = Augmenter(noise, args)

    rng = np.random.default_rng(args.seed)
    for i, r in enumerate(rng.choice(train, size=min(6, len(train)), replace=False)):
        y = augment(audio[r["audio_filepath"]], np.random.default_rng((args.seed, 999, i)))
        sf.write(run / "aug_samples" / f"{i}_{r['utt_id']}.wav", y, SR)

    # ---------------------------------------------------------------- model
    processor = WhisperProcessor.from_pretrained(args.model, local_files_only=True)
    # labels = <|startoftranscript|><|vi|><|transcribe|><|notimestamps|> text <|endoftext|>
    processor.tokenizer.set_prefix_tokens(language="vietnamese", task="transcribe", predict_timestamps=False)
    model =WhisperForConditionalGeneration.from_pretrained(args.model, local_files_only=True)
    model.config.apply_spec_augment = args.spec_augment > 0
    model.config.mask_time_prob = args.spec_augment
    model.config.mask_feature_prob = args.spec_augment
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    if args.freeze_encoder:
        model.freeze_encoder()
    if not args.no_grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"trainable params: {n_train / 1e6:.1f}M (freeze_encoder={args.freeze_encoder})")

    collate = Collator(processor, args.n_weight)
    train_ds = SpeechDataset(train, audio, augment, args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                          generator=torch.Generator().manual_seed(args.seed))
    val_dl = DataLoader(SpeechDataset(val, audio, None, 0), batch_size=args.eval_batch_size, collate_fn=collate)

    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)

    cfg = dict(vars(args), device=device, run_dir=str(run), git_commit=git_commit(),
               train_utts=len(train), val_utts=len(val), test_utts=len(test),
               steps_per_epoch=steps_per_epoch, total_steps=total_steps, noise_kinds=noise.kinds,
               started=datetime.now().isoformat(timespec="seconds"))
    (run / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf8")
    log.info(f"steps/epoch {steps_per_epoch}, total steps {total_steps}, effective batch "
             f"{args.batch_size * args.grad_accum}")

    # ---------------------------------------------------------------- epoch 0 = baseline on val
    val_waves = [audio[r["audio_filepath"]] for r in val]

    def evaluate_val(epoch: int, step: int, train_loss: float) -> dict:
        t0 = time.time()
        vl = val_loss(model, val_dl, device)
        m = score(val, transcribe(model, processor, val_waves, device, args.eval_batch_size))
        row = dict(epoch=epoch, step=step, train_loss=round(train_loss, 4), val_loss=round(vl, 4),
                   **{f"val_{k}": v for k, v in m.items()}, eval_sec=round(time.time() - t0, 1))
        log.info(f"[epoch {epoch}] train_loss {train_loss:.4f} | val_loss {vl:.4f} | "
                 f"val WER {m['wer_all']:.2%} (_r {m.get('wer_r', float('nan')):.2%}, "
                 f"_n {m.get('wer_n', float('nan')):.2%}) | CER {m['cer_all']:.2%}")
        return row

    step_rows: list[dict] = []
    epoch_rows = [evaluate_val(0, 0, float("nan"))]
    best_wer, best_epoch, bad_epochs = epoch_rows[0]["val_wer_r"], 0, 0
    model.save_pretrained(run / "checkpoints" / "best")
    processor.save_pretrained(run / "checkpoints" / "best")

    # ---------------------------------------------------------------- train
    step, t_last = 0, time.time()
    stop = False
    for epoch in range(1, args.epochs + 1):
        train_ds.epoch = epoch
        model.train()
        run_loss, run_n = 0.0, 0
        opt.zero_grad()
        for i, b in enumerate(train_dl):
            out = model(input_features=b["input_features"].to(device), labels=b["labels"].to(device))
            loss = weighted_loss(out.logits, b["labels"].to(device), b["weights"].to(device))
            (loss / args.grad_accum).backward()
            run_loss += loss.item()
            run_n += 1
            if (i + 1) % args.grad_accum and i + 1 != len(train_dl):
                continue
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            now = time.time()
            step_rows.append(dict(step=step, epoch=round(epoch - 1 + (i + 1) / len(train_dl), 3),
                                  lr=sched.get_last_lr()[0], train_loss=round(loss.item(), 4),
                                  grad_norm=round(gnorm, 3), sec_per_step=round(now - t_last, 2)))
            t_last = now
            if step % args.log_every == 0:
                log.info(f"step {step}/{total_steps} ep {epoch} | loss {loss.item():.4f} | "
                         f"lr {sched.get_last_lr()[0]:.2e} | gnorm {gnorm:.2f} | {step_rows[-1]['sec_per_step']}s/step")
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        row = evaluate_val(epoch, step, run_loss / max(run_n, 1))
        epoch_rows.append(row)
        write_csv(run / "metrics_steps.csv", step_rows)
        write_csv(run / "metrics_epochs.csv", epoch_rows)
        plot_curves(run, step_rows, epoch_rows)
        if row["val_wer_r"] < best_wer - 1e-4:
            best_wer, best_epoch, bad_epochs = row["val_wer_r"], epoch, 0
            model.save_pretrained(run / "checkpoints" / "best")
            processor.save_pretrained(run / "checkpoints" / "best")
            log.info(f"  new best val WER_r {best_wer:.2%} -> checkpoints/best")
        else:
            bad_epochs += 1
            log.info(f"  no improvement ({bad_epochs}/{args.patience}), best {best_wer:.2%} @ epoch {best_epoch}")
        if stop or bad_epochs >= args.patience:
            log.info("early stop" if not stop else "max steps reached")
            break

    model.save_pretrained(run / "checkpoints" / "last")
    processor.save_pretrained(run / "checkpoints" / "last")

    # ---------------------------------------------------------------- final eval: base vs best
    log.info(f"final evaluation: best epoch {best_epoch} (val WER_r {best_wer:.2%}) vs base model")
    snrs = [float(s) for s in args.test_snrs.split(",") if s]
    eval_noise = HospitalNoise([audio[r["audio_filepath"]] for r in train], files)
    conditions = {"sạch": [audio[r["audio_filepath"]] for r in test]}
    for snr in snrs:  # fixed noise per utterance so base and fine-tuned see identical input
        conditions[f"ồn {snr:g} dB"] = [
            add_noise(audio[r["audio_filepath"]], eval_noise(len(audio[r["audio_filepath"]]),
                      np.random.default_rng((7, int(snr), i)), n_layers=2), snr)
            for i, r in enumerate(test)
        ]
    models = {"base": args.model, "ft": str(run / "checkpoints" / "best")}
    final: dict = dict(best_epoch=best_epoch, best_val_wer_r=best_wer, test_by_condition={},
                       test_by_speaker={}, test_newtext={}, summary={})
    preds: dict = {}
    for tag, path in models.items():
        m = WhisperForConditionalGeneration.from_pretrained(path, local_files_only=True).to(device)
        for cname, waves in conditions.items():
            hyps = transcribe(m, processor, waves, device, args.eval_batch_size)
            preds[(tag, cname)] = hyps
            s = score(test, hyps)
            final["test_by_condition"].setdefault(cname, {})[tag] = s["wer_all"]
            final["summary"][f"{tag} | test {cname}"] = s
            log.info(f"  {tag:4s} test {cname:9s} WER {s['wer_all']:.2%} (_r {s['wer_r']:.2%}) CER {s['cer_all']:.2%}")
        hyps = transcribe(m, processor, [audio[r["audio_filepath"]] for r in test_new], device, args.eval_batch_size)
        s = score(test_new, hyps)
        final["test_newtext"][tag] = s
        final["summary"][f"{tag} | test_newtext"] = s
        log.info(f"  {tag:4s} test_newtext    WER {s['wer_all']:.2%} (_r {s.get('wer_r', float('nan')):.2%})")
        del m
        if device == "mps":
            torch.mps.empty_cache()

    for spk in sorted({r["speaker_id"] for r in test}):
        ix = [i for i, r in enumerate(test) if r["speaker_id"] == spk]
        final["test_by_speaker"][spk] = {
            tag: round(jiwer.wer([normalize(test[i]["text"]) for i in ix],
                                 [normalize(preds[(tag, "sạch")][i]) for i in ix]), 4) for tag in models}
    with open(run / "eval" / "test_predictions.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "speaker_id", "mode", "condition", "reference", "base_hyp", "ft_hyp",
                    "base_wer", "ft_wer"])
        for cname in conditions:
            for i, r in enumerate(test):
                ref = normalize(r["text"])
                b, ft = preds[("base", cname)][i], preds[("ft", cname)][i]
                w.writerow([r["utt_id"], r["speaker_id"], r["mode"], cname, r["text"], b, ft,
                            round(jiwer.wer(ref, normalize(b)), 4), round(jiwer.wer(ref, normalize(ft)), 4)])
    final["finished"] = datetime.now().isoformat(timespec="seconds")
    (run / "eval" / "summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf8")
    plot_final(run, final)
    log.info(f"done -> {run}")


if __name__ == "__main__":
    main()
