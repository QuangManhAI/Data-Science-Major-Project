"""Tổng hợp âm thanh nền bệnh viện cho Godot (chỉ để người xem nghe; STT tự trộn nhiễu riêng ở server).

Không dùng âm thanh tải từ ngoài (tránh bản quyền): mọi thứ tạo bằng numpy, riêng tiếng rì rầm lấy từ chính các
clip giọng của dự án (data/processed/wav), lọc thông thấp + vang + chồng nhiều người nên không nghe rõ chữ.
Cùng các loại nhiễu với augment khi fine-tune STT (scripts/train_asr_finetune.py, HospitalNoise): babble, máy lạnh
+ điện lưới 50 Hz, bíp máy theo dõi (gắn ở từng vital_monitor trong Godot), xe đẩy lạch cạch, chuông.

Đầu ra (godot/assets/sounds/):
  hospital_ambience.wav  60 s, vòng lặp liền mạch: máy lạnh ù, người nói xa, chuông loa thông báo, xe đẩy đi
                         ngang, điện thoại reo xa
  monitor_beep.wav       1 s, vòng lặp: một tiếng bíp máy theo dõi (≈ 60 nhịp/phút)

Chạy: python gen_hospital_ambience.py [--seed 7]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR: int = 22050
LOOP_S: float = 60.0
XFADE_S: float = 2.0
ROOT: Path = Path(__file__).resolve().parents[2]
WAV_DIR: Path = ROOT / "data" / "processed" / "wav"
OUT_DIR: Path = Path(__file__).resolve().parents[1] / "godot" / "assets" / "sounds"


def _norm(x: np.ndarray, rms_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-12
    return x * (10 ** (rms_db / 20.0) / rms)


def _filt(x: np.ndarray, kind: str, freq: float | tuple[float, float], order: int = 4) -> np.ndarray:
    sos = signal.butter(order, freq, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def _reverb(x: np.ndarray, rng: np.random.Generator, t60: float = 1.4) -> np.ndarray:
    """Vang hành lang: IR là nhiễu suy giảm mũ (đủ cho tiếng nền, không cần mô phỏng phòng thật)."""
    n = int(SR * t60)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir = _filt(ir, "lowpass", 3500.0, 2)
    wet = signal.fftconvolve(x, ir)[: len(x)]
    return 0.35 * x + _norm(wet, float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)))


def hvac(n: int, rng: np.random.Generator) -> np.ndarray:
    brown = np.cumsum(rng.standard_normal(n))
    brown = _filt(brown - np.mean(brown), "highpass", 20.0, 2)
    rumble = _filt(brown, "lowpass", 280.0)
    t = np.arange(n) / SR
    hum = sum(np.sin(2 * np.pi * h * 50.0 * t) / h for h in (1, 2, 3))  # điện lưới 50 Hz (như augment STT)
    air = _filt(rng.standard_normal(n), "bandpass", (400.0, 2500.0), 2) * 0.15      # gió qua cửa gió
    return _norm(_norm(rumble, -20) + _norm(hum, -38) + _norm(air, -36), -30)


def murmur(n: int, rng: np.random.Generator, n_streams: int = 7) -> np.ndarray:
    """Nhiều người nói xa: chuỗi clip ngẫu nhiên cho mỗi "người", lệch thời gian, lọc + vang."""
    clips = sorted(WAV_DIR.glob("*.wav"))
    if not clips:
        return np.zeros(n)
    out = np.zeros(n)
    for _ in range(n_streams):
        pos = int(rng.uniform(0, 3) * SR)
        stream = np.zeros(n)
        while pos < n:
            audio, sr = sf.read(str(clips[rng.integers(len(clips))]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = signal.resample_poly(audio, SR, sr) if sr != SR else audio
            seg = audio[: n - pos] * rng.uniform(0.4, 1.0)
            stream[pos: pos + len(seg)] += seg
            pos += len(seg) + int(rng.uniform(0.5, 4.0) * SR)
        out += stream
    out = _filt(out, "bandpass", (250.0, 1800.0))  # xa ➔ mất cao tần, không rõ chữ
    return _norm(_reverb(out, rng, 1.6), -36)


def chime(rng: np.random.Generator) -> np.ndarray:
    """Chuông loa thông báo hai nốt (ding-dong)."""
    out = []
    for f in (659.25, 523.25):  # E5 ➔ C5
        t = np.arange(int(0.9 * SR)) / SR
        tone = sum(a * np.sin(2 * np.pi * f * k * t) for k, a in ((1, 1.0), (2, 0.35), (3, 0.12)))
        out.append(tone * np.exp(-3.2 * t) * np.minimum(1.0, t / 0.01))
    x = np.concatenate([out[0][: int(0.45 * SR)], out[1]])
    return _norm(_reverb(x, rng, 1.8), -30)


def phone_ring() -> np.ndarray:
    """Điện thoại bàn reo xa: 2 hồi, mỗi hồi 0.4 s bật / 0.2 s tắt × 2."""
    t = np.arange(int(0.4 * SR)) / SR
    burst = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)) * (np.sin(2 * np.pi * 20 * t) > 0)
    gap = np.zeros(int(0.2 * SR))
    ring = np.concatenate([burst, gap, burst, np.zeros(int(1.8 * SR))])
    ring = np.concatenate([ring, ring])
    return _norm(_filt(ring, "bandpass", (300.0, 2000.0)), -44)


def trolley(rng: np.random.Generator, dur_s: float = 5.0) -> np.ndarray:
    """Xe đẩy đi ngang: nhiễu nâu lọc thấp × rung lạch cạch 4–12 Hz (như HospitalNoise.trolley), to dần rồi nhỏ dần."""
    n = int(dur_s * SR)
    brown = np.cumsum(rng.standard_normal(n))
    x = _filt(brown - np.mean(brown), "lowpass", float(rng.uniform(300, 700)))
    t = np.arange(n) / SR
    rattle = 1 + 0.6 * np.sin(2 * np.pi * rng.uniform(4, 12) * t) ** 2
    clank = _filt(rng.standard_normal(n), "bandpass", (1500.0, 4000.0), 2) * (rng.random(n) < 0.0015)
    passby = np.sin(np.pi * t / dur_s) ** 2
    return _norm(_reverb(x * rattle + 3.0 * clank, rng, 0.9) * passby, -38)


def place(dst: np.ndarray, src: np.ndarray, at_s: float) -> None:
    i = int(at_s * SR)
    j = min(len(dst), i + len(src))
    dst[i:j] += src[: j - i]


def make_loop(rng: np.random.Generator) -> np.ndarray:
    n = int((LOOP_S + XFADE_S) * SR)
    x = hvac(n, rng) + murmur(n, rng)
    place(x, chime(rng), 14.0)
    place(x, chime(rng), 44.5)
    place(x, phone_ring(), 27.0)
    place(x, trolley(rng), 6.0)
    place(x, trolley(rng, 6.5), 34.0)
    # Vòng lặp liền mạch: trộn đuôi XFADE_S giây vào đầu (equal-power) rồi cắt đúng LOOP_S.
    k = int(XFADE_S * SR)
    fade = np.linspace(0.0, np.pi / 2, k)
    head = x[:k] * np.sin(fade) + x[-k:] * np.cos(fade)
    loop = np.concatenate([head, x[k: n - k]])[: int(LOOP_S * SR)]
    return np.clip(loop * (0.89 / (float(np.max(np.abs(loop))) + 1e-9)), -1.0, 1.0)  # chuẩn hóa đỉnh 0.89


def monitor_beep() -> np.ndarray:
    t = np.arange(int(0.11 * SR)) / SR
    env = np.minimum(1.0, t / 0.005) * np.minimum(1.0, (0.11 - t) / 0.02)
    beep = np.sin(2 * np.pi * 980.0 * t) * env * 0.5
    return np.concatenate([beep, np.zeros(SR - len(beep))])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out_dir / "hospital_ambience.wav"), make_loop(rng).astype(np.float32), SR, subtype="PCM_16")
    sf.write(str(args.out_dir / "monitor_beep.wav"), monitor_beep().astype(np.float32), SR, subtype="PCM_16")
    print(f"➔ {args.out_dir}/hospital_ambience.wav ({LOOP_S:.0f} s), monitor_beep.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
