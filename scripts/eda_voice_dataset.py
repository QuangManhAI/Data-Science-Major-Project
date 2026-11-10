"""EDA for the STT fine-tuning dataset built by prepare_voice_dataset.py.

Reads data/processed/voice_dataset/manifest.csv, writes figures to
docs/figures/voice_eda/ and the numbers quoted in docs/VOICE_DATASET_EDA.md to
docs/figures/voice_eda/stats.json.

Usage: python scripts/eda_voice_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "voice_dataset" / "manifest.csv"
FIG = ROOT / "docs" / "figures" / "voice_eda"

# Reference palette (dataviz skill, light mode). <=3 categorical series per chart.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#a3a29c"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 9, "font.family": "DejaVu Sans",
    "figure.dpi": 110,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)


def step_hist(ax, data, bins, color, label):
    ax.hist(data, bins=bins, histtype="stepfilled", color=color, alpha=0.18)
    ax.hist(data, bins=bins, histtype="step", color=color, linewidth=2, label=label)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    allr = pd.read_csv(MANIFEST)
    d = allr[allr.usable].copy()
    d["syll"] = d.text.str.split().str.len()  # Vietnamese: one written word = one syllable
    d["rate"] = d.syll / d.speech_s.clip(lower=0.3)
    d["age_grp"] = d.age.map(lambda a: 10 if a <= 12 else 20 if a < 25 else 30)
    d["mode_l"] = d["mode"].map({"r": "_r đọc rõ", "n": "_n tự nhiên", "u": "không nhãn"})
    stats: dict = {}

    # ---- 0. overview
    stats["overview"] = dict(
        recordings_total=len(allr), usable=len(d),
        hours_usable=round(d.duration_s.sum() / 3600, 3),
        speakers_usable=d.speaker_id.nunique(),
        excluded=allr.exclude_reason.fillna("OK").value_counts().to_dict(),
        src_codec=d.src_codec.value_counts().to_dict(),
        src_sr=d.src_sr.astype(str).value_counts().to_dict(),
    )

    # ---- 1. duration distribution
    bins = np.arange(0, 11.5, 0.25)
    fig, ax = plt.subplots(figsize=(8, 4))
    step_hist(ax, d[d["mode"] == "r"].duration_s, bins, S1, "_r đọc rõ")
    step_hist(ax, d[d["mode"] == "n"].duration_s, bins, S2, "_n tự nhiên")
    med = d.duration_s.median()
    ax.axvline(med, color=INK2, linestyle="--", linewidth=1)
    ax.text(med + 0.1, ax.get_ylim()[1] * 0.92, f"median {med:.2f}s", color=INK2, fontsize=9)
    ax.set(title="Độ dài ghi âm theo chế độ thu", xlabel="giây", ylabel="số file")
    ax.legend()
    save(fig, "01_duration_hist.png")
    q = d.duration_s.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).round(2)
    stats["duration"] = dict(
        quantiles={str(k): v for k, v in q.items()}, mean=round(d.duration_s.mean(), 2),
        by_mode=d.groupby("mode").duration_s.agg(["count", "mean", "median"]).round(2).to_dict("index"),
        whisper_padding_ratio=round(1 - d.duration_s.mean() / 30, 3),
        speech_ratio_median=round((d.speech_s / d.duration_s).median(), 3),
    )

    # ---- 2. minutes per speaker (coloured by split)
    sp = (d.groupby("speaker_id").agg(minutes=("duration_s", lambda s: s.sum() / 60),
                                       n=("utt_id", "size"), split=("split", "first"),
                                       gender=("gender", "first"), age=("age", "first"))
            .sort_values("minutes"))
    colors = sp.split.map({"train": S1, "val": S2, "test": S3})
    fig, ax = plt.subplots(figsize=(8, 8.5))
    ax.barh(sp.index, sp.minutes, color=colors, height=0.72)
    for i, (m, n) in enumerate(zip(sp.minutes, sp.n)):
        ax.text(m + 0.08, i, f"{m:.1f}′ · {n}", va="center", fontsize=8, color=INK2)
    ax.set(title="Thời lượng mỗi người nói (màu = split)", xlabel="phút")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (S1, S2, S3)],
              labels=["train", "val", "test"], loc="lower right")
    save(fig, "02_minutes_per_speaker.png")
    stats["speakers"] = sp.round(2).reset_index().to_dict("records")
    top = sp.minutes.sort_values(ascending=False)
    stats["speaker_concentration"] = dict(
        gini=round(float(gini(sp.minutes.values)), 3),
        top5_share=round(top.head(5).sum() / top.sum(), 3),
        min_minutes=round(top.min(), 2), max_minutes=round(top.max(), 2),
    )
    stats["splits"] = (d.groupby("split").agg(utts=("utt_id", "size"),
                                              minutes=("duration_s", lambda s: round(s.sum() / 60, 1)),
                                              speakers=("speaker_id", "nunique")).to_dict("index"))

    # ---- 3. demographics
    demo = d.groupby(["age_grp", "gender"]).duration_s.sum().unstack(fill_value=0) / 60
    reg = d.assign(region=d.region.fillna("không rõ").replace("", "không rõ")).groupby("region").duration_s.sum() / 60
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    x = np.arange(len(demo.index))
    for j, (g, c) in enumerate([("Nam", S1), ("Nữ", S2)]):
        vals = demo.get(g, pd.Series(0, index=demo.index))
        a1.bar(x + (j - 0.5) * 0.38, vals, width=0.36, color=c, label=g)
        for xi, v in zip(x, vals):
            a1.text(xi + (j - 0.5) * 0.38, v + 0.6, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    a1.set_xticks(x, [f"{a} tuổi" if a > 10 else "trẻ em (10)" for a in demo.index])
    a1.set(title="Phút theo nhóm tuổi × giới", ylabel="phút")
    a1.grid(axis="x", visible=False)
    a1.legend()
    reg = reg.sort_values()
    a2.barh(reg.index, reg.values, color=S1, height=0.6)
    for i, v in enumerate(reg.values):
        a2.text(v + 0.6, i, f"{v:.0f}′", va="center", fontsize=8, color=INK2)
    a2.set(title="Phút theo vùng giọng", xlabel="phút")
    a2.grid(axis="y", visible=False)
    save(fig, "03_demographics.png")
    stats["demographics"] = dict(
        minutes_age_gender=demo.round(1).to_dict(), minutes_region=reg.round(1).to_dict(),
        speakers_gender=d.groupby("gender").speaker_id.nunique().to_dict(),
        speakers_age=d.groupby("age_grp").speaker_id.nunique().to_dict(),
        provinces=d.province.replace("", np.nan).dropna().unique().tolist(),
    )

    # ---- 4. loudness before normalisation
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    step_hist(a1, d.speech_rms_dbfs, np.arange(-60, -5, 1.5), S1, "speech RMS")
    a1.axvline(-23, color=INK2, linestyle="--", linewidth=1)
    a1.text(-22.5, a1.get_ylim()[1] * 0.9, "mục tiêu −23", fontsize=9, color=INK2)
    a1.set(title="Âm lượng gốc (RMS phần có tiếng)", xlabel="dBFS", ylabel="số file")
    order = d.groupby("speaker_id").speech_rms_dbfs.median().sort_values().index
    a2.boxplot([d[d.speaker_id == s].speech_rms_dbfs for s in order], vert=True, widths=0.6,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="#cde2fb", color=S1), medianprops=dict(color=INK),
               whiskerprops=dict(color=S1), capprops=dict(color=S1))
    a2.set_xticks(range(1, len(order) + 1), order, rotation=90, fontsize=7)
    a2.set(title="Âm lượng theo người nói", ylabel="dBFS")
    a2.grid(axis="x", visible=False)
    save(fig, "04_loudness.png")
    stats["loudness"] = dict(
        speech_rms_quantiles=d.speech_rms_dbfs.quantile([0.05, 0.5, 0.95]).round(1).to_dict(),
        speaker_median_range=[round(d.groupby("speaker_id").speech_rms_dbfs.median().min(), 1),
                              round(d.groupby("speaker_id").speech_rms_dbfs.median().max(), 1)],
        gain_db_quantiles=d.gain_db.quantile([0.05, 0.5, 0.95]).round(1).to_dict(),
        clipped_files=int((d.clip_ratio > 0.001).sum()),
        clipped_by_speaker=d[d.clip_ratio > 0.001].speaker_id.value_counts().to_dict(),
        peak_limited=int((d.gain_db < (-23 - d.speech_rms_dbfs) - 0.5).sum()),
    )

    # ---- 5. SNR & bandwidth by source
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for codec, c in [("opus", S1), ("aac", S2)]:
        step_hist(a1, d[d.src_codec == codec].snr_db, np.arange(10, 100, 3), c, codec)
    a1.axvline(30, color=INK2, linestyle="--", linewidth=1)
    a1.set(title="SNR ước lượng (p95 − p10 năng lượng khung)", xlabel="dB", ylabel="số file")
    a1.legend()
    for codec, c in [("opus", S1), ("aac", S2)]:
        step_hist(a2, d[d.src_codec == codec].cutoff_hz / 1000, np.arange(3, 8.25, 0.25), c, codec)
    a2.set(title="Băng thông hiệu dụng (ngưỡng −60 dB)", xlabel="kHz", ylabel="số file")
    a2.legend()
    save(fig, "05_snr_bandwidth.png")
    stats["snr"] = dict(
        quantiles=d.snr_db.quantile([0.05, 0.25, 0.5, 0.75]).round(1).to_dict(),
        below_30db=int((d.snr_db < 30).sum()),
        below_30_by_speaker=d[d.snr_db < 30].speaker_id.value_counts().head(8).to_dict(),
        by_codec=d.groupby("src_codec").snr_db.median().round(1).to_dict(),
        cutoff_below_7k=int((d.cutoff_hz < 7000).sum()),
        cutoff_by_speaker=d.groupby("speaker_id").cutoff_hz.median().round(0).sort_values().head(6).to_dict(),
    )

    # ---- 6. leading / trailing silence
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sb = np.arange(0, 4.05, 0.1)
    step_hist(ax, d.lead_sil_s.clip(upper=4), sb, S1, "im lặng đầu")
    step_hist(ax, d.trail_sil_s.clip(upper=4), sb, S2, "im lặng cuối")
    ax.set(title="Khoảng lặng đầu/cuối file", xlabel="giây (cắt ở 4s)", ylabel="số file")
    ax.legend()
    save(fig, "06_silence.png")
    stats["silence"] = dict(
        lead_median=round(d.lead_sil_s.median(), 2), trail_median=round(d.trail_sil_s.median(), 2),
        lead_gt_1s=int((d.lead_sil_s > 1).sum()), trail_gt_1s=int((d.trail_sil_s > 1).sum()),
        lead_lt_0_1s=int((d.lead_sil_s < 0.1).sum()), trail_lt_0_1s=int((d.trail_sil_s < 0.1).sum()),
        silence_share=round(1 - d.speech_s.sum() / d.duration_s.sum(), 3),
        trail_lt_0_1_by_speaker=d[d.trail_sil_s < 0.1].speaker_id.value_counts().head(6).to_dict(),
    )

    # ---- 7. speaking rate by age group x mode
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1, 1.2]))
    groups = [10, 20, 30]
    for j, (m, c, lab) in enumerate([("r", S1, "_r đọc rõ"), ("n", S2, "_n tự nhiên")]):
        data = [d[(d.age_grp == g) & (d["mode"] == m)].rate.values for g in groups]
        pos = np.arange(len(groups)) + (j - 0.5) * 0.36
        ok = [i for i, v in enumerate(data) if len(v)]
        bp = a1.boxplot([data[i] for i in ok], positions=pos[ok], widths=0.3, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK))
        for b in bp["boxes"]:
            b.set(facecolor=c, alpha=0.35, edgecolor=c)
        a1.plot([], [], color=c, linewidth=6, alpha=0.5, label=lab)
    a1.set_xticks(range(3), ["trẻ em", "20 tuổi", "30 tuổi"])
    a1.set(title="Tốc độ nói (âm tiết / giây có tiếng)", ylabel="âm tiết/s")
    a1.grid(axis="x", visible=False)
    a1.legend()
    a2.scatter(d.syll, d.duration_s, s=10, color=S1, alpha=0.35, linewidths=0)
    out = d[(d.rate > d.rate.quantile(0.99)) | (d.rate < d.rate.quantile(0.01))]
    a2.scatter(out.syll, out.duration_s, s=18, color=S2, label="1% ngoài cùng mỗi phía")
    a2.set(title="Số âm tiết transcript vs độ dài", xlabel="âm tiết trong câu kịch bản", ylabel="giây")
    a2.legend()
    save(fig, "07_speaking_rate.png")
    stats["rate"] = dict(
        by_age_mode=d.groupby(["age_grp", "mode"]).rate.median().round(2).unstack().to_dict(),
        overall_median=round(d.rate.median(), 2),
        corr_syll_duration=round(d[["syll", "duration_s"]].corr().iloc[0, 1], 3),
        outliers=out[["utt_id", "syll", "duration_s", "speech_s", "rate"]].round(2).to_dict("records"),
        syll_quantiles=d.syll.quantile([0, 0.5, 1]).to_dict(),
    )

    # ---- 8. pitch
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fb = np.arange(70, 420, 10)
    step_hist(a1, d[(d.age_grp > 10) & (d.gender == "Nam")].f0_hz.dropna(), fb, S1, "nam (người lớn)")
    step_hist(a1, d[(d.age_grp > 10) & (d.gender == "Nữ")].f0_hz.dropna(), fb, S2, "nữ (người lớn)")
    step_hist(a1, d[d.age_grp == 10].f0_hz.dropna(), fb, S3, "trẻ em")
    a1.set(title="Cao độ trung vị mỗi file (F0)", xlabel="Hz", ylabel="số file")
    a1.legend()
    spf0 = d.groupby("speaker_id").agg(f0=("f0_hz", "median"), gender=("gender", "first"),
                                        age=("age_grp", "first")).sort_values("f0")
    col = [S3 if a == 10 else S1 if g == "Nam" else S2 for g, a in zip(spf0.gender, spf0.age)]
    a2.scatter(spf0.f0, range(len(spf0)), color=col, s=30)
    a2.set_yticks(range(len(spf0)), spf0.index, fontsize=7)
    a2.set(title="F0 trung vị theo người nói", xlabel="Hz")
    save(fig, "08_pitch.png")
    pu = d[d.speaker_id.str.startswith("PU_F20")].groupby("speaker_id").f0_hz.describe()[["50%", "25%", "75%"]]
    stats["pitch"] = dict(
        median_by_group=d.groupby(["age_grp", "gender"]).f0_hz.median().round(0).unstack().to_dict(),
        pu_f20_compare=pu.round(1).to_dict("index"),
        male_speakers_f0=spf0[(spf0.gender == "Nam") & (spf0.age > 10)].f0.round(0).to_dict(),
    )

    # ---- 9. text: intents & sentence coverage
    it = d.groupby("intent").agg(n=("utt_id", "size"), minutes=("duration_s", lambda s: s.sum() / 60)).sort_values("n")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    a1.barh(it.index, it.n, color=S1, height=0.65)
    for i, v in enumerate(it.n):
        a1.text(v + 3, i, str(v), va="center", fontsize=8, color=INK2)
    a1.set(title="Số file theo intent", xlabel="file")
    a1.grid(axis="y", visible=False)
    per_sent = d.groupby("text").utt_id.size()
    a2.hist(per_sent, bins=np.arange(0, per_sent.max() + 5, 4), color=S1, edgecolor=SURF, linewidth=2)
    a2.set(title="Số bản ghi trên mỗi câu kịch bản", xlabel="bản ghi / câu", ylabel="số câu")
    save(fig, "09_text_coverage.png")
    tr = set(d[d.split == "train"].text)
    te = set(d[d.split == "test"].text)
    va = set(d[d.split == "val"].text)
    vocab = pd.Series(" ".join(d.text.str.lower().str.replace(r"[^\w\s]", "", regex=True)).split())
    stats["text"] = dict(
        unique_sentences=int(d.text.nunique()), intents=int(d.intent.nunique()),
        per_sentence=per_sent.describe().round(1).to_dict(),
        intent_counts=it.n.to_dict(),
        test_sentences_seen_in_train=round(len(te & tr) / len(te), 3),
        val_sentences_seen_in_train=round(len(va & tr) / len(va), 3),
        vocab_size=int(vocab.nunique()), tokens=int(len(vocab)),
        hapax=int((vocab.value_counts() == 1).sum()),
        by_script=d.code.str.extract(r"^([A-Z])")[0].value_counts().to_dict(),
        mode_counts=d["mode"].value_counts().to_dict(),
    )

    # ---- 10. dup / near-dup check: identical durations within a speaker+prompt pair
    pair = d.pivot_table(index=["speaker_id", "code"], columns="mode", values="duration_s", aggfunc="first")
    if {"r", "n"} <= set(pair.columns):
        pair = pair.dropna(subset=["r", "n"])
        stats["r_vs_n"] = dict(pairs=len(pair), n_longer_share=round(float((pair.n > pair.r).mean()), 3),
                               median_ratio_n_over_r=round(float((pair.n / pair.r).median()), 3),
                               identical_duration_pairs=int((abs(pair.n - pair.r) < 0.01).sum()))

    (FIG / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=str), encoding="utf8")
    print(json.dumps(stats, ensure_ascii=False, indent=1, default=str))


def gini(x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


if __name__ == "__main__":
    main()
