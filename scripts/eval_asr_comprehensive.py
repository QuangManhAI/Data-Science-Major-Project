"""scripts/eval_asr_comprehensive.py

Comprehensive evaluation benchmark for fine-tuned PhoASR-whisper-small models.
Supports:
  - Direct JSONL dataset ingestion (notebooks/data/voice_dataset_v2/test.jsonl or test_newtext.jsonl)
  - Offline local model weights loading (experiments/runs/.../checkpoints/best)
  - Smoke testing mode with --limit <N>
  - Streaming CSV writes and automatic crash-recovery resume (--resume)
  - Vietnamese acoustic text normalization (number-to-words, dialectal synonyms, NFC, lowercase)
  - Full multi-dimensional stratification:
      * mode (_r vs _n)
      * speaker_id (and speaker macro-WER)
      * hospital intent
      * duration buckets (<2s, 2-4s, >4s)
      * digit presence (room/floor numbers)
  - Deep error analysis: Levenshtein S-D-I, top worst predictions, hallucination detection
  - Summary outputs: predictions.csv, summary.json, and REPORT.md

Usage:
  # Smoke test (10 samples, takes ~5 seconds):
  python scripts/eval_asr_comprehensive.py --limit 10 --run-name smoke_test_epoch1

  # Full evaluation on real voice test set (1,726 samples):
  python scripts/eval_asr_comprehensive.py --split test --batch-size 16 --run-name eval_epoch1_test_real

  # Full evaluation on held-out text set (267 samples):
  python scripts/eval_asr_comprehensive.py --split test_newtext --batch-size 16 --run-name eval_epoch1_test_heldout
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "notebooks" / "data" / "voice_dataset_v2"
DEFAULT_MODEL_DIR = (
    ROOT / "experiments" / "runs" / "20261112_033805_ASR-v2-fresh" / "checkpoints" / "best"
)
DEFAULT_OUTPUT_BASE = ROOT / "experiments" / "results"
TARGET_SR = 16000

# ============================================================================
# Vietnamese Number to Words & Text Normalizer
# ============================================================================
DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def num_to_words(n: int) -> str:
    """Vietnamese reading of integers 0-999 for hospital prompts (rooms 410, 304, etc.)."""
    if n < 10:
        return DIGITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        head = "mười" if tens == 1 else f"{DIGITS[tens]} mươi"
        tail = {
            0: "",
            1: " mốt" if tens > 1 else " một",
            4: " tư" if tens > 1 else " bốn",
            5: " lăm",
        }.get(unit, f" {DIGITS[unit]}")
        return head + tail
    if n < 1000:
        hund, rest = divmod(n, 100)
        if rest == 0:
            return f"{DIGITS[hund]} trăm"
        return f"{DIGITS[hund]} trăm " + (
            f"linh {DIGITS[rest]}" if rest < 10 else num_to_words(rest)
        )
    return " ".join(DIGITS[int(c)] for c in str(n))


def normalize_acoustic(text: str) -> str:
    """Acoustic normalization for fair ASR scoring:
    - Unicode NFC normalization
    - Lowercase
    - Converts numbers to words (e.g. '410' -> 'bốn trăm mười')
    - Replaces punctuation with whitespace
    - Dialectal equivalence: 'lẻ' -> 'linh'
    - Collapse repeated whitespace
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\d+", lambda m: f" {num_to_words(int(m.group()))} ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\blẻ\b", "linh", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_raw(text: str) -> str:
    """Basic normalization preserving case and punctuation for raw comparison."""
    return unicodedata.normalize("NFC", text).strip()


# ============================================================================
# Audio Loader
# ============================================================================
def load_audio(path: Path | str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, float]:
    """Load WAV file, downmix to mono, and resample to target_sr if necessary."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    duration = float(len(x)) / float(target_sr)
    return x, duration


# ============================================================================
# Corpus & Breakdown Evaluation Helpers
# ============================================================================
def compute_corpus_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate aggregate WER, CER, Levenshtein S/D/I and Sentence Accuracy."""
    if len(df) == 0:
        return {
            "n": 0,
            "minutes": 0.0,
            "wer": 0.0,
            "cer": 0.0,
            "sub": 0,
            "dele": 0,
            "ins": 0,
            "ref_words": 0,
            "sent_acc": 0.0,
        }

    refs = df["ref_norm"].tolist()
    hyps = df["hyp_norm"].tolist()

    # jiwer process_words
    m = jiwer.process_words(refs, hyps)
    ref_words = sum(len(r.split()) for r in refs)
    sent_acc = float((df["ref_norm"] == df["hyp_norm"]).mean())

    cer_val = float(jiwer.cer(refs, hyps)) if len(refs) > 0 else 0.0

    return {
        "n": len(df),
        "minutes": round(float(df["duration_s"].sum()) / 60.0, 2),
        "wer": round(float(m.wer), 4),
        "cer": round(cer_val, 4),
        "sub": m.substitutions,
        "dele": m.deletions,
        "ins": m.insertions,
        "ref_words": ref_words,
        "sent_acc": round(sent_acc, 4),
    }


def compute_breakdown(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Compute corpus metrics grouped by column."""
    res = {}
    for key, group in df.groupby(col, dropna=False):
        k_str = str(key)
        res[k_str] = compute_corpus_metrics(group)
    return res


# ============================================================================
# Hospital Domain Spot-Check Keywords
# ============================================================================
HOSPITAL_KEYWORDS = [
    "x quang",
    "bảo hiểm",
    "bhyt",
    "siêu âm",
    "khoa sản",
    "khoa nhi",
    "cấp cứu",
    "phòng khám",
    "robot",
    "đăng ký khám",
    "thang máy",
    "nhà thuốc",
]


def spot_check_keywords(df: pd.DataFrame) -> dict[str, Any]:
    """Check recall rate for key hospital domain terminology."""
    results = {}
    for kw in HOSPITAL_KEYWORDS:
        # Check occurrences in ref_norm
        mask = df["ref_norm"].str.contains(rf"\b{kw}\b", regex=True)
        count = int(mask.sum())
        if count == 0:
            continue
        sub_df = df[mask]
        hits = int(sub_df["hyp_norm"].str.contains(rf"\b{kw}\b", regex=True).sum())
        recall = round(hits / count, 4) if count > 0 else 1.0
        results[kw] = {"total_occurrences": count, "hits": hits, "recall": recall}
    return results


# ============================================================================
# Main Evaluation Function
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of fine-tuned PhoASR model on voice test sets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Path to local PhoASR model checkpoint directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Path to dataset root folder containing test.jsonl",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "test_newtext", "val"],
        help="Dataset split to evaluate (test: 1726 utts, test_newtext: 267 utts)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Run identifier name for results folder. If empty, defaults to eval_epoch1_<split>",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="Base directory for output results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for model generation",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for Whisper generation (1 = greedy)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of utterances for smoke testing (0 = full dataset)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic resume from existing predictions.csv",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    data_dir = Path(args.data_dir).resolve()
    run_name = args.run_name if args.run_name else f"eval_epoch1_{args.split}"
    out_dir = Path(args.output_dir).resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PhoASR-whisper-small Comprehensive Evaluation Benchmark")
    print("=" * 80)
    print(f"Model Checkpoint : {model_path}")
    print(f"Dataset Root     : {data_dir}")
    print(f"Target Split     : {args.split}")
    print(f"Batch Size       : {args.batch_size} (Beams: {args.num_beams})")
    print(f"Smoke Test Limit : {args.limit if args.limit > 0 else 'None (Full Benchmark)'}")
    print(f"Output Directory : {out_dir}")

    # 1. Load Dataset Manifest
    jsonl_file = data_dir / f"{args.split}.jsonl"
    if not jsonl_file.exists():
        print(f"ERROR: Dataset file not found at {jsonl_file}", file=sys.stderr)
        sys.exit(1)

    items: list[dict[str, Any]] = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if args.limit > 0:
        items = items[: args.limit]

    total_utts = len(items)
    print(f"Loaded {total_utts} records from {jsonl_file.name}")

    # 2. Check for Resume
    pred_csv_path = out_dir / "predictions.csv"
    existing_preds: dict[str, dict[str, Any]] = {}
    if not args.no_resume and pred_csv_path.exists():
        try:
            prev_df = pd.read_csv(pred_csv_path)
            if "audio_filepath" in prev_df.columns:
                for _, row in prev_df.iterrows():
                    existing_preds[row["audio_filepath"]] = row.to_dict()
                print(f"Found existing predictions.csv with {len(existing_preds)} completed samples.")
                print("Auto-resume activated: skipping already evaluated samples.")
        except Exception as e:
            print(f"Warning: Could not read existing predictions.csv ({e}), starting fresh.")
            existing_preds = {}

    # Filter items that still need evaluation
    remaining_items = [
        it for it in items if it.get("audio_filepath") not in existing_preds
    ]
    print(f"Remaining samples to evaluate: {len(remaining_items)} / {total_utts}")

    # 3. Device & Model Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"\nHardware Device  : {device} ({device_name})")

    if remaining_items:
        print(f"Loading processor & model from {model_path} (local_files_only=True)...")
        t_load = time.time()
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        model = (
            AutoModelForSpeechSeq2Seq.from_pretrained(
                str(model_path), local_files_only=True
            )
            .to(device)
            .eval()
        )
        print(f"Model loaded successfully in {time.time() - t_load:.2f}s.")

    # 4. Evaluation Loop with Streaming CSV
    csv_headers = [
        "audio_filepath",
        "speaker_id",
        "mode",
        "intent",
        "duration_s",
        "infer_s",
        "rtf",
        "text",
        "hyp",
        "ref_norm",
        "hyp_norm",
        "wer_norm",
        "cer_norm",
        "exact_match",
        "has_digit",
        "heldout_text",
        "source",
    ]

    # Write header if file does not exist
    if not pred_csv_path.exists() or len(existing_preds) == 0:
        with open(pred_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(csv_headers) + "\n")

    completed_records: list[dict[str, Any]] = list(existing_preds.values())
    total_eval_time = 0.0

    print("\nStarting evaluation loop...")
    batch_size = args.batch_size
    n_remaining = len(remaining_items)

    for start_idx in range(0, n_remaining, batch_size):
        batch_items = remaining_items[start_idx : start_idx + batch_size]
        cur_batch_size = len(batch_items)

        # Load audio waveforms
        audios = []
        durations = []
        valid_batch_items = []
        for it in batch_items:
            rel_path = it["audio_filepath"]
            full_path = data_dir / rel_path
            if not full_path.exists():
                print(f"Warning: Audio file not found: {full_path}, skipping.", file=sys.stderr)
                continue
            try:
                wav_arr, dur = load_audio(full_path, target_sr=TARGET_SR)
                audios.append(wav_arr)
                durations.append(dur)
                valid_batch_items.append(it)
            except Exception as e:
                print(f"Error loading {full_path}: {e}", file=sys.stderr)
                continue

        if not audios:
            continue

        # Extract features
        feats = processor(
            audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            return_attention_mask=True,
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            sequences = model.generate(
                input_features=feats.input_features.to(device),
                attention_mask=feats.attention_mask.to(device),
                language="vi",
                task="transcribe",
                num_beams=args.num_beams,
                use_cache=True,
                suppress_tokens=None,
            )
        if device == "cuda":
            torch.cuda.synchronize()

        batch_infer_s = time.perf_counter() - t0
        total_eval_time += batch_infer_s
        per_item_infer_s = batch_infer_s / len(audios)

        decoded_hyps = [
            h.strip()
            for h in processor.batch_decode(sequences, skip_special_tokens=True)
        ]

        batch_rows: list[dict[str, Any]] = []
        for it, hyp_text, dur in zip(valid_batch_items, decoded_hyps, durations):
            ref_raw = it.get("text", "")
            ref_norm = normalize_acoustic(ref_raw)
            hyp_norm = normalize_acoustic(hyp_text)

            wer_score = float(jiwer.wer(ref_norm, hyp_norm))
            cer_score = float(jiwer.cer(ref_norm, hyp_norm))
            em = ref_norm == hyp_norm
            item_rtf = round(per_item_infer_s / dur, 4) if dur > 0 else 0.0

            row_data = {
                "audio_filepath": it["audio_filepath"],
                "speaker_id": it.get("speaker_id", "unknown"),
                "mode": it.get("mode", "unknown"),
                "intent": it.get("intent", "unknown"),
                "duration_s": round(dur, 3),
                "infer_s": round(per_item_infer_s, 4),
                "rtf": item_rtf,
                "text": ref_raw,
                "hyp": hyp_text,
                "ref_norm": ref_norm,
                "hyp_norm": hyp_norm,
                "wer_norm": round(wer_score, 4),
                "cer_norm": round(cer_score, 4),
                "exact_match": em,
                "has_digit": bool(re.search(r"\d", ref_raw)),
                "heldout_text": bool(it.get("heldout_text", False)),
                "source": it.get("source", "real"),
            }
            batch_rows.append(row_data)
            completed_records.append(row_data)

        # Streaming append to CSV file
        with open(pred_csv_path, "a", encoding="utf-8") as f:
            for row in batch_rows:
                # Escape commas and quotes for CSV
                vals = []
                for h in csv_headers:
                    val = str(row.get(h, ""))
                    if "," in val or '"' in val or "\n" in val:
                        val = '"' + val.replace('"', '""') + '"'
                    vals.append(val)
                f.write(",".join(vals) + "\n")

        # Progress reporting
        processed_so_far = len(completed_records)
        progress_pct = (processed_so_far / total_utts) * 100.0
        batch_wer = np.mean([r["wer_norm"] for r in batch_rows])
        sample_preview = (
            f"Ref: '{batch_rows[0]['ref_norm'][:35]}...' -> Hyp: '{batch_rows[0]['hyp_norm'][:35]}...'"
            if batch_rows
            else ""
        )
        print(
            f"  [{processed_so_far:>4}/{total_utts}] ({progress_pct:>5.1f}%) | "
            f"Batch infer: {batch_infer_s:.2f}s | "
            f"Batch WER: {batch_wer * 100:>5.2f}% | "
            f"{sample_preview}",
            flush=True,
        )

    # 5. Full Metrics Calculation & Diagnostics
    print("\nAll inferences complete! Calculating stratified statistics & diagnostics...")
    full_df = pd.DataFrame(completed_records)
    if len(full_df) == 0:
        print("ERROR: No records were evaluated.", file=sys.stderr)
        sys.exit(1)

    # Add duration buckets
    full_df["duration_bucket"] = pd.cut(
        full_df["duration_s"],
        bins=[0.0, 2.0, 4.0, 100.0],
        labels=["<2s (ngắn)", "2-4s (vừa)", ">4s (dài)"],
    )

    # Compute aggregate metrics
    overall_metrics = compute_corpus_metrics(full_df)
    total_audio_time = float(full_df["duration_s"].sum())
    total_infer_time = float(full_df["infer_s"].sum())
    corpus_rtf = (
        round(total_infer_time / total_audio_time, 4) if total_audio_time > 0 else 0.0
    )

    # Speaker macro metrics
    speaker_wers = [
        compute_corpus_metrics(g)["wer"] for _, g in full_df.groupby("speaker_id")
    ]
    speaker_cers = [
        compute_corpus_metrics(g)["cer"] for _, g in full_df.groupby("speaker_id")
    ]
    macro_wer_by_speaker = round(float(np.mean(speaker_wers)), 4) if speaker_wers else 0.0
    macro_cer_by_speaker = round(float(np.mean(speaker_cers)), 4) if speaker_cers else 0.0

    # Stratified breakdowns
    by_mode = compute_breakdown(full_df, "mode")
    by_speaker = compute_breakdown(full_df, "speaker_id")
    by_intent = compute_breakdown(full_df, "intent")
    by_duration = compute_breakdown(full_df, "duration_bucket")
    by_digit = compute_breakdown(full_df, "has_digit")
    by_heldout = compute_breakdown(full_df, "heldout_text")

    # Mode 'r' only by speaker
    df_r = full_df[full_df["mode"] == "r"]
    r_only_by_speaker = compute_breakdown(df_r, "speaker_id") if len(df_r) > 0 else {}

    # Keyword spot-check
    keyword_results = spot_check_keywords(full_df)

    # Error analysis: Top 20 worst samples by normalized WER
    worst_samples = (
        full_df.sort_values(by=["wer_norm", "duration_s"], ascending=[False, True])
        .head(20)[
            [
                "audio_filepath",
                "speaker_id",
                "mode",
                "intent",
                "duration_s",
                "wer_norm",
                "cer_norm",
                "text",
                "hyp",
                "ref_norm",
                "hyp_norm",
            ]
        ]
        .to_dict(orient="records")
    )

    # Hallucination check (Insertions > 2 or hyp length > 1.8 * ref length)
    full_df["ref_word_len"] = full_df["ref_norm"].apply(lambda s: len(str(s).split()))
    full_df["hyp_word_len"] = full_df["hyp_norm"].apply(lambda s: len(str(s).split()))
    full_df["len_ratio"] = full_df["hyp_word_len"] / (full_df["ref_word_len"] + 1e-5)

    hallucinations = full_df[
        (full_df["len_ratio"] > 1.8) & (full_df["hyp_word_len"] >= 4)
    ][
        [
            "audio_filepath",
            "speaker_id",
            "mode",
            "len_ratio",
            "ref_norm",
            "hyp_norm",
        ]
    ].to_dict(orient="records")

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(model_path),
        "dataset_split": args.split,
        "device": device,
        "device_name": device_name,
        "num_beams": args.num_beams,
        "batch_size": args.batch_size,
        "total_utterances": len(full_df),
        "total_audio_minutes": round(total_audio_time / 60.0, 2),
        "total_infer_seconds": round(total_infer_time, 2),
        "rtf": corpus_rtf,
        "overall": overall_metrics,
        "macro_wer_by_speaker": macro_wer_by_speaker,
        "macro_cer_by_speaker": macro_cer_by_speaker,
        "by_mode": by_mode,
        "by_speaker": by_speaker,
        "r_only_by_speaker": r_only_by_speaker,
        "by_intent": by_intent,
        "by_duration_bucket": by_duration,
        "by_has_digit": by_digit,
        "by_heldout_text": by_heldout,
        "hospital_keywords_recall": keyword_results,
        "hallucination_candidates_count": len(hallucinations),
        "hallucination_candidates": hallucinations[:10],
        "top_worst_predictions": worst_samples,
    }

    # Write summary.json
    summary_json_path = out_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    # 6. Generate Human-Readable Markdown Report
    report_md_path = out_dir / "REPORT.md"
    o = overall_metrics
    md_content = f"""# Báo cáo Đánh giá Chuyên sâu Mô hình ASR (PhoASR-Whisper-Small)

- **Model Checkpoint:** `{model_path.name}`
- **Tập kiểm thử:** `{args.split}` ({o['n']} câu, {o['minutes']} phút âm thanh)
- **Phần cứng:** `{device_name}` (Batch size: {args.batch_size}, Num beams: {args.num_beams})
- **Thời gian thực thi:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Kết quả Tổng quan & So sánh Baseline

| Chỉ số | Kết quả Checkpoint Epoch 1 | Baseline Chưa Fine-tune | Đánh giá / Tiến bộ |
|:---|:---:|:---:|:---:|
| **WER Chuẩn hóa Tổng thể** | **{o['wer'] * 100:.2f}%** | **19.10%** | {'✅ Giảm ngoạn mục' if o['wer'] < 0.1 else 'Cần cải thiện'} |
| **CER Chuẩn hóa** | **{o['cer'] * 100:.2f}%** | 12.90% | {'✅ Cực kỳ chính xác' if o['cer'] < 0.05 else 'Đạt'} |
| **Tỷ lệ Câu Đúng 100% (Sent Acc)** | **{o['sent_acc'] * 100:.2f}%** | 41.20% | {'✅ Vượt mục tiêu' if o['sent_acc'] > 0.8 else 'Khá'} |
| **Macro WER theo Người nói** | **{macro_wer_by_speaker * 100:.2f}%** | 17.90% | {'✅ Cân bằng giữa các speaker' if macro_wer_by_speaker < 0.1 else 'Đạt'} |
| **Real-Time Factor (RTF)** | **{corpus_rtf:.4f}** | 0.0540 (MPS) | {'✅ Đạt chuẩn Realtime (<0.08)' if corpus_rtf < 0.08 else 'Chậm'} |

### Chi tiết Phân rã Sai số (Levenshtein S-D-I):
- **Thay thế (Substitutions - S):** {o['sub']} từ
- **Nuốt từ (Deletions - D):** {o['dele']} từ
- **Thêm từ / Ảo giác (Insertions - I):** {o['ins']} từ
- **Tổng số từ trong nhãn:** {o['ref_words']} từ

---

## 2. Phân tầng theo Chế độ Thu âm (Mode)

| Chế độ | Số mẫu | WER (%) | CER (%) | Sent Acc (%) | Ghi chú |
|:---|:---:|:---:|:---:|:---:|:---|
| **Đọc chuẩn (`_r`)** | {by_mode.get('_r', by_mode.get('r', {})).get('n', 0)} | **{by_mode.get('_r', by_mode.get('r', {})).get('wer', 0) * 100:.2f}%** | {by_mode.get('_r', by_mode.get('r', {})).get('cer', 0) * 100:.2f}% | {by_mode.get('_r', by_mode.get('r', {})).get('sent_acc', 0) * 100:.2f}% | Phát âm chuẩn theo kịch bản |
| **Tự nhiên (`_n`)** | {by_mode.get('_n', by_mode.get('n', {})).get('n', 0)} | **{by_mode.get('_n', by_mode.get('n', {})).get('wer', 0) * 100:.2f}%** | {by_mode.get('_n', by_mode.get('n', {})).get('cer', 0) * 100:.2f}% | {by_mode.get('_n', by_mode.get('n', {})).get('sent_acc', 0) * 100:.2f}% | Tự nhiên, có nuốt chữ/nói nhanh |

---

## 3. Phân tầng theo Người nói (Speaker Breakdown)

| Speaker ID | Số câu | Thời lượng (phút) | WER (%) | CER (%) | Sent Acc (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
"""
    for spk, data in sorted(by_speaker.items(), key=lambda x: x[1].get("wer", 0)):
        md_content += f"| `{spk}` | {data.get('n', 0)} | {data.get('minutes', 0.0):.1f} | **{data.get('wer', 0) * 100:.2f}%** | {data.get('cer', 0) * 100:.2f}% | {data.get('sent_acc', 0) * 100:.2f}% |\n"

    md_content += f"""
---

## 4. Phân tầng theo Độ dài & Đặc tính Câu hỏi

| Phân nhóm | Số câu | WER (%) | Sent Acc (%) |
|:---|:---:|:---:|:---:|
| **Câu chứa số phòng/tầng (`has_digit = True`)** | {by_digit.get('True', {}).get('n', 0)} | **{by_digit.get('True', {}).get('wer', 0) * 100:.2f}%** | {by_digit.get('True', {}).get('sent_acc', 0) * 100:.2f}% |
| **Câu không chứa số (`has_digit = False`)** | {by_digit.get('False', {}).get('n', 0)} | **{by_digit.get('False', {}).get('wer', 0) * 100:.2f}%** | {by_digit.get('False', {}).get('sent_acc', 0) * 100:.2f}% |
"""
    for dur_lbl, ddata in by_duration.items():
        md_content += f"| **Thời lượng: {dur_lbl}** | {ddata.get('n', 0)} | **{ddata.get('wer', 0) * 100:.2f}%** | {ddata.get('sent_acc', 0) * 100:.2f}% |\n"

    md_content += f"""
---

## 5. Độ nhạy Từ vựng Chuyên ngành Bệnh viện (Domain Lexicon Recall)

| Từ khóa Bệnh viện | Số lần xuất hiện | Số lần nhận đúng | Tỷ lệ Nhận đúng (Recall) |
|:---|:---:|:---:|:---:|
"""
    for kw, kdata in keyword_results.items():
        md_content += f"| `{kw}` | {kdata['total_occurrences']} | {kdata['hits']} | **{kdata['recall'] * 100:.2f}%** |\n"

    md_content += f"""
---

## 6. Phân tích Các Trường hợp Lỗi Nặng Nhất (Top Worst Samples)

| # | Speaker | Mode | WER | Nhãn chuẩn hóa (Reference) | Mô hình nhận dạng (Hypothesis) |
|:---:|:---:|:---:|:---:|:---|:---|
"""
    for idx, w in enumerate(worst_samples[:10], 1):
        md_content += f"| {idx} | `{w['speaker_id']}` | `{w['mode']}` | {w['wer_norm'] * 100:.1f}% | \"{w['ref_norm']}\" | \"{w['hyp_norm']}\" |\n"

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 7. Print Terminal Summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Overall WER (Normalized)  : {o['wer'] * 100:.2f}%")
    print(f"Overall CER (Normalized)  : {o['cer'] * 100:.2f}%")
    print(f"Sentence Exact Match      : {o['sent_acc'] * 100:.2f}%")
    print(f"Speaker Macro WER         : {macro_wer_by_speaker * 100:.2f}%")
    print(f"Real-Time Factor (RTF)    : {corpus_rtf:.4f} (Avg infer: {total_infer_time / max(1, len(full_df)):.3f}s/utt)")
    print(f"Detailed CSV              : {pred_csv_path}")
    print(f"Summary JSON              : {summary_json_path}")
    print(f"Human Report              : {report_md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
