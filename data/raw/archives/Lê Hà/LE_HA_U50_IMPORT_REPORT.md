# Lê Hà U50 Import Report

**Date:** 2026-11-19 · **Verdict: PASS** (see F50 age-flag note below)

## A. Source Information

| ZIP | Declared Gender | Size | Audio Files | Other Members |
|---|---|---|---|---|
| `M50.zip` | Nam | 1,544,718 B | 10 | `metadata.json` |
| `F50.zip` | Nữ | 1,186,608 B | 10 | `metadata.json` |

Embedded `metadata.json` (authoritative, not inferred):

| Subject | realAge | Gender | Region | Province | Prompts |
|---|---|---|---|---|---|
| M50 | **50** | Nam | Bắc | Quảng Ninh | 70 (10 recorded) |
| F50 | **62** | Nữ | Bắc | Quảng Ninh | 70 (10 recorded) |

**Actual container format (verified):** all 20 `.wav` files are **WebM/Opus, 48 kHz, mono** — browser-recorder exports with a `.wav` extension. They are **not** PCM WAV. Files decode cleanly via ffmpeg; `.wav` label is cosmetic. No renaming performed (no established mapping rule per task constraints).

Naming structure observed: `M50_N01..N05_r.wav` + `M50_C01..C05_r.wav`; `F50_N01..N05_r.wav` + `F50_C01..C05_r.wav` except `F50_C01_n.wav` (suffix `n` vs `r` — mode flag consistent with existing repo convention where `n`/`r` distinguish recording modes; left untouched).

## B. Import Summary

| Category | Nam 50 | Nữ 50 |
|---|---|---|
| Total files in ZIP (audio) | 10 | 10 |
| Successfully extracted | 10 | 10 |
| New files | 10 | 10 |
| Duplicate files (raw-MD5 / PCM-MD5) | 0 | 0 |
| Invalid files | 0 (see format note) | 0 (see format note) |
| Extraction errors | 0 | 0 |
| Internal duplicates (Nam50 ↔ Nữ50) | — | 0 |
| Filename conflicts with existing corpus | 0 | 0 |

Destinations:

```
data/raw/archives/Lê Hà/Nam 50/M50.zip   (original preserved)
data/raw/archives/Lê Hà/Nữ 50/F50.zip    (original preserved)
data/raw/real_voice/Lê Hà/Nam 50/M50/{*.wav, metadata.json}
data/raw/real_voice/Lê Hà/Nữ 50/F50/{*.wav, metadata.json}
```

## C. Integrity Verification

- **ZIP integrity:** `unzip -t` — all members CRC-OK in both archives.
- **ZIP SHA-256:**
  - `M50.zip` = `b1a74a71116d125cc471f561d106fbd312f6aecbe5e948a48bf4a17d56894c78`
  - `F50.zip` = `faca24ab0f7f4daedab1325845c235a31e24b3858dbffb1db4addfa0be091501`
- **Reconciliation:** every extracted file byte-identical (MD5) to its ZIP member — 22/22 verified.
- **Raw-MD5 vs existing corpus** (98,945 hashes incl. `audio_inventory.csv` + `dataset_manifest.csv`): **0 duplicates**.
- **Canonical PCM-MD5** (decoded → 16 kHz mono s16; Opus sources required ffmpeg decode): **0 duplicates** vs 96,056 existing PCM hashes; **0 internal** duplicates.
- **Readability:** all 20 files decode via ffmpeg (durations 3.4–7.9 s); none corrupted.
- **No overwrites:** `Nam 50/` and `Nữ 50/` were newly created; zero existing files touched. `metadata.json` preserved alongside audio for lineage.

## D. Final Directory Structure (actual)

```
data/raw/
├── archives/Lê Hà/
│   ├── Nam 50/M50.zip
│   └── Nữ 50/F50.zip
└── real_voice/Lê Hà/
    ├── Nam 20/ … (unchanged)
    ├── Nam 30/ … (unchanged)
    ├── Nam 50/M50/           ← 10 WebM-Opus files (.wav ext) + metadata.json
    ├── Nữ 20/ … (unchanged)
    ├── Nữ 30/ … (unchanged)
    ├── Nữ 50/F50/            ← 10 WebM-Opus files (.wav ext) + metadata.json
    └── Open_Datasets/ … (unchanged)
```

## E. Final Verdict — **PASS**

All required checks completed: ZIP CRC ✓, SHA-256 recorded ✓, byte-identical extraction ✓, readability ✓, zero duplicates ✓, zero overwrites ✓.

**Flags for downstream phases:**
1. **F50 realAge = 62** — placed in `Nữ 50/` per user decision treating U50 as a 50+ umbrella group. Age-62 content now lives under a "50" label; downstream stratification should use `metadata.json` `realAge`, not the folder name.
2. **Files are Opus-in-WebM, not WAV** — any pipeline assuming `.wav` = PCM will fail to read them (libsndfile rejects). Conversion to canonical WAV/16k/PCM16 is deferred to the processing phase per task scope.
3. No transcripts/prompt references included in the zips (`totalPrompts: 70` refers to the prompt set used during recording; only 10 recorded).
