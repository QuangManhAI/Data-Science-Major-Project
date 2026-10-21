# Vuong Decision Log — AI-Assisted Architecture Records

- **Motivation/Background**: Captures architectural, toolchain, and model selection choices made during AI-assisted development to ensure long-term auditability and prevent unrecorded drift.
- **Purpose**: Maintain an immutable, attributable provenance log of decisions for the hospital navigation robot project.
- **Overview Pipeline**: Follows the decision log convention defined in `agents/templates/DECISION_LOG_TEMPLATE.md` and metadata header rules in `agents/rules/MD_CONVENTION.md`.
- **Detailed Plan**: §1 Decision Index Table; §2 Decision Log Entries (D-001 to D-005).
- **References**: [agents/templates/DECISION_LOG_TEMPLATE.md](../../agents/templates/DECISION_LOG_TEMPLATE.md), [agents/rules/MD_CONVENTION.md](../../agents/rules/MD_CONVENTION.md).
- **Created**: 2026-10-21T21:25:00+07:00
- **Last Updated**: 2026-10-21T21:25:00+07:00

---

## 1. Decision Index

| ID | Title | Decided by | Status |
| :--- | :--- | :--- | :---: |
| [D-001](#d-001--unify-python-dependencies-into-root-venv) | Unify Python dependencies into root .venv | human | accepted |
| [D-002](#d-002--store-full-phoasr-whisper-small-model-offline) | Store full PhoASR-whisper-small model offline under models/ | human | accepted |
| [D-003](#d-003--exclude-large-model-weights-from-git-tracking) | Exclude large model weights (>100MB) from Git tracking | human | accepted |
| [D-004](#d-004--enable-pytorch-cuda-acceleration-on-rtx-4060) | Enable PyTorch CUDA acceleration on RTX 4060 GPU | human | accepted |
| [D-005](#d-005--standardize-speech-audio-to-16khz-mono-pcm-wav) | Standardize speech audio to 16kHz mono PCM WAV via PyAV | agent-proposed, human-approved | accepted |

---

## 2. Decision Log Entries

### D-001 — Unify Python dependencies into root .venv
- **Status:** accepted
- **When:** 2026-10-21T19:52:42+07:00
- **Decided by:** human
- **Trigger:** "we keep sim_vision for now,but use root .venv"
- **Decision:** Keep `sim_vision/` directory structure intact, but execute all Python code and tests exclusively via the single root `.venv`.
- **Why:** Eliminates fragmented virtual environments across subfolders and enables IDEs and pytest to discover dependencies globally.
- **Rejected:** Moving `sim_vision/python/*` into `src/` — deferred by user directive; nested `.venv` in subfolders — breaks IDE discovery.
- **Touches:** `requirements/base.txt`, `requirements/dev.txt`, `pytest.ini`, `sim_vision/README.md`
- **Evidence:** `pytest` executed from root discovers and passes all tests; `sim_vision/python/protocol.py` imports cleanly.

### D-002 — Store full PhoASR-whisper-small model offline under models/
- **Status:** accepted
- **When:** 2026-10-21T20:06:06+07:00
- **Decided by:** human
- **Trigger:** "download me this model inprepare for testing voice to text, but the condition is that not loading it using huggface cache, i want the full model on /models"
- **Decision:** Download all 14 model files directly to `models/PhoASR-whisper-small` and load with `local_files_only=True`.
- **Why:** Guarantees offline self-contained reproducibility and eliminates hidden background downloads to `~/.cache/huggingface`.
- **Rejected:** Default Hugging Face cache directory — explicitly forbidden by user; `git clone` with LFS — creates nested `.git` submodules.
- **Touches:** `models/PhoASR-whisper-small/`, `scripts/download_phoasr.py`
- **Evidence:** `python scripts/download_phoasr.py` downloaded 927.6 MB directly into `models/PhoASR-whisper-small`.

### D-003 — Exclude large model weights from Git tracking
- **Status:** accepted
- **When:** 2026-10-21T20:06:24+07:00
- **Decided by:** human
- **Trigger:** "(Recommended) Ignore large weight binaries (e.g. models/**/*.safetensors...)"
- **Decision:** Add `/models/**/*.safetensors`, `/models/**/*.bin`, `/models/**/*.pt` to `.gitignore` while keeping configs, tokenizers, and docs tracked.
- **Why:** `model.safetensors` is 922.2 MB, which violates GitHub's 100 MB per-file push limit.
- **Rejected:** Ignoring the entire `models/PhoASR-whisper-small` directory — loses versioning for architecture configs and tokenizers.
- **Touches:** `.gitignore`
- **Evidence:** `git status -u` confirms `model.safetensors` is ignored while `config.json` and `tokenizer.json` remain tracked.

### D-004 — Enable PyTorch CUDA acceleration on RTX 4060 GPU
- **Status:** accepted
- **When:** 2026-10-21T20:24:18+07:00
- **Decided by:** human
- **Trigger:** "install the gpu version, but also, install all the dependency in requirements for me"
- **Decision:** Install PyTorch 2.6.0+cu124 with CUDA 12.4 support, configure `torch_dll.pth` for Windows DLL resolution, and inherit user site-packages.
- **Why:** Achieves 0.089x real-time inference on the RTX 4060 GPU (11.2× faster than speech) instead of CPU latency (~5–10s/utterance).
- **Rejected:** CPU-only PyTorch — unacceptable latency for real-time robotic conversational interaction.
- **Touches:** `.venv/pyvenv.cfg`, `.venv/Lib/site-packages/torch_dll.pth`, `requirements/base.txt`
- **Evidence:** `python scripts/evaluate_asr.py` processed 70 samples in 28.19s on `cuda` (NVIDIA GeForce RTX 4060 Laptop GPU).

### D-005 — Standardize speech audio to 16kHz mono PCM WAV via PyAV
- **Status:** accepted
- **When:** 2026-10-21T20:35:47+07:00
- **Decided by:** agent-proposed, human-approved
- **Trigger:** "also, the sample are in .m4a, is it the standard format for TTS? if not then convert it"
- **Decision:** Convert all raw `.m4a` files to 16kHz mono 16-bit PCM WAV in `data/processed/real_voice/` using PyAV with corrupt-packet tolerance.
- **Why:** Raw `.m4a` files have header anomalies causing standard loaders to fail; 16kHz mono PCM WAV is the universal standard for Whisper/ASR.
- **Rejected:** Processing raw `.m4a` directly without conversion — failed with `avcodec_send_packet()` on initial packet.
- **Touches:** `scripts/convert_audio.py`, `data/processed/real_voice/*.wav`
- **Evidence:** `python scripts/convert_audio.py` successfully converted all 70 `.m4a` files in 4.0s without data loss.
