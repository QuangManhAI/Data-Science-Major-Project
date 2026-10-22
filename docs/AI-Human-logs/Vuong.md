# Vuong Decision Log — AI-Assisted Architecture Records

- **Motivation/Background**: Captures architectural, toolchain, and model selection choices made during AI-assisted development to ensure long-term auditability and prevent unrecorded drift.
- **Purpose**: Maintain an immutable, attributable provenance log of decisions for the hospital navigation robot project.
- **Overview Pipeline**: Follows the decision log convention defined in `agents/templates/DECISION_LOG_TEMPLATE.md` and metadata header rules in `agents/rules/MD_CONVENTION.md`.
- **Detailed Plan**: §1 Decision Index Table; §2 Decision Log Entries (D-001 to D-011).
- **References**: [agents/templates/DECISION_LOG_TEMPLATE.md](../../agents/templates/DECISION_LOG_TEMPLATE.md), [agents/rules/MD_CONVENTION.md](../../agents/rules/MD_CONVENTION.md).
- **Created**: 2026-10-21T21:25:00+07:00
- **Last Updated**: 2026-10-21T21:38:00+07:00

---

## 1. Decision Index

| ID | Title | Decided by | Status |
| :--- | :--- | :--- | :---: |
| [D-001](#d-001--unify-python-dependencies-into-root-venv) | Unify Python dependencies into root .venv | human | accepted |
| [D-002](#d-002--store-full-phoasr-whisper-small-model-offline) | Store full PhoASR-whisper-small model offline under models/ | human | accepted |
| [D-003](#d-003--exclude-large-model-weights-from-git-tracking) | Exclude large model weights (>100MB) from Git tracking | human | accepted |
| [D-004](#d-004--enable-pytorch-cuda-acceleration-on-rtx-4060) | Enable PyTorch CUDA acceleration on RTX 4060 GPU | human | accepted |
| [D-005](#d-005--standardize-speech-audio-to-16khz-mono-pcm-wav) | Standardize speech audio to 16kHz mono PCM WAV via PyAV | agent-proposed, human-approved | accepted |
| [D-006](#d-006--initialize-godot-project-cache-and-asset-imports-via-headless-editor) | Initialize Godot project cache and asset imports via headless editor | agent-proposed, human-approved | accepted |
| [D-007](#d-007--cross-platform-windows-path-handling-in-godot-runlogger-tests) | Cross-platform Windows path handling in Godot RunLogger tests | agent-proposed, human-approved | accepted |
| [D-008](#d-008--instantiate-folder-structure-rule-from-template) | Instantiate FOLDER_STRUCTURE.md constitutional rule from template | agent-proposed, human-approved | accepted |
| [D-009](#d-009--download-and-locate-yolov8n-weights-locally) | Download and locate yolov8n.pt weights locally under models/ | agent-proposed, human-approved | accepted |
| [D-010](#d-010--confirm-fsm-proportional-controller-as-pre-rl-baseline) | Confirm FSM proportional controller as pre-RL baseline | human | accepted |
| [D-011](#d-011--document-godot-and-yolo-execution-manual-in-root-readmemd) | Document Godot and YOLO execution manual in root README.md | human | accepted |
| [D-012](#d-012--unified-simulation-launcher-and-websocket-send-resilience) | Unified simulation launcher and WebSocket send resilience | human | accepted |
| [D-013](#d-013--offline-tts-architecture-and-graceful-subtitle-fallback) | Offline TTS architecture and graceful subtitle fallback | human | accepted |

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

### D-006 — Initialize Godot project cache and asset imports via headless editor
- **Status:** accepted
- **When:** 2026-10-21T21:26:00+07:00
- **Decided by:** agent-proposed, human-approved
- **Trigger:** "Parse Error: Identifier 'Config' not declared in the current scope" on initial Godot launch
- **Decision:** Execute `godot --headless --editor --quit --path sim_vision/godot` on initial repository setup.
- **Why:** Godot 4 requires the editor scan to build `.godot/global_script_class_cache.cfg` and convert raw textures/models into `.ctex` and `.res` formats. Without this, class names cannot be resolved.
- **Rejected:** Manually writing `.godot/global_script_class_cache.cfg` — fragile, engine-version specific.
- **Touches:** `sim_vision/godot/.godot/`, `README.md`
- **Evidence:** All 81 GUT tests passed after running headless editor import.

### D-007 — Cross-platform Windows path handling in Godot RunLogger tests
- **Status:** accepted
- **When:** 2026-10-21T21:32:00+07:00
- **Decided by:** agent-proposed, human-approved
- **Trigger:** `test_unwritable_path_reports_once_and_keeps_running` failed on Windows with C++ engine error `Could not create directory '/'`.
- **Decision:** Check `OS.get_name() == "Windows"` and use `"C:/"` instead of POSIX `"/"` for read-only directory testing.
- **Why:** On Windows, `"/"` is an invalid drive syntax that triggers Godot engine internal errors, whereas `"C:/"` is an existing root that is write-protected for non-elevated user processes.
- **Rejected:** Skipping the test on Windows — reduces test coverage across developer OS environments.
- **Touches:** `sim_vision/godot/tests/test_run_logger.gd`
- **Evidence:** Test passed with zero unexpected engine errors.

### D-008 — Instantiate FOLDER_STRUCTURE.md constitutional rule from template
- **Status:** accepted
- **When:** 2026-10-21T21:29:00+07:00
- **Decided by:** agent-proposed, human-approved
- **Trigger:** Root smoke test `test_smoke.py::test_documentation_files_exist` failed looking for `agents/rules/FOLDER_STRUCTURE.md`.
- **Decision:** Copy `agents/templates/CREATE_FOLDER_STRUCTURE_TEMPLATE.md` to `agents/rules/FOLDER_STRUCTURE.md`.
- **Why:** Restores the mandatory constitutional folder layout rule and ensures repository conforms to smoke test specifications.
- **Rejected:** Removing the assertion from `test_smoke.py` — violates rule governance integrity.
- **Touches:** `agents/rules/FOLDER_STRUCTURE.md`
- **Evidence:** `pytest` smoke test passes (44/44 tests passed).

### D-009 — Download and locate yolov8n.pt weights locally under models/
- **Status:** accepted
- **When:** 2026-10-21T21:28:00+07:00
- **Decided by:** agent-proposed, human-approved
- **Trigger:** `yolo_server.py` threw `ERROR: model file not found: yolov8n.pt` and exited.
- **Decision:** Download `yolov8n.pt` (6.2 MB) to `models/yolov8n/yolov8n.pt` and place copy at `sim_vision/python/yolov8n.pt`.
- **Why:** `yolo_server.py` strictly prevents automatic runtime downloads by Ultralytics to ensure reproducible, offline execution.
- **Rejected:** Allowing Ultralytics to auto-download at runtime — explicitly prohibited by project specification (`docs/PROMPT/godot_robot_yolo_prompt.md`).
- **Touches:** `models/yolov8n/yolov8n.pt`, `sim_vision/python/yolov8n.pt`
- **Evidence:** Server starts cleanly and processes frames with 165ms inference latency.

### D-010 — Confirm FSM proportional controller as pre-RL baseline
- **Status:** accepted
- **When:** 2026-10-21T21:36:00+07:00
- **Decided by:** human
- **Trigger:** "it still not RL right? the moving is coded?"
- **Decision:** Confirm that current `controller.gd` is an intentional heuristic / proportional (P) FSM baseline, maintaining an isolated `step()` interface.
- **Why:** Phase 1 establishes the perception bridge and logs ground truth dataset (`sim_vision/logs/run_*.csv`). The clean `step(state, dets, time_since_seen_s)` interface allows direct drop-in replacement by `RLController` (via Godot RL Agents / Stable-Baselines3) in Phase 2.
- **Rejected:** Embedding RL dependencies before establishing simulator ground truth and camera perception pipeline.
- **Touches:** `sim_vision/godot/scripts/controller.gd`, `sim_vision/README.md`
- **Evidence:** Controller successfully navigates between `SEARCH`, `APPROACH`, and `ARRIVED` states using proportional control laws.

### D-011 — Document Godot and YOLO execution manual in root README.md
- **Status:** accepted
- **When:** 2026-10-21T21:40:00+07:00
- **Decided by:** human
- **Trigger:** "wait before that, in README.md documented details on how to start the running of godot and such, model/files to download, command to run"
- **Decision:** Expand root `README.md` to include detailed instructions on downloading model weights (YOLOv8n & PhoASR), running the WebSocket server, launching Godot 3D, and executing tests.
- **Why:** The root `README.md` serves as the central portfolio entry point and execution manual for any human engineer or AI agent cloning the repository.
- **Rejected:** Keeping instructions scattered only in `sim_vision/README.md` — causes developer confusion on initial checkout.
- **Touches:** `README.md`
- **Evidence:** `README.md` updated with exact commands for Windows PowerShell and Linux/macOS.

### D-012 — Unified simulation launcher and WebSocket send resilience
- **Status:** accepted
- **When:** 2026-10-22T07:50:00+07:00
- **Decided by:** human
- **Trigger:** "unififle evreything to run to 1 command/ scripts, because as it right now, i have to run command in 3 sperated tab, and its error prone. WARNING: VisionClient: send failed: Failed"
- **Decision:** Create `run_simulation.py` and `run.ps1` at repo root that orchestrate YOLO (8765), STT/NLU (8766), and Godot with health polling and automatic process cleanup. Guard `_send_frame` in `vision_client.gd` with ready-state verification and clean reconnection on send failures.
- **Why:** Running 3 separate terminal tabs manually is prone to human error and orphaned processes on abort. The launcher verifies that WebSocket servers are actively accepting connections before booting Godot, and terminates all child processes when Godot exits.
- **Rejected:** Running complex Docker Compose stacks on local development machine — incompatible with Godot 3D Vulkan rendering without specialized passthrough.
- **Touches:** `run_simulation.py`, `run.sh`, `run.ps1`, `run.bat`, `.gitattributes`, `sim_vision/godot/scripts/vision_client.gd`, `README.md`
- **Evidence:** `run.sh` (Linux/macOS) and `run.ps1` (Windows) boot all backend servers, verify ports via WebSocket ping, launch Godot across Windows/macOS/Linux paths, and cleanly terminate background servers on exit; 135/135 GUT tests pass.

### D-013 — Offline TTS architecture and graceful subtitle fallback
- **Status:** accepted
- **When:** 2026-10-22T07:55:00+07:00
- **Decided by:** human
- **Trigger:** "Do we have implemet TTS yet? i see the TTS status are disable"
- **Decision:** Document and maintain the offline VieNeu-TTS v3 Turbo subsystem (port 8767) with graceful degradation to 4-second on-screen subtitles when the TTS server or model weights are offline.
- **Why:** Large ONNX speech synthesis weights (~2GB) should remain optional so developers without high storage or specific TTS requirements can still test and run full vision and STT navigation simulation.
- **Rejected:** Hard-failing the simulation when TTS server is unreachable — degrades usability for vision and navigation testing.
- **Touches:** `sim_vision/godot/scripts/tts_client.gd`, `sim_vision/godot/scripts/robot_voice.gd`, `sim_vision/python/tts/tts_engine.py`, `scripts/download_tts.py`, `run_simulation.py`, `docs/guide.md`
- **Evidence:** `scripts/download_tts.py` downloaded VieNeu-TTS v3 Turbo ONNX weights & MOSS codec into `models/`; `vieneu` initialized with 25 preset Vietnamese voices; all 3 servers (YOLO 8765, STT 8766, TTS 8767) launched and connected successfully in unified launcher.
