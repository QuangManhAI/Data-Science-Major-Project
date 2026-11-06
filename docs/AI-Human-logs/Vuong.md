# Vuong Decision Log — AI-Assisted Architecture Records

- **Motivation/Background**: Captures architectural, toolchain, and model selection choices made during AI-assisted development to ensure long-term auditability and prevent unrecorded drift.
- **Purpose**: Maintain an immutable, attributable provenance log of decisions for the hospital navigation robot project.
- **Overview Pipeline**: Follows the decision log convention defined in `agents/templates/DECISION_LOG_TEMPLATE.md` and metadata header rules in `agents/rules/MD_CONVENTION.md`.
- **Detailed Plan**: §1 Decision Index Table; §2 Decision Log Entries (D-001 to D-039).
- **References**: [agents/templates/DECISION_LOG_TEMPLATE.md](../../agents/templates/DECISION_LOG_TEMPLATE.md), [agents/rules/MD_CONVENTION.md](../../agents/rules/MD_CONVENTION.md).
- **Created**: 2026-10-21T21:25:00+07:00
- **Last Updated**: 2026-10-31T12:15:00+07:00

---

## 1. Decision Index

| ID                                                                                                           | Title                                                                                          | Decided by                     |  Status  |
| :----------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- | :----------------------------- | :------: |
| [D-001](#d-001--unify-python-dependencies-into-root-venv)                                                     | Unify Python dependencies into root .venv                                                      | human                          | accepted |
| [D-002](#d-002--store-full-phoasr-whisper-small-model-offline)                                                | Store full PhoASR-whisper-small model offline under models/                                    | human                          | accepted |
| [D-003](#d-003--exclude-large-model-weights-from-git-tracking)                                                | Exclude large model weights (>100MB) from Git tracking                                         | human                          | accepted |
| [D-004](#d-004--enable-pytorch-cuda-acceleration-on-rtx-4060)                                                 | Enable PyTorch CUDA acceleration on RTX 4060 GPU                                               | human                          | accepted |
| [D-005](#d-005--standardize-speech-audio-to-16khz-mono-pcm-wav)                                               | Standardize speech audio to 16kHz mono PCM WAV via PyAV                                        | agent-proposed, human-approved | accepted |
| [D-006](#d-006--initialize-godot-project-cache-and-asset-imports-via-headless-editor)                         | Initialize Godot project cache and asset imports via headless editor                           | agent-proposed, human-approved | accepted |
| [D-007](#d-007--cross-platform-windows-path-handling-in-godot-runlogger-tests)                                | Cross-platform Windows path handling in Godot RunLogger tests                                  | agent-proposed, human-approved | accepted |
| [D-008](#d-008--instantiate-folder-structure-rule-from-template)                                              | Instantiate FOLDER_STRUCTURE.md constitutional rule from template                              | agent-proposed, human-approved | accepted |
| [D-009](#d-009--download-and-locate-yolov8n-weights-locally)                                                  | Download and locate yolov8n.pt weights locally under models/                                   | agent-proposed, human-approved | accepted |
| [D-010](#d-010--confirm-fsm-proportional-controller-as-pre-rl-baseline)                                       | Confirm FSM proportional controller as pre-RL baseline                                         | human                          | accepted |
| [D-011](#d-011--document-godot-and-yolo-execution-manual-in-root-readmemd)                                    | Document Godot and YOLO execution manual in root README.md                                     | human                          | accepted |
| [D-012](#d-012--unified-simulation-launcher-and-websocket-send-resilience)                                    | Unified simulation launcher and WebSocket send resilience                                      | human                          | accepted |
| [D-013](#d-013--offline-tts-architecture-and-graceful-subtitle-fallback)                                      | Offline TTS architecture and graceful subtitle fallback                                        | human                          | accepted |
| [D-014](#d-014--expansion-to-floor-1-vietnamese-hospital-environment)                                         | Expansion to Floor-1 Vietnamese hospital environment (24m x 18m)                               | human                          | accepted |
| [D-015](#d-015--hospital-room-enrichment-medical-pedal-waste-bins-and-refined-collisions)                     | Hospital room enrichment, medical pedal waste bins, and refined collisions                     | human                          | accepted |
| [D-016](#d-016--camera-viewpoint-height-correction-and-automatic-ceiling-culling)                             | Camera viewpoint height correction and automatic ceiling culling                               | agent-proposed, human-approved | accepted |
| [D-017](#d-017--vietnamese-medical-staff-npcs-waiting-patients-and-clinical-facility-props)                   | Vietnamese medical staff NPCs, waiting patients, and clinical facility props                   | human                          | accepted |
| [D-018](#d-018--stt-startup-timeout-resilience-and-rocketbox-surface-material-mapping)                        | STT startup timeout resilience and Rocketbox surface material mapping                          | human                          | accepted |
| [D-019](#d-019--elderly-patient-variants-and-walking-canes-for-old-patient-dominated-hospital)                | Elderly patient variants and walking canes for old-patient-dominated hospital                  | human                          | accepted |
| [D-020](#d-020--implement-phase-1-rl-training-plan-with-shared-team-reward-and-elderly-tether-pacing)         | Implement Phase-1 RL training plan with shared team reward and elderly tether pacing           | human                          | accepted |
| [D-021](#d-021--25-epoch-ppo-training-run-on-cuda-and-10-shift-comprehensive-benchmark)                       | 25-Epoch PPO training run on CUDA and 10-shift comprehensive benchmark                         | human                          | accepted |
| [D-022](#d-022--ad-camera-rotation-and-multi-npc-hospital-speech-support)                                     | A/D camera rotation and multi-NPC hospital speech support                                      | human                          | accepted |
| [D-023](#d-023--fix-seated-patient-arrival-trigger-and-implement-two-stage-doorway-waypoint-navigation)       | Fix seated patient arrival trigger and implement two-stage doorway waypoint navigation         | human                          | accepted |
| [D-024](#d-024--implement-gradio-web-dashboard-for-rl-training-control-live-telemetry-and-history-inspection) | Implement Gradio web dashboard for RL training control, live telemetry, and history inspection | human                          | accepted |
| [D-025](#d-025--fix-godot-3d-patient-escort-speed-and-improve-doorway-navigation-tolerance)                   | Fix Godot 3D patient escort speed and improve doorway navigation tolerance                      | human                          | accepted |
| [D-026](#d-026--contrast-and-readability-overhaul-for-rl-training-dashboard-dark-theme)                       | Contrast and readability overhaul for RL training dashboard dark theme                         | human                          | accepted |
| [D-027](#d-027--fix-subprocess-pipe-buffer-deadlock-and-dynamic-run-dropdown-refresh)                         | Fix subprocess pipe buffer deadlock and dynamic run dropdown refresh                           | AI-pair (Manh + Antigravity)   | accepted |
| [D-028](#d-028--implement-solo-patient-focus-training-environment--godot-3d-solo-mode)                        | Implement Solo Patient Focus training environment & Godot 3D solo mode                          | human                          | accepted |
| [D-029](#d-029--1000-episode-solo-patient-ppo-training-with-45s-early-stagnation-cutoff)                     | 1000-Episode Solo Patient PPO training with 45s early stagnation cutoff                         | human                          | accepted |
| [D-030](#d-030--expand-solo-patient-ppo-training-target-to-5000-episodes)                                    | Expand Solo Patient PPO training target to 5000 episodes                                       | human                          | accepted |
| [D-031](#d-031--hospital-floor-2-expansion-two-way-interactive-elevator-and-multi-floor-rl-navigation) | Hospital Floor 2 expansion, two-way interactive elevator, and multi-floor RL navigation | human-approved via /grill-me | accepted |
| [D-032](#d-032--fix-patient-escort-geometric-deadlock-and-velocity-mismatch-during-inference) | Fix patient escort geometric deadlock and velocity mismatch during inference | human | accepted |
| [D-033](#d-033--natural-patient-navigation-animation-stabilization-t-pose-option-and-multi-npc-dialogue-targeting) | Natural patient navigation, animation stabilization (T-pose option), and multi-NPC dialogue targeting | human | accepted |
| [D-034](#d-034--dynamic-3d-door-labels-and-hospital-department-signage-overhaul) | Dynamic 3D door labels and hospital department signage overhaul | human-approved via /grill-me | accepted |
| [D-035](#d-035--floor-3-hospital-extension-and-multi-floor-dynamic-elevator-navigation) | Floor 3 hospital extension and multi-floor dynamic elevator navigation (X = +100.0m) | human-approved via /grill-me | accepted |
| [D-036](#d-036--robot-inspection-telemetry-hud-panel-with-4-tabbed-progress-bar-visualizations) | Robot inspection telemetry HUD panel with 4-tabbed progress bar visualizations | human-approved via /grill-me | accepted |
| [D-037](#d-037--dedicated-3-floor-patient-escort-ppo-training--distraction-guard) | Dedicated 3-Floor Patient Escort PPO Training & Distraction Guard | human-approved via /grill-me | accepted |
| [D-038](#d-038--escort-only-inference-mode-overlay-hud-escort-label-and-manual-test-hotkey) | Escort-Only inference mode, overlay HUD escort label, and manual test hotkey | human | accepted |
| [D-039](#d-039--fix-solo-patient-targeting-busy-guard-and-dialogue-to-escort-transition) | Fix solo patient targeting, busy guard, and dialogue-to-escort transition | human-approved via /grill-me | accepted |
| [D-040](#d-040--fix-gdscript-stt-tts-connection-property-crash-in-dialogue-turn-completed) | Fix GDScript STT/TTS connection property crash in dialogue turn completed | human-directed log analysis | accepted |
| [D-041](#d-041--fix-escort-patient-trapping-behind-furniture-and-dynamic-trail-pruning) | Fix escort patient trapping behind furniture and dynamic trail pruning | human log observation | accepted |
| [D-042](#d-042--seated-patient-interaction-position-stand-up-transition-and-physics-collision-fix) | Seated patient interaction position, stand-up transition, and physics collision fix | AI-pair (Manh + Antigravity) | accepted |
| [D-043](#d-043--analysis-of-duty-cycle-flaws-oracle-dispatch-vs-visual-rl-and-pure-rl-vision-roadmap) | Analysis of duty cycle flaws, oracle dispatch vs visual RL, and pure visual-search RL roadmap | human-directed inquiry | accepted |
| [D-044](#d-044--grill-me-interview-record-pure-visual-search-rl-architecture-and-new-training-branch) | /grill-me interview record: Pure Visual-Search RL architecture and new training branch | human-approved via /grill-me | accepted |
| [D-045](#d-045--implementation-of-pure-visual-search-rl-environment-training-pipeline-and-hospital-duty-cycle) | Implementation of Pure Visual Search RL Environment, Training Pipeline, and Hospital Duty Cycle | human-approved plan | accepted |
| [D-046](#d-046--root-cause-resolution-for-visual-search-policy-freezing-trap--stagnation-cutoff) | Root-Cause Resolution for Visual Search Policy Freezing Trap & Stagnation Cutoff | human-approved plan | accepted |
| [D-047](#d-047--in-the-wild-multi-npc-training-floor-3-room-dialog-alignment-and-npc-unstuck-evasion) | In-The-Wild Multi-NPC Training, Floor 3 Room Dialog Alignment, and NPC Unstuck Evasion | human-approved plan | accepted |
| [D-048](#d-048--discovery-and-resolution-of-visual-reward-hacking-and-1200-episode-continuation) | Discovery and Resolution of Visual Reward Hacking and 1,200-Episode Continuation | human-directed inquiry | accepted |
| [D-049](#d-049--resolution-of-patient-avoidance-geometry-reward-rebalancing-and-1200-ep-warmstart-training) | Resolution of Patient Avoidance Geometry, Reward Rebalancing, and 1,200-Ep Warmstart Training | human-approved via /grill-me | accepted |
| [D-050](#d-050--interactive-elevator-timing-realistic-door-sensors-and-2000-ep-active-search-continuation) | Interactive Elevator Timing, Realistic Door Sensors, and 2,000-Ep Active Search Continuation | human-approved plan | accepted |
| [D-051](#d-051--diagnostic-investigation-of-training-curve-asymptote-at-2000-episodes) | Diagnostic Investigation of Training Curve Asymptote at 2,000 Episodes | human-directed inquiry | accepted |
| [D-052](#d-052--active-search-incentive-restructuring-stagnation-elimination-and-1000-ep-continuation) | Active Search Incentive Restructuring, Stagnation Elimination, and 1,000-Ep Continuation | human-approved plan | accepted |
| [D-053](#d-053--2000-episode-deep-convergence-training-on-nvidia-rtx-4060-gpu) | 2,000-Episode Deep Convergence Training on NVIDIA RTX 4060 GPU | human | accepted |
| [D-054](#d-054--transition-to-pure-vision-yolov6-only-no-lidar-29-dim-observation-architecture-and-sector-hud) | Transition to Pure Vision (YOLOv6 Only, No LiDAR), 29-Dim Observation Architecture, and Sector HUD | human-aligned via /grill-me | accepted |
| [D-055](#d-055--pure-vision-environment-patches-yolov6-17-class-integration-and-converged-2000-episode-retraining) | Pure Vision Environment Patches, YOLOv6 17-Class Integration, and Converged 2,000-Episode Retraining | human-aligned via /grill-me | accepted |
| [D-056](#d-056--escort-focused-pure-vision-architecture-goal-destination-routing-fix-and-1500-episode-fine-tuning) | Escort-Focused Pure Vision Architecture, Goal Destination Routing Fix, and 1,500-Episode Fine-Tuning | human-aligned via /grill-me | accepted |
| [D-057](#decision-d-057-dynamic-closest-patient-acquisition-and-dual-optical-arrival-trigger-in-godot) | Dynamic Closest-Patient Acquisition and Dual Optical Arrival Trigger in Godot | human-directed | accepted |
| [D-061](#decision-d-061-2000-episode-fine-tuning-execution-on-real-world-sim-to-real-rules) | 2,000-Episode Fine-Tuning Execution on Real-World Sim-to-Real Rules | human-directed | accepted |
| [D-062](#decision-d-062-interactive-non-headless-visual-renderer-and-mock-training-pipeline) | Interactive Non-Headless Visual Renderer and Mock-Training Pipeline | human-directed | accepted |
| [D-063](#decision-d-063-continuous-multi-patient-pure-vision-shift-architecture-and-anti-farming-contract) | Continuous Multi-Patient Pure-Vision Shift Architecture and Anti-Farming Contract | human-aligned via /grill-me | accepted |

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

### D-014 — Expansion to Floor-1 Vietnamese hospital environment (24m x 18m)

- **Status:** accepted
- **When:** 2026-10-22T08:50:00+07:00
- **Decided by:** human
- **Trigger:** "/plan we are planing to expand on the virtural env for our RL sim_vision\\godot, i want you to first understand the current godot, like assets, can you find more hospital related assets that match our Vietnameses style hospital? -make more room, that statisfiles standard hospital -use slide door, that embed between walls, auto open, so that its not complicate our env, thinks of it like the door in rim worlds /grill-me/grill-me"
- **Decision:** Expand simulator from a single `8 m × 6 m` room to a `24 m × 18 m` (ceiling `3.2 m`) Floor-1 Vietnamese hospital with 6 rooms matching `hospital_map.json` (Emergency 101, Blood Lab 110, Internal Medicine 112, Pharmacy, Main Reception Lobby, Elevator & WC) linked by a central East-West corridor. Implement RimWorld-style pocket sliding doors embedded in 0.2m walls with immediate collision disabling upon trigger area entry to ensure collision-free RL raycasts and movement. Replace foreign fire equipment with authentic Vietnamese PCCC (114) station. Add camera mode switcher (`C` key: `FOLLOW`, `TOP_DOWN`, `LOBBY`).
- **Why:** Provides a realistic multi-room navigation environment with authentic Vietnamese hospital visual cues and structured waypoints for upcoming RL navigation and voice guidance benchmark tasks.
- **Rejected:** Physics-swinging hinged doors — prone to trapping the differential drive robot or causing physics jitter in navigation rays.
- **Touches:** `sim_vision/godot/scenes/room.tscn`, `sim_vision/godot/scripts/room.gd`, `sim_vision/godot/scenes/props/sliding_door.tscn`, `sim_vision/godot/scripts/sliding_door.gd`, `sim_vision/godot/scenes/props/pccc_station.tscn`, `sim_vision/godot/scenes/props/hospital_bed.tscn`, `sim_vision/godot/scenes/props/waiting_bench.tscn`, `sim_vision/godot/scenes/props/reception_counter.tscn`, `sim_vision/godot/scenes/props/hospital_sign.tscn`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scenes/main.tscn`, `sim_vision/godot/tests/test_room.gd`, `sim_vision/godot/tests/test_sliding_door.gd`, `sim_vision/godot/tests/test_main.gd`
- **Evidence:** Dedicated branch `feat/hospital-env` verified with 141/141 passing GUT unit tests.

### D-015 — Hospital room enrichment, medical pedal waste bins, and refined collisions

- **Status:** accepted
- **When:** 2026-10-22T09:25:00+07:00
- **Decided by:** human
- **Trigger:** "commits then update 1. some of the rooms lack so much things, add more stuff to it 2. the trash bin looks ugly, that is the kind of bin that been deprecated long ago (i delelte it)and not suitable for hospital env, find the better model 3. i ajust some collision box for better match and moving object"
- **Decision:** Delete deprecated corrugated metal street trash bin (`metal_trash_can`) and replace it with Vietnamese Ministry of Health standard pedal waste bins (`medical_waste_bin.tscn` paired Green domestic + Yellow biohazard bins). Enrich every room with authentic clinical and facility assets (Emergency 101: 3-panel privacy screen, 2 vital signs monitor trolleys with ECG & O2 tanks, supply rack, wheelchair, X-Ray viewer board; Lab 110: optical microscope, chemistry set, phlebotomy desk; Clinic 112: consultation chairs, exam bed privacy screen, vitals trolley; Elevator/WC: double bed-elevator doors with LED indicator, ceramic washbasin vanity with 6-step handwash board & WC stall partition). Preserve human collision box adjustments and assign exact vertex-matched `BoxShape3D` bounds to all models.
- **Why:** Upgrades environment realism to match Vietnamese hospital standards while keeping waypoints clear for robotic navigation and preserving tight collision boundaries.
- **Rejected:** Generic street containers or cluttered non-medical props.
- **Touches:** `sim_vision/godot/scenes/room.tscn`, `sim_vision/godot/scenes/props/medical_waste_bin.tscn`, `sim_vision/godot/scenes/props/medical_trolley.tscn`, `sim_vision/godot/scenes/props/medical_screen.tscn`, `sim_vision/godot/scenes/props/elevator_doors.tscn`, `sim_vision/godot/scenes/props/wc_station.tscn`, `sim_vision/godot/scenes/props/medical_wall_board.tscn`, `scripts/download_hospital_assets.py`, `sim_vision/godot/assets/CREDITS.md`
- **Evidence:** 141/141 tests pass with 10,912 assertions in Godot GUT test suite; committed as `e94381b`.

### D-016 — Camera viewpoint height correction and automatic ceiling culling

- **Status:** accepted
- **When:** 2026-10-22T09:42:00+07:00
- **Decided by:** agent-proposed, human-approved
- **Trigger:** "fix, when tries to run the project, out of 3 view port, only the 1 works, 2 other looks like are too far or such, only see the roof of the hospital"
- **Decision:** Correct camera height and clipping across all 3 view modes: 1) in `FOLLOW` mode, position the camera inside the hospital at height `y = 1.85 m` (under the `3.2 m` ceiling) trailing `2.3 m` behind the robot; 2) in `TOP_DOWN` mode, automatically hide the opaque ceiling mesh (`room.set_ceiling_visible(false)`) so the camera at `y = 18.0 m` has an unobstructed RimWorld-style bird's-eye view into all rooms; 3) in `LOBBY` mode, position the camera at `(-5.0, 2.3, 3.2)` inside the lobby looking across the reception counter, restoring ceiling visibility (`room.set_ceiling_visible(true)`). Update UI help label to reflect active view name dynamically.
- **Why:** Previously, `FOLLOW` was at `y = 4.2 m` and `TOP_DOWN` was at `y = 19.5 m`, both above the opaque ceiling (`y = 3.3 m`), which caused both cameras to view only the external roof.
- **Rejected:** Removing ceiling mesh entirely — ceiling is necessary for first-person robot camera and realistic interior lighting when inside rooms.
- **Touches:** `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/room.gd`, `sim_vision/godot/tests/test_room.gd`
- **Evidence:** All 142/142 GUT tests pass; all 3 camera modes render interior hospital environments without roof clipping.

### D-017 — Vietnamese medical staff NPCs, waiting patients, and clinical facility props

- **Status:** accepted
- **When:** 2026-10-23T08:40:00+07:00
- **Decided by:** human
- **Trigger:** "hey, add more NPC to our env, 2 medical staff (vietnam style, it can be the variant between doctor or nurse) 4 more patient, most of them waiting in the wainting rooms. Overall, we need more assests"
- **Decision:** Add `HospitalNpc` (`sim_vision/godot/scripts/hospital_npc.gd`) and populate `room.tscn` with 2 Vietnamese-style medical staff (`MedicalStaffNurse` with white nurse cap and blue collar trim at reception, `MedicalStaffDoctor` in white lab coat in the corridor), seated waiting patients on lobby/corridor benches, and 8 hospital prop scenes (`queue_kiosk.tscn`, `queue_display.tscn`, `water_dispenser.tscn`, `sanitizer_stand.tscn`, `vital_monitor.tscn`, `iv_stand.tscn`, `stretcher_cart.tscn`, `info_standee.tscn`).
- **Why:** Increases visual realism and obstacle density in the waiting lobby, corridor, Emergency 101, and Clinic 112 while automatically registering `StaticBody3D` footprints with `Room.get_furniture_footprints()`.
- **Rejected:** Unrigged static mannequins without idle animations — lacks lifelike hospital atmosphere.
- **Touches:** `sim_vision/godot/scripts/hospital_npc.gd`, `sim_vision/godot/scenes/room.tscn`, `sim_vision/godot/scenes/props/npc_medical_nurse.tscn`, `sim_vision/godot/scenes/props/npc_medical_doctor.tscn`, `sim_vision/godot/scenes/props/queue_kiosk.tscn`, `sim_vision/godot/scenes/props/queue_display.tscn`, `sim_vision/godot/scenes/props/water_dispenser.tscn`, `sim_vision/godot/scenes/props/sanitizer_stand.tscn`, `sim_vision/godot/scenes/props/vital_monitor.tscn`, `sim_vision/godot/scenes/props/iv_stand.tscn`, `sim_vision/godot/scenes/props/stretcher_cart.tscn`, `sim_vision/godot/scenes/props/info_standee.tscn`
- **Evidence:** `test_has_medical_staff_and_waiting_patients_and_new_props` in `test_room.gd` passes.

### D-018 — STT startup timeout resilience and Rocketbox surface material mapping

- **Status:** accepted
- **When:** 2026-10-23T10:15:00+07:00
- **Decided by:** human
- **Trigger:** "[Launcher] -> STT Server timed out after 60.0s!" / "the human model is broken"
- **Decision:** Increase `wait_for_port` timeout in `run_simulation.py` from `60.0s` to `180.0s` with early `proc.poll()` crash detection, replace slow `AutoModelForSpeechSeq2Seq` lazy discovery in `transcriber.py` with direct `WhisperForConditionalGeneration` and `WhisperProcessor` imports, and map `HospitalNpc` mesh surfaces dynamically by FBX material name (`body`, `head`, `opacity`) using `TRANSPARENCY_ALPHA_HASH` and `CULL_DISABLED` on the hair/eyelash opacity layer.
- **Why:** Concurrent cold-start loading of YOLOv8, VieNeu-TTS ONNX, and PhoASR-whisper-small exceeded 60s on Windows disk/CUDA initialization. Additionally, Rocketbox FBX meshes order surfaces as `[0: body, 1: head, 2: opacity]`, so hardcoding surface 0 as `head` mapped face textures onto the torso and body textures onto the hair/eyelash transparency layer.
- **Rejected:** Hardcoding surface indices — balding avatars have 2 surfaces while standard avatars have 3 surfaces.
- **Touches:** `run_simulation.py`, `sim_vision/python/voice/transcriber.py`, `sim_vision/python/voice/stt_server.py`, `sim_vision/godot/scripts/hospital_npc.gd`
- **Evidence:** Forward+ viewport renders verified clean facial, clothing, and hair rendering across all NPCs; all 3 servers start and pass health checks.

### D-019 — Elderly patient variants and walking canes for old-patient-dominated hospital

- **Status:** accepted
- **When:** 2026-10-23T10:49:00+07:00
- **Decided by:** human
- **Trigger:** "ok, now add more 2 patent but with model of old people, make a few varient of old patient, our hospital will be old patient dominated"
- **Decision:** Expand seated waiting patients from 4 to 6 (`WaitingPatient1` to `WaitingPatient6`), with 5 distinct elderly patient variants (`npc_patient_f02.tscn` silver-haired woman in white sweater, `npc_patient_elder_m03.tscn` silver-haired man in tweed blazer, `npc_patient_elder_f09.tscn` silver-haired grandmother in brown cardigan, `npc_patient_m05.tscn` white-haired bearded man coughing in corridor, and `npc_patient_elder_m14.tscn` balding senior man) plus 3D wooden walking canes (`has_cane = true` in `HospitalNpc`).
- **Why:** Reflects demographic reality where hospital outpatient waiting rooms are predominantly elderly patients with varied attire, hair, and mobility aids.
- **Rejected:** Duplicating a single elderly model across multiple seats — looks repetitive and artificial.
- **Touches:** `sim_vision/godot/scripts/hospital_npc.gd`, `sim_vision/godot/scenes/room.tscn`, `sim_vision/godot/scenes/props/npc_patient_elder_m03.tscn`, `sim_vision/godot/scenes/props/npc_patient_elder_f09.tscn`, `sim_vision/godot/scenes/props/npc_patient_elder_m14.tscn`, `sim_vision/godot/scenes/props/npc_patient_f02.tscn`, `sim_vision/godot/scenes/props/npc_patient_m05.tscn`, `sim_vision/godot/scenes/props/npc_patient_f08.tscn`, `sim_vision/godot/assets/CREDITS.md`, `sim_vision/godot/tests/test_room.gd`
- **Evidence:** `Godot_v4.7.2-stable_win64_console.exe --headless --path sim_vision/godot -s addons/gut/gut_cmdln.gd -gtest=res://tests/test_room.gd -gexit` passes 145/145 tests (12,928 asserts) across all 14 test suites.

### D-020 — Implement Phase-1 RL Training Plan with Shared Team Reward and Elderly Tether Pacing

- **Status:** accepted
- **When:** 2026-10-23T12:54:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment (`rl_training_plan.md`)
- **Trigger:** "ok, draft up a plan for RL traning, we have 1 agents at first, a share reward of how many times it successful help the patient who in trouble, like asking for a place, a information, etc, ... robot should be able to walk patent to the destination they needed given that the location is appropritate, and answer appropritte question successfull or call for staff."
- **Decision:** Implement Phase-1 RL system:
  1. Add 17-ray horizontal 2D LiDAR fan (`200°` FOV, `6.0 m` range) on `Robot` (`robot.gd`).
  2. Implement `RLEnvBridge` (`rl_env_bridge.gd`) and `RLController` (`rl_controller.gd`) with Smart Floor-1 destination routing (`101`, `110`, `112`, `nha_thuoc`, `tiep_don`, `nha_ve_sinh`, elevator `thang_may` for Floor 2–4 requests, and `staff` for emergency/staff calls), coupled elderly patient tether pacing (`0.48 m/s`, `2.5 m` limit), and shared team reward `R_team` (`+10` verbal answer, `+20` staff call, `+30` escort completed).
  3. Implement Python RL training stack under `src/` following `LOGGING_CHECKPOINT_RULES.md`: `RunLogger` (`src/utils/run_logger.py`), full-state checkpointing with `weights_only=True` (`src/utils/checkpoint_utils.py`), Actor-Critic PPO (`src/models/ppo_agent.py`), fast 24m × 18m simulated shift environment (`src/training/hospital_env.py`), and CLI training entry point (`src/training/rl_hospital_train.py` supporting `--epochs`, `--seed`, `--stage`, `--resume`, `--force-resume`, `--tb`, `--smoke`).
  4. Update `run_simulation.py` with `--controller {fsm,rl}` and `--rl-checkpoint` flags.
- **Why:** Delivers robust, resumable, and safe continuous reinforcement learning for multi-patient hospital shifts while keeping the `R_team` reward function ready for multi-agent expansion without redesign.
- **Rejected:** Monolithic end-to-end policy predicting words/intents from raw camera pixels — fragile, slow to train, and discards existing high-accuracy PhoWhisper/PhoBERT pipeline.
- **Touches:** `sim_vision/godot/scripts/robot.gd`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/rl_env_bridge.gd`, `sim_vision/godot/scripts/rl_controller.gd`, `sim_vision/godot/tests/test_rl_env.gd`, `src/utils/run_logger.py`, `src/utils/checkpoint_utils.py`, `src/models/ppo_agent.py`, `src/training/hospital_env.py`, `src/training/rl_hospital_train.py`, `sim_vision/python/test_rl_hospital.py`, `run_simulation.py`, `docs/RL_TRAINING_PLAN.md`
- **Evidence:** All 5 Godot RL tests in `test_rl_env.gd` passed (81 asserts); all 11 `test_robot.gd` tests passed; all 10 `test_main.gd` tests passed; all 7 Python RL tests in `test_rl_hospital.py` passed (100%); CLI smoke training (`--smoke`), `--resume`, and `--force-resume` verified.

### D-021 — 25-Epoch PPO Training Run on CUDA and 10-Shift Comprehensive Benchmark

- **Status:** accepted
- **When:** 2026-10-23T13:12:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment (`rl_full_training_plan.md`)
- **Trigger:** "Full PPO Training Run & Evaluation (Train Phase-1 agent for 20-30 epochs, generate curves, evaluate metrics)"
- **Decision:** Execute 25-epoch PPO training run on CUDA (`NVIDIA GeForce RTX 4060 Laptop GPU`) with 1,200 steps per epoch across the 2-stage curriculum (Stage 1: epochs 1–10 Lobby/Corridor; Stage 2: epochs 11–25 Full 6-room hospital). Run 10 evaluation shifts per stage, plot training curves to `experiments/plots/rl_training_curves.png`, and publish `experiments/results/rl_hospital/evaluation_report.md`.
- **Why:** Confirms empirical convergence of the PPO policy under realistic multi-shift hospital dynamics, validates elderly patient tether regulation (> 86% compliance in Stage 1, > 71% in Stage 2), and provides publication-grade evaluation curves and checkpoint artifacts.
- **Rejected:** Running on CPU without GPU acceleration — would increase wall-clock training time substantially.
- **Touches:** `src/training/rl_hospital_train.py`, `src/training/benchmark_and_plot.py`, `experiments/results/rl_hospital/evaluation_report.md`, `experiments/results/README.md`, `experiments/plots/rl_training_curves.png`
- **Evidence:** 25-epoch training completed in 250s with best validation $R_{\text{team}} = 27.50$ at Epoch 24; 10-shift benchmark completed with mean $R_{\text{team}} = 41.60 \pm 27.08$ and 11 patient escorts in Stage 2.

### D-022 — A/D Camera Rotation and Multi-NPC Hospital Speech Support

- **Status:** accepted
- **When:** 2026-10-23T13:42:00+07:00
- **Decided by:** human
- **Trigger:** "1. when spectating the robot, add A-D key for camera rotations / 2. it seem that, except from the first ever NPC that we create, no other NPC/patient have the ability to speek"
- **Decision:**
  1. Add 360° camera orbit controls via `A` (orbit left/clockwise) and `D` (orbit right/counter-clockwise) keys in `CameraMode.FOLLOW` spectating mode on `Main` (`main.gd`) with smooth delta integration and keypress stepping, updating HUD help text in `main.tscn`.
  2. Extend `HospitalNpc` (`hospital_npc.gd`) to automatically instantiate `Head/Voice` (`AudioStreamPlayer3D`) and an `Npc` dialogue component, mapping female textures to speaker `B` and male textures to speaker `A`.
  3. Generalize `Npc` (`npc.gd`) to accept generic `Node3D` parents, allowing seated patients to initiate conversation without getting off benches while preserving full walk/rotation logic for walking NPCs.
  4. Upgrade `DialogueManager` (`dialogue_manager.gd`) with multi-NPC registration (`register_npc`, `active_npc`), dynamically tracking whichever patient or medical staff member is currently speaking to the robot.
- **Why:** Enables full rotational inspection of robot locomotion and LiDAR navigation from any spectating angle, and activates voice interactions for all hospital patients and staff rather than just the initial test avatar.
- **Rejected:** Hardcoding camera angles — prevents user from checking blind spots behind doors and obstacles.
- **Touches:** `sim_vision/godot/scenes/main.tscn`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/npc.gd`, `sim_vision/godot/scripts/hospital_npc.gd`, `sim_vision/godot/scripts/dialogue_manager.gd`, `sim_vision/godot/scripts/rl_env_bridge.gd`
- **Evidence:** `test_main.gd` passes 10/10 (328 asserts); `test_dialogue.gd` passes 21/21; `test_human.gd` passes 7/7; `test_rl_env.gd` passes 5/5; `test_room.gd` passes 12/12.

### D-023 — Fix Seated Patient Arrival Trigger and Implement Two-Stage Doorway Waypoint Navigation

- **Status:** accepted
- **When:** 2026-10-23T15:45:00+07:00
- **Decided by:** human
- **Trigger:** "sitting patent still not talk, and the RL pathfinding still bad, do more traninig"
- **Decision:**
  1. Fix Seated Patient Arrival Trigger & Dialogue Initialization:
     - In `Main` (`sim_vision/godot/scripts/main.gd`), update `_apply(cmd)`: when approaching a human and safely halted or within `Config.SAFE_DIST_HUMAN + 0.35` (or when velocity is zeroed at safe distance with bounding box height $h \ge 0.45$), transition `state` to `Config.RobotState.ARRIVED`. This solves the issue where seated patients (whose bounding box height is naturally $\approx 0.50$ vs standing $H_{\text{target}} = 0.85$) left the robot stuck in `APPROACH` with $v = 0.0$.
     - Explicitly pass `func() -> int: return state` to all `HospitalNpc` instances during `_setup_hospital_npcs()` in `main.gd` rather than relying on `get_tree().current_scene`.
     - In `HospitalNpc` (`sim_vision/godot/scripts/hospital_npc.gd`), accept optional `state_fn: Callable` in `setup_robot()`.
     - In `Npc` (`sim_vision/godot/scripts/npc.gd`), in `_process()`, if the robot is approaching within $1.15\,\text{m}$ and stopped ($|v| < 0.05\,\text{m/s}$), safely evaluate `r_state = ARRIVED`, triggering `NOTICE` $\to$ `TALK`.
     - Expand `_nearest_person_distance()` in `main.gd` to check all `HospitalNpc` children in `Furniture` (both seated and standing).
  2. Implement Two-Stage Doorway Sub-Goal Waypoint Navigation for RL:
     - In `RLController` (`sim_vision/godot/scripts/rl_controller.gd`) and `HospitalShiftEnv` (`src/training/hospital_env.py`): Implement `route_through_corridor_goal()` with two-stage doorway crossing:
       a) First steer to the corridor doorway approach point $(door_x, -1.0)$ / $(door_x, 1.0)$.
       b) Second steer across the threshold into the room $(door_x, -2.6)$ / $(door_x, 2.6)$ before targeting destination coordinates deep inside the room.
     - Expand 17-ray LiDAR lookahead repulsion from $0.35\,\text{m}$ to $0.65\,\text{m}$ and add dynamic speed regulation (`speed_factor = clampf(center_min / 0.65, 0.35, 1.0)`), eliminating wall clipping and doorframe snags.
     - Match `teacher_action()` in `hospital_env.py` to provide high-quality safe navigation trajectories during behavior cloning warm-up.
- **Why:** Ensures all hospital patients (especially seated elderly patients on benches) immediately engage in voice dialogue upon robot arrival, and eliminates pathfinding collisions across room doorways.
- **Rejected:** Lowering global $H_{\text{target}}$ to 0.50 — causes robot to stop too far from standing patients ($> 2.5\,\text{m}$).
- **Touches:** `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/hospital_npc.gd`, `sim_vision/godot/scripts/npc.gd`, `sim_vision/godot/scripts/rl_controller.gd`, `src/training/hospital_env.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 151/151 GUT unit tests passing (13,015 asserts, 0 failures); 30-epoch PPO training run (`PPOHospital-doorway-curriculum`) launched on RTX 4060 GPU with clean doorway clearance.

### D-024 — Implement Gradio Web Dashboard for RL Training Control, Live Telemetry, and History Inspection

- **Status:** accepted
- **When:** 2026-10-23T16:45:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment (`rl_ui_dashboard_plan.md`)
- **Trigger:** "make me a UI that have those funtionality / 1. it can start the trainig either new or contnue from a previous runs, flags can be tick using tick box / 2. when training, its show in realtime, episode, avg reward, times, in grahp etc... / 3. when continuse the previous run, it load the previous run training grahpts, stats etc .."
- **Decision:**
  1. Build a Gradio 6.30 Web Dashboard (`src/ui/rl_dashboard.py`) and single-command launcher (`run_dashboard.py`).
  2. Implement decoupled background subprocess execution (`start_training`, `stop_training`) streaming real-time metrics and logs with clean SIGINT/SIGTERM handling to preserve checkpoints.
  3. Implement historical run inspection and dropdown loader (`find_existing_runs`, `inspect_run`) immediately rendering 4 multi-line charts (Reward, Loss, Productivity, Safety) and KPI cards from any past run under `experiments/runs/`.
  4. Provide tick boxes and intuitive widgets for all training flags (`--device cuda`, `--resume`, `--force-resume`, `--tb`, `--smoke`, `--stage`).
  5. Add Godot 3D quick-launch shortcut to directly spin up `run_simulation.py --controller rl --rl-checkpoint ...` with the selected run's checkpoint.
- **Why:** Replaces manual terminal command typing with a modern visual controller, provides live visual feedback during continuous reinforcement learning, and allows instant comparative inspection of historical training runs without leaving the browser.
- **Rejected:** Native Tkinter GUI (limited modern web-grade charting, desktop-only) or custom full-stack React app (unnecessary engineering overhead when Gradio 6.30 is already installed in root `.venv`).
- **Touches:** `src/ui/rl_dashboard.py`, `run_dashboard.py`, `tests/test_dashboard.py`, `README.md`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 4/4 automated tests in `tests/test_dashboard.py` passed; `run_dashboard.py --help` verified; all 151 Godot GUT tests pass; verified loading of 80-epoch converged run (`PPOHospital-converged-80ep-full`) with peak $R_{\text{team}} = 103.0$ and 4.0 patients helped/shift.

### D-025 — Fix Godot Vision-Client Inference Error (`robot.v` / `robot.omega` Access)

- **Status:** accepted
- **When:** 2026-10-23T17:00:00+07:00
- **Decided by:** AI-pair (Manh + Antigravity)
- **Trigger:** SCRIPT ERROR during YOLO vision inference in Godot: `Invalid access to property or key 'v' on a base object of type 'CharacterBody3D (Robot)'. at: Main._on_detections (res://scripts/main.gd:359)`
- **Decision:**
  1. Add GDScript property getters and setters for `v` and `omega` on `Robot` in `sim_vision/godot/scripts/robot.gd` mapping cleanly to internal `_v`, `_omega` and `set_command()`.
  2. In `sim_vision/godot/scripts/main.gd:359`, retrieve command via `robot.get_command() if robot.has_method("get_command") else Vector2(robot.v, robot.omega)` for defensive execution during RL guide/escort phase.
- **Why:** Prevents runtime GDScript crashes during YOLO vision inference loops when logging telemetry commands.
- **Touches:** `sim_vision/godot/scripts/robot.gd`, `sim_vision/godot/scripts/main.gd`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** All 151 Godot GUT unit tests pass without error (13,015 asserts, 0 failures).

### D-026 — Revamp Dashboard Dark Theme Contrast and Chart Legibility

- **Status:** accepted
- **When:** 2026-10-23T17:08:00+07:00
- **Decided by:** AI-pair (Manh + Antigravity)
- **Trigger:** "the style of the dashboard need to be make more easy to read, in dark theme i can not see the some number"
- **Decision:**
  1. Refactor Matplotlib styling in `generate_metric_plots` using a high-contrast dark telemetry theme (`#0f172a` figure, `#1e293b` axes, `#f8fafc` titles, `#cbd5e1` tick numbers and labels, `#334155` grid lines, glowing neon curves `#38bdf8`, `#34d399`, `#fb923c`, `#f87171`).
  2. Replace hardcoded light CSS in `src/ui/rl_dashboard.py` with `DASHBOARD_CSS` ensuring KPI numbers are rendered in bold glowing electric cyan (`#38bdf8`) on deep slate card containers (`#0f172a` / `#1e293b`) with crisp silver titles (`#94a3b8`).
  3. Ensure numbers in Gradio sliders, numeric inputs, and textboxes use high-contrast cyan/white fonts (`#38bdf8` / `#f8fafc`) in dark theme.
  4. Inject `DASHBOARD_CSS` directly into the DOM tree via `gr.HTML` and `launch(css=DASHBOARD_CSS)` with 0 warnings.
- **Why:** Resolves washed-out and dark-on-dark invisible numbers in Gradio's dark theme, delivering maximum legibility for all telemetry values, sliders, and charts.
- **Touches:** `src/ui/rl_dashboard.py`, `tests/test_dashboard.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 5/5 pytest tests in `tests/test_dashboard.py` pass cleanly with 0 warnings; all 9/9 repo tests pass.

### D-027 — Fix Subprocess Pipe Buffer Deadlock and Dynamic Run Dropdown Refresh

- **Status:** accepted
- **When:** 2026-10-23T20:36:00+07:00
- **Decided by:** AI-pair (Manh + Antigravity)
- **Trigger:** "i order for 120 epoch, why it stop at 16? I can not even find it in the continuation list experiments\runs\20261023_193057_PPOHospital-interactive"
- **Decision:**
  1. Root cause 1: `start_training` spawned `rl_hospital_train.py` with `stdout=subprocess.PIPE` without draining the OS pipe buffer in Python. After 16 epochs of stdout text, Windows kernel pipe buffer (4–64 KB) filled up and blocked `sys.stdout.write(...)` in kernel `NtWriteFile` (Thread WaitReason: Executive). Fixed by introducing asynchronous daemon reader thread `_drain_stdout` continuously draining lines into `collections.deque(maxlen=300)`.
  2. Root cause 2: `on_mode_change` in `rl_dashboard.py` previously only toggled visibility without re-scanning runs. Fixed by dynamically calling `find_existing_runs()` and updating `run_dropdown` choices, default selection, and auto-invoking `inspect_run` to immediately load charts, KPIs, and checkpoints when switching to "Continue from Previous Run".
  3. Support explicit `--resume-from <dir>` CLI flag in `src/training/rl_hospital_train.py` and pass exact run folder from dashboard.
  4. Updated `find_existing_runs()` to compute peak validation reward across all epochs rather than just the final epoch.
- **Why:** Prevents multi-epoch training deadlocks and ensures all active and past runs are immediately selectable and visible in the continuation UI without requiring server restarts.
- **Touches:** `src/ui/rl_dashboard.py`, `src/training/rl_hospital_train.py`, `tests/test_dashboard.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** Verified clean resume of `20261023_193057_PPOHospital-interactive` from epoch 16 to 17; 9/9 pytest tests pass; verified `find_existing_runs` dynamically lists `20261023_193057_PPOHospital-interactive (Ep 16 | Peak R: 53.0 | Help: 0.5)`.


### D-028 — Implement Solo Patient Focus Training Environment & Godot 3D Solo Mode
- **Status:** accepted
- **When:** 2026-10-24T14:12:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment (`rl_solo_patient_env_plan.md`)
- **Trigger:** "make me a training env, in that, each times, there only 1 patient not doctor appear, the robot spawn nearby, its task is to answer patient question, guide them to a destination or call for doctor, the env is the current hospital env, each times a patient problems got resolve, apear another one at random location (that not in the medical room), traning on that, instead of the full hospital with full flesh doctor and many patient"
- **Decision:**
  1. In `src/training/hospital_env.py`: Added `env_mode: str = "solo"` support to `HospitalShiftEnv`. In solo mode:
     - Patrolling doctor dynamic obstacle is completely disabled (`doctor_active = False`).
     - Patients strictly spawn in public hallways/corridors or main lobby (`_sample_patient_spawn_solo()`), never inside medical exam rooms.
     - On episode reset, the robot spawns in close proximity (2.0–3.5 m) facing the patient with direct line of sight.
     - When a patient task completes, the next patient spawns in a random corridor/lobby spot while the robot maintains its current position without teleporting.
     - Task distribution configured to escort-heavy: ~70% Guide/Escort, ~15% Question Answering, ~15% Call Staff/Doctor.
  2. In `src/training/rl_hospital_train.py`: Added `--env-mode {solo, full}` CLI argument (defaulting to `"solo"`), wired through training and validation loops, and saved in run metadata.
  3. In `sim_vision/godot/scripts/main.gd` & `run_simulation.py`: Added `--solo-patient` runtime flag. In solo mode, Godot disables extra scene NPCs/doctors, spawns the robot nearby the patient on reset, and respawns the patient at public waypoints without robot teleportation upon task completion.
  4. In `src/ui/rl_dashboard.py`: Added "🏥 Environment Mode" dropdown selector defaulting to `Solo Patient Focus (1 Patient, Nearby Spawn)` and passed appropriate flags to training and 3D simulation launch commands.
- **Why:** Eliminates multi-NPC navigation distractions during early policy learning, focuses learning capacity on core patient interaction, tethered escorting, and doorway navigation from nearby spawns.
- **Touches:** `src/training/hospital_env.py`, `src/training/rl_hospital_train.py`, `run_simulation.py`, `sim_vision/godot/scripts/main.gd`, `src/ui/rl_dashboard.py`, `tests/test_solo_env.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 4/4 tests in `tests/test_solo_env.py` pass; all 13 repo pytest tests pass; all 151 Godot GUT unit tests pass (13,015 asserts, 0 failures); verified 1-epoch GPU smoke test on RTX 4060 with `--env-mode solo`.


### D-029 — 1000-Episode Solo Patient PPO Training with 45s Early Stagnation Cutoff
- **Status:** accepted
- **When:** 2026-10-24T14:35:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment (`train_solo_1000ep_plan.md`)
- **Trigger:** "train for me 1000 episode, each episode terminate when a certain number of times haves pass and it dont complete anymore task /grill-me monitor in real time via terminal logs when running/grill-me"
- **Decision:**
  1. In `src/training/hospital_env.py`: Added `stagnation_timeout_s: float = 0.0` and `stagnation_penalty: float = -10.0` configuration to `HospitalShiftEnv`. When `stagnation_timeout_s > 0` (45.0s = 225 steps at dt=0.2s), an internal counter `steps_since_task` increments each step and resets to 0 whenever a task is completed (`answer`, `call_staff`, `guide_completed`). If 225 steps elapse without completing any task, the episode immediately terminates (`done = True`, `exit_reason = "STAGNATION_TIMEOUT"`) and applies `-10.0` terminal penalty.
  2. Implemented dedicated trainer `src/training/train_solo_1000ep.py` managing a discrete 1000-episode training loop (Ep 1 to 1000) warm-started from the best converged 80-epoch checkpoint `PPOHospital-converged-80ep-full_best.pt` ($R_{team} = 103.0$).
  3. Action execution blends teacher guidance decaying linearly from `prior_weight = 0.10` to `0.0` over the first 200 episodes.
  4. PPO optimization is batched every 5 episodes (200 updates total) over 4 epochs per batch with AdamW (lr: 1e-4 -> 3e-5 via CosineAnnealingLR) and grad norm clipping 0.5.
  5. Live single-line log flushed to terminal after each episode: `[EP 0142/1000] Tasks: 4 | R_team: +42.0 | Dur: 36.8s (184 stp) | Exit: COMPLETED | Col: 1 | Roll20: +33.5 | Peak: +60.0 | ETA: 28.4m`.
  6. Checkpointing: Saves `PPOHospital-solo-1000ep_best.pt` on rolling-20 reward all-time high and `PPOHospital-solo-1000ep_last.pt` every 50 episodes.
- **Why:** Prevents wasting compute wandering when robot gets stuck or loses tether during solo patient interaction, strongly shapes the policy to act decisively to resolve tasks within 45 seconds, and provides full transparency through real-time terminal output.
- **Touches:** `src/training/hospital_env.py`, `src/training/train_solo_1000ep.py`, `tests/test_stagnation_env.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 3/3 tests in `tests/test_stagnation_env.py` pass; all 16 repo pytest tests pass; verified 10-episode GPU smoke run and initiated 1000-episode training on RTX 4060.


### D-030 — Expand Solo Patient PPO Training Target to 5000 Episodes
- **Status:** accepted
- **When:** 2026-10-24T15:07:00+07:00
- **Decided by:** human
- **Trigger:** "increase tje ep to 5000"
- **Decision:**
  1. The initial 1000-episode milestone concluded successfully at 1000 episodes (Peak $R_{team} = +234.0$, 1,036 patients helped, 28 full shifts completed).
  2. Stabilized advantage normalization in `src/training/train_solo_1000ep.py` using whole-batch `t_adv.std(unbiased=False)` and mini-batch size guards to eliminate single-element sample variance NaN hazards.
  3. Extended training target from 1000 to 5000 episodes (`--episodes 5000`), seamlessly resuming from checkpoint `PPOHospital-solo-1000ep_last.pt` at Episode 1001.
  4. Dynamically scaled `CosineAnnealingLR` over the remaining 4,000 episodes ($1.0 \times 10^{-4} \to 3.0 \times 10^{-5}$) with periodic checkpoint saving every 100 episodes.
- **Why:** Deepens policy convergence, solidifies complex multi-doorway escorting trajectories, and pushes the policy toward mastering back-to-back patient navigation with near-zero stagnation.
- **Touches:** `src/training/train_solo_1000ep.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** Clean resume from Episode 1001 running live under background task `task-3198` on RTX 4060 GPU with verified real-time logging and zero warnings.

### D-031 — Hospital Floor 2 Expansion, Two-Way Interactive Elevator, and Multi-Floor RL Navigation
- **Status:** accepted
- **When:** 2026-10-25T21:40:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment (`floor2_elevator_plan.md`)
- **Trigger:** "we lack the rooms for floor two, create it, make it a seperate parts, not directly on top of the floor 1, elevator should works a sa teleport to floor 2 both up and down should have animation of going up and down"
- **Decision:**
  1. **Side-by-Side Floor 2 Architecture ($X = +50.0\,\text{m}$):** Created `sim_vision/godot/scenes/room_floor2.tscn` ($24\,\text{m} \times 18\,\text{m}$) containing Rooms 201/204 (Imaging/CT/X-ray), 205 (Gastroenterology), 210 (ENT), 220 (Dermatology), Floor 2 Waiting Lobby (`sanh_t2`), Restrooms (`wc_t2`), and Elevator Station 2. Placed side-by-side at $X = +50.0\,\text{m}$ inside `room.tscn` (not stacked vertically) to avoid ceiling culling conflicts and allow seamless camera panning without reloading scenes.
  2. **Interactive Elevator Station (`elevator_station.tscn`):** Created full 3D cabin station with dual sliding steel doors (`DoorLeft`, `DoorRight`), dynamic 3D LED floor displays ("TẦNG 1 ▲ / TẦNG 2 ▼"), high-fidelity two-tone Ding-Dong chime (`elevator_chime.wav`), and low-frequency motor hum (`elevator_hum.wav`).
  3. **Two-Stage Multi-Floor Routing & Animation:** In `rl_env_bridge.gd` and `main.gd`, escorting requests across floors (Floor 1 $\leftrightarrow$ Floor 2) dynamically guide to the departure floor elevator entrance $\to$ trigger full animated elevator sequence (doors slide open $\to$ passenger boarding $\to$ doors close $\to$ hum audio & LED arrow update $\to$ coordinate teleportation $\to$ destination chime $\to$ doors slide open $\to$ passenger exit $\to$ doors close) $\to$ update navigation goal to the final clinic on the arrival floor.
  4. **Multi-Floor RL Training Environment:** In `src/training/hospital_env.py`, generated `FLOOR2_WALLS` offset at $+50.0\,\text{m}$, enabled cross-floor elevator transit simulation with `REWARD_ELEVATOR_TRANSIT = +10.0`, and updated corridor doorway routing so PPO learns both local hallway navigation and cross-floor elevator boarding.
- **Why:** Solves the core bottleneck where 1-floor-trained models got stuck hitting the east wall when trying to navigate to Floor 2, adds realistic medical transport aesthetics, and keeps 3D physics fully decoupled across floors.
- **Touches:** `sim_vision/godot/scenes/room.tscn`, `sim_vision/godot/scenes/room_floor2.tscn`, `sim_vision/godot/scenes/props/elevator_station.tscn`, `sim_vision/godot/scripts/elevator_station.gd`, `sim_vision/godot/scripts/room.gd`, `sim_vision/godot/scripts/rl_env_bridge.gd`, `sim_vision/godot/scripts/main.gd`, `src/training/hospital_env.py`, `tests/test_floor2_elevator.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 151/151 Godot GUT tests pass; 12/12 pytest unit tests pass (including multi-floor elevator transit test); verified GPU smoke training with warm-start weights from the 100k episode checkpoint.



### D-032 — Fix Patient Escort Geometric Deadlock and Velocity Mismatch During Inference
- **Status:** accepted
- **When:** 2026-10-26T16:25:00+07:00
- **Decided by:** human
- **Trigger:** "our newest training, but from what i saw when inference, the robot are literealy not works for escort patient, you test its our, by examining a specific case where patient ask for destination and see why the robot wont works experiments\runs\20261025_1248_PPOHospital"
- **Diagnosis:**
  1. **Geometric Arrival Deadlock (`GOAL_ARRIVE_PATIENT_M = 2.0m`):** In `main.gd` and `hospital_env.py`, goal arrival required both $d(\text{robot}, \text{goal}) \le 1.4\text{m}$ and $d(\text{patient}, \text{goal}) \le 2.0\text{m}$. However, `step_escorted_patient` halts the patient as soon as $d(\text{patient}, \text{robot}) \le 0.9\text{m}$. When the robot stopped at $1.2\text{m}$ from the goal, the trailing patient stopped at $1.2 + 0.85 = 2.05\text{m} > 2.0\text{m}$. Neither agent moved again, deadlocking task completion.
  2. **Escort Velocity Mismatch:** In `rl_policy_server.py`, robot speed uncapped reached $V_{\text{MAX}} = 0.80\,\text{m/s}$, while elderly patient walking speed is $0.48\,\text{m/s}$. Within 3 seconds, tether stretched past $2.2\,\text{m}$, triggering sharp policy speed collapse ($0.80 \to 0.21 \to 0.03\,\text{m/s}$) and severe right-turn steering bias ($\omega = -0.92\,\text{rad/s}$).
- **Decision:**
  1. Relaxed `GOAL_ARRIVE_PATIENT_M` from $2.0\,\text{m}$ to $2.5\,\text{m}$ in `src/training/hospital_env.py` and `sim_vision/godot/scripts/rl_env_bridge.gd`, safely accommodating $1.4\,\text{m} (\text{robot}) + 0.9\,\text{m} (\text{tether}) = 2.3\,\text{m} \le 2.5\,\text{m}$.
  2. In `sim_vision/python/rl_policy_server.py` and `src/training/hospital_env.py`, capped robot forward velocity during `TaskPhase.GUIDE_ESCORT` to `ESCORT_WALK_SPEED = 0.52 m/s` (matching patient walking speed of $0.48\,\text{m/s}$), preventing tether stretch and keeping $d_{\text{tether}}$ within the optimal $1.0 - 1.6\,\text{m}$ window.
- **Why:** Completely eliminates arrival deadlocks at destination clinics and elevators, avoids tether-stretch policy speed collapse, and raises multi-seed test success rate from 76.7% to 100.0%.
- **Touches:** `src/training/hospital_env.py`, `sim_vision/godot/scripts/rl_env_bridge.gd`, `sim_vision/python/rl_policy_server.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 30-seed deterministic benchmark verified: success rate improved from 76.7% to 100.0% (0 failures); average patients helped increased from 1.27 to 1.70 (+34%); all 18 pytest tests pass.

### D-033 — Natural Patient Navigation, Animation Stabilization (T-Pose Option), and Multi-NPC Dialogue Targeting
- **Status:** accepted
- **When:** 2026-10-26T23:45:00+07:00
- **Decided by:** human
- **Trigger:** "the robot only target the red shirt patient, finetune it so it can adopt all the new patient", "and human animation seem flicking, why? if its too complicated, use t pose for now", "dont teleport the red shirt patient to the nearby possition of the robot anymore, make him a normal patient like other"
- **Decision:**
  1. **Disable Artificial Red-Shirt Teleportation:** In `sim_vision/godot/scripts/main.gd`, removed `is_task_transition` forced teleportation of the red-shirt patient. Placed the red-shirt patient at natural corridor coordinates `(4.5, 0.0, -1.35)` with waypoint loop and pardon behavior enabled, treating him as a standard patient alongside the other 6 seated/walking NPCs.
  2. **Multi-NPC Dynamic Dialogue Targeting:** In `main.gd`, replaced hardcoded references to `human` in `human_distance()` and `human_angle_deg()`. During dialogue states (`LISTEN`, `THINK`, `RESPOND`), distance and angle are evaluated against the active speaking patient (`get_active_speaking_patient()`), preventing incorrect angle/distance telemetry when other patients speak.
  3. **Animation Stabilization (`use_t_pose`):** Added `@export var use_t_pose: bool = false` in `human.gd` and enabled it in `main.tscn` to eliminate skeletal bone interpolation flickering/jitter while retaining the physics capsule collision shape and standard unit test compatibility.
- **Why:** Removes artificial biases toward the red-shirt patient, stabilizes visual animation rendering, and ensures the robot properly engages and escorts whichever patient initiates dialogue.
- **Touches:** `sim_vision/godot/scripts/human.gd`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scenes/main.tscn`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** Verified all 7 human unit tests pass; red-shirt patient patrols corridor naturally; active speaking patient correctly becomes the focus of dialogue and escorting.

### D-034 — Dynamic 3D Door Labels and Hospital Department Signage Overhaul
- **Status:** accepted
- **When:** 2026-10-27T10:15:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment (`plan_ui_door_signs_floor3.md`)
- **Trigger:** "all the rooms that can be destination for patient, must be writen in its name, because as I see right now, all of them have the tabs name in the door to be 'Phòng khám'"
- **Decision:**
  1. **Dynamic Door Label Tooling:** In `sim_vision/godot/scripts/sliding_door.gd`, added `@tool` annotation, exported property `door_label: String` with setter `set(val)`, and implemented `_update_labels()` to synchronize both front (`SignLabel3D`) and back (`SignBackLabel3D`) 3D labels directly in the Godot Editor viewport and at runtime.
  2. **Department Designation Overhaul:** Replaced generic "Phòng khám" labels across all floors with specific room numbers and hospital department titles:
     - **Floor 1:** `Door_101` ("101 - Phòng Cấp cứu"), `Door_110` ("110 - Phòng Khám Nội"), `Door_112` ("112 - Phòng Khám Ngoại"), `Door_Pharmacy` ("105 - Nhà thuốc Bệnh viện"), `Door_Lobby` ("100 - Quầy Tiếp đón & BHYT").
     - **Floor 2:** `Door_201` ("201 - Khoa Mắt"), `Door_204` ("204 - Khoa Răng Hàm Mặt"), `Door_205` ("205 - Khoa Tiêu hóa"), `Door_210` ("210 - Khoa Tai Mũi Họng"), `Door_220` ("220 - Khoa Da liễu").
     - **Floor 3:** `Door_301` ("301 - Khoa Tim mạch"), `Door_302` ("302 - Phòng Siêu âm tim"), `Door_303` ("303 - Phòng Điện tim ECG"), `Door_315` ("315 - Khoa Cơ xương khớp").
- **Why:** Eliminates clinical room ambiguity for human inspection and robot navigation, matching the NLU slot destinations in hospital dialogue.
- **Touches:** `sim_vision/godot/scripts/sliding_door.gd`, `sim_vision/godot/scenes/room.tscn`, `sim_vision/godot/scenes/room_floor2.tscn`, `sim_vision/godot/scenes/room_floor3.tscn`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** Verified in Godot Editor and runtime; door labels show clear room numbers and specialty titles on both sides; sliding door unit tests pass.

### D-035 — Floor 3 Hospital Extension and Multi-Floor Dynamic Elevator Navigation (X = +100.0m)
- **Status:** accepted
- **When:** 2026-10-27T11:20:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment (`plan_ui_door_signs_floor3.md`)
- **Trigger:** "plan for adding floor number 3, which include all the destinated one for that floor"
- **Decision:**
  1. **Side-by-Side Floor 3 Modular Scene ($X = +100.0\,\text{m}$):** Created `sim_vision/godot/scenes/room_floor3.tscn` ($24\,\text{m} \times 18\,\text{m}$) containing Rooms 301 (Cardiology), 302 (Echocardiography), 303 (Electrocardiogram ECG), 315 (Orthopedics), Floor 3 Waiting Lobby (`sanh_t3`), Restrooms (`wc_t3`), and `Floor3Elevator` (`current_floor = 3`). Instantiated at $X = +100.0\,\text{m}$ inside `room.tscn`.
  2. **Generalized Multi-Floor Elevator Transit Formula:** In `sim_vision/godot/scripts/main.gd`, generalized elevator translation:
     $$\Delta X = (to\_floor - from\_floor) \times 50.0\,\text{m}$$
     Supporting any combination of Floor 1 ($0\,\text{m}$), Floor 2 ($50\,\text{m}$), and Floor 3 ($100\,\text{m}$) travel.
  3. **Dynamic Camera & Occlusion Handling:** Clamped camera orbit center dynamically ($X_{\text{center}} = 100\,\text{m}$ for $X \ge 75\,\text{m}$, $50\,\text{m}$ for $25 \le X < 75\,\text{m}$, and $0\,\text{m}$ for $X < 25\,\text{m}$) and cached Floor 3 interior walls for see-through transparency.
  4. **Multi-Floor RL Environment Routing:** Updated `Room.get_floor()`, `Room.get_room_floor()`, `rl_env_bridge.gd`, and `src/training/hospital_env.py` to route Floor 3 clinical keys (`301`, `302`, `303`, `315`, `dept_tim_mach`, `dept_co_xuong_khop`, `sanh_t3`) through the nearest elevator station and to destination clinic doors.
- **Why:** Fully supports Floor 3 departments referenced in clinical dialogue, expands hospital capacity to 3 complete floors, and keeps multi-floor physics decoupled and modular.
- **Touches:** `sim_vision/godot/scenes/room_floor3.tscn`, `sim_vision/godot/scenes/room.tscn`, `sim_vision/godot/scripts/room.gd`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/rl_env_bridge.gd`, `src/training/hospital_env.py`, `sim_vision/godot/tests/test_room.gd`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 18/18 GUT tests in `test_room.gd` pass (16,516 asserts); 6/6 GUT tests in `test_rl_env.gd` pass; 35/35 Python pytest tests pass.

### D-036 — Robot Inspection Telemetry HUD Panel with 4-Tabbed Progress Bar Visualizations
- **Status:** accepted
- **When:** 2026-10-27T14:40:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment (`plan_ui_door_signs_floor3.md`)
- **Trigger:** "update the UI in the top left conner when in the robot inspection cameras, it have a open/closable tab in that, if open it display the values of every inputs data sperate in tabs like you describe using process bar"
- **Decision:**
  1. **Component Design (`InspectionPanel`):** Created `sim_vision/godot/scripts/inspection_panel.gd` and `sim_vision/godot/scenes/inspection_panel.tscn` anchored at top-left (`UI/InspectionPanel` in `main.tscn`).
  2. **Collapsible Header & Shortcut Controls:** Added a toggle button `[▼] TELEMETRY / [▲] CLOSE` with status summary chips, supporting mouse click toggling and keyboard hotkeys (`I` / `Tab`).
  3. **4 Data Tabs with Real-Time Progress Bars and Numeric Labels:**
     - **Tab 1: LiDAR (17 Rays, 200° Fan):** 17 progress bars ($0.0 - 6.0\,\text{m}$) color-coded green ($\ge 1.5\,\text{m}$), yellow ($0.8 - 1.5\,\text{m}$), and red ($< 0.8\,\text{m}$).
     - **Tab 2: YOLO Vision:** Patient detection confidence bar ($0-100\%$), normalized centroid offset $C_x$ with left/right bias indicator, and bounding box height $H$ ($0.0 - 1.0$) with arrival threshold marking at $0.35$.
     - **Tab 3: RL Navigation:** Active task phase badge (`SEARCH_PATIENT`, `APPROACH_PATIENT`, `GUIDE_ESCORT`, `CALL_STAFF`), geodesic goal distance ($0 - 20\,\text{m}$), heading angle indicators ($\sin\theta, \cos\theta$), patient escort tether distance ($1.6\,\text{m}$ ideal, $2.5\,\text{m}$ max limit), linear velocity $V$, and angular velocity $\omega$.
     - **Tab 4: Voice & NLU:** STT & TTS connection chips, ASR transcribed hypothesis text, NLU intent & extracted slot, and target floor & clinic destination.
  4. **Performance Optimization:** Telemetry bars and text fields update only when the panel is expanded (`is_open == true`), incurring zero rendering overhead when collapsed.
- **Why:** Provides human operators and researchers with full, transparent visibility into the robot's real-time sensory and decision-making state during inspection camera view without cluttering the main 3D viewport.
- **Touches:** `sim_vision/godot/scripts/inspection_panel.gd`, `sim_vision/godot/scenes/inspection_panel.tscn`, `sim_vision/godot/scenes/main.tscn`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/tests/test_inspection_panel.gd`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 4/4 GUT tests in `test_inspection_panel.gd` pass; all 163 GUT suite tests pass across the repository (18,873 asserts); hotkeys `I` and `Tab` toggle the panel cleanly.

### D-037 — Dedicated 3-Floor Patient Escort PPO Training & Distraction Guard
- **Status:** accepted
- **When:** 2026-10-27T18:30:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment (`train_3floor_escort_2000ep_plan.md`)
- **Trigger:** "draft up a training plan, focus solly on escort patient on both 3 floors, train for 2k episode first /grill-me/grill-me", "to prevent deadlock (aka talk and accept too many escort, make the robot so that it will not be distract by other people until the current task is complete)"
- **Decision:**
  1. **Distraction Guard (Anti-Deadlock Mechanism):**
     - In `sim_vision/godot/scripts/main.gd`, added `is_task_busy() -> bool` returning `true` when `active_escorted_patient != null` or when the robot is actively executing an escort/call-staff task phase.
     - In `dialogue_manager.gd`, guarded `_on_talk_started()` and `register_npc()`: incoming NPC speech triggers and interaction popups are dropped/ignored whenever `main.is_task_busy()` is true, preventing dialogue interruption or task overwriting while an escort is underway.
  2. **Multi-Floor Environment & Dynamic Stagnation Reset:**
     - Updated `src/training/hospital_env.py` and `rl_controller.gd` with Floor 3 obstacles (31 rects, total 155 rects across 3 floors), north/south door routing, and elevator positions.
     - Enforced `escort_only=True` mode: 100% escort focus, balanced with 50% cross-floor elevator journeys ($F_1 \leftrightarrow F_2, F_1 \leftrightarrow F_3, F_2 \leftrightarrow F_3$) and 50% same-floor clinic escort.
     - Dynamic Stagnation Reset: 60s base timeout, with clock reset to 0 whenever an elevator transit occurs or the robot advances $\ge 1.0\,\text{m}$ toward its active goal.
  3. **Dedicated Trainer Script (`train_3floor_escort_2000ep.py`):**
     - Warm-started from full-patient policy (`20261026_214227_PPOHospital-full-patient`).
     - 2000 episodes on NVIDIA RTX 4060 GPU with Cosine Annealing LR ($1.0 \times 10^{-4} \to 3.0 \times 10^{-5}$), teacher prior decaying $0.05 \to 0.0$ over first 300 episodes, and deterministic evaluation every 200 episodes on 20 held-out seeds.
- **Why:** Eliminates deadlock caused by bystander NPC interruptions during patient transport, equips the PPO policy with multi-floor elevator transit navigation, and focuses learning entirely on patient escort delivery.
- **Touches:** `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/dialogue_manager.gd`, `sim_vision/godot/scripts/rl_controller.gd`, `src/training/hospital_env.py`, `src/training/hospital_layout.json`, `src/training/train_3floor_escort_2000ep.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** Full 2000-episode training run completed in 2h43m (9,839.7s) on NVIDIA RTX 4060 GPU (`experiments/runs/20261027_183421_PPOHospital-3floor-escort-2k`):
  - **Completed Shifts:** 2,000 / 2,000 (100% completion rate).
  - **Patients Escorted:** 7,362 (average 3.68 patients helped per 240s shift).
  - **Stagnation Timeouts:** 0 (0.0% stagnation rate across all 2,000 episodes and all evaluation checkpoints).
  - **Best Rolling-20 $R_{team}$:** +158.10 (Peak $R_{team}$: +246.00).
  - **Best Deterministic Eval Score:** +134.00 (3.30 tasks/shift, 0.0 collisions across 20 held-out seeds).
  - **Checkpoints Saved:** `PPOHospital-3floor-escort_best.pt` and `PPOHospital-3floor-escort_last.pt`.

### D-038 — Escort-Only Inference Mode, Overlay HUD Escort Label, and Manual Test Hotkey
- **Status:** accepted
- **When:** 2026-10-27T19:00:00+07:00
- **Decided by:** human
- **Trigger:** "hey, i see the inference the escort doest works at all, the robot just sat there, and the status got updated to arrived, no where to see the escort, make a modifle or flag for the inference/test code for patient to only ask for escort for me to check more throughtly"
- **Decision:**
  1. **Root Cause Identification:**
     - In speech dataset `sim_vision/godot/assets/voices/clips.json`, most clips map to non-escort intents (`dang_ky_kham`, `bao_hiem`, etc.), which trigger verbal answer and patient reset rather than an escort task.
     - In `main.gd`, `_step_policy()` passed `current_target_patient` rather than `active_escorted_patient`, triggering an artificial 10m tether break and stopping the robot during escort.
     - Overlay HUD displayed `Config.state_name(state)` which froze on `"ARRIVED"` even while actively navigating.
     - `_physics_process()` lacked `"thang_may_t3"` in elevator transition triggers.
  2. **`--escort-only` Simulation Flag:**
     - In `npc.gd`, added `const ESCORT_INTENTS = ["hoi_phong", "hoi_khoa", "dan_duong"]` and `escort_only` property to filter voice clips strictly to clinical destination escort requests.
     - In `run_simulation.py`, added `--escort-only` CLI argument forwarded to Godot.
     - In `main.gd`, parsed `--escort-only` and applied it to all hospital NPCs; if a non-escort intent somehow completes, promoted it to a multi-floor clinic escort.
  3. **Inference Navigation & HUD Overhaul:**
     - Fixed `_step_policy()` to measure distance and tether relative to `active_escorted_patient`.
     - Added `_display_state_name()` displaying `"ESCORT: <dest>"` or `"CALL_STAFF"` in the HUD overlay.
     - Added `thang_may_t3` to elevator transition checks.
     - Added `KEY_G` hotkey in `main.gd` calling `trigger_test_escort()` to instantly dispatch an escort task to Floors 1, 2, or 3 for immediate interactive testing.
  4. **Dedicated CLI Policy Test Script:**
     - Created `src/training/test_3floor_escort.py` to run headless multi-episode evaluations of trained escort checkpoints with step-by-step logs of elevator rides, patient pacing, and arrivals.
- **Why:** Guarantees 100% escort task testing without waiting through non-escort Q&A dialogues, corrects policy tether tracking in Godot 3D, and provides clear visual feedback of the active escort destination.
- **Touches:** `run_simulation.py`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/npc.gd`, `src/training/test_3floor_escort.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** `pytest` test suite passes (26/26 passed); `python -m src.training.test_3floor_escort --episodes 2` completes with 6/6 patients escorted and 5 elevator transits; verified in Godot runtime with `--escort-only` and `G` key.

### D-039 — Fix Solo Patient Targeting, Busy Guard, and Dialogue-to-Escort Transition
- **Status:** accepted
- **When:** 2026-10-30T07:00:00+07:00
- **Decided by:** human-approved via `/grill-me` alignment
- **Trigger:** "in the godot 3d simulation: the robot drive up to the patient? it movment is weird, the patient do speak, but no escort start, it sit there, after like 3-5s, it tries to find other patient"
- **Decision:**
  1. **Fix Solo-Patient Target Selection:**
     - In `sim_vision/godot/scripts/main.gd:get_available_patients()`, fixed logic so in `solo_patient_mode` it strictly returns `[human]` (`$Human`), preventing the robot from pursuing invisible or disabled furniture NPCs on other floors.
     - Updated `_nearest_person_distance()` in `main.gd` to ignore furniture NPCs during solo mode.
     - In `reset_positions(is_task_transition)`, kept the robot in place and re-positioned the solo `$Human` at a reachable corridor/lobby spot on the current floor ($\ge 3\,\text{m}$ away) instead of teleporting the robot back to spawn.
  2. **Refine Busy Guard (`APPROACH_PATIENT` Removal):**
     - In `main.gd:is_task_busy()`, removed `APPROACH_PATIENT` from busy states. This ensures that while the robot is driving toward a target patient, NPCs can evaluate the robot as receptive and trigger `_start_talk()` once the robot arrives.
  3. **Seamless Dialogue-to-Escort Transition:**
     - In `main.gd:_on_dialogue_turn_completed()`, prevented state overwriting during dialogue so `state == RESPOND` stays intact until audio and subtitles complete.
     - In `main.gd:set_dialogue_state()`, when returning to `ARRIVED` upon completion of dialogue speech, if `is_task_busy()` is true (an escort was scheduled), immediately transitions to `Config.RobotState.APPROACH` so the robot and patient begin walking without a 3-5s stall or abort reset.
     - In `sim_vision/godot/scripts/dialogue_manager.gd:_after_response()`, fixed the speaker reference to use `target_npc = active_npc if active_npc != null else npc`.
  4. **Escort Clip Filtering & Unit Test Conformance:**
     - In `sim_vision/godot/scripts/npc.gd`, excluded clip sentence `05` (symptom advice query lacking destination slot) in `escort_only` mode.
     - Fixed `pick_clip()` fallback to strictly adhere to GUT test specification by returning `""` on empty speaker clips.
- **Why:** Solves the root causes of the simulation stall where the robot reached the patient but failed to begin escorting, eliminated erratic navigation toward furniture NPCs, and ensures instantaneous start of patient walking upon speech completion.
- **Touches:** `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/dialogue_manager.gd`, `sim_vision/godot/scripts/npc.gd`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** All GUT unit tests pass cleanly (163/163 passed, 18,873 asserts); all Python pytest tests pass (26/26 passed); verified smooth immediate transition from voice completion to walking in solo-patient escort mode.

### D-040 — Fix GDScript STT/TTS Connection Property Crash in Dialogue Turn Completed
- **Status:** accepted
- **When:** 2026-10-30T07:40:00+07:00
- **Decided by:** human-directed log analysis
- **Trigger:** "SCRIPT ERROR: Invalid access to property or key 'connected' on a base object of type 'Node (SttClient)'. at: Main._on_dialogue_turn_completed (res://scripts/main.gd:438)"
- **Decision:**
  1. **Root Cause Analysis:**
     - During normal dialogue execution, when `dialogue.turn_completed` signal fired, `_on_dialogue_turn_completed` in `sim_vision/godot/scripts/main.gd` attempted to access `dialogue.stt.connected` and `dialogue.tts.connected` to update the HUD telemetry inspection panel.
     - `SttClient` and `TtsClient` encapsulate their connection status behind the public method `is_server_connected() -> bool` (`_connected` is private).
     - Accessing `.connected` raised an uncaught GDScript runtime exception that aborted `_on_dialogue_turn_completed` before reaching line 446. Consequently:
       - `active_escorted_patient` was never assigned.
       - `_rl_bridge.begin_post_dialogue_task(...)` was never called.
       - The task phase remained stuck at `SEARCH_PATIENT`, leaving the robot stationary until the 10s non-response timer timed out and reset the interaction.
     - In contrast, pressing `KEY_G` bypassed `_on_dialogue_turn_completed` and directly invoked `_rl_bridge.begin_post_dialogue_task()`, which explained why `G` escorted reliably while normal dialogue failed.
  2. **Code Resolution:**
     - In `sim_vision/godot/scripts/main.gd:439-440`, updated property access to method call:
       ```gdscript
       dialogue != null and dialogue.stt != null and dialogue.stt.is_server_connected(),
       dialogue != null and dialogue.tts != null and dialogue.tts.is_server_connected(),
       ```
- **Why:** Restores normal speech-driven dialogue turn completion and eliminates the script exception, allowing the escort state transition and RL policy dispatch to execute seamlessly when patients finish speaking.
- **Touches:** `sim_vision/godot/scripts/main.gd`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** All 163 GUT unit tests pass without error (18,873 asserts, 140.9s); confirmed no other invalid `.connected` accesses remain across the codebase.

### D-041 — Fix Escort Patient Trapping Behind Furniture and Dynamic Trail Pruning
- **Status:** accepted
- **When:** 2026-10-30T07:55:00+07:00
- **Decided by:** human log observation
- **Trigger:** "the patient not follow the bot, why? ... Robot=(-0.65, 3.94) Patient=(-2.72, 6.36) tether=3.19m"
- **Decision:**
  1. **Root Cause Analysis from Telemetry:**
     - The telemetry logs showed the patient moved from `(-3.24, 6.36)` to `(-2.72, 6.36)` and then completely froze while the robot moved towards the elevator.
     - Hospital layout inspection revealed `Furniture/WaitingBenchWestBack/Collision` sits at $X \in [-4.875, -2.725], Z \in [5.49, 6.11]$. The patient at $X = -2.72, Z = 6.36$ was physically pressing against the back of the waiting bench.
     - Why were they separated? `pick_solo_start_positions()` in `main.gd` had sampled `human_pos` using `get_free_position()` into the bench nook, then randomly spawned the robot across the bench in the front lobby ($Z \le 5.32$) without checking for line-of-sight obstruction. The robot approached to engage distance ($1.35\,\text{m}$) across the bench, spoke with the patient, and began escorting. The straight path from patient to robot's first crumb collided with the bench.
     - Furthermore, `step_escorted_patient()` in `rl_env_bridge.gd` strictly popped `trail[0]` only when distance was $\le 0.3\,\text{m}$ (`ESCORT_CRUMB_REACH_M`). Because the trapped patient stayed at $1.22\,\text{m}$ from the initial crumb, `trail.pop_front()` was never called; the patient remained permanently fixated on the blocked initial crumb rather than following the robot forward.
  2. **Code Resolution:**
     - **Lobby Patient Spawn Bounds (`room.gd`):** Trimmed `PATIENT_LOBBY_RECT` to `Rect2(Vector2(-3.2, 2.2), Vector2(6.4, 2.8))` ($Z \in [2.2, 5.0]$), matching Python training `HospitalShiftEnv._sample_patient_spawn_solo()` exactly and preventing patient spawns behind lobby benches.
     - **Solo Start Placement with Obstacle-Free LOS (`main.gd`):** Updated `pick_solo_start_positions()` to sample `human_pos` from `r.get_patient_spawn_position(1, rng)` and verify an unobstructed line of sight to candidate `robot_pos` using `AABB.intersects_segment()`.
     - **Fresh Trail Initialization (`rl_env_bridge.gd`):** In `begin_post_dialogue_task()`, cleared `escort_trail` and initialized with current robot position so past crumbs never linger.
     - **Adaptive Trail Progression (`rl_env_bridge.gd`):** Updated `step_escorted_patient()` while-loop to advance if `d0 <= ESCORT_CRUMB_REACH_M or d1 <= d0`. If the patient is already closer to the next waypoint `trail[1]`, obsolete crumbs are pruned automatically, preventing backtracking and deadlock.
- **Why:** Ensures the patient spawns in open walkable areas with clear access to the robot, and allows breadcrumb following to smoothly progress along the robot's trail without getting stuck on past crumbs.
- **Touches:** `sim_vision/godot/scripts/room.gd`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/rl_env_bridge.gd`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 163/163 GUT unit tests pass (18,873 asserts, 140.3s). Patient now has direct, open access to follow robot along the breadcrumbs.

### D-042 — Seated Patient Interaction Position, Stand-Up Transition, and Physics Collision Fix
- **Status:** accepted
- **When:** 2026-10-30T08:15:00+07:00
- **Decided by:** AI-pair (Manh + Antigravity)
- **Trigger:** "ok, this time I run without solo patient, and the robot pick up a random patient, but its did not follow ... Robot=(-0.99, 3.44) Patient=(-3.59, 3.50) tether=2.60m"
- **Decision:**
  1. **Root Cause Analysis from Telemetry:**
     - In full hospital simulation (with crowd NPCs), the robot selected `Patient_01` who was seated on `WaitingBenchWest` at $X = -4.55, Z = 4.20$.
     - `main.gd` previously passed raw `target_p.global_position` ($Z = 4.20$) to the RL policy and distance metrics. During Python training (`hospital_env.py:761`), seated targets were explicitly placed at `cand_z = chosen["z"] - 0.9` ($Z = 3.3$, in the open corridor aisle). In Godot, raw bench coordinates caused the robot to approach from the north passage behind the bench and stop at `Robot=(-3.47, 4.99)`.
     - Dialogue triggered across the backrest of the bench. When escort started, the robot moved towards the elevator, dropping its initial crumb behind the bench. `Patient_01` stood up at $Z = 3.50$ and attempted to move north towards the crumb, physically colliding with `WaitingBenchWest` ($Z \in [3.89, 4.51]$).
     - Furthermore, in `hospital_npc.gd:226-230`, `global_position += disp` ran unconditionally without an `else` branch after `move_and_slide()`, forcing the character capsule directly into the static bench collision geometry every frame and inducing physics lockup.
  2. **Code Resolution:**
     - **Interaction Position (`hospital_npc.gd`):** Added `get_interaction_position() -> Vector3`. If `is_seated`, returns `waypoints[1]` (which is explicitly placed in the open corridor aisle at $Z = 2.50$) or an offset $0.9\,\text{m}$ in front of the seat, matching training assumptions.
     - **Stand-Up Transition (`hospital_npc.gd`):** In `set_state(State.ESCORTED)` and `escort_step_towards()`, if `is_seated`, automatically clears seated state and places the NPC at their open aisle waypoint (`waypoints[1]`) before movement begins.
     - **Physics Collision Preservation (`hospital_npc.gd`):** Fixed `escort_step_towards()` and `_move_step_towards()` to place `global_position += disp` inside an `else:` branch, allowing Godot's `move_and_slide()` to resolve collisions naturally without forced penetration.
     - **Dialogue & Policy Alignment (`main.gd` & `npc.gd`):** Updated `_step_policy()`, `human_distance()`, `human_angle_deg()`, and `npc.gd:_process()` to use `get_interaction_position()`, ensuring the robot approaches the patient from the open corridor in front of the bench.
     - **Crumb Clearance (`rl_env_bridge.gd`):** Updated `ESCORT_CRUMB_REACH_M = 0.45` to permit smooth capsule navigation around lobby furniture.
- **Why:** Ensures seated patients are approached from the open aisle rather than from behind furniture, cleanly stand up into walkable space when escort begins, and smoothly follow the breadcrumb trail without physics collisions.
- **Touches:** `sim_vision/godot/scripts/hospital_npc.gd`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/npc.gd`, `sim_vision/godot/scripts/rl_env_bridge.gd`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** GUT unit tests passing; verified patient stands up in the corridor and follows breadcrumbs without colliding into bench geometry.

### D-043 — Analysis of Duty Cycle Flaws, Oracle Dispatch vs Visual RL, and Pure Visual-Search RL Roadmap
- **Status:** accepted
- **When:** 2026-10-30T08:50:00+07:00
- **Decided by:** human-directed architectural review
- **Trigger:** Human identified two fundamental operational and training flaws: 1) Patient repeated request after arriving at clinic room; 2) Robot stranded on upper floors without a return-to-lobby mechanism; 3) Revelation that current RL policy relies on oracle patient coordinates rather than visual search.
- **Decision:**
  1. **Root Cause Analysis — Patient Lifecycle & Duty Cycle:**
     - When arriving at destination (e.g. Room 205 on Floor 2), the escorted patient stayed standing outside the room without being flagged as completed (`helped = true`).
     - `main.gd:355` filtered next targets by current floor (`same_fl`). Since Floor 2/3 had no crowd patients, the single newly arrived patient was immediately re-targeted, triggering dialogue again.
     - In training (`hospital_env.py:756`), new patients were artificially spawned on the robot's current floor (`x_base = floor_of(robot)`), omitting the critical hospital duty cycle where the robot rides the elevator down to Floor 1 Lobby to collect new outpatients.
  2. **Architectural Contrast — Vision FSM vs Goal-Conditioned RL:**
     - FSM mode (`--controller fsm`) searches using YOLOv8 camera vision, scanning the room and steering via bounding box coordinates ($c_x, h$).
     - RL mode (`--controller rl`) operates as an Oracle-Dispatched Goal-Directed Agent: observation vector (31 dims) only takes 17 LiDAR rays and a goal vector $(d_{\text{goal}}, \Delta\theta)$ fed from simulation ground-truth coordinates. The camera/YOLO feed is only used for HUD overlay and CSV logging; the PPO network is blind to vision.
  3. **Strategic Direction (Pure RL Vision-Search Roadmap):**
     - Transition from oracle dispatch toward a Pure RL-driven robot that actively searches for patients using vision/detections without hardcoded or ground-truth patient coordinates.
     - Known static landmarks (rooms, elevator doors) may remain fixed map coordinates, but human target acquisition must be visual and learned.
- **Why:** Establishes complete parity between actual clinical hospital operations (Lobby $\to$ Escort $\to$ Clinic Room $\to$ Return to Lobby) and provides the architectural foundation for a true Visual Search RL agent.
- **Touches:** `docs/AI-Human-logs/Vuong.md`, `sim_vision/godot/scripts/main.gd`, `src/training/hospital_env.py`
- **Evidence:** Comprehensive codebase analysis confirming `_on_detections()` bypasses YOLO during RL mode; theoretical and empirical proof of duty cycle deadlock when multi-floor transition finishes.

### D-044 — /grill-me Interview Record: Pure Visual-Search RL Architecture and New Training Branch
- **Status:** accepted
- **When:** 2026-10-30T09:10:00+07:00
- **Decided by:** human-approved via /grill-me
- **Trigger:** Human requested a rigorous architectural interview (`/grill-me`) to design a Pure RL-driven robot that actively discovers patients through vision without oracle ground-truth coordinates, while preserving LiDAR obstacle avoidance and room escorting.
- **Decision:**
  1. **Q1 — Vision Observation Representation:**
     - *Question:* How should visual patient information be fed to the PPO policy?
     - *Decided:* **Feature-Level Vision Observation (YOLO Bounding Box Features + LiDAR + Nav Goals)**. Compact vector containing `[person_seen, norm_cx, norm_cy, box_w, box_h]` combined with 17 LiDAR rays and kinematic state. Enables ultra-fast PPO simulation (>5,000 steps/s in Python) and direct transfer to YOLOv8 inference in Godot.
  2. **Q2 — Motion during SEARCH (No Patient in FOV):**
     - *Question:* How should the robot move when searching and no patient is visible?
     - *Decided:* **Autonomous Exploration / Curiosity & Area Coverage Reward**. During SEARCH, the goal vector is neutral $(0, 0)$. The robot explores the hospital floor autonomously driven by intrinsic coverage incentives until a patient enters camera FOV. (Recommended patrol waypoints documented as fallback baseline).
  3. **Q3 — Multi-Floor Duty Cycle & Return-to-Lobby:**
     - *Question:* How does the robot learn to take the elevator down to Floor 1 after completing an escort on Floor 2/3?
     - *Decided:* **Implicit Opportunity Return (Sparse Penalty Formulation)**. No privileged multi-floor oracle state. Robot experiences a per-step time penalty (`-0.01/step`). Upper floors without waiting patients represent a reward desert; the robot must discover that riding the elevator down to Floor 1 unlocks high task completion rewards (+30.0).
  4. **Q4 — Visual Patient Acquisition Reward:**
     - *Question:* What reward guides the robot when a person is visible (`seen = 1`)?
     - *Decided:* **Visual Acquisition Reward**. Positive reward for detection (+2.0 when `seen=1`), camera centering penalty ($-|c_x|$ to align the robot's heading), and distance reduction shaping ($+h$ scaling up to $0.45$). Reaching $h \ge 0.45$ triggers engagement and dialogue.
  5. **Q5 — Fast Python Simulation of Vision:**
     - *Question:* How should the camera be modeled in fast Python simulation (`hospital_env.py`)?
     - *Decided:* **Geometric Pinhole Camera Model with Line-of-Sight and Domain Randomization**. $75^\circ$ FOV horizontal projection with $c_x = \tan\Delta\theta / \tan(\text{FOV}/2)$, $h \propto 1/D$, and raycast occlusion check. Domain randomization ($\pm 5\%$ jitter and frame dropout) ensures robustness against real-world YOLOv8 detection fluctuations during Godot inference.
  6. **Q6 — Implementation Roadmap:**
     - *Decided:* **Option 2: Focus 100% on the new Pure Visual RL environment on a dedicated Git branch (`feat/pure-visual-rl`)**, developing `hospital_env_vision.py` and `train_visual_rl.py`.
- **Why:** Replaces privileged oracle coordinates with genuine sensorimotor visual acquisition, uniting YOLO perception and PPO control into an authentic end-to-end autonomous robotic agent.
- **Touches:** `docs/AI-Human-logs/Vuong.md`, `src/training/hospital_env_vision.py`, `src/training/train_visual_rl.py`
- **Evidence:** Recorded interview consensus; architectural design verified against Pinhole optics, PPO sample complexity, and Godot 3D camera parameters.

### D-045 — Implementation of Pure Visual Search RL Environment, Training Pipeline, and Hospital Duty Cycle
- **Status:** accepted
- **When:** 2026-10-30T09:35:00+07:00
- **Decided by:** human-directed implementation on branch `feat/pure-visual-rl`
- **Trigger:** Implementation of approved plan `pure_visual_search_rl_plan.md` to eliminate oracle patient coordinate dispatch and fix multi-floor duty cycle.
- **Decision:**
  1. **Analytical Pinhole Perspective Camera Model (`src/training/camera_vision_model.py`):**
     - Implemented `compute_pinhole_detection()` modeling 70° VFOV, 86.4° HFOV camera at 1.0 m height matching Godot `robot.tscn`.
     - Projects 3D human bounding cylinders to normalized 2D image coordinates $[c_x, h]$ with line-of-sight raycasts against hospital walls.
     - Adds domain randomization: $\pm 4\%$ coordinate noise, $2\%$ random frame drop, and distance-dependent confidence ($0.95 - 0.03 \times d$).
  2. **Pure Visual-Search RL Environment (`src/training/hospital_env_vision.py`):**
     - Subclasses `HospitalShiftEnv`, strictly zeroing out oracle goal features (`obs[25..27] = [0.0, 0.0, 0.0]`) during `SEARCH_PATIENT` and `APPROACH_PATIENT`.
     - Reserves `obs[17..20]` for vision features `[has_det, cx_offset, h, conf]`.
     - Shaped visual rewards: $+2.0$ for first spotting, $-0.1 \times |c_x|$ camera centering penalty, and $+1.5 \times \Delta h$ approach expansion.
     - Grid cell coverage reward ($+0.05$ per newly visited $1\,\text{m} \times 1\,\text{m}$ tile) to motivate broad corridor and lobby exploration.
  3. **Dedicated Visual Training Pipeline (`src/training/train_visual_rl.py`):**
     - Supports `--episodes` (default 2000), warm-start from converged 3-floor checkpoint, `--device cuda`, and `--smoke` verification.
     - Implemented visual evaluation metrics (`spotting_rate`, `mean_search_time_s`, `r_team`, `patients_helped`).
  4. **Godot Runtime & Duty Cycle Fixes (`main.gd`, `rl_policy_client.gd`, `rl_policy_server.py`):**
     - Escorted patient arrival: When reaching clinic room destination, patient is marked `set_meta("helped", true)`, displays thank-you subtitle ("Cảm ơn robot, tôi đã tới nơi khám!"), and is excluded from `get_available_patients()`.
     - Multi-floor duty cycle: When robot completes escort on Floor 2 or Floor 3, it automatically calls the elevator and rides down alone (`with_patient = false`) to Floor 1 Lobby, resuming `SEARCH_PATIENT` for newly waiting outpatients.
     - In `rl_policy_server.py`, enforces zero-oracle observation vector during search, ensuring policy receives YOLO detections without leaked patient coordinates.
- **Why:** Solves both the repeated-dialogue issue at destination rooms, enables autonomous return to Floor 1, and removes the oracle coordinate cheat so the robot truly learns visual exploration and perception-driven approach.
- **Touches:** `src/training/camera_vision_model.py`, `src/training/hospital_env_vision.py`, `src/training/train_visual_rl.py`, `sim_vision/python/rl_policy_server.py`, `sim_vision/godot/scripts/rl_policy_client.gd`, `sim_vision/godot/scripts/main.gd`, `tests/test_camera_vision_model.py`, `tests/test_hospital_env_vision.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 10/10 vision unit tests pass; all 323 pytest unit tests pass (1 skipped); all 163 Godot GUT unit tests pass (18,873 asserts, 0 failures); verified 2-episode GPU smoke training run on RTX 4060.
### D-046 — Root-Cause Resolution for Visual Search Policy Freezing Trap & Stagnation Cutoff
- **Status:** accepted
- **When:** 2026-10-30T11:45:00+07:00
- **Decided by:** human-approved via fix_visual_search_rl_plan.md
- **Trigger:** Initial 2,000-episode pure visual RL run yielded 100% stagnation timeout (all episodes terminated at 300 steps / 60.0s with only 9 total patients helped). User requested empirical investigation to determine whether the approach is viable or a dead end.
- **Root Cause Discovered:**
  1. **Policy Freezing Trap:** The warm-started policy was trained with polar target vector obs[25..27] = [dist/25.0, sin_theta, cos_theta]. Zeroing obs[25..27] during visual search/approach caused the policy to infer distance = 0 (destination reached), commanding full reverse/brake (a_v = -0.92), freezing the robot in place the moment a human entered sight.
  2. **Zero-Oracle Pinhole Perception Formulation:** Deriving target distance d_vis = 1.20 / max(h, 0.05) and bearing angle_vis = -atan(cx * tan(HFOV/2)) from 2D pixel bounding box features provides a legitimate non-zero vector without leaking ground-truth world coordinates (zero-oracle).
  3. **Corridor Exploration Guidance:** Memoryless 2-layer MLPs cannot explore open hospital labyrinths without heading vectors. Introducing public hallway patrol waypoints (12 static nodes on Floors 1-3) gives purposeful directional sweep until a patient enters camera FOV.
  4. **Landmark Subgoal Routing Bug:** In HospitalVisualShiftEnv._get_obs(), during GUIDE_ESCORT, set_nav_target(*self._nav_final_target()) was missing, leaving goal_x, goal_z stale and 
av_remaining frozen after dialogue engagement.
  5. **Stagnation Timer Resets:** Stagnation counter steps_since_task was only reset on task completion. Added resets on: 1) visiting new 1m x 1m spatial cells, 2) first visual spotting, 3) visual approach expansion (h > best_h + 0.04), and 4) escort path progress.
- **Decision:**
  1. Updated src/training/hospital_env_vision.py with PUBLIC_PATROL_WAYPOINTS, pinhole vision conversion in _get_obs(), dynamic landmark path updates during GUIDE_ESCORT, and multi-condition stagnation resets.
  2. Updated sim_vision/python/rl_policy_server.py with identical pinhole vision formulas and corridor patrol loops for live Godot 3D inference.
  3. Enhanced unit test suite 	ests/test_hospital_env_vision.py verifying zero-oracle patrol routing, pinhole vision derivation, visual engagement, and exploration rewards.
- **Why:** Completely eliminates policy freezing, enables seamless transitions from autonomous corridor visual sweeping to perspective bounding-box approach, and maintains reliable tethered escort navigation.
- **Touches:** `src/training/hospital_env_vision.py`, `sim_vision/python/rl_policy_server.py`, `tests/test_hospital_env_vision.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 6/6 tests in `tests/test_hospital_env_vision.py` pass; verified multi-patient trace in scratch/trace_step.py completing 2 back-to-back patients in 400 steps; completed 2,000-episode visual training run: 5,748 patients helped (vs 9 in broken run), 100% visual spotting rate, Eval score +75.1 (vs 0.4), peak reward +213.9, best checkpoint saved at `experiments/runs/20261030_114837_PPOHospital-VisualSearch-3floor/checkpoints/PPOHospital-VisualSearch-3floor_best.pt`.

### D-047 — In-The-Wild Multi-NPC Training, Floor 3 Room Dialog Alignment, and NPC Unstuck Evasion
- **Status:** accepted
- **When:** 2026-10-30T17:30:00+07:00
- **Decided by:** human-approved via train_in_the_wild_2000ep_plan.md
- **Trigger:** During live 3D inference: 1) Robot dialogue stated "Khoa sản ở tầng 4" whereas the hospital has only 3 floors; 2) Patient_02 got wedged and completely stuck behind MedicalStaffNurse in the lobby corridor; 3) Training needed to transition from solo mode to crowded in-the-wild multi-NPC conditions with extended 300s shifts and automatic saving of 5 regular checkpoints.
- **Decision:**
  1. **Hospital Map & Dialog Alignment:** Updated `hospital_map.json`, `rl_env_bridge.gd`, and `room.gd` to align Khoa Sản to Floor 3, Room 315 and Khoa Nhi to Floor 3, Room 302.
  2. **NPC Geometric Collision & Evasion:** Relocated `MedicalStaffNurse` from walkway coordinate `(-1.9, 0, 7.45)` to behind reception desk `(-0.9, 0, 8.4)`. In `hospital_npc.gd`, added dynamic unstuck evasion in `escort_step_towards` (lateral evasion if movement blocked $<0.005\text{m}$ for $\ge 0.4\text{s}$; nudge past blocker if $\ge 1.2\text{s}$). Enhanced breadcrumb pruning in `rl_env_bridge.gd`.
  3. **In-The-Wild Training Architecture (`src/training/train_visual_rl_in_the_wild.py`):** Configured full multi-NPC mode (`env_mode="full"`), extended shift length to 300.0s (1,500 steps), and implemented dynamic saving of 5 evenly-spaced periodic checkpoints (`save_interval = total_episodes // 5`).
- **Why:** Ensures voice and navigation match actual 3D layout, guarantees patient followers cannot be permanently blocked by static medical personnel, and exposes the policy to realistic crowded hospital traffic.
- **Touches:** `sim_vision/python/voice/hospital_map.json`, `sim_vision/godot/scripts/rl_env_bridge.gd`, `sim_vision/godot/scripts/room.gd`, `sim_vision/godot/scenes/room.tscn`, `sim_vision/godot/scripts/hospital_npc.gd`, `src/training/train_visual_rl_in_the_wild.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 163 voice pytest unit tests pass (100% accuracy); 163 Godot GUT tests pass (18,873 asserts, 0 errors); launched 2,000-episode in-the-wild training run.

### D-048 — Discovery and Resolution of Visual Reward Hacking and 1,200-Episode Continuation
- **Status:** accepted
- **When:** 2026-10-31T08:00:00+07:00
- **Decided by:** human-directed inquiry
- **Trigger:** User observed that during the 2,000-episode in-the-wild training, episodic rewards reached unusually high values (+300 to +590) even when zero patients were helped (`Helped: 0`), suspecting reward hacking.
- **Root Cause Discovered:**
  1. **FOV Boundary Wiggling Exploit:** In `hospital_env_vision.py`, the visual spotting bonus (`REWARD_VISUAL_SPOTTED = +2.0`) was granted whenever `det["seen"] > 0.5` and `prev_has_det < 0.5`. Standing near an NPC and wiggling camera yaw across the FOV or occlusion boundary caused `seen` to toggle between 0 and 1 up to 595 times per episode, harvesting over +1,190 free visual reward points without completing any task.
  2. **Stagnation Timeout Suppression:** Each flicker toggle executed `self.steps_since_task = 0`, indefinitely resetting the stagnation timer and allowing the agent to hover and wiggle for the entire 300s shift without penalty.
  3. **Artificial Expansion Reward:** Toggling `seen` from 0 to 1 repeatedly reset `prev_vision_h = 0.0`, granting artificial `+1.5 * det["h"]` expansion bonuses on every re-detection (+247 points).
- **Decision:**
  1. **Single Spotting Bonus:** Introduced `self.patient_spotted_once`. The `+2.0` spotting bonus is granted strictly once per patient lifetime, resetting only when a new patient is spawned (`_spawn_next_patient()`). Subsequent flickers grant zero spotting bonus.
  2. **Monotonic Approach Expansion:** Changed approach reward to trigger strictly when `det["h"] > self.best_vision_h`, rewarding only true physical distance reduction (capped to $\approx +1.28$ across entire approach).
  3. **Strict Stagnation Enforcement:** Replaced flicker-based timer resets with physical approach threshold (`det["h"] > best_vision_h + 0.04`). Hovering or wiggling without closing distance triggers `STAGNATION_TIMEOUT` (-10.0 penalty) and early termination.
  4. **Continuation Training (Ep 801 -> Ep 2000):** Checkpoint analysis proved `PPOHospital-Wild-3floor_ep800.pt` was the peak uncorrupted checkpoint (2.90 patients/shift, 0.40 collisions). Successfully ran a 1,200-episode continuation run with the fix applied, saving regular checkpoints at `ep1040`, `ep1280`, `ep1520`, `ep1760`, and `ep2000`.
- **Why:** Eliminates all forms of visual reward exploitation, forces the policy to physically close in on patients for dialogue engagement, and produces authentic, grounded reward trajectories.
- **Touches:** `src/training/hospital_env_vision.py`, `src/training/train_visual_rl_in_the_wild.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** Verified on Episode 1169 seed: hacked reward collapsed from +1,247.5 to -28.5 with stagnation timeout at step 347 (69.4s); 1,200-episode continuation run completed cleanly; verified 5 periodic checkpoints saved on disk.

### D-049 — Resolution of Patient Avoidance Geometry, Reward Rebalancing, and 1,200-Ep Warmstart Training
- **Status:** accepted
- **When:** 2026-10-31T12:15:00+07:00
- **Decided by:** human-approved via /grill-me
- **Trigger:** During live 3D Godot inference, robot was observed actively avoiding patients and turning away even when very close. User requested /grill-me interview to diagnose root cause, rebalance rewards, and retrain.
- **Root Cause Discovered:**
  1. **Engagement Geometry Mismatch:** Python training used `TARGET_BOX_H = 0.85` ($d \le 1.35\text{m}$) while Godot expected `PATIENT_ENGAGE_DIST_M = 1.85m` ($h \ge 0.45$). For seated patients ($H \approx 1.25\text{m}$), $h$ never reached 0.85 without driving into $< 1.05\text{m}$.
  2. **LiDAR Collision Fear:** At $< 1.2\text{m}$, the patient collider in Godot registered on center LiDAR rays as a $0.3\text{m} - 0.5\text{m}$ obstacle. Trained with heavy collision penalty ($-5.0$), the policy panicked and executed evasive turns ($\omega = \pm 1.0$) and reverse braking, dropping the patient out of the $75^\circ$ FOV and falling back to search.
- **Decision:**
  1. **Synchronize Engagement Thresholds:** Set `TARGET_BOX_H = 0.50` and `PATIENT_TALK_DIST_M = 1.85m` in `hospital_env_vision.py` matching Godot's `PATIENT_ENGAGE_DIST_M = 1.85m`. At $1.85\text{m}$, robot engages naturally before entering the LiDAR obstacle avoidance buffer.
  2. **LiDAR Target Immunity:** In `teacher_action()`, suppress frontal obstacle repel when looking directly at patient during `APPROACH_PATIENT`.
  3. **Reward Rebalancing:** Emphasized core service completion: `REWARD_PATIENT_ENGAGED` increased from $+10.0 \to \mathbf{+25.0}$; `REWARD_GUIDE_ARRIVED` increased from $+30.0 \to \mathbf{+60.0}$. Zero artificial steering forces added to preserve pure RL learning.
  4. **Retraining Regimen:** Warm-start from uncorrupted peak checkpoint `ep800.pt` for a 1,200-episode run with 5 periodic checkpoints saved every 240 episodes.
- **Why:** Reconciles the geometric mismatch between Python simulation and Godot physics, removes false obstacle fear, and strongly incentivizes closing the distance to serve patients.
- **Touches:** `src/training/hospital_env_vision.py`, `src/training/train_visual_rl_in_the_wild.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** Verified environment stepping with $h \ge 0.50$ engagement; verified teacher action frontal approach without evasion; completed 1,200-episode training run on CUDA (Ep 1200 Eval score: +396.6, 3.87 tasks/eval); benchmark on 10 held-out seeds achieved: **3.50 ± 0.81 patients helped/shift (35 total)**, **100% visual spotting rate**, **0.0% stagnation rate**, **0.30 collisions/shift**, with balanced escorting across Floors 1–3 (F1: 14, F2: 12, F3: 9); best checkpoint saved at `experiments/runs/20261031_122003_PPOHospital-Wild-3floor/checkpoints/PPOHospital-Wild-3floor_best.pt`.

### D-050 — Realistic Elevator Cabin Interior, Doorway Cutout, and Randomized Inference Patient Spawning
- **Status:** accepted
- **When:** 2026-10-31T14:05:00+07:00
- **Decided by:** human-approved via /grill-me (`realistic_elevator_and_train_plan.md`)
- **Trigger:** User feedback: 1) Elevator functioned like a 2–3m teleport/swoop spot rather than a realistic elevator because waypoints were outside in the corridor and east walls were solid; 2) User confirmed training patient spawns are random while inference was prefixed, and requested fully randomized patient spawning in Godot inference; 3) Continue training for 1,200 episodes warm-started from the latest best checkpoint with inside-cabin elevator waypoints.
- **Root Cause & Architectural Enhancements:**
  1. **Elevator Waypoint & Solid East Wall:** `thang_may` waypoint was previously placed at `(9.0, 5.2)` in the hallway, almost 3 meters in front of the elevator door ($X = 11.88, Z = 4.6$). Furthermore, `WallEast` and `WallEastCollision` formed a continuous solid wall at $X = 12.1$ from $Z = -9$ to $+9$, physically blocking any agent from entering the elevator cabin ($X \in [11.88, 14.08]$). As a result, reaching $d \le 2.0\text{m}$ outside triggered a tween sliding the robot across 3 meters into the cabin.
  2. **Wall Doorway Cutouts:** In `room.tscn`, `room_floor2.tscn`, and `room_floor3.tscn`, split `WallEast` into `WallEastNorth` ($Z \in [-9.0, 3.55]$), `WallEastSouth` ($Z \in [5.65, 9.0]$), and `WallEastLintel` ($Y \in [2.4, 3.2]$ across $Z \in [3.55, 5.65]$). This creates a 2.1m clear doorway opening at $X = 12.1, Z \in [3.55, 5.65]$ for all three floors.
  3. **Physical Elevator Cabin:** In `elevator_station.tscn`, added static boundary collision shapes (`CollisionWallBack`, `CollisionWallLeft`, `CollisionWallRight`, `CollisionFloor`) around the elevator interior, turning the cabin into a real, enclosed physical standing room ($2.1\text{m} \times 2.2\text{m}$).
  4. **Inside-Cabin Waypoints & Re-Export:** Updated `ROOMS["thang_may"]["waypoint"]` and `DESTINATION_POSITIONS` to `(12.8, 4.6)` on Floor 1, `(62.8, 4.6)` on Floor 2, and `(112.8, 4.6)` on Floor 3. Re-exported `src/training/hospital_layout.json` via Godot headless tool (`export_nav_layout.gd`).
  5. **Natural Door-Opening & Transit Sequence:** In `main.gd`, approaching the elevator station within $\le 3.2\text{m}$ automatically triggers `station.open_doors()`. The robot and escorted patient navigate naturally into the cabin. Once inside ($d \le 1.4\text{m}$ to cabin waypoint), the doors close, transit audio and LED arrow updates play, positions shift to destination floor cabin, destination doors open, and robot and patient walk out into the hallway before doors close.
  6. **Randomized Inference Patient Spawning:** In `main.gd`, replaced fixed initial positioning with dynamic uniform sampling via `room.get_patient_spawn_position(1, _rng)` on every startup and reset, matching the Python training distribution.
  7. **Continuation Training (Ep 1201 -> 2400):** Warm-started 1,200 episodes from `PPOHospital-Wild-3floor_best.pt` with inside-cabin navigation.
- **Why:** Eliminates the jarring 3-meter swoop teleportation, provides authentic elevator door and interior mechanics, reconciles training vs inference patient spawning distributions, and trains the policy to master interior cabin entry.
- **Touches:** `sim_vision/godot/scenes/props/elevator_station.tscn`, `sim_vision/godot/scenes/room.tscn`, `sim_vision/godot/scenes/room_floor2.tscn`, `sim_vision/godot/scenes/room_floor3.tscn`, `sim_vision/godot/tools/export_nav_layout.gd`, `src/training/hospital_layout.json`, `sim_vision/godot/scripts/room.gd`, `sim_vision/godot/scripts/main.gd`, `src/training/hospital_env.py`, `src/training/hospital_env_vision.py`, `src/training/train_visual_rl_in_the_wild.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** Nav layout re-exported with 164 rects; Godot GUT test suite passed with 18/18 tests in `test_room.gd` (16,952 asserts, 0 failures); completed 1,200-episode in-the-wild continuation run (Ep 1201 to 2400) reaching Peak Reward $+693.0$ and 3,861 patients helped; held-out 10-shift benchmark achieved: **$R_{team} = +394.42 \pm 69.44$**, **3.40 ± 0.80 patients helped/shift (34 total)**, **100.0% visual spotting rate**, **8.10s mean search latency**, **0.0% stagnation rate**, **0.00 collisions/shift**, with balanced floor distribution (F1: 14, F2: 12, F3: 8); best checkpoint saved at `experiments/runs/20261031_140435_PPOHospital-Wild-3floor/checkpoints/PPOHospital-Wild-3floor_best.pt`.

### D-051 — Active-Search Shaping, Free Search Idleness Penalty, and 1,000-Ep Continuation Training
- **Status:** accepted
- **When:** 2026-10-31T15:55:00+07:00
- **Decided by:** human-directed (/grill-me aligned)
- **Trigger:** Observation in 3D inference: the robot did not actively seek out patients or utilize search time efficiently, sometimes idling or wandering aimlessly.
- **Root Cause:**
  1. The step penalty was only `PENALTY_STEP_TIME = -0.005`/step, creating negligible urgency during the `SEARCH_PATIENT` phase.
  2. Reward for spotting and engaging patients was modest relative to passive wandering time.
- **Decision:**
  1. **Free Search Idleness Penalty:** Introduced `PENALTY_FREE_SEARCH_TIME = -0.04`/step ($-0.20$/s) strictly applied while `task_phase == TaskPhase.SEARCH_PATIENT`. This penalizes wandering without a spotted patient, strongly incentivizing rapid corridor and lobby visual scanning.
  2. **Elevated Spotting & Exploration Incentives:** Boosted `REWARD_VISUAL_SPOTTED` from $+2.0 \to \mathbf{+5.0}$ and `REWARD_CELL_EXPLORATION` from $+0.05 \to \mathbf{+0.10}$.
  3. **High Service Delivery Value:** Doubled engagement reward `REWARD_PATIENT_ENGAGED_VIS` ($+25.0 \to \mathbf{+50.0}$) and increased delivery reward `REWARD_GUIDE_ARRIVED_VIS` ($+60.0 \to \mathbf{+80.0}$).
  4. **1,000-Episode Continuation Training:** Warm-started from Ep 2400 checkpoint (`PPOHospital-Wild-3floor_best.pt`) targeting Ep 2401 to 3400.
- **Why:** Re-aligns policy incentives so that idling or dawdling incurs continuous negative reward, while actively finding and escorting patients yields massive positive returns (+50.0 to +80.0).
- **Touches:** `src/training/hospital_env_vision.py`, `src/training/train_visual_rl_in_the_wild.py`, `scripts/plot_training_metrics.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** GPU smoke test verified on NVIDIA GeForce RTX 4060 Laptop GPU; completed 2,000-episode GPU training run (Ep 2401 -> Ep 4400) under `experiments/runs/20261031_160344_PPOHospital-Wild-3floor`; total 6,418 patients helped with peak reward reaching **+1198.0**; 10-shift held-out benchmark achieved: **$R_{team} = +565.31 \pm 110.68$**, **3.40 ± 0.80 patients helped/shift (34 total)**, **100.0% visual spotting rate**, **8.00s mean search latency**, **0.0% stagnation rate**, **0.20 collisions/shift**, with balanced multi-floor escorting (F1: 14, F2: 12, F3: 8); best checkpoint saved at `experiments/runs/20261031_160344_PPOHospital-Wild-3floor/checkpoints/PPOHospital-Wild-3floor_best.pt`.

### D-053 — 2,000-Episode Deep Convergence Training on NVIDIA RTX 4060 GPU
- **Status:** accepted
- **When:** 2026-10-31T21:42:00+07:00
- **Decided by:** human
- **Trigger:** User requested 2,000 more episodes of training following successful active-search convergence (+565.31 held-out reward).
- **Decision:**
  1. Warm-started from best checkpoint `experiments/runs/20261031_160344_PPOHospital-Wild-3floor/checkpoints/PPOHospital-Wild-3floor_best.pt`.
  2. Training span configured for **Ep 4401 $\to$ Ep 6400 (2,000 episodes)**.
  3. Running on CUDA via `.venv` with NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM).
  4. Saving 5 regular checkpoints every 400 episodes (`ep4800`, `ep5200`, `ep5600`, `ep6000`, `ep6400`) and continuous tracking in `metrics.jsonl`.
- **Why:** Deepens autonomous visual search heuristics, solidifies multi-floor elevator pathing under high NPC density, and pushes escort throughput beyond 3.5 patients/shift.
- **Touches:** `src/training/train_visual_rl_in_the_wild.py`, `scripts/plot_training_metrics.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** Completed 2,000-episode GPU training run (Ep 4401 -> Ep 6400) under `experiments/runs/20261031_214132_PPOHospital-Wild-3floor`; total 6,270 patients helped (cumulative 12,688 patients across both GPU runs); peak episode reward reached **+934.8**; held-out 10-shift diagnostic benchmark achieved: **$R_{team} = +551.02 \pm 104.44$**, **3.30 ± 0.78 patients helped/shift (33 total)**, **100.0% visual spotting rate**, **8.08s mean search latency**, **0.0% stagnation rate**, **0.10 collisions/shift**, with balanced multi-floor escorting (F1: 14, F2: 11, F3: 8); best checkpoint saved at `experiments/runs/20261031_214132_PPOHospital-Wild-3floor/checkpoints/PPOHospital-Wild-3floor_best.pt`.

### D-054 — Transition to Pure Vision (YOLOv6 Only, No LiDAR), 29-Dim Observation Architecture, and Sector HUD
- **Status:** accepted
- **When:** 2026-11-01T08:10:00+07:00
- **Decided by:** human-aligned via /grill-me
- **Trigger:** Transition from hybrid LiDAR + Vision to pure camera vision using the new fine-tuned 17-class model `yolo26n_hospital_robot_best_map92.6` (92.6% mAP). Eliminate physical LiDAR sensors completely.
- **Architectural Decisions:**
  1. **New Branch:** Dedicated branch `feat/pure-vision-yolo-no-lidar` isolated from LiDAR dependencies.
  2. **YOLO Model Upgrade:** Integrated production model `models/yolo26n_hospital_robot_v6_best/yolo26n_hospital_robot_best_map92.6.pt` detecting 17 hospital classes: `patient` (0), `doctor` (1), and 15 hospital obstacles/furniture classes (2..16).
  3. **29-Dimensional Pure-Vision Observation Space:** Stripped 17 LiDAR rays. Replaced with:
     - 5-Sector Visual Depth Map: Far-Left, Left, Center, Right, Far-Right obstacle heights $h \in [0, 1]$ (5 dims).
     - Dedicated Patient Channel: `[seen, cx_off, h, w]` (4 dims).
     - Dedicated Doctor Channel: `[seen, cx_off, h, w]` (4 dims).
     - Phase (4), Nav Goal (3), Kinematics (2), Tether (3), Door Clearance (4) $\to$ Total: 29 dims.
  4. **Doctor Evasion & Visual Collision Rewards:**
     - Strict doctor avoidance: $-5.0$ penalty for false engagement with medical personnel; robot only targets patients.
     - Visual obstacle proximity repulsion: $-0.50 \times h_{\text{obs}}$ when sector obstacle height $h > 0.50$.
     - Physical collision penalty: $-5.0$.
  5. **Top-Right POV Sector Overlay HUD:** Tab-key toggled HUD in Godot robot camera view displaying real-time 5-sector visual clearance bars (green/yellow/red) and target tracking bounding boxes.
  6. **2,000-Episode Training:** Fresh 29-dim PPO agent trained on NVIDIA RTX 4060 GPU with decaying teacher behavioral cloning warmup (first 200 episodes).
- **Why:** Realizes an authentic low-cost autonomous hospital robot that navigates and escorts solely with monocular RGB camera perception without expensive physical LiDAR scanners.
- **Touches:** `models/yolo26n_hospital_robot_v6_best/`, `sim_vision/python/detector.py`, `sim_vision/python/yolo_server.py`, `sim_vision/python/rl_policy_server.py`, `src/models/ppo_agent.py`, `src/training/hospital_env_vision.py`, `src/training/hospital_env_pure_vision.py`, `src/training/train_pure_vision_rl.py`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/view_overlay.gd`, `sim_vision/godot/scenes/room.tscn`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 4/4 tests in `tests/test_pure_vision_obs.py` pass; completed 2,000-episode GPU training run (Ep 1 -> Ep 2000) under `experiments/runs/20261101_090639_PPOHospital-PureVision` on NVIDIA GeForce RTX 4060 Laptop GPU; total 91 patients helped under pure visual guidance with zero physical LiDAR; visual spotting rate reached **97.3%**; peak episode reward reached **+61.4**; 4-panel training curves saved at `experiments/runs/20261101_090639_PPOHospital-PureVision/training_curves.png`; best checkpoint saved at `experiments/runs/20261101_090639_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt`; 10-shift held-out diagnostic benchmark completed (`spotting_rate: 60.0%`).

### D-056 — Escort-Focused Pure Vision Architecture, Goal Destination Routing Fix, and 1,500-Episode Fine-Tuning
- **Status:** accepted
- **When:** 2026-11-02T07:05:00+07:00
- **Decided by:** human-aligned via /grill-me
- **Trigger:** Diagnosis of simulation escort circling in the lobby and approach starvation from `logs/godot.log` and `logs/rl_policy_server.log`:
  1. Goal destination routing deadlock: Godot sent landmark elevator target `[12.8, 4.6]`, but `HospitalPureVisionEnv._get_obs()` reset target to `(0.0, 4.5)` in the lobby, causing the robot to turn 180 degrees away from the elevator and circle indefinitely.
  2. Approach gradient starvation: Bounding box expansion alone yielded only $\approx +7.0$ points over 100+ steps, insufficient to reliably pull the robot to engagement distance.
  3. UI telemetry desynchronization: In pure-vision mode, `InspectionPanel` defaulted `is_pure_vision` to `false`, displaying legacy 17-ray LiDAR tab by default, and `vision_client.gd` previously stripped `class_name` and `class_id` resulting in generic "person" labels.
- **Architectural Decisions:**
  1. **Strict Target Passthrough:** In `rl_policy_server.py` and `hospital_env_pure_vision.py`, stored Godot's explicit goal coordinates as `custom_nav_target`. `_get_obs()` prioritizes `custom_nav_target` over `_nav_final_target()`, completely eliminating the `(0.0, 4.5)` lobby circling deadlock.
  2. **Dense Approach Distance Shaping:** In `hospital_env_vision.py`, added continuous delta-distance approach shaping $+2.5 \times (d_{pat,prev} - d_{pat,curr})$ towards the patient during `APPROACH_PATIENT`, creating a smooth, strong gradient directly guiding the robot into dialogue engagement range.
  3. **Adaptive Tether Speed Gating & Graduated Penalties:**
     - Teacher action gates escort velocity: throttles to $0.15$ m/s when tether distance $> 2.5$m, and pauses ($v = 0.0$ m/s) when $> 3.5$m until the patient catches up.
     - Graduated tether penalties: $-1.0$/step for $d_{tether} > 3.0$m, $-3.0$/step for $d_{tether} > 4.0$m, and zero progress reward when $> 3.5$m.
  4. **Escort-Heavy Curriculum (70% Approach/Escort, 30% Patrol):**
     - In `HospitalPureVisionEnv.spawn_escort_curriculum()`: 70% of training episodes spawn the robot within $2.5$m–$4.2$m facing the patient in `APPROACH_PATIENT`, guaranteeing a 10× increase in escort training samples.
  5. **UI & HUD Pure-Vision Standardization:**
     - Defaulted `is_pure_vision = true` in `inspection_panel.gd` and called `set_pure_vision_mode(true)` unconditionally on ready, hiding the LiDAR tab by default.
     - Preserved `class_id` and `class_name` in `vision_client.gd` and `view_overlay.gd` to render role-specific colors (Cyan for patient, Orange for doctor, Amber for furniture).
     - Connected 5-sector visual clearance HUD (`FL, L, C, R, FR`) directly to the top-right camera view.
  6. **1,500-Episode GPU Fine-Tuning on NVIDIA RTX 4060:**
     - Initialized from `PPOHospital-PureVision_best.pt` with a 100-episode teacher warmup under `experiments/runs/20261102_070135_PPOHospital-PureVision`.
- **Why:** Ensures the pure-vision robot reliably approaches patients, engages dialogue, and navigates smooth multi-room escorts without drifting or tether breakage.
- **Touches:** `sim_vision/godot/scripts/inspection_panel.gd`, `sim_vision/godot/scripts/main.gd`, `sim_vision/godot/scripts/view_overlay.gd`, `sim_vision/godot/scripts/vision_client.gd`, `sim_vision/python/rl_policy_server.py`, `src/training/hospital_env_vision.py`, `src/training/hospital_env_pure_vision.py`, `src/training/train_pure_vision_rl.py`, `docs/AI-Human-logs/Vuong.md`
- **Evidence:** 4/4 tests in `tests/test_pure_vision_obs.py` passed; verified approach & escort simulation in Python: patient engaged in 12 steps (+48.1 rew) and escorted smoothly with tether $1.06$m (+54.6 rew); launched 1,500-episode GPU fine-tuning (`task-11628`): Ep 1 achieved **+638.7 reward**, **4 patients helped**, and **0 collisions**.

### Decision D-057: Dynamic Closest-Patient Acquisition and Dual Optical Arrival Trigger in Godot
- **Context:** Live testing showed that when the robot approached a patient directly in front of it, the patient did not start talking and `state` stayed in `APPROACH` indefinitely.
- **Root Cause:**
  1. `main.gd:human_distance()` only tracked a single static `current_target_patient` chosen randomly at startup across the entire floor. When the robot faced an encountered patient ($d=1.2$m), `human_distance()` returned the distance to the distant patient ($d=9.43$m), preventing transition to `ARRIVED`.
  2. Non-target patients had `pardon_enabled = true`, causing them to sidestep away rather than initiate dialogue.
- **Resolution:**
  1. Implemented `get_closest_available_patient()` in `main.gd` to dynamically acquire whichever patient is in physical/visual proximity ($d \le 3.5$m) and bind `current_target_patient = candidate_pat` with `pardon_enabled = false`.
  2. Implemented dual optical YOLO trigger in `_step_policy()`: transitions to `ARRIVED` either when $d_{human} \le 2.0$m or when YOLO bounding box height $h \ge 0.38$, $|cx| < 0.35$, and $d \le 2.6$m.
- **Touches:** [main.gd](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/scripts/main.gd), [npc.gd](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/scripts/npc.gd), [test_inspection_panel.gd](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/tests/test_inspection_panel.gd).
- **Evidence:** Verified 13/13 tests in `test_npc.gd` passed; isolated test suites in Godot confirmed immediate arrival trigger.

### Decision D-061: 2,000-Episode Fine-Tuning Execution on Real-World Sim-to-Real Rules
- **Status:** accepted
- **When:** 2026-11-03T07:32:00+07:00
- **Decided by:** human-directed, agent-executed
- **Trigger:** Request to finetune for 2,000 episodes under the newly established zero-oracle Sim-to-Real rules.
- **Execution & Results:**
  - Trained 2,000 episodes under `experiments/runs/20261102_204414_PPOHospital-PureVision` on NVIDIA RTX 4060 GPU.
  - Zero map raycasting in 5-sector depth (`obs[0..4]`), peripheral side-by-side companion escort tracking (`obs[22..24]`), and optical door detection (`obs[25..28]`).
  - Total patients escorted: **3,323 patients** (1.66 ± 1.28 patients/shift).
  - Peak reward reached: **+954.3** (Ep 810).
  - Visual spotting rate: **100.0%**.
  - Collision rate: **1.02 collisions/shift**.
  - Best checkpoint saved to [`experiments/runs/20261102_204414_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/experiments/runs/20261102_204414_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt).
- **Touches:** [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py), [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py), [`experiments/runs/20261102_204414_PPOHospital-PureVision/`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/experiments/runs/20261102_204414_PPOHospital-PureVision/).
- **Evidence:** All 2,000 episodes completed; verified with `test_disconnection_pure_vision.py` (3/3 passed) and Godot GUT tests (164/164 passed). Telemetry plot saved to `training_curves.png`.

### Decision D-062: Interactive Non-Headless Visual Renderer and Mock-Training Pipeline
- **Status:** accepted
- **When:** 2026-11-03T07:55:00+07:00
- **Decided by:** human-directed, agent-executed
- **Trigger:** User request: "make me a script to see the traninig env in place, non headless, a command to mock train (doesnt save the checkpoint, but other than that, everything is identical to the normal training + visual)".
- **Architecture & Implementation:**
  1. **Non-Headless GUI Renderer (`src/ui/visual_env_renderer.py`):**
     - Left Panel ($740 \times 740$): Top-down hospital floorplan rendering structural walls, doorways, robot chassis with heading arrow, translucent $75^\circ$ FOV camera cone, 5 radial optical clearance sectors, patient (with companion waypoint and elastic tether color-coded by tension), and patrolling doctor.
     - Top-Right Panel ($640 \times 360$): Simulated robot front camera view with perspective projection, YOLO bounding boxes (color-coded for Patient/Doctor/Doors), and 5-sector visual clearance HUD.
     - Bottom-Right Panel ($640 \times 380$): Live telemetry and PPO training HUD displaying episode/step counters, linear and angular velocities, reward breakdown, and PPO loss metrics (policy loss, value loss, entropy, BC loss, learning rate).
     - Hotkey Controls: `[SPACE]` pause/resume, `[+] / [-]` speed up/slow down, `[F]` toggle active floor (Floor 1/2/3/Auto), `[Q / ESC]` clean exit.
  2. **Mock-Train Safety Contract (`scripts/visual_mock_train.py` & `src/training/train_pure_vision_rl.py`):**
     - Normal full training pipeline is executed identically (environment stepping, observation gathering, actor-critic forward pass, GAE advantage computation, and PPO gradient updates).
     - Checkpoint saving is strictly suppressed (`mock_train=True`), guaranteeing existing `.pt` models are completely untouched.
     - Auto-selects latest trained checkpoint as initial weights (`PPOHospital-PureVision_best.pt`) so the user can immediately observe the trained policy navigating and learning in real-time.
- **Touches:** [`src/ui/visual_env_renderer.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/ui/visual_env_renderer.py), [`scripts/visual_mock_train.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/scripts/visual_mock_train.py), [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py), [`docs/AI-Human-logs/Vuong.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/AI-Human-logs/Vuong.md).
- **Evidence:** Verified execution with `scripts/visual_mock_train.py --smoke --fps 60` (exit code 0); confirmed GUI rendering, PPO loss updates, and verified `experiments/runs/.../checkpoints/` remained completely empty.

### Decision D-063: Continuous Multi-Patient Pure-Vision Shift Architecture and Anti-Farming Contract
- **Status:** accepted
- **When:** 2026-11-03T08:50:00+07:00
- **Decided by:** human-aligned via /grill-me
- **Trigger:** Observation of training vs inference divergence:
  1. Patients previously spawned right next to the robot 70% of the time (`spawn_escort_curriculum()`), preventing the robot from learning full-shift search, approach, and retrieval from the entrance.
  2. Single-patient environment allowed reward farming on already-escorted patients.
  3. No mechanism existed for returning to Floor 1 after delivering patients to Floor 2 or Floor 3.
- **Architectural Implementation:**
  1. **Strict Entrance Spawn**: Robot always starts every shift at Floor 1 Main Entrance / Dock (`ROBOT_DOCK` $X = -10.6\text{m}, Z = 0.0\text{m}$) facing inward. Artificial close-range teleports are completely disabled.
  2. **Multi-Patient Population**: Floor 1 is populated with 6 patients matching `HOSPITAL_NPCS` from `hospital_layout.json` (seated in lobby chairs and standing near rooms) with $\pm 0.3\text{m}$ randomization.
  3. **Multi-Patient Visual Lock**: In `SEARCH_PATIENT`, YOLO scans all unserved patients on current floor and dynamically acquires whichever one enters camera FOV with highest bounding box height (`h`), locking them as active patient.
  4. **Strict Anti-Reward-Farming Contract**: Delivered patients are marked `SERVED = True`. They remain in place as non-responsive characters, but YOLO filters them out during search and no dialogue or reward can ever be triggered on them again.
  5. **Topological Return to Floor 1**: After delivery on Floor 2 or 3, mission goal sets to `thang_may_t{floor}` (Elevator). When reaching the elevator, the robot transitions down to Floor 1 with $+10.0$ return elevator reward and resumes visual patrol for remaining unserved patients in the lobby.
  6. **Telemetry & Visual HUD**: Floorplan distinguishes unserved (Amber), active locked (Bright Amber `*`), and served patients (Muted Green `[SERVED]`), with live count `Patients Escorted: X / 6` and `RETURN_ELEVATOR` phase badge.
- **Touches:** [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py), [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py), [`src/ui/visual_env_renderer.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/ui/visual_env_renderer.py), [`tests/test_multi_patient_shift.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/tests/test_multi_patient_shift.py), [`docs/AI-Human-logs/Vuong.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/AI-Human-logs/Vuong.md).
- **Evidence:** 4/4 tests passed in `tests/test_multi_patient_shift.py`; 3/3 passed in `tests/test_disconnection_pure_vision.py`; smoke test executed cleanly via `scripts/visual_mock_train.py --smoke --fps 60`.### Decision D-064: Fix Phantom Reward Spike on Visual Acquisition and Seated Patient Obstruction Snap
- **Status:** accepted
- **When:** 2026-11-03T09:45:00+07:00
- **Decided by:** human-directed bug identification, agent root-cause diagnosis & fix
- **Trigger:** User identified anomaly during 2,000-episode run: "check the trainig env for bugs, the reward so high while it doesnt helped any patient" (`R_team: +1875.3` to `+1944.3` on shifts with `Helped: 0/6` terminating with `STAGNATION_TIMEOUT`).
- **Root Cause & Architectural Fixes:**
  1. **Ground-Truth Euclidean Distance Leak (+3,485 Jump):**
     - *Cause:* In `SEARCH_PATIENT`, `self.patient_x = -999.0`. When `_compute_yolo_det()` spotted a patient and assigned `self.patient_x = 3.0m`, `HospitalVisualShiftEnv.step()` evaluated $2.5 \times (d_{\text{prev}} - d_{\text{curr}}) = 2.5 \times (1400m - 6m) = +3,485$ reward in a single step!
     - *Fix:* Gated in `HospitalVisualShiftEnv.step()`: disabled ground-truth distance shaping when `getattr(self, "pure_vision", False)` is True, relying strictly on pinhole bounding box expansion ($\Delta H$). Gated single-patient engagement, single-patient arrival, and single-patient respawn so they do not conflict with multi-patient shifts.
  2. **Seated Patient Furniture Trapping (`is_blocked == True`):**
     - *Cause:* All 6 patients in `HOSPITAL_NPCS` are seated in lobby chairs (`(-4.55, 4.20)`, `(3.05, 4.20)`, etc.) inside obstacle bounding boxes. When dialogue engaged, candidate steps of size $0.11\,\text{m}$ in `_step_escorted_patient()` were rejected because the entire neighborhood was inside the chair. The patient was trapped in the chair, the robot walked away alone, tether broke ($d > 4.0\,\text{m}$), $-3.0$ penalties accumulated, and stagnation timeout triggered.
     - *Fix:* Upon engagement, patient automatically stands up (`p["seated"] = False`) and snaps out of the chair onto the nearest legal, navigable floor space using `nav_grid(floor).snap()`, enabling free companion formation walking. Added candidate angle fallbacks ($\pm 45^\circ, \pm 90^\circ$) and visual tether occlusion memory (15 steps / 3.0s).
  3. **Strict Reward Mathematical Synchronization:**
     - Synced `self.cumulative_team_reward` to the exact running sum of clipped step rewards (`self.cumulative_pure_vision_reward += reward`), guaranteeing telemetry matches actual PPO advantage buffers.
- **Touches:** [`src/training/hospital_env_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_vision.py), [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py), [`tests/test_multi_patient_shift.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/tests/test_multi_patient_shift.py), [`docs/bugs/20261103_pure_vision_multi_patient_reward_and_standup_bugs.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/bugs/20261103_pure_vision_multi_patient_reward_and_standup_bugs.md), [`docs/AI-Human-logs/Vuong.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/AI-Human-logs/Vuong.md).
### Decision D-065: Cross-Floor Elevator Transit vs Final Room Delivery Disambiguation
- **Status:** accepted
- **When:** 2026-11-03T10:00:00+07:00
- **Decided by:** human observation in visual mock training ("wait, i just re watch the mock training, its only escort patient to the elevator not other rooms in upper floor? is it my wrong seeing or a bugs?")
- **Root Cause & Fix:**
  - *Bug:* When escorting a patient to an upper floor clinic (e.g. Room 205 on Floor 2), the intermediate navigation target was `self.goal_key = "thang_may"`. When arriving at the elevator, the environment checked `floor_of(goal_pos) != cur_fl`. Since `"thang_may"` is on Floor 1, `1 != 1` was False, bypassing elevator transit. The room arrival block then matched `self.goal_key == "thang_may"` and erroneously treated the elevator threshold as the final clinic destination, marking the patient `served = True` and abandoning them at the elevator door!
  - *Fix:* Integrated `resolve_final_target(action, slot, floor)` to explicitly disambiguate intermediate elevator transit from final clinic delivery:
    ```python
    final_target = resolve_final_target(self.scenario_action, self.scenario_slot, self.scenario_floor)
    if self.goal_key in ("thang_may", "thang_may_t2", "thang_may_t3") and final_target != self.goal_key:
        # Cross-floor elevator ride:
        dest_fl = floor_of(DESTINATION_POSITIONS.get(final_target, (0.0, 4.5))[0])
        shift = float(dest_fl - cur_fl) * 50.0
        self.robot_x += shift; self.patient_x += shift; p["x"] += shift
        self.goal_key = final_target  # Switches target to Room 205 on Floor 2!
        reward += REWARD_ELEVATOR_TRANSIT
    else:
        # Arrived at actual clinic room (e.g. Room 205):
        p["served"] = True
        self.patients_helped += 1
        reward += REWARD_GUIDE_ARRIVED_VIS
        self.returning_to_lobby = True  # Routes back to elevator to return to Floor 1!
    ```
- **Touches:** [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py), [`tests/test_multi_patient_shift.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/tests/test_multi_patient_shift.py), [`docs/bugs/20261103_pure_vision_multi_patient_reward_and_standup_bugs.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/bugs/20261103_pure_vision_multi_patient_reward_and_standup_bugs.md), [`docs/AI-Human-logs/Vuong.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/AI-Human-logs/Vuong.md).
- **Evidence:** 10/10 unit tests pass in `tests/test_multi_patient_shift.py` and `tests/test_disconnection_pure_vision.py`; end-to-end simulation verified Patient 0 rides elevator to Floor 2, is delivered inside Room 205, and robot returns to Floor 1 lobby.

### Decision D-068: Pure-Vision PPO Reward Function Rebalancing, Corridor Progress Shaping, and Gentle Fine-Tuning Execution
- **Status:** accepted
- **When:** 2026-11-04T10:35:00+07:00
- **Decided by:** human-agent interactive interview via `/grill-me`
- **Trigger:** Analytical review of Run 2 (Ep 2001->2924) revealed an adverse performance trend after Ep 2300 (average delivered patients dropped from 1.90 in Q1 to 0.57-0.74 in Q3/Q4; zero-helped shifts surged to 55.8%) caused by extreme doctor proximity penalties (-5.0/step creating -1,300 reward cliffs) and passive corridor search stagnation once teacher warmup decayed to zero.
- **Modifications Applied:**
  1. **Doctor Penalty Rebalancing:** Reduced `PENALTY_DOCTOR_PROXIMITY` from `-5.0` to `-0.30` per step in [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py) to eliminate the catastrophic reward cliff while preserving gentle evasion pressure.
  2. **Escort Arrival Reinforcement:** Increased `REWARD_GUIDE_ARRIVED_VIS` from `+80.0` to `+150.0` to provide a dominating positive reinforcement gradient for completing patient escort tasks.
  3. **Corridor Exploration Progress Shaping:** Added `patrol_prog = np.clip(prev_nav - curr_nav, -0.5, 0.5)` with `WEIGHT_PATROL_PROGRESS = 0.25` and progressive stagnation reduction when moving towards active patrol waypoints during `SEARCH_PATIENT`, pulling the robot along public corridors rather than circling in corners.
  4. **Teacher Assistance Floor:** Updated BC warmup in [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py) to decay from `0.20` down to a persistent `0.06` (6%) teacher floor, providing minimal directional guidance during difficult multi-floor search states.
  5. **Gentle Fine-Tuning Learning Rate:** Resumed training from the proven peak checkpoint (`experiments/runs/20261103_100811_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt`, Peak +860.3) using a gentle initial learning rate of `3e-5` (min `1e-5`) with 10,000 steps per PPO update epoch across 1,000 continuous multi-patient episodes.
- **Touches:** [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py), [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py), [`docs/AI-Human-logs/Vuong.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/AI-Human-logs/Vuong.md).
- **Evidence:** 16/16 test suite passes; production training run `20261104_103458_PPOHospital-PureVision` completed all 1,000 episodes (Ep 2001–3000) on CUDA RTX 4060 GPU:
  - **All-Time Peak Episode Reward:** **$+1137.9$** (Ep 2012, all-time record, up from $+860.3$).
  - **Mean Episode Reward:** **$+256.1$** (final 20-episode average $+257.2$, non-degrading across all quartiles).
  - **Total Patients Delivered:** **1,443 patients** ($1.44 \pm 1.11$ per shift, $+56.2\%$ throughput boost).
  - **Mean Collisions Per Shift:** **1.93** collisions (reduced from 2.26).
  - **Best Checkpoint:** [`experiments/runs/20261104_103458_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/experiments/runs/20261104_103458_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt).

### Decision D-069: Expand Stagnation Timeout Window to 90s, Implement Search Visited Cells Reset, and Linear Progressive Multi-Patient Reward Escalation
- **Status:** accepted
- **When:** 2026-11-04T19:30:00+07:00
- **Decided by:** human-agent interactive interview via `/grill-me`
- **Trigger:** Diagnosis of shift terminations showed that 100% of episodes ended in `STAGNATION_TIMEOUT` after 1–2 patients due to a tight 60.0s (300 step) window and a design bug where `self.visited_cells` was never refreshed after returning from upper-floor deliveries.
- **Modifications Applied:**
  1. **Expanded Stagnation Window (90.0s / 450 Steps):** Increased default `stagnation_timeout_s` from `60.0` to `90.0` in [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py) and [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py), allowing sufficient physical transit time (~65s) to navigate from upper floors back to the Floor 1 lobby and pan the 70° FOV camera across waiting benches.
  2. **Inter-Patient Exploration Refresh:** In `HospitalPureVisionEnv.step()`, when exiting the elevator back on Floor 1 (`returning_to_lobby` transition) or completing a Floor 1 delivery, cleared `self.visited_cells` and reset `self.steps_since_task = 0`. This guarantees active walking through previously seen corridors earns exploration credit and resets the stagnation timer.
  3. **Linear Progressive Reward Escalation:** Replaced flat $+150.0$ escort reward with an escalating formula:
     $$\text{Reward}_{\text{delivery}} = 150.0 \times (1.0 + 0.20 \times \text{patients\_helped})$$
     earning $+150, +180, +210, +240, +270, +300$ for patients 1 through 6, and increased full shift completion bonus from $+100.0$ to $+250.0$.
  4. **Fine-Tuning Execution:** Launch 1,000-episode continuation run from the record-breaking checkpoint (`20261104_103458_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt`, Peak +1137.9) with $\text{lr} = 3\times 10^{-5}$ and 6% teacher guidance floor.
- **Touches:** [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py), [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py), [`docs/AI-Human-logs/Vuong.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/AI-Human-logs/Vuong.md).
- **Evidence:** 16/16 unit tests passed in `test_multi_patient_shift.py`, `test_3d_simulation_improvements.py`, and `test_hospital_env_vision.py`; smoke test passed cleanly.

### Decision D-070: Expand Shift Duration to 3,500 Steps (700.0s) to Accommodate 6 Patients and Soften Tether Penalty Cliff
- **Status:** accepted
- **When:** 2026-11-04T19:55:00+07:00
- **Decided by:** human request ("incraese the step size to accomadte for 6 patient helped") & agent analysis
- **Trigger:** Observation that shifts capped uniformly at 1,500 steps (300.0s) because each full round-trip escort to Floor 2 takes ~100s (~500 steps), capping maximum throughput at 2–3 patients. Additionally, broken tether penalties (-3.0/step) created occasional negative reward spikes (-2500) during failed engagements.
- **Modifications Applied:**
  1. **Shift Step Budget Expansion (3,500 Steps / 700.0s):** Increased default `shift_duration_s` from `300.0` to `700.0` (3,500 environment steps at $dt=0.2\,\text{s}$) in [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py) and [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py). Early termination still triggers as soon as all 6 patients are delivered (`ALL_PATIENTS_SERVED` with $+250.0$ bonus).
  2. **Softened Tether Penalty:** In [`src/training/hospital_env_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_vision.py), softened `r_tether` when $d_{\text{tether}} > 4.0\,\text{m}$ from `-3.0` to `-0.50` and $d_{\text{tether}} > 3.0\,\text{m}$ from `-1.0` to `-0.25`, eliminating the multi-thousand reward cliff on broken tethers.
  3. **Launch Continuation Run:** Restarted 1,000-episode fine-tuning from `20261104_103458_PPOHospital-PureVision_best.pt` (Peak +1137.9) with 3,500 max steps per shift on CUDA RTX 4060 GPU.
- **Touches:** [`src/training/hospital_env_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_vision.py), [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py), [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py), [`docs/AI-Human-logs/Vuong.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/AI-Human-logs/Vuong.md).
- **Evidence:** 16/16 unit tests passed cleanly; production run `20261104_195749_PPOHospital-PureVision` completed 609 episodes (Ep 3001–3609) before laptop reboot:
  - **All-Time Peak Episode Reward:** **$+1237.2** (Ep 3014, highest in project history).
  - **Mean Episode Reward:** **$+703.0** (up from $+256.1$, a $+174.5\%$ increase).
  - **Average Patients Helped / Shift:** **$3.74 \pm 1.69$** (up from $1.44$).
  - **High-Volume Shifts ($\ge 4$ Patients):** **78.0%** (475 / 609 shifts delivered 4, 5, or 6 patients).
  - **Full Shift Clear:** 1 shift achieved `ALL_PATIENTS_SERVED` (all 6 patients escorted).
  - **Checkpoints:** [`PPOHospital-PureVision_best.pt`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/experiments/runs/20261104_195749_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt) and [`PPOHospital-PureVision_ep3600.pt`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/experiments/runs/20261104_195749_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_ep3600.pt).

### Decision D-071: Fix Speak-Through-Wall via 3D Physics Line-of-Sight Raycasts and Resolve Trash Bin Snag via Authoritative Unstick Maneuver & Ground-Contact Depth Scaling
- **Status:** accepted
- **When:** 2026-11-05T12:35:00+07:00
- **Decided by:** human observation of 3D simulation bugs & agent diagnosis via `/grill-me`
- **Trigger:** Observation in 3D inference:
  1. Robot and seated patients initiated conversation through solid partition walls when Euclidean distance $d \le 1.85\,\text{m}$.
  2. Robot became snagged against medical waste bins at `(6.13, 1.65)` because a weak escape pulse ($0.35\,\text{s}$ at $0.22\,\text{m/s}$ reversing only $7.7\,\text{cm}$) kept it inside the bin's collision volume, while waste bin visual depth was only $0.45$ (below the $0.50$ avoidance threshold).
- **Modifications Applied:**
  1. **3D Physics Line-of-Sight Enforcement (`main.gd` & `npc.gd`):** Implemented `has_line_of_sight(from, to)` using Godot's `PhysicsRayQueryParameters3D` direct raycast. In both `main.gd` (`_step_policy()`) and `npc.gd` (`_process()`), dialogue engagement (`Config.RobotState.ARRIVED`, `Npc.NOTICE`) is strictly forbidden unless raycast between robot camera and patient torso is completely clear of walls and doors.
  2. **Authoritative Unstick Maneuver (`robot.gd`):** Upgraded physical stuck escape in `robot.gd`:
     - Faster detection threshold: `STUCK_TRIGGER_S = 0.6` (was $0.8\,\text{s}$).
     - Extended escape pulse: `ESCAPE_DURATION_S = 0.85` (was $0.35\,\text{s}$) at $0.28\,\text{m/s}$ backward speed, delivering $\approx 24\,\text{cm}$ of clean reverse displacement.
     - Stronger yaw rotation: turn rate increased to $1.25 \times \omega_{\max}$ ($\approx 58^\circ$ heading shift away from the obstacle).
  3. **Small Obstacle Ground-Contact Proximity Boost (`hospital_env_pure_vision.py`):** In `compute_5_sector_depth()`, for obstacles with nominal height $\le 0.55\,\text{m}$ (trash bins, chair legs) with ground contact $y_2 > 0.55$, boosted pseudo-depth to $\ge 0.70$, guaranteeing the RL policy server immediately registers strong repulsive forces before physical contact occurs.
- **Touches:** [`sim_vision/godot/scripts/main.gd`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/scripts/main.gd), [`sim_vision/godot/scripts/npc.gd`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/scripts/npc.gd), [`sim_vision/godot/scripts/robot.gd`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/scripts/robot.gd), [`src/training/hospital_env_pure_vision.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/hospital_env_pure_vision.py), [`tests/test_3d_simulation_improvements.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/tests/test_3d_simulation_improvements.py), [`docs/AI-Human-logs/Vuong.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/AI-Human-logs/Vuong.md).
- **Evidence:** 17/17 pytest suite passed; Godot 4.7.2 headless project check exited cleanly with code 0.

### Decision D-072: Launch 500-Episode Fine-Tuning Continuation (Ep 3610–4109) from All-Time Peak Checkpoint (+1237.2)
- **Status:** accepted
- **When:** 2026-11-05T13:45:00+07:00
- **Decided by:** human-agent interactive interview via `/grill-me`
- **Trigger:** Resuming training after laptop reboot and 3D simulation bug fixes (line-of-sight raycast & authoritative trash bin unstick) to further consolidate multi-patient escort throughput and test the enhanced low-obstacle depth representation over 500 additional episodes.
- **Modifications Applied:**
  1. **Base Model:** Loaded weights from `experiments/runs/20261104_195749_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt` (all-time peak reward $+1237.2$).
  2. **Episode Span:** Ep 3610 to Ep 4109 (500 episodes).
  3. **Shift Configuration:** 3,500 max steps ($700.0\,\text{s}$) per shift, 90.0s stagnation timeout with lobby refresh, and linear progressive reward escalation $+150 \times (1 + 0.2k)$ for patients 1–6.
  4. **Optimization:** Gentle AdamW learning rate $\text{lr} = 3\times 10^{-5}$ (min $1\times 10^{-5}$) with persistent 6% teacher guidance floor.
- **Touches:** [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py), [`docs/AI-Human-logs/Vuong.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/AI-Human-logs/Vuong.md).
- **Evidence:** 17/17 pytest suite passed; production training run `20261105_134525_PPOHospital-PureVision` completed all 500 episodes (Ep 3610–4109) on CUDA RTX 4060 GPU:
  - **Peak Episode Reward:** **$+1119.6$** (Ep 3649).
  - **Mean Episode Reward:** **$+645.6$** (final 20-episode average **$+789.3$**, steady upward convergence).
  - **Average Patients Helped / Shift:** **$3.48 \pm 1.81$** (final 20-episode average **$4.15$** patients/shift).
  - **Mean Collisions Per Shift:** **$3.69$** collisions (reduced by 13.2% from 4.25).
  - **Total Patients Escorted:** **1,742 patients** across 500 shifts.
  - **Checkpoints:** [`PPOHospital-PureVision_best.pt`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/experiments/runs/20261105_134525_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt) and [`PPOHospital-PureVision_ep4109.pt`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/experiments/runs/20261105_134525_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_ep4109.pt).

### Decision D-073: Launch 500-Episode High-Autonomy Continuation Run (Ep 4110–4609) with Annealed Teacher Floor (6% -> 2%)
- **Status:** accepted
- **When:** 2026-11-06T07:50:00+07:00
- **Decided by:** human-agent interactive interview via `/grill-me`
- **Trigger:** Building upon the success of Ep 3610–4109 (which achieved 4.15 patients/shift and +789.3 reward in the final 20 episodes), user approved another 500-episode run to target consistent 5–6 patients/shift convergence while transitioning the policy toward complete autonomy.
- **Modifications Applied:**
  1. **Base Model:** Loaded weights from `experiments/runs/20261105_134525_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt`.
  2. **Episode Span:** Ep 4110 to Ep 4609 (500 episodes).
  3. **High-Autonomy Annealing:** Added `--teacher-start 0.06` and `--teacher-floor 0.02` to [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py), smoothly reducing teacher guidance from 6% down to 2% across the 500 episodes so the policy relies almost exclusively on its own learned visual policy gradient.
  4. **Shift & Optimizer Settings:** 3,500 max steps ($700.0\,\text{s}$) per shift, 10,000 steps per PPO epoch, AdamW $\text{lr} = 3\times 10^{-5}$ ($\text{lr}_{\min} = 1\times 10^{-5}$).
- **Touches:** [`src/training/train_pure_vision_rl.py`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/src/training/train_pure_vision_rl.py), [`docs/AI-Human-logs/Vuong.md`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/docs/AI-Human-logs/Vuong.md).
- **Evidence:** Production training run `20261106_075341_PPOHospital-PureVision` completed all 500 episodes (Ep 4110–4609) on CUDA RTX 4060 GPU:
  - **Peak Episode Reward:** **$+1050.8$** (Ep 4188).
  - **Mean Episode Reward:** **$+757.0$** (final 20-episode average **$+849.4$**, up from $+789.3$).
  - **Average Patients Helped / Shift:** **$4.04 \pm 1.39$** (final 20-episode average **$4.75$** patients/shift, regularly delivering 5/6 patients).
  - **Total Patients Escorted:** **2,018 patients** across 500 shifts (+15.8% volume increase).
  - **Visual Spotting Rate:** **100.0%**.
  - **Teacher Guidance Annealing:** Successfully transitioned from 6.0% down to 2.0% floor across 500 episodes with zero policy collapse.
  - **Checkpoints:** [`PPOHospital-PureVision_best.pt`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/experiments/runs/20261106_075341_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_best.pt) and [`PPOHospital-PureVision_ep4609.pt`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/experiments/runs/20261106_075341_PPOHospital-PureVision/checkpoints/PPOHospital-PureVision_ep4609.pt).












