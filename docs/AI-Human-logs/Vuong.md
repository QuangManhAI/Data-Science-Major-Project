# Vuong Decision Log — AI-Assisted Architecture Records

- **Motivation/Background**: Captures architectural, toolchain, and model selection choices made during AI-assisted development to ensure long-term auditability and prevent unrecorded drift.
- **Purpose**: Maintain an immutable, attributable provenance log of decisions for the hospital navigation robot project.
- **Overview Pipeline**: Follows the decision log convention defined in `agents/templates/DECISION_LOG_TEMPLATE.md` and metadata header rules in `agents/rules/MD_CONVENTION.md`.
- **Detailed Plan**: §1 Decision Index Table; §2 Decision Log Entries (D-001 to D-019).
- **References**: [agents/templates/DECISION_LOG_TEMPLATE.md](../../agents/templates/DECISION_LOG_TEMPLATE.md), [agents/rules/MD_CONVENTION.md](../../agents/rules/MD_CONVENTION.md).
- **Created**: 2026-10-21T21:25:00+07:00
- **Last Updated**: 2026-10-23T11:23:00+07:00

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
| [D-014](#d-014--expansion-to-floor-1-vietnamese-hospital-environment) | Expansion to Floor-1 Vietnamese hospital environment (24m x 18m) | human | accepted |
| [D-015](#d-015--hospital-room-enrichment-medical-pedal-waste-bins-and-refined-collisions) | Hospital room enrichment, medical pedal waste bins, and refined collisions | human | accepted |
| [D-016](#d-016--camera-viewpoint-height-correction-and-automatic-ceiling-culling) | Camera viewpoint height correction and automatic ceiling culling | agent-proposed, human-approved | accepted |
| [D-017](#d-017--vietnamese-medical-staff-npcs-waiting-patients-and-clinical-facility-props) | Vietnamese medical staff NPCs, waiting patients, and clinical facility props | human | accepted |
| [D-018](#d-018--stt-startup-timeout-resilience-and-rocketbox-surface-material-mapping) | STT startup timeout resilience and Rocketbox surface material mapping | human | accepted |
| [D-019](#d-019--elderly-patient-variants-and-walking-canes-for-old-patient-dominated-hospital) | Elderly patient variants and walking canes for old-patient-dominated hospital | human | accepted |
| [D-020](#d-020--implement-phase-1-rl-training-plan-with-shared-team-reward-and-elderly-tether-pacing) | Implement Phase-1 RL training plan with shared team reward and elderly tether pacing | human | accepted |
| [D-021](#d-021--25-epoch-ppo-training-run-on-cuda-and-10-shift-comprehensive-benchmark) | 25-Epoch PPO training run on CUDA and 10-shift comprehensive benchmark | human | accepted |
| [D-022](#d-022--ad-camera-rotation-and-multi-npc-hospital-speech-support) | A/D camera rotation and multi-NPC hospital speech support | human | accepted |
| [D-023](#d-023--fix-seated-patient-arrival-trigger-and-implement-two-stage-doorway-waypoint-navigation) | Fix seated patient arrival trigger and implement two-stage doorway waypoint navigation | human | accepted |
| [D-024](#d-024--implement-gradio-web-dashboard-for-rl-training-control-live-telemetry-and-history-inspection) | Implement Gradio web dashboard for RL training control, live telemetry, and history inspection | human | accepted |

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
