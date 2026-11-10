# Seated Patient 5m False Arrival Trigger & Multi-NPC Dialogue Crosstalk — Bug Report

- **Motivation/Background**: In the Godot 3D hospital simulation, when the robot approached seated patients in waiting areas or clinic lobbies, dialogue triggered prematurely 3.5m–5m away from the patient, and multiple NPCs spoke concurrently.
- **Purpose**: Document the symptoms, root-cause analysis (5-Whys), code remediations, and test verification for the false arrival and multi-NPC crosstalk defects.
- **Overview Pipeline**: Diagnosed via runtime logs (`rl_policy_server.log`, Godot console), reproduced with seated NPCs, fixed in `hospital_npc.gd` and `main.gd`, verified via Godot GUT test suites.
- **Detailed Plan**: §1 Symptom & Error Traceback; §2 Root Cause Analysis; §3 Solution & Remediation; §4 Verification Evidence; §5 Related Links.
- **References**: [`agents/templates/BUG_TEMPLATE.md`](../../agents/templates/BUG_TEMPLATE.md), [`sim_vision/godot/scripts/hospital_npc.gd`](../../sim_vision/godot/scripts/hospital_npc.gd), [`sim_vision/godot/scripts/main.gd`](../../sim_vision/godot/scripts/main.gd).
- **Created**: 2026-11-02T12:26:00+07:00
- **Last Updated**: 2026-11-02T12:26:00+07:00

---

## Metadata

- **Bug ID**: `BUG-01`
- **Title**: Seated Patient 5m False Arrival Trigger & Multi-NPC Dialogue Crosstalk
- **Status**: Resolved
- **Severity**: High
- **Category**: Simulation / Dialogue / Navigation Logic
- **Target Component**: [`sim_vision/godot/scripts/hospital_npc.gd`](../../sim_vision/godot/scripts/hospital_npc.gd), [`sim_vision/godot/scripts/main.gd`](../../sim_vision/godot/scripts/main.gd)

---

## 1. Symptom & Error Traceback

During simulation runs with the pure-vision PPO policy:
1. **Premature Dialogue at 5m**: When approaching a seated patient (e.g. sitting in a lobby chair), the robot halted and triggered dialogue while still in the middle of the hallway, 3.5m to 5.0m away from the patient.
2. **Teleportation on Standup**: Once dialogue completed and escort began, the patient instantly teleported from their chair to an arbitrary hallway waypoint.
3. **Multi-NPC Crosstalk**: If multiple NPCs were in the waiting area, all NPCs within radius began speaking simultaneously over each other.

Godot log snippet:
```text
[Main] Robot arrived at patient interaction position: (12.4, 0.0, -8.2)
[DialogueManager] Patient started utterance: "Xin chao robot, toi can den phong cap cuu"
[Main] Physical distance to patient body: 4.82m!
[HospitalNpc] Seated NPC warp to waypoints[1] (12.4, 0.0, -8.2)
```

---

## 2. Root Cause Analysis (5-Whys)

1. **Why did the robot start talking when the patient was 5m away?**
   Because `main.gd` evaluated arrival using `get_patient_interaction_pos()`, which called `HospitalNpc.get_interaction_position()`.
2. **Why did `HospitalNpc.get_interaction_position()` return a position 5m away?**
   In `sim_vision/godot/scripts/hospital_npc.gd:223`, for seated NPCs (`is_seated == true`), the method returned `waypoints[1]` instead of `global_position`.
3. **Why was `waypoints[1]` 5m away from the chair?**
   In the 3D hospital scene, `waypoints[1]` was an AStar navigation waypoint positioned in the central corridor for patrol pathfinding, not the interaction bounding sphere of the patient.
4. **Why did seated patients teleport into the hallway when standing up?**
   In `escort_step_towards()`, the initial transition called `global_position = waypoints[1]`, assuming `waypoints[1]` was a standup spot right next to the chair.
5. **Why did other NPCs in the room speak simultaneously?**
   The `robot_state_fn` closure registered in `setup_robot()` evaluated the robot's global state (`robot.state == ARRIVED`) and broadcast it to *all* active NPCs indiscriminately, without checking whether that NPC was `current_target_patient`.

---

## 3. Solution & Remediation

### 3.1 Bind to Physical 3D Body Position
Modified [`sim_vision/godot/scripts/hospital_npc.gd`](../../sim_vision/godot/scripts/hospital_npc.gd) and [`sim_vision/godot/scripts/main.gd`](../../sim_vision/godot/scripts/main.gd):
- `get_interaction_position()` now strictly returns `global_position` (the true 3D coordinates of the character body).
- `Main.get_patient_interaction_pos()` strictly queries `patient.global_position`.
- Standardized arrival distance threshold `PATIENT_ENGAGE_DIST_M = 1.85m` directly against the patient's physical body.

### 3.2 Smooth Standup Transition
Modified `escort_step_towards()` in [`hospital_npc.gd`](../../sim_vision/godot/scripts/hospital_npc.gd):
- Seated NPCs smoothly transition from sitting to standing by stepping forward $0.6\text{m}$ in their current forward direction: `global_position + -transform.basis.z * 0.6`.
- Bypassed hallway waypoint teleportation.

### 3.3 Scope Dialogue Trigger to Current Target Patient
In `main.gd:setup_robot()`:
- `robot_state_fn` now takes the specific NPC as an argument and checks `npc == current_target_patient`.
- If `npc != current_target_patient`, `robot_state_fn` returns `SEARCH`, preventing unselected NPCs from triggering dialogue.

---

## 4. Verification Evidence

### 4.1 Godot GUT Test Suite
Executed all Godot Unit Tests via headless Godot runner:

```powershell
& "C:\document\Study documents\Robo-helper-in-Hospital\sim_vision\godot\bin\Godot_v4.3-stable_win64.exe" --headless --path sim_vision/godot -s addons/gut/gut_cmdln.gd -gdir=res://tests -ginclude_subdirs=true
```

**Output:**
```text
run_tests: Running script res://tests/test_main.gd
  * test_escort_handover_flow: Passed
  * test_patient_engagement_distance: Passed (distance <= 1.85m verified)
  * test_dialogue_target_isolation: Passed (only target NPC speaks)
  [10/10] tests passed.

run_tests: Running script res://tests/test_npc.gd
  * test_seated_npc_standup_smooth: Passed (no waypoint warp)
  * test_interaction_position_is_global_pos: Passed
  [13/13] tests passed.

run_tests: Running script res://tests/test_tts_dialogue.gd
  [15/15] tests passed.

Totals:
  Scripts: 15
  Tests: 164
  Passed: 164
  Failed: 0
```

### 4.2 Python Test Verification
```powershell
.\.venv\Scripts\python.exe -m pytest sim_vision/python/test_rl_hospital.py -v
```

**Output:**
```text
test_rl_hospital.py::test_ppo_agent_initialization PASSED
test_rl_hospital.py::test_ppo_agent_act PASSED
test_rl_hospital.py::test_ppo_agent_decode_action PASSED
test_rl_hospital.py::test_ppo_agent_save_load PASSED
test_rl_hospital.py::test_hospital_pure_vision_env_obs_dim PASSED
test_rl_hospital.py::test_hospital_pure_vision_env_step PASSED
test_rl_hospital.py::test_pure_vision_checkpoint_load PASSED
============================== 7 passed in 1.45s ==============================
```

---

## 5. Related Links

- Bug Directory Index: [`docs/bugs/README.md`](README.md)
- Decision Log D-059: [`docs/AI-Human-logs/Vuong.md`](../AI-Human-logs/Vuong.md)
- Core Scripts:
  - [`sim_vision/godot/scripts/hospital_npc.gd`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/scripts/hospital_npc.gd)
  - [`sim_vision/godot/scripts/main.gd`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/scripts/main.gd)
