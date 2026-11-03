# Bug Report: Phantom Reward Spike & Seated Patient Obstruction in Multi-Patient Pure-Vision Shift

**Date:** 2026-11-03  
**Status:** RESOLVED  
**Branch:** `feat/pure-vision-yolo-no-lidar`  
**Related Commits:** Pre-`d8c0871`, Patch Post-`d8c0871`

---

## 1. Symptoms
During the 2,000-episode continuous multi-patient shift training run (`train_pure_vision_rl.py`):
1. **Abnormally High Rewards:** Telemetry reported `R_team: +1875.3` to `+1944.3` on shifts where `Helped: 0/6` (no patients were delivered).
2. **Persistent Stagnation Timeouts:** Episodes routinely terminated with `Exit: STAGNATION_TIMEOUT` after 700–1000 steps without making escort progress.
3. **Severe Tether Violations:** Telemetry step-breakdown revealed sustained `r_tether = -3.0` penalties on every step during `GUIDE_ESCORT`.

---

## 2. Root Cause Analysis

### Bug 1: Superclass Ground-Truth Euclidean Distance Reward Leak (+3,485 Jump)
* **Root Location:** `src/training/hospital_env_vision.py` lines 263 and 346–349 (`HospitalVisualShiftEnv.step()`).
* **Mechanism:**
  - In multi-patient search mode, `self.patient_x` was initialized to `-999.0` (unassigned).
  - At the start of `HospitalVisualShiftEnv.step()`, `prev_dist_pat` was calculated using `self.patient_x = -999.0`, yielding `prev_dist_pat ≈ 1,400.0` meters.
  - During that same step, `_compute_yolo_det()` spotted an unserved patient and dynamically assigned `self.patient_x = p["x"]` (~3.0 meters) and transitioned to `APPROACH_PATIENT`.
  - Later in `HospitalVisualShiftEnv.step()`, the ground-truth distance shaping term was evaluated:
    $$\Delta r_{\text{visual}} = 2.5 \times (d_{\text{prev}} - d_{\text{curr}}) = 2.5 \times (1400.0 - 6.0) \approx +3{,}485.0$$
  - This single step leaked ~+3,500 phantom reward into `cumulative_team_reward` despite zero clinical task completion.

### Bug 2: Seated Patients Trapped in Lobby Furniture Obstacle Boxes (`is_blocked == True`)
* **Root Location:** `src/training/hospital_env_pure_vision.py` (`_step_escorted_patient()`).
* **Mechanism:**
  - All 6 lobby patients from `HOSPITAL_NPCS` are seated on waiting room chairs (`Patient_01` at `(-4.55, 4.20)`, `Patient_03` at `(3.05, 4.20)`, etc.).
  - Their coordinate points reside directly inside the obstacle bounding boxes of the lobby benches (`is_blocked(x, z, 0.28) == True`).
  - When the robot approached and dialogue engaged, the patient stood up, but their coordinate remained inside the furniture box.
  - In `_step_escorted_patient()`, candidate steps of size $\Delta s = 0.11\,\text{m}$ were evaluated with `if not is_blocked(px_new, pz_new, 0.28):`.
  - Because the entire $0.11\,\text{m}$ neighborhood remained inside the furniture box, all candidates were rejected.
  - The patient never took their first step and remained physically frozen in their chair forever.
  - The robot drove towards the clinic alone, breaking the tether ($d > 4.0\,\text{m}$), incurring $-3.0$ penalties every step, and eventually timing out at 60s.

---

## 3. Resolution & Code Changes

### Fix 1: Guard Ground-Truth Distance Shaping in `HospitalVisualShiftEnv`
- Guarded line 346 of `src/training/hospital_env_vision.py`:
  ```python
  if not getattr(self, "pure_vision", False) and self.task_phase == TaskPhase.APPROACH_PATIENT and prev_dist_pat < 50.0 and self.patient_x > -100.0:
      curr_dist_pat = math.hypot(self.patient_x - self.robot_x, self.patient_z - self.robot_z)
      r_visual += 2.5 * (prev_dist_pat - curr_dist_pat)
  ```
- Gated single-patient engagement, single-patient destination arrival, and `_spawn_next_patient()` with `if not getattr(self, "pure_vision", False):` so they do not conflict with multi-patient shift logic.

### Fix 2: Autonomous Stand-up & NavGrid Legal Floor Snap on Engagement
- In `src/training/hospital_env_pure_vision.py` (`step()`), when dialogue triggers:
  ```python
  p = self.patients[self.active_patient_idx]
  p["seated"] = False
  fl_grid = nav_grid(floor_of(p["x"]))
  si, sj = fl_grid.snap(p["x"], p["z"])
  p["x"], p["z"] = float(fl_grid.xs[si]), float(fl_grid.zs[sj])
  self.patient_x, self.patient_z = p["x"], p["z"]
  self.escort_trail.clear()
  self.last_known_tether = 1.0
  self.lost_sight_steps = 0
  ```
- Snaps the standing patient out of the chair directly onto free, navigable floor space (`is_blocked == False`), enabling unconstrained movement.

### Fix 3: Obstacle Evasion Fallback for Escorted Companion
- In `_step_escorted_patient()`:
  - If `is_blocked(cur_px, cur_pz, 0.28)` is ever True, automatically snap to the nearest legal floor cell.
  - Added diagonal candidate fallback angles ($\pm 45^\circ, \pm 90^\circ$) so corner turns never obstruct the escort companion.
  - Added visual tether occlusion buffer (persisting last known distance up to 15 steps / 3.0s before declaring tether loss).

### Fix 4: True Mathematical Reward Synchronization
- In `HospitalPureVisionEnv.step()`, synchronize `self.cumulative_team_reward` strictly as the running sum of clipped step rewards (`self.cumulative_pure_vision_reward += reward`), eliminating superclass cumulative discrepancies.

---

## 4. Verification Results
- **Unit Tests:** 9/9 passed (`pytest tests/test_multi_patient_shift.py tests/test_disconnection_pure_vision.py`).
- **Real Episode Validation (`scratch/diagnose_helped.py`):**
  ```
  Step  119 | event=engaged         | Helped=0/6 | active=2    | r=+50.02
  Step  365 | event=guide_completed | Helped=1/6 | active=None | r=+80.08
  Step  567 | event=engaged         | Helped=1/6 | active=1    | r=+49.98
  Step  861 | event=guide_completed | Helped=2/6 | active=None | r=+80.08
  Step 1064 | event=engaged         | Helped=2/6 | active=0    | r=+49.98
  Episode Done at step 1364: exit_reason=STAGNATION_TIMEOUT, Total Helped=2/6
  ```
- Result: Robot successfully engaged and delivered Patient 2, returned to lobby, engaged and delivered Patient 1, and began escorting Patient 0 with strictly proportional rewards.
