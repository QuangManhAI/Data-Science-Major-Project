### D-032 — Main Entrance + Robot Dock, Grid Path Planner, Polite NPCs, and the Strongest Full-Mode PPO Policy
- **Status:** accepted
- **When:** 2026-10-26T17:30:00+07:00
- **Decided by:** human
- **Trigger:** "cải thiện tiếp model train full lên để có RL phiên bản mạnh nhất ! ... làm cái bệnh viện có cái cửa ra vào từ bên ngoài và thường robot sẽ xuất phát ở đó."
- **Decision:**
  1. **Main entrance + robot dock (`room.tscn`):** West end of the floor-1 corridor now has a sliding door ("LỐI VÀO CHÍNH") with a canopy, an outdoor plaza and a sign. `SlidingDoor.robot_can_open = false`: people open it, the robot treats it as a wall, so it never leaves the building. `Entrance/RobotDock` sits at (-10.6, 0) facing the corridor. In Godot every shift starts at the dock; training starts there in 75% of episodes (`DOCK_START_PROB`) and mid-shift elsewhere otherwise. `tools/export_nav_layout.gd` exports the dock and the entrance door (and skips Area3D trigger shapes).
  2. **Geodesic navigation (`hospital_env.NavGrid`):** Replaces the hand-written door/corridor routing. A 0.2 m grid of walls and furniture is inflated by the robot radius (0.34 m). Cells closer than 0.8 m to an obstacle cost up to 4x, so paths keep to the middle of passages. Dijkstra distance fields are cached per target. The sub-goal is the farthest point up to 2 m along the path that is in straight-line sight with at least 0.55 m clearance. Observation `obs[25]` is the remaining path length, and progress shaping and the stagnation clock use path length, so detours around furniture are no longer punished. `rl_policy_server.py` calls the same `set_nav_target()`, and training observations are now rebuilt at the end of each step exactly like the server builds them (no stale sub-goal).
  3. **NPC behaviour (Python env and `hospital_npc.gd`):**
     - NPCs only side-step when they actually block the robot's lane (0.75 m), to 0.8 m off its line.
     - They never cross in front of the robot, and never pick a side-step spot within 1.2 m of a doorway.
     - A side-step is abandoned after 4 s, with a 3 s cooldown before the next one.
     - NPCs never step within 0.9 m of the robot. A patrolling NPC kept waiting for 2 s turns back to its other waypoint.
     - The robot can always back out of an NPC that walked into it.
     - These rules fix the doctor and robot deadlocks in front of room 110.
  4. **Sim/train parity:**
     - Godot patient spawns use the training regions: 60% corridor, 40% lobby, on the robot's floor (`Room.get_patient_spawn_position`).
     - CALL_STAFF targets the nearest nurse or doctor in both Godot and Python.
     - When the staff member is on another floor, the robot rides the elevator alone.
     - The Godot stagnation timer also covers SEARCH/APPROACH (it previously looped forever) and measures progress toward the elevator for cross-floor legs.
     - NLU targets `room_<n>` and `dept_<slug>` are normalized in `RLEnvBridge`. Previously every "phòng XXX" request fell back to the reception desk.
  5. **Training protocol:**
     - Round 1: 4 runs (3 fine-tunes from the previous full model plus 1 from scratch with teacher BC).
     - Round 2: 4 fine-tunes from the round-1 bests on the final env, evaluated every 500 episodes on 100 held-out seeds, with early stopping (patience 20).
     - Every new `best.pt` was snapshotted (`experiments/checkpoints/round2/`).
     - Selection used a separate 1000-seed test on fresh seeds (base 500,000,000) never used in training or best.pt selection.
  6. **Chosen model:** round-2 run A, episode 16000 → `experiments/checkpoints/PPOHospital-full_strongest.pt`. It is now the default checkpoint of `rl_policy_server.py` and `run_simulation.py --controller rl`.
- **Why:** The previous policy stalled behind furniture (about 5% of shifts stuck) because routing was heuristic and shaping was Euclidean. Godot also spawned patients where training never did. A sliding-door start at the entrance matches how a hospital guide robot is actually deployed.
- **Touches:** `src/training/hospital_env.py`, `src/training/hospital_layout.json`, `src/training/train_solo_patient.py` (checkpoint names `PPOHospital-<env_mode>_*.pt`), `sim_vision/python/rl_policy_server.py`, `run_simulation.py`, `sim_vision/godot/scenes/room.tscn`, `sim_vision/godot/scripts/{main,room,sliding_door,hospital_npc,human,rl_env_bridge,view_overlay}.gd`, `sim_vision/godot/tools/export_nav_layout.gd`, `tests/test_nav_layout.py`, `sim_vision/godot/tests/{test_room,test_main,test_sliding_door,test_rl_env}.gd`
- **Evidence:** 1000 fresh seeds, full mode (8 NPCs), deterministic policy:

  | Model | Tasks / shift | Collisions / shift | Zero-collision shifts | Stuck | Score (R_team − collisions) |
  |---|---|---|---|---|---|
  | Previous full model (`20261025_225529…/PPOHospital-solo_best.pt`) | 3.71 | 0.62 | 86.6% | 5.0% | 110.6 |
  | **Strongest (round 2 A, ep 16000)** | **4.02** | 0.89 | 85.4% | **0.0%** | **120.7** |
  | Round 2 D, ep 6000 | 4.05 | 1.18 | 84.0% | 0.0% | 120.7 |
  | Round 2 C, ep 19500 | 4.04 | 1.05 | 83.6% | 0.0% | 120.7 |
  | Round 2 B, ep 1000 (fewest collisions) | 3.97 | 0.65 | 88.4% | 0.2% | 119.2 |

  In solo mode (500 seeds) the strongest model reaches 3.91 tasks, 0.09 collisions and 0% stuck, against 3.61, 0.18 and 3.2% for the previous model. Python tests 30/30 and Godot GUT tests 157/157 pass. An 8-minute Godot full-mode run produced 32 task events with no stuck aborts. It covered answers, a staff call, escorts that reached their destination, and an elevator escort from floor 1 to the floor-2 lobby, after which the robot kept serving patients on floor 2.

### D-074 — Pure-Vision V2: Camera-Depth Observation, Redesigned Reward, No Hard-Coded Unstick
- **Status:** accepted
- **When:** 2026-11-09T12:40:00+07:00
- **Decided by:** human ("tách branch riêng, thiết kế lại ... bỏ cái hardcode - thiết kế lại thưởng"), agent analysis
- **Trigger:** Review of `PPOHospital-PureVision_best.pt` (Vuong, D-073, ep 4587, commit `75245f0`) found that the robot escorted patients at 0.27 m/s (about half the 0.52 m/s cap), never served all 6 patients, and relied on a hard-coded "back up and spin" maneuver in `robot.gd`.
- **Root causes found in V1:**
  1. **Obstacles were invisible during training.** `_get_obs` only put the doctor into the 5 obstacle sectors, so the policy never saw walls or furniture. That is why Godot needed the D-071 unstick maneuver. At inference, the server fed all YOLO obstacles into the sectors, which the policy had never seen in training.
  2. **The reward paid for slow escorts.** The escort earned +0.15 per step while the patient stayed 0.9–2.5 m away, so a slower escort earned more per metre. Rewards were also clipped at +100, so the +150·(1+0.2k) delivery escalation (D-069) and the +250 full-shift bonus never reached the agent. With gamma 0.99, a delivery about 400 steps away was worth about 2% of its value.
  3. **The patient follower got stuck.** The patient walked in a straight line to an ahead-right point and stuck behind furniture. Out of camera view, the "tether broken" flag fired, and the teacher and policy crept at 0.08–0.15 m/s. In Godot the patient walked at 0.48 m/s, slower than the robot's 0.52 m/s cap.
  4. **The teacher's repulsion sign was inverted** (+omega means turn left, but left obstacles produced +omega).
  5. **PPO stored and evaluated different actions.** The buffer stored the blended, clipped action, but the log-prob was computed on the raw sample. Time-limit truncation was treated as terminal, and `best.pt` was chosen by the training avg20 (`best_eval` stayed `-inf`).
- **Decision:**
  1. **Observation, 21-D** (`src/training/hospital_env_pure_vision_v2.py`):
     - 5 depth-proximity sectors, built from 15 rays across the 86.4° camera HFOV with a 4 m range.
     - Patient box [4], phase one-hot [4], nav goal [3], kinematics [2].
     - Tether [3]: last-seen distance, visible flag, seconds unseen.
     - The doctor channel and the door channel are removed. Geometric depth covers both.
     - On the real robot, the depth comes from an RGB-D camera or a monocular depth model. In Godot it comes from `Robot.get_depth_readings()`.
     - The tracked patient is masked out of the depth (they are the goal, not an obstacle): `Robot.depth_ignore` in Godot, `_patient_discs(skip_target_in=...)` in the sim.
  2. **Reward:**
     - Time cost −0.005 per step.
     - Path-length progress shaping: 0.2 per metre for approach, escort and return; 0.05 per metre for patrol. It uses the undiscounted difference: with gamma<1, a negative potential pays (1−γ)·K·d for standing still.
     - Event rewards: +0.5 spotted, +2 engaged, +0.5 elevator transit, +10 delivered, +5 all served, −2 stagnation.
     - Safety: −1 per new collision, −0.1 per step of sustained contact, up to −0.05 when the front depth is under 0.5 m.
     - Tether is penalty-only: −0.05 per metre beyond 2 m, and no progress credit when the patient is more than 3 m behind. No reward clip below ±20.
     - gamma = 0.995.
  3. **Escort companion** (sim and `rl_env_bridge.gd`):
     - The patient walks at 0.70 m/s to (1.2 m ahead, 0.45 m right) when the straight path is clear; otherwise they walk the robot's breadcrumb trail.
     - In the sim they also slide around the robot body.
     - A patient stuck for more than 2 s is moved back to the nearest crumb.
  4. **Hard-coded unstick removed** from `robot.gd`. The robot now only integrates (v, omega), exactly like training.
  5. **Training** (`src/training/train_pure_vision_v2.py`):
     - The stored action is the evaluated action.
     - Truncated shifts bootstrap V(s_T).
     - The teacher's BC weight decays from 0.5 to 0 over 300 episodes.
     - `best.pt` is chosen by a deterministic 10-seed eval every 50 episodes.
     - Learning rate 3e-4 with cosine decay, 8192 steps per update.
  6. **Server:** `rl_policy_server.py` detects the 21-D policy from the checkpoint and uses the `depth` field sent by `rl_policy_client.gd`. If it is missing, it falls back to the LiDAR rays inside the HFOV.
- **Is the policy redundant with full vision?**
  - With depth alone, a ~60-line rule-based controller (the V2 teacher) already serves 3.5 patients per shift (10/50 full shifts). V1 RL served 4.2 per shift and never cleared a full shift.
  - Most of the V1 RL's value was compensating for missing perception. The doctor and door channels and the class-height sector heuristic become redundant.
  - RL still earns its place: V2 RL beats the rule baseline by a wide margin, mainly on escort pacing and dead-lock recovery.
  - The baseline is kept as the bar any future policy must beat.
- **Touches:** `src/training/hospital_env_pure_vision_v2.py` (new), `src/training/train_pure_vision_v2.py` (new), `src/training/hospital_env_pure_vision.py` (patient escort speed is configurable), `src/models/ppo_agent.py` (`OBS_DIM_PURE_VISION_V2`), `sim_vision/python/rl_policy_server.py`, `sim_vision/godot/scripts/{robot,rl_env_bridge,rl_policy_client,main}.gd`, `tests/test_pure_vision_v2.py` (new), `experiments/checkpoints/PPOHospital-PureVisionV2_best.pt`
- **Evidence:**
  - Run `experiments/runs/20261109_113048_PPOHospital-PureVisionV2`: 2000 episodes on CPU (about 40 min). `best.pt` is from ep 1950.
  - Test set: 50 fresh seeds (base 123,000,000, never used for selection), deterministic policy. Escort v is the mean robot speed during GUIDE_ESCORT.

  | System | DR | Patients/shift | 6/6 shifts | Collisions | Escort v (m/s) | Stagnation |
  |---|---|---|---|---|---|---|
  | V1 RL (Vuong ep 4587) | off | 4.20 | 0/50 | 6.06 | 0.265 | 11/50 |
  | V1 RL (Vuong ep 4587) | on | 4.00 | 0/50 | 5.46 | 0.259 | 15/50 |
  | V2 rule-based teacher | off | 3.48 | 10/50 | 4.66 | 0.329 | 40/50 |
  | V2 rule-based teacher | on | 3.36 | 8/50 | 12.08 | 0.341 | 42/50 |
  | **V2 RL** | off | **6.00** | **50/50** | **0.00** | **0.492** | 0/50 |
  | **V2 RL** | on | **5.86** | **48/50** | **0.04** | **0.492** | 2/50 |

  - Tests: Python 68 pass and 1 fail (`test_pure_vision_obs.py::test_doctor_channel_and_avoidance`, which already fails on `75245f0` because D-068 lowered the doctor penalty to −0.3). The new `tests/test_pure_vision_v2.py` passes 9/9. Godot GUT passes 168/168.
- **Limits:**
  - The hospital layout and the 6 patient seats are fixed (only ±0.3 m jitter), so the policy may have memorised the building. Generalisation to a new layout is untested.
  - The V2 policy has not yet been run end-to-end in Godot with real YOLO boxes.
  - In Godot the patient's straight-path check raycasts through the robot body, so they follow the trail more often than in the sim.
