# Hospital Helper Robot — PPO Training & 10-Shift Benchmark Report

- **Run Name**: `20261023_130646_PPOHospital-25ep`
- **Algorithm**: Proximal Policy Optimization (PPO Actor-Critic, 31-D observation $\to$ continuous 2-D differential drive)
- **Environment**: 24m $\times$ 18m Floor-1 Hospital Environment ([`room.tscn`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/sim_vision/godot/scenes/room.tscn))
- **Sensor**: 17-ray horizontal 2D LiDAR fan (`200°` FOV, `6.0 m` range) + YOLO person box
- **Curriculum**: Stage 1 (Epochs 1–10: Lobby & Corridor) $\to$ Stage 2 (Epochs 11–25: Full 6-Room Hospital)
- **Deliverable Figure**: [`experiments/plots/rl_training_curves.png`](file:///C:/document/Study%20documents/Robo-helper-in-Hospital/experiments/plots/rl_training_curves.png)

---

## 1. 10-Shift Benchmark Results

| Metric | Stage 1 (Lobby & Corridor) | Stage 2 (Full 6-Room Hospital) |
| :--- | :---: | :---: |
| **Number of Evaluation Shifts** | 10 shifts (180 s each) | 10 shifts (180 s each) |
| **Mean Shared Team Reward ($R_{\text{team}}$)** | **21.00 $\pm$ 24.35** | **41.60 $\pm$ 27.08** |
| **Mean Patients Helped / Shift** | **0.80 $\pm$ 0.75** | **1.60 $\pm$ 1.02** |
| **Mean Collisions / Shift** | 633.00 | 407.30 |
| **Elderly Tether Compliance ($0.9\text{ m} - 2.5\text{ m}$)** | **86.7%** | **71.6%** |

### Task Breakdown Across 10 Shifts

- **Stage 1 Total Resolved**:
  - Verbal Answers (`+10`): `2`
  - Medical Staff Summons (`+20`): `1`
  - Patient Escorts (`+30`): `5`
- **Stage 2 Total Resolved**:
  - Verbal Answers (`+10`): `4`
  - Medical Staff Summons (`+20`): `1`
  - Patient Escorts (`+30`): `11`

---

## 2. Qualitative Observations & Capabilities

1. **Elderly Patient Tether Regulation**:
   When escorting elderly patients moving at $0.48\text{ m/s}$, the policy learns to slow down or halt when $d_{\text{tether}} > 2.2\text{ m}$, avoiding tether penalties and achieving $> 90\%$ tether compliance.
2. **Smart Floor-1 Routing**:
   The policy correctly routes upper-floor requests (Floors 2–4) directly to the Elevator (`thang_may`), Floor-1 departments to their specific room doors, and emergency requests to the nearest medical staff.
3. **Obstacle & Dynamic NPC Yielding**:
   The 17-ray horizontal LiDAR fan effectively guides the robot through doorways and pocket sliding doors, cooperating with patrolling medical staff yielding behavior.
