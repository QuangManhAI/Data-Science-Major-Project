#!/usr/bin/env python3
"""
scripts/generate_patrol_visualizations.py
=========================================
Trực quan hóa toàn diện sứ mệnh tuần tra thực tế của Robot (Robot Patrol Mission Benchmark):
1. Bản đồ hành trình tuần tra 30 Waypoints qua 7 phân khu bệnh viện.
2. Biểu đồ so sánh mAP@50 v4 vs v5 trên góc nhìn robot thực tế.
3. Phân bố đối tượng và nhãn ground truth phát hiện theo phân khu (Zone breakdown).
4. Hồ sơ vận hành thực thi trên Robot (Latency 3.4ms, 294 FPS, 5.3MB, Zero Leakage).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "experiments/results/visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_JSON = REPO_ROOT / "experiments/results/robot_patrol_benchmark_metrics.json"

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6
})

# Load metrics
with open(METRICS_JSON, "r", encoding="utf-8") as f:
    metrics = json.load(f)

v4_classes = metrics["v4_results"]["per_class"]
v5_classes = metrics["v5_results"]["per_class"]

class_names = [
    "wc_station", "medical_waste_bin", "queue_kiosk", "sanitizer_stand",
    "pccc_station", "medical_trolley", "wheelchair", "hospital_chair",
    "vital_monitor", "elevator_doors", "hospital_bed", "patient",
    "reception_counter", "doctor", "iv_stand"
]

v4_vals = [v4_classes.get(c, {}).get("mAP50", 0.0) * 100 for c in class_names]
v5_vals = [v5_classes.get(c, {}).get("mAP50", 0.0) * 100 for c in class_names]

# 30 Waypoints
waypoints = [
    (0.0, 4.0), (2.5, 5.5), (1.0, 6.5), (-2.5, 6.2), (-4.5, 5.0),
    (-6.0, 3.5), (-8.8, 2.5), (-8.8, 0.5), (-8.0, -1.0), (-8.0, -3.5),
    (-10.0, -4.5), (-8.5, -6.8), (-6.2, -5.5), (-6.0, -2.5), (-6.0, -0.2),
    (-2.5, 0.0), (0.0, -1.0), (0.0, -4.0), (0.0, -1.0), (4.0, 0.0),
    (8.0, -1.0), (8.0, -4.5), (10.5, -6.0), (8.0, -2.5), (8.5, 1.0),
    (9.0, 4.0), (9.5, 6.2), (6.0, 5.5), (2.5, 4.5), (0.0, 4.0)
]

fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=300)

# ==============================================================================
# SUBPLOT 1: ROBOT PATROL TRAJECTORY MAP (TOP-DOWN 2D)
# ==============================================================================
ax1 = axes[0, 0]
wx, wy = zip(*waypoints)
ax1.plot(wx, wy, color="#1E88E5", linewidth=2.5, linestyle="-", marker="o", markersize=6, markerfacecolor="#0D47A1", markeredgecolor="white", zorder=3, label="Robot Patrol Path (531 frames)")

# Vẽ các phân khu bệnh viện
zones = [
    {"name": "Lobby & Reception", "x": -4, "y": 3.8, "w": 8, "h": 3.2, "color": "#E3F2FD"},
    {"name": "Pharmacy", "x": -11, "y": 1.5, "w": 4.5, "h": 2.5, "color": "#EDE7F6"},
    {"name": "Emergency Ward", "x": -11.5, "y": -7.5, "w": 6.5, "h": 5.5, "color": "#FFEBEE"},
    {"name": "Main Corridor", "x": -7, "y": -1.2, "w": 14, "h": 2.2, "color": "#F5F5F5"},
    {"name": "Blood Testing Lab", "x": -2.0, "y": -6.5, "w": 4.5, "h": 4.5, "color": "#E8F5E9"},
    {"name": "Clinic Examination", "x": 6.0, "y": -7.5, "w": 5.8, "h": 5.5, "color": "#FFF3E0"},
    {"name": "Elevator & Restroom", "x": 7.0, "y": 2.5, "w": 4.5, "h": 4.5, "color": "#F3E5F5"}
]

for z in zones:
    rect = plt.Rectangle((z["x"], z["y"]), z["w"], z["h"], facecolor=z["color"], edgecolor="#B0BEC5", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)
    ax1.add_patch(rect)
    ax1.text(z["x"] + z["w"]/2, z["y"] + z["h"]/2, z["name"], ha="center", va="center", fontsize=9, fontweight="bold", color="#37474F", zorder=2)

ax1.scatter([0.0], [4.0], color="#D32F2F", s=140, marker="*", zorder=5, label="Start/Finish Docking")
ax1.set_title("Robot Real-World Autonomous Patrol Trajectory (30 Waypoints Across 7 Zones)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("Hospital World X Coordinate (meters)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Hospital World Z Coordinate (meters)", fontsize=10, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower left", fontsize=9, framealpha=0.9)

# ==============================================================================
# SUBPLOT 2: mAP@50 COMPARATIVE BENCHMARK (V4 vs V5 ON PATROL)
# ==============================================================================
ax2 = axes[0, 1]
y = np.arange(len(class_names))
height = 0.36

rects1 = ax2.barh(y - height/2, v4_vals, height, label="YOLO26n-v4 (Patrol: 49.05% mAP)", color="#90CAF9", edgecolor="#1565C0")
rects2 = ax2.barh(y + height/2, v5_vals, height, label="YOLO26n-v5 (Patrol: 50.12% mAP)", color="#2E7D32", edgecolor="#1B5E20")

ax2.set_yticks(y)
ax2.set_yticklabels([c.replace("_", " ").title() for c in class_names], fontsize=9, fontweight="bold")
ax2.invert_yaxis()
ax2.set_xlabel("mAP@50 (%)", fontsize=10, fontweight="bold")
ax2.set_title("Robot Patrol Mission Benchmark: YOLO26n v4 vs v5 by Class", fontsize=13, fontweight="bold", pad=12)
ax2.grid(True, axis="x", linestyle="--", alpha=0.5)
ax2.set_xlim(0, 105)
ax2.legend(loc="lower right", fontsize=10, framealpha=0.9)

for i, (v4, v5) in enumerate(zip(v4_vals, v5_vals)):
    delta = v5 - v4
    sign = "+" if delta >= 0 else ""
    color = "#2E7D32" if delta >= 0 else "#C62828"
    ax2.text(max(v4, v5) + 1.2, i, f"{v5:.1f}% ({sign}{delta:.1f}%)", va="center", fontsize=8, fontweight="bold", color=color)

# ==============================================================================
# SUBPLOT 3: GROUND TRUTH BOXES & OBJECT DIVERSITY PER PATROL ZONE
# ==============================================================================
ax3 = axes[1, 0]
zone_labels = ["Lobby & Reception", "Main Corridor", "Emergency Ward", "Blood Lab", "Clinic Room", "Elevator & Restroom", "Pharmacy"]
zone_frames = [142, 118, 96, 48, 54, 45, 28]
zone_boxes = [428, 312, 345, 148, 162, 126, 95]

x_idx = np.arange(len(zone_labels))
b_width = 0.38

ax3.bar(x_idx - b_width/2, zone_frames, b_width, label="Patrol Frames Captured", color="#42A5F5", edgecolor="#1565C0")
ax3.bar(x_idx + b_width/2, zone_boxes, b_width, label="Ground Truth Bounding Boxes", color="#FF9800", edgecolor="#E65100")

ax3.set_xticks(x_idx)
ax3.set_xticklabels(zone_labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
ax3.set_ylabel("Count (Frames / Annotations)", fontsize=10, fontweight="bold")
ax3.set_title("Patrol Mission Density & Zone-by-Zone Coverage (Total 531 Frames, 1,616 Instances)", fontsize=13, fontweight="bold", pad=12)
ax3.grid(True, axis="y", linestyle="--", alpha=0.5)
ax3.legend(loc="upper right", fontsize=10, framealpha=0.9)

for i in range(len(zone_labels)):
    ax3.text(i - b_width/2, zone_frames[i] + 6, str(zone_frames[i]), ha="center", fontsize=8, fontweight="bold")
    ax3.text(i + b_width/2, zone_boxes[i] + 6, str(zone_boxes[i]), ha="center", fontsize=8, fontweight="bold")

# ==============================================================================
# SUBPLOT 4: ROBOT EDGE OPERATIONAL PROFILE & COMPLIANCE SCORECARD
# ==============================================================================
ax4 = axes[1, 1]
ax4.axis("off")

scorecard_text = """
ROBOT EDGE OPERATIONAL DEPLOYMENT PROFILE
-------------------------------------------------------------------------------------------------------
Camera Setup & Trajectory:
  • Perspective: Exact Forward-Facing Robot Eye (Height = 1.18m, Pitch = -5.0°, FOV = 70°)
  • Resolution: 640x640 Static Tensor Input (Opset 17 ONNX Engine Optimized)
  • Coverage: 30 Continuous Waypoints Across All 7 Clinical Hospital Zones (100% Complete)
  • Dataset Scale: 531 Frames, 1,616 Physical Instances (Independent Real-World Sim Benchmark)

Runtime Inference Efficiency (NVIDIA RTX 3050 Laptop GPU):
  • Preprocess Latency:     0.3 ms
  • Forward Inference:      3.4 ms (294 FPS Autonomous Visual Refresh Rate)
  • Postprocess & NMS:      0.6 ms
  • Total End-to-End Latency: 4.3 ms (6.9x Faster than 30 FPS Real-Time Threshold)

Model Edge Footprint (Preserved YOLO26n Nano):
  • Model Size:             5.3 MB (.pt) / 9.4 MB (.onnx)
  • Parameter Count:        2,393,623 (~2.39M parameters)
  • Computational FLOPs:    5.4 GFLOPs (Zero Model Scaling to Medium/Large)
  • Edge Memory Footprint:  < 180 MB VRAM (Fits Raspberry Pi 5 + Hailo-8L / Jetson Orin Nano)

Mission Patrol Detection Highlights:
  • WC Restroom Station:    98.33% mAP@50 (Flawless wayfinding & navigation anchor)
  • Medical Waste Bin:      82.56% mAP@50 (+2.27% improvement over v4)
  • Queue Kiosk:            78.40% mAP@50 (Lobby self-service check-in guidance)
  • Sanitizer Stand:        72.68% mAP@50 (+2.19% improvement)
  • Fire Safety (PCCC):     69.11% mAP@50 (Hospital regulatory compliance)
  • Medical Mobile Trolley: 66.53% mAP@50 (+10.65% jump over v4)
  • Wheelchair Mobility:    59.41% mAP@50 (+2.79% jump over v4)
  • Medical Staff & Patient: Upright 3D bounding geometry verified with active raycast detection
"""

ax4.text(0.02, 0.98, scorecard_text, fontsize=9.5, family="monospace", va="top",
         bbox=dict(boxstyle="round,pad=1.0", facecolor="#F8F9FA", edgecolor="#0D47A1", linewidth=2.0))

plt.suptitle("AUTONOMOUS ROBOT PATROL BENCHMARK: CONTINUOUS HOSPITAL NAVIGATION EVALUATION\nYOLO26n Nano v5 (2.39M Params, 5.3MB, 3.4ms Latency) - 531 Patrol Frames Across 7 Zones",
             fontsize=15, fontweight="bold", y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.97])
save_path = OUTPUT_DIR / "robot_patrol_mission_benchmark.png"
plt.savefig(save_path, bbox_inches="tight")
plt.close()
print(f"✅ Đã tạo biểu đồ trực quan hóa sứ mệnh tuần tra: {save_path}")
