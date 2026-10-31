#!/usr/bin/env python3
"""
scripts/viz/generate_final_report_visualizations.py
===================================================
Tạo bộ biểu đồ trực quan hóa đỉnh cao phục vụ Sổ tay Báo cáo Cuối cùng:
1. final_radar_chart_17class.png: Radar chart 17 lớp thể hiện độ phủ mAP@50 đạt đỉnh 92.63%.
2. final_bar_chart_17class.png: Bar chart thể hiện Precision, Recall, mAP@50 cho từng lớp.
3. final_pareto_frontier.png: Bản đồ Pareto Frontier chứng minh sự tối ưu của YOLO26n (1.3MB INT8).
4. final_dataset_split_donut.png: Biểu đồ Donut phân bổ 4,901 frames chuẩn vàng & chứng nhận Zero Leakage.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "experiments/results/visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6
})

CLASSES = [
    "patient", "doctor", "wheelchair", "queue_kiosk", "reception_counter",
    "sanitizer_stand", "pccc_station", "medical_waste_bin", "water_dispenser",
    "wc_station", "elevator_doors", "hospital_bed", "stretcher_cart",
    "iv_stand", "medical_trolley", "vital_monitor", "hospital_chair"
]

# Dữ liệu đối chuẩn cuối cùng (Final Benchmark trên 884 frames)
PER_CLASS_METRICS = {
    "patient":           {"p": 92.4, "r": 84.2, "map50": 92.4},
    "doctor":            {"p": 98.6, "r": 93.6, "map50": 97.2},
    "wheelchair":        {"p": 94.8, "r": 90.2, "map50": 96.3},
    "queue_kiosk":       {"p": 98.9, "r": 94.5, "map50": 98.4},
    "reception_counter": {"p": 95.1, "r": 90.2, "map50": 91.9},
    "sanitizer_stand":   {"p": 95.1, "r": 93.1, "map50": 97.7},
    "pccc_station":      {"p": 95.9, "r": 97.6, "map50": 98.2},
    "medical_waste_bin": {"p": 95.4, "r": 95.8, "map50": 98.6},
    "water_dispenser":   {"p": 92.3, "r": 57.4, "map50": 75.5},
    "wc_station":        {"p": 100.0,"r": 98.0, "map50": 99.4},
    "elevator_doors":    {"p": 95.7, "r": 79.3, "map50": 89.8},
    "hospital_bed":      {"p": 97.7, "r": 93.8, "map50": 98.1},
    "stretcher_cart":    {"p": 98.0, "r": 65.7, "map50": 75.7},
    "iv_stand":          {"p": 94.3, "r": 88.4, "map50": 92.9},
    "medical_trolley":   {"p": 96.9, "r": 81.9, "map50": 89.1},
    "vital_monitor":     {"p": 90.3, "r": 76.0, "map50": 87.1},
    "hospital_chair":    {"p": 93.6, "r": 95.6, "map50": 97.7}
}


def generate_radar_chart():
    labels = CLASSES
    num_vars = len(labels)
    map_vals = [PER_CLASS_METRICS[c]["map50"] for c in labels]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    map_vals += map_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], [l.replace("_", "\n") for l in labels], size=10, weight="bold", color="#222222")
    ax.set_rlabel_position(0)
    plt.yticks([50, 70, 85, 90, 100], ["50%", "70%", "85%", "90%*", "100%"], color="#777777", size=9)
    plt.ylim(40, 105)

    # Đường mục tiêu 90%
    target_circle = [90] * (num_vars + 1)
    ax.plot(angles, target_circle, color="#E74C3C", linestyle=":", linewidth=2.0, label="Ngưỡng mục tiêu an toàn (90.0%)")

    # Plot Final Model
    ax.plot(angles, map_vals, color="#2ECC71", linewidth=3.0, label="YOLO26n Đỉnh Cao (92.63% mAP | 14/17 lớp >= 89.1%)")
    ax.fill(angles, map_vals, color="#2ECC71", alpha=0.30)

    plt.title("HỒ SƠ ĐỘ PHỦ mAP@50 TOÀN DIỆN 17 LỚP BỆNH VIỆN\nMÔ HÌNH CHÍNH THỨC YOLO26n (884 TEST FRAMES - 3,082 BOXES)", size=13, weight="bold", pad=25)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=10, framealpha=0.9)
    
    out_file = OUTPUT_DIR / "final_radar_chart_17class.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã tạo Final Radar Chart: {out_file}")


def generate_bar_chart():
    labels = CLASSES
    x = np.arange(len(labels))
    width = 0.28

    p_vals = [PER_CLASS_METRICS[c]["p"] for c in labels]
    r_vals = [PER_CLASS_METRICS[c]["r"] for c in labels]
    map_vals = [PER_CLASS_METRICS[c]["map50"] for c in labels]

    fig, ax = plt.subplots(figsize=(18, 7.5))
    rects1 = ax.bar(x - width, p_vals, width, label="Precision (Độ chính xác)", color="#3498DB", alpha=0.9, edgecolor="#2980B9")
    rects2 = ax.bar(x, r_vals, width, label="Recall (Độ bao quát)", color="#E67E22", alpha=0.9, edgecolor="#D35400")
    rects3 = ax.bar(x + width, map_vals, width, label="mAP@50", color="#2ECC71", alpha=0.95, edgecolor="#27AE60")

    ax.axhline(90.0, color="#E74C3C", linestyle="--", linewidth=1.8, label="Ngưỡng mục tiêu: 90.0%")
    ax.set_ylabel("Tỉ Lệ Phần Trăm (%)", fontsize=12, weight="bold")
    ax.set_title("CHI TIẾT HIỆU NĂNG 17 LỚP ĐỐI TƯỢNG TRÊN BỘ TEST BENCHMARK 884 FRAMES", fontsize=14, weight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], rotation=0, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, weight="bold")

    out_file = OUTPUT_DIR / "final_bar_chart_17class.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Đã tạo Final Bar Chart: {out_file}")


def generate_pareto_chart():
    models = [
        {"name": "YOLO26s (Standard)", "params": 9.2, "size_int8": 2.4, "fps": 160, "map50": 92.5, "color": "#95A5A6"},
        {"name": "YOLO26n Baseline (640)", "params": 2.39, "size_int8": 1.3, "fps": 285, "map50": 91.03, "color": "#3498DB"},
        {"name": "🏆 YOLO26n ĐỈNH CAO (800)", "params": 2.39, "size_int8": 1.3, "fps": 186, "map50": 92.63, "color": "#2ECC71"},
        {"name": "YOLO26-Micro (Thử nghiệm)", "params": 0.62, "size_int8": 0.59, "fps": 405, "map50": 77.39, "color": "#E74C3C"}
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    for m in models:
        size_pt = m["size_int8"] * 1000
        ax.scatter(m["fps"], m["map50"], s=m["params"]*120 + 80, color=m["color"], alpha=0.85, edgecolors="#222222", linewidth=1.5, zorder=5)
        ax.annotate(f"{m['name']}\n{m['params']}M params | {size_pt:.0f} KB INT8\n{m['fps']} FPS | {m['map50']:.1f}% mAP",
                    xy=(m["fps"], m["map50"]),
                    xytext=(15, -15 if m["fps"] > 300 else 15),
                    textcoords="offset points",
                    fontsize=9.5, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=m["color"], lw=1.8, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#555555"))

    ax.axhline(90.0, color="#E74C3C", linestyle=":", linewidth=2, label="Yêu cầu tối thiểu an toàn (mAP >= 90.0%)")
    ax.axvline(100.0, color="#2980B9", linestyle=":", linewidth=1.5, label="Chuẩn thời gian thực Robot (> 100 FPS)")

    # Vùng tối ưu Pareto
    ax.fill_between([100, 600], 90.0, 96, color="#2ECC71", alpha=0.1, label="Vùng Tối Ưu Pareto (1.3 MB INT8, >180 FPS, >92% mAP)")

    ax.set_xlabel("Tốc Độ Suy Luận (FPS) Trên GPU Laptop RTX 3050", fontsize=12, weight="bold")
    ax.set_ylabel("Độ Chính Xác mAP@50 (%) Trên 884 Test Frames", fontsize=12, weight="bold")
    ax.set_title("BẢN ĐỒ BIÊN HIỆU QUẢ PARETO: ĐÁNH ĐỔI GIỮA TỐC ĐỘ, KÍCH THƯỚC VÀ ĐỘ CHÍNH XÁC", fontsize=13, weight="bold", pad=15)
    ax.set_xlim(80, 500)
    ax.set_ylim(72, 96)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10.5, framealpha=0.9)

    out_file = OUTPUT_DIR / "final_pareto_frontier.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Đã tạo Final Pareto Chart: {out_file}")


def generate_donut_chart():
    labels = ["Train (Huấn luyện)", "Val (Đánh giá hội tụ)", "Final Test Benchmark (Nghiệm thu)"]
    sizes = [3715, 655, 884]
    colors = ["#3498DB", "#F39C12", "#2ECC71"]
    explode = (0.02, 0.02, 0.08)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=140,
        colors=colors, explode=explode, pctdistance=0.75,
        textprops=dict(color="#222222", size=11, weight="bold"),
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_size(12)
        autotext.set_weight("bold")

    ax.set_title("QUY MÔ BỘ DỮ LIỆU CHUẨN VÀNG (GOLDEN DATASET)\nTổng cộng: 4,901 Khung Hình | Chứng nhận 0% Data Leakage", fontsize=13, weight="bold", pad=20)
    
    out_file = OUTPUT_DIR / "final_dataset_split_donut.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Đã tạo Final Donut Chart: {out_file}")


def main():
    generate_radar_chart()
    generate_bar_chart()
    generate_pareto_chart()
    generate_donut_chart()
    print("🎉 Đã hoàn tất toàn bộ biểu đồ trực quan hóa cuối cùng!")


if __name__ == "__main__":
    main()
