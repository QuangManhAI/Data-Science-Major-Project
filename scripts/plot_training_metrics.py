"""Training Curves Plotter for Visual RL In-The-Wild.

Reads metrics.jsonl from an experiment run directory and generates
high-contrast publication-grade curves for:
  - Episodic Team Reward (raw & rolling-20 average)
  - Patients Helped per Episode (raw, rolling-20 average, cumulative)
  - Collisions per Episode
  - Task Exit / Stagnation Statistics

Usage:
  python scripts/plot_training_metrics.py [--run-dir DIR] [--output PNG_PATH] [--show]
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def find_latest_wild_run(experiments_root: Path = Path("experiments/runs")) -> Path | None:
    if not experiments_root.exists():
        return None
    runs = sorted(
        [d for d in experiments_root.iterdir() if d.is_dir() and "PPOHospital-Wild" in d.name],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def load_metrics(metrics_file: Path) -> dict[str, np.ndarray]:
    episodes = []
    r_team = []
    roll20 = []
    helped = []
    collisions = []
    spotted = []
    steps = []
    exit_reasons = []

    with metrics_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                episodes.append(rec.get("episode", len(episodes) + 1))
                r_team.append(float(rec.get("r_team", 0.0)))
                roll20.append(float(rec.get("roll20_mean", rec.get("r_team", 0.0))))
                helped.append(int(rec.get("patients_helped", 0)))
                collisions.append(int(rec.get("collisions", 0)))
                spotted.append(bool(rec.get("spotted", False)))
                steps.append(int(rec.get("steps", 0)))
                exit_reasons.append(str(rec.get("exit_reason", "")))
            except Exception:
                continue

    return {
        "episode": np.asarray(episodes, dtype=int),
        "r_team": np.asarray(r_team, dtype=float),
        "roll20": np.asarray(roll20, dtype=float),
        "helped": np.asarray(helped, dtype=int),
        "collisions": np.asarray(collisions, dtype=int),
        "spotted": np.asarray(spotted, dtype=bool),
        "steps": np.asarray(steps, dtype=int),
        "exit_reasons": exit_reasons,
    }


def compute_rolling_mean(arr: np.ndarray, window: int = 20) -> np.ndarray:
    if len(arr) == 0:
        return np.array([])
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    rolling = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        w = min(i + 1, window)
        rolling[i] = (cumsum[i + 1] - cumsum[i + 1 - w]) / w
    return rolling


def plot_metrics(
    metrics: dict[str, np.ndarray],
    output_path: Path,
    title_suffix: str = "",
    show: bool = False,
) -> None:
    ep = metrics["episode"]
    if len(ep) == 0:
        print("[ERROR] No metrics data found to plot.")
        return

    r_team = metrics["r_team"]
    roll20_r = metrics["roll20"]
    helped = metrics["helped"]
    cum_helped = np.cumsum(helped)
    roll20_helped = compute_rolling_mean(helped, 20)
    col = metrics["collisions"]
    roll20_col = compute_rolling_mean(col, 20)

    # Styling setup: high-contrast dark theme
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=(12, 10), sharex=True, dpi=150, constrained_layout=True
    )
    fig.patch.set_facecolor("#0b0f19")

    accent_cyan = "#38bdf8"
    accent_emerald = "#34d399"
    accent_amber = "#fbbf24"
    accent_rose = "#f87171"
    subtle_gray = "#475569"

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#111827")
        ax.grid(True, linestyle="--", alpha=0.3, color="#374151")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#374151")

    # 1. Team Reward
    ax1.plot(ep, r_team, color=accent_cyan, alpha=0.25, linewidth=1.0, label="Raw Ep Reward")
    ax1.plot(ep, roll20_r, color=accent_cyan, linewidth=2.2, label="Rolling-20 Avg Reward")
    peak_idx = int(np.argmax(r_team))
    ax1.scatter(
        [ep[peak_idx]],
        [r_team[peak_idx]],
        color="#f43f5e",
        s=80,
        zorder=5,
        label=f"Peak: {r_team[peak_idx]:+.1f} (Ep {ep[peak_idx]})",
    )
    ax1.set_ylabel("Episodic Reward ($R_{team}$)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Visual RL Training Dynamics — Multi-Floor In-The-Wild {title_suffix}",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.legend(loc="upper left", facecolor="#1f2937", edgecolor="#374151", labelcolor="#e2e8f0")

    # 2. Patients Helped
    ax2.bar(ep, helped, color=accent_emerald, alpha=0.35, width=0.8, label="Patients Helped / Shift")
    ax2.plot(ep, roll20_helped, color=accent_emerald, linewidth=2.2, label="Rolling-20 Avg Helped")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(ep, cum_helped, color=accent_amber, linestyle="--", linewidth=1.8, label="Cumulative Total")
    ax2_twin.set_ylabel("Cumulative Patients", color=accent_amber, fontsize=10, fontweight="bold")
    ax2_twin.tick_params(colors=accent_amber, labelsize=9)
    ax2_twin.spines["right"].set_color(accent_amber)
    ax2.set_ylabel("Patients Helped", color="#f8fafc", fontsize=11, fontweight="bold")

    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2t, labels_2t = ax2_twin.get_legend_handles_labels()
    ax2.legend(
        lines_2 + lines_2t,
        labels_2 + labels_2t,
        loc="upper left",
        facecolor="#1f2937",
        edgecolor="#374151",
        labelcolor="#e2e8f0",
    )

    # 3. Collisions & Safe Navigation
    ax3.plot(ep, col, color=accent_rose, alpha=0.3, linewidth=1.0, label="Raw Collisions / Shift")
    ax3.plot(ep, roll20_col, color=accent_rose, linewidth=2.0, label="Rolling-20 Avg Collisions")
    ax3.set_ylabel("Obstacle Collisions", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Training Episode", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", facecolor="#1f2937", edgecolor="#374151", labelcolor="#e2e8f0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"\n[OK] Training curves plot saved to: {output_path.resolve()}")

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[WARN] Unable to display plot window: {e}")
    plt.close(fig)


def print_summary_table(metrics: dict[str, np.ndarray]) -> None:
    ep = metrics["episode"]
    if len(ep) == 0:
        return
    r_team = metrics["r_team"]
    helped = metrics["helped"]
    col = metrics["collisions"]
    spotted = metrics["spotted"]
    reasons = metrics["exit_reasons"]

    total_shifts = len(ep)
    total_helped = int(np.sum(helped))
    peak_r = float(np.max(r_team))
    mean_r = float(np.mean(r_team))
    last20_r = float(np.mean(r_team[-20:]))
    mean_helped = float(np.mean(helped))
    last20_helped = float(np.mean(helped[-20:]))
    mean_col = float(np.mean(col))
    spot_rate = float(np.mean(spotted)) * 100.0
    stagnation_count = sum(1 for r in reasons if "STAGNATION" in r)

    print("\n" + "=" * 70)
    print(" 📊 TRAINING PERFORMANCE TELEMETRY SUMMARY")
    print("=" * 70)
    print(f" Episode Span          : Ep {ep[0]} -> Ep {ep[-1]} ({total_shifts} shifts)")
    print(f" Peak Episode Reward   : {peak_r:+.1f} (Ep {ep[np.argmax(r_team)]})")
    print(f" Mean Episode Reward   : {mean_r:+.1f} (Last 20 Avg: {last20_r:+.1f})")
    print(f" Total Patients Helped : {total_helped} patients")
    print(f" Patients / Shift      : {mean_helped:.2f} ± {np.std(helped):.2f} (Last 20 Avg: {last20_helped:.2f})")
    print(f" Mean Collisions/Shift : {mean_col:.2f} collisions")
    print(f" Visual Spotting Rate  : {spot_rate:.1f}%")
    print(f" Stagnation Timeouts   : {stagnation_count} / {total_shifts} ({stagnation_count/total_shifts*100:.1f}%)")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot visual RL training metrics from metrics.jsonl.")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to experiment run directory.")
    parser.add_argument("--output", type=str, default=None, help="Output image file path (.png).")
    parser.add_argument("--show", action="store_true", help="Display interactive plot window.")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_wild_run()
        if run_dir is None:
            print("[ERROR] No visual RL run directory found in experiments/runs/")
            sys.exit(1)

    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"[ERROR] metrics.jsonl not found in {run_dir}")
        sys.exit(1)

    print(f"[INFO] Loading metrics from: {metrics_file}")
    data = load_metrics(metrics_file)
    print_summary_table(data)

    out_path = Path(args.output) if args.output else (run_dir / "training_curves.png")
    plot_metrics(data, out_path, title_suffix=f"({run_dir.name})", show=args.show)


if __name__ == "__main__":
    main()
