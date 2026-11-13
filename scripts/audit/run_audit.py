"""Automated Audit Runner with Video Evidence Capture.

Orchestrates YOLO, STT/NLU backend servers and Godot 3D simulation with `--audit-mode`.
Runs stress audit episodes (lagging patient, crowded corridors, elevator transit),
collects captured .mp4 dashcam videos from AuditWatchdog, and compiles AUDIT_REPORT.md.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = REPO_ROOT / "sim_vision" / "python"
GODOT_DIR = REPO_ROOT / "sim_vision" / "godot"
EVIDENCE_DIR = REPO_ROOT / "sim_vision" / "reports" / "audit_evidence"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("run_audit")

CHILD_PROCESSES: List[subprocess.Popen] = []


def cleanup() -> None:
    """Cleanly terminate all spawned server and simulation processes."""
    if not CHILD_PROCESSES:
        return
    log.info("Shutting down audit child processes...")
    for p in CHILD_PROCESSES:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    time.sleep(0.5)
    for p in CHILD_PROCESSES:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
    CHILD_PROCESSES.clear()
    log.info("All audit processes terminated.")


atexit.register(cleanup)


def find_python() -> Path:
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return Path(sys.executable)


def find_godot() -> Path:
    candidates = [
        Path("/home/bush/.local/bin/godot"),
        REPO_ROOT / "godot",
    ]
    custom = os.environ.get("GODOT_BIN")
    if custom and Path(custom).is_file():
        return Path(custom)
    for c in candidates:
        if c.is_file():
            return c
    import shutil
    which = shutil.which("godot")
    if which:
        return Path(which)
    return Path("/home/bush/.local/bin/godot")


def wait_for_port(port: int, timeout_s: float = 10.0) -> bool:
    import socket
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.2)
    return False


def start_backend_servers(py_exe: Path) -> bool:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PYTHON_DIR)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. YOLO Server
    yolo_model = REPO_ROOT / "release_models" / "yolo26n_hospital_robot_v6_best" / "yolo26n_hospital_robot_best_map92.6.pt"
    if not yolo_model.is_file():
        yolo_model = REPO_ROOT / "models" / "yolov8n" / "yolov8n.pt"

    log.info("Starting YOLO Server on port 8765 (%s)...", yolo_model.name)
    yolo_log = open(EVIDENCE_DIR / "yolo_server.log", "w", encoding="utf-8")
    yolo_proc = subprocess.Popen(
        [str(py_exe), "yolo_server.py", "--model", str(yolo_model)],
        cwd=PYTHON_DIR,
        env=env,
        stdout=yolo_log,
        stderr=subprocess.STDOUT,
    )
    CHILD_PROCESSES.append(yolo_proc)

    # 2. STT Server with rule-based NLU
    log.info("Starting STT / Rule NLU Server on port 8766...")
    stt_log = open(EVIDENCE_DIR / "stt_server.log", "w", encoding="utf-8")
    stt_proc = subprocess.Popen(
        [str(py_exe), "-m", "voice.stt_server", "--nlu", "rule"],
        cwd=PYTHON_DIR,
        env=env,
        stdout=stt_log,
        stderr=subprocess.STDOUT,
    )
    CHILD_PROCESSES.append(stt_proc)

    # 3. RL Policy Server (PPO Hospital-EngageV3)
    rl_checkpoint = REPO_ROOT / "experiments" / "checkpoints" / "PPOHospital-EngageV3_best.pt"
    ok_rl = True
    if rl_checkpoint.is_file():
        log.info("Starting RL Policy Server on port 8768 (%s)...", rl_checkpoint.name)
        rl_log = open(EVIDENCE_DIR / "rl_server.log", "w", encoding="utf-8")
        rl_proc = subprocess.Popen(
            [str(py_exe), "rl_policy_server.py", "--checkpoint", str(rl_checkpoint)],
            cwd=PYTHON_DIR,
            env=env,
            stdout=rl_log,
            stderr=subprocess.STDOUT,
        )
        CHILD_PROCESSES.append(rl_proc)
        ok_rl = wait_for_port(8768, timeout_s=10.0)

    ok_yolo = wait_for_port(8765, timeout_s=15.0)
    ok_stt = wait_for_port(8766, timeout_s=30.0)

    if not ok_yolo or not ok_stt or not ok_rl:
        log.error("Failed to connect to backend servers (YOLO=%s, STT=%s, RL=%s)", ok_yolo, ok_stt, ok_rl)
        return False
    log.info("Backend servers online and healthy.")
    return True


def run_episode(
    godot_exe: Path,
    episode_idx: int,
    duration_s: float,
    extra_flags: list[str],
) -> None:
    """Runs a single audit episode in Godot with --audit-mode."""
    log.info(">>> Launching Audit Episode %d (Duration: %.1fs, Flags: %s)...", episode_idx, duration_s, extra_flags)
    godot_cmd = [
        str(godot_exe),
        "--path", str(GODOT_DIR),
        "--",
        "--audit-mode",
        f"--audit-duration={duration_s}",
    ] + extra_flags

    log_path = EVIDENCE_DIR / f"episode_{episode_idx}.log"
    with open(log_path, "w", encoding="utf-8") as f_log:
        proc = subprocess.Popen(godot_cmd, stdout=f_log, stderr=subprocess.STDOUT, text=True)
        CHILD_PROCESSES.append(proc)

        start_t = time.time()
        # Wait for Godot to compile shaders and quit gracefully via --audit-duration
        max_wait = duration_s + 45.0
        try:
            while time.time() - start_t < max_wait:
                if proc.poll() is not None:
                    log.info("Godot concluded episode %d with code %d", episode_idx, proc.returncode)
                    break
                time.sleep(1.0)
        finally:
            if proc.poll() is None:
                log.info("Episode timeout exceeded, stopping Godot...")
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if proc in CHILD_PROCESSES:
                CHILD_PROCESSES.remove(proc)


def generate_audit_summary_report(output_dir: Path) -> Path:
    """Scans output_dir for .mp4 and .json files, compiles AUDIT_REPORT.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_files = sorted(output_dir.glob("*.mp4"))
    
    report_path = output_dir / "AUDIT_REPORT.md"
    lines = [
        "# 📑 BÁO CÁO NGHIỆM THU KIỂM TOÁN VÀ VIDEO BẰNG CHỨNG (AUDIT EVIDENCE REPORT)",
        "",
        f"- **Thời gian tạo:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Tổng số video bằng chứng đã ghi nhận:** {len(mp4_files)} clip",
        f"- **Thư mục lưu trữ:** `{output_dir}`",
        "",
        "---",
        "",
        "## 🎥 Danh Sách Video Bằng Chứng Đã Ghi Nhận",
        "",
        "| STT | Video Bằng Chứng (.mp4) | Loại Lỗi / Hiện Tượng | Thời Lượng | Bệnh nhân / Chi tiết Telemetry | Tệp Dữ Liệu |",
        "| :---: | :--- | :--- | :---: | :--- | :---: |",
    ]

    if not mp4_files:
        lines.append("| - | *Không có lỗi nghiêm trọng nào kích hoạt trong đợt chạy này* | N/A | - | - | - |")
    else:
        for i, mp4 in enumerate(mp4_files, 1):
            json_file = mp4.with_suffix(".json")
            details = "N/A"
            err_type = "UNKNOWN"
            dur_str = "N/A"
            if json_file.is_file():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    err_type = meta.get("error_type", "EVENT")
                    dur = meta.get("duration_s", 0.0)
                    dur_str = f"{dur:.1f}s"
                    tether = meta.get("tether_distance_m", None)
                    stt_name = meta.get("robot_state", "")
                    details = f"State: `{stt_name}`"
                    if tether is not None:
                        details += f", Tether: `{tether:.2f}m`"
                except Exception as e:
                    details = f"Lỗi đọc JSON: {e}"

            lines.append(f"| {i} | [{mp4.name}](file://{mp4.resolve()}) | **`{err_type}`** | {dur_str} | {details} | [{json_file.name}](file://{json_file.resolve()}) |")

    lines.extend([
        "",
        "---",
        "",
        "## 📌 Hướng Dẫn Kiểm Tra Bằng Mắt Cho Người Đánh Giá",
        "1. Click trực tiếp vào các liên kết video `.mp4` ở bảng trên để mở bằng trình phát video hệ thống (VLC, Videos, Chrome).",
        "2. Kiểm tra góc nhìn toàn cảnh thứ ba: robot di chuyển, người bệnh đi sau lưng, và bảng thông số HUD góc trên.",
        "3. Đọc kèm file `.json` tương ứng để đối chiếu tọa độ $(x, y, z)$ và khoảng cách dây dắt tại thời điểm xảy ra sự cố.",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Generated master audit report at %s", report_path)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audit episodes and collect video evidence.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of audit episodes to run (default: 3)")
    parser.add_argument("--scenario", type=int, default=0, help="Run only specific scenario index (1, 2, or 3). Default 0 (run all).")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds per episode (default: 30.0)")
    parser.add_argument("--output-dir", default=str(EVIDENCE_DIR), help="Output directory for videos")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    py_exe = find_python()
    godot_exe = find_godot()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting Audit Harness (Python: %s, Godot: %s)", py_exe, godot_exe)
    if not start_backend_servers(py_exe):
        log.error("Failed to start backend servers. Exiting.")
        return 1

    # Scenarios for audit testing
    scenarios = [
        {
            "desc": "Kịch bản 1: Stress Test TETHER_BREAK - RL Policy V3 bỏ rơi người bệnh khi người bệnh tụt lại (--rl, --escort-only, --no-reid, --lag-prob=0.8, --escort-target=201)",
            "flags": ["--rl", "--escort-only", "--no-reid", "--lag-prob=0.8", "--escort-target=201"],
        },
        {
            "desc": "Kịch bản 2: Stress Test STAGNATION - Robot kẹt/đóng băng không tiến triển khi dẫn đường (--rl, --escort-only, --no-reid, --stagnation-test, --escort-target=301)",
            "flags": ["--rl", "--escort-only", "--no-reid", "--stagnation-test", "--escort-target=301"],
        },
        {
            "desc": "Kịch bản 3: Stress Test REID MISMATCH - Mất dấu danh tính / bế tắc góc nhìn sau trong hành lang đông người (--rl, --escort-only, --lag-prob=0.5, --escort-target=112)",
            "flags": ["--rl", "--escort-only", "--lag-prob=0.5", "--escort-target=112"],
        },
    ]

    try:
        run_indices = [args.scenario] if args.scenario > 0 else list(range(1, min(args.episodes + 1, len(scenarios) + 1)))
        for idx in run_indices:
            sc = scenarios[idx - 1]
            log.info("=== %s ===", sc["desc"])
            run_episode(
                godot_exe=godot_exe,
                episode_idx=idx,
                duration_s=args.duration,
                extra_flags=sc["flags"],
            )
            time.sleep(2.0)  # Wait for stitcher process to finish
    except KeyboardInterrupt:
        log.info("Audit interrupted by user.")
    finally:
        cleanup()

    report_path = generate_audit_summary_report(output_dir)
    print(f"\n=======================================================")
    print(f"AUDIT RUN FINISHED!")
    print(f"Master Report: file://{report_path.resolve()}")
    print(f"=======================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
