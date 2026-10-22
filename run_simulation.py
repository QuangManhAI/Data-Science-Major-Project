"""Unified simulation launcher for Robo-helper-in-Hospital.

Launches all backend Python servers (YOLO, STT/NLU, optional TTS) and Godot
in a single command, waits for WebSocket servers to be ready, and cleanly shuts down
all child processes when Godot exits or Ctrl+C is pressed.
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = REPO_ROOT / "sim_vision" / "python"
GODOT_DIR = REPO_ROOT / "sim_vision" / "godot"

CHILD_PROCESSES: List[subprocess.Popen] = []


def log(msg: str) -> None:
    """Print immediately with flush."""
    print(msg, flush=True)


def cleanup() -> None:
    """Terminate all spawned server subprocesses."""
    if not CHILD_PROCESSES:
        return
    log("\n[Launcher] Shutting down background servers...")
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
    log("[Launcher] All servers stopped.")


def handle_signal(sig, frame) -> None:
    cleanup()
    sys.exit(0)


def find_python() -> Path:
    """Find the preferred Python executable with project dependencies."""
    candidates = [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "bin" / "python",
        PYTHON_DIR / ".venv" / "Scripts" / "python.exe",
        PYTHON_DIR / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return Path(sys.executable)


def find_godot(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Find the Godot binary across Windows, macOS, and Linux."""
    if explicit_path and Path(explicit_path).is_file():
        return Path(explicit_path)

    env_path = os.environ.get("GODOT_BIN")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    import shutil

    # 1. System PATH lookup (works on Linux, macOS with brew, Windows)
    for name in ["godot", "godot4", "godot-bin", "godot.exe", "godot4.exe"]:
        found = shutil.which(name)
        if found:
            return Path(found)

    # 2. macOS common application bundles and paths
    if sys.platform == "darwin":
        mac_candidates = [
            Path("/Applications/Godot.app/Contents/MacOS/Godot"),
            Path("/Applications/Godot_4.app/Contents/MacOS/Godot"),
            Path.home() / "Applications" / "Godot.app" / "Contents" / "MacOS" / "Godot",
            Path("/opt/homebrew/bin/godot"),
            Path("/usr/local/bin/godot"),
        ]
        for c in mac_candidates:
            if c.is_file():
                return c

    # 3. Windows Downloads folder search
    if sys.platform == "win32":
        downloads_dir = Path.home() / "Downloads"
        download_candidates = list(downloads_dir.glob("**/Godot*.exe"))
        # Prefer console executable if available (outputs logs to terminal)
        for c in sorted(download_candidates, key=lambda x: "console" not in x.name.lower()):
            if c.is_file():
                return c

    # 4. Linux common user paths & Flatpak
    if sys.platform.startswith("linux"):
        linux_candidates = [
            Path.home() / ".local" / "bin" / "godot",
            Path.home() / ".local" / "bin" / "godot4",
            Path.home() / "bin" / "godot",
            Path("/usr/bin/godot"),
            Path("/usr/local/bin/godot"),
            Path("/var/lib/flatpak/exports/bin/org.godotengine.Godot"),
            Path.home() / ".local" / "share" / "flatpak" / "exports" / "bin" / "org.godotengine.Godot",
        ]
        for c in linux_candidates:
            if c.is_file():
                return c

    return None


def wait_for_ws(port: int, host: str = "127.0.0.1", timeout: float = 60.0, desc: str = "") -> bool:
    """Wait until WebSocket server is ready and responds to handshake."""
    start = time.time()
    url = f"ws://{host}:{port}"
    log(f"[Launcher] Waiting for {desc} on {url}...")
    while time.time() - start < timeout:
        try:
            from websockets.sync.client import connect
            with connect(url, open_timeout=1.0, close_timeout=1.0) as ws:
                log(f"[Launcher] -> {desc} is ready!")
                return True
        except Exception:
            time.sleep(0.5)
    log(f"[Launcher] -> {desc} timed out after {timeout}s!")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Simulation Launcher")
    parser.add_argument(
        "--nlu",
        choices=["rule", "phobert"],
        default="rule",
        help="NLU engine to use ('rule' or 'phobert', default: rule)",
    )
    parser.add_argument(
        "--with-tts",
        action="store_true",
        default=None,
        help="Force launch TTS server (port 8767). Default: auto-detect if VieNeu model is present",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS server even if models are present (runs in subtitle fallback mode)",
    )
    parser.add_argument(
        "--godot",
        type=str,
        default=None,
        help="Path to Godot executable",
    )
    parser.add_argument(
        "--no-godot",
        action="store_true",
        help="Only launch backend servers without starting Godot",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional Godot scene to launch (e.g. res://tools/dialogue_batch.tscn)",
    )
    args = parser.parse_args()

    # Register exit hooks
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    py_exe = find_python()
    log(f"[Launcher] Python executable: {py_exe}")

    sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    # 1. Start YOLO server (Port 8765)
    log("[Launcher] Starting YOLO server on port 8765...")
    yolo_proc = subprocess.Popen(
        [str(py_exe), "yolo_server.py"],
        cwd=PYTHON_DIR,
        env=sub_env,
    )
    CHILD_PROCESSES.append(yolo_proc)

    # 2. Start STT + NLU server (Port 8766)
    log(f"[Launcher] Starting STT + NLU server on port 8766 (NLU mode: {args.nlu})...")
    stt_proc = subprocess.Popen(
        [str(py_exe), "-m", "voice.stt_server", "--nlu", args.nlu],
        cwd=PYTHON_DIR,
        env=sub_env,
    )
    CHILD_PROCESSES.append(stt_proc)

    # 3. TTS server (Port 8767) — auto-detected or explicitly enabled
    tts_onnx = REPO_ROOT / "models" / "Vie-neu-v3-turbo" / "onnx_update" / "vieneu_prefill.onnx"
    should_run_tts = (args.with_tts or (tts_onnx.is_file() and not args.no_tts)) and not args.no_tts

    if should_run_tts:
        log("[Launcher] Starting TTS server on port 8767 (VieNeu-TTS v3 Turbo)...")
        tts_proc = subprocess.Popen(
            [str(py_exe), "-m", "tts.tts_server"],
            cwd=PYTHON_DIR,
            env=sub_env,
        )
        CHILD_PROCESSES.append(tts_proc)
    else:
        log("[Launcher] TTS server disabled/skipped. Simulation will run in subtitle fallback mode.")

    # Wait for WebSocket servers to be healthy
    ok_yolo = wait_for_ws(8765, desc="YOLO Server")
    ok_stt = wait_for_ws(8766, desc="STT Server")
    if should_run_tts:
        wait_for_ws(8767, desc="TTS Server")

    if not ok_yolo or not ok_stt:
        log("[Launcher] ERROR: One or more backend servers failed to start.")
        cleanup()
        sys.exit(1)

    log("[Launcher] All required backend servers are online.")

    if args.no_godot:
        log("[Launcher] Running in server-only mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    # Find and launch Godot
    godot_exe = find_godot(args.godot)
    if not godot_exe:
        log("[Launcher] ERROR: Could not find Godot executable.")
        log("          Please specify it using --godot <path_to_godot_exe>")
        log("          or set the GODOT_BIN environment variable.")
        cleanup()
        sys.exit(1)

    log(f"[Launcher] Launching Godot: {godot_exe}")
    godot_cmd = [str(godot_exe), "--path", str(GODOT_DIR)]
    if args.scene:
        godot_cmd.append(args.scene)

    godot_proc = subprocess.Popen(godot_cmd)
    CHILD_PROCESSES.append(godot_proc)

    try:
        godot_proc.wait()
        log(f"[Launcher] Godot exited with code {godot_proc.returncode}.")
    except KeyboardInterrupt:
        log("\n[Launcher] Interrupted by user.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
