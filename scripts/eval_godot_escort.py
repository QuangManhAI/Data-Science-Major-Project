"""Đánh giá end-to-end trong Godot: robot RL (PureVisionV2) + YOLO + STT/NLU/TTS ở chế độ --escort-only.

Chạy sim N phút (thời gian thực), tắt sạch, gom log rồi tính:
  dẫn đường   số chuyến bắt đầu / tới nơi, thời gian mỗi chuyến, đi thang máy, lần "kẹt" (45 s không tiến triển)
  an toàn     giây có khoảng trống LiDAR < 0.40 m (= ngưỡng va chạm lúc train), giây va vào tường, giây robot
              ra lệnh chạy mà không nhúc nhích
  bệnh nhân   khoảng cách tới robot khi dẫn, % thời gian > 2 m, số lần phải đưa về dấu chân robot
  lái         |Δω| giữa hai bước policy liên tiếp, tỉ lệ đảo chiều lái mạnh
  giọng nói   WER STT so với câu gốc, độ chính xác intent, dẫn đúng phòng được hỏi

Dùng:
  sim_vision/python/.venv/bin/python scripts/eval_godot_escort.py --minutes 25
  sim_vision/python/.venv/bin/python scripts/eval_godot_escort.py --analyze-only experiments/results/godot_eval_<stamp>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import signal
import statistics as st
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "sim_vision" / "python" / ".venv" / "bin" / "python"
COLLISION_M = 0.40
TETHER_FAR_M = 2.0

DEBUG_RE = re.compile(
    r"\[ESCORT_DEBUG\] t=\s*([\d.]+)s \| State=(\S+)\s*\| Phase=(\S+)\s*\| Robot=\(\s*([-\d.]+),\s*([-\d.]+)\) "
    r"cmd=\(v=([-\d.]+), w=\s*([-+\d.]+)\) vel=([\d.]+) \| Patient=.*tether=([\d.]+)m"
    r"(?:.*clear_min=([\d.inf]+)m bump=(\d+) floor=(\d+))?")
START_RE = re.compile(r"Post-dialogue task initiated: act='(\w+)', target='([^']*)', floor=(\d+).*final='([^']*)'")
ARRIVE_RE = re.compile(r"Arrived at escort goal '([^']*)'")
ELEV_RE = re.compile(r"Elevator transit triggered: Floor (\d+) -> Floor (\d+) \((?:FinalTarget|Solo)")
STEP_RE = re.compile(r"step=\s*(\d+)\s+phase=(\w+).*?w=([+-][\d.]+)")


# ----------------------------------------------------------------------------- run
def run_sim(minutes: float, out: Path) -> None:
    started = time.time()
    proc = subprocess.Popen([str(PY), "run_simulation.py", "--controller", "rl", "--nlu", "phobert", "--escort-only"],
                            cwd=ROOT, stdout=open(out / "launcher.log", "w"), stderr=subprocess.STDOUT)
    print(f"[eval] sim chạy {minutes:.0f} phút (pid {proc.pid}) — đừng đóng cửa sổ Godot", flush=True)
    try:
        deadline = started + minutes * 60
        while time.time() < deadline:
            if proc.poll() is not None:
                print("[eval] sim dừng sớm (cửa sổ bị đóng?) — phân tích phần đã chạy", flush=True)
                break
            time.sleep(5)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)  # launcher tắt server + Godot gọn gàng
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    for name in ("godot.log", "rl_policy_server.log", "stt.log", "yolo.log", "tts.log"):
        if (ROOT / "logs" / name).exists():
            shutil.copy(ROOT / "logs" / name, out / name)
    for f in (ROOT / "sim_vision" / "logs").glob("dialogue_*.csv"):
        if f.stat().st_mtime >= started:
            shutil.copy(f, out / f.name)


# ----------------------------------------------------------------------------- parse
def parse_godot(path: Path) -> dict:
    t = 0.0
    rows, escorts, elevators, stalls, snaps = [], [], [], 0, 0
    cur = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = DEBUG_RE.search(line)
        if m:
            t = float(m[1])
            rows.append(dict(t=t, state=m[2], phase=m[3], v_cmd=float(m[6]), w_cmd=float(m[7]), vel=float(m[8]),
                             tether=float(m[9]), clear=float(m[10]) if m[10] and m[10] != "inf" else None,
                             bump=int(m[11]) if m[11] else 0, floor=int(m[12]) if m[12] else None))
            continue
        if (m := START_RE.search(line)) and m[1] == "guide":
            if cur is not None:
                escorts.append(cur)
            cur = dict(start=t, target=m[4] or m[2], floor=int(m[3]), end=None, ok=False, stalls=0, elevator=0)
        elif (m := ARRIVE_RE.search(line)) and cur is not None:
            cur.update(end=t, ok=True)
            escorts.append(cur)
            cur = None
        elif m := ELEV_RE.search(line):
            elevators.append((int(m[1]), int(m[2]), "Solo" in line))
            if cur is not None and "Solo" not in line:
                cur["elevator"] += 1
        elif "timeout suppressed" in line:
            stalls += 1
            if cur is not None:
                cur["stalls"] += 1
        elif "Patient stuck" in line:
            snaps += 1
    if cur is not None:
        cur["in_progress"] = True
        escorts.append(cur)
    return dict(rows=rows, escorts=escorts, elevators=elevators, stalls=stalls, snaps=snaps, duration_s=t)


def jitter(path: Path) -> dict:
    steps = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if m := STEP_RE.search(line):
            steps.append((int(m[1]), m[2], float(m[3])))
    out = {}
    for ph in ("GUIDE_ESCORT", "APPROACH_PATIENT", "SEARCH_PATIENT"):
        pairs = [(a, b) for a, b in zip(steps, steps[1:]) if a[1] == b[1] == ph and b[0] == a[0] + 1]
        if pairs:
            d = [abs(b[2] - a[2]) for a, b in pairs]
            flips = sum(1 for a, b in pairs if a[2] * b[2] < 0 and abs(a[2] - b[2]) > 0.3)
            out[ph] = dict(n=len(d), mean_abs_dw=round(st.mean(d), 3), flip_rate=round(flips / len(d), 3))
    return out


def _words(s: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", s.lower()).split()


def _wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)], len(r)


def dialogue(files: list[Path]) -> dict:
    sys.path.insert(0, str(ROOT / "sim_vision" / "python"))
    from nlu import policy, slots  # noqa: E402
    from nlu.nlu import IntentResult  # noqa: E402
    with open(ROOT / "data" / "processed" / "manifest.csv", encoding="utf-8") as f:
        man = {r["clip_id"]: r for r in csv.DictReader(f)}
    turns = [r for f in files for r in csv.DictReader(open(f, encoding="utf-8"))]
    errs = words = n_int = ok_int = n_guide = ok_guide = n_err = 0
    by_gender: dict[str, list[int]] = {}
    wrong: list[str] = []
    for r in turns:
        ref = man.get(r["clip_id"])
        if r.get("error"):
            n_err += 1
        if ref is None or not r.get("hyp_text"):
            continue
        e, n = _wer(ref["text_ref"], r["hyp_text"])
        errs, words = errs + e, words + n
        g = by_gender.setdefault(ref["gender"], [0, 0])
        g[0], g[1] = g[0] + e, g[1] + n
        n_int += 1
        ok_int += r["intent_pred"] == ref["intent"]
        if ref.get("escort_ok") == "True":
            n_guide += 1
            want = policy.decide(IntentResult(ref["intent"], 1.0, slots.extract(ref["text_ref"]), "rule", []), 0, None)
            good = r["action"] == "guide" and r["target"] == want.target
            ok_guide += good
            if not good:
                wrong.append(f"{r['clip_id']}: '{ref['text_ref']}' ➔ nghe '{r['hyp_text']}' ➔ {r['intent_pred']}/"
                             f"{r['action']}/{r['target'] or '-'} (đúng: {want.target})")
    return dict(turns=len(turns), errors=n_err, wer=round(errs / max(words, 1), 3),
                wer_by_gender={k: round(v[0] / max(v[1], 1), 3) for k, v in by_gender.items()},
                intent_acc=round(ok_int / max(n_int, 1), 3), n_intent=n_int,
                guide_correct=round(ok_guide / max(n_guide, 1), 3), n_guide=n_guide, wrong_guides=wrong[:12])


def summarize(out: Path) -> dict:
    g = parse_godot(out / "godot.log")
    rows = g["rows"]
    esc_rows = [r for r in rows if r["phase"] == "GUIDE_ESCORT"]
    moving = [r for r in rows if r["state"] not in ("LISTEN", "THINK", "RESPOND", "ARRIVED")]
    done = [e for e in g["escorts"] if not e.get("in_progress")]
    ok = [e for e in done if e["ok"]]
    durs = [e["end"] - e["start"] for e in ok]
    teth = [r["tether"] for r in esc_rows]
    clear = [r["clear"] for r in rows if r["clear"] is not None]
    res = {
        "sim_minutes": round(g["duration_s"] / 60, 1),
        "escort": {
            "started": len(g["escorts"]), "finished_or_abandoned": len(done), "arrived": len(ok),
            "success_rate": round(len(ok) / max(len(done), 1), 3),
            "duration_s_median": round(st.median(durs), 1) if durs else None,
            "duration_s_max": round(max(durs), 1) if durs else None,
            "with_elevator": sum(1 for e in ok if e["elevator"]),
            "elevator_transits": len(g["elevators"]),
            "stall_events_45s": g["stalls"],
            "targets": [f"{e['target']}{'✓' if e['ok'] else ('…' if e.get('in_progress') else '✗')}" for e in g["escorts"]],
        },
        "safety": {
            "seconds_logged": len(rows),
            "close_calls_s_lt_0.40m": sum(1 for c in clear if c < COLLISION_M),
            "min_clearance_m": round(min(clear), 2) if clear else None,
            "wall_bump_s": sum(1 for r in rows if r["bump"] > 0),
            "blocked_s_cmd_but_still": sum(1 for r in moving if r["v_cmd"] > 0.2 and r["vel"] < 0.05),
        },
        "patient_follow": {
            "tether_mean_m": round(st.mean(teth), 2) if teth else None,
            "tether_p90_m": round(sorted(teth)[int(0.9 * (len(teth) - 1))], 2) if teth else None,
            "tether_max_m": round(max(teth), 2) if teth else None,
            "pct_time_over_2m": round(100 * sum(1 for x in teth if x > TETHER_FAR_M) / max(len(teth), 1), 1),
            "stuck_snaps": g["snaps"],
        },
        "steering": jitter(out / "rl_policy_server.log") if (out / "rl_policy_server.log").exists() else {},
        "voice": dialogue(sorted(out.glob("dialogue_*.csv"))),
    }
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minutes", type=float, default=25.0)
    p.add_argument("--analyze-only", type=Path, default=None, help="thư mục kết quả đã có log, chỉ phân tích lại")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.analyze_only:
        out = args.analyze_only
    else:
        out = ROOT / "experiments" / "results" / f"godot_eval_{datetime.now():%Y%m%d_%H%M%S}"
        out.mkdir(parents=True, exist_ok=True)
        run_sim(args.minutes, out)
    res = summarize(out)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[eval] ➔ {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
