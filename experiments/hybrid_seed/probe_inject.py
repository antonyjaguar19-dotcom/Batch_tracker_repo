"""PROBE: can SynthEyes be handed external seed points and made to track them?

Smallest thing that answers the question. No bot code is modified; this only imports
`app.syntheyes_engine` for its connect / Sizzle-run plumbing.

Mechanism under test (both idioms are from SynthEyes' own shipped scripts):
  scripts/Trackers/tkgrid.szl   -- `tk = new ob.trk` / `tk.key = Point(u,v)` / `tk.Run()`
  scripts/Trackers/trackbyx.szl -- `Scene.RunTrackersFwd()` stepping one frame at a time

Run from the repo root:
    runtime\\python311\\python.exe experiments\\hybrid_seed\\probe_inject.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.syntheyes_engine import SynthEyesEngine  # noqa: E402

MP4 = r"D:\Jefrin\IN\SH004.mp4"
N_SEEDS = 10
N_FRAMES = 20          # probe only tracks a short span; full run comes later
OUT_DIR = os.path.join(HERE, "out")
OUT_TXT = "probe_inject.txt"


def seed_points(mp4: str, n: int):
    """goodFeaturesToTrack on frame 0 -- stands in for TAPNext seeds for the probe."""
    cap = cv2.VideoCapture(mp4)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame 0 of {mp4}")
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=n, qualityLevel=0.05,
                                  minDistance=120, blockSize=5)
    if pts is None:
        raise RuntimeError("no features found on frame 0")
    return pts.reshape(-1, 2).astype(np.float32), w, h


def to_uv(px: float, py: float, w: int, h: int):
    """Inverse of the engine's export mapping (syntheyes_engine.py:2062):
    px = 0.5*(u+1)*w  ->  u = 2*px/w - 1 ;  py = 0.5*(1-v)*h  ->  v = 1 - 2*py/h"""
    return (2.0 * px / w) - 1.0, 1.0 - (2.0 * py / h)


def build_inject_script(uv, n_frames: int) -> str:
    """Sizzle: create one area-match tracker per seed at the shot start, key it, then
    step forward calling RunTrackersFwd once per frame."""
    lines = [
        "//SIZZLET BTRInjectProbe",
        "ob = Scene.activeObj",
        "shot = ob.shot",
        "start = shot.start",
        "stop = shot.stop",
        "Scene.playRate = 1",
        "frame = start",
        "nmade = 0",
    ]
    for i, (u, v) in enumerate(uv):
        lines += [
            "frame = start",
            "tk = new ob.trk",
            f'tk.nm = "HYB{i:04d}"',
            "tk.kind = 0",          # area match
            "tk.size = 0.02",
            "tk.asp = 1.0",
            "tk.srchu = 0.015",
            "tk.srchv = 0.020",
            "tk.smooth = 20",
            "tk.autokey = 20",
            "tk.isSel = 2",
            f"tk.key = Point({u:.6f},{v:.6f})",
            "tk.isEnabled = 1",
            "tk.Run()",
            "nmade = nmade + 1",
        ]
    # Track forward, one frame per RunTrackersFwd, exactly as trackbyx.szl does it.
    lines += [
        "frame = start",
        f"last = start + {int(n_frames) - 1}",
        "if (last > stop)",
        "    last = stop",
        "end",
        "for (f = start; f < last; f++)",
        "    frame = f + 1",
        "    Scene.RunTrackersFwd()",
        "end",
    ]
    diag = os.path.join(OUT_DIR, "probe.diag").replace("\\", "/")
    lines += [
        f'openout("{diag}")',
        'printf("%d %d %d\\n", nmade, #ob.trk, last)',
        "closeout()",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    log = lambda m: print(f"SE: {m}", flush=True)

    pts, w, h = seed_points(MP4, N_SEEDS)
    print(f"plate {w}x{h}, {len(pts)} seed points on frame 0", flush=True)
    uv = [to_uv(float(x), float(y), w, h) for (x, y) in pts]

    settings = {
        "syntheyes_exe": os.environ.get(
            "BTR_SYNTHEYES_EXE",
            r"C:\Program Files\BorisFX\SynthEyes 2026\SynthEyes64.exe"),
        "port": int(os.environ.get("BTR_SE_PORT", 2222)),
        "pin": os.environ.get("BTR_SE_PIN", "listen"),
        "startup_wait": 3,
    }
    eng = SynthEyesEngine(settings, on_log=log)
    if not eng.setup_sypy():
        return 2
    if not eng.connect_or_launch():
        return 3

    eng.set_writable_folder(OUT_DIR)

    print("-> loading movie", flush=True)
    shot = eng.hlev.NewSceneAndShot(os.path.normpath(MP4))
    if shot is None:
        print("FAIL: NewSceneAndShot returned None")
        return 4
    time.sleep(5)
    try:
        eng.hlev.Redraw()
    except Exception:
        pass

    script = build_inject_script(uv, N_FRAMES)
    with open(os.path.join(OUT_DIR, "probe_inject.szl"), "w", encoding="utf-8") as f:
        f.write(script)

    print(f"-> injecting {len(uv)} trackers + tracking {N_FRAMES} frames", flush=True)
    t0 = time.time()
    eng._run_sizzle(script, watchdog_secs=600)
    dt = time.time() - t0
    print(f"   inject+track wall time: {dt:.1f}s", flush=True)

    diag = os.path.join(OUT_DIR, "probe.diag")
    if os.path.isfile(diag):
        print("   diag (nmade, #ob.trk, last_frame): " + open(diag).read().strip(), flush=True)
    else:
        print("   diag: NOT WRITTEN (script likely errored before the end)", flush=True)

    out_txt = os.path.join(OUT_DIR, OUT_TXT)
    print("-> exporting via Sizzle", flush=True)
    n = eng._sizzle_export_3de(out_txt)
    print(f"RESULT: export reported {n} trackers -> {out_txt}", flush=True)

    if n and n > 0 and os.path.isfile(out_txt):
        with open(out_txt) as f:
            head = [next(f, "").rstrip() for _ in range(6)]
        print("   first lines: " + " | ".join(head), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
