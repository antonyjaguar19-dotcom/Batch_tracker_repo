"""Steps 2 and 3: the two SynthEyes behaviours the hybrid depends on.

Both failed (or were untestable) on the Demo build. They are re-run here against whatever
licence this machine has, and each prints a PASS/FAIL verdict so the bat can stop early
instead of producing numbers that describe a broken mechanism.

    probes.py midshot   --plate <mp4 or frames dir>
    probes.py reacquire --plate <mp4 or frames dir>

Exit codes: 0 pass, 1 fail, 3 could not connect / no scene.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plate_io import Plate, to_uv, pick_feature  # noqa: E402
from sylab import connect, run_szl, OUT_DIR, DIAG  # noqa: E402

DIAG_FWD = DIAG.replace("\\", "/")

# Tracker fields shared by both probes. Same values the hybrid uses, so a probe result
# transfers to the real run.
TK_SETUP = [
    "tk.kind = 0",
    "tk.size = 0.020",
    "tk.asp = 1.0",
    "tk.srchu = 0.015",
    "tk.srchv = 0.020",
    "tk.smooth = 20",
    "tk.autokey = 20",
    "tk.isSel = 2",
]


def _load(eng, plate: Plate):
    if eng.hlev.NewSceneAndShot(plate.load_path) is None:
        raise RuntimeError("NewSceneAndShot returned None")
    time.sleep(6)
    for sh in eng.hlev.Shots():
        eng.hlev.Validate(sh)
    time.sleep(2)


# --------------------------------------------------------------------------- midshot

def probe_midshot(eng, plate: Plate) -> bool:
    """Can a tracker be CREATED at a mid-shot frame and become valid?

    On the Demo build it never did, which is why the hybrid forces seed staggering off.
    If this passes, staggered seeds come back and the hybrid can cover content that only
    appears part-way through a shot.
    """
    mid = min(80, max(1, plate.count // 2))
    f0 = plate.frame(0)
    fm = plate.frame(mid)
    if f0 is None or fm is None:
        raise RuntimeError("could not read the probe frames")
    (cx0, cy0) = pick_feature(f0)[0]
    (cxm, cym) = pick_feature(fm)[0]
    u0, v0 = to_uv(cx0, cy0, plate.w, plate.h)
    um, vm = to_uv(cxm, cym, plate.w, plate.h)

    lines = ["//SIZZLET LabMidshot",
             "ob = Scene.activeObj", "shot = ob.shot", "start = shot.start",
             f'openout("{DIAG_FWD}")']
    # CONTROL: creation at the shot start is known to work. If this fails too, the failure
    # is something else entirely and the mid-shot result means nothing.
    lines += ["frame = start", "tc = new ob.trk", 'tc.nm = "CTRL"']
    lines += [s.replace("tk.", "tc.") for s in TK_SETUP]
    lines += [f"tc.key = Point({u0:.6f},{v0:.6f})", "tc.isEnabled = 1", "x = tc.Run()",
              'printf("control %d\\n", tc.valid)']
    # TEST: creation at a mid-shot frame, using tkgrid.szl's documented pattern (key
    # isEnabled off at the start, on at the creation frame) plus an explicit mainFrame.
    lines += [f"frame = {mid}", "td = new ob.trk", 'td.nm = "MIDSHOT"']
    lines += [s.replace("tk.", "td.") for s in TK_SETUP]
    lines += [f"td.mainFrame = {mid}",
              f"td.key = Point({um:.6f},{vm:.6f})",
              "frame = start", "td.isEnabled = 0",
              f"frame = {mid}", "td.isEnabled = 1", "x = td.Run()",
              'printf("midshot %d\\n", td.valid)',
              f"frame = {mid + 1}", "x = td.Run()",
              'printf("midshot_next %d\\n", td.valid)',
              "closeout()"]
    out = run_szl(eng, "\n".join(lines) + "\n")
    vals = dict(_pairs(out))
    print(out.strip() or "(no diag)")

    if vals.get("control") != 1:
        print("VERDICT: INCONCLUSIVE - even the control tracker at the shot start is "
              "invalid, so this says nothing about mid-shot creation.")
        return False
    ok = vals.get("midshot") == 1 and vals.get("midshot_next") == 1
    print(f"VERDICT: mid-shot creation {'PASS' if ok else 'FAIL'} (frame {mid})"
          + ("" if ok else " - seed staggering must stay OFF; seeds all go on frame 0."))
    return ok


# ------------------------------------------------------------------------- reacquire

def probe_reacquire(eng, plate: Plate) -> bool:
    """Can a tracker that is ALREADY ALIVE be re-keyed mid-shot and carry on?

    This is what "replant the same seed when the feature comes back" needs, and it is a
    different question from mid-shot CREATION above: the tracker already exists and is
    valid, we are only giving it a new key further along. One tracker, a hole in the
    middle, which is what the 3DE export already supports.
    """
    a = min(20, max(2, plate.count // 8))          # track to here
    b = min(a + 20, max(a + 2, plate.count - 3))   # replant here
    c = min(b + 19, plate.count - 1)               # then carry on to here
    fb = plate.frame(b)
    f0 = plate.frame(0)
    if f0 is None or fb is None:
        raise RuntimeError("could not read the probe frames")
    (cx0, cy0) = pick_feature(f0)[0]
    (cxb, cyb) = pick_feature(fb)[0]
    u0, v0 = to_uv(cx0, cy0, plate.w, plate.h)
    ub, vb = to_uv(cxb, cyb, plate.w, plate.h)

    lines = ["//SIZZLET LabReacquire",
             "ob = Scene.activeObj", "shot = ob.shot", "start = shot.start",
             f'openout("{DIAG_FWD}")',
             "frame = start", "tk = new ob.trk", 'tk.nm = "REACQ"']
    lines += TK_SETUP
    lines += [f"tk.key = Point({u0:.6f},{v0:.6f})", "tk.isEnabled = 1", "x = tk.Run()",
              # phase 1: track normally
              f"for (f = start + 1; f <= {a}; f++)", "    frame = f", "    x = tk.Run()", "end",
              f"frame = {a}", 'printf("phase1_end %d\\n", tk.valid)',
              # phase 2: simply do not run it -- this is the gap
              f"frame = {b - 1}", 'printf("gap %d\\n", tk.valid)',
              # phase 3: replant the SAME tracker and resume
              f"frame = {b}", f"tk.key = Point({ub:.6f},{vb:.6f})", "x = tk.Run()",
              'printf("replant %d\\n", tk.valid)',
              f"for (f = {b} + 1; f <= {c}; f++)", "    frame = f", "    x = tk.Run()", "end",
              f"frame = {c}", 'printf("phase3_end %d\\n", tk.valid)',
              # full validity count, so a partial recovery is visible rather than binary
              "cnt = 0",
              "for (frame = start; frame <= shot.stop; frame++)",
              "    if (tk.valid)", "        cnt = cnt + 1", "    end", "end",
              'printf("valid_total %d\\n", cnt)',
              "closeout()"]
    out = run_szl(eng, "\n".join(lines) + "\n")
    vals = dict(_pairs(out))
    print(out.strip() or "(no diag)")
    print(f"(tracked to {a}, gap {a + 1}-{b - 1}, replanted at {b}, resumed to {c})")

    if vals.get("phase1_end") != 1:
        print("VERDICT: INCONCLUSIVE - the tracker was already dead before the gap.")
        return False
    ok = vals.get("replant") == 1 and vals.get("phase3_end") == 1
    print(f"VERDICT: re-acquisition {'PASS' if ok else 'FAIL'}"
          + ("" if ok else " - a dead track cannot be resumed; --reacquire will do nothing."))
    return ok


# ------------------------------------------------------------------------------ main

def _pairs(text: str):
    for ln in (text or "").splitlines():
        parts = ln.split()
        if len(parts) == 2:
            try:
                yield parts[0], int(parts[1])
            except ValueError:
                continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["midshot", "reacquire"])
    ap.add_argument("--plate", required=True)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    plate = Plate(args.plate, ifl_dir=OUT_DIR)
    print(f"plate {plate.name}: {plate.w}x{plate.h}, {plate.count} frames")
    if plate.count < 12:
        print("FAIL: need at least 12 frames to probe anything meaningful.")
        return 3

    try:
        eng = connect(quiet=True)
        _load(eng, plate)
    except Exception as e:
        print(f"FAIL: {e}")
        return 3

    try:
        ok = probe_midshot(eng, plate) if args.which == "midshot" else probe_reacquire(eng, plate)
    finally:
        plate.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
