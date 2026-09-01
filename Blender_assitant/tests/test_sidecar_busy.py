"""A busy sidecar must not quietly turn CoTracker off.

From a real session, marking three stretches and tracking them:

    [runs] guess failed: a guess job is already running (CoTracker: tracking 1 point(s)...)
    [runs] Track: f1-f14: reached f14; f25-f31: CoTracker offered nothing;
                  f40-f65: CoTracker offered nothing
    WARNING short: f25-f31 got 2 of 7, f40-f65 got 2 of 26

The sidecar runs ONE job at a time and answers 409 BUSY to anything else. That was read as a
refusal, so two of the artist's three stretches were tracked by Blender alone -- with the
engine the whole mode exists for never once running -- and came back with 2 frames each.

Busy is not a failure. It is a queue of one, and the job in front is a CoTracker pass that
takes minutes. Only a sidecar that stays busy past `BUSY_WAIT_S` is a real problem.

This drives `_start` directly with a stand-in that refuses the way the sidecar does, because
reproducing the race against a real GPU job is slower and less certain than testing the rule
it broke.

    blender.exe --background -noaudio --python tests/test_sidecar_busy.py
"""

import os
import sys
import time

import bpy

EXT = "bl_ext.user_default.btr_assist"

FAILED = []


def check(name, got, want):
    ok = got == want
    print("[sb] %-56s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


class Fake:
    """Enough of the operator for `_start`, which is the rule under test."""

    def __init__(self, cls):
        object.__setattr__(self, "_cls", cls)

    def __getattr__(self, name):
        v = getattr(object.__getattribute__(self, "_cls"), name)
        return v.__get__(self, type(self)) if callable(v) and not isinstance(v, type) else v


def main():
    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    om = importlib.import_module(EXT + ".ops_mark")
    op = Fake(om.CLIP_OT_btr_track_runs)

    check("the wait is generous enough for a CoTracker pass in front",
          om.BUSY_WAIT_S >= 300.0, True)

    # Busy twice, then free -- what the artist hit, and what it should do about it.
    state = {"n": 0}

    def busy_then_free():
        state["n"] += 1
        if state["n"] <= 2:
            raise RuntimeError("a guess job is already running (CoTracker: tracking 1 "
                               "point(s) over 32 frames)")
        return {"id": "job-1"}

    t0 = time.time()
    job, err = op._start(busy_then_free)
    took = time.time() - t0
    check("a busy sidecar is waited for, not given up on", job, {"id": "job-1"})
    check("  and no error is reported for it", err, None)
    check("  it really retried rather than passing first time", state["n"], 3)
    print("[sb] waited %.1f s across 2 refusals" % took)

    # A real error must still come straight back -- waiting on one would hang the artist.
    def broken():
        raise RuntimeError("plate not found: D:/nope.mp4")

    job, err = op._start(broken)
    check("a real error is not mistaken for busy", job, None)
    check("  and it is reported as itself", "plate not found" in (err or ""), True)

    # Busy forever is a real problem and has to end as one, not hang.
    old = om.BUSY_WAIT_S
    om.BUSY_WAIT_S = 1.0
    try:
        def always_busy():
            raise RuntimeError("a fill job is already running (decoding)")

        t0 = time.time()
        job, err = op._start(always_busy)
        took = time.time() - t0
    finally:
        om.BUSY_WAIT_S = old
    check("a sidecar that never frees is reported, not waited on forever", job, None)
    check("  and it gives up near the limit", took < 10.0, True)
    check("  saying what it was waiting for", "already running" in (err or ""), True)

    print("")
    if FAILED:
        print("SIDECAR BUSY: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("[sb] SIDECAR BUSY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
