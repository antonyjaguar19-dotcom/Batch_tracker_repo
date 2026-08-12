"""Step 1: is this SynthEyes licensed, or a Demo?

A Demo build tracks about 10 real frames per tracker and then holds a frozen coordinate.
The export still looks full length, so every number downstream describes the licence
instead of the tracker. That is the exact shape of plausible-but-wrong this repo's tooling
exists to prevent, so the experiment hard-stops here rather than producing numbers nobody
should trust.

Exit codes: 0 licensed, 2 Demo, 3 could not connect.

    runtime\\python311\\python.exe experiments\\hybrid_seed\\check_licence.py
"""
from __future__ import annotations

import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sylab import connect  # noqa: E402


def window_titles(eng) -> list[str]:
    user32 = ctypes.windll.user32
    out = []
    for h in eng._syntheyes_top_windows():
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(h, buf, 512)
        if buf.value:
            out.append(buf.value)
    return out


def main() -> int:
    try:
        eng = connect(quiet=True)
    except SystemExit as e:
        print(f"FAIL: {e}")
        return 3

    try:
        version = eng.hlev.Version()
    except Exception:
        version = "?"
    titles = window_titles(eng)
    print(f"SynthEyes version: {version}")
    for t in titles:
        print(f"window: {t}")

    if not titles:
        print("VERDICT: UNKNOWN - no SynthEyes window title could be read.")
        return 3
    if any("demo" in t.lower() for t in titles):
        print("VERDICT: DEMO - tracking is capped at ~10 frames per tracker and the rest of "
              "each track is a held coordinate. Stopping: nothing measured past this point "
              "would describe the tracker.")
        return 2
    print("VERDICT: LICENSED - full-length tracking expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
