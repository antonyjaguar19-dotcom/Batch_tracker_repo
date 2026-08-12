"""Shared plumbing for the Blender hybrid: plate access, coordinates, Blender launch.

Nothing here imports bpy -- this is the *outside* half. `bl_track.py` is the inside half
and imports nothing from this repo, because it runs under Blender's own interpreter.

The plate reader is `experiments/hybrid_seed/plate_io.py` rather than a second copy: it
already handles movies, EXR/DPX sequences and the linear-float gamma trap, and it was
verified on this box during the SynthEyes hybrid.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HYBRID = os.path.join(ROOT, "experiments", "hybrid_seed")
for _p in (ROOT, HERE, HYBRID):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plate_io import Plate  # noqa: E402,F401  (sets OPENCV_IO_ENABLE_OPENEXR before cv2)

OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Where Blender is. Overridable, because this is a portable unzip, not an installer build.
DEFAULT_BLENDER = os.environ.get(
    "BTR_BLENDER_EXE",
    r"C:\Users\jefrin\Downloads\blender-5.2.0-windows-x64\blender-5.2.0-windows-x64\blender.exe")


# ------------------------------------------------------------------ coordinates
# The bot exports 3DE ASCII with flip_y_for_3de on: y_3de = (H-1) - y_topdown, i.e. the
# origin is bottom-left and y counts up, in PIXELS, with a pixel centre on the integer.
# Blender's movie-clip space is also bottom-left-origin but NORMALISED, with the centre of
# the bottom row at 0.5/H. So the two differ by a half-pixel as well as a scale, and
# dropping that half pixel is a systematic 0.5px bias -- exactly the size of the effect
# this experiment is trying to measure. Both directions are exact inverses; the round-trip
# is asserted in check_roundtrip.py rather than assumed.

def px_to_uv(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    """3DE pixel coords -> Blender normalised clip coords."""
    return (x + 0.5) / float(w), (y + 0.5) / float(h)


def uv_to_px(u: float, v: float, w: int, h: int) -> tuple[float, float]:
    """Blender normalised clip coords -> 3DE pixel coords."""
    return u * float(w) - 0.5, v * float(h) - 0.5


# ------------------------------------------------------------------ blender call

def run_blender(script: str, args: list[str], blender: str = "",
                timeout: float = 3600.0, quiet: bool = False) -> tuple[int, str]:
    """Run `script` under Blender in background mode and return (rc, combined output).

    `--factory-startup` so a user's saved preferences (add-ons, units, a different default
    scene) cannot change the result between machines, and `-noaudio` because the portable
    build otherwise probes audio devices on a box that may have none.

    Background mode is deliberate and was measured, not assumed: `bpy.ops.clip.*` are
    normally called out as needing a real CLIP_EDITOR, and the usual advice is to run
    Blender windowed. In 5.2 `--background` still builds one ghost window with a screen,
    so an area can be retyped to CLIP_EDITOR and `track_markers` returns FINISHED. That
    keeps this headless -- unlike the SynthEyes path, nothing here takes the mouse.
    """
    exe = blender or DEFAULT_BLENDER
    if not os.path.isfile(exe):
        raise RuntimeError(f"blender.exe not found: {exe}  (set BTR_BLENDER_EXE)")
    cmd = [exe, "--background", "--factory-startup", "-noaudio",
           "--python-exit-code", "3", "--python", script, "--"] + [str(a) for a in args]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    lines = out.splitlines()
    if not quiet:
        # Blender is chatty on startup, so normal runs print only what the worker prefixed.
        # A FAILED run prints the whole tail instead: filtering a traceback down to the
        # lines that match a prefix drops the exception message itself, which is the one
        # line worth having.
        if p.returncode != 0:
            print("\n".join(lines[-40:]))
        else:
            for line in lines:
                if line.startswith("BL "):
                    print(line)
    print(f"  blender rc={p.returncode} in {time.time() - t0:.1f}s")
    return p.returncode, out


# ------------------------------------------------------------------ 3DE ascii io

def read_3de(path: str) -> dict:
    """3DE 2D-track ASCII -> {name: {frame: (x, y)}}, via the bot's own parser."""
    from app.compare_tracks import load_tracks
    return load_tracks(path)


def write_3de(path: str, tracks: dict) -> int:
    """{name: {frame: (x, y)}} -> 3DE ASCII, via the bot's own writer.

    Importing the writer instead of reimplementing it means a format change in the bot
    cannot silently make this experiment's exports unreadable by the same tools that read
    the bot's.
    """
    from app.export_3de import write_tracks_txt
    conv = {}
    end = 1
    for name, pts in tracks.items():
        if len(pts) < 2:
            continue
        conv[name] = [(int(f), float(x), float(y)) for f, (x, y) in sorted(pts.items())]
        end = max(end, max(pts))
    write_tracks_txt(path, conv, end_frame=end)
    return len(conv)


def dump_json(path: str, obj) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
