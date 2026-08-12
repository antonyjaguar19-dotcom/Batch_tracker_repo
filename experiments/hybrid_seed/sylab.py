"""Fast Sizzle iteration harness: run a .szl against the ALREADY-RUNNING SynthEyes.

Reuses the live instance and whatever scene is loaded, so a snippet can be tried in
about a second instead of reloading a plate each time.

    runtime\\python311\\python.exe experiments\\hybrid_seed\\sylab.py snippet.szl [--load MP4]

Any file the snippet writes to out/lab.diag is printed afterwards.
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.syntheyes_engine import SynthEyesEngine  # noqa: E402

OUT_DIR = os.path.join(HERE, "out")
DIAG = os.path.join(OUT_DIR, "lab.diag")


def wait_until_ready(eng, timeout: float = 120.0, quiet: bool = True) -> bool:
    """Wait for a freshly launched SynthEyes to be able to RUN something.

    The SyPy3 socket comes up well before the app is usable: a cold start leaves a
    `SplashPopup` child window on the main window, and while it is up RunScriptFile does not
    come back -- which looks exactly like a hang, on any script, including a two-line one.
    Warm instances never show this, which is why it only bites the first run after a launch.

    Returns True when the splash is gone (or was never there).
    """
    import ctypes
    u = ctypes.windll.user32
    EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    said = False
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = []

        def _cb(hh, _l):
            c = ctypes.create_unicode_buffer(64)
            u.GetClassNameW(hh, c, 64)
            if c.value == "SplashPopup" and u.IsWindowVisible(hh):
                found.append(hh)
            return True

        try:
            for h in eng._syntheyes_top_windows():
                u.EnumChildWindows(h, EnumProc(_cb), 0)
        except Exception:
            return True
        if not found:
            return True
        if not said and not quiet:
            print("   waiting for the SynthEyes splash to clear...", flush=True)
            said = True
        for hh in found:                 # nudge it; it also times out on its own
            u.PostMessageW(hh, 0x0010, 0, 0)          # WM_CLOSE
        time.sleep(1.0)
    if not quiet:
        print("   WARNING: the SynthEyes splash never cleared; scripts may hang.", flush=True)
    return False


def connect(quiet: bool = True):
    log = (lambda m: None) if quiet else (lambda m: print(f"SE: {m}", flush=True))
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
        raise SystemExit("SyPy3 not found")
    if not eng.connect_or_launch():
        raise SystemExit("could not connect to SynthEyes")
    wait_until_ready(eng, quiet=quiet)
    eng.set_writable_folder(OUT_DIR)
    return eng


def run_szl(eng, text: str, watchdog: int = 900) -> str:
    """Run a snippet; return the contents of out/lab.diag ('' if not written)."""
    try:
        os.remove(DIAG)
    except OSError:
        pass
    # Several SyPy3 calls (Validate, room nav, spinner writes) leave the 2026.2.4679 socket
    # desynced, after which RunScriptFile silently no-ops -- which reads as "the snippet did
    # nothing" and sends you hunting the wrong bug. Always start from a fresh socket.
    eng._resync_socket()
    t0 = time.time()
    eng._run_sizzle(text, watchdog_secs=watchdog)
    dt = time.time() - t0
    print(f"[sylab] {dt:.1f}s", flush=True)
    if os.path.isfile(DIAG):
        return open(DIAG, encoding="utf-8", errors="ignore").read()
    return ""


def main() -> int:
    args = sys.argv[1:]
    load = None
    validate = "--validate" in args
    if validate:
        args.remove("--validate")
    if "--load" in args:
        i = args.index("--load")
        load = args[i + 1]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    eng = connect(quiet=False)
    if load:
        print(f"-> loading {load}", flush=True)
        if eng.hlev.NewSceneAndShot(os.path.normpath(load)) is None:
            print("FAIL: NewSceneAndShot returned None")
            return 4
        time.sleep(5)
    if validate:
        shots = eng.hlev.Shots()
        print(f"-> Validate() on {len(shots)} shot(s)", flush=True)
        for sh in shots:
            eng.hlev.Validate(sh)
        time.sleep(2)
    text = open(args[0], encoding="utf-8").read()
    out = run_szl(eng, text)
    print("---- lab.diag ----")
    print(out if out else "(nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
