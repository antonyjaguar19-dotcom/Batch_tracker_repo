# -*- coding: utf-8 -*-
"""
SynthEyes tracking backend for the Batch Tracker.

Ported from the standalone production tool `autotrack_to_3de.py` (Tkinter GUI +
SynthEyesEngine). The Tkinter UI is dropped — this module exposes only the engine
so `app/app.py:worker_track` can drive SynthEyes over the SyPy3 socket API, in
place of the CoTracker3 path (`app/tracker_core.py`).

Differences vs the original standalone script:
  * No Tkinter / no folder scanner / no `Shot/mid/cmm/bot_MMpoints/` convention.
  * Input frames come from the Batch Tracker's per-shot image sequence (jpg/exr/png),
    not from `in/plates/vXXX/...`.
  * The matte is the SAM3 mask sequence (`<out>/<shot>/<mask_subdir>/*.png`) instead
    of a hand-authored `mid/mattes/` sequence. Gated by `use_sam3_matte`.
  * Export name + flat output dir follow the Batch Tracker convention
    (`<stem>__syntheyes.txt`) so existing QC (`compute_track_metrics`) finds it.

SyPy3 is a pure-python socket client that ships with SynthEyes; it imports fine
under the project's embeddable Python 3.11 once its path is on sys.path (the `._pth`
isolation blocks auto-paths, so `setup_sypy` inserts it at runtime).
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import glob
import site
import sysconfig
import ctypes
from ctypes import wintypes
import subprocess

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".exr", ".png", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")

# Win32 message constants for PostMessage button clicks (blip/peel on 2026.2.4679,
# where the SyPy3 ByID().ClickAndWait() dispatch silently no-ops).
_WM_LBUTTONDOWN = 0x0201
_WM_LBUTTONUP = 0x0202
_MK_LBUTTON = 0x0001


# SendInput payloads, for the Advanced-dialog spinner. That control ignores WM_SETTEXT and
# posted WM_CHAR outright, and keybd_event reached it but committed an EMPTY value, so real
# injected input is the only thing that sets it. See _set_advanced_max_tracks.
class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("pad", ctypes.c_byte * 32)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]


# ============================================================
#  SyPy3 discovery
# ============================================================

def _find_sypy3_paths(exe_hint=None):
    """Locate the SyPy3 library. Returns (sypy3_dir, parent_dir).

    SynthEyes ships SyPy3 inside its install folder (e.g. BorisFX SynthEyes 2026:
    `C:\\Program Files\\BorisFX\\SynthEyes 2026\\SyPy3`). Import is `from SyPy3.sylevel
    import SyLevel`, so the PARENT dir must go on sys.path. Best source is the folder of
    the SynthEyes .exe the user pointed at — version-proof across BorisFX/Andersson builds.
    """
    cands = []
    # 0. explicit override via env var (BTR_SYPY3_DIR points at the SyPy3 folder)
    env_dir = os.environ.get("BTR_SYPY3_DIR", "").strip()
    if env_dir:
        cands.append(env_dir)
    # 0a. user-placed copy off C: (kept alongside project assets)
    cands.append(r"D:\Jefrin\Assets\SyPy3")
    # 0b. next to the SynthEyes exe (most reliable when installed)
    if exe_hint:
        d = os.path.dirname(exe_hint)
        cands += [os.path.join(d, "SyPy3"), os.path.join(d, "SyPy")]
    # 1. currently running interpreter site-packages
    try:
        for sp in site.getsitepackages():
            cands.append(os.path.join(sp, "SyPy3"))
    except Exception:
        sp = sysconfig.get_path("purelib")
        if sp:
            cands.append(os.path.join(sp, "SyPy3"))
    # 2. common SynthEyes install dirs (BorisFX 2023+, Andersson older) + bundled pythons
    for pat in (r"C:\Program Files\BorisFX\SynthEyes*\SyPy3",
                r"C:\Program Files\Andersson Technologies LLC\SynthEyes*\SyPy3",
                r"C:\Program Files\*\SynthEyes*\SyPy3",
                r"C:\Python3*\Lib\site-packages\SyPy3"):
        cands += glob.glob(pat)

    for c in cands:
        # A valid SyPy3 dir contains sylevel.py (the SyLevel client lives there).
        if c and os.path.isfile(os.path.join(c, "sylevel.py")):
            return c, os.path.dirname(c)

    # fallback so import never crashes the host app; error message stays clear
    return r"C:\Python39\Lib\site-packages\SyPy3", r"C:\Python39\Lib\site-packages"


SYPY3_PATH, SITE_PACKAGES = _find_sypy3_paths()


# ============================================================
#  Frame scanning
# ============================================================

def scan_frames(folder, extensions=IMAGE_EXTENSIONS):
    """Scan a folder for image frames -> sorted list of (frame_num, filepath)."""
    files = []
    if not folder or not os.path.isdir(folder):
        return files
    for f in os.listdir(folder):
        if f.lower().endswith(extensions):
            filepath = os.path.normpath(os.path.join(folder, f))
            stem = os.path.splitext(f)[0]
            match = re.search(r"(\d+)$", stem)
            frame_num = int(match.group(1)) if match else 0
            files.append((frame_num, filepath))
    files.sort(key=lambda x: x[0])
    return files


def find_shot_frames(shot_dir):
    """Locate the image sequence for a shot.

    Looks for frames directly in `shot_dir`, else one level deep (first subfolder
    that contains frames). Returns a dict like the original scanner's version_data,
    or None.
    """
    frames = scan_frames(shot_dir)
    if not frames:
        for sub in sorted(os.listdir(shot_dir)) if os.path.isdir(shot_dir) else []:
            sub_dir = os.path.join(shot_dir, sub)
            if os.path.isdir(sub_dir):
                frames = scan_frames(sub_dir)
                if frames:
                    break
    if not frames:
        return None
    return {
        "first_frame": frames[0][1],
        "all_frames":  [fp for (_n, fp) in frames],
        "frame_count": len(frames),
        "start_frame": frames[0][0],
        "end_frame":   frames[-1][0],
        "extension":   os.path.splitext(frames[0][1])[1],
    }


def read_image_size(path):
    """Return (w, h) of an image, or (None, None). Uses cv2 if available."""
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    return None, None


# ============================================================
#  Defaults
# ============================================================

SE_DEFAULTS = {
    "port":           2222,
    "pin":            "listen",
    "threshold":      0.02,
    "separation":     8,
    "startup_wait":   10,
    "sensor_width":   36.0,
    "sensor_height":  24.0,
    "focal_length":   35.0,
    "fps":            24.0,
    # Long-shot chunking: blipping thousands of 4K frames at once OOMs SynthEyes
    # ('Imminent Crash'). Above chunk_threshold frames, blip/peel per playback-range
    # window (bounded blip memory) and accumulate trackers; chunk_long_shots False disables.
    # Window SIZE is ADAPTIVE (not fixed): seeded from resolution (chunk_frames_per_mp),
    # self-calibrated from live SynthEyes RSS, shrunk toward a RAM ceiling + on the crash
    # dialog. chunk_size>0 forces a fixed window (manual override).
    "chunk_long_shots":    True,
    "chunk_threshold":     1000,
    "chunk_size":          0,       # 0 = adaptive; >0 = fixed window size (override)
    "chunk_overlap":       24,
    "chunk_frames_per_mp": 5000,    # seed: ~600 frames @ 4K (8.3 MP); HD bigger, 8K smaller
    "chunk_ram_frac":      0.5,     # SynthEyes RSS headroom = this fraction of available RAM
    "chunk_min_window":    120,     # never shrink a window below this
}


# ============================================================
#  SYNTHEYES ENGINE
# ============================================================

class SynthEyesEngine:
    """Drives a SynthEyes instance over SyPy3 for batch 2D tracking + export."""

    def __init__(self, settings, on_log=None):
        self.settings = dict(settings or {})
        self.on_log = on_log or print
        self.hlev = None
        self._stop_requested = False

    def log(self, msg):
        safe = str(msg).encode("ascii", "replace").decode("ascii")
        self.on_log(safe)

    def request_stop(self):
        self._stop_requested = True

    def _s(self, key):
        return self.settings.get(key, SE_DEFAULTS.get(key))

    # ------ SyPy3 setup ------

    def setup_sypy(self):
        global SYPY3_PATH, SITE_PACKAGES
        # Re-resolve using the SynthEyes .exe the user set (version-proof).
        p, sp = _find_sypy3_paths(self._s("syntheyes_exe"))
        SYPY3_PATH, SITE_PACKAGES = p, sp
        for pth in [SITE_PACKAGES, SYPY3_PATH]:
            if os.path.isdir(pth) and pth not in sys.path:
                sys.path.insert(0, pth)
        if not os.path.isfile(os.path.join(SYPY3_PATH, "sylevel.py")):
            self.log(f"ERROR: SyPy3 not found near the SynthEyes .exe (looked at {SYPY3_PATH}). "
                     f"Check the SynthEyes .exe path in Settings.")
            return False
        self.log(f"OK  SyPy3 found: {SYPY3_PATH}")
        return True

    # ------ Win32 input helpers (fallbacks for dialogs without SyPy3 API) ------

    def _hardware_key(self, vk_code):
        try:
            user32 = ctypes.windll.user32
            user32.keybd_event(vk_code, 0, 0, 0)
            time.sleep(0.05)
            user32.keybd_event(vk_code, 0, 0x0002, 0)  # key up
        except Exception as e:
            self.log(f"   WARNING: Hardware key failed: {e}")

    def _hardware_type_string(self, text):
        try:
            user32 = ctypes.windll.user32
            VK_SHIFT = 0x10
            KEYEVENTF_KEYUP = 0x0002
            for char in text:
                vk_packed = user32.VkKeyScanW(ord(char))
                vk_code = vk_packed & 0xFF
                shift_pressed = (vk_packed >> 8) & 1
                if shift_pressed:
                    user32.keybd_event(VK_SHIFT, 0, 0, 0)
                    time.sleep(0.01)
                user32.keybd_event(vk_code, 0, 0, 0)
                time.sleep(0.01)
                user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)
                time.sleep(0.01)
                if shift_pressed:
                    user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
                    time.sleep(0.01)
        except Exception as e:
            self.log(f"   WARNING: Hardware typing failed: {e}")

    # ------ Win32 panel-button clicks (bypass the broken SyPy3 dispatch) ------
    # On SynthEyes 2026.2.4679 the SyPy3 return-value protocol is broken: blip/peel via
    # hlev.Main().ByID(ActionID(...)).ClickAndWait() returns instantly WITHOUT running (log
    # shows Blip+Peel "done" in the same second -> ~1 tracker). SynthEyes panel buttons are
    # real Win32 child windows of class "Butt"; PostMessage(WM_LBUTTONDOWN/UP) fires them for
    # real, even when the window is backgrounded (no foreground focus needed).

    def _syntheyes_top_windows(self):
        """All visible top-level windows owned by the SynthEyes process (main window +
        docked control panels are separate HWNDs; the blip/peel buttons live on a panel,
        NOT the main toolbar, so we must search every one)."""
        try:
            import psutil
        except ImportError:
            return []
        pids = {p.info["pid"] for p in psutil.process_iter(["name", "pid"])
                if "syntheyes" in (p.info["name"] or "").lower()}
        if not pids:
            return []
        user32 = ctypes.windll.user32
        found = []
        EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _cb(hwnd, _lp):
            if not user32.IsWindowVisible(hwnd):
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in pids:
                found.append(hwnd)
            return True

        user32.EnumWindows(EnumProc(_cb), 0)
        return found

    def _enum_butt_buttons(self, main_hwnd):
        """[(hwnd, text, (l,t,r,b)), ...] for every 'Butt' descendant of main_hwnd."""
        user32 = ctypes.windll.user32
        out = []
        EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _cb(hwnd, _lp):
            cls = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, cls, 64)
            if cls.value == "Butt":
                txt = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, txt, 256)
                r = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(r))
                out.append((hwnd, txt.value, (r.left, r.top, r.right, r.bottom)))
            return True

        user32.EnumChildWindows(main_hwnd, EnumProc(_cb), 0)
        return out

    def _win32_click_button(self, label, variants, settle=0.5):
        """Click a Features-room panel button by its text via PostMessage. Returns bool.

        Searches Butt children across ALL of SynthEyes' top-level windows (the panel
        buttons are not on the main window's toolbar).
        """
        tops = self._syntheyes_top_windows()
        if not tops:
            self.log(f"   WARNING: SynthEyes windows not found (Win32 {label})")
            return False
        buttons = []
        for w in tops:
            buttons.extend(self._enum_butt_buttons(w))
        norm = lambda s: "".join((s or "").lower().split())
        wants = {norm(v) for v in variants}
        match = next((b for b in buttons if norm(b[1]) in wants), None)
        if match is None:
            named = sorted({b[1] for b in buttons if b[1]})
            self.log(f"   WARNING: Win32 button '{label}' not found among {len(buttons)} "
                     f"Butt children across {len(tops)} windows. Labels: {named}")
            self._dump_all_children()
            return False
        hwnd, text, (l, t, r, b) = match
        cx, cy = (r - l) // 2, (b - t) // 2          # center in the button's client space
        lp = ((cy & 0xFFFF) << 16) | (cx & 0xFFFF)
        user32 = ctypes.windll.user32
        user32.PostMessageW(hwnd, _WM_LBUTTONDOWN, _MK_LBUTTON, lp)
        time.sleep(0.05)
        user32.PostMessageW(hwnd, _WM_LBUTTONUP, 0, lp)
        self.log(f"   OK  Win32 click '{text}'")
        time.sleep(settle)
        return True

    def _interruptible_sleep(self, seconds):
        """Sleep in 0.5s steps, honoring a stop request."""
        end = time.time() + max(0.0, float(seconds))
        while time.time() < end:
            if getattr(self, "_stop_requested", False):
                return False
            time.sleep(min(0.5, end - time.time()))
        return True

    # ------ active completion detection (replaces fixed blip/peel waits) ------
    # On 2026.2.4679 there's no working API to poll blip/peel completion (broken SyPy3).
    # A probe (2026-07) found two OS-level signals instead: (1) blip/peel spawn a transient
    # #32770 progress dialog ('Computing Blips' -> 'Linking'; present ~1s for 300 frames,
    # ~40s+ for 2114) that closes on completion, and (2) the process CPU pegs to thousands
    # of % during the op and drops to ~idle when done. We wait on BOTH (dialog-present OR
    # CPU-pegged = working; dialog-gone AND CPU-idle = done), abort on a crash/error dialog
    # (long-shot OOM shows 'Imminent Crash'), and cap with a max timeout so it can't hang.

    def _syntheyes_pid(self):
        try:
            import psutil
        except ImportError:
            return None
        for p in psutil.process_iter(["name", "pid"]):
            if "syntheyes" in (p.info["name"] or "").lower():
                return p.info["pid"]
        return None

    # Match SPECIFIC titles, not "any dialog": a benign startup dialog (crash-recovery
    # after a force-kill, license, tip) must count as NEITHER progress (else the wait never
    # ends) NOR error (else we abort the wait too early). Progress dialogs seen in the probe:
    # 'Computing Blips', 'Linking' (blip), 'Peeling'/'Computing' (peel). The real abort
    # signal is the long-shot OOM dialog 'SynthEyes Imminent Crash' -> match it precisely.
    _OP_PROGRESS_TERMS = ("computing", "linking", "peel", "blip", "solv", "refin",
                          "processing", "please wait", "working")
    _OP_ERROR_TERMS = ("imminent", "out of memory")

    def _op_dialog_state(self):
        """(progress_present, error_present) across SynthEyes #32770 dialogs, matched by
        KNOWN titles so benign dialogs (crash-recovery, license, tips) are ignored."""
        user32 = ctypes.windll.user32
        try:
            import psutil
            pids = {p.info["pid"] for p in psutil.process_iter(["name", "pid"])
                    if "syntheyes" in (p.info["name"] or "").lower()}
        except ImportError:
            return (False, False)
        if not pids:
            return (False, False)
        state = {"prog": False, "err": False}
        EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _cb(hwnd, _lp):
            if not user32.IsWindowVisible(hwnd):
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in pids:
                c = ctypes.create_unicode_buffer(64)
                user32.GetClassNameW(hwnd, c, 64)
                if c.value == "#32770":
                    t = ctypes.create_unicode_buffer(160)
                    user32.GetWindowTextW(hwnd, t, 160)
                    title = (t.value or "").strip().lower()
                    if title:
                        if any(b in title for b in self._OP_ERROR_TERMS):
                            state["err"] = True
                        elif any(b in title for b in self._OP_PROGRESS_TERMS):
                            state["prog"] = True
            return True

        user32.EnumWindows(EnumProc(_cb), 0)
        return (state["prog"], state["err"])

    def _tracker_count(self):
        """How many trackers the scene holds, or None if SyPy3 won't say.

        This is what Peel produces, so it is the one signal that proves peel ran. Only
        called when the op is already idle -- never mid-op, because a SyPy3 call while
        SynthEyes is busy is what desyncs the socket on 2026.2.4679.
        """
        hlev = getattr(self, "hlev", None)
        if hlev is None:
            return None
        for get in (lambda: len(list(hlev.Trackers())),
                    lambda: int(hlev.NumTrackers())):
            try:
                return int(get())
            except Exception:
                continue
        return None

    def _made_progress(self, label, progress_fn, before):
        """True when the op's own output changed, i.e. it really did run.

        Measured 2026-08 on build 2026.2.4679: Blip All peaks at ~3162% CPU and raises a
        progress dialog; Peel All peaks at ~47% and raises none, yet takes the scene from
        0 to 120 trackers. A single CPU bar cannot cover both, so an op that reports no OS
        signal gets asked whether it produced anything before being declared dead.
        """
        if progress_fn is None or before is None:
            return False
        after = progress_fn()
        if after is None or after == before:
            return False
        self.log(f"   {label}: no CPU peak or dialog, but its output changed "
                 f"({before} -> {after}) -> it ran")
        return True

    def _wait_for_operation(self, label, min_wait=3.0, max_timeout=600.0,
                            peak_cpu=250.0, idle_hold=1.5, start_grace=6.0,
                            progress_fn=None, progress_before=None):
        """Wait for a SynthEyes long-op (blip/peel) to finish via live signals instead of a
        fixed sleep. Falls back to a fixed sleep if psutil is unavailable.

        Returns a STATUS STRING, not a bool, because the caller needs to tell apart the two
        very different ways this returns "not done":

          "done"          the op ran and finished (CPU peaked, progress dialog seen, or --
                          when neither fired -- progress_fn shows its output changed).
          "never-started" released or hit the ceiling having seen NEITHER a CPU peak nor a
                          progress dialog, AND produced no output. The op never actually
                          began -- almost always a PostMessage click SynthEyes ignored
                          (window not foreground, a modal up). This used to be
                          indistinguishable from "done", so Peel ran on zero blips and the
                          shot exported 0 trackers with no error anywhere.
          "error"         a crash/OOM ('Imminent Crash') dialog appeared.
          "stopped"       the user pressed Stop.

        progress_fn / progress_before are the escape hatch for a cheap op: the CPU bar is
        tuned for the massively-parallel blip, and peel finishes far below it with no
        dialog at all, so without this peel is reported dead every single time.
        """
        pid = self._syntheyes_pid()
        proc = None
        try:
            import psutil
            if pid:
                proc = psutil.Process(pid)
                proc.cpu_percent(None)  # prime the delta
        except Exception:
            proc = None

        t0 = time.time()
        peaked = False
        seen_dialog = False
        low_since = None
        while time.time() - t0 < max_timeout:
            if getattr(self, "_stop_requested", False):
                return "stopped"
            prog, err = self._op_dialog_state()
            if err:
                self.log(f"   {label}: SynthEyes crash/error dialog -> aborting op")
                return "error"
            if proc is not None:
                try:
                    cpu = proc.cpu_percent(interval=0.3)
                except Exception:
                    self.log(f"   {label}: SynthEyes process gone during wait")
                    return "error"
            else:
                cpu = 0.0
                time.sleep(0.3)
            el = time.time() - t0
            if prog:
                seen_dialog = True
            if cpu >= peak_cpu:
                peaked = True
            working = prog or (cpu >= peak_cpu)
            if working:
                low_since = None
            else:
                if low_since is None:
                    low_since = time.time()
                idled = time.time() - low_since
                # release once idle has held long enough, past min_wait, and we've either
                # seen real work (CPU peak / progress dialog) or waited out start_grace (for
                # an instant/no-op op like peel on a short shot, or no-psutil fallback).
                ready = peaked or seen_dialog or (proc is None) or (el >= start_grace)
                if el >= min_wait and idled >= idle_hold and ready:
                    # No psutil = no way to tell work from idle, so the old fixed-sleep
                    # behavior stands (assume it ran). With psutil, seeing neither signal
                    # means the click never landed -- report that instead of a false "done".
                    if proc is not None and not (peaked or seen_dialog):
                        # Neither OS signal fired. That is NOT proof the click was ignored:
                        # a cheap single-threaded op finishes below the CPU bar and puts up
                        # no dialog. Ask the op itself whether it produced anything before
                        # calling it dead.
                        if self._made_progress(label, progress_fn, progress_before):
                            return "done"
                        self.log(f"   {label}: released at ~{el:.1f}s but NO work was ever "
                                 f"observed (no CPU peak, no progress dialog) -> the click "
                                 f"most likely did not register")
                        return "never-started"
                    self.log(f"   {label}: complete at ~{el:.1f}s (dialog gone + CPU idle)")
                    return "done"
        if proc is not None and not (peaked or seen_dialog):
            if self._made_progress(label, progress_fn, progress_before):
                return "done"
            self.log(f"   {label}: hit {max_timeout:.0f}s ceiling having never seen the op run")
            return "never-started"
        self.log(f"   {label}: hit {max_timeout:.0f}s ceiling -> proceeding")
        return "done"

    # ------ blip/peel (single-pass + long-shot chunked) ------

    def _run_panel_op(self, label, variants, fallback_action, timeout, attempts=2,
                      progress_fn=None):
        """Click a Features-panel button and WAIT for the op to really run, retrying when it
        didn't. Raises RuntimeError if it never ran.

        _win32_click_button only proves the button was found and a message was posted -- not
        that SynthEyes acted on it. A click that silently doesn't register looked exactly like
        a completed op, so Peel then ran on zero blips and the shot exported 0 trackers with
        nothing in the log to say why. This re-asserts the room + foreground and re-clicks,
        which is the same recover-and-retry shape _blip_peel_chunked already uses.
        """
        for attempt in range(1, attempts + 1):
            # The panel's child controls only exist while the window is up and in the
            # Features room; a lost foreground is exactly why a click gets dropped.
            self._ensure_features_room()
            self._bring_foreground()
            # Baseline BEFORE the click, and re-read per attempt: a retry follows an attempt
            # that may itself have produced some output.
            before = progress_fn() if progress_fn else None
            if not self._win32_click_button(label, variants):
                self.log(f"   Win32 '{label}' unavailable - falling back to SyPy3 dispatch")
                if self._click_and_wait(fallback_action, label, timeout=timeout):
                    return
                raise RuntimeError(f"{label} failed: button not found and SyPy3 dispatch failed")
            self.log(f"   {label} dispatched... (waiting for completion signal)")
            st = self._wait_for_operation(label, min_wait=3.0, max_timeout=timeout,
                                          progress_fn=progress_fn, progress_before=before)
            if st == "done":
                return
            if st == "stopped":
                return          # user pressed Stop; the caller's own stop checks take over
            if st == "error":
                # crash/OOM dialog -> clear it and the half-built blips before retrying
                self._dismiss_error_dialog()
                self._win32_click_button("Clear blips", ["Clear all blips"])
                if not self.is_alive():
                    raise RuntimeError(f"SynthEyes crashed during {label}")
            if attempt < attempts:
                self.log(f"   {label}: '{st}' -> retrying ({attempt + 1}/{attempts})")
        raise RuntimeError(
            f"{label} never ran after {attempts} attempts (last status '{st}'). "
            f"SynthEyes ignored the panel click - the shot would have exported 0 tracks.")

    def _blip_peel_full(self, frame_count, blip_timeout, peel_timeout):
        """Single-pass: blip ALL frames, peel, clear. Fine for shots that fit in memory."""
        self.log("-> Blip All frames (Win32 PostMessage)...")
        self._run_panel_op("Blip All", ["Blips all frames", "Blip all frames"],
                           "Feature/Blips all frames", blip_timeout)

        self.log("-> Peel All (Win32 PostMessage)...")
        # Peel gets a progress probe: it draws too little CPU and raises no dialog, so the
        # OS signals alone report it dead on every shot (measured on 2026.2.4679).
        self._run_panel_op("Peel All", ["Peel all", "Peel All"],
                           "Feature/Peel All", peel_timeout,
                           progress_fn=self._tracker_count)

        self.log("-> Clearing blips...")
        if not self._win32_click_button("Clear blips", ["Clear all blips"]):
            self._click_and_wait("Feature/Clear all blips", "Clear blips", timeout=10)

    def _dismiss_error_dialog(self):
        """Click OK / close any SynthEyes crash/OOM #32770 dialog so a chunked run can
        recover and retry with a smaller window."""
        user32 = ctypes.windll.user32
        try:
            import psutil
            pids = {p.info["pid"] for p in psutil.process_iter(["name", "pid"])
                    if "syntheyes" in (p.info["name"] or "").lower()}
        except ImportError:
            return
        targets = []
        EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _cb(hwnd, _lp):
            if not user32.IsWindowVisible(hwnd):
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in pids:
                c = ctypes.create_unicode_buffer(64)
                user32.GetClassNameW(hwnd, c, 64)
                if c.value == "#32770":
                    t = ctypes.create_unicode_buffer(160)
                    user32.GetWindowTextW(hwnd, t, 160)
                    if any(b in (t.value or "").lower() for b in self._OP_ERROR_TERMS):
                        targets.append(hwnd)
            return True

        user32.EnumWindows(EnumProc(_cb), 0)
        for h in targets:
            ok = user32.GetDlgItem(h, 1)  # IDOK
            if ok:
                user32.PostMessageW(ok, _WM_LBUTTONDOWN, _MK_LBUTTON, 0)
                user32.PostMessageW(ok, _WM_LBUTTONUP, 0, 0)
            else:
                user32.PostMessageW(h, 0x0010, 0, 0)  # WM_CLOSE
            self.log("   dismissed a SynthEyes crash/error dialog")
        if targets:
            time.sleep(0.5)

    def _blip_peel_chunked(self, shot, start, end, frame_count, blip_timeout, peel_timeout,
                           img_w=None, img_h=None):
        """Long-shot path: blip/peel one playback-range WINDOW at a time so SynthEyes never
        holds blips for the whole shot at once (that OOMs -> 'Imminent Crash'). Blips are
        cleared after each window's peel to free their (heavy) memory; peeled trackers
        ACCUMULATE (they are light) and the full-range Sizzle export afterward covers the
        whole shot. 'Blips playback range' + SetAnimStart/End confine the blip to the window
        (probe-verified).

        Window size is ADAPTIVE, not a fixed constant:
          A) seeded from plate resolution -- chunk_frames_per_mp / megapixels, anchored to a
             known-safe ~600 frames @ 4K, so HD gets bigger windows and 8K smaller;
          B) self-calibrated live -- the first window's SynthEyes RSS growth gives real
             MB/frame on THIS machine+settings, then later windows are sized to a RAM ceiling
             and shrink as accumulated trackers raise the baseline (never grows past the
             resolution-safe seed);
          C) crash-guarded -- if the 'Imminent Crash' dialog fires mid-blip, dismiss it,
             halve the window, and retry.
        chunk_size>0 forces a fixed window (manual override)."""
        import threading
        try:
            import psutil
        except ImportError:
            psutil = None
        proc = None
        if psutil:
            pid = self._syntheyes_pid()
            try:
                proc = psutil.Process(pid) if pid else None
            except Exception:
                proc = None

        def rss():
            try:
                return proc.memory_info().rss if proc else 0
            except Exception:
                return 0

        overlap = int(self._s("chunk_overlap") or 24)
        min_win = max(30, int(self._s("chunk_min_window") or 120))
        fixed = int(self._s("chunk_size") or 0)   # >0 => fixed window (disable adaptive size)

        # A) resolution-seeded window
        mp = 8.3
        if img_w and img_h:
            mp = max(0.5, (float(img_w) * float(img_h)) / 1e6)
        frames_per_mp = float(self._s("chunk_frames_per_mp") or 5000.0)
        res_seed = fixed if fixed > 0 else int(frames_per_mp / mp)

        # B) RAM ceiling for adaptive sizing
        ram_frac = float(self._s("chunk_ram_frac") or 0.5)
        base0 = rss()
        ceiling = None
        if proc and psutil:
            try:
                ceiling = base0 + int(psutil.virtual_memory().available * ram_frac)
            except Exception:
                ceiling = None

        # Cap the SEED window by the RAM ceiling too (via a resolution-based MB/frame prior),
        # so the FIRST window -- before live calibration -- can't itself OOM when free RAM is
        # tight (the full pipeline keeps torch/SAM3 resident in-process, leaving less free RAM
        # than a bare tracking run; that's what hung the first full-bot attempt on SH005).
        win0 = max(min_win, min(frame_count, res_seed))
        if fixed <= 0 and ceiling:
            prior_bytes_per_frame = 7.5e6 * mp    # ~60 MB/frame @ 4K (8.3 MP), scales w/ res
            room = max(0, ceiling - base0)
            win0 = max(min_win, min(win0, int(room / prior_bytes_per_frame)))
        rss_per_frame = None

        self.log(f"-> Long shot ({frame_count} frames, ~{mp:.1f}MP): "
                 f"{'fixed' if fixed > 0 else 'adaptive'} chunking, seed window {win0}"
                 + (f", RSS base {base0/1e9:.1f}GB" if proc else " (no psutil)"))

        # SynthEyes frames + playback range are 0-BASED (probe: a 300-frame shot reports
        # shot.start 0 / shot.stop 299). The caller's start/end are 1-based plate numbers, so
        # window the loop in 0-based indices -- otherwise 'Blips playback range' on [1..] skips
        # raw frame 0 (= plate frame 1) and that frame exports with no tracks.
        s0 = max(0, start - 1)
        e0 = max(s0, end - 1)
        a = s0
        idx = 0
        win = win0
        while a <= e0:
            if getattr(self, "_stop_requested", False):
                self.log("   chunked blip/peel stopped by user."); return
            idx += 1
            # size this window: shrink toward the ceiling as the baseline creeps up with
            # accumulated trackers; never exceed the resolution-safe seed.
            if fixed <= 0 and rss_per_frame and ceiling:
                room = max(0, ceiling - rss())
                win = max(min_win, min(win0, int(room / rss_per_frame)))
            b = min(a + win - 1, e0)

            # Per-window HANG WATCHDOG: a stuck SynthEyes blocks SyPy3 socket calls with no
            # timeout (that hung the first full-bot SH005 run at _set_frame_range). If a window
            # overruns its hard deadline, kill SynthEyes to unblock the socket, then abort the
            # shot cleanly (worker_track logs it) instead of hanging forever.
            wd_fired = {"v": False}
            deadline = max(300.0, (b - a + 1) * 2.0)

            def _watchdog(a=a, b=b, deadline=deadline):
                wd_fired["v"] = True
                self.log(f"   window {a}-{b}: watchdog {deadline:.0f}s exceeded -> killing hung SynthEyes")
                try:
                    self.kill_syntheyes()
                except Exception:
                    pass

            wd = threading.Timer(deadline, _watchdog)
            wd.daemon = True
            wd.start()
            try:
                # C) blip the window; on a crash/OOM dialog, shrink and retry
                attempt = 0
                rss_before = rss()
                while True:
                    attempt += 1
                    self._set_frame_range(shot, a, b)
                    self._ensure_features_room()
                    self._bring_foreground()
                    wto = max(120, int((b - a + 1) * 0.3))
                    if self._win32_click_button("Blips playback range",
                                                ["Blips playback range", "Blip playback range"]):
                        st = self._wait_for_operation(f"Blip w{idx}", min_wait=2.0, max_timeout=wto)
                    else:
                        self.log("   'Blips playback range' not found - falling back to Blip All")
                        self._win32_click_button("Blip All", ["Blips all frames", "Blip all frames"])
                        st = self._wait_for_operation(f"Blip w{idx}", min_wait=2.0, max_timeout=blip_timeout)
                    # 'never-started' joins 'error' on the retry path: both mean this window
                    # produced nothing, and a re-click is exactly the right recovery.
                    ok = (st == "done")
                    if ok or st == "stopped" or getattr(self, "_stop_requested", False):
                        break
                    # crash/OOM signal -> recover + shrink + retry
                    self._dismiss_error_dialog()
                    self._win32_click_button("Clear blips", ["Clear all blips"])
                    if not self.is_alive():
                        raise RuntimeError("SynthEyes crashed during chunked blip (OOM)")
                    shrunk = max(min_win, (b - a + 1) // 2)
                    if shrunk >= (b - a + 1) or attempt > 6:
                        self.log(f"   window {a}-{b}: cannot shrink further -> proceeding"); break
                    win = shrunk
                    b = min(a + win - 1, e0)
                    why = "click did not register" if st == "never-started" else "memory pressure"
                    self.log(f"   {why} -> shrink window to plate {a+1}-{b+1} ({win}f) and retry")

                # B) calibrate MB/frame from the first successful window
                if fixed <= 0 and rss_per_frame is None and proc:
                    peak = rss()
                    frames = max(1, b - a + 1)
                    if peak > rss_before:
                        rss_per_frame = (peak - rss_before) / frames
                        self.log(f"   calibrated ~{rss_per_frame/1e6:.1f} MB/frame @ {mp:.1f}MP; "
                                 f"later windows sized to the RAM ceiling")

                self.log(f"   [window {idx}] plate frames {a+1}-{b+1} ({b - a + 1}f) blipped")
                trk_before = self._tracker_count()
                if self._win32_click_button("Peel All", ["Peel all", "Peel All"]):
                    pst = self._wait_for_operation(f"Peel w{idx}", min_wait=2.0,
                                                   max_timeout=max(200, int((b - a + 1) * 1.2)),
                                                   progress_fn=self._tracker_count,
                                                   progress_before=trk_before)
                    # Trackers ACCUMULATE across windows, so one dead peel loses only this
                    # window -- warn (so a pattern is visible in the log) rather than abort
                    # the shot, which is the caller's call once the export count is known.
                    if pst == "never-started":
                        self.log(f"   WARNING: window {idx} peel never ran - that window "
                                 f"contributed no trackers")
                self._win32_click_button("Clear blips", ["Clear all blips"])  # free blip memory
            finally:
                wd.cancel()

            if wd_fired["v"] or not self.is_alive():
                raise RuntimeError(f"SynthEyes hung/crashed on window {a}-{b} (watchdog); shot aborted")

            if b >= e0:
                break
            a = b + 1 - overlap

        # restore the full (0-based) range so the Sizzle export walks the whole shot
        self._set_frame_range(shot, s0, e0)
        self.log(f"-> Adaptive chunked blip/peel done ({idx} windows); trackers accumulated.")

    def _bring_foreground(self):
        """Show + foreground the SynthEyes main window so the Features panel actually
        paints and its blip/peel child controls get instantiated (they don't exist while
        the window is backgrounded). Uses the alt-key trick to defeat the Win32
        foreground lock, then gives it a moment to repaint."""
        tops = self._syntheyes_top_windows()
        if not tops:
            return
        user32 = ctypes.windll.user32

        def _area(h):
            r = wintypes.RECT()
            user32.GetWindowRect(h, ctypes.byref(r))
            return (r.right - r.left) * (r.bottom - r.top)

        main = max(tops, key=_area)
        SW_RESTORE = 9
        try:
            user32.ShowWindow(main, SW_RESTORE)
            user32.keybd_event(0x12, 0, 0, 0)          # ALT down (unlock SetForegroundWindow)
            user32.SetForegroundWindow(main)
            user32.keybd_event(0x12, 0, 0x0002, 0)     # ALT up
            user32.BringWindowToTop(main)
        except Exception as e:
            self.log(f"   (foreground nudge failed: {e})")
        time.sleep(0.8)
        try:
            self.hlev.Redraw()
        except Exception:
            pass
        time.sleep(0.4)

    def _dump_all_children(self):
        """One-time full window-tree dump (all top windows, child class -> texts) to locate
        where the blip/peel controls live when text-matching fails."""
        if getattr(self, "_tree_dumped", False):
            return
        self._tree_dumped = True
        user32 = ctypes.windll.user32
        for wi, top in enumerate(self._syntheyes_top_windows()):
            cb = ctypes.create_unicode_buffer(64); user32.GetClassNameW(top, cb, 64)
            tb = ctypes.create_unicode_buffer(128); user32.GetWindowTextW(top, tb, 128)
            self.log(f"   [win{wi}] class={cb.value} text='{tb.value}'")
            classes = {}
            EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

            def _cb(h, _l):
                c = ctypes.create_unicode_buffer(64); user32.GetClassNameW(h, c, 64)
                t = ctypes.create_unicode_buffer(64); user32.GetWindowTextW(h, t, 64)
                classes.setdefault(c.value, []).append(t.value)
                return True

            user32.EnumChildWindows(top, EnumProc(_cb), 0)
            for cls, texts in sorted(classes.items()):
                named = [x for x in texts if x][:16]
                self.log(f"       {cls} x{len(texts)} {named}")

    def _winrect(self, hwnd):
        r = wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
        return r

    def _find_child_by_class(self, parent, cls_name):
        user32 = ctypes.windll.user32
        out = []
        EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _cb(h, _l):
            c = ctypes.create_unicode_buffer(64); user32.GetClassNameW(h, c, 64)
            if c.value == cls_name:
                r = wintypes.RECT(); user32.GetWindowRect(h, ctypes.byref(r))
                out.append((h, (r.left, r.top, r.right, r.bottom)))
            return True

        user32.EnumChildWindows(parent, EnumProc(_cb), 0)
        return out

    def _click_features_tab(self):
        """Reach the Features room by sweep-clicking the TopTabber room-selector until the
        blip button appears (SetRoom is dead on 2026.2.4679, and TopTabber is a custom
        control with no per-tab child window, so we hit-test it by PostMessage). Self-finds
        the Features tab - no hardcoded pixel."""
        tops = self._syntheyes_top_windows()
        if not tops:
            return False
        main = max(tops, key=lambda h: (lambda r: (r.right - r.left) * (r.bottom - r.top))(self._winrect(h)))
        tabbers = self._find_child_by_class(main, "TopTabber")
        if not tabbers:
            self.log("   WARNING: TopTabber room-selector not found")
            return False
        thwnd, (l, t, r, b) = tabbers[0]
        w, h = r - l, b - t
        self.log(f"   TopTabber found: {w}x{h} px - sweeping for Features tab")
        user32 = ctypes.windll.user32
        y = h // 2
        step = max(20, w // 30)
        x = step // 2
        while x < w:
            lp = ((y & 0xFFFF) << 16) | (x & 0xFFFF)
            user32.PostMessageW(thwnd, _WM_LBUTTONDOWN, _MK_LBUTTON, lp)
            time.sleep(0.03)
            user32.PostMessageW(thwnd, _WM_LBUTTONUP, 0, lp)
            time.sleep(0.2)
            try:
                self.hlev.Redraw()
            except Exception:
                pass
            time.sleep(0.15)
            if self._features_blip_button_present():
                self.log(f"   OK  Features room reached via TopTabber click at x={x}")
                return True
            x += step
        self.log("   WARNING: swept the whole TopTabber, Features room not reached")
        return False

    def _features_blip_button_present(self):
        """True if the Features-room 'Blips all frames' Butt child currently exists."""
        for w in self._syntheyes_top_windows():
            for _h, txt, _r in self._enum_butt_buttons(w):
                if "".join((txt or "").lower().split()) in ("blipsallframes", "blipallframes"):
                    return True
        return False

    def _ensure_features_room(self):
        """Get SynthEyes onto the Features room so its blip/peel buttons instantiate.

        IMPORTANT: do NOT use SyPy3 SetRoom here. On 2026.2.4679 SetRoom is a no-op that
        DESYNCS the socket, which then breaks the later RunScriptFile Sizzle export. Reach
        Features purely by clicking the TopTabber room tab (Win32, no socket calls) so the
        socket stays clean for export."""
        self._bring_foreground()
        if self._features_blip_button_present():
            self.log("   OK  Features room already active - blip button present")
            return True
        if self._click_features_tab():
            return True
        self.log("   WARNING: Features-room blip button never appeared (TopTabber tab click failed)")
        return False

    # ------ process lifecycle ------

    def kill_syntheyes(self):
        killed = False
        try:
            import psutil
            for p in psutil.process_iter(["name", "pid"]):
                if "syntheyes" in (p.info["name"] or "").lower():
                    try:
                        self.log(f"   Terminating SynthEyes PID {p.info['pid']}...")
                        p.terminate()
                        try:
                            p.wait(timeout=5)
                        except Exception:
                            p.kill()
                            p.wait(timeout=5)
                        killed = True
                    except Exception as e:
                        self.log(f"   WARNING: kill failed: {e}")
        except ImportError:
            pass
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", "SynthEyes64.exe"],
                capture_output=True, text=True
            )
            if "success" in result.stdout.lower():
                killed = True
            subprocess.run(["taskkill", "/F", "/IM", "SynthEyes.exe"],
                           capture_output=True, text=True)
        except Exception:
            pass
        if killed:
            self.log("   SynthEyes terminated. Waiting 3s...")
            time.sleep(3)
        else:
            self.log("   No existing SynthEyes instance found.")

    def launch(self):
        exe = self._s("syntheyes_exe")
        port = self._s("port")
        pin = self._s("pin")
        wait = self._s("startup_wait")
        self.kill_syntheyes()
        if not exe or not os.path.isfile(exe):
            self.log(f"ERROR: SynthEyes not found at: {exe}")
            return False
        self.log("-> Launching SynthEyes...")
        subprocess.Popen([exe, "-l", str(port), "-pin", pin])
        grace = min(int(wait or 3), 3)  # short spawn grace; connect() polls for readiness
        self.log(f"   Launched; giving {grace}s to spawn, then polling for the listener...")
        time.sleep(grace)
        return True

    def connect(self, timeout=60, interval=2.0):
        """Poll for the SynthEyes SyPy listener. Cold-start (license/Synthia/splash) can
        take far longer than a single attempt — OpenExisting makes a fresh SyPyCore each
        call, so retrying until the socket is up is safe."""
        from SyPy3.sylevel import SyLevel
        port = self._s("port")
        pin = self._s("pin")
        self.log(f"-> Connecting port={port} (up to {timeout}s)...")
        deadline = time.time() + timeout
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                self.hlev = SyLevel()
                if self.hlev.OpenExisting(port, pin):
                    self.log(f"OK  Connected - SynthEyes {self.hlev.Version()} (after {attempt} tries)")
                    return True
            except Exception as e:
                if attempt == 1:
                    self.log(f"   (waiting for SynthEyes... {e})")
            if self._stop_requested:
                self.log("   Connect aborted (stop requested).")
                return False
            time.sleep(interval)
        self.log(f"   Connection failed (no listener on port {port} after {timeout}s).")
        return False

    def connect_or_launch(self):
        self.log("-> Checking for existing SynthEyes instance...")
        try:
            from SyPy3.sylevel import SyLevel
            port = self._s("port")
            pin = self._s("pin")
            test = SyLevel()
            if test.OpenExisting(port, pin):
                self.hlev = test
                self.log(f"OK  Reusing existing SynthEyes - {self.hlev.Version()}")
                return True
            self.log("   No existing instance found.")
        except Exception as e:
            self.log(f"   Could not connect: {e}")
        if not self.launch():
            return False
        if not self.connect():
            self.log("ERROR: Launched SynthEyes but could not connect!")
            return False
        return True

    def is_alive(self):
        # Check the OS process FIRST: if SynthEyes is gone, never call Version() — a SyPy3
        # socket read against a dead/hung instance can block the caller indefinitely.
        if self._syntheyes_pid() is None:
            return False
        try:
            v = self.hlev.Version()
            return v is not None and len(str(v)) > 0
        except Exception:
            return False

    def restart(self):
        self.log("-> RESTARTING SynthEyes (memory cleanup)...")
        try:
            self.hlev.Close()
        except Exception:
            pass
        self.hlev = None
        self.kill_syntheyes()
        time.sleep(2)
        if not self.launch():
            self.log("   ERROR: Relaunch failed!")
            return False
        if not self.connect():
            self.log("   ERROR: Reconnect failed after relaunch!")
            return False
        self.log("   OK  SynthEyes restarted with clean memory")
        return True

    def set_writable_folder(self, folder_path):
        out = os.path.normpath(folder_path)
        os.makedirs(out, exist_ok=True)
        try:
            self.hlev.BeginPref()
            self.hlev.SetFolderPref("SCENE", out)
            self.hlev.SetFolderPref("EXPORT", out)
            self.hlev.AcceptPref()
            self.log(f"   OK  Writable folders -> {out}")
        except Exception as e:
            self.log(f"   WARNING: Could not set folder prefs: {e}")
            try:
                self.hlev.AcceptPref()
            except Exception:
                pass

    def flush_after_shot(self, shot):
        try:
            self.hlev.FlushShot(shot)
            self.log("   OK  RAM cache flushed")
        except Exception as e:
            self.log(f"   WARNING: FlushShot failed: {e}")
        try:
            self.hlev.FlushUndo()
            self.log("   OK  Undo history flushed")
        except Exception as e:
            self.log(f"   WARNING: FlushUndo failed: {e}")

    # ------ per-shot processing ------

    def process_shot(self, shot_name, output_dir, out_txt_name,
                     first_frame, all_frames, frame_count, start_frame, end_frame,
                     track_count=500, mask_dir=None, image_width=None, image_height=None,
                     movie_path=None):
        """Track one shot in SynthEyes and export 2D tracks (native format).

        Input is either an image sequence (built into an IFL) or a movie file
        (`movie_path`, loaded directly — SynthEyes decodes movies natively).

        Returns (tracker_count, out_txt_path).
        """
        hlev = self.hlev
        os.makedirs(output_dir, exist_ok=True)

        track_threshold = float(self._s("threshold"))
        track_separation = int(self._s("separation"))

        self.log(f"\n{'='*55}")
        self.log(f"  Processing: {shot_name}")
        self.log(f"  Max Tracks: {track_count}")
        self.log(f"  Output:     {os.path.join(output_dir, out_txt_name)}")
        self.log(f"{'='*55}")

        self.set_writable_folder(output_dir)

        if movie_path:
            load_path = os.path.normpath(movie_path)
            if not os.path.isfile(load_path):
                raise RuntimeError(f"Movie file NOT FOUND: {load_path}")
            if not os.access(load_path, os.R_OK):
                raise RuntimeError(f"No READ permission on: {load_path}")
            first_frame = load_path
            self.log(f"-> Loading movie: {os.path.basename(load_path)}")
        else:
            first_frame = os.path.normpath(first_frame)
            all_frames = [os.path.normpath(f) for f in all_frames]
            if not os.path.isfile(first_frame):
                raise RuntimeError(f"First frame file NOT FOUND: {first_frame}")
            if not os.access(first_frame, os.R_OK):
                raise RuntimeError(f"No READ permission on: {first_frame}")
            ifl_path = os.path.join(output_dir, os.path.splitext(out_txt_name)[0] + ".ifl")
            self.log(f"-> Creating IFL ({len(all_frames)} frames)...")
            try:
                with open(ifl_path, "w", encoding="utf-8") as f:
                    for frame_path in all_frames:
                        safe_path = os.path.abspath(frame_path).replace("\\", "/")
                        f.write(safe_path + "\n")
            except Exception as e:
                raise RuntimeError(f"Could not write IFL file: {e}")
            load_path = ifl_path
            self.log("-> Loading sequence via IFL...")

        shot = hlev.NewSceneAndShot(load_path)
        if shot is None:
            raise RuntimeError("NewSceneAndShot returned None - check input path")

        if frame_count < 300:
            try:
                hlev.Validate(shot)
                self.log("   OK  Validated shot RAM cache")
            except Exception as e:
                self.log(f"   WARNING: Validate failed: {e}")

        try:
            hlev.Redraw()
        except Exception:
            pass

        load_wait = 10 if frame_count > 500 else 5
        self.log(f"   Waiting {load_wait}s for image plane to load...")
        time.sleep(load_wait)

        self._set_frame_range(shot, start_frame, end_frame)
        self.log(f"   Frames: {start_frame}-{end_frame} ({frame_count} total)")

        try:
            hlev.Redraw()
        except Exception:
            pass

        # NOTE: SAM3 masks are applied as a Python POST-FILTER after export (see
        # _apply_sam3_postfilter), NOT as a SynthEyes roto matte. The roto path
        # (_load_sam3_matte) switches to the "Roto Masking" room, which desyncs the SyPy3
        # socket on 2026.2.4679 and is fragile; the Python gating is more robust + exact.

        self.log("-> Switching to Features room (via TopTabber tab click)...")
        # Hard-fail if the room can't be reached. Without it the blip/peel buttons don't
        # exist, every click falls through to the SyPy3 dispatch that no-ops on 2026.2.4679,
        # and the shot silently exports 0 trackers AFTER burning the whole blip/peel budget.
        # One retry first: the usual cause is a lost foreground, which _bring_foreground fixes.
        if not self._ensure_features_room():
            self.log("   Features room not reached - retrying once after a foreground nudge")
            self._bring_foreground()
            if not self._ensure_features_room():
                raise RuntimeError(
                    "Could not reach the SynthEyes Features room (blip/peel buttons never "
                    "appeared). Tracking this shot would produce 0 trackers.")
        # Applies the tracker count via the Advanced dialog; see _configure_features.
        # (It used to be disabled here because the old implementation opened that dialog
        # with ByID(1238).ClickAndContinue(), which desynced the SyPy3 socket. It now opens
        # it with the same Win32 button click blip/peel use, so that hazard is gone.)
        self._configure_features(track_count, track_threshold, track_separation)

        # Blip All + Peel All via Win32 PostMessage. The SyPy3 dispatch (_click_and_wait)
        # silently no-ops on 2026.2.4679, so PostMessage the real "Butt" panel buttons and
        # wait a frame-scaled budget (there is no reliable API to poll blip completion).
        # SyPy3 remains the fallback if the buttons can't be located.
        # Foreground the window so the Features panel paints and its blip/peel child
        # controls instantiate (they don't exist while backgrounded on 2026.2.4679).
        self.log("-> Bringing SynthEyes foreground for panel clicks...")
        self._bring_foreground()

        blip_timeout = max(400, int(frame_count * 0.2))
        peel_timeout = max(600, int(frame_count * 1.0))
        chunk_threshold = int(self._s("chunk_threshold") or 1000)
        if bool(self._s("chunk_long_shots")) and frame_count > chunk_threshold:
            # Long shots: blip a playback-range window at a time so SynthEyes never holds
            # blips for thousands of 4K frames at once (that OOMs -> 'Imminent Crash').
            # Pass the plate resolution so the window size adapts to it (see _blip_peel_chunked).
            cw, ch = image_width, image_height
            if not cw or not ch:
                cw, ch = read_image_size(first_frame)
            self._blip_peel_chunked(shot, start_frame, end_frame, frame_count,
                                    blip_timeout, peel_timeout, img_w=cw, img_h=ch)
        else:
            self._blip_peel_full(frame_count, blip_timeout, peel_timeout)

        # 2026: there is no "Trackers" room -- trackers live on the Summary panel.
        # Feeding SetRoom() an invalid name DESYNCS the SyPy3 socket (all later
        # Run() return bogus True), which corrupted the export step. Use "Summary".
        # Export via Sizzle (server-side openout+printf). This reads obj.trk directly, so it
        # does NOT depend on the broken SyPy3 Trackers()/menu-export path, and works from the
        # Features room (no Summary switch needed).
        out_txt_path = os.path.join(output_dir, out_txt_name)
        self.log("-> Exporting 2D tracks via Sizzle...")
        n_trk = self._sizzle_export_3de(out_txt_path)

        # SAM3 gating in Python (yesterday's proven method): drop mover points, truncate
        # the Demo frozen tail. Beats the fragile SynthEyes roto matte.
        #
        # Best-effort, like _filter_exported_tracks: a crash in the gating must not throw
        # away a finished export. It did once -- a sampling TypeError after a clean
        # 120-tracker export failed the shot, sent it to the TAPNext retry, and published
        # TAPNext tracks for a shot SynthEyes had actually tracked. The export is the
        # expensive part; keep it. The tracks are then UNGATED, which matters (mover points
        # survive), so say so loudly rather than letting it pass as a normal run.
        if n_trk > 0 and mask_dir:
            try:
                res = self._apply_sam3_postfilter(out_txt_path, mask_dir)
                if res is not None:  # (0, 0) means "gated to nothing" -- still a real result
                    n_trk = res[0]
            except Exception as e:
                self.log(f"   WARNING: SAM3 gating failed ({e}) - keeping all {n_trk} exported "
                         f"tracker(s) UNGATED. Points on masked movers were NOT removed; "
                         f"check this shot before solving.")

        if n_trk > 0:
            if image_width is None or image_height is None:
                image_width, image_height = read_image_size(first_frame)
            sidecar_path = self._write_3de_sidecar(
                output_dir=output_dir, shot_name=shot_name, txt_filename=out_txt_name,
                first_frame=first_frame, frame_count=frame_count,
                start_frame=start_frame, end_frame=end_frame,
                image_width=image_width, image_height=image_height,
            )
            if self._s("auto_3de") and sidecar_path:
                self._run_3de_import(sidecar_path, output_dir, shot_name, out_txt_name)
        else:
            self.log("   WARNING: Sizzle export returned no trackers")

        return max(0, n_trk), out_txt_path

    # ------ SAM3 matte loading (UI automation) ------

    def _load_sam3_matte(self, shot, mask_dir):
        """Load the SAM3 mask sequence as a SynthEyes alpha matte to gate trackers.

        Mechanism mirrors the original `_load_matte`: switch to Roto Masking, click
        "+ Alpha", and inject the path of the first mask frame into the native file
        dialog. SAM3 masks are white=keep / black=ignore — if SynthEyes treats the
        loaded alpha with the opposite polarity, invert the masks upstream or flip
        the matte sense in the Roto room (left as a follow-up if observed).
        """
        hlev = self.hlev
        frames = scan_frames(mask_dir, MASK_EXTENSIONS)
        if not frames:
            self.log(f"   No SAM3 masks in {mask_dir} - tracking without matte")
            return

        matte_first = frames[0][1]
        self.log(f"-> Loading SAM3 matte ({len(frames)} masks)")
        self.log(f"   Path: {matte_first}")

        matte_path_abs = os.path.abspath(matte_first)
        matte_dir = os.path.dirname(matte_path_abs)
        matte_file = os.path.basename(matte_path_abs)

        try:
            self.log("   Switching to Roto Masking tab...")
            self._switch_room("Roto Masking", "Roto", "RotoMasking")
            time.sleep(1)

            # set IMAGE folder pref to the mask dir before opening the dialog
            try:
                hlev.BeginPref()
                hlev.SetFolderPref("IMAGE", matte_dir)
                hlev.AcceptPref()
            except Exception as pref_e:
                self.log(f"   WARNING: Could not set matte folder pref: {pref_e}")
                try:
                    hlev.AcceptPref()
                except Exception:
                    pass

            alpha_clicked = False
            for btn_name in ["+ Alpha", "+Alpha", "+ alpha", "Alpha"]:
                try:
                    btn = hlev.Main().ByName(btn_name)
                    if btn.IsValid():
                        btn.ClickAndContinue()
                        alpha_clicked = True
                        self.log(f"   OK  '{btn_name}' clicked")
                        break
                except Exception:
                    pass
            if not alpha_clicked:
                self.log("   WARNING: '+ Alpha' button not found - matte not loaded")
                return

            time.sleep(2.0)  # let native file dialog construct
            self._inject_file_dialog_path(matte_path_abs, matte_file)

            try:
                hlev.Redraw()
            except Exception:
                pass
        except Exception as e:
            self.log(f"   WARNING: SAM3 matte loading failed: {e}")

    def _inject_file_dialog_path(self, path_abs, fname):
        """Push a path into the standard Windows file-open dialog (#32770)."""
        try:
            user32 = ctypes.windll.user32
            WM_SETTEXT = 0x000C
            BM_CLICK = 0x00F5

            dialog_hwnd = 0
            for _ in range(10):
                time.sleep(0.5)
                dialog_hwnd = user32.FindWindowW("#32770", None)
                if dialog_hwnd:
                    break

            if not dialog_hwnd:
                self.log("   WARNING: file dialog not found - blind typing")
                self._hardware_type_string(path_abs)
                time.sleep(0.5)
                self._hardware_key(0x0D)
                time.sleep(1.0)
                return

            edit_hwnd = user32.GetDlgItem(dialog_hwnd, 1148)
            ok_hwnd = user32.GetDlgItem(dialog_hwnd, 1)
            if edit_hwnd and ok_hwnd:
                user32.SendMessageW(edit_hwnd, WM_SETTEXT, 0, ctypes.c_wchar_p(path_abs))
                time.sleep(0.1)
                user32.SendMessageW(ok_hwnd, BM_CLICK, 0, 0)
                time.sleep(1.0)
                self.log(f"   OK  Matte file loaded: {fname}")
            else:
                self._hardware_type_string(path_abs)
                time.sleep(0.5)
                self._hardware_key(0x0D)
                time.sleep(1.0)
        except Exception as e:
            self.log(f"   WARNING: file dialog interaction failed: {e}")

    # ------ frame range ------

    def _set_frame_range(self, shot, start, end):
        hlev = self.hlev
        try:
            hlev.SetAnimStart(start)
            hlev.SetAnimEnd(end)
            hlev.Begin()
            for a_s, a_e in [("startFrame", "endFrame"), ("firstFrame", "lastFrame"), ("inFrame", "outFrame")]:
                try:
                    shot.Set(a_s, start)
                    shot.Set(a_e, end)
                    break
                except Exception:
                    pass
            for attr in ["matchFrameNumbers", "matchFrameNum", "useFrameNumbers"]:
                try:
                    shot.Set(attr, 1)
                    break
                except Exception:
                    pass
            hlev.Accept("Set frame range")
        except Exception as e:
            try:
                hlev.Cancel()
            except Exception:
                pass
            self.log(f"   WARNING: frame range skipped: {e}")

    # ------ rooms / features ------

    def _switch_room(self, *names):
        for name in names:
            try:
                self.hlev.SetRoom(name)
                time.sleep(1)
                self.log(f"   Room -> '{name}'")
                return
            except Exception:
                pass
        self.log("   WARNING: Could not switch room")

    def _configure_features(self, count, threshold, separation):
        """Write the Features-room spinners and VERIFY the writes landed.

        This used to skip a missing spinner silently, so a failed 'Count' write was
        invisible: SynthEyes kept whatever count was left over from a previous session and
        the UI slider looked broken (or the shot tracked far fewer points than asked).
        Read back where the API allows so the log states what SynthEyes actually holds.
        """
        self.log(f"   Features: count={count}, threshold={threshold}, separation={separation}")
        # There are no Count/Threshold/Separation controls in the 2026 Features room --
        # enumerating it (2026-08) turns up only buttons and three Boxers (Camera, motion
        # preset, camera name). ByName('Count') was therefore invalid on every shot and all
        # three values were silently dropped, three WARNING lines at a time.
        #
        # The one that actually governs output is 'Maximum tracker count', which lives in
        # the Advanced Feature Control dialog and defaults to 120 -- exactly the tracker
        # count every export produced, i.e. this cap, not the plate, was the limit.
        self._set_advanced_max_tracks(count)
        # Threshold/separation have no equivalent field on this build (the dialog exposes
        # blip size/density instead, which are not the same quantity). Say so once, at the
        # level of a note, rather than warning per shot about a control that cannot exist.
        if not getattr(self, "_feat_note_logged", False):
            self._feat_note_logged = True
            self.log(f"   note: threshold={threshold} / separation={separation} are not exposed "
                     f"as controls on this SynthEyes build; leaving its own feature settings "
                     f"alone. Only the tracker count is applied.")

    # ----- Advanced dialog (known control IDs, Win32 input) -----
    ADV_BUTTON_ID = 1238
    ADV_MAX_TRACKS_ID = 1266

    def _send_input(self, vk, scan, flags):
        inp = _INPUT(type=1)
        inp.ki = _KEYBDINPUT(vk, scan, flags, 0, None)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        time.sleep(0.015)

    def _send_vk(self, vk, up=False):
        self._send_input(vk, 0, 0x0002 if up else 0)

    def _send_unicode(self, ch):
        for flags in (0x0004, 0x0004 | 0x0002):      # KEYEVENTF_UNICODE [| KEYUP]
            self._send_input(0, ord(ch), flags)

    def _adv_spinner_text(self, hwnd):
        """Read a SynthEyes Spinner. GetWindowTextW returns '' across processes for these
        custom controls; WM_GETTEXT does answer. Note it only reflects a new value AFTER
        the edit is committed with Enter, so read back at the end, not while typing."""
        buf = ctypes.create_unicode_buffer(64)
        ctypes.windll.user32.SendMessageW(ctypes.c_void_p(int(hwnd)), 0x000D, 64, buf)
        return buf.value.strip()

    def _find_butt(self, variants):
        """(hwnd, rect) of a Features-panel button by its text, or (None, None)."""
        norm = lambda s: "".join((s or "").lower().split())
        wants = {norm(v) for v in variants}
        for w in self._syntheyes_top_windows():
            for hwnd, text, rect in self._enum_butt_buttons(w):
                if norm(text) in wants:
                    return hwnd, rect
        return None, None

    def _real_click(self, rect):
        """Physically click the centre of a screen rect and put the pointer back.

        PostMessage is enough for blip/peel, but NOT for 'Advanced': it reports success and
        the dialog never opens, because a posted click on a button that is not genuinely
        foreground is simply dropped. A real click is the same mechanism that turned out to
        be the only thing the Advanced spinner accepts.
        """
        user32 = ctypes.windll.user32
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        try:
            cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
            user32.SetCursorPos(cx, cy)
            time.sleep(0.15)
            user32.mouse_event(0x0002, 0, 0, 0, 0)   # LEFTDOWN
            time.sleep(0.05)
            user32.mouse_event(0x0004, 0, 0, 0, 0)   # LEFTUP
            time.sleep(0.25)
        finally:
            user32.SetCursorPos(pt.x, pt.y)

    def _open_advanced_dialog(self, attempts=3):
        """Click 'Advanced' and return the dialog HWND, or 0.

        Uses a REAL click, not PostMessage. The posted click logs 'OK Win32 click' and then
        no dialog appears -- the button needs genuine foreground input, unlike blip/peel.
        """
        user32 = ctypes.windll.user32
        for attempt in range(1, attempts + 1):
            dlg = user32.FindWindowW(None, "Advanced Feature Control")
            if dlg:
                return dlg
            self._ensure_features_room()
            self._bring_foreground()
            time.sleep(0.4)
            hwnd, rect = self._find_butt(["Advanced"])
            if not hwnd:
                self.log("   WARNING: 'Advanced' button not found on the Features panel")
                return 0
            self._real_click(rect)
            for _ in range(20):
                time.sleep(0.4)
                dlg = user32.FindWindowW(None, "Advanced Feature Control")
                if dlg:
                    return dlg
            if attempt < attempts:
                self.log(f"   Advanced dialog did not open - retrying ({attempt + 1}/{attempts})")
        return 0

    def _type_into_spinner(self, dlg, spin, text):
        """Click the field, select all, type, commit. Returns what it reads back.

        AttachThreadInput first: without sharing the input queue, SetForegroundWindow is
        refused by the foreground lock, SetFocus does nothing, and the injected digits go
        to whatever really had focus -- which is how this committed an EMPTY value.
        """
        user32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
        our = k32.GetCurrentThreadId()
        se_tid = user32.GetWindowThreadProcessId(ctypes.c_void_p(dlg), None)
        attached = bool(user32.AttachThreadInput(our, se_tid, True)) if our != se_tid else False
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))   # this steals the pointer; put it back after
        try:
            user32.ShowWindow(ctypes.c_void_p(dlg), 5)
            user32.SetForegroundWindow(ctypes.c_void_p(dlg))
            time.sleep(0.35)
            user32.SetFocus(ctypes.c_void_p(spin))
            time.sleep(0.15)

            r = wintypes.RECT()
            user32.GetWindowRect(ctypes.c_void_p(spin), ctypes.byref(r))
            cx, cy = (r.left + r.right) // 2, (r.top + r.bottom) // 2
            user32.SetCursorPos(cx, cy)
            time.sleep(0.2)
            for _ in range(2):                      # double-click = enter edit mode
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                time.sleep(0.12)
            time.sleep(0.35)

            self._send_vk(0x11)                                  # ctrl down
            self._send_vk(0x41); self._send_vk(0x41, up=True)    # A
            self._send_vk(0x11, up=True)                         # ctrl up
            time.sleep(0.15)
            for ch in text:
                self._send_unicode(ch)
            time.sleep(0.25)
            self._send_vk(0x0D); self._send_vk(0x0D, up=True)    # Enter commits
            time.sleep(0.6)
            return self._adv_spinner_text(spin)
        finally:
            user32.SetCursorPos(pt.x, pt.y)
            if attached:
                user32.AttachThreadInput(our, se_tid, False)

    def _set_advanced_max_tracks(self, max_tracks, attempts=3):
        """Set 'Maximum tracker count' in the Advanced Feature Control dialog.

        This is the setting that caps how many trackers a shot can produce. It defaults to
        120, and 120 is exactly what every export returned regardless of the count asked
        for, because nothing was writing it: the Features room has no such control, so the
        old ByName('Count') lookup was invalid on every shot.

        Everything cheaper was tried and does not work on build 2026.2.4679 (2026-08):
        SyPy3 cannot see this dialog (Popup() is invalid), WM_SETTEXT returns 1 and changes
        nothing, posted WM_CHAR is ignored, and keybd_event reached the field but committed
        an EMPTY value. Injected input against an attached input queue is what works.

        Verified by read-back, retried, and -- crucially -- restored if it cannot be set:
        leaving the cap blank would be worse than leaving it alone.
        """
        user32 = ctypes.windll.user32
        target = str(int(max_tracks))

        dlg = self._open_advanced_dialog()
        if not dlg:
            self.log("   WARNING: Advanced Feature Control dialog did not open - tracker "
                     "count left at its current value")
            return False

        spin = user32.GetDlgItem(ctypes.c_void_p(dlg), self.ADV_MAX_TRACKS_ID)
        if not spin:
            self.log(f"   WARNING: spinner {self.ADV_MAX_TRACKS_ID} (Maximum tracker count) "
                     f"not in the dialog - tracker count left as-is")
            user32.PostMessageW(ctypes.c_void_p(dlg), 0x0010, 0, 0)
            return False

        original = self._adv_spinner_text(spin)
        ok = False
        try:
            if original == target:
                self.log(f"   OK  Maximum tracker count already {target}")
                return True
            got = ""
            for attempt in range(1, attempts + 1):
                got = self._type_into_spinner(dlg, spin, target)
                if got == target:
                    self.log(f"   OK  Maximum tracker count {original or '?'} -> {got}")
                    ok = True
                    break
                if attempt < attempts:
                    self.log(f"   Maximum tracker count read back {got!r}, wanted {target} "
                             f"- retrying ({attempt + 1}/{attempts})")
            if not ok:
                self.log(f"   WARNING: could not set Maximum tracker count to {target} "
                         f"(reads {got!r}); SynthEyes will cap this shot at its own value")
                # Never leave the field blank/garbage -- put back what was there.
                if original and got != original:
                    back = self._type_into_spinner(dlg, spin, original)
                    self.log(f"   restored Maximum tracker count to {back!r}")
        finally:
            if user32.IsWindow(ctypes.c_void_p(dlg)):
                user32.PostMessageW(ctypes.c_void_p(dlg), 0x0010, 0, 0)   # WM_CLOSE
                time.sleep(0.3)
        return ok


    def _click_and_wait(self, action_name, label, timeout=300):
        # SynthEyes 2026: two-part completion detection.
        #  1) Dispatch the action with the server-side "AndWait" API. This waits
        #     out / dismisses any confirm dialog on the SynthEyes side (the old
        #     ClickAndContinue() + python `Popup().IsValid()` poll never cleared on
        #     2026 -> false `timed out after 400s`). AndWait is focus-independent
        #     (no foreground window) -> headless-safe.
        #  2) Blip/Peel then run an ASYNC frame-by-frame PLAYBACK that AndWait does
        #     NOT block on (it returns as soon as playback is kicked off). So poll
        #     IsPlaying() for the real end -> without this, Peel fired before any
        #     blips existed and only ~1 tracker survived. This also restores the
        #     `timeout` + mid-op STOP that bare AndWait dropped.
        # 2026: use the documented SyPyManual idiom -- resolve the action id, then
        # click the panel BUTTON with ClickAndWait() (CLICKBUTTON1), which blocks
        # server-side until the op completes and waits out its progress/dialogs.
        #   Manual example: bid = hlev.ActionID("Solve/Clear")
        #                   hlev.Main().ByID(bid).ClickAndWait()
        # (ClickAndContinue+Popup poll false-hung 400s; PerformActionByNameAndWait
        #  no-op'd instantly -> ~1 tracker. ClickAndWait is the correct variant.)
        hlev = self.hlev
        idno = hlev.ActionID(action_name)
        if idno <= 0:
            self.log(f"   ERROR: Action '{action_name}' not found")
            return False
        try:
            hlev.Main().ByID(idno).ClickAndWait()
            self.log(f"   OK  {label} done")
            return True
        except Exception as e:
            self.log(f"   ERROR: {label}: {e}")
            return False

    # ------ export ------

    def _run_sizzle(self, script_text, watchdog_secs=90):
        """Run a //SIZZLET script server-side via RunScriptFile. A watchdog thread dismisses
        the 'Sizzle Scripting' error modal (#32770) that would otherwise hang RunScriptFile's
        Sync on any script error. Returns the .szl path written."""
        import threading
        tmp = os.environ.get("TEMP") or os.path.dirname(os.path.abspath(__file__))
        szl = os.path.join(tmp, "btr_export.szl")
        with open(szl, "w", encoding="utf-8") as f:
            f.write(script_text)
        state = {"done": False}

        def _watch():
            user32 = ctypes.windll.user32
            t0 = time.time()
            while not state["done"] and time.time() - t0 < watchdog_secs:
                h = user32.FindWindowW(None, "Sizzle Scripting")
                if h:
                    # read the error message (static-text children) before dismissing
                    msgs = []
                    EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

                    def _cb(hh, _l):
                        t = ctypes.create_unicode_buffer(512)
                        user32.GetWindowTextW(hh, t, 512)
                        if t.value and t.value not in ("OK", "Cancel"):
                            msgs.append(t.value)
                        return True

                    user32.EnumChildWindows(h, EnumProc(_cb), 0)
                    if msgs:
                        self.log(f"   Sizzle error: {' | '.join(msgs)}")
                    ok = user32.GetDlgItem(h, 1)  # IDOK
                    if ok:
                        user32.PostMessageW(ok, _WM_LBUTTONDOWN, _MK_LBUTTON, 0)
                        user32.PostMessageW(ok, _WM_LBUTTONUP, 0, 0)
                    else:
                        user32.PostMessageW(h, 0x0010, 0, 0)  # WM_CLOSE
                    self.log("   (dismissed a Sizzle error modal)")
                    time.sleep(0.5)
                time.sleep(0.3)

        wd = threading.Thread(target=_watch, daemon=True)
        wd.start()
        try:
            self.hlev.RunScriptFile(szl.replace("\\", "/"))
        except Exception as e:
            self.log(f"   WARNING: RunScriptFile: {e}")
        finally:
            state["done"] = True
        return szl

    def _apply_sam3_postfilter(self, txt_path, mask_dir, min_len=5):
        """Gate the exported 3DE tracks against the SAM3 masks in Python (yesterday's proven
        method - more robust than SynthEyes roto matte). For each track point (frame,x,y),
        sample the mask at (x,y): white(>=127)=background=KEEP, black=mover=DROP. Coords index
        the mask directly (px=0.5*(u+1)*w, py=0.5*(1-v)*h, NOT Y-flipped). Rewrites txt_path in
        place. Returns (kept_tracks, kept_points)."""
        import glob
        import cv2  # type: ignore
        import numpy as np
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        if not masks:
            self.log(f"   SAM3 post-filter: no masks in {mask_dir} - leaving tracks ungated")
            return None
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                toks = f.read().split("\n")
        except Exception as e:
            self.log(f"   SAM3 post-filter: cannot read export: {e}")
            return None

        # parse classic 3DE: <N> then per track: name / color / count / count*(frame x y)
        it = iter(toks)

        def nxt():
            for t in it:
                if t.strip() != "":
                    return t.strip()
            return None

        try:
            ntr = int(nxt())
        except Exception:
            self.log("   SAM3 post-filter: bad export header")
            return None

        mask_cache = {}
        # Frame-alignment counters. keep_point maps export frame N -> masks[N-1], so a plate
        # numbered 1001.. against 120 masks lands every point out of range and the gating
        # silently does NOTHING. Counting both makes "mask did nothing" and "mask dropped
        # everything" distinguishable in the log instead of looking alike.
        span = {"matched": 0, "out_of_range": 0}
        squeezed = []   # marker so the shape-correction is reported once, not per mask

        def keep_point(frame, x, y):
            # export frame is 1-based; masks are a sorted 1-based sequence
            idx = int(frame) - 1
            if idx < 0 or idx >= len(masks):
                span["out_of_range"] += 1
                return True  # no mask for this frame -> don't drop
            span["matched"] += 1
            if idx in mask_cache:
                m = mask_cache[idx]
            else:
                m = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)
                # IMREAD_GRAYSCALE is *supposed* to return 2-D, and these files read back
                # 2-D standalone -- but in a live run (SAM3 + torch already loaded in this
                # process) one came back with a channel axis, and sampling it raised
                # "only 0-dimensional arrays can be converted to Python scalars" AFTER a
                # successful 120-tracker export, failing the shot to the TAPNext retry.
                # Don't trust the flag: collapse to one channel here, once per frame.
                if m is not None and getattr(m, "ndim", 2) > 2:
                    if not squeezed:
                        # Once per run, not once per frame: it is the same cause for every
                        # mask in the folder, and a 40-line burst per task buries the log.
                        self.log(f"   SAM3 post-filter: masks read back with shape {m.shape} "
                                 f"(not 2-D); using their first channel")
                        squeezed.append(1)
                    m = m[:, :, 0]
                mask_cache[idx] = m
            if m is None:
                return True
            h, w = m.shape[:2]
            xi = min(max(int(round(x)), 0), w - 1)
            yi = min(max(int(round(y)), 0), h - 1)
            return int(m[yi, xi]) >= 127  # white = background = keep

        out_tracks = []
        for _ in range(ntr):
            name = nxt()
            color = nxt()
            cnt = int(nxt())
            pts = []
            for _p in range(cnt):
                parts = nxt().split()
                fr, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                pts.append((fr, x, y))
            # SAM3 gating: keep only background points. (The Demo build's "frozen tail"
            # truncation was removed for the Pro license: Pro produces real full-length
            # tracks with no held-coord padding, and a genuinely static feature -- a
            # locked-off plate, a bolt on a wall -- legitimately holds identical coords,
            # so truncating on the first repeated coord would chop real tracks.)
            gated = [(fr, x, y) for (fr, x, y) in pts if keep_point(fr, x, y)]
            if len(gated) >= min_len:
                out_tracks.append((name, color, gated))

        if span["out_of_range"]:
            self.log(f"   SAM3 post-filter: {span['matched']} point(s) had a matching mask "
                     f"frame, {span['out_of_range']} fell outside the {len(masks)} mask(s) "
                     f"and were kept ungated (export frame N maps to mask N-1 - a plate that "
                     f"starts at 1001 will not line up)")

        # Gating wiped the shot -> FAIL it and keep nothing, rather than overwrite the good
        # ungated export with an empty file (which then published over a previous good one).
        if not out_tracks:
            self.log(f"   SAM3 post-filter: ALL {ntr} track(s) were gated away - failing the "
                     f"shot and removing the export (masks may be inverted or misaligned)")
            try:
                os.remove(txt_path)
            except Exception:
                pass
            return 0, 0

        # rewrite in place (only now that we know there is something worth writing)
        lines = [str(len(out_tracks))]
        for name, color, gated in out_tracks:
            lines.append(name)
            lines.append(color)
            lines.append(str(len(gated)))
            for fr, x, y in gated:
                lines.append(f"{fr} {x:.4f} {y:.4f}")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        kept_pts = sum(len(g) for _n, _c, g in out_tracks)
        self.log(f"   SAM3 post-filter: {ntr} -> {len(out_tracks)} tracks, {kept_pts} points kept")
        return len(out_tracks), kept_pts

    def _resync_socket(self):
        """Recreate the SyPy3 socket (fresh SyLevel + OpenExisting) to the SAME running
        SynthEyes. The Win32 room nav + spinner writes leave the 2026.2.4679 socket desynced
        so RunScriptFile no-ops; a fresh socket resets the protocol state while the scene +
        peeled trackers stay intact."""
        try:
            from SyPy3.sylevel import SyLevel
            port = self._s("port")
            pin = self._s("pin")
            fresh = SyLevel()
            if fresh.OpenExisting(port, pin):
                self.hlev = fresh
                self.log("   OK  SyPy3 socket resynced for export")
                return True
        except Exception as e:
            self.log(f"   WARNING: socket resync failed: {e}")
        return False

    def _sizzle_export_3de(self, out_txt_path):
        """Write classic 3DE 2D-track ASCII directly from SynthEyes via Sizzle (openout +
        printf), bypassing the broken SyPy3 export. Pixel: px=w*0.5*(u+1), py=h*0.5*(1-v).
        Returns the real tracker count (parsed from the file), or -1 on failure."""
        try:
            if os.path.isfile(out_txt_path):
                os.remove(out_txt_path)
        except Exception:
            pass
        if not self._resync_socket():
            # A desynced socket makes RunScriptFile a silent no-op, which reads downstream as
            # "0 tracks". Say so here so the log names the real culprit.
            self.log("   WARNING: socket resync failed - the Sizzle export may silently no-op")
        p = out_txt_path.replace("\\", "/")
        # Raw vs exported tracker counts, written to a sidecar the caller reads and deletes.
        # Without this, 0 tracks is ambiguous: 'blip/peel never produced anything' and
        # 'trackers exist but none are flagged exportable' look identical. #obj.trk is the
        # shipped idiom (see SynthEyes' own scripts/Trackers/*.szl).
        diag = p + ".diag"
        # Sizzle syntax (from SynthEyes' own tdexport.szl / trkpath.szl): newline-terminated
        # statements, for(...)...end / if(...)...end (NO semicolons, NO braces). openout()
        # redirects printf to the file; pixel = 0.5*(u+1)*width, 0.5*(1-v)*height. Classic 3DE
        # 2D-track ASCII: <N> then per track name / color 0 / point-count / "frame x y" lines.
        script = (
            "//SIZZLET BTRExport\n"   # tool-script header REQUIRED - without it RunScriptFile silently no-ops
            "obj = Scene.activeObj\n"
            "shot = obj.shot\n"
            "start = shot.start\n"
            "stop = shot.stop\n"
            "width = shot.width\n"
            "height = shot.height\n"
            f'openout("{p}")\n'
            "nt = 0\n"
            "for (tk in obj.trk)\n"
            "    if (tk.isExported)\n"
            "        nt = nt + 1\n"
            "    end\n"
            "end\n"
            'printf("%d\\n", nt)\n'
            "for (tk in obj.trk)\n"
            "    if (tk.isExported)\n"
            "        cnt = 0\n"
            "        for (frame = start; frame <= stop; frame++)\n"
            "            if (tk.valid)\n"
            "                cnt = cnt + 1\n"
            "            end\n"
            "        end\n"
            '        printf("%s\\n", tk.nm)\n'
            '        printf("0\\n")\n'
            '        printf("%d\\n", cnt)\n'
            "        for (frame = start; frame <= stop; frame++)\n"
            "            if (tk.valid)\n"
            '                printf("%d %.4f %.4f\\n", frame+1, 0.5*(tk.u+1)*width, 0.5*(1-tk.v)*height)\n'
            "            end\n"
            "        end\n"
            "    end\n"
            "end\n"
            "closeout()\n"
            f'openout("{diag}")\n'
            'printf("%d %d\\n", #obj.trk, nt)\n'
            "closeout()\n"
        )
        self._run_sizzle(script)
        self._log_export_diag(diag)
        if not os.path.isfile(out_txt_path):
            self.log("   ERROR: Sizzle export produced no file")
            return -1
        try:
            with open(out_txt_path, "r", encoding="utf-8", errors="ignore") as f:
                first = f.readline().strip()
            count = int(first)
            self.log(f"   OK  Sizzle export -> {os.path.basename(out_txt_path)} ({count} trackers)")
            return count
        except Exception as e:
            self.log(f"   WARNING: could not parse Sizzle export header: {e}")
            return -1

    def _log_export_diag(self, diag_path):
        """Read + delete the raw/exported sidecar and say what it means. Best-effort: if the
        second openout didn't work on this build, stay quiet rather than invent a diagnosis."""
        try:
            if not os.path.isfile(diag_path):
                return
            with open(diag_path, "r", encoding="utf-8", errors="ignore") as f:
                raw, exp = (int(x) for x in f.readline().split()[:2])
            self.log(f"   trackers: raw={raw} exported={exp}")
            if raw == 0:
                self.log("   -> SynthEyes holds NO trackers: blip/peel produced nothing "
                         "(panel automation failed), not a tracking-quality problem.")
            elif exp == 0:
                self.log(f"   -> {raw} tracker(s) exist but none are flagged exportable: "
                         "tracking quality / too few valid frames, not an automation failure.")
        except Exception as e:
            self.log(f"   (export diagnostic unavailable: {e})")
        finally:
            try:
                os.remove(diag_path)
            except Exception:
                pass

    def _export_tracks(self, output_dir, filename):
        out = os.path.normpath(output_dir)
        full_path = os.path.join(out, filename)
        hlev = self.hlev
        self.log(f"   Export -> {full_path}")

        try:
            hlev.BeginPref()
            hlev.SetFolderPref("EXPORT", out)
            hlev.AcceptPref()
        except Exception as e:
            self.log(f"   WARNING: folder pref: {e}")
            try:
                hlev.AcceptPref()
            except Exception:
                pass

        hlev.InitMenu()
        hlev.ClickMainMenuAndContinue("All Tracker Paths")
        time.sleep(2)

        try:
            popup = hlev.Popup()
            if not popup.IsValid():
                self.log("   ERROR: Save dialog did not appear")
                return False
        except Exception as e:
            self.log(f"   ERROR: popup: {e}")
            return False

        try:
            popup.ByID(1001).SetEditValue(filename)
            time.sleep(0.3)
        except Exception as e:
            self.log(f"   ERROR: filename field: {e}")
            try:
                popup.CloseAndWait()
            except Exception:
                pass
            return False

        try:
            popup.ByID(1).ClickAndWait()
            time.sleep(1)
        except Exception as e:
            self.log(f"   ERROR: Save button: {e}")
            return False

        if os.path.exists(full_path):
            self.log(f"   OK  Exported -> {full_path}")
            return True
        self.log("   WARNING: file not found after save")
        return False

    # ------ 3DE sidecar + auto-import ------

    def _write_3de_sidecar(self, output_dir, shot_name, txt_filename,
                           first_frame, frame_count, start_frame, end_frame,
                           image_width, image_height):
        if not image_width or not image_height:
            image_width = image_width or 1920
            image_height = image_height or 1080
            self.log(f"   WARNING: image size unknown, defaulting {image_width}x{image_height}")

        plate_folder = os.path.dirname(first_frame)
        sidecar = {
            "shot_name":        shot_name,
            "txt_file":         txt_filename,
            "plate_path":       plate_folder.replace("\\", "/"),
            "first_frame_file": os.path.basename(first_frame),
            "frame_start":      start_frame,
            "frame_end":        end_frame,
            "frame_count":      frame_count,
            "image_width":      int(image_width),
            "image_height":     int(image_height),
            "sensor_width":     float(self._s("sensor_width")),
            "sensor_height":    float(self._s("sensor_height")),
            "focal_length":     float(self._s("focal_length")),
            "fps":              float(self._s("fps")),
        }
        json_path = os.path.join(output_dir, os.path.splitext(txt_filename)[0] + "_3de_sidecar.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2)
            self.log(f"   OK  3DE sidecar -> {os.path.basename(json_path)}")
            return json_path
        except Exception as e:
            self.log(f"   WARNING: Could not write 3DE sidecar: {e}")
            return None

    def _run_3de_import(self, json_path, output_dir, shot_name, out_txt_name):
        tde4_exe = self._s("tde4_exe")
        if not tde4_exe or not os.path.isfile(tde4_exe):
            self.log(f"   SKIP 3DE import: exe not found at {tde4_exe}")
            return

        tde4_bin_dir = os.path.dirname(tde4_exe)
        tde4_root = os.path.dirname(tde4_bin_dir)
        tde4_scripts_dir = os.path.join(tde4_root, "sys_data", "py_scripts")
        if not os.path.isdir(tde4_scripts_dir):
            self.log(f"   SKIP 3DE import: scripts folder not found at {tde4_scripts_dir}")
            return

        tde_filename = os.path.splitext(out_txt_name)[0] + ".3de"
        tde_path = os.path.join(output_dir, tde_filename).replace("\\", "/")
        json_path_safe = json_path.replace("\\", "/")

        temp_script_name = f"_temp_auto_import_{shot_name}.py"
        temp_script_path = os.path.join(tde4_scripts_dir, temp_script_name)
        status_file = os.path.join(output_dir, f"_3de_import_status_{shot_name}.txt")
        status_file_safe = status_file.replace("\\", "/")

        script_content = f'''#
# 3DE4.script.name:    _temp_auto_import
# 3DE4.script.startup: true
# 3DE4.script.hide:    true
#

import tde4
import os
import json

def run_import():
    status_path = r"{status_file_safe}"
    json_path = r"{json_path_safe}"

    try:
        with open(json_path, "r") as f:
            info = json.load(f)
    except Exception as e:
        with open(status_path, "w") as sf:
            sf.write("ERROR: Could not read JSON: " + str(e))
        import sys; sys.exit(1)
        return

    shot_name      = info.get("shot_name", "Unknown")
    txt_file       = info.get("txt_file", "")
    plate_path     = info.get("plate_path", "")
    first_frame    = info.get("first_frame_file", "")
    frame_start    = info.get("frame_start", 1)
    frame_end      = info.get("frame_end", 100)
    image_width    = info.get("image_width", 1920)
    image_height   = info.get("image_height", 1080)
    sensor_width   = info.get("sensor_width", 36.0)
    sensor_height  = info.get("sensor_height", 24.0)
    focal_length   = info.get("focal_length", 35.0)
    fps            = info.get("fps", 24.0)

    json_dir = os.path.dirname(json_path)
    if txt_file:
        txt_path = os.path.join(json_dir, txt_file)
    else:
        txt_path = json_path.replace("_3de_sidecar.json", ".txt")

    if not os.path.isfile(txt_path):
        with open(status_path, "w") as sf:
            sf.write("ERROR: .txt not found: " + txt_path)
        import sys; sys.exit(1)
        return

    first_frame_full = os.path.join(plate_path, first_frame)
    plate_exists = os.path.isfile(first_frame_full)

    import re as _re
    seq_pattern = ""
    if plate_exists:
        fname = first_frame
        match = _re.search(r'(\\d+)(\\.[^.]+)$', fname)
        if match:
            num_str = match.group(1)
            ext = match.group(2)
            prefix = fname[:match.start(1)]
            hashes = "#" * len(num_str)
            seq_pattern = os.path.join(plate_path, prefix + hashes + ext)
        else:
            seq_pattern = first_frame_full

    tde4.newProject()
    for cam_id in tde4.getCameraList(0):
        tde4.deleteCamera(cam_id)
    for lens_id in tde4.getLensList(0):
        tde4.deleteLens(lens_id)
    for pg_id in tde4.getPGroupList(0):
        tde4.deletePGroup(pg_id)

    cam = tde4.createCamera("SEQUENCE")
    tde4.setCameraName(cam, shot_name)
    if seq_pattern:
        tde4.setCameraPath(cam, seq_pattern)

    frame_count = frame_end - frame_start + 1
    tde4.setCameraSequenceAttr(cam, frame_start, frame_end, 1)
    tde4.setCameraFrameOffset(cam, frame_start)
    tde4.setCameraImageWidth(cam, image_width)
    tde4.setCameraImageHeight(cam, image_height)
    tde4.setCameraFPS(cam, fps)

    lens = tde4.createLens()
    tde4.setLensName(lens, shot_name + "_lens")
    tde4.setLensFBackWidth(lens, sensor_width)
    tde4.setLensFBackHeight(lens, sensor_height)
    tde4.setLensFocalLength(lens, focal_length)
    tde4.setCameraLens(cam, lens)

    pg = tde4.createPGroup("CAMERA")
    tde4.setPGroupName(pg, "SynthEyes_Tracks")

    tracks = {{}}
    current_tracker = None
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) == 1:
                current_tracker = parts[0]
                tracks[current_tracker] = []
            elif len(parts) >= 3 and current_tracker:
                frame = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                tracks[current_tracker].append((frame, x, y))

    w = float(image_width)
    h = float(image_height)
    count = 0
    skipped = 0
    for trk_name, pts in tracks.items():
        if not pts: continue
        pts_sorted = sorted(pts, key=lambda p: p[0])
        first_f_internal = pts_sorted[0][0] + 1
        last_f_internal  = pts_sorted[-1][0] + 1
        if last_f_internal > frame_count or first_f_internal < 1:
            skipped += 1
            continue
        try:
            point = tde4.createPoint(pg)
            tde4.setPointName(pg, point, trk_name)
            curve = []
            for pt in pts_sorted:
                curve.append([pt[1] / w, pt[2] / h])
            tde4.setPointPosition2DBlock(pg, point, cam, first_f_internal, curve)
            tde4.setPointStatus2D(pg, point, cam, last_f_internal, "POINT_KEYFRAME_END")
            count += 1
        except:
            skipped += 1

    tde4.saveProject(r"{tde_path}")

    import sys
    with open(status_path, "w") as sf:
        msg = "OK: " + str(count) + " trackers saved"
        if skipped > 0: msg += " (" + str(skipped) + " skipped)"
        sf.write(msg)
    sys.exit(0)

run_import()
'''

        try:
            with open(temp_script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
        except Exception as e:
            self.log(f"   ERROR: Could not write startup script: {e}")
            return

        self.log(f"-> Launching 3DE for auto-import ({shot_name})...")
        tde_process = None
        try:
            tde_process = subprocess.Popen([tde4_exe])
            max_wait = 300
            poll_interval = 2
            elapsed = 0
            while elapsed < max_wait:
                if tde_process.poll() is not None:
                    break
                if os.path.isfile(status_file):
                    self.log("   OK  Import completed - closing 3DE...")
                    time.sleep(1)
                    try:
                        tde_process.terminate()
                        tde_process.wait(timeout=10)
                    except Exception:
                        subprocess.run(["taskkill", "/F", "/IM", "3DE4.exe"], capture_output=True)
                    break
                time.sleep(poll_interval)
                elapsed += poll_interval
            else:
                self.log(f"   WARNING: 3DE timed out after {max_wait}s - killing process")
                try:
                    tde_process.terminate()
                except Exception:
                    pass
        except Exception as e:
            self.log(f"   ERROR: 3DE launch failed: {e}")

        if os.path.isfile(status_file):
            try:
                with open(status_file, "r") as sf:
                    self.log(f"   3DE report: {sf.read().strip()}")
                os.remove(status_file)
            except Exception:
                pass

        if os.path.isfile(os.path.join(output_dir, tde_filename)):
            self.log(f"   OK  3DE project saved -> {tde_filename}")

        try:
            os.remove(temp_script_path)
        except Exception:
            pass


# ============================================================
#  Tracking presets (preset name -> max tracker count)
# ============================================================

TRACKING_PRESETS = {
    "Locked / Tripod":   100,
    "Slow / Dolly":      500,
    "Normal / Handheld": 800,
    "Fast / Action":     2000,
    "Custom":            None,
}
PRESET_NAMES = list(TRACKING_PRESETS.keys())
DEFAULT_PRESET = "Normal / Handheld"


def preset_track_count(preset_name, custom_count=500):
    cnt = TRACKING_PRESETS.get(preset_name)
    return int(custom_count) if cnt is None else int(cnt)
