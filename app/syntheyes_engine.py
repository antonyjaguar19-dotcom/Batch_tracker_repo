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

    def _wait_for_operation(self, label, min_wait=3.0, max_timeout=600.0,
                            peak_cpu=250.0, idle_hold=1.5, start_grace=6.0):
        """Wait for a SynthEyes long-op (blip/peel) to finish via live signals instead of a
        fixed sleep. Returns True when done (or on the timeout ceiling, to never hang the
        batch); False on stop-request or a crash/error dialog. Falls back to a fixed sleep if
        psutil is unavailable."""
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
                return False
            prog, err = self._op_dialog_state()
            if err:
                self.log(f"   {label}: SynthEyes crash/error dialog -> aborting op")
                return False
            if proc is not None:
                try:
                    cpu = proc.cpu_percent(interval=0.3)
                except Exception:
                    self.log(f"   {label}: SynthEyes process gone during wait")
                    return False
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
                    self.log(f"   {label}: complete at ~{el:.1f}s (dialog gone + CPU idle)")
                    return True
        self.log(f"   {label}: hit {max_timeout:.0f}s ceiling -> proceeding")
        return True

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
        self._ensure_features_room()
        self._configure_features(track_count, track_threshold, track_separation)
        # NOTE (2026 debug): _set_advanced_max_tracks does ByID(1238).ClickAndContinue()
        # on a control that no longer exists on 2026 ("Advanced dialog not found").
        # Suspected to DESYNC the SyPy3 socket -> subsequent IsPlaying()/Popup() return
        # bogus True -> Blip All false-hangs 400s with flat CPU. Skipping to test.
        # self._set_advanced_max_tracks(track_count)

        # Blip All + Peel All via Win32 PostMessage. The SyPy3 dispatch (_click_and_wait)
        # silently no-ops on 2026.2.4679, so PostMessage the real "Butt" panel buttons and
        # wait a frame-scaled budget (there is no reliable API to poll blip completion).
        # SyPy3 remains the fallback if the buttons can't be located.
        # Foreground the window so the Features panel paints and its blip/peel child
        # controls instantiate (they don't exist while backgrounded on 2026.2.4679).
        self.log("-> Bringing SynthEyes foreground for panel clicks...")
        self._bring_foreground()

        blip_timeout = max(400, int(frame_count * 0.2))
        self.log("-> Blip All frames (Win32 PostMessage)...")
        if self._win32_click_button("Blip All", ["Blips all frames", "Blip all frames"]):
            self.log(f"   blipping {frame_count} frames... (waiting for completion signal)")
            self._wait_for_operation("Blip All", min_wait=3.0, max_timeout=blip_timeout)
        else:
            self.log("   Win32 blip unavailable - falling back to SyPy3 dispatch")
            if not self._click_and_wait("Feature/Blips all frames", "Blip All", timeout=blip_timeout):
                raise RuntimeError("Blip All failed")

        peel_timeout = max(600, int(frame_count * 1.0))
        self.log("-> Peel All (Win32 PostMessage)...")
        if self._win32_click_button("Peel All", ["Peel all", "Peel All"]):
            self.log("   peeling... (waiting for completion signal)")
            self._wait_for_operation("Peel All", min_wait=3.0, max_timeout=peel_timeout)
        else:
            self.log("   Win32 peel unavailable - falling back to SyPy3 dispatch")
            if not self._click_and_wait("Feature/Peel All", "Peel All", timeout=peel_timeout):
                raise RuntimeError("Peel All failed")

        self.log("-> Clearing blips...")
        if not self._win32_click_button("Clear blips", ["Clear all blips"]):
            self._click_and_wait("Feature/Clear all blips", "Clear blips", timeout=10)

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
        if n_trk > 0 and mask_dir:
            res = self._apply_sam3_postfilter(out_txt_path, mask_dir)
            if res:
                n_trk = res[0]

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
        self.log(f"   Features: count={count}, threshold={threshold}, separation={separation}")
        try:
            ui = self.hlev.Main()
            for name, val in [("Count", count), ("Threshold", threshold), ("Separation", separation)]:
                spinner = ui.ByName(name)
                if spinner.IsValid():
                    spinner.SetSpnValue(val)
        except Exception as e:
            self.log(f"   WARNING: Could not set features: {e}")

    # ----- Advanced dialog (known control IDs, Win32 input) -----
    ADV_BUTTON_ID = 1238
    ADV_MAX_TRACKS_ID = 1266

    def _set_advanced_max_tracks(self, max_tracks):
        hlev = self.hlev
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self.log(f"-> Setting Advanced Max Tracks to: {max_tracks}")

        try:
            hlev.Main().ByID(self.ADV_BUTTON_ID).ClickAndContinue()
            time.sleep(1.5)
        except Exception as e:
            self.log(f"   ERROR: Could not click Advanced button: {e}")
            return False

        dialog_hwnd = 0
        for _ in range(10):
            time.sleep(0.5)
            dialog_hwnd = user32.FindWindowW(None, "Advanced Feature Control")
            if dialog_hwnd:
                break
        if not dialog_hwnd:
            self.log("   WARNING: Advanced dialog not found - skipping")
            return False

        spinner_hwnd = user32.GetDlgItem(dialog_hwnd, self.ADV_MAX_TRACKS_ID)
        if not spinner_hwnd:
            self.log("   WARNING: Spinner 1266 not found")
            return False

        our_tid = kernel32.GetCurrentThreadId()
        se_tid = user32.GetWindowThreadProcessId(dialog_hwnd, None)
        attached = False
        if our_tid != se_tid:
            attached = bool(user32.AttachThreadInput(our_tid, se_tid, True))

        try:
            user32.ShowWindow(dialog_hwnd, 5)
            user32.SetForegroundWindow(dialog_hwnd)
            time.sleep(0.3)
            user32.SetFocus(spinner_hwnd)
            time.sleep(0.2)

            lParam = (10 << 16) | 30
            user32.SendMessageW(spinner_hwnd, 0x0201, 1, lParam)
            user32.SendMessageW(spinner_hwnd, 0x0202, 0, lParam)
            time.sleep(0.2)

            VK_CONTROL = 0x11
            VK_A = 0x41
            VK_RETURN = 0x0D
            KEYEVENTF_KEYUP = 0x0002

            def press_key(vk):
                user32.keybd_event(vk, 0, 0, 0)
                time.sleep(0.02)
                user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
                time.sleep(0.02)

            user32.keybd_event(VK_CONTROL, 0, 0, 0)
            time.sleep(0.02)
            press_key(VK_A)
            user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
            time.sleep(0.1)

            for char in str(int(max_tracks)):
                press_key(ord(char))
                time.sleep(0.05)

            time.sleep(0.1)
            press_key(VK_RETURN)
            time.sleep(0.3)

            ok_btn = user32.GetDlgItem(dialog_hwnd, 1)
            if ok_btn:
                user32.SendMessageW(ok_btn, 0x0201, 1, 0)
                time.sleep(0.05)
                user32.SendMessageW(ok_btn, 0x0202, 0, 0)
                self.log(f"   OK  Set Max Tracks = {max_tracks}")
            else:
                press_key(VK_RETURN)
                self.log(f"   OK  Set Max Tracks = {max_tracks} (Enter to close)")

            time.sleep(0.3)
            return True
        finally:
            if attached:
                user32.AttachThreadInput(our_tid, se_tid, False)

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

        def keep_point(frame, x, y):
            # export frame is 1-based; masks are a sorted 1-based sequence
            idx = int(frame) - 1
            if idx < 0 or idx >= len(masks):
                return True  # no mask for this frame -> don't drop
            m = mask_cache.get(idx)
            if m is None:
                m = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)
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

        # rewrite in place
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
        self._resync_socket()  # clean socket so RunScriptFile actually runs
        p = out_txt_path.replace("\\", "/")
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
        )
        self._run_sizzle(script)
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
