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
import subprocess

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".exr", ".png", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")


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

        # ---- Optional SAM3 matte ----
        if mask_dir:
            self._load_sam3_matte(shot, mask_dir)

        self.log("-> Switching to Features room...")
        self._switch_room("Features", "Feature")
        self._configure_features(track_count, track_threshold, track_separation)
        # NOTE (2026 debug): _set_advanced_max_tracks does ByID(1238).ClickAndContinue()
        # on a control that no longer exists on 2026 ("Advanced dialog not found").
        # Suspected to DESYNC the SyPy3 socket -> subsequent IsPlaying()/Popup() return
        # bogus True -> Blip All false-hangs 400s with flat CPU. Skipping to test.
        # self._set_advanced_max_tracks(track_count)

        blip_timeout = max(400, int(frame_count * 0.2))
        self.log(f"-> Blip All frames... (timeout {blip_timeout}s)")
        if not self._click_and_wait("Feature/Blips all frames", "Blip All", timeout=blip_timeout):
            raise RuntimeError("Blip All failed")

        peel_timeout = max(600, int(frame_count * 1.0))
        self.log(f"-> Peel All ({frame_count} frames, timeout {peel_timeout}s)...")
        if not self._click_and_wait("Feature/Peel All", "Peel All", timeout=peel_timeout):
            raise RuntimeError("Peel All failed")

        self.log("-> Clearing blips...")
        self._click_and_wait("Feature/Clear all blips", "Clear blips", timeout=10)

        # 2026: there is no "Trackers" room -- trackers live on the Summary panel.
        # Feeding SetRoom() an invalid name DESYNCS the SyPy3 socket (all later
        # Run() return bogus True), which corrupted the export step. Use "Summary".
        self.log("-> Switching to Summary room...")
        self._switch_room("Summary")

        tracker_list = hlev.Trackers()
        self.log(f"   Trackers found: {len(tracker_list)}")

        out_txt_path = os.path.join(output_dir, out_txt_name)
        if len(tracker_list) > 0:
            self.log("-> Exporting 2D tracks...")
            self._export_tracks(output_dir, out_txt_name)

            # 3DE sidecar + optional auto-import
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
            self.log("   WARNING: 0 trackers - skipping export")

        return len(tracker_list), out_txt_path

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
