"""
==============================================================
  SynthEyes Batch Auto-Tracker  —  Shot Selector UI  v4.5
  Python 3.9 | Tkinter (zero dependencies)

  CHANGELOG v4.5:
   - FIXED: Added AttachThreadInput() before SetFocus/keybd_event.
     This was the missing piece — without it, Windows silently
     blocks cross-thread focus changes, so hardware keystrokes
     were landing on the wrong window.
   - FIXED: popup.GetText() -> popup.Name() (SyPy has no GetText)
   - Retained all v4.4 hardware keystroke logic (keybd_event).
==============================================================
"""

import os
import sys
import re
import csv
import time
import glob
import subprocess
import threading
import traceback
import ctypes
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ============================================================
#  TRACKING PRESETS
# ============================================================

TRACKING_PRESETS = {
    "Locked / Tripod": {
        "description": "Static camera, minimal movement",
        "count": 300,
        "threshold": 0.015,
        "separation": 6,
    },
    "Slow / Dolly": {
        "description": "Slow smooth camera moves, dolly, crane",
        "count": 600,
        "threshold": 0.015,
        "separation": 6,
    },
    "Normal / Handheld": {
        "description": "Standard handheld or steadicam",
        "count": 500,
        "threshold": 0.020,
        "separation": 8,
    },
    "Fast / Action": {
        "description": "Fast pans, whips, running, shaky",
        "count": 800,
        "threshold": 0.035,
        "separation": 12,
    },
    "Custom": {
        "description": "Use manual Count / Threshold / Separation below",
        "count": None,
        "threshold": None,
        "separation": None,
    },
}

PRESET_NAMES = list(TRACKING_PRESETS.keys())
DEFAULT_PRESET = "Normal / Handheld"


# ============================================================
#  DEFAULT SETTINGS
# ============================================================

DEFAULTS = {
    "syntheyes_exe":  r"C:\Program Files\BorisFX\SynthEyes 2025.5\SynthEyes64.exe",
    "port":           2222,
    "pin":            "listen",
    "tracker_count":  500,
    "threshold":      0.02,
    "separation":     8,
    "startup_wait":   10,
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".exr")

SYPY3_PATH    = r"C:\Python39\Lib\site-packages\SyPy3"
SITE_PACKAGES = r"C:\Python39\Lib\site-packages"


# ============================================================
#  OUTPUT PATH HELPERS
# ============================================================

def get_shot_output_dir(shot_path):
    out_dir = os.path.normpath(os.path.join(shot_path, "mid", "cmm", "bot_MMpoints"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_next_version_number(output_dir, shot_name):
    pattern = re.compile(
        re.escape(shot_name) + r"_v(\d{3})\.(txt|ifl)$",
        re.IGNORECASE
    )
    max_ver = 0
    if os.path.isdir(output_dir):
        for fname in os.listdir(output_dir):
            m = pattern.match(fname)
            if m:
                ver_num = int(m.group(1))
                if ver_num > max_ver:
                    max_ver = ver_num
    return max_ver + 1


def make_versioned_filename(shot_name, version_num, ext):
    return f"{shot_name}_v{version_num:03d}{ext}"


# ============================================================
#  FOLDER SCANNER
# ============================================================

def scan_show_folder(show_path):
    results = []
    if not os.path.isdir(show_path):
        return results

    for shot_name in sorted(os.listdir(show_path)):
        shot_dir = os.path.join(show_path, shot_name)
        if not os.path.isdir(shot_dir):
            continue
        plates_dir = os.path.join(shot_dir, "in", "plates")
        if not os.path.isdir(plates_dir):
            continue

        versions = {}
        for ver_name in sorted(os.listdir(plates_dir)):
            ver_dir = os.path.join(plates_dir, ver_name)
            if not os.path.isdir(ver_dir):
                continue
            for plate_name in os.listdir(ver_dir):
                plate_dir = os.path.join(ver_dir, plate_name)
                if not os.path.isdir(plate_dir):
                    continue
                for res_folder in os.listdir(plate_dir):
                    res_dir = os.path.join(plate_dir, res_folder)
                    if not os.path.isdir(res_dir):
                        continue
                    frames = _scan_frames(res_dir)
                    if frames:
                        versions[ver_name] = {
                            "plate_name":  plate_name,
                            "res_folder":  res_folder,
                            "first_frame": frames[0][1],
                            "all_frames":  [fp for (_num, fp) in frames],
                            "frame_count": len(frames),
                            "start_frame": frames[0][0],
                            "end_frame":   frames[-1][0],
                            "extension":   os.path.splitext(frames[0][1])[1],
                        }
                        break
                if ver_name in versions:
                    break

        if versions:
            ver_list = sorted(versions.keys())
            results.append({
                "shot_name":    shot_name,
                "shot_path":    shot_dir,
                "versions":     ver_list,
                "version_data": versions,
            })
    return results


def _scan_frames(folder):
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith(IMAGE_EXTENSIONS):
            filepath = os.path.normpath(os.path.join(folder, f))
            stem = os.path.splitext(f)[0]
            match = re.search(r'(\d+)$', stem)
            frame_num = int(match.group(1)) if match else 0
            files.append((frame_num, filepath))
    files.sort(key=lambda x: x[0])
    return files


# ============================================================
#  SYNTHEYES ENGINE
# ============================================================

class SynthEyesEngine:
    def __init__(self, settings, on_log=None):
        self.settings = settings
        self.on_log = on_log or print
        self.hlev = None
        self._stop_requested = False

    def log(self, msg):
        safe = str(msg).encode("ascii", "replace").decode("ascii")
        self.on_log(safe)

    def request_stop(self):
        self._stop_requested = True

    def setup_sypy(self):
        for p in [SITE_PACKAGES, SYPY3_PATH]:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        if not os.path.isdir(SYPY3_PATH):
            self.log(f"ERROR: SyPy3 not found at {SYPY3_PATH}")
            return False
        self.log("OK  SyPy3 found")
        return True

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
                        except:
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
        except:
            pass
        if killed:
            self.log("   SynthEyes terminated. Waiting 3s...")
            time.sleep(3)
        else:
            self.log("   No existing SynthEyes instance found.")

    def launch(self):
        exe = self.settings["syntheyes_exe"]
        port = self.settings["port"]
        pin = self.settings["pin"]
        wait = self.settings["startup_wait"]
        self.kill_syntheyes()
        if not os.path.isfile(exe):
            self.log(f"ERROR: SynthEyes not found at: {exe}")
            return False
        self.log("-> Launching SynthEyes...")
        subprocess.Popen([exe, "-l", str(port), "-pin", pin])
        self.log(f"   Waiting {wait}s for startup...")
        time.sleep(wait)
        self.log("OK  SynthEyes launched")
        return True

    def connect(self):
        from SyPy3.sylevel import SyLevel
        port = self.settings["port"]
        pin = self.settings["pin"]
        self.hlev = SyLevel()
        self.log(f"-> Connecting port={port}...")
        if not self.hlev.OpenExisting(port, pin):
            self.log("   Connection failed.")
            return False
        self.log(f"OK  Connected — SynthEyes {self.hlev.Version()}")
        return True

    def connect_or_launch(self):
        self.log("-> Checking for existing SynthEyes instance...")
        try:
            from SyPy3.sylevel import SyLevel
            port = self.settings["port"]
            pin = self.settings["pin"]
            test = SyLevel()
            if test.OpenExisting(port, pin):
                self.hlev = test
                self.log(f"OK  Reusing existing SynthEyes — {self.hlev.Version()}")
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
            except:
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

    def is_alive(self):
        try:
            v = self.hlev.Version()
            return v is not None and len(str(v)) > 0
        except:
            return False

    def restart(self):
        self.log("-> RESTARTING SynthEyes (memory cleanup)...")
        try:
            self.hlev.Close()
        except:
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

    # ------ processing ------

    def process_shot(self, shot_name, shot_path, first_frame, all_frames,
                     frame_count, start_frame, end_frame,
                     preset_name="Normal / Handheld",
                     custom_count=500, custom_threshold=0.02, custom_separation=8,
                     row_override_count=None, row_override_thresh=None, row_override_sep=None):
        hlev = self.hlev

        output_dir = get_shot_output_dir(shot_path)
        version_num = get_next_version_number(output_dir, shot_name)
        ifl_filename = make_versioned_filename(shot_name, version_num, ".ifl")
        txt_filename = make_versioned_filename(shot_name, version_num, ".txt")
        ifl_path = os.path.join(output_dir, ifl_filename)

        preset = TRACKING_PRESETS.get(preset_name, TRACKING_PRESETS[DEFAULT_PRESET])
        if preset_name == "Custom":
            track_count = custom_count
            track_threshold = custom_threshold
            track_separation = custom_separation
        else:
            track_count = preset["count"]
            track_threshold = preset["threshold"]
            track_separation = preset["separation"]

        if row_override_count is not None:
            track_count = row_override_count
        if row_override_thresh is not None:
            track_threshold = row_override_thresh
        if row_override_sep is not None:
            track_separation = row_override_sep

        self.log(f"\n{'='*55}")
        self.log(f"  Processing: {shot_name}")
        self.log(f"  Preset:     {preset_name}")
        self.log(f"  Settings:   count={track_count}, threshold={track_threshold}, separation={track_separation}")
        self.log(f"  Output dir: {output_dir}")
        self.log(f"  Version:    v{version_num:03d}")
        self.log(f"{'='*55}")

        self.set_writable_folder(output_dir)

        first_frame = os.path.normpath(first_frame)
        all_frames  = [os.path.normpath(f) for f in all_frames]

        if not os.path.isfile(first_frame):
            raise RuntimeError(f"First frame file NOT FOUND: {first_frame}")
        if not os.access(first_frame, os.R_OK):
            raise RuntimeError(f"No READ permission on: {first_frame}")
        self.log(f"   OK  Read access confirmed on plates")

        self.log(f"-> Creating IFL: {ifl_filename}")
        self.log(f"   Path:   {ifl_path}")
        self.log(f"   Frames: {len(all_frames)} files")
        try:
            with open(ifl_path, "w", encoding="utf-8") as f:
                for frame_path in all_frames:
                    abs_path = os.path.abspath(frame_path)
                    safe_path = abs_path.replace("\\", "/")
                    f.write(safe_path + "\n")
            self.log(f"   OK  IFL written ({len(all_frames)} lines)")
        except Exception as e:
            raise RuntimeError(f"Could not write IFL file: {e}")

        self.log("-> Loading sequence via IFL...")
        shot = hlev.NewSceneAndShot(ifl_path)
        if shot is None:
            raise RuntimeError("NewSceneAndShot returned None — check IFL file")
        self.log(f"   OK  Shot loaded (index {shot.Index()})")

        if frame_count < 300:
            try:
                hlev.Validate(shot)
                self.log("   OK  Validated shot RAM cache")
            except Exception as e:
                self.log(f"   WARNING: Validate failed: {e}")
        else:
            self.log("   Skipped RAM cache validation (sequence too long, preventing black screen).")

        try:
            hlev.Redraw()
            self.log("   OK  Redraw triggered")
        except Exception as e:
            self.log(f"   WARNING: Redraw failed: {e}")

        load_wait = 10 if frame_count > 500 else 5
        self.log(f"   Waiting {load_wait}s for image plane to fully load...")
        time.sleep(load_wait)

        try:
            check_path = hlev.ShotSingleImageName(shot, start_frame)
            if check_path:
                self.log(f"   VERIFIED: SynthEyes sees frame {start_frame} at: {check_path}")
            else:
                self.log(f"   WARNING: SynthEyes returned empty path for frame {start_frame}")
        except Exception as e:
            self.log(f"   WARNING: Could not verify image path: {e}")

        self._set_frame_range(shot, start_frame, end_frame)
        self.log(f"   Frames: {start_frame}-{end_frame} ({frame_count} total)")

        try:
            hlev.Redraw()
        except:
            pass

        self.log("-> Switching to Features room...")
        self._switch_room("Features", "Feature")

        self._configure_features(track_count, track_threshold, track_separation)

        # ---- Set max tracker count in Advanced dialog ----
        self._set_advanced_max_tracks(track_count)

        blip_timeout = max(120, int(frame_count * 0.2))
        self.log(f"-> Blip All frames... (timeout {blip_timeout}s)")
        if not self._click_and_wait("Feature/Blips all frames", "Blip All", timeout=blip_timeout):
            raise RuntimeError("Blip All failed")

        peel_timeout = max(600, int(frame_count * 1.0))
        self.log(f"-> Peel All ({frame_count} frames, timeout {peel_timeout}s)...")
        if not self._click_and_wait("Feature/Peel All", "Peel All", timeout=peel_timeout):
            raise RuntimeError("Peel All failed")

        self.log("-> Clearing blips...")
        self._click_and_wait("Feature/Clear all blips", "Clear blips", timeout=10)

        self.log("-> Switching to Trackers room...")
        self._switch_room("Trackers", "Tracker")

        tracker_list = hlev.Trackers()
        self.log(f"   Trackers found: {len(tracker_list)}")

        if len(tracker_list) > 0:
            self.log("-> Exporting 2D tracks...")
            self._export_tracks(output_dir, txt_filename)
        else:
            self.log("   WARNING: 0 trackers — skipping export")

        return len(tracker_list), output_dir, version_num

    # ------ frame range ------

    def _set_frame_range(self, shot, start, end):
        hlev = self.hlev
        try:
            hlev.SetAnimStart(start)
            hlev.SetAnimEnd(end)
        except Exception as e:
            self.log(f"   WARNING: SetAnimStart/End failed: {e}")
        try:
            hlev.Begin()
            for a_s, a_e in [("startFrame", "endFrame"), ("firstFrame", "lastFrame"), ("inFrame", "outFrame")]:
                try:
                    shot.Set(a_s, start)
                    shot.Set(a_e, end)
                    break
                except:
                    pass
            for attr in ["matchFrameNumbers", "matchFrameNum", "useFrameNumbers"]:
                try:
                    shot.Set(attr, 1)
                    break
                except:
                    pass
            hlev.Accept("Set frame range")
        except Exception as e:
            try:
                hlev.Cancel()
            except:
                pass
            self.log(f"   WARNING: frame range skipped: {e}")

    def _switch_room(self, *names):
        for name in names:
            try:
                self.hlev.SetRoom(name)
                time.sleep(1)
                self.log(f"   Room -> '{name}'")
                return
            except:
                pass
        self.log(f"   WARNING: Could not switch room")

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

    # ==============================================================
    #  ADVANCED DIALOG — Known Control IDs (v4.4 Hardware Input)
    # ==============================================================

    ADV_BUTTON_ID        = 1238   
    ADV_MAX_TRACKS_ID    = 1266   

    def _set_advanced_max_tracks(self, max_tracks):
        """
        Sets the Maximum Tracker Count in the Advanced dialog
        using OS-level hardware keystrokes (keybd_event).

        Key fix (v4.5): AttachThreadInput() is called BEFORE
        SetFocus/SetForegroundWindow. Without it, Windows silently
        blocks all cross-thread focus changes, and keybd_event
        keystrokes land on the wrong window.
        """
        hlev = self.hlev
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self.log(f"-> Setting Advanced Max Tracks to: {max_tracks}")

        # ---- 1) Click the Advanced Button ----
        try:
            hlev.Main().ByID(self.ADV_BUTTON_ID).ClickAndContinue()
            time.sleep(1.5)
        except Exception as e:
            self.log(f"   ERROR: Could not click Advanced button: {e}")
            return False

        # ---- 2) Find the dialog via FindWindowW (reliable cross-thread) ----
        dialog_hwnd = 0
        for wait in range(1, 11):
            time.sleep(0.5)
            dialog_hwnd = user32.FindWindowW(None, "Advanced Feature Control")
            if dialog_hwnd:
                self.log(f"   OK  Found dialog after {wait * 0.5:.1f}s (HWND=0x{dialog_hwnd:X})")
                break

        if not dialog_hwnd:
            self.log("   WARNING: Advanced dialog not found — skipping")
            return False

        # ---- 3) Find the spinner (ID 1266) ----
        spinner_hwnd = user32.GetDlgItem(dialog_hwnd, self.ADV_MAX_TRACKS_ID)
        if not spinner_hwnd:
            self.log("   WARNING: Spinner 1266 not found")
            return False
        self.log(f"   Spinner HWND: 0x{spinner_hwnd:X}")

        # ---- 4) AttachThreadInput — THE CRITICAL FIX ----
        #   Without this, SetFocus() and SetForegroundWindow() are
        #   silently ignored across threads, and keybd_event goes
        #   to whatever window currently has focus (not our spinner).
        our_tid = kernel32.GetCurrentThreadId()
        se_tid = user32.GetWindowThreadProcessId(dialog_hwnd, None)
        attached = False
        if our_tid != se_tid:
            attached = bool(user32.AttachThreadInput(our_tid, se_tid, True))
            self.log(f"   AttachThreadInput: {'OK' if attached else 'FAILED'}")

        try:
            # ---- 5) Focus the dialog and spinner ----
            user32.ShowWindow(dialog_hwnd, 5)  # SW_SHOW
            user32.SetForegroundWindow(dialog_hwnd)
            time.sleep(0.3)
            user32.SetFocus(spinner_hwnd)
            time.sleep(0.2)

            # Click inside the spinner control to position cursor in the text
            lParam = (10 << 16) | 30  # MAKELPARAM(x=30, y=10)
            user32.SendMessageW(spinner_hwnd, 0x0201, 1, lParam)  # WM_LBUTTONDOWN
            user32.SendMessageW(spinner_hwnd, 0x0202, 0, lParam)  # WM_LBUTTONUP
            time.sleep(0.2)

            # ---- 6) Hardware keystrokes: Ctrl+A, type digits, Enter ----
            VK_CONTROL = 0x11
            VK_A = 0x41
            VK_RETURN = 0x0D
            KEYEVENTF_KEYUP = 0x0002

            def press_key(vk):
                user32.keybd_event(vk, 0, 0, 0)
                time.sleep(0.02)
                user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
                time.sleep(0.02)

            # Ctrl+A (select all text in the spinner)
            user32.keybd_event(VK_CONTROL, 0, 0, 0)
            time.sleep(0.02)
            press_key(VK_A)
            user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
            time.sleep(0.1)

            # Type each digit of the new value
            for char in str(int(max_tracks)):
                vk_code = ord(char)  # VK codes for 0-9 match ASCII
                press_key(vk_code)
                time.sleep(0.05)

            time.sleep(0.1)

            # Press Enter to commit the value
            press_key(VK_RETURN)
            time.sleep(0.3)

            # ---- 7) Click OK (ID=1) via mouse message ----
            ok_btn = user32.GetDlgItem(dialog_hwnd, 1)
            if ok_btn:
                user32.SendMessageW(ok_btn, 0x0201, 1, 0)  # WM_LBUTTONDOWN
                time.sleep(0.05)
                user32.SendMessageW(ok_btn, 0x0202, 0, 0)  # WM_LBUTTONUP
                self.log(f"   OK  Set Max Tracks = {max_tracks} (hardware keystrokes)")
            else:
                press_key(VK_RETURN)  # Fallback: Enter to close dialog
                self.log(f"   OK  Set Max Tracks = {max_tracks} (Enter to close)")

            time.sleep(0.3)
            return True

        finally:
            # ---- 8) Detach thread input ----
            if attached:
                user32.AttachThreadInput(our_tid, se_tid, False)

    def _click_and_wait(self, action_name, label, timeout=300):
        hlev = self.hlev
        idno = hlev.ActionID(action_name)
        if idno <= 0:
            self.log(f"   ERROR: Action '{action_name}' not found")
            return False
        try:
            hlev.Main().ByID(idno).ClickAndContinue()
            time.sleep(0.5)
            elapsed = 0
            while hlev.Popup().IsValid():
                time.sleep(1)
                elapsed += 1
                if elapsed >= timeout:
                    self.log(f"   WARNING: {label} timed out after {timeout}s")
                    return False
            self.log(f"   OK  {label} done (~{elapsed}s)")
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
            except:
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
            except:
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
        else:
            self.log(f"   WARNING: file not found after save")
            return False


# ============================================================
#  SHOT ROW
# ============================================================

class ShotRow:
    def __init__(self, parent, shot_data, row_index):
        self.data = shot_data
        self.var_selected = tk.BooleanVar(value=True)
        self.var_version = tk.StringVar()
        self.var_preset = tk.StringVar(value=DEFAULT_PRESET)
        self.var_custom_count = tk.StringVar(value="")
        self.var_custom_thresh = tk.StringVar(value="")
        self.var_custom_sep = tk.StringVar(value="")

        versions = shot_data["versions"]
        self.var_version.set(versions[-1])

        self.frame = tk.Frame(parent, bg="#313338", highlightbackground="#202225",
                              highlightthickness=1)
        self.frame.pack(fill="x", padx=8, pady=3)

        self.chk = tk.Checkbutton(self.frame, variable=self.var_selected,
                                  bg="#313338", activebackground="#313338", selectcolor="#1E1F22")
        self.chk.pack(side="left", padx=(8, 4))

        self.status_lbl = tk.Label(self.frame, text="  —  ", bg="#313338",
                                   font=("Consolas", 10), fg="#949BA4", width=5)
        self.status_lbl.pack(side="left", padx=(0, 4))

        self.name_lbl = tk.Label(self.frame, text=shot_data["shot_name"],
                                 font=("Segoe UI", 10, "bold"), bg="#313338", fg="#DBDEE1",
                                 width=18, anchor="w")
        self.name_lbl.pack(side="left", padx=(0, 8))

        if len(versions) > 1:
            tk.Label(self.frame, text="Ver:", bg="#313338", fg="#DBDEE1",
                     font=("Segoe UI", 8)).pack(side="left")
            self.ver_menu = tk.OptionMenu(self.frame, self.var_version,
                                          *versions, command=lambda _: self._update_info())
            self.ver_menu.config(width=5, font=("Segoe UI", 8), bg="#1E1F22", fg="#DBDEE1", activebackground="#2B2D31", highlightthickness=0)
            self.ver_menu["menu"].config(bg="#1E1F22", fg="#DBDEE1")
            self.ver_menu.pack(side="left", padx=(2, 6))
        else:
            tk.Label(self.frame, text=versions[0], bg="#313338",
                     fg="#949BA4", font=("Segoe UI", 8)).pack(side="left", padx=(0, 6))

        tk.Label(self.frame, text="Preset:", bg="#313338", fg="#DBDEE1",
                 font=("Segoe UI", 8)).pack(side="left")
        self.preset_menu = tk.OptionMenu(self.frame, self.var_preset, *PRESET_NAMES)
        self.preset_menu.config(width=16, font=("Segoe UI", 8), bg="#1E1F22", fg="#DBDEE1", activebackground="#2B2D31", highlightthickness=0)
        self.preset_menu["menu"].config(bg="#1E1F22", fg="#DBDEE1")
        self.preset_menu.pack(side="left", padx=(2, 8))

        tk.Label(self.frame, text="Tracks:", bg="#313338", fg="#DBDEE1", font=("Segoe UI", 8)).pack(side="left", padx=(8, 2))
        self.entry_count = tk.Entry(self.frame, textvariable=self.var_custom_count, width=5, bg="#1E1F22", fg="#DBDEE1", insertbackground="#DBDEE1", font=("Segoe UI", 9))
        self.entry_count.pack(side="left")

        tk.Label(self.frame, text="Thresh:", bg="#313338", fg="#DBDEE1", font=("Segoe UI", 8)).pack(side="left", padx=(6, 2))
        self.entry_thresh = tk.Entry(self.frame, textvariable=self.var_custom_thresh, width=5, bg="#1E1F22", fg="#DBDEE1", insertbackground="#DBDEE1", font=("Segoe UI", 9))
        self.entry_thresh.pack(side="left")

        tk.Label(self.frame, text="Sep:", bg="#313338", fg="#DBDEE1", font=("Segoe UI", 8)).pack(side="left", padx=(6, 2))
        self.entry_sep = tk.Entry(self.frame, textvariable=self.var_custom_sep, width=4, bg="#1E1F22", fg="#DBDEE1", insertbackground="#DBDEE1", font=("Segoe UI", 9))
        self.entry_sep.pack(side="left")

        self.info_lbl = tk.Label(self.frame, text="", bg="#313338",
                                 font=("Consolas", 8), fg="#949BA4", anchor="w")
        self.info_lbl.pack(side="left", fill="x", expand=True, padx=(8, 8))

        self._update_info()

    def _update_info(self):
        ver = self.var_version.get()
        vd = self.data["version_data"].get(ver)
        if vd:
            info = (f"{vd['frame_count']} fr  "
                    f"[{vd['start_frame']}-{vd['end_frame']}]  |  "
                    f"{vd['res_folder']}  |  "
                    f"{vd['extension']}")
        else:
            info = "(no data)"
        self.info_lbl.config(text=info)

    def set_status(self, status, tracker_count=0, version_str=""):
        icons = {
            "pending":  ("  —  ", "#949BA4"),
            "running":  ("  ...  ", "#F39C12"),
            "done":     (f" \u2714 {tracker_count}", "#2ECC71"),
            "failed":   (" \u2718 FAIL", "#E74C3C"),
        }
        text, color = icons.get(status, ("  ?  ", "#949BA4"))
        try:
            self.status_lbl.config(text=text, fg=color)
        except:
            pass

    def get_selected_data(self):
        if not self.var_selected.get():
            return None
        ver = self.var_version.get()
        vd = self.data["version_data"].get(ver)
        if not vd:
            return None
        override_cnt = None
        if self.var_custom_count.get().strip().isdigit():
            override_cnt = int(self.var_custom_count.get().strip())
        override_thr = None
        try:
            override_thr = float(self.var_custom_thresh.get().strip())
        except ValueError:
            pass
        override_sep = None
        if self.var_custom_sep.get().strip().isdigit():
            override_sep = int(self.var_custom_sep.get().strip())
        return (self.data["shot_name"], self.data["shot_path"], vd, self.var_preset.get(), override_cnt, override_thr, override_sep)


# ============================================================
#  MAIN UI
# ============================================================

class BatchTrackerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SynthEyes Batch Auto-Tracker  v4.5")
        self.root.geometry("1100x720")
        self.root.resizable(True, True)
        self.root.configure(bg="#202225")

        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        self._rows = []
        self._engine = None
        self._thread = None
        self._stop_flag = False
        self._batch_results = []
        self._show_path = ""

        self.var_show_path  = tk.StringVar(value="")
        self.var_exe        = tk.StringVar(value=DEFAULTS["syntheyes_exe"])
        self.var_port       = tk.IntVar(value=DEFAULTS["port"])
        self.var_pin        = tk.StringVar(value=DEFAULTS["pin"])
        self.var_count      = tk.IntVar(value=DEFAULTS["tracker_count"])
        self.var_threshold  = tk.DoubleVar(value=DEFAULTS["threshold"])
        self.var_separation = tk.IntVar(value=DEFAULTS["separation"])
        self.var_status     = tk.StringVar(value="Ready — browse a show folder to begin.")
        self.var_global_preset = tk.StringVar(value=DEFAULT_PRESET)

        self._build_ui()

    def _build_ui(self):
        top = tk.LabelFrame(self.root, text="  Show Folder  ", font=("Segoe UI", 10, "bold"),
                            bg="#202225", fg="#DBDEE1", padx=10, pady=6)
        top.pack(fill="x", padx=10, pady=(10, 4))

        tk.Label(top, text="Show Path:", bg="#202225", fg="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.var_show_path, width=80,
                 bg="#1E1F22", fg="#DBDEE1", insertbackground="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=0, column=1, sticky="we", padx=6)
        tk.Button(top, text="Browse Show Folder", command=self._browse_show,
                  font=("Segoe UI", 9, "bold"), bg="#4A90D9", fg="white",
                  relief="flat", padx=12).grid(row=0, column=2, padx=(4, 0))
        top.grid_columnconfigure(1, weight=1)

        self.var_output_info = tk.StringVar(value="")
        tk.Label(top, textvariable=self.var_output_info, bg="#202225",
                 fg="#949BA4", font=("Segoe UI", 8)).grid(
                     row=1, column=1, columnspan=2, sticky="w", padx=6, pady=(2, 0))

        mid_frame = tk.LabelFrame(self.root, text="  Shots  ", font=("Segoe UI", 10, "bold"),
                                  bg="#202225", fg="#DBDEE1", padx=6, pady=6)
        mid_frame.pack(fill="both", expand=True, padx=10, pady=4)

        hdr = tk.Frame(mid_frame, bg="#202225")
        hdr.pack(fill="x", pady=(0, 4))

        self.shot_count_lbl = tk.Label(hdr, text="No shots loaded", bg="#202225",
                                       font=("Segoe UI", 9), fg="#949BA4")
        self.shot_count_lbl.pack(side="left")

        tk.Label(hdr, text="   Global Preset:", bg="#202225", fg="#DBDEE1",
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(16, 4))
        self.global_preset_menu = tk.OptionMenu(hdr, self.var_global_preset, *PRESET_NAMES)
        self.global_preset_menu.config(width=16, font=("Segoe UI", 9), bg="#1E1F22", fg="#DBDEE1", activebackground="#2B2D31", highlightthickness=0)
        self.global_preset_menu["menu"].config(bg="#1E1F22", fg="#DBDEE1")
        self.global_preset_menu.pack(side="left")
        tk.Button(hdr, text="Apply to All Shots", command=self._apply_preset_to_all,
                  font=("Segoe UI", 8, "bold"), bg="#8E44AD", fg="white",
                  relief="flat", padx=8).pack(side="left", padx=(6, 0))

        tk.Button(hdr, text="Select All", command=self._select_all,
                  bg="#2B2D31", fg="#DBDEE1", relief="flat", font=("Segoe UI", 8)).pack(side="right", padx=4)
        tk.Button(hdr, text="Select None", command=self._select_none,
                  bg="#2B2D31", fg="#DBDEE1", relief="flat", font=("Segoe UI", 8)).pack(side="right")

        self.canvas = tk.Canvas(mid_frame, bg="#2B2D31", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(mid_frame, orient="vertical", command=self.canvas.yview)
        self.list_frame = tk.Frame(self.canvas, bg="#2B2D31")

        self.list_frame.bind("<Configure>",
                             lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._canvas_win = self.canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        settings_frame = tk.LabelFrame(self.root, text="  Settings  ",
                                       font=("Segoe UI", 10, "bold"),
                                       bg="#202225", fg="#DBDEE1", padx=10, pady=6)
        settings_frame.pack(fill="x", padx=10, pady=4)

        r = 0
        tk.Label(settings_frame, text="SynthEyes .exe:", bg="#202225", fg="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=0, sticky="w")
        tk.Entry(settings_frame, textvariable=self.var_exe, width=55,
                 bg="#1E1F22", fg="#DBDEE1", insertbackground="#DBDEE1",
                 font=("Segoe UI", 8)).grid(row=r, column=1, columnspan=5, sticky="we", padx=4)
        tk.Button(settings_frame, text="...", width=3, bg="#2B2D31", fg="#DBDEE1", relief="flat",
                  command=self._browse_exe).grid(row=r, column=6)

        r = 1
        tk.Label(settings_frame, text="Custom Settings:", bg="#202225", fg="#DBDEE1",
                 font=("Segoe UI", 9, "bold")).grid(row=r, column=0, sticky="w", pady=(6, 0))
        tk.Label(settings_frame, text="(only used when a shot's preset is 'Custom')",
                 bg="#202225", fg="#949BA4", font=("Segoe UI", 8)).grid(
                     row=r, column=1, columnspan=5, sticky="w", padx=4, pady=(6, 0))

        r = 2
        tk.Label(settings_frame, text="Tracker Count:", bg="#202225", fg="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=0, sticky="w", pady=2)
        tk.Entry(settings_frame, textvariable=self.var_count, width=8,
                 bg="#1E1F22", fg="#DBDEE1", insertbackground="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=1, sticky="w", padx=4)
        tk.Label(settings_frame, text="Threshold:", bg="#202225", fg="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=2, sticky="w")
        tk.Entry(settings_frame, textvariable=self.var_threshold, width=8,
                 bg="#1E1F22", fg="#DBDEE1", insertbackground="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=3, sticky="w", padx=4)
        tk.Label(settings_frame, text="Separation:", bg="#202225", fg="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=4, sticky="w")
        tk.Entry(settings_frame, textvariable=self.var_separation, width=8,
                 bg="#1E1F22", fg="#DBDEE1", insertbackground="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=5, sticky="w", padx=4)

        r = 3
        tk.Label(settings_frame, text="SyPy Port:", bg="#202225", fg="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=0, sticky="w", pady=2)
        tk.Entry(settings_frame, textvariable=self.var_port, width=8,
                 bg="#1E1F22", fg="#DBDEE1", insertbackground="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=1, sticky="w", padx=4)
        tk.Label(settings_frame, text="Pin:", bg="#202225", fg="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=2, sticky="w")
        tk.Entry(settings_frame, textvariable=self.var_pin, width=10,
                 bg="#1E1F22", fg="#DBDEE1", insertbackground="#DBDEE1",
                 font=("Segoe UI", 9)).grid(row=r, column=3, sticky="w", padx=4)

        settings_frame.grid_columnconfigure(1, weight=1)

        prog_frame = tk.Frame(self.root, bg="#202225")
        prog_frame.pack(fill="x", padx=10, pady=(4, 0))
        self.progress_bar = ttk.Progressbar(prog_frame, orient="horizontal",
                                            length=300, mode="determinate")
        self.progress_bar.pack(fill="x", pady=2)
        self.var_progress_text = tk.StringVar(value="")
        tk.Label(prog_frame, textvariable=self.var_progress_text, bg="#202225",
                 font=("Segoe UI", 8), fg="#949BA4").pack(anchor="w")

        log_frame = tk.LabelFrame(self.root, text="  Log  ", font=("Segoe UI", 10, "bold"),
                                  bg="#202225", fg="#DBDEE1", padx=6, pady=4)
        log_frame.pack(fill="x", padx=10, pady=4)
        self.log_text = tk.Text(log_frame, height=6, font=("Consolas", 9),
                                bg="#111214", fg="#CCCCCC", insertbackground="white",
                                state="disabled", wrap="word", highlightthickness=0)
        log_scroll = tk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        bot = tk.Frame(self.root, bg="#202225")
        bot.pack(fill="x", padx=10, pady=(4, 10))
        self.btn_start = tk.Button(bot, text="  Start Batch  ", font=("Segoe UI", 10, "bold"),
                                   bg="#2ECC71", fg="white", relief="flat", padx=16, pady=4,
                                   command=self._start_batch)
        self.btn_start.pack(side="left")
        self.btn_stop = tk.Button(bot, text="  Stop  ", font=("Segoe UI", 10),
                                  bg="#E74C3C", fg="white", relief="flat", padx=16, pady=4,
                                  command=self._stop_batch, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))
        self.btn_open_show = tk.Button(bot, text="  Open Show Folder  ",
                                       font=("Segoe UI", 9),
                                       bg="#3498DB", fg="white", relief="flat", padx=12, pady=4,
                                       command=self._open_show_folder)
        self.btn_open_show.pack(side="left", padx=(8, 0))
        tk.Label(bot, textvariable=self.var_status, bg="#202225",
                 font=("Segoe UI", 9), fg="#4A90D9").pack(side="left", padx=16)

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self._canvas_win, width=event.width)

    def _browse_show(self):
        path = filedialog.askdirectory(title="Select Show Folder")
        if path:
            self.var_show_path.set(path)
            self._show_path = os.path.normpath(path)
            self.var_output_info.set(
                f"Output: each shot saves to  ShotName/mid/cmm/bot_MMpoints/  (auto-versioned)")
            self._scan_and_populate(path)

    def _browse_exe(self):
        path = filedialog.askopenfilename(
            title="Select SynthEyes Executable",
            filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if path:
            self.var_exe.set(path)

    def _open_show_folder(self):
        if self._show_path and os.path.isdir(self._show_path):
            os.startfile(self._show_path)
        else:
            messagebox.showinfo("Show Folder", "No show folder selected yet.")

    def _scan_and_populate(self, show_path):
        for child in self.list_frame.winfo_children():
            child.destroy()
        self._rows.clear()
        self._log_append(f"Scanning: {show_path}")
        shots = scan_show_folder(show_path)
        if not shots:
            tk.Label(self.list_frame, text="  No shots found. Check folder structure.",
                     bg="#2B2D31", fg="#E74C3C", font=("Segoe UI", 10)).pack(
                         anchor="w", padx=10, pady=20)
            self.shot_count_lbl.config(text="0 shots found")
            self._log_append("No shots found — expected: Show > Shot > in > plates > vXXX > ...")
            return
        for i, shot_data in enumerate(shots):
            row = ShotRow(self.list_frame, shot_data, i)
            self._rows.append(row)
        count = len(shots)
        self.shot_count_lbl.config(text=f"{count} shot(s) found")
        self._log_append(f"Found {count} shot(s)")
        self.var_status.set(f"{count} shots loaded — select shots and hit Start Batch.")

    def _select_all(self):
        for r in self._rows:
            r.var_selected.set(True)

    def _select_none(self):
        for r in self._rows:
            r.var_selected.set(False)

    def _apply_preset_to_all(self):
        preset = self.var_global_preset.get()
        for r in self._rows:
            r.var_preset.set(preset)
        self._log_append(f"Applied preset '{preset}' to all shots")

    def _log_append(self, msg):
        def _do():
            self.log_text.config(state="normal")
            self.log_text.insert("end", str(msg) + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        try:
            self.root.after(0, _do)
        except:
            pass

    def _gather_settings(self):
        return {
            "syntheyes_exe": self.var_exe.get().strip(),
            "port":          int(self.var_port.get()),
            "pin":           self.var_pin.get().strip(),
            "tracker_count": int(self.var_count.get()),
            "threshold":     float(self.var_threshold.get()),
            "separation":    int(self.var_separation.get()),
            "startup_wait":  DEFAULTS["startup_wait"],
        }

    def _start_batch(self):
        selected = []
        for row in self._rows:
            result = row.get_selected_data()
            if result:
                selected.append((result, row))
        if not selected:
            messagebox.showwarning("No Shots Selected", "Please tick at least one shot to process.")
            return
        if not self._show_path:
            messagebox.showerror("Error", "Please browse a Show Folder first.")
            return
        settings = self._gather_settings()
        names = [s[0][0] for s in selected]
        msg = f"Start batch tracking {len(selected)} shot(s)?\n\n" + "\n".join(names)
        if not messagebox.askyesno("Confirm Batch", msg):
            return
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = len(selected)
        self.var_progress_text.set("")
        self._batch_results = []
        for _, row in selected:
            row.set_status("pending")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._stop_flag = False
        self.var_status.set("Running...")
        self._thread = threading.Thread(
            target=self._run_batch, args=(selected, settings), daemon=True)
        self._thread.start()

    def _stop_batch(self):
        self._stop_flag = True
        if self._engine:
            self._engine.request_stop()
        self.var_status.set("Stop requested...")
        self.btn_stop.config(state="disabled")

    def _run_batch(self, selected, settings):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        shot_log_file = None

        def batch_log(msg):
            self._log_append(msg)
            if shot_log_file:
                try:
                    shot_log_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
                    shot_log_file.flush()
                except:
                    pass

        def open_shot_log(output_dir, shot_name, ver_num):
            log_name = f"{shot_name}_v{ver_num:03d}_log.txt"
            log_path = os.path.join(output_dir, log_name)
            try:
                return open(log_path, "w", encoding="utf-8"), log_path
            except Exception as e:
                self._log_append(f"   WARNING: Could not create shot log: {e}")
                return None, ""

        def close_shot_log():
            nonlocal shot_log_file
            if shot_log_file:
                try:
                    shot_log_file.close()
                except:
                    pass
                shot_log_file = None

        try:
            engine = SynthEyesEngine(settings, on_log=batch_log)
            self._engine = engine
            batch_log("=" * 55)
            batch_log("  SynthEyes Batch Auto-Tracker  v4.5")
            batch_log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            batch_log(f"  Show: {self._show_path}")
            batch_log("=" * 55)

            if not engine.setup_sypy():
                self._finish("ERROR: SyPy3 not found")
                return
            if not engine.connect_or_launch():
                self._finish("ERROR: Could not start/connect to SynthEyes")
                return

            total = len(selected)
            prev_was_heavy = False
            shot_times = []

            for i, ((shot_name, shot_path, vd, preset_name, override_cnt, override_thr, override_sep), row) in enumerate(selected):
                if self._stop_flag:
                    batch_log("STOPPED by user.")
                    break

                frame_count = vd["frame_count"]
                self._update_status(f"Processing {i+1}/{total}: {shot_name} ({frame_count} frames)")
                self._update_row_status(row, "running")

                if prev_was_heavy and i > 0:
                    batch_log("-> Previous shot was heavy (1000+ frames) — restarting SynthEyes...")
                    if not engine.restart():
                        batch_log("ERROR: Could not restart SynthEyes! Aborting batch.")
                        self._update_row_status(row, "failed")
                        break

                if not engine.is_alive():
                    batch_log("WARNING: SynthEyes not responding — attempting restart...")
                    if not engine.restart():
                        batch_log("ERROR: Could not restart SynthEyes! Aborting batch.")
                        self._update_row_status(row, "failed")
                        break

                shot_output_dir = get_shot_output_dir(shot_path)
                version_num = get_next_version_number(shot_output_dir, shot_name)

                shot_log_file, shot_log_path = open_shot_log(shot_output_dir, shot_name, version_num)
                if shot_log_path:
                    batch_log(f"   Log -> {shot_log_path}")

                shot_start_time = time.time()
                tracker_count = 0
                status = "done"
                output_dir = shot_output_dir

                try:
                    tracker_count, output_dir, version_num = engine.process_shot(
                        shot_name=shot_name,
                        shot_path=shot_path,
                        first_frame=vd["first_frame"],
                        all_frames=vd["all_frames"],
                        frame_count=frame_count,
                        start_frame=vd["start_frame"],
                        end_frame=vd["end_frame"],
                        preset_name=preset_name,
                        custom_count=settings["tracker_count"],
                        custom_threshold=settings["threshold"],
                        custom_separation=settings["separation"],
                        row_override_count=override_cnt,
                        row_override_thresh=override_thr,
                        row_override_sep=override_sep
                    )
                    batch_log(f"   DONE: {shot_name} — {tracker_count} trackers (v{version_num:03d})")
                except Exception as e:
                    batch_log(f"ERROR on '{shot_name}': {e}")
                    batch_log(traceback.format_exc())
                    tracker_count = 0
                    status = "failed"

                shot_elapsed = time.time() - shot_start_time
                shot_times.append(shot_elapsed)
                self._update_row_status(row, status, tracker_count)

                shot_result = {
                    "shot_name": shot_name, "preset": preset_name,
                    "frames": frame_count, "trackers": tracker_count,
                    "version": f"v{version_num:03d}" if version_num > 0 else "",
                    "output_dir": output_dir,
                    "status": "OK" if status == "done" else "FAILED",
                    "time_sec": round(shot_elapsed, 1),
                }
                self._batch_results.append(shot_result)
                self._write_shot_summary_csv(output_dir, shot_name, version_num, shot_result)
                close_shot_log()

                try:
                    shots_list = engine.hlev.Shots()
                    if shots_list:
                        engine.flush_after_shot(shots_list[0])
                except Exception as e:
                    batch_log(f"   WARNING: Post-shot flush failed: {e}")

                prev_was_heavy = (frame_count >= 1000)
                completed = i + 1
                self._update_progress(completed, total, shot_times)

            batch_log("\n" + "=" * 55)
            batch_log("  BATCH COMPLETE")
            batch_log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            batch_log("=" * 55)
            self._finish(f"Done — processed {total} shot(s)")

        except Exception as e:
            batch_log(f"FATAL ERROR: {e}")
            batch_log(traceback.format_exc())
            self._finish(f"Error: {e}")
        finally:
            close_shot_log()

    def _update_row_status(self, row, status, tracker_count=0):
        try:
            self.root.after(0, lambda: row.set_status(status, tracker_count))
        except:
            pass

    def _update_progress(self, completed, total, shot_times):
        def _do():
            self.progress_bar["value"] = completed
            if completed < total and len(shot_times) > 0:
                avg_time = sum(shot_times) / len(shot_times)
                remaining = total - completed
                eta_sec = avg_time * remaining
                if eta_sec >= 60:
                    eta_str = f"~{int(eta_sec // 60)}m {int(eta_sec % 60)}s remaining"
                else:
                    eta_str = f"~{int(eta_sec)}s remaining"
                self.var_progress_text.set(f"{completed}/{total} shots done  |  {eta_str}")
            else:
                self.var_progress_text.set(f"{completed}/{total} shots done")
        try:
            self.root.after(0, _do)
        except:
            pass

    def _write_shot_summary_csv(self, output_dir, shot_name, version_num, result):
        csv_name = f"{shot_name}_v{version_num:03d}_summary.csv"
        csv_path = os.path.join(output_dir, csv_name)
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "shot_name", "preset", "frames", "trackers",
                    "version", "output_dir", "status", "time_sec"])
                writer.writeheader()
                writer.writerow(result)
            self._log_append(f"   Summary -> {csv_path}")
        except Exception as e:
            self._log_append(f"   WARNING: Could not write shot summary: {e}")

    def _update_status(self, msg):
        try:
            self.root.after(0, lambda: self.var_status.set(msg))
        except:
            pass

    def _finish(self, msg):
        def _do():
            self.var_status.set(msg)
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
        try:
            self.root.after(0, _do)
        except:
            pass

    def run(self):
        self.root.mainloop()


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = BatchTrackerApp()
    app.run()