"""Talking to the sidecar. Stdlib only.

`urllib.request`, not `requests`, even though Blender 5.2 bundles requests -- this way the
addon has zero non-bpy dependencies and cannot be broken by somebody's Blender build. And
nothing but JSON crosses the wire: the sidecar reads plate pixels off disk itself, so the
addon sends a path and a frame range, never an image.
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request


class SidecarError(Exception):
    """Carries a sentence an artist can act on. The traceback stays in the sidecar log."""


def _portfile(assist_root):
    return os.path.join(assist_root, "logs", "sidecar.json")


def read_portfile(assist_root):
    try:
        with open(_portfile(assist_root), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _request(info, path, payload=None, timeout=10.0):
    url = "http://127.0.0.1:%d%s" % (int(info["port"]), path)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("X-BTR-Token", info.get("token", ""))
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
            raise SidecarError(body.get("error", {}).get("message", str(exc)))
        except (ValueError, KeyError):
            raise SidecarError("sidecar returned HTTP %d" % exc.code)
    except (urllib.error.URLError, OSError) as exc:
        raise SidecarError("sidecar is not answering (%s)" % exc)


def health(assist_root, timeout=3.0):
    info = read_portfile(assist_root)
    if info is None:
        return None
    try:
        return _request(info, "/health", timeout=timeout)
    except SidecarError:
        return None


def ensure(assist_root, python_exe, port=0, timeout=60.0):
    """Return a live sidecar, spawning one if needed.

    60 s because the first health check imports torch, and a cold CUDA init on a spinning
    disk is genuinely slow. A stale portfile from a killed process is the normal case, not
    an error -- it is overwritten.
    """
    h = health(assist_root)
    if h is not None:
        return read_portfile(assist_root), h
    if not python_exe or not os.path.isfile(python_exe):
        raise SidecarError("no sidecar Python configured -- run bootstrap.bat, then set it "
                           "in Preferences > Add-ons > Tracking Assistant")

    pf = _portfile(assist_root)
    try:
        os.remove(pf)
    except OSError:
        pass
    os.makedirs(os.path.dirname(pf), exist_ok=True)
    log = open(os.path.join(assist_root, "logs", "sidecar.log"), "ab", buffering=0)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # Launched by FILE, not as `-m sidecar`. The embeddable CPython that bootstrap builds
    # ships a `._pth` that blocks the automatic sys.path entries, so `-m` cannot find a
    # package in the working directory -- it fails with "No module named sidecar" no matter
    # what cwd is. `__main__.py` puts its own directory on sys.path itself.
    entry = os.path.join(assist_root, "sidecar", "__main__.py")
    if not os.path.isfile(entry):
        raise SidecarError("sidecar not found at %s" % entry)
    subprocess.Popen(
        [python_exe, entry, "--port", str(int(port)),
         "--portfile", pf, "--parent", str(os.getpid())],
        cwd=assist_root, stdout=log, stderr=log, creationflags=creationflags)

    deadline = time.time() + timeout
    while time.time() < deadline:
        h = health(assist_root)
        if h is not None:
            return read_portfile(assist_root), h
        time.sleep(0.4)
    raise SidecarError("sidecar did not come up in %.0fs -- see logs/sidecar.log" % timeout)


def start_seed(assist_root, clip_info, params):
    info = read_portfile(assist_root)
    if info is None:
        raise SidecarError("sidecar is not running")
    return _request(info, "/jobs/seed", {"clip": clip_info, "params": params},
                    timeout=30.0)


def start_motion(assist_root, clip_info, params=None, seeds=None):
    info = read_portfile(assist_root)
    if info is None:
        raise SidecarError("sidecar is not running")
    return _request(info, "/jobs/motion",
                    {"clip": clip_info, "params": params or {}, "seeds": seeds or []},
                    timeout=30.0)


def start_hold(assist_root, clip_info, reqs, params=None):
    info = read_portfile(assist_root)
    if info is None:
        raise SidecarError("sidecar is not running")
    return _request(info, "/jobs/hold",
                    {"clip": clip_info, "requests": reqs, "params": params or {}},
                    timeout=30.0)


def poll(assist_root, job_id):
    info = read_portfile(assist_root)
    if info is None:
        raise SidecarError("sidecar is not running")
    return _request(info, "/jobs/%s" % job_id, timeout=10.0)


def cancel(assist_root, job_id):
    info = read_portfile(assist_root)
    if info is None:
        return
    try:
        _request(info, "/jobs/%s/cancel" % job_id, {}, timeout=5.0)
    except SidecarError:
        pass


def shutdown(assist_root):
    info = read_portfile(assist_root)
    if info is None:
        return
    try:
        _request(info, "/shutdown", {}, timeout=5.0)
    except SidecarError:
        pass


def start_reacquire(assist_root, clip_info, requests, params=None):
    info = read_portfile(assist_root)
    if info is None:
        raise SidecarError("sidecar is not running")
    return _request(info, "/jobs/reacquire",
                    {"clip": clip_info, "requests": requests, "params": params or {}},
                    timeout=30.0)


def start_patcheck(assist_root, clip_info, requests, params=None):
    """Ask why a pattern box changed size. CPU only -- no model is loaded for this."""
    info = read_portfile(assist_root)
    if info is None:
        raise SidecarError("sidecar is not running")
    return _request(info, "/jobs/patcheck",
                    {"clip": clip_info, "requests": requests, "params": params or {}},
                    timeout=30.0)


def start_leash(assist_root, clip_info, request, params=None):
    """A guide path for one track's whole span, with a trust verdict from round-trip closure.

    One track per call, matching the sidecar: the verdict describes THAT path, and batching
    would average a trustworthy guide with an untrustworthy one into a number true of
    neither.
    """
    info = read_portfile(assist_root)
    if info is None:
        raise SidecarError("sidecar is not running")
    return _request(info, "/jobs/leash",
                    {"clip": clip_info, "request": request, "params": params or {}},
                    timeout=30.0)


def start_report(assist_root, clip_info, tracks, params=None):
    """Ask whether a set of tracks will solve. CPU only -- no plate is read, no model loaded."""
    info = read_portfile(assist_root)
    if info is None:
        raise SidecarError("sidecar is not running")
    return _request(info, "/jobs/report",
                    {"clip": clip_info, "tracks": tracks, "params": params or {}},
                    timeout=60.0)


def start_pin(assist_root, clip_info, requests, params=None):
    """Register every frame of each track against the artist's own pattern box.

    Positions only -- the sidecar never proposes deleting a frame, so a pin can be applied
    without reviewing it.
    """
    info = read_portfile(assist_root)
    if info is None:
        raise SidecarError("sidecar is not running")
    return _request(info, "/jobs/pin",
                    {"clip": clip_info, "requests": requests, "params": params or {}},
                    timeout=30.0)


def start_ctrack(assist_root, clip_info, requests, params=None):
    """Track seeds with CoTracker as the primary engine, pinned to the artist's pattern."""
    info = read_portfile(assist_root)
    if info is None:
        raise SidecarError("sidecar is not running")
    return _request(info, "/jobs/ctrack",
                    {"clip": clip_info, "requests": requests, "params": params or {}},
                    timeout=30.0)
