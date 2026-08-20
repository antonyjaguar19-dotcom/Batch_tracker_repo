"""The sidecar: a localhost HTTP server that owns the GPU, so Blender never has to.

Why a separate process at all:
  * torch + CUDA is ~2.5 GB and Blender 5.2's Python is 3.13, not the 3.11 the bot runs on
  * a CUDA OOM kills this process, not the artist's session
  * every licence boundary lands on a process boundary -- the addon is GPL and imports bpy,
    the models are Apache/BSD and never share an interpreter with it

Stdlib only. `http.server` is enough for seven endpoints, and in self-contained mode every
avoided dependency is a smaller bootstrap.

Security: bound to 127.0.0.1, port 0 (the OS picks), and every request must carry the token
written into `logs/sidecar.json`. Without that, any local process -- including a web page --
could drive a service that reads client footage off a studio share.
"""

import json
import os
import secrets
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
VERSION = "0.1.0"

TOKEN = ""
JOBS = {}
JOBS_LOCK = threading.Lock()
STARTED = time.time()


class Job:
    """One unit of work. Only one runs at a time -- there is one A4000."""

    def __init__(self, kind):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.state = "queued"
        self.message = ""
        self.stage = ""
        self.result = None
        self.error = None
        self.cancel = threading.Event()
        self.started = time.time()
        self.log = []

    def say(self, msg):
        self.stage = str(msg)
        self.log.append("%.1fs  %s" % (time.time() - self.started, msg))
        del self.log[:-200]
        print("[job %s] %s" % (self.id, msg), flush=True)

    def public(self):
        return {"id": self.id, "kind": self.kind, "state": self.state,
                "stage": self.stage, "message": self.message,
                "seconds": round(time.time() - self.started, 1),
                "result": self.result, "error": self.error,
                "log": self.log[-12:]}


def busy_job():
    with JOBS_LOCK:
        for j in JOBS.values():
            if j.state in ("queued", "running"):
                return j
    return None


def run_job(job, fn):
    def body():
        job.state = "running"
        try:
            job.result = fn(job)
            job.state = "cancelled" if job.cancel.is_set() else "done"
        except Exception as exc:                     # noqa: BLE001
            job.state = "error"
            # The traceback goes to the log; the addon gets one sentence. A traceback in a
            # Blender popup is not information an artist can act on.
            job.error = {"code": type(exc).__name__, "message": str(exc)[:400],
                         "log": os.path.join(ASSIST, "logs", "sidecar.log")}
            job.say("ERROR %s: %s" % (type(exc).__name__, exc))
            traceback.print_exc()
    t = threading.Thread(target=body, daemon=True)
    t.start()
    return job


# ------------------------------------------------------------------ job bodies

def job_seed(payload):
    def fn(job):
        import sys
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import autoseed                                              # noqa: PLC0415
        clip = payload.get("clip") or {}
        params = payload.get("params") or {}
        plate = clip.get("path", "")
        if not plate:
            raise ValueError("no clip path")
        if not os.path.exists(plate):
            # A relative filepath from Blender (`//plate/...`) that the addon failed to
            # resolve lands here. Failing on the argument beats failing on a decode.
            raise FileNotFoundError("plate not found: %s" % plate)
        out_dir = os.path.join(ASSIST, "logs", "seed", job.id)
        data = autoseed.build_seeds(
            plate, out_dir,
            target=int(params.get("target", 150)),
            spacing_px=float(params.get("spacing_px", 15)),
            max_tracks=int(params.get("max_tracks", 1200)),
            reject_movers=bool(params.get("reject_movers", False)),
            on_status=job.say)
        w, h = int(data["width"]), int(data["height"])
        cw, ch = int(clip.get("width", w)), int(clip.get("height", h))
        if (cw, ch) != (w, h):
            # Hard stop. With a different size the normalised coordinates still "work" and
            # every tracker starts on the wrong feature, which looks like bad tracking
            # rather than like a bug.
            raise RuntimeError("Blender reports %dx%d but the plate reads %dx%d -- "
                               "different plate, or a proxy is on" % (cw, ch, w, h))
        return data
    return fn


# ------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):                  # quiet; jobs do their own logging
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        if self.headers.get("X-BTR-Token", "") == TOKEN:
            return True
        self._send(403, {"error": {"code": "BAD_TOKEN",
                                   "message": "wrong or missing token"}})
        return False

    def _payload(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, health())
        if not self._auth():
            return None
        if self.path.startswith("/jobs/"):
            jid = self.path.split("/")[2]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if job is None:
                return self._send(404, {"error": {"code": "NO_JOB",
                                                  "message": "unknown job %s" % jid}})
            return self._send(200, job.public())
        return self._send(404, {"error": {"code": "NO_ROUTE", "message": self.path}})

    def do_POST(self):
        if not self._auth():
            return None
        try:
            payload = self._payload()
        except ValueError as exc:
            return self._send(400, {"error": {"code": "BAD_JSON", "message": str(exc)}})

        if self.path == "/shutdown":
            self._send(200, {"ok": True})
            threading.Thread(target=lambda: (time.sleep(0.2),
                                             os._exit(0)), daemon=True).start()
            return None

        if self.path == "/jobs/seed":
            b = busy_job()
            if b is not None:
                return self._send(409, {"error": {
                    "code": "BUSY",
                    "message": "a %s job is already running (%s)" % (b.kind, b.stage)}})
            job = Job("seed")
            with JOBS_LOCK:
                JOBS[job.id] = job
            run_job(job, job_seed(payload))
            return self._send(200, job.public())

        if self.path.startswith("/jobs/") and self.path.endswith("/cancel"):
            jid = self.path.split("/")[2]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if job is None:
                return self._send(404, {"error": {"code": "NO_JOB", "message": jid}})
            job.cancel.set()
            job.say("cancel requested")
            return self._send(200, {"ok": True})

        return self._send(404, {"error": {"code": "NO_ROUTE", "message": self.path}})


def health():
    """Everything the addon needs to decide whether to offer a button, in one call."""
    out = {"ok": True, "version": VERSION, "uptime_s": round(time.time() - STARTED, 1),
           "pid": os.getpid()}
    try:
        import sys
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import repo                                                  # noqa: PLC0415
        p = repo.paths()
        out["repo_root"] = p["repo_root"]
        out["tapnext_ckpt"] = ("present" if os.path.isfile(p["tapnext_ckpt"] or "")
                               else "MISSING")
        try:
            repo.require_repo()
            out["repo"] = "ok"
        except RuntimeError as exc:
            out["repo"] = str(exc)[:300]
            out["ok"] = False
    except Exception as exc:                         # noqa: BLE001
        out["repo"] = "%s: %s" % (type(exc).__name__, exc)
        out["ok"] = False
    try:
        import torch                                                 # noqa: PLC0415
        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.is_available()
        if out["cuda"]:
            free, total = torch.cuda.mem_get_info()
            out["device"] = torch.cuda.get_device_name(0)
            out["vram_free_mb"] = int(free / 1e6)
            out["vram_total_mb"] = int(total / 1e6)
    except Exception as exc:                         # noqa: BLE001
        out["torch"] = "%s: %s" % (type(exc).__name__, exc)
        out["cuda"] = False
        out["ok"] = False
    b = busy_job()
    out["busy"] = b.public() if b else None
    return out


def serve(port=0, portfile=None, token=None, parent_pid=0):
    global TOKEN
    TOKEN = token or secrets.token_hex(16)
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    actual = srv.server_address[1]
    info = {"port": actual, "token": TOKEN, "pid": os.getpid()}
    pf = portfile or os.path.join(ASSIST, "logs", "sidecar.json")
    os.makedirs(os.path.dirname(pf), exist_ok=True)
    with open(pf, "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    print("[sidecar] listening on 127.0.0.1:%d  (portfile %s)" % (actual, pf), flush=True)

    if parent_pid:
        # If Blender goes away, so does this. Otherwise a crashed session leaves a process
        # holding 2.5 GB of VRAM that nothing will ever ask to stop.
        def watch():
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x00100000, False, int(parent_pid))
            if h:
                k.WaitForSingleObject(h, 0xFFFFFFFF)
                print("[sidecar] parent %s exited -- shutting down" % parent_pid,
                      flush=True)
                os._exit(0)
        threading.Thread(target=watch, daemon=True).start()

    srv.serve_forever()
