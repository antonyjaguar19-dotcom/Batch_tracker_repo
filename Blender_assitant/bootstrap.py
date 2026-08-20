"""Build everything the sidecar needs, inside `Blender_assitant/`, and prove it works.

Two modes, and the difference matters:

  --self-contained (default)
      Downloads its own CPython 3.11.9 embeddable + torch cu121 into
      `Blender_assitant/runtime/`, and TAPNext++ code and weights into `vendor/` and
      `weights/`. ~2.7 GB, once. The folder can then be copied to another workstation.

  --reuse-repo
      Points at the repo's existing `runtime/python311` and `pipeline/tapnext-main`.
      Zero download on this machine, but nothing is portable.

**Self-contained refers to the RUNTIME, not to the repo.** The auto-seed sidecar imports
`app.tracker_core`, `app.track_meta` and friends from batch_tracker itself, and vendoring
copies of those would recreate the two-copies problem this project already refuses
elsewhere. So the repo is a hard dependency of auto-seed either way. 3DE import/export in
the addon needs none of this and works on a bare Blender.

Everything downloaded lands under this folder. Nothing is written outside it, and nothing
in the existing bot is modified.

    bootstrap.bat                       self-contained, the default
    bootstrap.bat --reuse-repo          use the bot's runtime instead
    bootstrap.bat --check               report only, download nothing
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

PY_VER = "3.11.9"
PY_ZIP_URL = ("https://www.python.org/ftp/python/%s/python-%s-embed-amd64.zip"
              % (PY_VER, PY_VER))
GETPIP_URL = "https://bootstrap.pypa.io/get-pip.py"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
TORCH_PINS = ["torch==2.5.1+cu121", "torchvision==0.20.1+cu121"]
# Pinned to what the bot runs today (requirements.txt:49-53). A sidecar on a different
# torch than the code it imports is a debugging session nobody asked for.
OTHER_PINS = [
    "numpy==2.4.4",
    "opencv-python-headless==4.13.0.92",
    "scipy==1.17.1",
    "pillow==12.2.0",
    "einops==0.8.2",          # vendored tapnet/tapnext_torch.py imports it
    "timm==1.0.27",
    "imageio-ffmpeg==0.6.0",  # frame extraction when the plate is a movie
    "pandas==3.0.3",
]
# Pinned to the bot's own requirements.txt, because the sidecar imports the bot's code and
# a version drift lands inside an import it cannot see fail cleanly. Left unpinned, pip
# installed opencv 5.0.0 -- a major version ahead of the 4.x that `app.*` is written
# against.
#
# Deliberately NOT installed: ultralytics (AGPL-3.0) and transformers/accelerate. They are
# only needed for SAM 3 masking and Qwen analysis, both out of scope here -- which is
# exactly what keeps this addon commercially clean.
TAPNET_REPO = "https://github.com/google-deepmind/tapnet"
TAPNEXT_CKPT_URL = ("https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt")
TAPNEXT_CKPT_NAME = "tapnextpp_ckpt.pt"

DEFAULT_BLENDER = (r"C:\Users\jefrin\Downloads\blender-5.2.0-windows-x64"
                   r"\blender-5.2.0-windows-x64\blender.exe")

ROWS = []


def row(name, ok, detail=""):
    ROWS.append((name, bool(ok), detail))
    print("  %-28s %s  %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)
    return ok


def run(cmd, **kw):
    print("  $ %s" % " ".join(str(c) for c in cmd[:6]), flush=True)
    return subprocess.run(cmd, **kw)


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print("\r    %s  %5.1f%%  (%.0f/%.0f MB)"
                      % (os.path.basename(dest), pct, done / 1e6, total / 1e6),
                      end="", flush=True)
    print()
    os.replace(tmp, dest)
    return dest


# ------------------------------------------------------------------ interpreter

def build_runtime():
    """Portable CPython 3.11.9 embeddable + pip, mirroring setup_bot.bat:55-71."""
    py_dir = os.path.join(HERE, "runtime", "python311")
    py = os.path.join(py_dir, "python.exe")
    if os.path.isfile(py):
        print("  runtime python already present")
        return py

    os.makedirs(py_dir, exist_ok=True)
    zip_path = os.path.join(HERE, "runtime", "python-embed.zip")
    print("  downloading CPython %s embeddable ..." % PY_VER)
    download(PY_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(py_dir)
    os.remove(zip_path)

    # The embeddable build ships `._pth` with `import site` commented out, which blocks
    # both site-packages and pip. Uncommenting it is what makes this a usable interpreter.
    for fn in os.listdir(py_dir):
        if fn.endswith("._pth"):
            p = os.path.join(py_dir, fn)
            with open(p, encoding="utf-8") as fh:
                txt = fh.read()
            txt = txt.replace("#import site", "import site")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(txt)

    getpip = os.path.join(HERE, "runtime", "get-pip.py")
    download(GETPIP_URL, getpip)
    run([py, getpip, "--no-warn-script-location"], check=True)
    os.remove(getpip)
    return py


def pip_install(py, args):
    env = dict(os.environ)
    env["PIP_CACHE_DIR"] = os.path.join(HERE, "runtime", "pipcache")
    env["TMP"] = env["TEMP"] = os.path.join(HERE, "runtime", "tmp")
    os.makedirs(env["TMP"], exist_ok=True)
    return run([py, "-m", "pip", "install", "--no-warn-script-location"] + args,
               env=env).returncode == 0


def probe(py):
    """What does this interpreter actually have? Versions, not assumptions."""
    code = (
        "import json,sys\n"
        "o={'python':sys.version.split()[0]}\n"
        "for m in ('torch','torchvision','numpy','cv2','scipy'):\n"
        "    try:\n"
        "        mod=__import__(m); o[m]=getattr(mod,'__version__','?')\n"
        "    except Exception as e: o[m]='MISSING (%s)'%type(e).__name__\n"
        "try:\n"
        "    import torch; o['cuda']=torch.cuda.is_available()\n"
        "    o['device']=torch.cuda.get_device_name(0) if o['cuda'] else None\n"
        "except Exception: o['cuda']=False; o['device']=None\n"
        "print(json.dumps(o))\n"
    )
    p = subprocess.run([py, "-c", code], capture_output=True, text=True)
    try:
        return json.loads((p.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": (p.stderr or p.stdout or "no output").strip()[:400]}


# ------------------------------------------------------------------ tapnext

def get_tapnext(self_contained):
    """Code (Apache-2.0) and the 256 checkpoint. `BTR_TAPNEXT_CKPT` always wins."""
    env_ckpt = os.environ.get("BTR_TAPNEXT_CKPT", "")
    repo_code = os.path.join(REPO, "pipeline", "tapnext-main")
    repo_ckpt = os.path.join(repo_code, "checkpoints", TAPNEXT_CKPT_NAME)

    if not self_contained:
        return repo_code, (env_ckpt if os.path.isfile(env_ckpt) else repo_ckpt)

    code = os.path.join(HERE, "vendor", "tapnet")
    if not os.path.isdir(os.path.join(code, ".git")):
        os.makedirs(os.path.dirname(code), exist_ok=True)
        if os.path.isdir(repo_code):
            print("  copying tapnet from the repo checkout ...")
            shutil.copytree(repo_code, code,
                            ignore=shutil.ignore_patterns("checkpoints", "__pycache__"),
                            dirs_exist_ok=True)
        else:
            run(["git", "clone", "--depth", "1", TAPNET_REPO, code])

    ckpt = os.path.join(HERE, "weights", TAPNEXT_CKPT_NAME)
    if not os.path.isfile(ckpt):
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        src = env_ckpt if os.path.isfile(env_ckpt) else repo_ckpt
        if os.path.isfile(src):
            # Same bytes either way, and 200 MB off the local disk beats 200 MB off the
            # internet. Downloading is only for a machine that has neither.
            print("  copying %s from the repo ..." % TAPNEXT_CKPT_NAME)
            shutil.copy2(src, ckpt)
        else:
            print("  downloading %s ..." % TAPNEXT_CKPT_NAME)
            download(TAPNEXT_CKPT_URL, ckpt)
    return code, ckpt


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--self-contained", action="store_true", default=True)
    g.add_argument("--reuse-repo", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="report what is present; download nothing")
    ap.add_argument("--blender", default=os.environ.get("BTR_BLENDER_EXE", DEFAULT_BLENDER))
    args = ap.parse_args()
    self_contained = not args.reuse_repo

    print("Blender_assitant bootstrap")
    print("  mode        : %s" % ("self-contained" if self_contained else "reuse-repo"))
    print("  folder      : %s" % HERE)
    print("  repo        : %s" % REPO)
    print()

    # ---- interpreter
    if self_contained:
        py = os.path.join(HERE, "runtime", "python311", "python.exe")
        if not os.path.isfile(py) and not args.check:
            py = build_runtime()
    else:
        py = os.path.join(REPO, "runtime", "python311", "python.exe")
    row("python 3.11", os.path.isfile(py), py if os.path.isfile(py) else "not built yet")

    info = probe(py) if os.path.isfile(py) else {"error": "no interpreter"}
    need_torch = "torch" not in info or str(info.get("torch", "")).startswith("MISSING")
    if need_torch and self_contained and not args.check and os.path.isfile(py):
        print("  installing torch cu121 (~2.5 GB, once) ...")
        pip_install(py, ["--upgrade", "pip"])
        pip_install(py, TORCH_PINS + ["--index-url", TORCH_INDEX])
        pip_install(py, OTHER_PINS)
        info = probe(py)

    row("torch + CUDA", info.get("cuda") is True,
        "%s on %s" % (info.get("torch"), info.get("device")) if info.get("cuda")
        else str(info.get("torch") or info.get("error", ""))[:80])
    row("numpy", not str(info.get("numpy", "MISSING")).startswith("MISSING"),
        str(info.get("numpy", "")))
    row("opencv", not str(info.get("cv2", "MISSING")).startswith("MISSING"),
        str(info.get("cv2", "")))

    # ---- tapnext
    if args.check:
        code = (os.path.join(HERE, "vendor", "tapnet") if self_contained
                else os.path.join(REPO, "pipeline", "tapnext-main"))
        ckpt = (os.path.join(HERE, "weights", TAPNEXT_CKPT_NAME) if self_contained
                else os.path.join(REPO, "pipeline", "tapnext-main", "checkpoints",
                                  TAPNEXT_CKPT_NAME))
    else:
        code, ckpt = get_tapnext(self_contained)
    row("TAPNext++ code", os.path.isdir(code), code)
    row("TAPNext++ weights", os.path.isfile(ckpt),
        "%.0f MB" % (os.path.getsize(ckpt) / 1e6) if os.path.isfile(ckpt) else ckpt)

    # ---- the repo, which auto-seed needs whatever the mode
    row("repo importable (app/)", os.path.isfile(os.path.join(REPO, "app", "tracker_core.py")),
        "auto-seed imports app.tracker_core, app.track_meta")

    row("blender.exe", os.path.isfile(args.blender), args.blender)

    # ---- write what the addon reads
    paths = {
        "assist_root": HERE,
        "repo_root": REPO,
        "python_exe": py,
        "tapnext_code": code,
        "tapnext_ckpt": ckpt,
        "blender_exe": args.blender,
        "mode": "self-contained" if self_contained else "reuse-repo",
        "probe": info,
    }
    os.makedirs(os.path.join(HERE, "config"), exist_ok=True)
    with open(os.path.join(HERE, "config", "paths.json"), "w", encoding="utf-8") as fh:
        json.dump(paths, fh, indent=2)
    row("config/paths.json", True, "written")

    # ---- addon zip
    if not args.check:
        p = run([sys.executable, os.path.join(HERE, "build.py"),
                 "--blender", args.blender])
        row("addon zip built+validated", p.returncode == 0)

    ok = all(r[1] for r in ROWS)
    print()
    print("=" * 64)
    print("BOOTSTRAP: %s" % ("PASS" if ok else "FAIL -- see the rows above"))
    if ok:
        print()
        print("Install the addon:")
        print("  Blender > Edit > Preferences > Add-ons > Install from Disk")
        print("  %s" % os.path.join(HERE, "dist"))
        print()
        print("Smoke test:")
        print('  "%s" --background --factory-startup -noaudio ^' % args.blender)
        print('     --python "%s" -- ^' % os.path.join(HERE, "tests", "smoke_addon.py"))
        print('     --plate <frames folder> --tracks <a 3DE .txt>')
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
