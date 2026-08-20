"""Package `addon/btr_assist/` as a Blender extension, and refuse to ship an invalid zip.

Follows `experiments/blender_track/build_extensions.py`, with one change: that script zips
a single `__init__.py` because both of its addons are one file. This one is a package, so
the staging step walks the directory.

Kept from it, because each was learned the hard way:
  * the version is regexed out of `bl_info`, so the manifest and the source cannot disagree
  * the tagline guard -- Blender rejects one over 64 chars or ending in punctuation, and
    says so only at validation time
  * validation via `blender --command extension validate`; a failing zip is not shipped
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "addon", "btr_assist")
DIST = os.path.join(HERE, "dist")
PKG_ID = "btr_assist"

NAME = "Tracking Assistant"
TAGLINE = "AI-assisted 2D tracking: auto-seed, repair and 3DE transfer"
TAGS = ["Tracking", "Import-Export"]

DEFAULT_BLENDER = (r"C:\Users\jefrin\Downloads\blender-5.2.0-windows-x64"
                   r"\blender-5.2.0-windows-x64\blender.exe")

MANIFEST = """schema_version = "1.0.0"

id = "{pkg_id}"
version = "{version}"
name = "{name}"
tagline = "{tagline}"
maintainer = "batch_tracker"
type = "add-on"

tags = [{tags}]

blender_version_min = "4.2.0"
license = ["SPDX:GPL-2.0-or-later"]
"""


def version_from_source(path):
    """Read the version out of bl_info, so the manifest and the file cannot disagree."""
    with open(path, encoding="utf-8") as fh:
        m = re.search(r'"version"\s*:\s*\((\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?\)', fh.read())
    return "%s.%s.%s" % (m.group(1), m.group(2), m.group(3) or "0") if m else "0.1.0"


def write_paths_json(stage):
    """Bake the repo-side paths into the package.

    Once Blender copies the zip into its extensions folder, walking up from __file__ lands
    in Blender's config rather than the repo, so the addon cannot find Blender_assitant on
    its own. Bootstrap writes config/paths.json; build copies it in.
    """
    src = os.path.join(HERE, "config", "paths.json")
    data = {"assist_root": HERE}
    try:
        with open(src, encoding="utf-8") as fh:
            data.update(json.load(fh))
    except (OSError, ValueError):
        pass                                  # not bootstrapped yet; 3DE still works
    data.setdefault("assist_root", HERE)
    with open(os.path.join(stage, "paths.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def build(exe):
    if len(TAGLINE) > 64 or TAGLINE[-1] in ".!?,":
        print("tagline is %d chars and ends %r -- Blender will reject it"
              % (len(TAGLINE), TAGLINE[-1]))
        return None
    if not os.path.isdir(SRC):
        print("missing %s" % SRC)
        return None

    version = version_from_source(os.path.join(SRC, "__init__.py"))
    stage = os.path.join(DIST, PKG_ID)
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)

    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith((".pyc", ".pyo")):
                continue
            rel = os.path.relpath(os.path.join(root, fn), SRC)
            dst = os.path.join(stage, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(root, fn), dst)

    write_paths_json(stage)
    with open(os.path.join(stage, "blender_manifest.toml"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(MANIFEST.format(pkg_id=PKG_ID, version=version, name=NAME,
                                 tagline=TAGLINE,
                                 tags=", ".join('"%s"' % t for t in TAGS)))

    zip_path = os.path.join(DIST, "%s-%s.zip" % (PKG_ID, version))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(stage):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, stage)
                z.write(full, os.path.join(PKG_ID, rel))
    print("built %s  (%.1f KB)" % (os.path.basename(zip_path),
                                   os.path.getsize(zip_path) / 1024))

    if not os.path.isfile(exe):
        print("[skip] Blender not found -- zip built but NOT validated")
        return zip_path
    p = subprocess.run([exe, "--command", "extension", "validate", zip_path],
                       capture_output=True, text=True)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    if p.returncode != 0:
        print("VALIDATION FAILED -- do not ship this zip\n%s" % out)
        return None
    print("validated by Blender")
    return zip_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=os.environ.get("BTR_BLENDER_EXE", DEFAULT_BLENDER))
    args = ap.parse_args()
    os.makedirs(DIST, exist_ok=True)
    return 0 if build(args.blender) else 1


if __name__ == "__main__":
    sys.exit(main())
