"""Score the re-acquire's candidates against a hand track, at full plate resolution.

The first reference in this project for an OCCLUSION. Everything before it was measured
against Blender's own output, a synthetic plate, or an algorithm sharing the same NCC bias.
A hand track through an occluder answers the one question none of those could: when the
feature comes back, does the loop put the marker on it?

    runtime\python311\python.exe tests\eval_reacquire.py --manual tests\<file>.txt \
        --plate D:\Jefrin\IN\SH006.mp4 --seed-frame 1
"""

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))


def read_3de(path):
    tok = open(path, encoding="utf-8", errors="ignore").read().split()
    i = 0
    n = int(tok[i]); i += 1
    out = []
    for _ in range(n):
        name = tok[i]; i += 1
        i += 1
        cnt = int(tok[i]); i += 1
        pts = []
        for _ in range(cnt):
            pts.append((int(tok[i]), float(tok[i + 1]), float(tok[i + 2]))); i += 3
        out.append((name, pts))
    return out


def call(port, token, path, payload=None, timeout=300.0):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 data=None if payload is None
                                 else json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          "X-BTR-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True)
    ap.add_argument("--plate", required=True)
    ap.add_argument("--seed-frame", type=int, default=1)
    ap.add_argument("--pattern", type=float, default=41.2)
    ap.add_argument("--search", type=float, default=113.1)
    ap.add_argument("--min-match", type=float, default=0.6622)
    ap.add_argument("--max-px", type=float, default=5.0,
                    help="how far from the hand track still counts as ON the feature. The "
                         "reference is a human clicking on 4K frames, so a couple of px is "
                         "its own noise; landing on the wrong feature is hundreds")
    a = ap.parse_args()

    info = json.load(open(os.path.join(ASSIST, "logs", "sidecar.json"), encoding="utf-8"))
    port, token = int(info["port"]), info["token"]
    h = call(port, token, "/health")
    W, H = None, None

    name, pts = read_3de(a.manual)[0]
    # The hand track is 3DE ASCII: y-UP. Everything the sidecar speaks is image px, y-DOWN.
    truth_up = {f: (x, y) for f, x, y in pts}
    frames = sorted(truth_up)
    runs = []
    for f in frames:
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    print("hand track %s: %d samples, runs %s" % (name, len(pts), runs))

    # The track is alive up to the end of the first run; that is where the loop must
    # hand over, and where the artist's own last good position is.
    last_good = runs[0][1]
    back_at = runs[1][0] if len(runs) > 1 else None
    print("occlusion: last good f%d, feature back at f%d" % (last_good, back_at))

    import subprocess
    probe = subprocess.run([sys.executable, "-c",
                            "import cv2,sys;c=cv2.VideoCapture(sys.argv[1]);"
                            "print(int(c.get(3)),int(c.get(4)),int(c.get(7)))", a.plate],
                           capture_output=True, text=True)
    W, H, NF = (int(v) for v in probe.stdout.split())
    print("plate %dx%d, %d frames" % (W, H, NF))

    def down(f):
        x, y = truth_up[f]
        return (x, H - y)

    sx, sy = down(a.seed_frame)
    lx, ly = down(last_good)
    reqs = [{"id": name, "query_frame": a.seed_frame, "query_x": sx, "query_y": sy,
             "last_good_frame": last_good, "last_good_x": lx, "last_good_y": ly,
             "gap": 3,
             "pattern": {"frame": a.seed_frame, "cx": sx, "cy": sy,
                         "w": a.pattern, "h": a.pattern},
             "last_box": {"frame": last_good, "cx": lx, "cy": ly,
                          "w": a.pattern, "h": a.pattern},
             "search_px": a.search}]
    r = call(port, token, "/jobs/reacquire",
             {"clip": {"path": os.path.abspath(a.plate), "width": W, "height": H,
                       "frames": NF},
              "requests": reqs,
              "params": {"frame_hi": NF, "verify_pattern": True,
                         "min_match": a.min_match}})
    jid = r["id"]
    while True:
        st = call(port, token, "/jobs/%s" % jid)
        if st["state"] not in ("queued", "running"):
            break
        time.sleep(0.5)
    if st["state"] != "done":
        print("FAILED:", st.get("error"))
        return 1

    res = st["result"]
    print()
    for res_r in res.get("resumes") or []:
        cands = res_r.get("candidates") or []
        chosen = (int(res_r["frame"]), float(res_r["x"]), float(res_r["y"]))
        print("the loop proposes f%d  (match %s)" % (chosen[0], res_r.get("match_score")))
        print()
        print("%-4s %-6s %-8s %-22s %s"
              % ("#", "frame", "score", "error vs hand track", "verdict"))
        rows = [{"frame": chosen[0], "x": chosen[1], "y": chosen[2],
                 "score": res_r.get("match_score")}]
        rows += [c for c in cands if int(c["frame"]) != chosen[0]]
        for i, c in enumerate(rows, 1):
            f = int(c["frame"])
            if f in truth_up:
                tx, ty = down(f)
                err = ((c["x"] - tx) ** 2 + (c["y"] - ty) ** 2) ** 0.5
                verdict = ("ON the feature" if err <= 5 else
                           "near (%.0f px)" % err if err <= 25 else "WRONG feature")
                print("%-4d f%-5d %-8.2f %-22.1f %s"
                      % (i, f, c["score"] or 0, err, verdict))
            else:
                print("%-4d f%-5d %-8.2f %-22s hand track has no sample here "
                      "(feature still hidden)" % (i, f, c["score"] or 0, "-"))
    for m in res.get("misses") or []:
        print("MISS %s: %s" % (m["id"], m.get("reason")))

    # ---- the gate ---------------------------------------------------------------
    # Two things have to hold, and they are different claims. The resume must land on a
    # frame the artist's hand track actually has -- landing inside the occlusion is the
    # failure this reference was brought in to catch -- and it must land ON the feature
    # there, not merely near it.
    failures = []
    got = res.get("resumes") or []
    if not got:
        failures.append("no resume proposed at all")
    for res_r in got:
        f = int(res_r["frame"])
        if f not in truth_up:
            failures.append("resumed at f%d, where the hand track has no sample -- that is "
                            "inside the occlusion" % f)
            continue
        tx, ty = down(f)
        err = ((float(res_r["x"]) - tx) ** 2 + (float(res_r["y"]) - ty) ** 2) ** 0.5
        if err > a.max_px:
            failures.append("resumed at f%d but %.1f px from the hand track (limit %.1f)"
                            % (f, err, a.max_px))
        else:
            print()
            print("resume landed f%d, %.1f px from the hand track" % (f, err))
    print()
    for m in failures:
        print("FAILURE: %s" % m)
    print("REACQUIRE vs HAND TRACK: %s" % ("FAIL" if failures else "PASS"))
    return 1 if failures else 0


sys.exit(main())
