# -*- coding: utf-8 -*-
"""Ground-truth deviation: manual artist tracks vs tracker (ISOLATED experiment).

Seeds the tracker at each MANUAL track's start point, then measures per-frame
euclidean deviation (native px) against the artist's actual path. Compares:

  BL   baseline whole-frame TAPNext (current GPU fallback engine)
  MT   guided moving-tile (native crop follows point)
  CB   winning combo = guided moving-tile + NCC sub-pixel refine (sharpest anchor)

This is the decider for wiring: it shows, in real pixels vs a human track, how much
each layer actually helps on a 4K plate.

Plate is a 4K image sequence; frames are streamed native (no full-RAM 4.4GB decode).
"""
from __future__ import annotations
import os, sys, math, json, argparse, importlib.util
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "training", "extract"))
from _common import parse_tracks_txt  # noqa: E402


def load_helpers():
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    def by_path(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    eng = by_path("btr_tapnext_engine", os.path.join(_REPO, "app", "tapnext_engine.py"))
    mt = by_path("btr_mt", os.path.join(_HERE, "track_moving_tile.py"))
    rp = by_path("btr_rp", os.path.join(_HERE, "refine_pipeline.py"))
    return eng.TapNextEngine, mt, rp


def load_seq(dir_path):
    files = sorted(f for f in os.listdir(dir_path) if f.lower().endswith((".jpg", ".png", ".jpeg")))
    paths = [os.path.join(dir_path, f) for f in files]
    h, w = cv2.imread(paths[0]).shape[:2]
    return paths, h, w


def decode_seq(paths):
    T = len(paths)
    h, w = cv2.imread(paths[0]).shape[:2]
    frames = np.empty((T, h, w, 3), np.uint8)
    for t, p in enumerate(paths):
        frames[t] = cv2.imread(p)
    return frames


def to_tracks_dict(A, V, names):
    """arrays (T,N,2)/(T,N) -> {name: {f,x,y}} using frame index+1."""
    out = {}
    for i, nm in enumerate(names):
        m = V[:, i] & ~np.isnan(A[:, i, 0])
        if m.sum() < 2:
            continue
        out[nm] = {"f": (np.where(m)[0] + 1).astype(int), "x": A[m, i, 0], "y": A[m, i, 1]}
    return out


def deviation(pred, manual):
    """pred/manual dict name->list[(f,x,y)] or {f,x,y}. Returns per-track + overall stats."""
    per = {}
    all_err = []
    for nm, mtrk in manual.items():
        if nm not in pred:
            per[nm] = None
            continue
        mm = {int(f): (x, y) for (f, x, y) in mtrk}
        pf, px, py = pred[nm]["f"], pred[nm]["x"], pred[nm]["y"]
        errs = []
        for k in range(len(pf)):
            f = int(pf[k])
            if f in mm:
                errs.append(math.hypot(px[k] - mm[f][0], py[k] - mm[f][1]))
        if errs:
            errs = np.array(errs)
            per[nm] = dict(n=len(errs), mean=float(errs.mean()), med=float(np.median(errs)),
                           p90=float(np.percentile(errs, 90)), mx=float(errs.max()))
            all_err.extend(errs.tolist())
        else:
            per[nm] = None
    ae = np.array(all_err) if all_err else np.array([0.0])
    overall = dict(mean=float(ae.mean()), med=float(np.median(ae)),
                   p90=float(np.percentile(ae, 90)), mx=float(ae.max()), n=len(all_err))
    return per, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default=os.path.join(_REPO, "training", "Assets for training", "Plates", "Shot_01"))
    ap.add_argument("--txt", default=os.path.join(_REPO, "training", "Assets for training", "Tracks", "Tracks_Shot_01.txt"))
    ap.add_argument("--out", default=os.path.join(_HERE, "out", "gt"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    Engine, mt, rp = load_helpers()
    paths, H, W = load_seq(args.plates)
    T = len(paths)
    print(f"[data] {T} frames {W}x{H}", flush=True)

    _, manual_raw = parse_tracks_txt(args.txt)
    # 3DE Y-up bottom-left -> image coords: y_img = (H-1) - y_txt ; x unchanged
    manual = {}
    for nm, pts in manual_raw.items():
        manual[nm] = [(int(f), float(x), float((H - 1) - y)) for (f, x, y) in pts]
    names = sorted(manual.keys())
    seeds = np.array([[manual[nm][0][1], manual[nm][0][2]] for nm in names], np.float32)
    print(f"[manual] {len(names)} tracks seeded at frame 1", flush=True)

    print("[decode] loading 4K sequence to RAM ...", flush=True)
    frames = decode_seq(paths)
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], 0)

    print("[engine] loading ...", flush=True)
    engine = Engine(tool_root=_REPO, device="cuda")

    # BL baseline whole-frame
    q = np.zeros((1, len(names), 3), np.float32); q[0, :, 1] = seeds[:, 0]; q[0, :, 2] = seeds[:, 1]
    print("[BL] whole-frame ...", flush=True)
    bl, blv = engine.track_queries(frames, q)
    # MT guided moving-tile (baseline guides itself)
    print("[MT] guided moving-tile ...", flush=True)
    gd, gdv = mt.moving_tile_track_guided(engine, frames, seeds, bl, blv)
    # BN = baseline (whole-frame) + NCC refine  == what the bot ALREADY does
    print("[BN] baseline + NCC refine ...", flush=True)
    bn_tracks = rp.refine_all(rp.arrays_to_tracks(bl, blv), grays, half=15, search=28,
                              motion_key="translation", anchor="sharpest", min_len=6)
    # CB combo = MT + NCC refine (translation, sharpest anchor)
    print("[CB] NCC sharpest-anchor refine ...", flush=True)
    mt_tracks = rp.arrays_to_tracks(gd, gdv)
    cb_tracks = rp.refine_all(mt_tracks, grays, half=15, search=28,
                              motion_key="translation", anchor="sharpest", min_len=6)

    bl_d = to_tracks_dict(bl, blv, names)
    mt_d = to_tracks_dict(gd, gdv, names)
    bn_d = {names[t["id"]]: {"f": t["f"], "x": t["x"], "y": t["y"]} for t in bn_tracks}
    cb_d = {names[t["id"]]: {"f": t["f"], "x": t["x"], "y": t["y"]} for t in cb_tracks}

    res = {}
    for tag, pred in (("BL", bl_d), ("MT", mt_d), ("BN", bn_d), ("CB", cb_d)):
        per, ov = deviation(pred, manual)
        res[tag] = dict(per=per, overall=ov)
        print(f"[{tag}] overall mean={ov['mean']:.2f}px med={ov['med']:.2f} "
              f"p90={ov['p90']:.2f} max={ov['mx']:.2f} (n={ov['n']}) tracks={sum(1 for v in per.values() if v)}",
              flush=True)

    # per-track table
    print("\n==== PER-TRACK MEAN DEVIATION (px vs manual) ====", flush=True)
    print(f"{'trk':>4} {'BL':>8} {'MT':>8} {'BN':>8} {'CB':>8}", flush=True)
    for nm in names:
        def g(tag):
            v = res[tag]["per"].get(nm)
            return f"{v['mean']:.2f}" if v else "lost"
        print(f"{nm:>4} {g('BL'):>8} {g('MT'):>8} {g('BN'):>8} {g('CB'):>8}", flush=True)

    json.dump(res, open(os.path.join(args.out, "deviation.json"), "w"), indent=2)

    # overlay video (downscaled to 1080p for viewing): manual red, BL blue, CB green
    scale = 1920.0 / W
    ow, oh = int(W * scale), int(H * scale)
    vw = cv2.VideoWriter(os.path.join(args.out, "Shot_01_gt_overlay.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (ow, oh))
    man_by_f = {nm: {int(f): (x, y) for (f, x, y) in manual[nm]} for nm in names}
    def draw(img, d, nm, f, col):
        if nm in d:
            k = np.where(d[nm]["f"] == f)[0]
            if len(k):
                cv2.drawMarker(img, (int(d[nm]["x"][k[0]] * scale), int(d[nm]["y"][k[0]] * scale)),
                               col, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
    for t in range(T):
        img = cv2.resize(frames[t], (ow, oh), interpolation=cv2.INTER_AREA)
        f = t + 1
        for nm in names:
            if f in man_by_f[nm]:
                x, y = man_by_f[nm][f]
                cv2.circle(img, (int(x * scale), int(y * scale)), 5, (0, 0, 255), 1, cv2.LINE_AA)
            draw(img, bl_d, nm, f, (255, 130, 0))
            draw(img, cb_d, nm, f, (0, 230, 0))
        cv2.putText(img, "RED(o)=manual  BLUE=baseline  GREEN=combo   f%d/%d" % (f, T),
                    (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
