"""Is the resume on the ARTIST'S feature, or merely on a trackable one?

CoTracker answers "where did this go". It does not answer "is that the same thing you
picked" -- and measured across three shots, a re-acquired point lands on the right feature
only 26-47 % of the time. A wrong one then tracks perfectly well, so survival proves
nothing. Everything downstream of that number (the muted resume, the Keep/Drop pass) exists
because the loop had no way to tell the two apart.

This module is that way. The artist's marker carries a pattern box, and the pixels inside it
at the marker's keyframe are exactly what Blender draws in the Track panel's preview -- the
feature, as set. Normalised cross-correlation of that patch against the plate around a
candidate resume gives a number for "is this the same feature":

  * `TM_CCOEFF_NORMED` subtracts the mean and divides by the norm, so an exposure or grain
    change between the seed frame and the resume does not read as a different feature. That
    is the same normalisation Blender's own tracker runs with (`use_normalization`) and the
    same family as the bot's `pattern_refine`.
  * The match runs at ORIGINAL plate resolution. CoTracker's 768 px guide only has to land
    inside a search box; deciding identity is a pixel question and gets full pixels.
  * The peak is refined to sub-pixel by a parabola through its three neighbours per axis,
    so a passing match also improves the plant rather than only judging it.

What it cannot do, stated so it is not discovered later: the reference is a fixed patch from
one frame, so a feature that has genuinely changed appearance -- rotated, scaled, lit
differently, half-occluded -- scores low even when the resume is correct. That is the price
of anchoring to what the artist set. The score is therefore REPORTED for every candidate,
not only compared against a threshold: a run of near-threshold misses means the threshold is
wrong for that plate, and the artist can see it rather than guess.
"""

import numpy as np


def _gray(img):
    """BGR uint8 -> float32 2-D. Defensive squeeze: in-process, with torch loaded, OpenCV
    hands back (H, W, 1) where standalone it returns (H, W) -- the trap already recorded in
    the SynthEyes mask sampler."""
    import cv2                                                        # noqa: PLC0415
    if img is None:
        return None
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] >= 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    a = np.squeeze(a)
    if a.ndim != 2:
        return None
    return a.astype(np.float32)


def reference_patch(plate, frame, cx, cy, pw, ph, min_side=7):
    """The artist's pattern, as pixels.

    `frame` is 1-based; `cx, cy` are plate pixels, y-DOWN (image space, the same convention
    the addon sends); `pw, ph` the pattern box size in pixels. Blender's pattern is a quad
    and may be sheared -- the addon sends its axis-aligned bounding box, which is what the
    preview effectively shows for the unrotated boxes this addon creates.

    Returns (patch, (ox, oy)) where (ox, oy) is the patch centre's offset from (cx, cy)
    after clamping to the image, or None if the box does not fit.
    """
    img = _gray(plate.frame(int(frame) - 1))
    if img is None:
        return None
    h, w = img.shape
    x0 = int(round(cx - pw / 2.0))
    y0 = int(round(cy - ph / 2.0))
    x1 = x0 + max(min_side, int(round(pw)))
    y1 = y0 + max(min_side, int(round(ph)))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if (x1 - x0) < min_side or (y1 - y0) < min_side:
        return None
    patch = img[y0:y1, x0:x1]
    # A patch with no contrast correlates with everything; TM_CCOEFF_NORMED divides by its
    # standard deviation and would return nan. Refuse instead of scoring noise.
    if float(patch.std()) < 1e-3:
        return None
    ox = (x0 + x1) / 2.0 - cx
    oy = (y0 + y1) / 2.0 - cy
    return patch, (ox, oy)


def _subpixel(score, px, py):
    """Parabola through the peak and its two neighbours, per axis. Returns (dx, dy) in
    pixels, clamped to +/-1 -- a fit that wants to move further than one cell means the
    surface is not peaked there and the integer position is the honest answer."""
    h, w = score.shape
    dx = dy = 0.0
    if 0 < px < w - 1:
        a, b, c = float(score[py, px - 1]), float(score[py, px]), float(score[py, px + 1])
        den = a - 2.0 * b + c
        if abs(den) > 1e-9:
            dx = 0.5 * (a - c) / den
    if 0 < py < h - 1:
        a, b, c = float(score[py - 1, px]), float(score[py, px]), float(score[py + 1, px])
        den = a - 2.0 * b + c
        if abs(den) > 1e-9:
            dy = 0.5 * (a - c) / den
    return max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy))


def match(plate, frame, patch, cx, cy, radius=32.0, offset=(0.0, 0.0)):
    """Find `patch` near (cx, cy) on `frame` (1-based). Plate pixels, y-down.

    Returns (x, y, score) -- the patch CENTRE's best position and its normalised
    correlation in [-1, 1] -- or None if the search window does not fit.
    """
    img = _gray(plate.frame(int(frame) - 1))
    if img is None:
        return None
    return match_in(img, patch, cx, cy, radius=radius, offset=offset)


def match_in(img, patch, cx, cy, radius=32.0, offset=(0.0, 0.0)):
    """`match` against an ALREADY DECODED grey frame.

    Split out because the reappearance sweep is frame-major: one decode, every track that is
    still looking tested against it. Per-track decoding re-read the same 4K frame once per
    track and made a full-window search cost what it did not need to.
    """
    import cv2                                                        # noqa: PLC0415
    if img is None:
        return None
    ih, iw = img.shape
    ph, pw = patch.shape
    # The window is the pattern plus the search radius on every side: the correlation
    # surface it produces is exactly (2*radius+1) wide, one score per candidate position.
    r = max(1, int(round(radius)))
    ccx, ccy = cx + offset[0], cy + offset[1]
    x0 = int(round(ccx - pw / 2.0)) - r
    y0 = int(round(ccy - ph / 2.0)) - r
    x1, y1 = x0 + pw + 2 * r, y0 + ph + 2 * r
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(iw, x1), min(ih, y1)
    if (x1 - x0) < pw or (y1 - y0) < ph:
        return None
    win = img[y0:y1, x0:x1]
    score = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(score).any():
        return None
    score = np.nan_to_num(score, nan=-1.0, posinf=-1.0, neginf=-1.0)
    py, px = np.unravel_index(int(np.argmax(score)), score.shape)
    dx, dy = _subpixel(score, int(px), int(py))
    # matchTemplate indexes the patch's top-left corner inside the window; the caller wants
    # its centre back in plate pixels, and the reference was taken `offset` off the marker.
    x = x0 + px + dx + pw / 2.0 - offset[0]
    y = y0 + py + dy + ph / 2.0 - offset[1]
    return float(x), float(y), float(score[py, px])


def best_candidate(plate, patch, offset, candidates, radius=32.0):
    """Score every candidate resume and return the best.

    `candidates` is [(frame, (x, y)), ...] -- CoTracker's guesses, in the order it thinks
    the feature came back. The FIRST visible frame is not automatically the best one: a
    point emerging from behind an occluder is often still half-covered there, and one or two
    frames later it is whole. Scoring several and keeping the peak costs one small
    correlation each and is the difference between a resume the artist keeps and one they
    drop.

    Returns (frame, x, y, score, tried) with `tried` the per-candidate scores, or None.
    """
    best, tried = None, []
    for f, (gx, gy) in candidates:
        got = match(plate, f, patch, gx, gy, radius=radius, offset=offset)
        if got is None:
            tried.append((int(f), None))
            continue
        x, y, s = got
        tried.append((int(f), s))
        if best is None or s > best[3]:
            best = (int(f), x, y, s)
    if best is None:
        return None
    return best[0], best[1], best[2], best[3], tried


def find_reappearance(plate, jobs, min_match=0.60, settle=4, on_status=None):
    """When does each feature come back, and where?

    This is the answer to "Blender lost it -- skip ahead to where it is again". Each job is

        {"id", "patch", "offset", "radius", "path": [(frame, (x, y)), ...]}

    where `path` is the guide's predicted position for EVERY frame after the failure, in
    frame order. The sweep walks those frames ONCE, decoding each frame a single time and
    testing every job still looking, and resolves a job at the FIRST frame whose correlation
    against `patch` reaches `min_match`.

    `patch` is the LOCALISATION reference and is normally the feature as the track last saw
    it, not as it was seeded -- the caller decides. This function does not check identity;
    finding where something is and deciding whether it is the right something are separate
    questions, answered with different pictures. See `server.py`'s reacquire handler.

    First, not best. Best-over-the-whole-window would skip past a perfectly good return in
    favour of a marginally sharper frame fifty frames later, and every frame it skipped is a
    frame the artist has to track by hand. But the first frame over the line is often the
    feature only half back, so once one passes, the next `settle` frames are also scored and
    the best of that short run wins -- earliest return, best landing within it.

    Jobs are processed in whatever order they are given and share the decode, so the cost is
    one pass over the window regardless of how many tracks are looking.

    Returns {id: {"frame", "x", "y", "score", "first_frame", "scanned", "best_seen",
                  "best_frame"}} -- `frame` is None when the feature never came back, and
    the `best_*` fields say how close it got, which is what makes a refusal readable.
    """
    say = on_status or (lambda m: None)
    state = {}
    for j in jobs:
        state[j["id"]] = {"job": j, "path": dict(j["path"]), "done": False,
                          "frame": None, "x": None, "y": None, "score": None,
                          "first_frame": None, "settle_left": 0, "scanned": 0,
                          "best_seen": -1.0, "best_frame": None}

    frames = sorted({f for j in jobs for f, _ in j["path"]})
    if not frames:
        return {k: {kk: vv for kk, vv in v.items() if kk not in ("job", "path")}
                for k, v in state.items()}

    say("pattern sweep: %d frame(s) x %d track(s)" % (len(frames), len(jobs)))
    for f in frames:
        live = [s for s in state.values() if not s["done"] and f in s["path"]]
        if not live:
            continue
        img = _gray(plate.frame(int(f) - 1))
        if img is None:
            continue
        for s in live:
            px, py = s["path"][f]
            got = match_in(img, s["job"]["patch"], px, py,
                           radius=s["job"].get("radius", 32.0),
                           offset=s["job"]["offset"])
            s["scanned"] += 1
            if got is None:
                continue
            x, y, sc = got
            if sc > s["best_seen"]:
                s["best_seen"], s["best_frame"] = sc, int(f)
            if s["first_frame"] is None:
                if sc < min_match:
                    continue
                # First crossing: remember it, then keep looking for `settle` more frames
                # in case the feature is still emerging.
                s["first_frame"] = int(f)
                s["frame"], s["x"], s["y"], s["score"] = int(f), x, y, sc
                s["settle_left"] = int(settle)
                continue
            if sc > s["score"]:
                s["frame"], s["x"], s["y"], s["score"] = int(f), x, y, sc
            s["settle_left"] -= 1
            if s["settle_left"] <= 0:
                s["done"] = True
        if all(s["done"] for s in state.values()):
            break

    out = {}
    for k, s in state.items():
        out[k] = {"frame": s["frame"], "x": s["x"], "y": s["y"], "score": s["score"],
                  "first_frame": s["first_frame"], "scanned": s["scanned"],
                  "best_seen": None if s["best_seen"] < -0.5 else float(s["best_seen"]),
                  "best_frame": s["best_frame"]}
    return out


# ---------------------------------------------------------------- pattern drift

#: A scaled reference has to beat the unscaled one by more than noise before "the feature
#: got bigger" is a better explanation than "the box got bigger". Measured on the synthetic
#: cases in `tests/test_scale_drift.py` the two are nowhere near this margin: a true 1.45x
#: approach scores 0.92 scaled against 0.21 unscaled, and a box swelling onto its
#: neighbours scores 0.97 unscaled against 0.16 scaled. The margin exists for the ambiguous
#: middle, not for the cases it was built from.
SCALE_MARGIN = 0.05

#: Scale band inside which a flagged swell is small enough to be nothing. Measured over 36
#: LocScale tracks on SH004 (160 frames): tracks that never lose their patch hold a box
#: scale of p50 0.97, p90 1.08, p99 1.33 -- so 1.25 keeps the ordinary breathing of a
#: healthy track out of the "bad-box" bucket.
CLEAN_BAND = 1.25


def _resized(patch, scale, min_side=7, max_side=512):
    import cv2                                                        # noqa: PLC0415
    h, w = patch.shape
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if nw < min_side or nh < min_side or nw > max_side or nh > max_side:
        return None
    if (nw, nh) == (w, h):
        return patch
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(patch, (nw, nh), interpolation=interp)


def classify_drift(score_ref, score_scaled, offset_px, scale,
                   min_match=0.60, margin=SCALE_MARGIN, band=CLEAN_BAND):
    """What does a swollen pattern box mean? Three answers, two of them actionable.

      * `lost`    -- the artist's patch is not there at any size. The box grew because
                     there was nothing holding it, and the frames it measured on the way
                     are not worth keeping.
      * `grown`   -- the patch matches BETTER when resized by the amount the box grew. The
                     feature really is approaching (or leaving) camera; the box is right and
                     only the baseline was wrong.
      * `bad-box` -- the patch is there, at the size the artist set, and the box is not.
                     Swollen onto the surroundings, or collapsed to nothing; either way
                     putting the artist's box back is the fix.
      * `clean`   -- the patch is there at its own size and the box is near enough to it.
                     The flag was grain or a lighting step; carry on.

    `offset_px` -- how far the patch's correlation peak sits from the box centre -- is
    REPORTED and deliberately does NOT decide anything. It looked like the obvious drift
    measure and it is not: measured over 36 LocScale tracks on SH004, tracks whose patch
    stays findable for all 160 frames still show a peak p50 4.2 px and p90 25.0 px away
    from their own tracked position (search radius 24 px). A fixed patch matched against a
    plate 150 frames later cannot arbitrate a sub-pixel position -- gating on it would have
    dragged healthy tracks tens of pixels onto whatever the seed patch liked best, undoing
    exactly the per-frame precision Blender is here for. It answers presence and size; the
    position stays Blender's measurement.

    Kept separate from the pixels so its rules can be exercised on cases whose answer is
    known by construction rather than only observed on a plate.
    """
    scores = [s for s in (score_ref, score_scaled) if s is not None]
    if not scores or max(scores) < min_match:
        return "lost"
    if (score_scaled is not None and score_ref is not None
            and score_scaled - score_ref > margin):
        return "grown"
    if score_scaled is not None and score_ref is None:
        return "grown"
    dev = scale if scale >= 1.0 else (1.0 / scale if scale > 0 else 99.0)
    return "clean" if dev <= band else "bad-box"


def drift_report(plate, ref_frame, ref_box, frame, cx, cy, cur_box,
                 radius=32.0, min_match=0.60):
    """Why did this track's pattern box change size?

    `ref_box` and `cur_box` are (cx, cy, w, h) in plate pixels, y-DOWN: the artist's box on
    the frame they set it, and the box the tracker is holding now. The artist's patch is
    correlated against `frame` twice -- at its own size, and resized by exactly the amount
    the box grew -- and the two scores are what separate a feature that got bigger from a
    box that got bigger.

    Everything is reported, including both scores and both positions, because a verdict
    without the numbers behind it cannot be argued with when it is wrong. The positions in
    particular are reported and NOT acted on -- see `classify_drift` for the measurement
    that took the position out of the decision.
    """
    ref = reference_patch(plate, int(ref_frame), float(ref_box[0]), float(ref_box[1]),
                          float(ref_box[2]), float(ref_box[3]))
    if ref is None:
        return {"ok": False, "verdict": "no-reference",
                "reason": "the pattern box you set is off-plate or has no contrast"}
    patch, offset = ref
    img = _gray(plate.frame(int(frame) - 1))
    if img is None:
        return {"ok": False, "verdict": "no-frame",
                "reason": "frame %d could not be read" % int(frame)}

    rw, rh = float(ref_box[2]), float(ref_box[3])
    cw, ch = float(cur_box[2]), float(cur_box[3])
    scale = ((cw * ch) / (rw * rh)) ** 0.5 if rw > 0 and rh > 0 and cw > 0 and ch > 0 else 1.0

    got = match_in(img, patch, cx, cy, radius=radius, offset=offset)
    x = y = None
    score_ref = None
    off_px = 0.0
    if got is not None:
        x, y, score_ref = got
        off_px = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5

    score_scaled = None
    xs = ys = None
    big = _resized(patch, scale)
    if big is not None and abs(scale - 1.0) > 0.02:
        gs = match_in(img, big, cx, cy,
                      radius=radius, offset=(offset[0] * scale, offset[1] * scale))
        if gs is not None:
            xs, ys, score_scaled = gs

    verdict = classify_drift(score_ref, score_scaled, off_px, scale, min_match=min_match)
    return {"ok": True, "verdict": verdict, "scale": float(scale),
            "score_ref": None if score_ref is None else float(score_ref),
            "score_scaled": None if score_scaled is None else float(score_scaled),
            "x": None if x is None else float(x), "y": None if y is None else float(y),
            "x_scaled": None if xs is None else float(xs),
            "y_scaled": None if ys is None else float(ys),
            "offset_px": float(off_px),
            "patch_std": float(patch.std())}
