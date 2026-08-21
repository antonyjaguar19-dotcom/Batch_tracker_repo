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
    against the artist's own patch reaches `min_match`.

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
