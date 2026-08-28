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

import math

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


# ---------------------------------------------------------------- pinning through a warp

#: ECC iterations and convergence. 60/1e-4 converges on a 41x41 patch in well under a
#: millisecond; the numbers are a cost ceiling, not a tuned quantity.
ECC_ITERS = 60
ECC_EPS = 1e-4

#: A patch with no contrast has no gradient for ECC to descend, and it fails noisily rather
#: than returning a bad answer. Below this it is not worth asking.
ECC_MIN_STD = 1.0


def warp_match_in(img, patch, cx, cy, offset=(0.0, 0.0), motion=None, init=None):
    """Align the artist's patch to this frame ALLOWING IT TO WARP, and score the fit.

    `match_in` slides a rigid patch and takes the best correlation. That assumes the feature
    still looks like the day it was seeded -- and it stops being true the moment the camera
    moves round it or towards it. Measured on the artist's own 250-frame hand track, where
    every frame is correct by construction and any low score is therefore the MATCHER
    failing rather than the track:

        frame   plain NCC   ECC affine
        f216      0.651       0.873
        f230      0.535       0.825
        f231      0.082         --
        f250      0.690       0.851
        worst     0.535       0.807   (over 27 sampled frames)

    Plain correlation falls under 0.60 at positions the artist tracked by hand. Allowing an
    affine warp, it never does. Every cut threshold in this addon was fitted against the
    first column, which is why a feature turning towards the camera reads as drift -- and why
    `first_loss` needed a five-frame settle to survive a four-frame dip that was never in the
    footage at all.

    Returns (x, y, score, warp) -- the patch centre corrected by the warp's translation, the
    enhanced correlation coefficient, and the 2x3 affine -- or None.
    """
    import cv2                                                        # noqa: PLC0415
    if img is None:
        return None
    mode = cv2.MOTION_AFFINE if motion is None else motion
    ph, pw = patch.shape[:2]
    P = np.ascontiguousarray(patch, dtype=np.float32)
    if float(P.std()) < ECC_MIN_STD:
        return None
    ih, iw = img.shape
    ccx, ccy = cx + offset[0], cy + offset[1]
    x0 = int(round(ccx - pw / 2.0))
    y0 = int(round(ccy - ph / 2.0))
    if x0 < 0 or y0 < 0 or x0 + pw > iw or y0 + ph > ih:
        return None
    W = np.ascontiguousarray(img[y0:y0 + ph, x0:x0 + pw], dtype=np.float32)
    warp = np.eye(2, 3, dtype=np.float32) if init is None else init.astype(np.float32).copy()
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)
    try:
        cc, warp = cv2.findTransformECC(P, W, warp, mode, crit, None, 5)
    except cv2.error:
        # Non-convergence is normal on a covered or featureless frame and is not an error --
        # the caller reads "no warped answer" and falls back to the rigid one.
        return None
    if not np.isfinite(cc) or not np.isfinite(warp).all():
        return None
    # ECC maps the TEMPLATE into the window, so the patch centre lands at the warp applied to
    # the patch's own centre. That displacement is the sub-pixel correction.
    pc = np.array([pw / 2.0, ph / 2.0, 1.0], dtype=np.float64)
    wc = warp.astype(np.float64) @ pc
    x = x0 + float(wc[0]) - offset[0]
    y = y0 + float(wc[1]) - offset[1]
    return float(x), float(y), float(cc), warp


def pinned(plate, frame, patch, cx, cy, radius=3.0, offset=(0.0, 0.0), motion=None):
    """`match_pinned` against a frame number rather than a decoded image.

    The frame-major callers decode once and share; the ones that ask about a single frame
    want this. It exists because the gap fill was handed `match` -- the RIGID matcher -- and
    a rigid score is exactly what fails on a feature that has changed shape, which is the
    case the fill is reaching for. Measured on the artist's reference: at f233 the rigid
    score at their own hand-tracked position is 0.212 and the pinned one is 0.938, so the
    walk stopped one frame short of a frame that was plainly there.
    """
    img = _gray(plate.frame(int(frame) - 1))
    if img is None:
        return None
    return match_pinned(img, patch, cx, cy, radius=radius, offset=offset, motion=motion)


def match_pinned(img, patch, cx, cy, radius=3.0, offset=(0.0, 0.0), motion=None):
    """The best the artist's pattern can do here, rigid OR warped -- position and score.

    Takes whichever fits better rather than always warping. An affine has four more degrees
    of freedom than a translation, and on a frame where nothing has changed those spend
    themselves on noise: over the same 27 frames the warped score sat 0.01-0.09 BELOW the
    rigid one wherever the rigid one was already above 0.95. It only wins where the feature
    has genuinely changed shape, and there it wins by 0.16-0.29.

    A rigid match is an affine with the extra parameters pinned at zero, so taking the better
    of the two is choosing the best fit within the family, not averaging two opinions.

    Returns (x, y, score, warp_or_None) or None.
    """
    rigid = match_in(img, patch, cx, cy, radius=radius, offset=offset)

    # ECC is a local optimiser, so WHERE IT STARTS decides which answer it finds. Starting it
    # only at the rigid peak makes it inherit the rigid pass's mistakes -- and the rigid pass
    # is exactly what fails when the feature has changed shape, which is the case this
    # function exists for. Measured on the artist's reference at f233: the rigid peak sits
    # 11.6 px from the feature and refining from it converges to 0.546, while the same patch
    # at the position the caller predicted scores 0.938. One start point, one wrong answer.
    #
    # So both are tried: the rigid peak, and the prediction handed in. Two extra correlations
    # on a 41x41 patch, against a frame that would otherwise be dropped.
    starts = [(cx, cy)]
    if rigid is not None and math.hypot(rigid[0] - cx, rigid[1] - cy) > 0.5:
        starts.append((rigid[0], rigid[1]))
    best = None
    for sx, sy in starts:
        got = warp_match_in(img, patch, sx, sy, offset=offset, motion=motion)
        if got is not None and (best is None or got[2] > best[2]):
            best = got
    if rigid is None and best is None:
        return None
    if best is None:
        return rigid[0], rigid[1], rigid[2], None
    if rigid is None or best[2] > rigid[2]:
        return best[0], best[1], best[2], best[3]
    return rigid[0], rigid[1], rigid[2], None


def warp_shape(warp):
    """How much the patch had to change shape, as (scale, non-uniformity).

    `scale` is the area ratio the warp applies and `skew` is how far the two axes were scaled
    differently -- which is what perspective does to a pattern and pure distance does not.
    Reported so a cut can say "this feature turned" rather than only "the score fell".
    """
    if warp is None:
        return 1.0, 0.0
    a = np.asarray(warp, dtype=np.float64)[:, :2]
    sx = float(math.hypot(a[0, 0], a[1, 0]))
    sy = float(math.hypot(a[0, 1], a[1, 1]))
    area = abs(float(np.linalg.det(a)))
    skew = abs(sx - sy) / max(1e-6, 0.5 * (sx + sy))
    return math.sqrt(max(0.0, area)), skew


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


def find_reappearance(plate, jobs, min_match=0.60, settle=4, collect=40,
                      band=0.04, on_status=None):
    """When does each feature come back, and where?

    This is the answer to "Blender lost it -- skip ahead to where it is again". Each job is

        {"id", "patch", "offset", "radius", "path": [(frame, (x, y)), ...]}

    where `path` is the guide's predicted position for EVERY frame after the failure, in
    frame order. The sweep walks those frames ONCE, decoding each frame a single time and
    testing every job still looking, and resolves a job at the EARLIEST frame that scores
    within `band` of the best score it saw.

    `patch` is the LOCALISATION reference and is normally the feature as the track last saw
    it, not as it was seeded -- the caller decides. This function does not check identity;
    finding where something is and deciding whether it is the right something are separate
    questions, answered with different pictures. See `server.py`'s reacquire handler.

    This used to be FIRST over the line, on the reasoning that best-over-the-window would
    skip past a perfectly good return for a marginally sharper frame fifty frames later, and
    every skipped frame is one the artist tracks by hand. Measured against the first hand
    track this project has through an occlusion (SH006, feature hidden f15-24, back at f25):

        chosen by "first over the line"   f17  score 0.84   INSIDE the occlusion, no
                                                            hand-track sample exists there
        every other candidate             f26  0.99   1.7 px from the hand track
                                          f31  0.98   1.4 px
                                          f45  0.95   1.2 px

    The "perfectly good return" the old rule protected was the occluder. Every alternative
    was on the feature within 2.4 px; only the selection was wrong.

    So: the earliest frame scoring within `band` of the best. That keeps what the old rule
    was defending -- it will not skip to frame 45 when 26 is just as good -- while refusing
    a marginal crossing when something clearly better exists. On the case above it picks
    f26 (0.99) over f31 (0.98) and never considers f17 (0.84).

    A caution that still stands: a score is comparable WITHIN one sweep, where every
    candidate is the same patch against the same track. It is NOT comparable across tracks
    or shots -- measured separately, the worst landings across a reference set scored
    0.85-0.98. This rule only ever compares within a sweep.

    Jobs are processed in whatever order they are given and share the decode, so the cost is
    one pass over the window regardless of how many tracks are looking.

    `collect` frames keep being scored AFTER a job resolves, so the alternatives in
    `candidates` reach past the first crossing. They do not affect which frame is returned.

    Returns {id: {"frame", "x", "y", "score", "first_frame", "scanned", "best_seen",
                  "best_frame", "candidates"}} -- `frame` is None when the feature never came
    back, and the `best_*` fields say how close it got, which makes a refusal readable.
    """
    say = on_status or (lambda m: None)
    state = {}
    for j in jobs:
        state[j["id"]] = {"job": j, "path": dict(j["path"]), "done": False, "seen": [], "collect_left": int(collect),
                          "frame": None, "x": None, "y": None, "score": None,
                          "first_frame": None, "settle_left": 0, "scanned": 0,
                          "best_seen": -1.0, "best_frame": None,
                          # Where the best-scoring frame put it, even when that score never
                          # reached the gate. A refusal that cannot say WHERE it looked
                          # cannot be turned into anything an artist could judge.
                          "best_x": None, "best_y": None}

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
            # Every frame that was scored, kept. The sweep already did this work and threw
            # it away, which is why a refused resume used to leave an artist with one wrong
            # answer and no second opinion. See `top_candidates`.
            s["seen"].append((int(f), float(x), float(y), float(sc)))
            if sc > s["best_seen"]:
                s["best_seen"], s["best_frame"] = sc, int(f)
                s["best_x"], s["best_y"] = x, y
            if s["first_frame"] is None:
                if sc < min_match:
                    continue
                # First crossing: remember it, then keep looking for `settle` more frames
                # in case the feature is still emerging.
                s["first_frame"] = int(f)
                s["frame"], s["x"], s["y"], s["score"] = int(f), x, y, sc
                s["settle_left"] = int(settle)
                continue
            # Only the SETTLE frames may improve the answer. The collect frames after them
            # exist to fill `candidates` and must not quietly turn this into
            # best-over-the-window -- that is a different rule with a documented reason
            # against it, and changing it by accident is how a behaviour nobody chose ships.
            if s["settle_left"] > 0 and sc > s["score"]:
                s["frame"], s["x"], s["y"], s["score"] = int(f), x, y, sc
            s["settle_left"] -= 1
            if s["settle_left"] <= 0:
                # Resolved -- but keep SCORING for a while. The first crossing is the answer
                # this function returns, and it is not always the right one: measured on
                # SH006 the sweep resolved at frame 17 and the artist confirmed by eye that
                # it was the wrong feature, while the real reappearance was frame 25. Stopping
                # at the crossing meant 25 was never scored and could not be offered as an
                # alternative. These extra frames change nothing about the resume; they exist
                # so `top_candidates` has somewhere else to point.
                if s["collect_left"] > 0:
                    s["collect_left"] -= 1
                else:
                    s["done"] = True
                s["settle_left"] = 0
        if all(s["done"] for s in state.values()):
            break

    out = {}
    for k, s in state.items():
        # Pick the earliest frame within `band` of the best this sweep saw. `first_frame`
        # keeps recording the first crossing, so a report can still say where it started.
        cands = [c for c in top_candidates(s["seen"], k=24, min_gap=1)
                 if c["score"] >= min_match]
        if cands:
            best = max(c["score"] for c in cands)
            pick = min((c for c in cands if c["score"] >= best - band),
                       key=lambda c: c["frame"])
            s["frame"], s["x"], s["y"] = pick["frame"], pick["x"], pick["y"]
            s["score"] = pick["score"]
        out[k] = {"frame": s["frame"], "x": s["x"], "y": s["y"], "score": s["score"],
                  "first_frame": s["first_frame"], "scanned": s["scanned"],
                  "best_seen": None if s["best_seen"] < -0.5 else float(s["best_seen"]),
                  "best_frame": s["best_frame"],
                  "best_x": s["best_x"], "best_y": s["best_y"],
                  "candidates": top_candidates(s["seen"])}
    return out


def peak_margin(img, patch, cx, cy, offset=(0.0, 0.0), radius=120.0, sep=30.0):
    """How much better is the best match than the best OTHER match nearby?

    A high correlation score means nothing on repeating texture. Measured on SH006, where a
    thin wire crosses the feature and the track walks onto a lookalike 52 px away: the score
    stays 0.90+ while the two candidates converge to within 0.006 of each other. At that
    point the pixels do not determine the position and any answer is a coin toss.

    `sep` keeps the runner-up from being the same peak one pixel over.

    Returns (best, second, margin) or (best, None, None) when nothing is far enough away.
    """
    import cv2                                                        # noqa: PLC0415
    if img is None:
        return None, None, None
    ih, iw = img.shape
    ph, pw = patch.shape
    r = max(1, int(round(radius)))
    ccx, ccy = cx + offset[0], cy + offset[1]
    x0 = int(round(ccx - pw / 2.0)) - r
    y0 = int(round(ccy - ph / 2.0)) - r
    x1, y1 = x0 + pw + 2 * r, y0 + ph + 2 * r
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(iw, x1), min(ih, y1)
    if (x1 - x0) < pw or (y1 - y0) < ph:
        return None, None, None
    sc = cv2.matchTemplate(img[y0:y1, x0:x1], patch, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(sc).any():
        return None, None, None
    sc = np.nan_to_num(sc, nan=-1.0, posinf=-1.0, neginf=-1.0)
    order = np.dstack(np.unravel_index(np.argsort(-sc, axis=None), sc.shape))[0]
    by, bx = order[0]
    best = float(sc[by, bx])
    bxp, byp = x0 + bx + pw / 2.0, y0 + by + ph / 2.0
    for yy, xx in order[1:]:
        px, py = x0 + xx + pw / 2.0, y0 + yy + ph / 2.0
        if math.hypot(px - bxp, py - byp) >= sep:
            second = float(sc[yy, xx])
            return best, second, best - second
    return best, None, None


def hold_check(plate, patch, offset, path, radius=3.0, margin_radius=120.0,
               probe_radius=0.0):
    """Score the artist's own patch at every position a track claims, in frame order.

    Blender tracks with PREV_FRAME: each frame is matched against the one before it. That is
    what gives it precision, and it is also why an occluder captures a track without the
    correlation ever failing -- the occluder slides in over a few frames and each step looks
    like a small, plausible move. The track never dies, so nothing downstream ever asks for
    a re-acquire, and the drift is written to the file as if it were data.

    The question nobody was asking is whether it is still the ARTIST'S feature. Measured on
    SH006, a seed occluded at frame 14:

        f13  0.985    f14  0.908    f15  0.218    f16  0.210  ...  f22  0.178

    A collapse from 0.91 to 0.22 in one frame. The signal does not need a careful threshold;
    it needs to be looked at.

    `radius` is deliberately tiny. This is not a search -- the position is Blender's and is
    not in question here. It asks what is AT that position.

    Returns [(frame, score or None), ...] in the order given.
    """
    out = []
    for f, x, y in path:
        img = _gray(plate.frame(int(f) - 1))
        if img is None:
            out.append((int(f), None, None))
            continue
        # Pinned, not rigid. This is the score every cut in the addon is decided on, and a
        # rigid patch reads a feature turning towards the camera as a feature being lost --
        # the artist's report that "the QC checker confuses perspective change with drift and
        # deletes the tracks in those areas". On their own hand track the rigid score falls
        # to 0.535 and 0.082 at positions they tracked by hand; allowing the pattern to warp,
        # the worst of the same frames is 0.807.
        got = match_pinned(img, patch, float(x), float(y), radius=radius, offset=offset)
        _b, _s, mg = peak_margin(img, patch, float(x), float(y), offset=offset,
                                 radius=margin_radius)
        gain = dist = None
        if probe_radius > 0.0 and got is not None:
            # Is there somewhere BETTER nearby? The question that survives a feature
            # legitimately changing appearance. A track sitting on its feature is at the
            # local optimum even when its absolute score has fallen; a track that has slid
            # off has a much better answer a short distance away. Measured on SH006 at each
            # track's last frame: the drifted one scores 0.501 where 0.979 sits 23 px away,
            # while the artist's hand track scores 0.731 with only 0.786 available -- a gain
            # of 0.478 against 0.055.
            # The probe stays RIGID on purpose. It asks "is there a better answer a short
            # way off", and letting it warp too would let it find a good affine fit to some
            # other piece of the plate and report a gain that is about the model's freedom
            # rather than about a better feature. The comparison has to be like for like, so
            # the gain is measured against the rigid score at the claimed position.
            here = match_in(img, patch, float(x), float(y), radius=radius, offset=offset)
            near = match_in(img, patch, float(x), float(y), radius=probe_radius,
                            offset=offset)
            if near is not None and here is not None:
                gain = float(near[2]) - float(here[2])
                dist = math.hypot(near[0] - float(x), near[1] - float(y))
        out.append((int(f), None if got is None else float(got[2]), mg, gain, dist))
    return out




#: A failing run is only cleared by a CONVINCING recovery, as a fraction of the track's own
#: opening score -- not merely by a frame that scrapes over the floor.
#:
#: Without this the cut lands late and the frames in between are written to the file as data.
#: Measured on the artist's track3 Track.001, at the track's own positions: f28-f32 score
#: 0.92-0.96 and are correct, then f33 0.435 and f34 0.328 are the occluder. f35 scores 0.596
#: -- above the 0.50 floor, and above 0.6 x 0.96 = 0.58 BY 0.016 -- so it reset the run, the
#: cut slid from f33 to f37, and f33-f36 shipped as real samples. Four wrong frames turned on
#: sixteen thousandths of a correlation score.
#:
#: 0.8 of the opening is a recovery nobody would argue with; 0.62 (what f35 managed) is not.
CLEAR_SHARE = 0.8

#: A COLLAPSE needs no second opinion. Every other trigger here requires `better_exists` --
#: proof that a better match sits nearby -- because a feature turning towards the camera loses
#: score without having gone anywhere, and cutting on that deleted correct work. But an
#: occluder that resembles the pattern IS the best match in its neighbourhood, so nothing
#: better exists, and the track is judged healthy while sitting on a sign.
#:
#: Measured, and the two cases do not overlap:
#:
#:     a feature turning towards camera, correct    worst warped score  0.81 of baseline
#:     the occluder on track3 Track.001 at f33      0.435 against 0.99  0.44 of baseline
#:
#: So a fall past half of what the track was holding is a collapse, and a collapse is allowed
#: to cut on its own. Four wrong frames on that reference hung on this.
COLLAPSE_SHARE = 0.5


#: Frames of failure before a cut. It was FIVE, and five was fitted to the RIGID matcher:
#: the artist's own correct hand track dipped to 0.132 there, so anything shorter cut their
#: own work. `hold_check` now scores with `match_pinned`, and the same dip bottoms out at
#: 0.807 -- the reason for five went with the matcher it was measured on.
#:
#: Two, measured across all three references. Every wrong frame the loop produced sat in the
#: window five was waiting through:
#:
#:                       settle=5              settle=2
#:     track3 T.001      46 on, 5 OFF          45 on, 1 OFF
#:     250-frame ref     250 on, 0 off         249 on, 0 off
#:     ref1              46 on, 0 off          46 on, 1 OFF
#:     total             342 on, 5 off         340 on, 2 off
#:
#: Three wrong frames removed for two right ones, and the trade is not symmetric: a gap is
#: honest and 3DE solves through it, while a wrong marker has to be found by hand and
#: corrupts the solve if it is not. Note it is NOT monotonic -- cutting earlier moves the
#: resume, which changes everything downstream, which is why ref1 gains a bad frame while
#: track3 loses four.
HOLD_SETTLE = 2


def first_loss(scores, floor=0.5, drop=0.6, settle=HOLD_SETTLE, head=10, look=5,
               min_fall=0.20,
               ambig_margin=0.05, ambig_drop=0.85, probe_gain=0.10,
               clear_share=CLEAR_SHARE, collapse_share=COLLAPSE_SHARE):
    """The first frame where a track stopped being on the artist's feature.

    Two conditions, and the second is what keeps this honest on difficult footage. An
    absolute floor alone would condemn every track on a low-contrast plate -- measured on
    SH013, patches there score 0.53-0.72 against the very NEXT frame while tracking
    perfectly well. So a loss is a score that is BOTH below `floor` and below `drop` times
    what this track was holding AT ITS START. A track that never scored well cannot fall.

    On the artist's SH006 reference this puts the cut at f33 -- the first frame of their
    second occlusion -- and leaves f32 (0.608, a real frame in their hand track) alone,
    because 0.6 x 0.99 = 0.594 sits between them.

    `settle` frames must agree before it counts. It was two, fitted to a synthetic case, and
    real data moved it: the artist's own 250-frame hand track -- every frame correct by
    definition -- contains a **four-frame** run where the seed patch scores as low as 0.132 at
    a position they tracked by hand, and then recovers to 0.731. A 230-frame-old patch simply
    stops describing the feature for a moment; blur, a light change, something passing. Two
    frames of agreement cut that track at f230.

    Five frames is what separates it from the drift on the same shot, which runs five and
    never recovers. That number comes from measured footage, not from taste, and it is the
    reason a transient dip is survivable at all.

    Returns the frame number, or None.
    """
    # `scores` may be [(frame, score)], [(frame, score, margin)] or
    # [(frame, score, margin, gain, distance)].
    scores = [(r[0], r[1], (r[2] if len(r) > 2 else None),
               (r[3] if len(r) > 3 else None)) for r in scores]
    good = [s for _f, s, _m, _g in scores if s is not None]
    if not good:
        return None
    # The baseline comes from the track's FIRST `head` scored frames, not the median of all
    # of them. Measured on the artist's SH006 reference, a track that drifted after its
    # second occlusion:
    #
    #     baseline from the whole track   0.61   <- half the track was already drift
    #     baseline from the first 14      0.99
    #
    # By the time this runs, the drift is IN the scores, so a median over all of them lets
    # the wrong frames define what normal looks like -- and nothing can then fall below half
    # of it. The head of a track is the part anchored to the frame the artist seeded, which
    # is the only part known to be their feature.
    head_scores = [s for _f, s, _m, _g in scores
                   if s is not None][:max(1, int(head))]
    base = sorted(head_scores)[len(head_scores) // 2]
    # And it has to be a FALL, not a slide. A feature going soft -- defocus, a light change,
    # a plate getting grainier -- declines gently and is still the artist's feature; an
    # occluder arrives. Measured on the SH006 reference the second occlusion reads
    # 0.86 -> 0.61 -> 0.35 across two frames, while a defocus decline moves about 0.02 a
    # frame. Requiring the drop to be at least `min_fall` below the recent level separates
    # them without another threshold on the score itself.
    recent = []
    bad_run = 0
    first_bad = None
    for f, s, mg, gain in scores:
        if s is None:
            continue
        # A score falling is not on its own a reason to cut. A feature approaching camera
        # changes PERSPECTIVE -- it is the same feature, correctly tracked, and it stops
        # resembling the patch taken when it was small and far away. Cutting there ends a
        # good track exactly where it starts to matter.
        #
        # What separates that from drift is whether somewhere better exists. A track still on
        # its feature sits at the local optimum however much its score has fallen; a track
        # that slid off has a much better answer a short distance away. Measured on the
        # artist's SH006 pair at the last frame of each: the drifted track scores 0.501 with
        # 0.979 sitting 23 px away, their hand track scores 0.731 with only 0.786 available
        # -- a gain of 0.478 against 0.055.
        #
        # `gain is None` means nobody looked, and then the older rules stand alone.
        better_exists = (gain is None) or (gain > probe_gain)
        # AMBIGUITY. A margin this small means the plate itself cannot say which of two
        # places the feature is -- measured on SH006, a wire crossing the feature leaves two
        # candidates 52 px apart within 0.006 of each other. The margin alone cannot condemn
        # a track: it is a property of the PLATE and reads identically for a correct track
        # and a drifting one at the same frame. What separates them is the score at the
        # position the track claims. Measured at f91-95 there: the drifting track scores
        # 0.80 -> 0.50 while the artist's own hand track holds 0.91 -> 0.89, with the same
        # margin. Both conditions, or neither.
        # SOMEWHERE BETTER. The plainest statement of drift there is: the artist's patch
        # matches far better a short distance from where the track claims to be. It needs no
        # rate and no margin, which is what makes it the one rule that catches a SLOW slide
        # onto a neighbour -- gradual enough never to trip the fall test, and on texture
        # distinct enough never to trip the ambiguity test.
        #
        # Guarded by the score having given way at all, so a healthy track with a marginally
        # better neighbour is left alone.
        # A collapse first, and without asking whether anything better is nearby. See
        # COLLAPSE_SHARE: on an occluder that matches, nothing better IS nearby, which is
        # exactly why the other three triggers stay silent through an occlusion.
        if s < base * collapse_share:
            bad_run += 1
            if first_bad is None:
                first_bad = f
            if bad_run >= settle:
                return first_bad
            continue

        if (gain is not None and gain > probe_gain and s < base * ambig_drop):
            bad_run += 1
            if first_bad is None:
                first_bad = f
            if bad_run >= settle:
                return first_bad
            recent.append(s)
            if len(recent) > look:
                recent.pop(0)
            continue

        if (mg is not None and mg < ambig_margin and s < base * ambig_drop
                and better_exists):
            bad_run += 1
            if first_bad is None:
                first_bad = f
            if bad_run >= settle:
                return first_bad
            recent.append(s)
            if len(recent) > look:
                recent.pop(0)
            continue

        fell = True
        if recent:
            prev = sorted(recent)[len(recent) // 2]
            fell = s < prev - min_fall
        if s < floor and s < base * drop and fell and better_exists:
            bad_run += 1
            if first_bad is None:
                first_bad = f
            if bad_run >= settle:
                return first_bad
            # Do NOT let a rejected frame into `recent`. It used to, and the bad frames then
            # dragged the recent level down to their own value -- by the third one the drop
            # was no longer a fall against it, the run reset, and a long failure could never
            # reach the settle count. Invisible while settle was 2; fatal at 5.
            continue
        else:
            # HYSTERESIS. A run of failing frames is only cleared by a CONVINCING recovery,
            # not by one frame that scrapes over the threshold. Measured on the artist's
            # track3 Track.001: f33 (0.435) and f34 (0.328) are the occluder, and f35 at
            # 0.596 cleared both the 0.50 floor and 0.6 x 0.96 = 0.58 -- by 0.016 -- so the
            # run reset, the cut slid from f33 to f37, and four frames of occluder shipped as
            # data. A frame that has recovered to 62 % of what the track was holding has not
            # recovered.
            if bad_run and s < base * clear_share:
                # Still failing as far as this is concerned: keep the run and keep the frame
                # out of `recent`, for the same reason rejected frames are kept out above.
                bad_run += 1
                if bad_run >= settle:
                    return first_bad
                continue
            bad_run = 0
            first_bad = None
        # Only frames that were ACCEPTED shape the recent level, so a drift cannot quietly
        # become the new normal one frame at a time.
        recent.append(s)
        if len(recent) > look:
            recent.pop(0)
    return None


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


def top_candidates(seen, k=6, min_gap=3, floor=0.2):
    """The best few places the feature might be, as alternatives to offer the artist.

    A re-acquire lands on the wrong feature often enough that one answer is not enough --
    measured on SH006, the top match was frame 17 and the artist confirmed by eye that it was
    the wrong feature entirely, while the real reappearance was frame 25. The sweep had
    already scored frame 25; nothing kept it.

    `min_gap` stops the list being six frames of the same peak: adjacent frames of one
    reappearance score almost identically and would fill every slot with the same answer.
    Candidates are taken best-first, each at least `min_gap` frames from every one already
    taken, so the list spans the window instead of clustering.

    Ranked by score, NOT by frame. The artist is choosing between places, and the most
    likely one should be the first thing they see.
    """
    out = []
    for f, x, y, sc in sorted(seen, key=lambda r: -r[3]):
        if sc < floor:
            break
        if any(abs(f - g) < min_gap for g, _x, _y, _s in out):
            continue
        out.append((int(f), float(x), float(y), float(sc)))
        if len(out) >= k:
            break
    return [{"frame": f, "x": x, "y": y, "score": round(sc, 3)} for f, x, y, sc in out]


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

      * `unknown` -- the artist's patch is not there at any size, which sounds like proof
                     the track is lost and is not. It is the ABSENCE of evidence, and the
                     two causes are indistinguishable from here: the tracker slid off, or
                     the feature stopped looking like its seed frame. Measured on SH013
                     (59.94 fps chase plate) the second is routine -- a foreground patch
                     scores 0.53-0.72 against the NEXT frame, let alone against a seed
                     fifty frames back -- and a correctly tracked feature reads exactly the
                     same as a lost one. This verdict therefore does NOT delete anything.
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
        # Absence of evidence. This used to return "lost" and the caller deleted every frame
        # back to the swell onset -- measured on SH013 that cut a track which otherwise ran
        # the whole 303-frame shot down to 5 markers, because the seed patch is simply not
        # findable on that footage. Nothing here can tell "slid off" from "changed
        # appearance", so nothing here may destroy frames.
        return "unknown"
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
