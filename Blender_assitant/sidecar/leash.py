"""A guide path for the WHOLE track, and an honest statement of when to believe it.

`cotrack.track_points` already produces a position for every frame; the re-acquire keeps
one of them. Keeping the rest is tempting and, measured, only half safe:

  * On a 250-frame shot with no real occlusion the guide never left the feature -- 250/250
    within 25 px -- and its DISPLACEMENT over a short gap agrees with a hand track to
    ~2 px at gap 1 and ~7 px at gap 30. Re-anchored per frame that is a usable leash.
  * On a 64-frame shot with two real occlusions the same model walks onto the OCCLUDER at
    the first cover and never comes back: 4.5 px error at gap 1, 266 px by the end. A leash
    built from it would drag correct tracks off their feature -- run against the artist's
    own hand track it demanded a cut at f27.

So the guide cannot be used unconditionally, and the model's own visibility head does not
separate the two cases (it is a threshold on covered-ness, not on identity).

**Closure does.** Track forward from the seed, then track BACK from where the forward pass
ended, and compare. Measured on those same two shots, using only what exists at runtime --
the backward query is the forward pass's own endpoint, not a known-good position:

    shot                        closure p50 / p90 / max      true error p50 / max
    2 occlusions (bad guide)      21.5 /  62.1 / 103.6         152.5 / 264.7
    250 frames   (good guide)      8.5 /  11.1 /  12.7           5.1 /  12.2

A factor of 5.6 on p90, which is what `TRUST_P90_PX` sits inside. Note closure UNDERSTATES
the true error badly (21.5 against 152.5) -- the two passes share a model and therefore
share its faults, exactly the limitation `tools/make_lk_reference.py` records for round-trip
closure. It is a detector, not a measurement, and it is used here only to answer yes/no.

Both faults it cannot see are real: a guide that fails identically in both directions reads
as trustworthy, and a guide that is right forward and wrong backward reads as untrustworthy
and is merely discarded. The second is safe. The first is why nothing here overrides the
pattern check -- the leash PROPOSES positions and `patmatch` still disposes.
"""

import math

# Closure above this (p90 over the span, in original plate px) means the two passes disagree
# about which feature they are on. Basis is the table above: 11.1 on a guide that was right,
# 62.1 on one that was wrong. Sitting at 25 leaves better than 2x headroom either side rather
# than splitting the difference, because the cost is asymmetric -- distrusting a good guide
# loses an optimisation, trusting a bad one moves correct tracks onto the wrong feature.
TRUST_P90_PX = 25.0

# Per-frame tolerance for the re-anchored leash: TOL_A + TOL_B*sqrt(gap). Fitted to the
# guide's measured displacement error against a hand track (p90 2.02 px at gap 1, 3.54 at 5,
# 4.57 at 12, 7.20 at 30), then rounded up. sqrt and not linear because the error is
# sublinear in the gap -- a linear fit through gap 1 would be 4x too wide by gap 30.
TOL_A = 1.2
TOL_B = 2.1


def tolerance(gap):
    """How far the leash may legitimately be wrong `gap` frames after its anchor."""
    return TOL_A + TOL_B * math.sqrt(max(1, int(gap)))


def _pct(vals, p):
    if not vals:
        return float("nan")
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def _chain(plate, seed_frame, seed_px, frame_lo, frame_hi, max_side, chain, on_status, step):
    """One direction, in windows, re-querying each window at the previous one's last point.

    `step` is +1 or -1 and is passed in rather than inferred from the frame range. Inferring
    it was a real bug: the return pass runs from the far end back to `frame_lo`, so its seed
    frame IS its upper bound, every comparison said "forward", the pass re-ran the direction
    it had just done, and closure came out empty -- which reads as `nan`, fails the `<=`
    test, and reports every guide as untrustworthy. A leash that never engages looks exactly
    like a leash that is being careful.

    Windowing is not an optimisation: offline CoTracker attends across the whole clip and a
    312-frame pass saturated a 16 GB A4000 without finishing. Error compounds across the
    seams, which is part of what closure then measures.
    """
    import cotrack                                                    # noqa: PLC0415
    out = {}
    step = 1 if int(step) > 0 else -1
    f0, q = int(seed_frame), (float(seed_px[0]), float(seed_px[1]))
    while True:
        f1 = min(frame_hi, f0 + chain - 1) if step > 0 else max(frame_lo, f0 - chain + 1)
        lo, hi = (f0, f1) if step > 0 else (f1, f0)
        if hi - lo + 1 < 2:
            break
        g = cotrack.track_points(plate, [(f0, q[0], q[1])], lo, hi,
                                 max_side=max_side, on_status=on_status)
        seg = g["tracks"][0]
        out.update(seg)
        if f1 == f0:
            break
        f0, q = f1, seg[f1]
        if (step > 0 and f0 >= frame_hi) or (step < 0 and f0 <= frame_lo):
            break
    return out


def compute(plate, seed_frame, seed_px, frame_lo, frame_hi, max_side=768, chain=120,
            trust_px=TRUST_P90_PX, on_status=None):
    """Guide path over [frame_lo, frame_hi] plus per-frame closure and a trust verdict.

    `seed_px` is in ORIGINAL plate pixels, y-down, and is the artist's own seed click --
    the same query the re-acquire uses, so the guide follows the feature the artist chose.

    Returns a dict; `trusted` False means every other field is diagnostic only.
    """
    say = on_status or (lambda m: None)
    seed_frame = int(seed_frame)
    frame_lo, frame_hi = int(frame_lo), int(frame_hi)
    if not (frame_lo <= seed_frame <= frame_hi):
        raise ValueError("seed frame %d is outside %d..%d" % (seed_frame, frame_lo, frame_hi))
    if frame_hi - frame_lo < 1:
        raise ValueError("need at least two frames")

    say("leash: forward pass f%d..f%d" % (seed_frame, frame_hi))
    fwd = _chain(plate, seed_frame, seed_px, frame_lo, frame_hi, max_side, chain, say, +1)
    if seed_frame > frame_lo:
        say("leash: pass f%d back to f%d (before the seed)" % (seed_frame, frame_lo))
        fwd.update(_chain(plate, seed_frame, seed_px, frame_lo, seed_frame,
                          max_side, chain, say, -1))
    if not fwd:
        raise RuntimeError("the guide produced no positions")

    end_f = max(fwd)
    say("leash: return pass from f%d back to f%d" % (end_f, frame_lo))
    # Queried at the FORWARD PASS'S OWN endpoint. Using a known-good position here would
    # measure something unavailable at runtime and report a closure the addon can never
    # reproduce.
    bwd = _chain(plate, end_f, fwd[end_f], frame_lo, end_f, max_side, chain, say, -1)

    closure = {}
    for f, (x, y) in fwd.items():
        b = bwd.get(f)
        if b is not None:
            closure[f] = math.hypot(x - b[0], y - b[1])
    vals = list(closure.values())
    p50, p90 = _pct(vals, 0.5), _pct(vals, 0.9)
    trusted = bool(vals) and p90 <= float(trust_px)
    reason = (("closure p90 %.1f px is within %.1f -- both passes agree which feature "
               "this is") % (p90, trust_px)) if trusted else \
             (("closure p90 %.1f px exceeds %.1f -- the passes disagree, so the guide "
               "steers nothing") % (p90, trust_px))
    say("leash: closure p50 %.1f  p90 %.1f  max %.1f px -> %s"
        % (p50, p90, max(vals) if vals else float("nan"),
           "TRUSTED" if trusted else "not trusted"))
    return {"path": fwd, "closure": closure, "trusted": trusted,
            "closure_p50": p50, "closure_p90": p90,
            "closure_max": max(vals) if vals else float("nan"),
            "reason": reason, "seed_frame": seed_frame,
            "frame_lo": frame_lo, "frame_hi": frame_hi}


def predict(guide, anchor_frame, anchor_px, frame):
    """Where the leash says the feature is at `frame`, anchored at a frame we believe.

    Displacement, never the guide's absolute coordinates: the guide accumulates ~0.1 px per
    frame of drift, so by frame 250 its absolute position is 24 px out while its motion over
    the last few frames is still good to 2 px. Re-anchoring to the last frame the correlator
    confirmed is what keeps the leash tight -- only the gap since THAT frame accrues error.

    Returns (x, y, tolerance_px) or None.
    """
    a = guide.get(int(anchor_frame))
    b = guide.get(int(frame))
    if a is None or b is None:
        return None
    return (anchor_px[0] + (b[0] - a[0]), anchor_px[1] + (b[1] - a[1]),
            tolerance(int(frame) - int(anchor_frame)))


def frames_between(guide, lo, hi):
    """Frames the guide covers in (lo, hi], in order -- the gap a leash may fill."""
    return [f for f in sorted(guide) if int(lo) < f <= int(hi)]


# The fill probes WIDER than it will accept: the peak is searched for within
# `probe_mult` x tolerance (at least `PROBE_MIN_PX`) and then required to come back inside
# the tolerance. Searching only as far as the tolerance would guarantee an in-range answer
# and prove nothing -- the same "is there somewhere better" reasoning the hold check and the
# QC pass already use. Measured on the artist's two shots, this is the gate that separates
# them: across a real occlusion the peak lands 3.7-19.8 px from where the leash said, across
# a gap where the feature was visible the whole time it lands 1.8-4.0 px away.
PROBE_MULT = 2.0
PROBE_MIN_PX = 12.0


def trust_over(closure, frames, trust_px=TRUST_P90_PX):
    """Is the guide believable over THESE frames?

    Local, not global, and that is the whole point. Measured on the artist's 250-frame shot:
    the guide's closure over the 151-frame window after f90 is 66.0 px -- untrustworthy --
    while over the five frames actually being filled it is 15.6 px, and the leash is accurate
    to 2 px there. A verdict taken over the whole window condemns a guide that is excellent
    where it is about to be used, because the guide's reliability falls off with distance
    from its query and the fill only ever reaches a few frames.
    """
    vals = [closure[f] for f in frames if f in closure]
    if not vals:
        return False, float("nan")
    p90 = _pct(vals, 0.9)
    return bool(p90 <= float(trust_px)), p90


def _anchor_on(matcher, frame, px, radius=6.0):
    """Where does the FILL'S OWN patch sit at the end it is anchored on?

    Without this the fill inherits an offset it then reads as error. The patch that fills is
    not always the patch that produced the anchor position -- after a drift cut the resume is
    localised with the artist's seed box while the track's own position came from a box it
    had been carrying for 200 frames -- and the two peak at different sub-positions on the
    same feature. Measured on the artist's 250-frame reference, that alone put the first
    filled frame 7.8 px from the prediction against a 3.3 px tolerance: a constant bias being
    refused as if it were a slide.

    Returns the patch's own peak, or None if it cannot see the feature there at all -- which
    is itself a reason not to fill from this end.
    """
    got = matcher(int(frame), float(px[0]), float(px[1]), float(radius))
    return None if got is None else (float(got[0]), float(got[1]))


def _walk(matcher, guide, vis, anchor_frame, anchor_px, frames, min_match, visible_gate):
    """Fill `frames` in order, stopping at the first one that fails. Never skips.

    A fill that skipped a frame and carried on would be an island across an occlusion --
    markers on the occluder with correct-looking neighbours either side, which is the single
    worst thing this can produce. `pattern_refine._extend_ends` stops at the first failure
    for the same reason.
    """
    out = []
    for f in frames:
        p = predict(guide, anchor_frame, anchor_px, f)
        if p is None:
            return out, "the guide has no position at f%d" % f
        if visible_gate and not vis.get(f, False):
            return out, "CoTracker calls the feature covered at f%d" % f
        got = matcher(f, p[0], p[1], max(PROBE_MULT * p[2], PROBE_MIN_PX))
        if got is None:
            return out, "nothing to correlate at f%d (box off-plate)" % f
        snap = math.hypot(got[0] - p[0], got[1] - p[1])
        if got[2] < min_match:
            return out, ("your pattern only reaches %.2f at f%d, under %.2f"
                         % (got[2], f, min_match))
        if snap > p[2]:
            return out, ("the best match at f%d sits %.1f px from where the guide said, "
                         "outside the %.1f px it is allowed to be wrong by"
                         % (f, snap, p[2]))
        out.append({"frame": f, "x": float(got[0]), "y": float(got[1]),
                    "score": float(got[2]), "snap": float(snap)})
    return out, ""


#: How far a guide-free step may move a marker between adjacent frames, in plate pixels. It
#: has to cover real inter-frame motion and must not reach a neighbouring feature -- the same
#: two-sided constraint as the pin, and the same measured answer. On the artist's reference
#: this is what refuses f232: the best match there scores 0.917 and sits 24.9 px away.
LOCAL_RADIUS_PX = 12.0


def _walk_local(matcher, anchor_frame, anchor_px, frames, radius, min_match,
                vis=None):
    """Step frame by frame with a PREVIOUS-FRAME prior and no guide at all.

    The guide walks answer "where does CoTracker think this went". This answers the simpler
    question a tracker asks: it was here last frame, is it near here now. It exists because
    the guide's prediction can be too far off to pass the snap gate on exactly the frames
    worth having -- measured on the artist's reference the bridge stopped one frame short of
    f233, where the seed patch scores 0.938 at their own hand-tracked position. The guide was
    wrong; the feature was plainly there.

    No drift, because the template is always the artist's seed patch and never the previous
    frame's pixels. What moves is only where the search starts.

    Stops at the first frame that fails, and the move cap is what makes that safe: on f232 of
    the same reference -- genuinely occluded -- this finds something scoring 0.917 that sits
    24.9 px away. The score alone would have taken it.
    """
    out, cur = [], (float(anchor_px[0]), float(anchor_px[1]))
    for f in frames:
        if vis is not None and not vis.get(f, False):
            break
        got = matcher(f, cur[0], cur[1], radius)
        if got is None:
            break
        moved = math.hypot(got[0] - cur[0], got[1] - cur[1])
        if got[2] < min_match or moved > radius:
            break
        out.append({"frame": f, "x": float(got[0]), "y": float(got[1]),
                    "score": float(got[2]), "snap": float(moved)})
        cur = (float(got[0]), float(got[1]))
    return out


def fill_gap(matcher, guide, vis, closure, anchor_frame, anchor_px, resume_frame,
             resume_px=None, back_guide=None, back_vis=None,
             min_match=0.60, trust_px=TRUST_P90_PX, visible_gate=True):
    """Positions for the frames between a cut and its resume, or nothing.

    A gap is not automatically an occlusion. Measured on the artist's 250-frame reference,
    5 of its 7 gap frames sit where the feature is plainly visible and their hand track has
    a sample: the loop cut as a precaution at f91 and re-acquired at f96, leaving work the
    artist then had to do. The other reference's gaps ARE occlusions and must stay empty --
    a marker there is on the occluder.

    **Filled from BOTH ends, each stopping at its own first failure.** Working forward from
    the cut alone is not enough, and the reason is measured: when the track died to drift,
    the last-good frame is ALREADY on the wrong feature, so a guide queried there follows the
    wrong feature too. On the artist's f91 cut, a guide queried at their hand-tracked f90
    closes to 15.2 px, and one queried where Blender actually was closes to 132.9 -- the same
    gap, the same frames, and the trust gate correctly refuses the second. The resume end has
    no such problem: it was verified against the artist's own pattern at 0.96 before anything
    was planted. So the resume is usually the good end of a drift gap, and the cut is usually
    the good end of an occlusion gap, and neither is reliably the one to work from.

    The two walks never cross: whatever neither reaches is left empty, which through a real
    occlusion is the correct answer.

    Gates, and every one of them earns its place on those two shots:

      * the guide must be trustworthy OVER THIS GAP (`trust_over`), judged per direction;
      * CoTracker must call the feature visible -- a report, not a gate, everywhere else in
        this codebase, but a fill is optional, so a false negative costs an empty gap that
        was already empty, where the same flag used to gate the SEARCH killed whole tracks;
      * the correlation peak, probed wider than the tolerance, must come back inside it;
      * and the run must be contiguous from its end.

    `matcher(frame, x, y, radius) -> (x, y, score) or None` is injected so this is testable
    without a plate, a GPU or a model.

    Returns (filled, reason), filled sorted by frame.
    """
    anchor_frame, resume_frame = int(anchor_frame), int(resume_frame)
    span = list(range(anchor_frame + 1, resume_frame))
    if not span:
        return [], "no frames between the cut and the resume"

    got, notes = {}, []
    for tag, end_f, end_px, order, g, v, gated in (
            ("forward from the cut", anchor_frame, anchor_px, span, guide, vis, True),
            ("back from the resume", resume_frame, resume_px, list(reversed(span)),
             back_guide if back_guide is not None else guide,
             back_vis if back_vis is not None else vis,
             back_guide is None)):
        if end_px is None:
            continue
        want = [end_f] + order
        ok, p90 = trust_over(closure, want, trust_px)
        # The closure gate applies to the guide anchored on the CUT, which is the one with
        # no independent support: when a track dies to drift its last good frame is already
        # on the wrong feature and a guide queried there follows the wrong feature.
        #
        # A guide anchored on the RESUME is a different case. That position was correlated
        # against the artist's own pattern and passed before anything was planted, so
        # refusing it for disagreeing with the cut-anchored guide would be refusing the
        # trustworthy one for disagreeing with the untrustworthy one -- and closure cannot
        # say which side is wrong. What rules on it instead is the per-frame evidence, which
        # is what separated the two shots when it was measured: across a real occlusion the
        # peak lands 3.7-19.8 px from the prediction, across a precautionary cut 1.8-4.0 px.
        if gated and not ok:
            notes.append("%s: guide closure is %.1f px, over %.1f" % (tag, p90, trust_px))
            continue
        if not ok:
            notes.append("%s: closure %.1f px (the ends disagree; per-frame evidence "
                         "decides)" % (tag, p90))
        seat = _anchor_on(matcher, end_f, end_px)
        if seat is None:
            notes.append("%s: your pattern does not fit at f%d" % (tag, end_f))
            continue
        # Only frames the other direction has not already taken, so the two walks meet in
        # the middle instead of one of them re-deciding the other's frames.
        todo = [f for f in order if f not in got]
        run, why = _walk(matcher, g, v, end_f, seat, todo, min_match, visible_gate)
        for r in run:
            got[r["frame"]] = r
        if why:
            notes.append("%s: %s" % (tag, why))

    # ---- finish from both ends, without the guide -------------------------------------
    # A re-acquire that only tracks FORWARD leaves the frames just before it empty even when
    # the feature is plainly visible on them. The guide walks above are meant to be that, but
    # they can only reach as far as the guide is right. Where one stopped early, step in from
    # that end with a previous-frame prior instead.
    #
    # Measured on the artist's reference: recovers f233 at 0.71 and 3.3 px from their hand
    # track, which the guide walk refused because its own prediction there was 11.6 px out.
    # It stops at f232, correctly -- that frame is occluded and the thing it finds scores
    # 0.917 twenty-five pixels away.
    for tag, end_f, end_px, order in (
            ("forward from the cut", anchor_frame, anchor_px, span),
            ("back from the resume", resume_frame, resume_px, list(reversed(span)))):
        if end_px is None:
            continue
        # Start from the last frame this END has actually reached -- the end itself if the
        # guide walk got nowhere, otherwise the furthest CONTIGUOUS frame it accepted. Every
        # such frame passed the same gates, so the walk is still anchored on something
        # verified; the point of the rule is only that it must never begin part-way into a
        # gap on a frame nothing has ruled on, which is how a walk marches into an occluder
        # with a confident-looking score at every step.
        #
        # Requiring the END itself was too strict and cost the frame this exists for: on the
        # artist's reference the guide walk accepted f235 and f234 and stopped, so the local
        # walk's first target was f233 -- not the end -- and it refused to run at all.
        seat_f, seat_px = end_f, end_px
        idx = 0
        while idx < len(order) and order[idx] in got:
            seat_f = order[idx]
            seat_px = (got[seat_f]["x"], got[seat_f]["y"])
            idx += 1
        todo = [f for f in order[idx:] if f not in got]
        if not todo:
            continue
        seat = _anchor_on(matcher, seat_f, seat_px)
        if seat is None:
            continue
        end_f = seat_f
        run = _walk_local(matcher, seat_f, seat, todo, LOCAL_RADIUS_PX, min_match,
                          vis=v if visible_gate else None)
        for r in run:
            got.setdefault(r["frame"], r)
        if run:
            notes.append("%s: %d more frame(s) stepped in without the guide"
                         % (tag, len(run)))

    filled = [got[f] for f in sorted(got)]
    if not filled:
        return [], "; ".join(notes) or "nothing to fill"
    # The notes survive a full fill. Reporting only "filled every frame" would hide that the
    # guide was refused and the frames came from the step-in instead, which is the difference
    # between a gap CoTracker crossed and one the artist's pattern crossed on its own.
    done = "filled every frame in the gap"
    if len(filled) == len(span):
        return filled, ("%s (%s)" % (done, "; ".join(notes))) if notes else done
    return filled, "; ".join(notes)
