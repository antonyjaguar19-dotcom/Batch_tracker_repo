"""CoTracker3 as the re-acquisition guide.

Role in the loop: Blender measures every frame and dies when it loses correlation. It has
no notion of a feature coming back. CoTracker does -- it carries a queried point through an
occlusion and reports, per frame, whether it thinks the point is VISIBLE. So it answers the
one question Blender cannot: where did my feature go, and when is it back.

    artist places a seed
      -> Blender tracks it and dies at frame N
      -> CoTracker, queried at the artist's own point, says where it is at N+k
      -> Blender resumes there and tracks on
      -> repeat

Two things carried over from measurements already in this project:

  * **Use the guide's DISPLACEMENT, not its absolute position.** By the time a track dies,
    Blender and the guide have drifted apart -- median 6 px on SH004, p90 35 px, max 242.
    Planting at the guide's absolute position teleports the track onto whatever the guide
    was following and throws away the better of the two localisations: 13 % on-feature
    against 41 % for applying the guide's motion to Blender's last good position.

  * **A coarse guide is fine.** The existing TAPNext guide runs at 256 px and Blender still
    refines to sub-pixel from it, because the resume only has to land inside a widened
    search box. So downscaling a 4K plate for CoTracker costs nothing that matters.

LICENCE, stated plainly and once: CoTracker is **CC-BY-NC 4.0** (verified from
facebookresearch/co-tracker's own README: "The majority of CoTracker is licensed under
CC-BY-NC"). NonCommercial restricts USE, not merely distribution, so unlike a GPL component
"we only run it in-house" does not make it commercial-safe. This module exists because the
tool's owner asked for it after that was raised twice; the decision and its consequences are
theirs. The code and weights live under `vendor/` and `weights/`, both gitignored, so
nothing NC-licensed enters the repository.
"""

import os

import numpy as np

_MODEL = None
_MODEL_KEY = None


def _paths():
    from repo import paths                                            # noqa: PLC0415
    p = paths()
    root = p["assist_root"]
    return (os.path.join(root, "vendor", "co-tracker"),
            os.environ.get("BTR_COTRACKER_CKPT")
            or os.path.join(root, "weights", "cotracker3_scaled_offline.pth"))


def available():
    code, ckpt = _paths()
    return os.path.isdir(code) and os.path.isfile(ckpt)


def load(offline=True):
    """Build the predictor once and keep it. Offline sees the whole window at once, which
    is what makes it useful across an occlusion; the online model streams and cannot."""
    global _MODEL, _MODEL_KEY
    import sys
    import torch                                                      # noqa: PLC0415

    code, ckpt = _paths()
    if not os.path.isdir(code):
        raise RuntimeError("CoTracker is not installed: %s missing. "
                           "Run bootstrap.bat --with-cotracker" % code)
    if not os.path.isfile(ckpt):
        raise RuntimeError("CoTracker checkpoint missing: %s" % ckpt)
    if code not in sys.path:
        sys.path.insert(0, code)

    key = (ckpt, bool(offline))
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    from cotracker.predictor import CoTrackerPredictor                # noqa: PLC0415
    model = CoTrackerPredictor(checkpoint=ckpt, offline=bool(offline), v2=False,
                               window_len=60)
    if torch.cuda.is_available():
        model = model.cuda()
    _MODEL, _MODEL_KEY = model, key
    return model


def free():
    global _MODEL, _MODEL_KEY
    _MODEL, _MODEL_KEY = None, None
    try:
        import torch                                                  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:                                                 # noqa: BLE001
        pass


def track_points(plate, queries_px, frame_lo, frame_hi, max_side=768, on_status=None,
                 backward=True):
    """Track points through [frame_lo, frame_hi] (1-based, inclusive).

    `queries_px` is [(frame, x, y), ...] in ORIGINAL plate pixels, y-down (image space).
    Returns {"tracks": {i: {frame: (x, y)}}, "vis": {i: {frame: bool}}, "scale": s}, also in
    original plate pixels.

    The clip is decoded at `max_side` on its long edge. 4K at 768 is a 5x reduction, which
    the resume tolerates -- see the module docstring -- and is the difference between
    fitting in VRAM and not.
    """
    import torch                                                      # noqa: PLC0415

    say = on_status or (lambda m: None)
    frame_lo, frame_hi = int(frame_lo), int(frame_hi)
    n = frame_hi - frame_lo + 1
    if n <= 1:
        raise ValueError("need at least two frames")

    scale = min(1.0, float(max_side) / float(max(plate.w, plate.h)))
    w, h = int(round(plate.w * scale)), int(round(plate.h * scale))

    # Offline CoTracker attends across the whole clip and adds a support grid, so cost grows
    # steeply with frame count. Measured the hard way: 312 frames at 768 px filled a 16 GB
    # A4000 (16080 / 16376 MiB) and had not finished after nine minutes. Refuse rather than
    # thrash -- a job that never returns is worse than one that says why.
    budget = int(os.environ.get("BTR_COTRACKER_MAX_FRAMES", "160"))
    if n > budget:
        raise RuntimeError(
            "CoTracker window is %d frames; the practical limit here is %d. "
            "Re-acquisition only needs the frames just after the failure -- reduce "
            "search_len, or raise BTR_COTRACKER_MAX_FRAMES if you have the VRAM." % (n, budget))

    say("CoTracker: decoding %d frames at %dx%d (scale %.3f)" % (n, w, h, scale))

    import cv2                                                        # noqa: PLC0415
    buf = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        img = plate.frame(frame_lo - 1 + i)          # Plate.frame is 0-based
        if img is None:
            raise RuntimeError("could not read frame %d" % (frame_lo + i))
        if scale != 1.0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        buf[i] = img[:, :, ::-1]                     # BGR -> RGB

    video = torch.from_numpy(buf).permute(0, 3, 1, 2)[None].float()
    q = []
    for (f, x, y) in queries_px:
        t = int(f) - frame_lo
        if not (0 <= t < n):
            raise ValueError("query frame %d is outside %d..%d" % (f, frame_lo, frame_hi))
        q.append([float(t), float(x) * scale, float(y) * scale])
    queries = torch.tensor(q, dtype=torch.float32)[None]

    if torch.cuda.is_available():
        video, queries = video.cuda(), queries.cuda()
    model = load(offline=True)

    say("CoTracker: tracking %d point(s) over %d frames" % (len(q), n))
    with torch.no_grad():
        tracks, vis = model(video, queries=queries, backward_tracking=bool(backward))
    # tracks (1, T, N, 2) in the resized pixel space; vis (1, T, N)
    tr = tracks[0].detach().cpu().numpy()
    vs = vis[0].detach().cpu().numpy()
    del video, queries, tracks, vis
    torch.cuda.empty_cache()

    inv = 1.0 / scale if scale else 1.0
    out_tracks, out_vis = {}, {}
    for j in range(tr.shape[1]):
        t_map, v_map = {}, {}
        for t in range(tr.shape[0]):
            f = frame_lo + t
            t_map[f] = (float(tr[t, j, 0]) * inv, float(tr[t, j, 1]) * inv)
            v_map[f] = bool(vs[t, j] > 0.5)
        out_tracks[j], out_vis[j] = t_map, v_map
    say("CoTracker: done")
    return {"tracks": out_tracks, "vis": out_vis, "scale": scale,
            "frame_lo": frame_lo, "frame_hi": frame_hi}


def resume_position(guide_track, guide_vis, last_good_frame, last_good_px, gap=3,
                    max_search=200):
    """Where should a track that died at `last_good_frame` be resumed?

    Applies the GUIDE'S DISPLACEMENT to Blender's last good position rather than jumping to
    the guide's own coordinates -- 41 % on-feature against 13 % for the absolute version.
    Walks forward to the first frame the guide calls visible, so the resume lands after the
    occluder rather than inside it.

    Returns (resume_frame, (x, y)) or None if the guide never comes back.
    """
    g0 = guide_track.get(int(last_good_frame))
    if g0 is None:
        return None
    f = int(last_good_frame) + max(1, int(gap))
    limit = int(last_good_frame) + int(max_search)
    while f <= limit:
        g = guide_track.get(f)
        if g is not None and guide_vis.get(f, False):
            dx, dy = g[0] - g0[0], g[1] - g0[1]
            return f, (last_good_px[0] + dx, last_good_px[1] + dy)
        f += 1
    return None
