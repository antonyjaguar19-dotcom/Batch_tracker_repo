"""How far does the plate move between frames, and where?

`track_core.KIND_GEOM` is a fixed table: a corner gets a 21 px pattern in a 41 px search box,
scaled only by plate width. It was tuned on SH004, where the camera is slow, and it carries
**no motion term at all**. On a chase plate that is not a tuning choice, it is a wall --
measured on SH013 (motocross, 59.94 fps, camera chasing the bike) the near-ground moves
21-53 px between frames, p95 67, while the shipped corner box can reach +-13 px. Every
foreground seed died on its FIRST step, not because the feature was hard but because it was
never inside the box being searched. Widening the box took the same seeds from span 1 to
spans 13-47, each ending with its search box off the edge of the plate -- which is the
feature leaving frame, the honest end of a near-ground track on a shot like that.

Correlation is not the problem on that footage and this must not be mistaken for a fix to
it: at the CORRECT position those same foreground patches score 0.88-0.93 NCC between
consecutive frames. They are findable. They were simply not searched for.

Deliberately coarse. This sizes a search box; it does not need sub-pixel flow, and paying
for accuracy here would make it too slow to run before every job.
"""

import numpy as np


#: Sample this many frame PAIRS across the shot. Motion is not constant -- SH013 runs
#: 2-20 px in the same band depending on where the camera is in the chase -- so a single
#: pair at the head of the shot would size the whole run from its quietest moment.
DEFAULT_SAMPLES = 9

#: Analysis width. Farneback on a 4K frame costs seconds; the answer is a box size in tens
#: of pixels, so the precision lost by running at ~900 px wide does not reach the result.
ANALYSIS_W = 896

#: Grid the plate into this many cells each way. A chase plate's motion is wildly
#: non-uniform -- on SH013 the sky reads 0.0 px/frame while the near-ground reads 20 -- so
#: one number for the plate would either starve the foreground or bloat every background
#: box into a false-match machine.
GRID = (6, 4)


def _gray(img):
    import cv2                                                        # noqa: PLC0415
    if img is None:
        return None
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] >= 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    a = np.squeeze(a)
    return None if a.ndim != 2 else a


def measure(plate, n_frames, samples=DEFAULT_SAMPLES, grid=GRID, on_status=None):
    """Per-cell inter-frame motion in FULL-RES plate pixels.

    Returns {"grid": [gx, gy], "p95": [[...]], "median": [[...]], "global_p95": float,
             "pairs": n} with rows top-to-bottom in image space, matching the y-down
    convention the addon sends positions in.
    """
    import cv2                                                        # noqa: PLC0415
    say = on_status or (lambda m: None)
    gx, gy = grid
    n_frames = int(n_frames)
    if n_frames < 2:
        return None

    # Spread the pairs over the middle of the shot rather than the whole of it: the first and
    # last frames of a handheld plate are often the least representative (a camera settling,
    # or an operator stopping) and sizing every box from them is how a fast shot gets a slow
    # box.
    lo, hi = 1, max(2, n_frames - 1)
    picks = np.linspace(lo, hi, num=min(samples, hi - lo + 1), dtype=int)

    scale = None
    acc = []
    for f in picks:
        a = _gray(plate.frame(int(f) - 1))
        b = _gray(plate.frame(int(f)))
        if a is None or b is None:
            continue
        if scale is None:
            scale = min(1.0, float(ANALYSIS_W) / float(a.shape[1]))
        if scale < 1.0:
            size = (int(a.shape[1] * scale), int(a.shape[0] * scale))
            a = cv2.resize(a, size)
            b = cv2.resize(b, size)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 31, 3, 7, 1.5, 0)
        acc.append(np.linalg.norm(flow, axis=2) / scale)      # back to full-res px
    if not acc:
        return None

    say("motion: %d frame pair(s) over %d frames" % (len(acc), n_frames))
    stack = np.stack(acc)                                     # (pairs, h, w)
    h, w = stack.shape[1], stack.shape[2]
    p95 = np.zeros((gy, gx), dtype=float)
    med = np.zeros((gy, gx), dtype=float)
    for j in range(gy):
        y0, y1 = int(h * j / gy), int(h * (j + 1) / gy)
        for i in range(gx):
            x0, x1 = int(w * i / gx), int(w * (i + 1) / gx)
            cell = stack[:, y0:y1, x0:x1]
            # p95 per cell, ACROSS pairs as well as pixels: a box has to survive the fastest
            # moment the feature lives through, not the average one.
            p95[j, i] = float(np.percentile(cell, 95))
            med[j, i] = float(np.median(cell))
    return {"grid": [gx, gy], "p95": p95.tolist(), "median": med.tolist(),
            "global_p95": float(np.percentile(stack, 95)), "pairs": len(acc)}


def cell_for(mo, x, y, w, h):
    """Motion at a plate position (image px, y-DOWN). Returns (p95, median)."""
    if not mo:
        return (0.0, 0.0)
    gx, gy = mo["grid"]
    i = min(gx - 1, max(0, int(x / max(1.0, float(w)) * gx)))
    j = min(gy - 1, max(0, int(y / max(1.0, float(h)) * gy)))
    return (float(mo["p95"][j][i]), float(mo["median"][j][i]))
