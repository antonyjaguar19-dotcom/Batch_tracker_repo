# SpatialTracker V2 on SH006 — smoke test

Downloaded, wired up, run on SH006, and exported to a `.blend`. This is a first run, not a
comparison: no accuracy claim below is measured against a hand track, and the only number
that means anything yet is the model's own internal consistency.

**Licence: CC BY-NC 4.0, NonCommercial** (`vendor/SpaTrackerV2/LICENSE.txt`, despite the
repo's Apache-flavoured README badges). Same position as MFT: fine for evaluating, cannot
ship inside a studio tool without permission from the authors. Read the rest of this with
that in mind.

## What it is, and why it is not a TAPNext replacement

TAPNext, MFT and the Blender tracker all answer *where did this pixel go* in 2D.
SpaTrackV2 answers it in 3D. A VGGT-derived front end predicts, for the whole window at
once, a camera pose per frame, an intrinsic per frame, and a metric point map; the tracker
then follows query points through that reconstructed space and reports a 3D position, a 2D
position and an occlusion score per frame.

So the deliverable is a **camera plus a moving 3D point cloud**, which is why the export
here is a `.blend` and the 3DE ASCII is a by-product.

## Setup

| | |
|---|---|
| upstream | `github.com/henry123-boy/SpaTrackerV2` @ `7e12274` → `vendor/` |
| weights | `Yuxihenry/SpatialTrackerV2_Front` (4.63 GB) + `SpatialTrackerV2-Offline` (276 MB) → `weights/`, 4.6 GB on disk |
| deps | `pydeps/` (286 MB): kornia, einx, sklearn, pycolmap, pyceres, moviepy, utils3d, decord, omegaconf, easydict, jaxtyping, flow_vis |
| GPU | RTX A4000, 16 GB, torch 2.5.1+cu121 |
| Blender | 5.2.0 LTS |

Nothing was installed into the app runtime and nothing in `app/` was touched. `pydeps/` is
prepended to `sys.path` inside `run_spatracker.py` because the embeddable interpreter is in
isolated mode and ignores `PYTHONPATH`. xformers is absent and unused — every import of it
in the vendored tree is already inside a `try`.

## Runs

SH006 is 312 frames of 3840×2160. The front end works at 518 px wide, so every frame is
fed at 518×294 and the 2D side is scaled back to plate resolution on export.

The self-check is the model against itself: take the 3D track, push it through the camera
the model predicted, and compare to the 2D track the same model produced. It cannot show
the reconstruction is *right*, only that it has not fallen apart — but that turned out to
be the thing worth watching.

| window | frames | front | track | total | reproj median (518-space / plate) | behind camera | visible |
|---|---|---|---|---|---|---|---|
| f1–48, step 1 | 48 | 16.9 s | 25.0 s | **77 s** | **4.2 px / 31 px** | 3.7 % | 84.8 % |
| f1–96, step 1 | 96 | 650.9 s | 385.8 s | 1090 s | 132 px / 977 px | 64.4 % | 72.1 % |
| f1–305, step 8 | 39 | 8.1 s | 12.8 s | 54 s | 6.8 px / 50 px | 80.7 % | 5.3 % |

### Two ways it breaks, both silent

**96 frames is past the edge on 16 GB.** The front end went from 16.9 s to 650.9 s — 38×
for 2× the frames. That is not attention cost, that is the allocator spilling into shared
host memory. It did not raise; it got slow and then produced a reconstruction where **two
thirds of the tracked points ended up behind the camera**, while still exporting a
perfectly well-formed `.blend`. This is why `run_spatracker.py` reprojects and prints
`DEGENERATE` rather than trusting a run that completed.

**Striding to cover the whole shot does not work.** Step 8 over all 312 frames looks
attractive — 39 frames, 54 s, whole-shot camera — and it collapses: 80.7 % of samples
behind the camera, 5.3 % of samples visible. It is still a *tracker*; an 8-frame jump is
past what it can associate. Covering a 312-frame shot means several overlapping ~48 frame
windows and stitching them, which is not built here.

The usable envelope on this machine is therefore **contiguous, ~48 frames**.

### The 48-frame window, read properly

- **31 px at plate resolution** is the model's own 3D↔2D disagreement. For scale, the bot's
  refine chain sits at 1.30 px against 4K ground truth. SpaTrackV2 is not competing for
  that job and should not be pointed at it.
- **The lens drifts.** `fx` goes 817.4 → 718.9 over 48 frames (12 %) — the front end
  re-estimates the intrinsic every frame and nothing ties them together. On a prime-lens
  plate that is wrong by construction. The export keys `lens` per frame so what the model
  actually said is visible in the file rather than averaged away.
- **Pixels are not square** in its estimate (fx 817.4 vs fy 811.9, 0.7 %). Blender has one
  focal value, so the export uses `fx`; that costs about a pixel in y and is most of the
  gap between the 4.2 px numpy check and the 5.2 px Blender check.
- **Baseline is 0.14 of median point depth** over the window, i.e. a small move — consistent
  with the shot, and a reminder that the depth here is mostly the network's prior, not
  triangulation.
- 95 of 144 points survive all 48 frames unbroken; the rest are broken by the occlusion
  score, which is a first-class output and does not have to be inferred from a tracker
  dying.

### The .blend is verified, not assumed

The one step that can be wrong in a way that *looks* right is the OpenCV→Blender camera
convention: flip it wrong and you get a mirrored world that behaves plausibly until
something real is matched to it. So `check_blend.py` opens the saved file, asks Blender's
own camera where each point projects, and compares against the model's 2D track:

```
5726 samples, median 5.16 px, p90 29.54 px at 518x294
```

Same order as the 4.21 px numpy check, so the file agrees with the model.

## What this is actually good for

Not per-track precision. What it gives that nothing else in this repo does:

- a **camera without a solve** — no seeding, no cleanup, no solve pass, 77 s from plate to
  animated camera;
- **depth for every pixel**, and 3D positions for moving points, so it is not restricted to
  the static parallax a matchmove solve needs;
- **an honest occlusion signal** per point per frame.

The obvious next question is whether that camera is worth anything as a *starting point* —
initialise a real solve from it, or use its depth to sort tracks into layers, rather than
using its 2D output at all. Nothing here answers that yet.

## Rerun

```bat
runtime\python311\python.exe "Spatial tracker\run_spatracker.py" ^
    --plate experiments\blender_track\out\SH006\plate ^
    --name SH006_f1-48 --start 1 --end 48 --grid 12
```

Writes into `Spatial tracker\out\`:

- `<name>__spatrack.blend` — camera, `SPT_####` empties keyed per frame and hidden while
  occluded, plate wired in as a camera background. Blender frame numbers are the plate's.
- `<name>__spatrack.npz` — world coords, c2w, intrinsics, visibility, 2D tracks.
- `<name>__spatrack_2d.txt` — 3DE ASCII at plate resolution, y-up, gaps where occluded.

Verify a file with:

```bat
"%BTR_BLENDER_EXE%" --background --factory-startup --python-exit-code 1 ^
    --python "Spatial tracker\check_blend.py" -- ^
    --blend "Spatial tracker\out\SH006_f1-48__spatrack.blend" ^
    --npz   "Spatial tracker\out\SH006_f1-48__spatrack.npz"
```

Watch the `[check]` line on every run. A median over ~10 px, or any noticeable fraction of
samples behind the camera, means the window was too long or too sparse and the `.blend` is
not worth opening.
