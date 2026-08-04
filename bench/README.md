# bench — synthetic ground truth for tracking accuracy

`refs/` holds hand-tracked references, which are the real thing and stay the final word.
They also have two limits this exists to cover:

1. **One track.** `refs/gt4k` has a single corner, and `eval_refs` says so on every run:
   "per-class conclusions need one per class". A rule that helps corners and hurts blobs is
   invisible in one number.
2. **The plate has to be on the box.** `refs/gt4k`'s footage is not on this machine, so its
   locked baseline can be re-read but never re-run. Nothing new can be measured against it.

A real plate frame warped by a **known homography** has neither problem. The scene is a
rigid plane, so the true position of any point on any frame is exact arithmetic — for every
seed the tracker chooses to make, with no upper limit and no human error term.

## What it does and does not measure

Measures: **localisation** — sub-pixel accuracy, drift, wobble, and the fast/slow damping
pair that `refs/gt4k/baseline.json` reads as "position pulled toward where the point was".

Does **not** measure, so never conclude these from it:

- **parallax** — one plane, nothing moves relative to anything else.
- **occlusion** — nothing passes in front of anything.
- **bad tracks** — every track on it comes out good (0.03–0.12px), so it can prove a metric
  reports *false* signal, but it cannot yet validate that a metric *ranks* tracks correctly.
  A bench with genuine failures is the next thing this needs.

A constant offset applied to a whole track scores zero error, because ground truth is
anchored on that track's own seed. That is a property of the scene rather than a gap: on a
plane, the point 1px from the corner is a valid scene point, and following it rigidly is a
correct observation. Seeding off the intended corner still shows up, as worse `mean_err`.

## Use

```bat
REM 1. build a shot (once). --hard adds noise + motion blur + exposure drift.
runtime\python311\python.exe bench\make_synth.py --out bench\synth\lab02 --frames 100 --hard

REM 2. run the production TAPNext path over it
runtime\python311\python.exe bench\run_bench.py --shot bench\synth\lab02 --tag base
runtime\python311\python.exe bench\run_bench.py --shot bench\synth\lab02 --tag test --set per_track_policy=1

REM 3. score, per feature class, against exact truth
runtime\python311\python.exe bench\score_synth.py bench\synth\lab02 --run test --baseline base
```

`run_bench.py` calls `app.py:_track_shots_tapnext` rather than building a `RunnerConfig`
itself — that mapping is ~80 fields wide and changes whenever a setting is added, so a bench
that re-implemented it would be measuring a configuration nobody ships. `--set FIELD=VALUE`
reaches any `AppState` field by name.

Metric definitions are shared with the hand-track path (`fast` / `slow` / `jerk` come
straight out of `app.compare_tracks.compare`), so a column here means the same thing as the
same-named column in `refs/gt4k/baseline.json`.

Renders are PNG and the noise seed is fixed: a bench whose input changes between runs cannot
attribute a delta to a code change, which is the only thing it exists to do.

## What it found

Baseline on `lab02` (2560×1440, 100f, noise+blur+exposure): **0.06px** median error, 0/23
tracks over 1px. Planar localisation is not where the quality problem is.

Two defects in the **quality signals**, both proven against exact truth:

- **`measure_wobble` was reporting the plate's motion, not tracking error.** Fed the exact
  ground truth — tracks that are correct by construction — it reported **1.585px** of wobble,
  the same number it gave the bot. Cause: it detrended with a moving average, which is exact
  only for constant velocity and so leaves a residual proportional to path curvature. Now a
  local quadratic (Savitzky–Golay) detrend, exact for constant acceleration: ground truth
  reads **0.363px**, and the pipeline's own log line dropped 1.585 → 0.364. The remainder is
  genuine handheld jitter that no single-track metric can separate from error; measuring each
  track against its neighbours' consensus gets it to ~0.09px and is the way to finish this.
- **The certainty gate's "two populations" test fired on ordinary spread.** Largest-gap alone
  does not establish bimodality — in a sparsely sampled tail it is routinely several times
  the typical spacing. On `lab02` a 0.075 gap (3.7× the median gap) isolated the top **3 of
  23** tracks, and because a "clean split" is allowed to override the `certainty_max_cut`
  rail, it dropped **20 tracks that were all accurate to 0.06px**. A split now also requires
  both sides to hold ≥25% of the tracks. A genuine soft-background split still cuts (verified
  on a bimodal sample); the sparse-tail artefact no longer does.

Neither change moved accuracy: every class scores identically before and after, as they
should — both are diagnostic/gating, not tracking.

**Still open:** even after the fix the relative bar (`certainty_rel` against P90) drops 13 of
23 tracks that are accurate to 0.06px. Export count is unaffected because backfill tops it
back up, but those tracks ship flagged `weak`, which misinforms the artist. Whether the
relative bar should apply at all when the numbers show no split is a shipping-behaviour
decision, and validating it needs a bench that contains genuinely bad tracks.
