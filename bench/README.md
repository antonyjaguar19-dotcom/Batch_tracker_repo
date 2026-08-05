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
- **bad tracks among the EXPORTS.** `--hazards` bakes defocused, repetitive and
  low-contrast regions into the source, and they do their job — but the seeding and quality
  stages reject those seeds long before export, so what actually ships is still uniformly
  good (0.03–0.09px). That is the pipeline behaving correctly, and it means this bench can
  prove a quality metric reports *false* signal but still cannot show it *ranks* real
  failures correctly. Judging a ranker needs bad tracks that survive to the export.

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

Then three more in the certainty gate, all found by asking why accurate tracks were being
condemned:

- **Certainty `0.0` meant "not measured", and the gate read it as "measured, and terrible".**
  `_refine_one_multi` reports `min(seg_certs)`, and `seg_certs` only gains an entry when a
  segment refines with reason `"ok"`. A track whose segments all come back `"no-anchor"`
  keeps its input points *by design* ("better raw than deleted") and scored 0.0. That
  assumption predates `moving_tile_refine`: those points are now native-resolution
  moving-tile positions, so they are good. On `lab03`, **23 of 32 tracks scored exactly
  0.0000 while being accurate to 0.044px** — indistinguishable from the 9 scoring 0.79–1.00.
  Worse, the 0.0-versus-real chasm read as a textbook bimodal split, so the gate overrode its
  own `max_cut` rail and dropped all 23. Now `NaN` = unknown: excluded from the distribution,
  never dropped for lacking a measurement. `lab03` went from **23 of 32 shipped flagged weak
  to 1**.
- **The relative bar cut a continuum.** With no clean split, `rel * P90` slices an arbitrary
  point off one population. On `lab02` that cut 13 of 23 tracks accurate to 0.06px, where
  certainty correlates with true error at only **−0.22** — a cut made on noise. The relative
  bar now applies only when a split is actually present; the absolute floor always does.
  `lab02` went from **13 weak flags to 0**.
- **A narrow spread skipped the absolute floor.** The "spread too narrow to separate
  anything" guard returned early, past the floor as well as the relative bar — so a uniformly
  defocused plate, every track alike and every one too soft to trust, passed the gate
  untouched. That is precisely the case the floor exists for. Both paths now fall through to
  it.

No change moved accuracy: every class scores identically before and after on both shots, as
it should — all of these are diagnostic and gating, not tracking. Verified against a six-case
matrix: a genuine soft/sharp split still cuts the soft cluster, a uniformly soft plate now
gets cut by the floor (capped so the shot is not emptied), and unmeasured tracks are kept
without being judged.

**Still open:** the wobble metric's remaining 0.363px floor on ground truth (neighbour
consensus is the fix, prototyped at ~0.09px), and a bench whose *exported* tracks include
genuine failures, without which no ranker can be validated.

## Comparing two runs on real footage: one trap, measured

Neighbour-relative wobble — each track's motion minus the median motion of its nearest
neighbours — is the right idea for footage with no ground truth, because the neighbours
carry the camera move and subtracting it leaves the track's own error. It is **not
comparable between two runs whose exports differ**, and the failure is not subtle.

Measured on SH004 (real plate, sharp subject on a defocused background), comparing the
pre-change tracker against the current one:

- whole-export medians said 1.66px -> 2.60px, i.e. a large regression;
- the same 16 features present in BOTH exports said 7 got worse and 1 better;
- but the positions of those tracks were **bit-identical between the two runs**
  (0.000000px, over 134 and 160 frames). Nothing about them had changed.

The two exports shared only 19 of 47/41 track ids, so each track was being judged against a
*different set of neighbours* — a different consensus, hence a different residual, on an
unchanged track. Recomputed with the shared ids as a fixed neighbour pool, the medians are
4.70px versus 5.00px: no regression, mixed direction, within noise.

So: hold the neighbour pool fixed across the runs being compared, and check whether the
positions actually differ before believing any per-track delta. A whole-export median of
this metric compares two different measuring sticks.

## What predicts real error, and one fix that did not work

`tools/verify_against_lk.py` re-tracks every exported track independently, so on SH004 each
of them has a measured error. Against those numbers:

| predictor | pearson | spearman |
|---|---|---|
| `score` | **+0.398** | +0.291 |
| `certainty` | −0.236 | −0.133 |
| `wobble` (after the S-G fix) | +0.127 | **+0.573** |

`score` is **positively** correlated with error — it ranks bad tracks as good, because its
coverage term is weighted 0.5 and long tracks are where drift accumulates. Wobble is the
only usable ranker, and only became usable after the moving-average detrend was replaced.

That suggested capping the backfill: only pad a thin export with rejects whose wobble is
near the tracks that passed. **Measured, it made things worse** — SH004 went from a 2.34px
median over 24 measurable tracks to 2.61px over 10, and the export fell from 41 tracks to 13.

The reason is worth keeping: the worst track on that shot, 20.51px against a reference
closing to 0.71px, **passed the certainty gate**. It was never a backfill reject, so a
backfill cap could not touch it, while good backfilled tracks (0.32px, 0.63px) were cut.
Bad tracks are entering through the GATE, not through the padding, and a simulation that
ranks the whole export cannot be used to justify a change that only filters part of it.
The open question is a gate that would reject a 20px track, which certainty at −0.236 does
not do.

### The wobble gate did not answer it either

Wobble ranks error better than anything else available (spearman +0.573), so the obvious
next step was to gate on it: drop tracks deviating more than 1.5x the shot's own median.
Simulated on the 24 measurable SH004 tracks that gave 2.34px -> 1.08px with nothing above
3px. **Run for real it cut 14 of 29 tracks and left the worst one exactly where it was:**

    median 2.76px   p90 7.65px   worst 20.51px   over 3px 5/12

`BWD_0247` — the 20.51px track — survived, and it was always going to. Wobble is deviation
from a track's OWN smooth path, and that track drifts *smoothly*: it slides off its feature
a fraction of a pixel per frame and never jitters. Its wobble is near zero and its error is
the largest in the shot. Wobble is structurally blind to smooth drift, which is precisely
the failure mode that produces the worst numbers here.

Rank correlation was not enough to justify a threshold. +0.573 over a whole population says
wobble is *usually* informative; it says nothing about the tail, and the tail is the only
part a gate acts on. Two changes have now been reverted for the same reason — a statistic
computed over the whole export used to justify a rule that fires on a subset of it.

The gate remains in the code as a slider, defaulting to **off**, because it does remove
visible jitter when that is the complaint.

### What did catch it: seed identity

One NCC between a track's patch at its FIRST frame and at its LAST. A point that slid off
its feature ends on unrelated pixels; a point that held on still matches even when lighting
and perspective have moved. On SH004 the drifter scored **−0.055 while every other exported
track scored 0.50 to 0.99** — separated by a margin no threshold has to be tuned to find.

    before   median 2.34px   p90 5.44px   worst 20.51px   over 3px 7/24
    after    median 2.28px   p90 5.19px   worst  8.19px   over 3px 6/23

Worst-case error down 60% for the loss of exactly one track (41 → 40), with median and p90
both improving. On `lab03`, where every track is good, it drops nothing and every class
scores identically — so it is an outlier catch, not a tax on healthy footage.

It is used as an outlier test, never a ranker: pearson −0.81 against measured error in the
tail, but spearman only −0.38 through the middle. It answers "is this still the same
feature" and nothing finer, and the floor is **absolute** (0.25) because that question means
the same thing on every plate — unlike certainty, which is why that bar must self-calibrate
and this one must not.

The ordering is the part worth remembering. The first attempt ran it after the certainty
gate but before `backfill_to_floor`, which draws from a pool snapshotted *earlier* — so the
drifter was dropped and immediately re-added, and the run came out byte-identical. That is
the same failure as the backfill ceiling and the wobble simulation: **a filter applied at one
stage while the bad track re-enters through another.** Three times on one task. When adding
a gate here, the question to ask first is not "does this test work" but "what else can put
the track back".

## Where the remaining error is, and where it is not

With the identity gate in, SH004 delivers a median of 2.28px against an independent
re-track. That residual is **not** a filtering problem — every remaining lever was measured
and none of them help:

| change | median | worst | verdict |
|---|---|---|---|
| **shipped** (41px box, adaptive peak fit) | **2.22px** | **8.19px** | best measured |
| backfill quality ceiling | 2.61px | 20.51px | reverted |
| wobble gate on | 2.76px | 20.51px | off by default |
| pattern box 31px | 2.64px | 16.59px | not shipped |
| pattern box 61px | 2.92px | 13.84px | not shipped |
| template_frames 9 (from 5) | 2.29px | 8.19px | not shipped |
| match_ambiguity_ratio 0.80 (from 0.90) | 2.38px | 8.19px | not shipped |
| min_corner_anisotropy 0.15 (from 0.08) | 1.25px* | 4.03px* | per-shot dial, not a default |

\* trustworthy band (reference closure < 0.2px), against 1.01px / 8.19px shipped.

The box-size curve has a clear minimum at 41px, which is exactly what `shot_profile.tune()`
already chooses for a soft plate — so that rule is correct and does not need revisiting.
More template averaging, which the UI calls the best-value setting on a grainy plate, is
also already at its best value: 9 frames moved the median the wrong way and put more tracks
over 3px, while improving p90. Both dials are at their optimum.

Two changes did survive measurement, and both attack localisation rather than selection:
the **identity gate** (worst track 20.51 -> 8.19px) and the **adaptive sub-pixel fit**
(median 2.28 -> 2.22px, and equal-or-better on every synthetic peak width).

### That 2.22px is not the tracker's error

It is the *disagreement between two trackers*, and it inherits whatever the LK reference got
wrong. `verify_against_lk` calls a row usable when its closure is under half the
disagreement, which lets a reference that is only good to 0.9px sit in the median unremarked.
Stratifying by how tight the reference actually is:

    all usable rows        n=25   median 2.22px
    closure < 0.5px        n=20   median 1.35px
    closure < 0.3px        n=13   median 1.25px
    closure < 0.2px        n=12   median 1.01px   <- reference good to ~0.2px

**Over half of what was being reported as tracker error was the reference's own
imprecision.** Where the reference can be trusted the bot sits at ~1px, with the best tracks
at 0.37-0.62px. The tool now prints this breakdown, because the single headline figure
flattered the reference and slandered the tracker — the same class of mistake as the wobble
metric that measured the plate, caught this time in the measuring instrument rather than in
the pipeline.

Read the tightest band that still has rows in it. The `closure < 0.1px` line is noisier than
the one above it (n=5, and LK closes tightest on the low-contrast features where it is also
least accurate), which is why the bands are printed rather than reduced to one number.

What is left is sub-pixel localisation on defocused features. Nine hypotheses have now been
measured against real footage; six were rejected outright, one was a correction to the
measurement itself, and the two that shipped moved the worst case a long way.

The last one is worth keeping because it rules a cause OUT. The worst surviving track on
SH004 (`BWD_0037`, 8.19px against a reference closing to 0.19px) has an identity score of
0.935 -- it ends on a patch that matches the one it started on -- which is the signature of a
point that snapped to an identical neighbour. Tightening `match_ambiguity_ratio` from 0.90
to 0.80 is the setting built for exactly that, and it cost 10 tracks, moved the trustworthy
median from 1.01px to 1.54px, and left `BWD_0037` completely unchanged. So that track is not
a look-alike snap, and the rival-peak defence is already tight enough on this footage. The remaining error does not respond
to thresholds or to the accuracy dials, which points at the model rather than the
configuration — TAPNext's fixed 256px stage and NCC's behaviour on low-frequency texture
are the two candidates, and neither is reachable from a settings file.

The durable result is the loop itself: `tools/verify_against_lk.py` re-tracks each exported
track with a different algorithm (pyramidal LK) and reports per-track error with the
reference's own round-trip closure beside it, so a claim can be checked in ten minutes.
Five plausible changes would have shipped on reasoning alone without it.
