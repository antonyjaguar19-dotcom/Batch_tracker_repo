# Hybrid tracking: TAPNext seeds and replants, Blender tracks

Question asked: the bot already seeds with TAPNext and re-acquires after occlusion. What
if the per-frame *measurement* in between were done by Blender's movie-clip tracker
instead of the bot's own refine chain?

Answer: **it replaces them.** On exact synthetic ground truth Blender takes raw TAPNext
from 2.71px to 0.05px, where the bot's own two refine stages reach 0.06px — so the
expensive half of the bot run can be dropped rather than added to. On SH004 that is 165s
instead of 580s, for 122 tracks instead of 40 and 10.8x the exported point-samples.
Nothing in `app/` was modified.

The split under test:

| stage | engine | why |
|---|---|---|
| which features, and where they start | TAPNext (the bot's own seeder + passes) | globally robust; picks trackable points |
| where a dead feature reappears | TAPNext | it carries a position through an occlusion; a local tracker cannot |
| every per-frame measurement between those | Blender | classic NCC/affine planar tracker with sub-pixel refinement |

## What is proven

Measured on Blender 5.2.0 LTS (portable build, hash `fbe6228777e7`).

| Step | Result |
|---|---|
| Blender tracks with **no GUI at all** | works — `--background`, `track_markers` returns `FINISHED` |
| the operator is synchronous there | yes — it runs to completion inside the call, not modally |
| inject external seed points | works — `tracks.new(frame=N)` + `markers[0].co` |
| seed lands where asked | **max round-trip 0.0001px**, checked on every run |
| track them | works — one operator call per group, not per track |
| replant a dead track from the guide | works — 10 resumes over 4 rounds on SH004 |
| gaps survive | yes — muted/absent markers are dropped, so the export has real holes |
| speed | 160 frames x 40 trackers in **17.6s** (256 tracker-frames/s) |

Unlike the SynthEyes hybrid, this path **never touches the mouse and needs no licence**.
That was the blocker there (see `experiments/hybrid_seed/FINDINGS.md`), and it is simply
absent here.

## Accuracy

### Synthetic, exact ground truth — `bench/synth/lab02`, 2560x1440, 100 frames

Seeded from the bot's own `runs/base` export, so both sides are tracking the same 23
features on the same pixels; the only difference is who does the per-frame measurement.

| run | ALL mean_err | p90 | worst track |
|---|---|---|---|
| `base` — the shipping bot | 0.06px | 0.11px | 0.12px |
| `bl_kind` — hybrid, per-class geometry | **0.05px** | **0.08px** | **0.10px** |
| `bl_flat` — hybrid, one geometry for all | **0.05px** | **0.07px** | **0.10px** |

Per class the hybrid is level or slightly ahead everywhere except `dense-corner`
(+0.01px) and `flat` first-step (+0.04px). `fast`/`slow` are 1.00 on every class, so
neither engine shows the "pulled toward where the point was" damping that
`refs/gt4k/baseline.json` records. Peak locking is +0.004px, the same as the bot.

**Per-seed tracker geometry buys nothing measurable.** `bl_kind` maps the bot's seed class
onto Blender's pattern size, search size and motion model; `bl_flat` gives every tracker
the same box. They score identically, and the *flat* control is marginally better on p90.
This was the strongest argument for the SynthEyes hybrid and it does not survive contact
with a measurement here. Caveat: lab02 is one plane with uniformly good features, which is
the case where a per-class rule has least to do.

### Real plate — SH004, 2560x1440, 160 frames, vs `refs/SH004_lk`

```
ref052  corner   0.72 -> 0.75      (reference closure 0.15px)
ref069  corner  22.48 -> 21.75     (reference closure 0.27px)
ref014, ref017, ref048            MISSED by both
scored 2/5
```

Read carefully, this measures very little:

- Three references are MISSED by **both** engines, because the guide only holds 40 tracks
  and none land within 25px of those features. That is the bot's seeding/gating, inherited
  wholesale by the hybrid — not a difference between the two.
- `ref069` is 22px off on **both**, against a reference whose own closure is 0.27px. A
  22px deviation with a 20px first-step is the proximity pairing latching onto a different
  nearby feature, not a tracker verdict. Excluded.
- That leaves **one** usable row, `ref052`: 0.72px bot vs 0.75px hybrid. A tie, and both
  are five times the reference's own precision.

So the real-plate comparison is not a result. `bench/` is where the accuracy claim above
comes from, and it is worth remembering what that cannot cover: one plane, so no parallax
and no occlusion.

### Replant

Replant is the half the synthetic bench cannot test at all — lab02 has nothing to occlude,
and it fired **0** times there. On SH004:

| run | median track length | full-length tracks | tracked frames |
|---|---|---|---|
| `--no-replant` | 97 / 160 | 17 / 40 | 4239 |
| replant on (10 resumes, 4 rounds) | **152 / 160** | 17 / 40 | 4519 |

Median span rises 97 → 152 frames. The count of full-length tracks does not move, which is
the expected shape: replant does not rescue a track that was already clean, it extends the
ones that died partway. Whether the resumed segments are *accurate* is not established —
neither reference could measure it.

## Blender replaces both refine stages — the result that matters

The bot spends most of a run refining: on SH004, `moving_tile_refine` took 5.5 min and
`pattern_refine` 1.9 min of a 9.7 min run. Asking Blender to do that job instead, on
`bench/synth/lab02` with exact ground truth:

| run | mean_err | p90 | worst | tracks over 1px |
|---|---|---|---|---|
| bot, **refines off** (`moving_tile=0 pattern_refine=0`) | **2.71px** | 5.02 | 6.66 | **23 / 23** |
| bot, full refine chain | 0.06px | 0.11 | 0.12 | 0 / 23 |
| **Blender tracking that same raw guide** | **0.05px** | **0.07** | 0.13 | 0 / 23 |

Raw TAPNext is unusable on its own (every track over 1px). The bot's own refine chain
takes it to 0.06px. Blender takes the *same raw input* to 0.05px — it is a complete
substitute for both stages, not an addition to them.

That changes what the hybrid costs. It no longer needs the expensive bot run it was
seeded from:

### SH004, 2560x1440, 160 frames — the same shot, three ways

| | bot, as shipped | bot raw, dense | **hybrid** (dense raw guide + Blender) |
|---|---|---|---|
| time | 580s | 139s | **139 + 26 = 165s** |
| tracks | 40 | 122 | **122** |
| median track length | 20 frames | 134 | **152 / 160** |
| median span (first→last) | 20 frames | 134 | **160** |
| point-samples exported | 1457 | — | **15697** |
| full-length tracks | 2 | 38 | 27 |

**3.5x faster, 3x the tracks, 10.8x the usable point-samples.** The shipping run's 40
tracks have a median length of *20 frames* on a 160-frame shot, because `defragment` split
16 of them into short continuous runs (28 → 40 in the log). The hybrid's tracks carry
gaps instead of being cut into pieces, which is what 3DE ASCII is for.

Two things the dense guide needed, both config, no code:

- `track_spacing_px=15` (default 60, scaled by plate width → 80px at 2560). Spacing was the
  binding constraint, not quality: the log reads `1278 past quality bar -> 28 after
  spacing`. At 15 it is `-> 122`.
- `moving_tile=0 pattern_refine=0`, since Blender is doing that work.

## Blender-side tuning is already at its optimum

`sweep.py` runs ten configurations against exact ground truth. Every "smarter" option is
worse than the plain defaults:

| config | mean_err | p90 | worst |
|---|---|---|---|
| **default** (Loc/LocScale per class, KEYFRAME, brute, corr 0.75) | 0.050 | **0.080** | **0.100** |
| `locscale`, `srch_big`, `corr_high` | 0.050 | 0.080 | 0.100–0.110 |
| `affine` | 0.050 | 0.100 | 0.300 |
| `prevframe` | 0.060 | 0.090 | 0.150 |
| `perspective` | 0.060 | 0.100 | 0.360 |
| `affine_prev` | 0.060 | 0.150 | 0.440 |
| `pat_big` | 0.070 | 0.110 | 0.210 |

The pattern is consistent: **every extra degree of freedom costs accuracy.** Affine and
Perspective have the freedom to explain a bad match by deforming the patch, and their worst
tracks are 3–4x the default's. `PREV_FRAME` matching drifts, as expected. A bigger pattern
box averages the feature away. Nothing here is worth changing.

## The disagreement gate does not work, and why

Two independent trackers on one feature is a quality signal neither engine has alone, so
`--max-guide-dev` truncates a track where Blender and TAPNext stop agreeing. Measured on
SH004 against the dense raw guide:

| threshold | tracks cut | median length after |
|---|---|---|
| off | 0 | 152 |
| 8px | 104 / 122 | 52 |
| 15px | 77 | 71 |
| 25px | 45 | 93 |
| 40px | 27 | 139 |

It destroys the run at every useful setting. The reason is structural: the guide is *raw*
TAPNext, which the bench measures at 2.71px mean and 6.66px worst — so when the two
disagree, the guide is usually the one that is wrong, and the gate trims the better
measurement. It would need a refined guide to arbitrate, and the refine is exactly what
this arrangement removed. **Left off by default**, kept because the negative result is worth
having written down.

## Do the disappeared tracks come back on the right feature?

This was the open question above, and the answer is **partly, and it needed a bug fixed
first**. Longer tracks are worthless if the resumed half is following something else, and
nothing in the span table can tell the two apart.

`check_replants.py` measures it without touching the guide — the guide is where the resume
position came from, so checking against it would confirm anything. For each internal gap it
takes the patch on the last frame before it, searches the neighbourhood of the resume
position on the first frame after it, and reports how far the best match is from where the
track actually resumed. Each gap is paired with a **control** on the same track over the
same number of frames with no gap in it, so "NCC 0.65" is read against what that plate
scores over that separation anyway.

The metric is self-checked on every plate (`--selfcheck L` punches synthetic gaps into
*continuous* tracks, where the answer is known to be ~0px):

| plate | self-check "on the feature" | median offset |
|---|---|---|
| SH004 | 100% | 0.35px |
| SH016 | 98% | 0.17px |
| SH008 | 100% | 0.19px |

### The bug: the resume teleported the track

The first version planted the resumed marker at **the guide's absolute position**. By the
time a track dies the two trajectories have long since diverged — measured on SH004, Blender
and the guide sit a median **6px** apart on the last frame before the gap, p90 **35px**, max
**242px**. So every replant threw away Blender's localisation and jumped onto whatever
TAPNext happened to be following.

Applying the guide's **displacement** to Blender's own last good position instead:

| SH004 | on the feature | clearly wrong | median offset | NCC vs control |
|---|---|---|---|---|
| guide's absolute position | 13% | 57% | 7.35px | 0.579 vs 0.851 |
| **guide's displacement** | **41%** | **22%** | **2.75px** | **0.812 vs 0.873** |

`--replant-absolute` keeps the old behaviour as the control.

A second change — widening the search box on the resume marker and **not exporting the
resume frame itself**, since it is an estimate rather than a measurement — came out
**neutral** (44% / 25% / 2.89px, inside the noise). It is kept anyway, because publishing an
estimate as though it were tracked data is wrong whether or not it moves the number, but it
is not an improvement and is not claimed as one.

### Where it stands, on three shots

| | SH004 2560x1440 | SH016 4096x2160 | SH008 1920x1080 |
|---|---|---|---|
| frames | 160 | 127 | 258 |
| tracks | 122 | 228 | 137 |
| gaps measured | 168 | 225 | 249 |
| unjudgeable (repetitive texture) | 48% | 39% | 27% |
| **reappearance on the feature** | 44% | **26%** | **47%** |
| off by 1.5–6px | 31% | 35% | 28% |
| **clearly on the wrong thing** | 25% | **39%** | 25% |
| median offset | 2.89px | 4.13px | 1.82px |
| NCC across gap vs control | 0.78 / 0.83 | 0.65 / 0.86 | 0.75 / 0.88 |

**So: no, reappearance is not working reliably.** Roughly a quarter to a half of gaps resume
on the right feature and a quarter to two-fifths resume on the wrong one. SH016 is the worst
of the three — a 4K plate where the NCC across a gap (0.65) falls far below its own control
(0.86), meaning the resumed patch genuinely does not look like the one that disappeared.

It is still much better than what it is built on. The same measurement on the raw TAPNext
guide's own gaps:

| | SH004 | SH016 | SH008 |
|---|---|---|---|
| guide, on the feature | 45% | 17% | 14% |
| guide, clearly wrong | 18% | 60% | 62% |
| guide, median offset | 3.50px | 12.93px | 16.40px |
| **hybrid, median offset** | **2.89px** | **4.13px** | **1.82px** |

The hybrid roughly halves to a quarters the offset on the two hard shots. But note the
comparison is against the **raw** guide: the bot's own re-acquisition logic lives in
`pattern_refine` (`refine_ncc_reacquire`), which this configuration switches off. The
shipping bot with refines on produced **zero gaps** on SH004 — it does not re-acquire so
much as decline to, letting `defragment` split the track into short continuous runs instead
(median 20 frames). That is the honest comparison: the bot avoids the problem, the hybrid
attempts it and gets it right about half the time.

### What would fix it

The resume position is a pure prediction — guide displacement, never checked against the
plate. The obvious next step is to snap it to the NCC peak of the pre-gap patch before
tracking on, which is exactly what `refine_ncc_reacquire` does inside the bot. That cannot
be done inside `bl_track.py` (Blender's Python has no cheap pixel access at 4K), so it needs
the replant loop moved out of Blender and into the orchestrator, with Blender re-invoked per
round. Not attempted here.

## Two traps, both measured

**1. `scene.frame_set()` does not move the frame the tracker reads.**
The clip editor keeps its own frame in `space.clip_user.frame_current`, which normally
follows the scene through a UI redraw. In `--background` there is no redraw, so it stays
pinned at 1 while `scene.frame_current` says 25 — and `track_markers` reads the *space*.
Every call therefore started from frame 1.

That alone would be a visible failure. What made it dangerous is the second half: Blender's
marker lookup returns the **nearest** marker rather than failing, so a track seeded at
frame 67 was silently re-anchored onto frame 1 and tracked forward from there. The result
was a clean, full-length, plausible-looking 100-frame track that had never touched its own
seed.

Measured both ways, in isolation: space frame 1 → span 1..100; space frame 25 → span
25..100. The fix is one line, `space.clip_user.frame_current = frame`, alongside the
`scene.frame_set` that looks like it should be sufficient.

**The bench could not see this.** Its ground truth is anchored on each track's own seed
(`bench/README.md` says so explicitly), so a track planted on the wrong feature entirely
still scored 0.05px — a *better* number than the correct run produced later. The **seed
round-trip check** is what caught it, at 218px, which is why it now runs on every run
instead of being a script somebody remembers. This is the third time in this repo that a
metric looked plausible while measuring the wrong thing.

**2. Selection is honoured — the obvious suspect was innocent.**
Blender counts a track as selected if any of `select` / `select_pattern` / `select_search`
is set, which made "the operator is tracking everything" the natural first theory for the
above. It is wrong: measured, `track.select = False` alone does deselect (its RNA setter
clears the other two), and a deselected track is left with its single seed marker
untouched. Recorded because it cost time, and because the code sets all three flags and
that could otherwise read as a fix for something real.

## What this does not settle

- **Whether the longer tracks are right.** The hybrid's median span is 152 frames against
  the shipping bot's 20, and the bench says its localisation is good — but the bench has no
  occlusion and no movers. A track that stays long *because it is drifting through an
  occluder* would look exactly like this in the span table. The one clean real-plate
  reference pairing cannot separate them.
- **Occlusion accuracy.** Replant demonstrably extends spans; nothing here shows the
  resumed segments land on the right feature.
- **Dense output makes `eval_refs` less reliable, not more.** At 122 tracks the
  proximity pairing starts matching a reference to a *neighbouring* feature: the dense run
  scores 4/5 instead of 2/5, but the new rows carry 12–16px first-step, which is the
  signature of a wrong pairing rather than a wrong track. Read `first_step` before
  believing any `mean_err` on this reference set.
- **Parallax and movers.** Neither the bench (one plane) nor the surviving SH004 reference
  touches it.
- **Mask gating.** The hybrid ignores SAM3 mattes entirely. Every gating decision the bot
  makes is inherited through the seeds and then not re-applied.

## How to re-run

```bat
REM synthetic, exact truth -- no GPU needed, seeds come from an export already on disk
runtime\python311\python.exe experiments\blender_track\run_blender_hybrid.py ^
    --plate bench\synth\lab02\plate --name lab02 ^
    --reuse-tapnext bench\synth\lab02\runs\base\lab02__tapnext.txt ^
    --tag kind --out bench\synth\lab02\runs\bl_kind\lab02__tapnext.txt
runtime\python311\python.exe bench\score_synth.py bench\synth\lab02 --run bl_kind --baseline base

REM Blender vs the bot's own refine chain, on exact truth
runtime\python311\python.exe bench\run_bench.py --shot bench\synth\lab02 --tag raw ^
    --set moving_tile=0 --set pattern_refine=0
runtime\python311\python.exe experiments\blender_track\run_blender_hybrid.py ^
    --plate bench\synth\lab02\plate --name lab02 ^
    --reuse-tapnext bench\synth\lab02\runs\raw\lab02__tapnext.txt ^
    --tag rawseed --out bench\synth\lab02\runs\bl_raw\lab02__tapnext.txt
runtime\python311\python.exe bench\score_synth.py bench\synth\lab02 --run bl_raw --baseline base

REM the tuning sweep (10 configurations, ~3 min, no GPU)
runtime\python311\python.exe experiments\blender_track\sweep.py --shot bench\synth\lab02

REM real plate: dense raw guide + Blender. This is the configuration in the table above.
runtime\python311\python.exe bench\run_bench.py --shot experiments\blender_track\out\SH004 ^
    --tag dense_raw --set track_spacing_px=15 --set moving_tile=0 --set pattern_refine=0
runtime\python311\python.exe experiments\blender_track\run_blender_hybrid.py ^
    --plate experiments\blender_track\out\SH004\plate --name SH004 ^
    --reuse-tapnext experiments\blender_track\out\SH004\runs\dense_raw\SH004__tapnext.txt ^
    --tag dense
```

Every run must print `seed round-trip : max <n>px PASS`. If it says FAIL, stop — the
trackers are not on the features they were given and no other number means anything.

## Files

| File | What it is |
|---|---|
| `run_blender_hybrid.py` | the e2e: bot seeder/guide -> Blender -> 3DE export, with the round-trip check |
| `bl_track.py` | runs inside Blender; seeds, tracks, replants. Imports nothing from this repo |
| `blio.py` | plate access, pixel<->clip coordinates, the Blender subprocess call |
| `sweep.py` | tunes the Blender side against exact ground truth, one row per config |
| `render_overlay.py` | burns hybrid + guide over the plate; in-gap points drawn hollow |
| `README.md` | how to run it |
