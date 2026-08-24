# Tracking Assistant — AI-assisted 2D tracking inside Blender

An addon for Blender's Movie Clip Editor. A model decides **where** the trackers go and
roughly how the plate moves; **Blender measures every frame**, at full plate resolution.
Tracks go in and out as classic 3DE 2D-track ASCII.

Everything it needs lives in this folder. Nothing in `app/` is touched.

---

## What it does, honestly

| | |
|---|---|
| **Auto-seed** | TAPNext++ picks features and supplies a motion guide; Blender's tracker measures them. Working. |
| **3DE import / export** | Round-trips exactly (0.000000000 px, 15697 samples). Working. |
| **Repair / re-acquire** | Designed, not built. See *Not built yet* below. |
| **Mover rejection** | Method decided and off by default — it has no reference set to be scored against yet. |

**Accuracy: 2.20 px against artist hand tracks** on real footage (366 frames, 31 % of frames
within 1 px), and sub-pixel on the features it holds. The synthetic bench says 0.05 px and
is optimistic by roughly 40× — it rated another method at 0.16 px where real footage
measured 16.90 px. Quote the 2.20.

**Robustness is the weak axis, not precision.** Blender's tracker dies 1.0–2.3 times per
track where the AI guide dies 0.3. This tool makes a matchmover faster; it does not replace
one.

---

## Install

```bat
bootstrap.bat
```

Downloads CPython 3.11.9 + torch cu121 + TAPNext++ into this folder (~9.6 GB, once), then
builds and validates the addon zip. It prints a PASS/FAIL row per dependency and exits
non-zero if any row fails.

Then in Blender: **Edit ▸ Preferences ▸ Add-ons ▸ Install from Disk** ▸
`dist\btr_assist-0.1.0.zip`, and enable it. The panel is in the **Movie Clip Editor**
sidebar (press <kbd>N</kbd>), tab **Assist**.

Other modes:

```bat
bootstrap.bat --check         REM report what is present, download nothing
bootstrap.bat --reuse-repo    REM use the bot's runtime/python311 instead of building one
```

### "Self-contained" means the runtime, not the repo

Auto-seed imports `app.tracker_core` and `app.track_meta` from batch_tracker. Vendoring
copies of those would mean a second copy of an ~80-field config that changes whenever a
setting is added — the exact problem the parity gate below exists to police. So **the repo
is a hard dependency of auto-seed**, in both modes.

3DE import/export needs none of it and works on a bare Blender with no sidecar at all.

### If you have the older addons installed

`experiments/blender_track/addon_3de_io.py` registers `clip.export_3de`. This addon uses
`clip.btr_*` throughout so the two cannot collide, but there is no reason to run both.
(That older one also has a live bug on Blender 5.2 — its "Delete existing tracks" calls
`tracks.remove()`, which does not exist.)

---

## Using it

1. Load your plate in the Movie Clip Editor.
2. **Assist ▸ Auto-seed ▸ Start** — spawns the model process. It reports torch, CUDA and
   free VRAM. It shuts itself down when Blender exits.
3. **Place seeds only** — puts the markers down and stops, so you can look at where the
   model chose before spending the tracking time. Cheap, and worth doing first.
4. **Auto-seed and track** — same, then tracks. The progress line reads
   `frame 87/160  114 live  8 dead  3 clamped`; <kbd>Esc</kbd> cancels between frames.
5. **Assist ▸ 3DE 2D tracks ▸ Export 3DE file**.

### Track + re-acquire, and the box the track carries

Select your markers and press **Track N selected + re-acquire**. Location *and scale* are
animated, so Blender solves a size for the pattern box on every frame — and that size is
watched. A tracker sliding off its feature does not announce it: the position stays
plausible and the track stays alive, but the box swells, because correlation is being
satisfied by the surroundings instead of by the feature.

So a box that changes size unusually fast (>10 % in a frame), or wanders past 1.6× the box
you set, **stops that track on the spot**. The patch from the frame you seeded is then
correlated against the plate there — at its own size, and resized by exactly how much the
box grew — and the two scores decide:

| what the plate says | what happens |
|---|---|
| your patch is not there at any size | the frames back to where the box started growing are dropped, and re-acquire takes over from the last frame your own box measured |
| the patch matches better *resized* | the feature really did approach camera: the box is kept and tracking carries on |
| the patch is there at your size, the box is not | your box is put back and tracking carries on from the position Blender measured |
| the patch is there and the box is near enough | false alarm; nothing is touched |

The marker is **never moved** by this and only the first row deletes anything. That is a
measurement, not caution — see FINDINGS, "a metric measured and REMOVED before it shipped".
Two toggles and the growth limit are in the panel; turn `Animate loc + scale` off and the
whole thing is inert, because under `Loc` the box never changes.

Two things the UI does silently on your behalf:

- **Full resolution is forced** while a job runs, then your proxy setting is put back. A
  50 % proxy would halve tracking precision with nothing on screen to show for it.
- **Every tracking setting is written explicitly** per job. The headless path is protected
  by `--factory-startup`; your Blender is not, and a scene left on Affine/PREV_FRAME would
  quietly reproduce the worst configuration in the sweep.

### The warnings box is not decoration

It flags a proxy, a relative clip path, and a clip reporting one frame. Two of those
produce a plausible-looking track file that is quietly wrong.

---

## Defaults, and what they cost to change

| setting | value | why |
|---|---|---|
| `spacing_px` | **15** | Spacing caps the count, not quality. One shot logged `1278 past quality bar -> 28 after spacing` at the default 60; at 15 the same shot yields 122 tracks. |
| `pattern_match` | **PREV_FRAME** | The bench prefers KEYFRAME (0.050 px vs 0.060 px). Real plates reverse it: KEYFRAME dies **2.6–2.9× more often**, because it matches the seed patch forever. |
| `leash` | **20 px** | PREV_FRAME's cost is accumulated drift; the leash bounds it. The two are one decision — PREV_FRAME with `leash = 0` is the worst of both. |
| `scale_rate` / `scale_ratio` | **0.10 / 1.6** | Chosen from the distribution over 36 tracks on SH004: ordinary frame-to-frame box movement is p90 0.065, a healthy track's cumulative size p99 1.33. At 0.10/1.6 all 8 tracks that go on to lose their patch are flagged; 0.15 misses 2. False alarms are cheap — 16 of 28 flags change nothing. |
| repair moves the marker | **never** | The seed patch's peak sits p50 4.2 px / p90 25.0 px from a *healthy* track's own position 150 frames later. Snapping to it would wreck good tracks; presence and size are all a fixed patch can answer. |
| motion model | **per-class** (Loc / LocScale) | A 10-config sweep: Affine worst-track 0.300, Perspective 0.360, bigger patterns 0.210, against 0.100 for these. |
| moving-tile + pattern refine | **off** | Blender replaces both. Bot with refines off 2.71 px, with them 0.06 px, Blender on the same raw guide 0.05 px — and they were the expensive half of a run. |
| `max-guide-dev` | **not implemented** | Measured harmful at every threshold: against a raw guide it is the guide that is wrong. |

---

## Verifying it

```bat
REM addon registers, and 3DE survives a round trip
blender.exe --background --factory-startup -noaudio ^
    --python tests\smoke_addon.py -- ^
    --plate <frames folder> --tracks <a 3DE .txt>
REM expect: 0.000000000 px, 0 problems

REM the parity gate -- the addon's tracking loop vs the original, marker for marker
runtime\python311\python.exe tests\test_track_core_parity.py ^
    --seeds ..\experiments\blender_track\out\SH004__dense__seeds.json
REM expect: 0.000000000 px, 0 span mismatches. Anything else is a real divergence.

REM foreground behaviour of Blender's tracker (already answered; re-run after a Blender upgrade)
blender.exe --factory-startup -noaudio --python blender_scripts\spike_foreground.py -- ^
    --plate <frames folder> --out logs\m0_spike.json

REM accuracy, against artist hand tracks -- the number that counts
runtime\python311\python.exe ..\experiments\blender_track\eval_vs_manual.py --pair seeded ...
REM reference: 2.20 px overall, 31%% of frames within 1 px

REM the pattern-box watch: when it stops a track, and what the stop means (CPU, seconds)
runtime\python311\python.exe tests	est_scale_watch.py
runtime\python311\python.exe tests	est_scale_drift.py
REM expect: all checks passed. Neither needs Blender, a plate, or the sidecar.

REM robustness, no ground truth needed
runtime\python311\python.exe ..\experiments\blender_track\track_stats.py <export>.txt
REM reference deaths/track: SH004 2.02, SH006 2.32, SH016 1.06, SH008 1.91
```

**The seed round-trip gate runs on every job and must pass before any other number means
anything.** A seed that fails to land is silently re-read from the nearest marker, and the
result is a full-length track that never touched its own feature — which every position
metric scores as good.

Use `--pair seeded`, never `nearest`. Past ~120 tracks, proximity pairing starts matching a
reference against a *neighbouring* feature.

---

## Not built yet

**Repair / re-acquire.** The design is settled and deliberately artist-in-the-loop, because
of what the method measures at:

- autonomous re-acquire: **315.73 px** against hand tracks, 2 % of frames within 3 px;
  three of five resumes landed on a *different* feature (1625 px, 500 px, 381 px)
- reappearances land on the right feature **26–47 %** of the time across three shots
- and every AI matcher tried was **worse than plain geometry**: local neighbour motion
  8.6 px at 10-frame gaps, DINOv2 36.2 px *while reporting 0.98 confidence*, RoMa 163.4 px

A track dies because its feature is hard, so a matcher that needs a clear feature fails on
exactly the features that died. The planned flow is therefore: detect the death, predict
from local neighbour motion, fan candidates out at several distances, let Blender try them
all and keep the longest survivor, cull anything that disagrees with the motion field over
its whole resumed segment — and then **show the artist the plate at the resume frame and
wait for a click**. Proposals arrive muted. Nothing un-mutes itself.

**Mover rejection.** Local neighbour motion again, not a global homography (144.3 px — it
cannot fit foreground dirt and a distant mountain at once, so it flags parallax as motion)
and not a segmentation model. It stays off until it has a hand-labelled reference set to be
scored against; shipping an unmeasured detector is how two 2026-08 metric defects happened.

**Shots with no static background** (a beauty shot, say) break both features outright —
there is no neighbour-motion field to work from. Detect and refuse, rather than return
numbers.

---

## Licensing

The addon imports `bpy`, so it is **GPL-2.0-or-later**. Everything under a different licence
runs in the sidecar process and is reached over localhost — it never shares an interpreter
with `bpy`.

Commercially clean, and the feature scope is why: dropping AI masking removed both of the
repo's licence landmines at once — SAM 3 (`license: other`, gated) and Ultralytics
(**AGPL-3.0**, and the only route to SAM 3).

| | |
|---|---|
| TAPNext++ (`google-deepmind/tapnet`, `tapnextpp_ckpt.pt`) | Apache-2.0 ✅ |
| torch / torchvision | BSD-3 ✅ |
| numpy, OpenCV | BSD / Apache-2.0 ✅ |
| MFT | CC BY-NC-SA — **NonCommercial**, never used here |
| CoTracker | CC-BY-NC — already out of the repo |
| SAM 3, Ultralytics | avoided; masking is out of scope |

`vendor/`, `weights/`, `runtime/`, `dist/`, `logs/` and `config/paths.json` are gitignored.

---

## Layout

```
bootstrap.bat / bootstrap.py   build the runtime, fetch weights, build the zip
build.py                       package addon/ as a Blender extension, validate, refuse if invalid
addon/btr_assist/              the addon (GPL, bpy + stdlib only)
  track_core.py                seeding + the resumable tracking generator
  client.py                    sidecar client (urllib, never `requests`)
  ops_seed.py                  the modal auto-seed operator
  three_de.py / ops_3de.py     3DE ASCII
sidecar/                       the model process (torch, TAPNext), stdlib HTTP
tests/                         smoke, parity
blender_scripts/               the M0 foreground spike
FINDINGS.md                    what was measured here, including what failed
```
