# Blender_assitant — measurements taken here

Only NEW measurements live in this file. Everything about the headless hybrid stays in
`experiments/blender_track/FINDINGS.md` and `experiments/report/FINDINGS_REPORT.md`; this
file cites those rather than restating them.

---

## M0 — foreground tracking (2026-08-20)

**Question.** Every tracking measurement this project owns was taken in `--background`.
`bl_track.py:14-18` records that `bpy.ops.clip.track_markers` runs *synchronously* there,
and warns that windowed "the same operator is modal and would return before the tracking
finished". If that were true of a Python call, an interactive addon would be impossible and
everything would have to shell out to a second Blender.

**Method.** `blender_scripts/spike_foreground.py`, run in a real window on SH004
(2560x1440, 160 frames):

```bat
blender.exe --factory-startup -noaudio --python blender_scripts\spike_foreground.py -- ^
    --plate experiments\blender_track\out\SH004\plate --out logs\m0_spike.json
```

`--factory-startup` here only makes the spike reproducible. The shipping addon gets no such
protection — see G.

### B — is a Python `track_markers()` call synchronous, windowed? **YES**

| | |
|---|---|
| operator result | `FINISHED` |
| wall clock | **11.0 s** for 20 trackers |
| markers produced | median 14, **max 160 of 160 frames** |
| verdict | **synchronous** |

The docstring's warning describes the **INVOKE** path — what pressing Blender's own *Track*
button runs. `bpy.ops.x.y()` from Python defaults to `EXEC_DEFAULT`, which is the same
`exec` callback background mode takes. Background mode never had a different operator.

**Consequence: Path A.** Tracking runs in-process, in the artist's Blender. The headless
subprocess fallback is not needed. The real foreground problem is that the call **freezes
the UI for its duration** — a UX problem, solved by draining a per-frame generator from a
modal timer, not an architecture problem.

Throughput here was 50.8 tracker-frames/s, but this is not a speed measurement: the seeds
were a blind 4x5 grid, not features, so most died within 20 frames. The measured figures
stay 256 tracker-frames/s typical / 48 on the hard shot (`experiments/blender_track/FINDINGS.md:25-34`).

### C — is `frames_limit` a usable chunk lever? **YES, but not needed**

Cap 8 forward from the seed (= 9 markers): counts `[4, 5, 9, 9, 9, 9]`. Nothing exceeded
the cap; four tracks reached it. The two short ones died — the tracker failing, not the
limit misbehaving.

So the lever exists. **The plan still does not use it**, because chunking moves the leash
checkpoint from every frame to every N frames, and that is a behaviour change with no
measurement behind it. Draining `track_stepped`'s existing per-frame loop as a generator
changes only the yield point, so the 2.20px and deaths-per-track numbers carry over intact.

### E — is `scene.frame_set()` alone enough, windowed? **YES — but keep the workaround**

Seeds at frame 40, cap 5:

| | first frames | last frames |
|---|---|---|
| with `space.clip_user.frame_current` | all 40 | 40,41,45,45,45,45 |
| `scene.frame_set` only | all 40 | 40,41,45,45,45,45 |

Identical. The silent re-anchor onto frame 1 documented at `bl_track.py:249-257` is a
**background-mode** failure: there, no redraw ever propagates the scene frame into the
space, so it stays pinned at 1 while the scene says 40.

**Keep setting it anyway.** It costs one assignment, it is required by the headless path
that shares this code, and the modal driver deliberately suppresses redraws between frames —
the condition that produced the bug in the first place. The seed round-trip gate stays
mandatory regardless; this test does not retire it.

### G — inherited settings

Recorded from a factory-startup scene so the addon can prove it overwrites each one. In an
artist's Blender there is no `--factory-startup`: `blio.run_blender():81` passes it so a
user's preferences cannot change results, and the addon has no equivalent. **Every setting
at `bl_track.py:360-364` and every per-track setting at `:376-381` must be written
explicitly on every job, never inherited.**

### F — proxies: NOT MEASURED

If an artist has `clip.use_proxy` on with `proxy_render_size = PROXY_50`, tracking may run
against half-resolution pixels and halve precision invisibly. No experiment in this repo has
ever run with a proxy.

Rather than measure it, **the addon forces `PROXY_100` for the duration of a job and
restores the artist's setting afterwards** — cheaper than the measurement and safe either
way. The test is written (`test_F_proxy`, `--proxy`) and left unrun; it needs
`bpy.ops.clip.rebuild_proxy` first, or it silently compares full-res against full-res and
reports a false all-clear.

### Incidental: `MovieTrackingTracks` has no `.remove()`

On 5.2 the collection is add-only from RNA (`tracks.new()` exists, `tracks.remove()` raises
`AttributeError: bpy_prop_collection: attribute "remove" not found`). Deletion is
`bpy.ops.clip.delete_track()` under the same context override tracking needs.

---

## M1 — the addon registers and 3DE round-trips (2026-08-20)

`tests/smoke_addon.py` on SH004, headless:

| | |
|---|---|
| register / unregister | PASS |
| import `SH004__dense__blender.txt` | 122 tracks |
| export and re-read | 15697 samples, **worst 0.000000000 px**, 0 problems |

Exact, which is what a correct round trip looks like — 3DE ASCII carries 12 decimal places.
Anything above 1e-6 px would be a real bug, most likely a lost half-pixel centre (which
reads as 0.5) or a flipped axis.

### Incidental: the existing addon's "Delete existing tracks" is broken on 5.2

`experiments/blender_track/addon_3de_io.py` clears tracks with `tracks.remove(tr)`, and
that method does not exist — see the M0 note. The lift into
`addon/btr_assist/three_de.py` replaces it with `delete_all_tracks()`, which selects and
calls `bpy.ops.clip.delete_track()` under a context override.

---

## M2a — parity: `track_core.py` reproduces `bl_track.py` exactly (2026-08-20)

The addon needs a second copy of the tracking body, because the original imports argparse
and runs as a headless script while the addon needs a resumable generator. Two copies is
two chances to disagree, so `tests/test_track_core_parity.py` runs both over the same seeds
in their own headless Blenders and diffs marker-for-marker.

**SH004, 122 seeds, leash 20, replants off, PREV_FRAME:**

| | |
|---|---|
| tracks | 122 vs 122 |
| samples compared | 16774 |
| worst difference | **0.000000000 px** |
| span mismatches | **0** |

Exact is the right bar here: both sides run the same C tracker on the same pixels through
the same operator, so any non-zero difference would be a divergence in the Python around
it, not floating-point noise.

### The bug this gate caught, which nothing else would have

First run: **positions pixel-exact over 15030 samples, and 40 of 122 tracks missing their
entire head.** Every number that existed was perfect; a third of the footage was silently
absent.

Cause: the forward pass runs `sequence=False`, and that does not produce exactly one marker
per call — a track seeded at frame 40 comes back with a marker at **39**, below its own
seed. (Same artefact recorded at `bl_track.py:415-424`, which measured markers at 39..44
after three forward single-steps.) The backward pass anchored on `markers[0].frame`, which
was now that stray 39 rather than the seed — and it is not a match, so tracking backward
from it failed on frame one and the entire backward pass did nothing.

`bl_track.py` avoids this by capturing the seed frames *before* the forward pass rather than
by knowing about it. `track_core.track_backward_pass` now anchors on the `seed_frame`
recorded at creation, which is the same fix without the ordering trap.

**Why this matters beyond the bug:** it is a second instance of the failure mode
`bench/README.md` is about. A track that keeps every position it has and loses half its span
scores *perfectly* on any position-error metric — mean error, p90, worst frame, all clean.
Only a span check sees it. Accuracy metrics do not detect missing data.

---

## Bootstrap — self-contained runtime (2026-08-20)

`bootstrap.py --self-contained`, one run, everything inside `Blender_assitant/`:

| row | result |
|---|---|
| python 3.11 | 3.11.9 embeddable, built here |
| torch + CUDA | **2.5.1+cu121 on NVIDIA RTX A4000** |
| numpy / opencv | 2.4.6 / **4.13.0** |
| TAPNext++ code + weights | Apache-2.0, `tapnextpp_ckpt.pt` 2532 MB |
| addon zip | built and validated by Blender |

9.6 GB on disk. Left unpinned, pip installed **opencv 5.0.0**, a major version ahead of the
4.13 the repo's `app.*` code is written against — inside an import the sidecar cannot see
fail cleanly. `OTHER_PINS` now pins it.

**"Self-contained" refers to the runtime, not to the repo.** Auto-seed imports
`app.tracker_core` and `app.track_meta` from batch_tracker; vendoring copies of those would
recreate exactly the two-copies problem the parity gate above exists to police. So the repo
stays a hard dependency of auto-seed. The addon's 3DE import/export needs none of it.

---

## M2b — sidecar (2026-08-20)

`sidecar/` spawned from `client.ensure()`, exercised over HTTP:

| check | result |
|---|---|
| spawn + health | port 65168, `ok=true` |
| torch / CUDA | 2.5.1+cu121, **NVIDIA RTX A4000, 16002 MB free** |
| repo imports (`app.tracker_core`, `app.track_meta`, `app.compare_tracks`, `app.export_3de`) | ok |
| unauthenticated POST | **HTTP 403** |
| bad plate path | `state=error`, `"plate not found: D:\nope\missing"` — a sentence, not a traceback |
| shutdown | clean; health goes to None |

### `python -m sidecar` does not work on the embeddable interpreter

`No module named sidecar`, whatever the cwd. The embeddable CPython that bootstrap builds
ships a `._pth` that blocks the automatic sys.path entries — the same mechanism
`CLAUDE.md` records for the bot's `_bootstrap_paths()`. The sidecar is therefore launched
**by file path** (`sidecar/__main__.py`, which puts its own directory on sys.path), not with
`-m`.

### Design notes worth keeping

- **`urllib.request`, not `requests`**, even though Blender 5.2 bundles requests — the addon
  then has zero non-`bpy` dependencies and cannot be broken by somebody's Blender build.
- **Token + portfile.** Port 0, bound to 127.0.0.1, token written to `logs/sidecar.json`.
  Without it any local process — including a web page — could drive a service that reads
  client footage off a studio share.
- **Parent-PID watchdog.** A crashed Blender would otherwise leave a process holding VRAM
  that nothing will ever ask to stop.
- **One job at a time.** There is one A4000; a second request gets 409 with a readable
  message rather than an OOM.

---

## M2c — auto-seed runs end to end (2026-08-20)

First real job, SH004 (2560x1440, 160 frames), `target=150 spacing_px=15`:

```
plate 2560x1440  160 frames
spread: 15px @1920 -> 20px at this plate width (2560)
tracks: 2578 seeded -> 2456 past motion filter -> 2456 past mask gate
        -> 1278 past quality bar -> 122 after spacing
classified 122 of 122 seeds
kinds: blob 14, corner 17, dense-blob 2, dense-corner 2, dense-edge 1, edge 34, flat 52
auto-seed done in 160.5s
```

Then those seeds through the addon's own tracking loop:

```
seed round-trip: max 0.000076 px  PASS
forward: 159 calls, 122 entered, 33 deaths, 517 clamps
backward: 21 calls (sequence mode)
122 tracks, mean length 137.5/160
```

**122 tracks is the number the headless hybrid recorded on this shot**, and the spacing log
reproduces `1278 past quality bar -> 122` exactly. The pipeline is the measured one.

### Three bugs, and what they have in common

All three were in the ~40 lines of glue between the sidecar and code that already worked.

1. **`Plate(path, ifl_dir)`** takes two arguments, and its attributes are `.w` / `.h` /
   `.count`, not `width`/`height`/`frames`. Failed loudly on the first call.
2. **`classify_seeds()` labels its argument in place and returns a COUNT**, not the list.
   Rebinding its result replaced 122 seeds with the integer 122 — `TypeError` one line
   later, which was lucky; a function that returned a list of the wrong thing would have
   run to completion.
3. **The recipe was silently not applied.** The first version set `RunnerConfig` class
   attributes, and `track_spacing_px` is not a RunnerConfig field at all — it is an
   `AppState` knob that `app.py:2424` maps to **`spread_min_dist_px`**. Setting a
   non-existent attribute on a dataclass raises nothing and changes nothing, so the run
   would have used the default 40 and produced a fraction of the tracks while looking
   entirely healthy.

Number 3 is the one worth remembering: **it is the only one that would not have crashed.**
The fix is to pass the recipe as named constructor arguments, which fail on a typo, rather
than as monkeypatched defaults, which do not. `spread: 15px @1920 -> 20px at this plate
width` in the log is now the proof the setting arrived.

### Dependencies the sidecar actually needs

TAPNext died on `No module named 'einops'`. `OTHER_PINS` now carries einops, timm,
imageio-ffmpeg, pillow, pandas, all pinned to the bot's `requirements.txt`.

**Not installed, deliberately: `ultralytics` (AGPL-3.0) and transformers/accelerate.** They
exist only for SAM 3 masking and Qwen analysis, both out of scope — which is what keeps
this addon's dependency tree commercially clean rather than merely its intent.

---

## M2d — movies, not just image sequences (2026-08-20)

First run on a real artist clip failed with `no image sequence found under: D:\Jefrin\IN`.

`clip.filepath` means two different things depending on what was loaded. For an image
sequence it names one numbered still and the sidecar wants the containing folder; for a
movie it names the movie itself. `clip_info()` took `os.path.dirname()` whenever the path
was a file, so `D:\Jefrin\IN\SH002.mp4` became `D:\Jefrin\IN` — a folder holding a dozen
unrelated mp4s and no image sequence at all.

Fixed by asking Blender instead of inferring: `clip.source` is `SEQUENCE` or `MOVIE`, and
only `SEQUENCE` gets the dirname.

**Verified on the real clip, SH002.mp4, 3840x2160, 180 frames:**

```
spread: 15px @1920 -> 30px at this plate width (3840)
tracks: 5976 seeded -> 4155 past motion filter -> 4155 past mask gate
        -> 2893 past quality bar -> 183 after spacing
kinds: edge 49, blob 28, corner 22, flat 41, dense 10
150 seeds (target) in 296.7s
```

Worth noting for expectations: **4K/180 frames is ~5 minutes**, against 160 s for
2.5K/160. And `spread_min_dist_px` is quoted against 1920, so 15 becomes 30 px at 3840 —
the density dial scales with the plate rather than staying absolute.

Every measurement in this project until now came from a folder of PNGs, because that is
what the bench and the extracted shots are. The movie path had never been exercised from
the addon.

---

## M2e — two things found while chasing a stale addon (2026-08-20)

The movie-path fix from M2d was correct, and the same error came back. It was not the fix.

### Blender keeps a disabled addon's submodules

Installing a new zip and re-enabling does **not** reload `ops_seed`, `track_core` and the
rest — they stay in `sys.modules`, so the OLD code keeps running with nothing anywhere to
say so. **A restart is required.** Verified separately that the packaged 0.1.1 computed the
right path all along:

```
[src] source     : 'MOVIE'
[src] SENT TO SIDECAR: D:\Jefrin\IN\SH002.mp4
[src] installed addon version: (0, 1, 1)
```

The panel now prints `v0.1.3   source: MOVIE`, so "which code is actually loaded" and
"which kind of clip is this" are both answerable by looking, rather than by a round trip.

### `PROXY_100` is not full resolution

Measured enum: `['PROXY_25','PROXY_50','PROXY_75','PROXY_100','FULL']`, default `FULL`.

**`FULL` is the original footage. `PROXY_100` is a rendered 100 %-size proxy FILE** — still
a re-encode, and one that may not exist. `FullResolution` was setting `PROXY_100`, i.e. the
proxy guard written to protect precision would itself have tracked against a re-encode.

Now verified as a round trip, with an artist's 50 % proxy simulated:

```
warnings   : INFO  Proxy PROXY_50 is on; jobs will run at full res
during job : use_proxy=False  size=FULL
restored   : use_proxy=True   size=PROXY_50
```

Both of these are the same shape as the recipe bug in M2c: **code that looked right, ran
without error, and quietly did the wrong thing.** None of the three would have been caught
by a position-accuracy check.

### The reinstall never happened

Two rounds were spent on a fix that was already correct. The installed extension was still
**0.1.0**, and none of the fixes were on disk:

```
C:\Users\jefrin\AppData\Roaming\Blender Foundation\Blender\5.2\extensions\user_default\btr_assist
  "version": (0, 1, 0)          <- installed 15:01, before any fix
  clip.source == "SEQUENCE"     : absent
```

Checking the install directory should have been the FIRST move, not the third. "Reinstall
and restart" is an instruction whose completion nobody can see — including the person who
followed it. `grep '"version"' <install dir>/__init__.py` is one command and answers it.

Verified after copying 0.1.3 in, by enabling the real installed extension rather than the
source tree:

```
enabled: bl_ext.user_default.btr_assist version (0, 1, 3)
panel line: v0.1.3   source: MOVIE
path sent : D:\Jefrin\IN\SH002.mp4
```

Note the module name: an extension is `bl_ext.user_default.<id>`, not `<id>`. Also delete
`__pycache__` when replacing files in place, or Blender can load stale bytecode.

---

## M3 — the artist loop, with CoTracker re-acquiring (2026-08-20)

Requested shape: **the artist places the seed, Blender tracks, CoTracker gets it back, Blender
tracks on, repeat.** No other model in the loop.

### Licence, recorded once

CoTracker is **CC-BY-NC 4.0** — verified against the source, not memory
(`facebookresearch/co-tracker` README: *"The majority of CoTracker is licensed under
CC-BY-NC"*). NonCommercial restricts **use**, not merely distribution, so unlike a GPL
component "in-house only" does not make it commercial-safe. Raised twice, reaffirmed by the
tool's owner; the decision is theirs. Code and weights live under `vendor/` and `weights/`,
both gitignored, so nothing NC-licensed enters the repository.

(The same README notes **LocoTrack is Apache-2.0** — a permissive alternative in the same
family, if the licence ever needs to go away.)

### CoTracker is cheap enough to run per-death

3 points across 60 frames of a 3840x2160 plate: **5.2 s**, at `max_side=768` (a 5x
reduction). It reports per-frame **visibility**, which is the occlusion signal Blender does
not have.

That changes the design: the headless prototype needed a dense throwaway motion field of
hundreds of tracks to predict a return. CoTracker answers directly for the artist's own
points, so no field is built at all.

Two things carried in from earlier measurements:
- **Resume by the guide's DISPLACEMENT, not its absolute position** — 41 % on-feature
  against 13 %, because by the time a track dies the two trajectories have drifted apart
  (median 6 px, p90 35 px, max 242 px).
- **A coarse guide is fine.** The existing TAPNext guide runs at 256 px and Blender still
  refines to sub-pixel, because a resume only has to land inside a widened search box.

### End to end, SH002.mp4, 12 seeds, 2 rounds

```
coord round-trip (uv <-> image px, y flip): max 0.000119 px  PASS
round 0: tracked in 4.5s, deaths 3, spans [1, 3, 13, 180 x9]
  CoTracker: 3 resume(s), 0 miss(es)
    USER_08 died f13 -> resume f16, 0 occluded
round 1: tracked in 3.3s, deaths 3, spans [2, 7, 23, 180 x9]
    USER_08 died f25 -> resume f28, 0 occluded
round 2: spans [4, 9, 40, 180 x9]
```

The mechanism works: every dead track was re-acquired, every round, and Blender extended it
(USER_08: 13 → 25 → 40 frames).

**But read `0 occluded`.** CoTracker never called any of these points hidden. They did not
die to an occluder — they died to blur and low contrast, on arbitrary grid positions no
artist would choose. Re-acquisition succeeds and the track dies again immediately, so the
loop crawls forward ~12 frames per round on a feature that is simply not trackable.

Hence `min_resume_len` (default 12, from the headless prototype): if a resumed segment
survives fewer frames than that, stop re-acquiring **that** track. A bad feature needs a
hand, not a retry.

### What is proven and what is not

Proven: the plumbing. Seed → Blender → death detection → CoTracker query at the artist's own
point → displacement resume → Blender continues → repeat, with the coordinate round trip
(y-up clip space vs y-down image space) gated at 0.001 px.

**Not proven: whether a resume lands on the RIGHT feature.** Every point here died to blur,
so none of them tested an occlusion crossing, which is the case the feature exists for. The
standing measurement for autonomous re-acquisition on this footage is **315.73 px against
hand tracks, 26–47 % on-feature** — that number was produced by a different method, and
CoTracker has not been scored against hand tracks here at all.

That is why resumes arrive **muted** and the panel states the failure rate at the point of
decision. Scoring CoTracker properly needs `eval_vs_manual.py` against the artist's own hand
tracks on a shot with a real occlusion.

### Float32, not a bug

The coordinate gate first failed at `0.000069 px` against a `1e-6` threshold. Blender stores
`marker.co` as float32, so a round trip carries ~7e-5 px on a 3840-wide plate — storage
precision. The bar is now 0.001 px, which still catches everything real: a y-flip is
thousands of px, a dropped half-pixel centre is 0.5, a resolution mismatch is a fixed
fraction of the width.

### A panel that raises does so on every redraw

`from . import bl_info` inside `draw()` raised `ImportError` in the artist's session:
Blender 4.2+ extensions are described by `blender_manifest.toml`, and the extension loader
does not guarantee `bl_info` survives on the module. The failure repeats on **every
redraw**, so one bad label buries the console and makes the whole addon look broken.

Fixes: `VERSION = bl_info["version"]` is captured while the module is still executing (the
literal stays literal, so `build.py`'s `version_from_source` regex still keeps the manifest
and the source from disagreeing), and `_version_str()` falls back rather than raising —
nothing inside `draw()` may throw.

**The gap that let it through:** the earlier check enabled the addon and listed registered
classes, and passed. It never called `draw()`. `tests/test_panels_draw.py` now calls every
panel's `draw()` against the **installed** extension, with and without a selection, using a
recording stub layout (`cls()` cannot make a Panel — `bpy_struct.__new__` refuses — and
Blender will not hand out a UILayout headless). It tests the Python in `draw()`, which is
where this lived; it does not test Blender's layout engine and does not pretend to.

That test immediately found a second, cosmetic one: **three user-visible strings contained
`%%`** (`"26-47%% land correctly"`) copied from format-string habit into plain literals,
which Blender renders verbatim. Plus a stale preference description still promising
`PROXY_100` after that was corrected to `FULL`.

---

## M3a — the whole-clip window was the bug (2026-08-20)

Reported from a live session: a track reached frame 72 and "CoTracker is not re-acquiring".

It was not failing. Querying the running sidecar showed a job **still running after 566
seconds**, with the GPU at **99 % and 16080 of 16376 MiB used**. It was thrashing, not hung.

Cause: the re-acquire job fed CoTracker the **entire clip** — all 312 frames — to answer
"where did this one point go after frame 72". Offline CoTracker attends across the whole
sequence and adds a support grid, so cost climbs steeply with length, and none of that
length was needed. The question only concerns the frames just after the failure.

### Three changes

1. **Window, not clip.** `[last_good - 2, last_good + 120]` instead of `[1, n_frames]`.
2. **Query at the last good frame**, using Blender's own position there, rather than at the
   artist's original seed. Blender matched that point by correlation on every frame up to
   it, so it is the same feature measured better — and it keeps the window short instead of
   forcing it back to frame 1.
3. **Forward only.** Backward tracking doubled the cost to answer a question Blender had
   already settled.

Plus a refusal (`BTR_COTRACKER_MAX_FRAMES`, 160): a job that never returns is worse than one
that says why.

That refusal immediately caught the next problem — **tracks do not all die together.** Deaths
at frame 1 and frame 91 give a 211-frame window on a 312-frame clip, which is the original
bug again. So requests are now grouped by where they died and each group gets its own short
pass: total work scales with the number of distinct failure points, not with the shot.

### Same clip, same length, after

```
round 0:  4.8s track, 5 deaths, spans [  1,  4,  26,  31,  91, 312]
round 1:  5.6s track, 4 deaths, spans [  3,  8, 106, 171, 310, 312]
round 2:  6.9s track, 2 deaths, spans [  3,  8, 172, 201, 310, 312]
round 3: 11.0s track, 0 deaths, spans [  3,  8, 172, 301, 310, 312]
```

USER_05 went **91 → 108 → 205 → 301 of 312 frames** across three resumes. USER_03 went
26 → 301. Whole run in well under a minute, against nine minutes that never finished.

**And the first real occlusion crossing:** USER_05's last resume reported **6 occluded
frames** — CoTracker called the point hidden and resumed it after. Every earlier test
reported `0 occluded`, i.e. blur deaths rather than occlusions.

Two honest refusals in the same run: USER_00 and USER_01 (dead at frames 5 and 10) came back
as *"CoTracker never calls it visible again"*. They were junk grid points on a blurred
region. Refusing beats inventing a position.

### Still not measured

Whether a resume lands on the RIGHT feature. Spans growing proves a resume was *trackable*,
which is exactly what `FINDINGS_REPORT.md` §8.4 warns is not the same thing — that mistake
produced a clean-looking file in which the resumes were 315.73 px out. Scoring this needs
hand tracks through an occlusion.

---

## Re-acquire now has to prove it is the artist's feature (2026-08-21)

Answers the "still not measured" note directly above it. Spans growing proved a resume was
*trackable*; nothing checked it was the thing the artist picked.

The check: the marker's pattern box — the patch Blender draws in the Track panel's preview,
captured at the keyframe **before any tracking runs** — is correlated (`TM_CCOEFF_NORMED`,
full plate resolution) against every candidate resume CoTracker offers. Below `min_match`
the track is left dead instead of resumed. `sidecar/patmatch.py`, gated in
`sidecar/server.py:job_reacquire`.

Two things came free with it:

* **CoTracker's first visible frame is not the best one.** `resume_candidates` now offers
  the first six, and the peak is chosen by correlation. On SH004 USER_04 moved from frame 24
  (0.746) to frame 27 (0.814) — the point was still half-covered when the guide first called
  it visible.
* **Sub-pixel.** A passing match refines the plant by a parabola fit on the correlation
  peak, so the check improves the position it approves rather than only judging it.

### Synthetic, answers known by construction (`tests/test_patmatch.py`, CPU, seconds)

| case | result |
|---|---|
| same feature, +17,+9 px | found to **0.008 px**, score 0.988 |
| same feature, +17.4,+9.65 px | found to **0.014 px**, score 0.880 |
| same feature, 2 stops down + grain σ6 | found to **0.010 px**, score 0.939 |
| **different feature** where the guide pointed | **0.186** — refused |
| flat / off-plate box | refused, not scored |

The decoy at 0.186 against the truth at 0.939 is the whole point: the two are separated by
0.75, not by a hair.

### SH004, real plate, 6 seeds, 2 rounds

One refusal, and it was right. USER_00 was planted on flat sky — the 28×28 patch spans
**130..135 grey levels**, std 0.42. Blender still tracked it 40 frames on brute+normalised
correlation, and the old path then re-acquired it and "tracked" it for **116 more frames on
nothing**. With the check on it scores 0.48–0.52 across all six candidates, its own peak
jumps ±10 px between adjacent frames, and it is refused. That is the failure this was built
for, caught on the first real run.

Refusals now say which of the two problems it is, because the fix differs: the same
candidate is re-scored against the patch at the **last good frame**. Both low ⇒ no feature
in the box (move the marker). Seed low, last-good high ⇒ the feature changed appearance
(re-key, or lower `min_match`). USER_00 scored 0.58 against frame 40 as well — no feature.

### Where the 0.60 default comes from

12 real SH004 features, each scored against its own frame-1 patch at its true position every
20 frames to frame 160 (`guide` track from `SH004__final__seeds.json`):

```
           f1     f10    f20    f40    f60    f80    f100   f120   f140   f160
median     1.000  0.944  0.906  0.895  0.877  0.917  0.819  0.868  0.932  0.928
min        1.000  0.723  0.497  0.621  0.487  0.707  0.545  0.725  0.595  0.512
```

A real feature stays at **0.82–0.94 median** over the whole shot, so 0.60 leaves a wide
margin — but the worst single feature dips to **0.487**, and the featureless seed's noise
floor was **0.52**. Those overlap. 0.60 is therefore a default, not a law: it will
occasionally refuse a hard feature, which is why the threshold is exposed in the panel and
why every score the run saw is printed.

### What this still does not settle

Whether an *approved* resume is on the right feature — only that a rejected one is not.
Nothing here is scored against hand tracks through an occlusion, so the resume still arrives
**muted** and the Keep/Drop pass stays. The pattern check narrows what reaches that pass; it
does not replace it.

---

## The re-acquire stopped at the frames where the feature was still covered (2026-08-21)

Reported from use: "CoTracker is not re-acquiring the track when Blender loses the pattern —
it should skip to the frame where the feature reappears, snap the point, and ask me."

**The bug.** `resume_candidates` offered only the first **six frames CoTracker's visibility
head flipped back to visible**, and the pattern check then ruled on exactly those. Through a
real occlusion those are the frames where the feature is still covered: all six fail, the
track is abandoned, and the artist is left to do it by hand while the feature is plainly back
a dozen frames later. Visibility was gating which frames were allowed to be examined; it is
now a report only.

**What replaces it.** CoTracker supplies a predicted position for **every** frame after the
failure (`cotrack.resume_path`), and `patmatch.find_reappearance` sweeps the whole window
frame by frame — one decode per frame, every track still looking tested on it — taking the
FIRST frame whose correlation against the artist's pattern crosses `min_match`, then the best
of the next four. First, not best-overall: best-overall skips past a good return in favour of
a marginally sharper frame fifty frames later, and every frame skipped is a frame the artist
has to track by hand.

Window raised 120 → 150 frames (still inside the 160-frame VRAM budget). A sweep that runs
out of window hands back the guide's position at its end; the next round re-queries CoTracker
from there and sweeps the next window, so an occlusion longer than one window is crossed in
stages instead of ending the track.

### Against the independent reference (`refs/SH004_lk`, closure 0.15–0.91 px)

Death declared at frame 40, prediction path = LK truth pushed off by `jitter` px in a random
direction — the guide error the sweep has to survive.

| guide error | found | position vs LK (median / max) |
|---|---|---|
| 0 px | 5/5 | 1.12 / 4.54 px |
| 8 px | 5/5 | 1.12 / 4.54 px |
| 25 px | 5/5 | 1.12 / 4.54 px |
| 60 px | 5/5 | 1.12 / 4.54 px |

**Identical to two decimals at every jitter.** That is the property worth having: the resume
no longer depends on CoTracker's accuracy, only on its rough vicinity — the correlation peak
is where the feature is regardless of where the window was centred. Two tracks land at 0.44
and 0.80 px, i.e. at the reference's own noise floor; ref014 at 4.54 px is a real
disagreement, above its 0.91 px closure. Caveat: this measures **re-landing**, 4–7 frames
after the death, not crossing a long occlusion — SH004 has no hand-tracked occlusion.

### Synthetic occlusion, answer known by construction

Feature covered by a different texture for six frames, then back and moving
(`tests/test_patmatch.py`): found on **the frame it returns, never inside the occluder**,
position **0.003 px**. A sweep that finds nothing reports how close it got and over which
frames.

### Two metrics measured and REJECTED

* **Peak-to-second-peak ratio** as a "is this distinctive" gate. Looks obviously right; does
  not work on this plate. Real correctly-tracked features score **0.93–0.99** (their
  surroundings are repeating structure) while the flat-sky seed scores **0.72** — the metric
  is inverted relative to what it would need to be.
* **Patch contrast (std) as a gate.** A seed on flat sky (std 0.42) and a usable one (std
  0.43) are indistinguishable by it, so refusing on contrast throws away real tracks. It is
  reported as a **warning** instead — real artist-placed features on this plate measure std
  40–64, so `std < 5` says "the score cannot mean much here" without vetoing anything.

Both were caught by feeding them cases whose answer was already known. Neither reached the
gate.

### Confirm at the snap, not in a batch afterwards

`confirm_resumes` (default ON) stops the run at each reappearance: the clip jumps to that
frame, the marker is snapped and selected, and the operator waits — **Enter** tracks on,
**D** drops it and stops re-acquiring that track, **A** accepts the rest, **Esc** stops. The
question "is that my feature?" takes two seconds on the frame it happened; the same question
in a batch afterwards, out of context, is what the old mute-everything pass was asking.

With it on, the confirmed segment is no longer re-muted at the end — the artist has already
answered. Only the resume frame itself is removed, because it is the guide's estimate rather
than a measurement, which is exactly what `Keep` did by hand. With it off, nothing has been
looked at and the old batch Keep/Drop behaviour is unchanged.

---

## The pattern box is a measurement, and nobody was reading it (2026-08-24, v0.5.0)

Once a seed is placed, the addon now tracks with **location and scale both animated**
(`LocScale`) and watches the box size on every frame. A tracker that has started sliding off
its feature does not announce it — the position stays plausible, the track stays alive, and
the run finishes with a file that looks fine. But under `LocScale` the pattern box swells,
because correlation is being satisfied by the surroundings instead of by the feature. So the
box size is an early-warning channel that was already there and was being thrown away.

Verified first, before anything was built on it: **Blender really does rewrite
`pattern_corners` every frame under `LocScale`** — 6/6 tracks on SH004, one distinct size
per frame, and the sizes move a long way (a 160-frame track ranging 19–32 px from a 31 px
seed). Under `Loc` the box never changes and the whole watch is inert, which is why the two
are one switch in the UI.

### What the watch does

`addon/btr_assist/scale_watch.py` — no bpy, so it is testable without Blender. Two limits,
because they catch two different failures, plus an onset:

* `rate` (default **0.10**) — fractional size change from one frame to the next.
* `ratio` (default **1.6**) — cumulative size against the box the artist set, either way up.
* `onset` — the first frame after which the box never came back inside 6 % of the artist's
  size. The frame that trips a limit is where the evidence crossed a line; the onset is
  where the trouble started, and it is what a repair cuts back to. On a 3 %/frame creep the
  two are **13 frames apart**.

A flag stops that track where it is. It is not a death and it deletes nothing.

### What a flag means is decided on the plate, not in the addon

`sidecar/patmatch.drift_report` correlates the patch from the seed frame against the flag
frame **twice** — at its own size, and resized by exactly how much the box grew — and the
two scores separate a feature that got bigger from a box that got bigger. Four verdicts
(`classify_drift`), only one of which deletes anything:

| verdict | evidence | what happens |
|---|---|---|
| `lost` | patch not findable at any size | cut back to the onset, leave dead → re-acquire takes it from a frame that can be trusted |
| `grown` | scaled patch beats unscaled by > 0.05 | keep the box, re-baseline, carry on |
| `bad-box` | patch is there at its own size, box is not | put the artist's box back, keep Blender's position, carry on |
| `clean` | patch is there and the box is within 1.25× | carry on, nothing touched |

Synthetic cases with answers known by construction (`tests/test_scale_drift.py`): a true
1.45× approach scores **0.92 scaled vs 0.21 unscaled**; a box swelling onto its neighbours
scores **0.97 unscaled vs 0.16 scaled**. The separation is not close, which is why the
0.05 margin is not a delicate number.

### A metric measured and REMOVED before it shipped

The obvious drift measure — how far the seed patch's correlation peak sits from where the
tracker has the box — **does not work on real footage, and gating on it would have been
destructive**. Measured over 36 `LocScale` tracks on SH004 (2416 sampled frames, search
radius 24 px): tracks whose patch stays findable for all 160 frames still show that peak
**p50 4.2 px, p90 25.0 px** from their own tracked position. A fixed patch matched 150
frames later cannot arbitrate a sub-pixel position — snapping to it would have dragged
healthy tracks tens of pixels and undone exactly the per-frame precision Blender is here
for. The offset is now **reported and never acted on**; presence and size are all the patch
is asked. This is the third metric this project has caught by feeding it known-good input
(peak ratio and patch contrast were the first two).

The same measurement is why a repair **never moves the marker** and only `lost` deletes
frames.

### Thresholds, chosen from the distribution rather than from taste

36 tracks × 160 frames on SH004, box size logged every frame with the watch off:

```
per-frame |step-1|   p50 0.0135   p90 0.0650   p99 0.4128
box scale, healthy   p50 0.97     p90 1.08     p99 1.33     max 1.46
```

`rate 0.10` sits just above the p90 of ordinary frame-to-frame movement; `ratio 1.6` sits
above the p99 of a healthy track's cumulative size. Sweeping rate × ratio against "does this
track eventually lose its patch": **0.10/1.6 flags all 8 of the 8 tracks that go bad**;
0.15 and above misses 2 of them. The cost of the sensitive setting is false alarms, and a
false alarm is cheap by construction — it costs one correlation and changes nothing.

### What the shipped defaults do on SH004

28 of 36 tracks flag at least once. Verdicts at the first flag:

```
clean 16    bad-box 5    grown 4    lost 3
```

**All 3 `lost` verdicts — the only destructive one — landed on tracks that genuinely lose
their patch. It did not fire once on a track that stayed findable for 160 frames.** 16 of
28 flags do nothing at all, which is the intended shape: stop cheaply, decide on evidence.

`/jobs/patcheck` end to end over the wire: 8 tracks, **3.5 s** including sidecar start and
first plate decode. No model and no GPU — it is two correlations per track.

### What this does not settle

* The verdicts were scored against the seed patch itself, which is the same signal the
  verdict uses. A track whose appearance genuinely changed reads as `lost` to both. The
  independent check would be hand tracks or `refs/SH004_lk`, and it has not been run.
* The modal wiring — flag → sidecar → repair → carry on — is not covered by a headless
  test, because the watch runs inside Blender and the correlation runs in the sidecar and
  no single interpreter has both. Both halves are tested; the seam is exercised by hand.
* Repair is capped at 2 per track (`drift_fixes`), which is a guess, not a measurement.

---

## The confirm phase was mistaken for a hang, because it was one (2026-08-24)

Reported from a real session: "i tried tracking a seed and blender froze". The sidecar log
had nothing wrong in it — the last job finished cleanly and returned a resume:

```
[job 22510a90cb64] 1 resume(s), 0 without one; pattern match 0.62..0.62
[sidecar] parent 21752 exited -- shutting down
```

The sidecar was healthy, answered, and then watched Blender get killed. So the fault was
entirely on the addon side, in whatever runs *immediately after* a resume arrives — which
is the confirm phase.

### What it was

`_tick_confirm` returned `RUNNING_MODAL` for every event that was not <kbd>Enter</kbd>,
<kbd>D</kbd>, <kbd>A</kbd>, or <kbd>Esc</kbd>. A modal operator that returns `RUNNING_MODAL`
consumes the event: nothing reaches the editor. So while the loop waited for an answer,
Blender took **no** mouse move, no click, no wheel, no middle-drag, no playhead. The only
sign it was alive was one line of status-bar text at the bottom edge of the window.

The comment above it named the reason — passing keys through would let <kbd>Enter</kbd> and
<kbd>D</kbd> reach the clip editor's own keymap — and that reason is real, but the cure was
applied to *every* event rather than to those keys.

The irony is the point. The question being asked is **"is that your feature?"**, and the
only honest way to answer it is to zoom in and look. The phase that exists to make the
artist look was the one phase that would not let them.

### What it is now

Consume the four answer keys and their releases; `PASS_THROUGH` on everything else. Zoom,
pan, scrub and play all work while the question is up.

<kbd>Space</kbd> was dropped from the accept keys in the same change. It used to accept,
which was harmless only because nothing else worked either — now that navigation reaches
the editor, an artist pressing Space to play the shot and see the motion would have
silently accepted the proposal they were about to judge.

The prompt is also drawn **in the clip editor** now (`overlay.py`), not only in the status
bar, and says in as many words that nothing is frozen. `_draw` is wrapped and never raises:
a draw handler that throws fires again on the next redraw, and unlike a panel there is
nothing to collapse to stop it.

### One thing measured here rather than assumed

The obvious other suspect was the polling. `_tick_waiting` issues a synchronous HTTP
`poll()` on Blender's main thread on **every 0.05 s timer tick — 20 requests a second** —
each with a 10 s timeout, while the sidecar runs CoTracker on the GPU. That looks exactly
like a freeze waiting to happen, and it is not: the sidecar is a `ThreadingHTTPServer` and
every job runs on its own daemon thread, so a poll never queues behind the work it is
asking about. It is wasteful, not blocking. It was left alone; the freeze was elsewhere.

`client.ensure()` *can* block the main thread for up to 60 s while a cold sidecar imports
torch, and that is real — but the log shows the sidecar was already up and serving, so it
was not this either.

### Gates

* `tests/test_confirm_keys.py` — 19 navigation keys × 3 event values all pass through and
  change no state; Enter/D/A do what they say; an answer key's RELEASE is swallowed too, so
  it cannot leak into the editor; dropping the last proposal ends the run instead of
  starting an empty tracking pass; and <kbd>Space</kbd> is pinned as pass-through, since it
  used to accept.
* `tests/test_overlay_draw.py` — `_draw` called in every state the module can be in: never
  shown, shown, hidden, hidden twice, re-shown. `blf` is stubbed, and not for convenience:
  calling the real one outside a GPU draw context does not raise, it takes the process down
  with `EXCEPTION_ACCESS_VIOLATION` (measured, headless, 5.2). The stub also checks the
  arguments — three lines, first one highlighted, stacked upward off the bottom edge.
* Unchanged and re-run: parity **25100 samples, 0.000000000 px, 0 span mismatches**, panel
  draw PASS, `test_scale_watch` / `test_scale_drift` / `test_patmatch` PASS.

### What this does not settle

The modal itself still has no headless test — an Operator instance cannot be constructed,
so `_tick_confirm` is called unbound with a stand-in `self`, the same compromise
`test_panels_draw` makes for `draw()`. That covers the decision, not Blender's event
routing: whether `PASS_THROUGH` actually reaches the clip editor's keymap was confirmed by
hand, not by a test.

---

## The re-acquire was looking for a 250-frame-old picture (2026-08-24)

Reported: a track ran clean to frame 253, died at 254, and the re-acquire snapped it back
at 256 — but "not exactly as the 1st frames which i seeded". Match score 0.62 against a 0.60
gate, so it passed and was planted.

### Three suspects, measured in order

**1. The pattern box size is ignored.** The seed patch is cut at the seed frame, at the seed
box, and correlated at that size for the entire sweep. Under LocScale — the shipped default
— Blender has been solving a per-frame size for 253 frames and none of it is sent. Measured
on the 122 real SH004 seeds, box scale at the last good frame:

| | p10 | p50 | p90 | max |
|---|---|---|---|---|
| died (n=35) | 0.29× | **1.52×** | 2.09× | 2.39× |
| survived (n=87) | 0.26× | 0.89× | 1.34× | 1.96× |

28 of 35 dead tracks are >10 % off their seed box. Real gap. But fixing *only* this does not
work — see the table below, and note it LOSES in the >50 % band, which is where dying tracks
actually sit. **Measured, then not shipped on its own.**

**2. The match score does not predict the landing.** The 14 known-answer landings worse than
20 px scored **p50 0.85, max 0.98**. Raising the gate 0.60 → 0.90 moves median error 3.87 →
3.11 px, still leaves 8 % beyond 20 px, and throws away 40 % of resumes. `min_match` is an
identity gate and nothing more; it was never going to fix this.

**3. The reference is stale.** `track_core`'s own docstring already measured this effect on
the tracker itself: KEYFRAME matching — correlate against the seed patch forever — dies
2.6–2.9× more often than PREV_FRAME, because appearance drifts. `find_reappearance` is
KEYFRAME matching at its most extreme: one patch, 250 frames later.

### The known-answer harness

A resume has no ground truth — the track is dead, so there is nothing to score a landing
against. So the answer is borrowed from tracks that did NOT die: a track Blender held for all
160 frames has a measured position on every one of them. Pretend it died at frame F,
re-acquire at F+3, and its own marker there is the answer. Both arms sweep the same frames,
from the same guide path, at the same radius, scored against the same position; the only
difference is the picture they are looking for.

158 cases — 59 surviving SH004 tracks × simulated deaths at f40/f80/f120:

| arm | found | err p50 | err p90 | err max | ≤2 px |
|---|---|---|---|---|---|
| seed patch (shipped) | 155/158 | 3.87 | 18.96 | 112.69 | 30 % |
| seed patch, resized to the track's box | 151/158 | 3.36 | 17.28 | 77.95 | 34 % |
| best of 9 scales | 158/158 | 3.24 | 18.57 | 105.17 | 30 % |
| **the last good frame's patch** | 153/158 | **0.46** | **5.41** | 54.67 | **85 %** |

**8.4× better median, 30 % → 85 % within 2 px.** It also disposes of suspect 1: the last-good
patch is cut at the box the track was carrying, so the size correction arrives free.
Staleness was the fault; scale was a symptom of looking at the wrong frame.

Stratified, the size-only arm is a wash and the reason is visible — it wins 30/19 in the
25–50 % band and loses 5/6 above 50 %:

| box scale at death | n | seed p50/p90 | resized p50/p90 | resized W/L/T |
|---|---|---|---|---|
| within 10 % | 41 | 2.56 / 9.00 | 2.53 / 9.99 | 16/19/6 |
| 10–25 % off | 48 | 3.23 / 14.26 | 3.15 / 14.86 | 20/28/0 |
| 25–50 % off | 49 | 5.45 / 20.10 | 3.73 / 17.45 | 30/19/0 |
| >50 % off | 11 | 6.16 / 20.56 | 7.01 / 17.79 | 5/6/0 |

### What shipped: two patches, two jobs

Localising with the last-good patch alone would be a regression of the thing the pattern
check exists for. If a track has already drifted onto the wrong feature before dying, its
last-good patch IS the wrong feature, and re-acquire would confirm it enthusiastically. So:

* **Localise** with the feature as the track last saw it (`last_box`, sent by the addon from
  the marker on the frame it died).
* **Verify identity** with the artist's seed patch, at the position localisation found,
  within `VERIFY_RADIUS = 6 px` — a second opinion on one spot, not another search.
* The seed patch is **resized to the track's box** for that check. Scored at positions known
  to be correct, the seed patch at seed size falsely refuses **11 %** of correct resumes; at
  the track's size, **7 %**. Same patch, same position — the size is the difference. This is
  where the scale measurement earns its place.
* A refusal now says what happened ("found something at frame N but your own pattern only
  scores X there") instead of reporting a bare number, and does not retry — the feature was
  found, it is simply not the artist's, and another sweep finds the same thing.
* `match_score` keeps its meaning — the artist's own pattern — so the confirm prompt still
  reads the number worth reading. `locate_score` is new and reports what localisation scored.

With no `last_box` (a continuation, or a track whose marker is missing) the path is
byte-identical to before.

### Gates

End to end on SH004 through the real sidecar: resume at f58, identity 0.804, locate 0.702;
the round-2 miss reports correctly and the shot finishes. `test_assist_loop` now seeds
**LocScale** and sends `last_box`, matching the shipped operator — under `Loc` the box never
changes size and the gate would have been testing a configuration no artist runs. Unchanged
and re-run: parity 25100 samples / 0.000000000 px / 0 span mismatches, panel draw PASS,
confirm keys PASS, overlay draw PASS, scale watch / scale drift / patmatch PASS.

### What this does not settle

* **The truth is Blender's own continuous track, and the winning arm is NCC like Blender is.**
  They share a bias, so the absolute 0.46 px is optimistic. The 8.4× gap *between arms* is the
  finding, since both were scored the same way. An independent check needs hand tracks or
  `refs/SH004_lk`.
* The harness only covers tracks that survived, whose boxes barely move (p50 0.87). The
  population that actually dies sits at p50 1.52× and cannot be measured this way, because a
  dead track has no answer to score against.
* The identity check runs at one position. A resume that localises onto a lookalike a few
  pixels from the real feature is inside `VERIFY_RADIUS` and will not be caught by it.
* Re-acquire still lands within 2 px only 30 % of the time with the shipped seed patch, and
  85 % with this change — on healthy tracks with a correct guide. It remains a proposal the
  artist confirms, and the confirm step is still the thing that makes it safe.

---

## Keep un-muted the one marker it was written to discard (2026-08-24)

Found by the first live run in a windowed Blender, not by any headless gate.

A track seeded at frame 1 came back with a **muted marker at frame 0** — the `sequence=False`
artefact `track_core.track_backward_pass` already documents ("a track seeded at frame 40 came
back with markers at 39..44"). Nothing downstream noticed it: 3DE export runs `skip_muted=True`,
so it can never reach a track file, and in the viewport it is invisible.

It broke `Keep`. That button un-mutes a resumed segment while deliberately leaving the FIRST
muted frame muted, because a resume frame is the guide's ESTIMATE of where the feature went,
and the frame after it is the first one Blender actually matched. It finds that frame with
`min(m.frame for m in muted)` — so with the artefact present, `first` is frame 0, and the
estimate at frame 54 is above it and gets un-muted onto the track. The artist presses Keep and
silently gets the guessed position they were meant to be spared.

Fix: muted markers below the track's first LIVE frame are not resumes, and are excluded before
`first` is taken.

`tests/test_confirm_resumes.py` pins both directions — the new rule spares frame 54 and
un-mutes only 55/56, and the old rule provably un-muted 54. It drives the rule rather than the
operator: a `MovieClip` cannot be synthesised headless without a file, and the test says so
rather than pretending to cover more than it does.

### What the live run also showed

The whole loop, in a real windowed Blender, 24.9 s: modal timer ticking, clip-editor context
resolving, overlay module loading, scale watch flagging at f9 and f10 and both coming back
`clean` from the sidecar, re-acquire landing at f58 after a death at f53, and round 2 refusing
honestly ("nothing reached 0.60 ... best 0.32 at frame 158"). The two false alarms in the first
ten frames each cost a sidecar round trip, which is worth watching if the watch is ever tuned.

Two process notes worth keeping:

* **A running Blender does not pick up a reinstalled addon.** Extension modules are imported at
  enable time; rebuilding the zip while Blender is open leaves the old code in memory. A live
  test against a session started before the install tests the previous build — check the
  process start time against the install time before believing any live result.
* Installed as an extension the package is `bl_ext.user_default.btr_assist`. A bare
  `import btr_assist` is a `ModuleNotFoundError`, which is how the first live attempt failed —
  and, because the console output was behind a buffering pipe, it briefly looked like a
  successful run whose sidecar jobs actually belonged to a different Blender.

---

## Autonomous re-acquire, scored against something that is not us (2026-08-24)

The standing number — **315.73 px, 26–47 % on-feature** — is why every resume stops and asks.
It was produced by a different method, and it predates today's localisation change. So it was
re-run properly.

### Getting a case at all

Seeded from `refs/SH004_lk/manual.txt` (5 corners, pyramidal Lucas-Kanade, gradient-based —
a different algorithm family from the NCC used here, so pixel-locking bias is not shared;
round-trip closure 0.15–0.91 px). Seeding from the reference's own first-frame positions is
what makes `eval_vs_manual --pair seeded` mean what it says.

**All 5 tracks survived all 160 frames. Zero deaths, so nothing to re-acquire.** Worth
recording on its own: on this shot Blender alone holds every reference corner end to end.
Autonomous, no confirming, scored over 800 samples: **median 0.77 px, 55 % within 1 px** — the
first number in this project scoring the assist loop against an independent reference rather
than against itself.

So deaths were made: `--kill-at F` cuts every track at frame F after the first pass. The
truncated track looks dead to `dead_tracks` for the same reason a real one does, and the LK
reference still holds the answer for every frame after the cut.

### The result, frames after the cut only

| cut at | tracks | frames | median | p90 | <1 px | <3 px | baseline, same frames |
|---|---|---|---|---|---|---|---|
| f40 | 5 | 535 | 3.44 | 6.14 | 27 % | 48 % | 1.15 px |
| f80 | 5 | 323 | 1.96 | 5.21 | 43 % | 73 % | 1.13 px |
| f120 | 5 | 131 | 2.58 | 5.06 | 27 % | 69 % | 1.52 px |

Autonomous re-acquire costs **1–2 px of median** against a track that never died. Across all
10 resumes the **worst single frame is 6.66 px**, and not one landed on a different feature.
Against 315.73 px and 3-of-5-wrong, that is a different failure mode entirely: right feature,
slightly off, instead of confidently tracking something else.

### What the number is allowed to justify — and what it is not

**Every one of those 10 resumes reports `occluded_frames = 0`.** The guide called the feature
visible throughout; these are loss-of-correlation deaths, not occlusion crossings. Truncating
a track does not hide anything, so this harness *cannot* produce an occlusion case, and
occlusion is precisely what CoTracker is in the loop for.

So the rule shipped is exactly as wide as the evidence: **`confirm_only_occluded` (default ON)
stops for the resumes that crossed an occlusion and takes the rest without asking.** The
common death — blur, contrast, a feature going soft — no longer interrupts, because that case
is measured and it holds. A resume across frames the guide calls hidden is unmeasured and
still stops. The count taken without asking is reported at the end, so nothing is accepted on
the artist's behalf silently.

Turning `Only when it was hidden` off restores stop-and-ask for every resume.

### What this does not settle

* **No occlusion was tested.** The one case the confirm phase exists for is still unmeasured,
  on this shot and every other reference here. It needs footage with a real occluder and a
  hand track across it.
* 5 tracks, all corners, one shot, one plate. `refs.json` labels every one `corner`; blobs and
  edges are not represented.
* Deaths are simulated by truncation. A real death happens because correlation failed, which
  usually means the picture changed — the last-good patch a real death leaves behind may be
  worse than the clean one a cut leaves.
* The baseline column is Blender tracking the same frames without dying, not ground truth.
  Both columns are measured against LK, whose own closure is 0.15–0.91 px, so differences
  below about 1 px should not be read as real.

---

## The foreground was never searched for (2026-08-24, SH013)

Reported on a real shot with a real occluder: "the tracks have a gap and is not tracked in
the FG when the camera started chasing the bike". Two separate faults, and the loud one is
not the AI's.

SH013 is motocross — 303 frames, 2562x1440, **59.94 fps**, camera chasing a bike over dirt.

### What the tracker was being asked to do

Inter-frame motion by image band, full-res plate px:

| frame | sky/BG | mid | FG band | FG p95 |
|---|---|---|---|---|
| 30 | 0.0 | 7.8 | 10.6 | 43.5 |
| 80 | 0.0 | 2.8 | 17.8 | 46.8 |
| 130 | 0.0 | 3.5 | 20.0 | **66.7** |

Against what the shipped geometry can reach:

    corner  pattern 28 px  search 55 px  ->  +-13 px
    blob    pattern 41 px  search 68 px  ->  +-13 px
    edge    pattern 33 px  search 81 px  ->  +-24 px

**The feature is outside the search box before Blender looks for it.** Not drift, not
rejection — it is never searched for. `KIND_GEOM` is a fixed table tuned on SH004, where the
camera is slow, and it carries no motion term at all. The bot's own tracker derives geometry
from a measured shot profile (`app/shot_profile.py`); the addon never got that.

Correlation is emphatically NOT the problem, and this must not be mistaken for a fix to it:
at the CORRECT position those same foreground patches score **0.88-0.93 NCC** between
consecutive frames. Direct probe in Blender, one foreground seed:

    search= 55 (shipped)   markers [1]        died on the first step
    search=200             markers [1..13]    tracks

### The tracks end where the plate ends

With the box widened, five foreground seeds ran 13-47 frames and **every one finished with
its search box off the edge of the plate** (15-38 px from it). They do not fail; they leave
frame. On a chase plate the near-ground is only on screen for that long, and short foreground
tracks are the correct answer there — 3DE solves from them. The bug was getting **one** frame
where the footage had 13-47 to give.

The correlation floor barely moves this: at a 3.5x box, dropping 0.75 -> 0.40 takes spans
from 7/10/13/16/23 to 13/13/26/28/43, and the ends are still the frame edge.

### What shipped

`sidecar/motion.py` + `/jobs/motion`: coarse optical flow over sampled frame pairs, reported
on a 6x4 grid. A grid because motion is wildly non-uniform here — 0.0 px/frame in the sky
against 20 in the near-ground — so one number for the plate would starve the foreground or
bloat every background box into a lookalike-matcher.

The operator measures before tracking, as a **modal phase**, and widens any box too small to
reach: `search = 2 * (p95 * 1.5 + pattern/2)`, capped at a quarter of the plate width.

**Only ever enlarges.** A small box an artist set deliberately is a statement about how far
that feature may move. A box too small to reach the feature at all is not a statement, it is
a track that dies on its first step.

Live on SH013 through the real sidecar: `plate moves 62 px/frame here, search box 55 -> 213
px`, and the seed went from **span 1 to span 13**, ending at the frame edge. The re-acquire
then refused correctly — the feature is off the bottom of the plate and is not coming back —
rather than inventing a resume.

`tests/test_motion_fit.py` checks both halves, because they are different claims: flow reads
2.0 / 12.0 / 39.8 on synthetic plates shifted by a known 2 / 12 / 40 px per frame; the grid
separates a still half from a moving half; a fast plate is widened to 163 px (reaching 68);
**SH004's slow plate stays at 55 px**; an artist's larger box is kept; runaway motion is
capped. The "leave slow plates alone" cases matter as much as the rest — inflating every box
would trade a rare failure for a permanent one.

### Three ways a fix can be invisible, all hit in one session

Each presented as "the change did nothing", and each was the change never being loaded:

* **A running Blender does not pick up a reinstalled addon.** Extension modules import at
  enable time. Check the process start time against the install time.
* **A running sidecar does not pick up new sidecar code.** `client.ensure` health-checks,
  finds the old process alive, and reuses it — so `/jobs/motion` fails on a machine where the
  feature is installed correctly, and the addon quietly falls back to the built-in boxes
  forever. The fallback logs and degrades rather than failing, which is right, but the
  staleness deserves a version stamp in `ensure`. NOT DONE.
* **A test driver that reads a setting but seeds with its own hardcoded geometry.** A
  search-box A/B ran three identical arms and was reported as "the box is not the cause". It
  was the cause. `--search-scale` and `--correlation` now reach the markers.

Three identical arms is the shape of a broken harness, not a null result.

### What this does not settle

* **The gaps are still unmeasured.** On this shot a track that stopped mid-chase looked
  exactly like a failed occlusion recovery and was not one; measuring re-acquire here before
  the foreground tracked would have scored the wrong fault. SH013 is still the only footage
  present with a real occluder, and it is now the right instrument for that question.
* `MOTION_HEADROOM = 1.5` is derived from one shot, where the worst single-frame motion ran
  about 1.4x the cell p95. A measured starting point, not a settled constant.
* Widening costs time and false-match risk, and neither has been quantified: no run here has
  scored ACCURACY with widened boxes against a reference, only span.
* The grid is 6x4 and static for the shot. A feature crossing from a slow cell into a fast
  one is sized by where it STARTED.

### The box was sized for where the feature started (2026-08-24, SH013)

Reported straight after the fix above: *"the feature is still in the frame but track is not
completely tracked until it exits."* Correct on both counts, and it exposes the limitation
that shipped with it — the grid was read **once, at the seed**.

Measured, box sized once from the seed cell, eight foreground seeds:

| trk | span | p95 @seed | p95 @death | box | needed | why it stopped |
|---|---|---|---|---|---|---|
| P00 | 13 | 47.6 | 47.0 | 171 | 169 | left frame, 15 px from edge |
| P04 | 24 | 22.7 | 47.6 | 96 | 171 | left frame, 22 px from edge |
| **P06** | **42** | **22.7** | **47.6** | **96** | **171** | **box too small there — 248 px of frame left** |
| P07 | 63 | 22.7 | 37.1 | 96 | 139 | left frame, 17 px from edge |

Seven of eight really do leave frame. P06 does not: seeded where the plate moves 22.7
px/frame, swept into a region moving 47.6, and died holding a box that no longer reached.

So `track_job` now re-fits each marker's search box every frame from the cell it has actually
reached. `Opts.motion` defaults to None and the loop is byte-identical without it, which is
what keeps the parity gate meaningful.

### It buys nothing on its own

| | spans | tracked frames |
|---|---|---|
| corr 0.75, sized once | 13 17 10 8 24 32 42 63 | 209 |
| corr 0.75, re-fit | 13 17 10 8 24 32 42 63 | 209 — **identical** |
| corr 0.40, sized once | 13 17 25 17 24 44 48 63 | 251 |
| corr 0.40, re-fit | 13 17 25 17 24 **129** 49 63 | **337** |

At the shipped correlation floor the track dies of **appearance** before it ever dies of
reach, so a bigger box does not extend one span — P06 included, which then reports "box was
big enough, something else". Relax the floor and reach starts to matter: one track goes 44 →
129 frames, +34 % tracked frames overall.

**The two limits are coupled and neither is worth much alone.** That is the finding. A reach
fix reads as useless while correlation is binding; a correlation fix reads as marginal while
reach is binding.

Shipped regardless, on narrow grounds: a box that cannot reach the feature is wrong whatever
is currently killing the track first, and it is measured to change nothing at the default.

**The correlation floor is deliberately NOT lowered.** 0.40 accepts weaker matches, weaker
matches mean drift, and there is no reference on SH013 to measure drift against — the span
numbers above say how many frames survive, not whether they are on the feature. Lowering a
quality floor because it produces longer tracks, without checking where those tracks are, is
the exact mistake this project's conventions exist to prevent. It stays an artist's knob with
these numbers written next to it.

### What this does not settle

* No accuracy number exists for SH013 at any setting. Everything here is span.
* `MOTION_HEADROOM = 1.5` still comes from one shot.
* What actually kills these tracks at 0.75 is unidentified — "appearance" covers motion blur,
  perspective foreshortening and scale change, and no measurement here separates them.
* A latent crash was fixed in passing: `/jobs/motion`'s fallback read `plate.n`, which does
  not exist (`Plate` exposes `.count`). It never fired because the addon always sends
  `frames`.

### Foreground continuity on SH013: what it is not, and what it is (2026-08-24)

Reported after the box fixes: *"track continuity is still bad"*. It was. Four candidate
causes were measured and eliminated in order, all on the same eight foreground seeds.

**Not the search box.** Re-fitting per frame changed not one span at the 0.75 floor
(209 tracked frames either way).

**Not the motion model.** `LocScale` — what the operator already ships — is the best of them.
Note the earlier probes in this file used `Loc` and therefore understated the shipped config:

| model | tracked frames |
|---|---|
| Loc | 209 |
| **LocScale** | **302** |
| LocRotScale | 224 |
| Affine | 214 |
| Perspective | 213 |

**Not the pattern size.** Bigger is strictly worse — 28 px 302, 56 px 270, 84 px 203,
112 px 201. A larger patch straddles more of a ground plane that is shearing under it.

**It is appearance.** On the frame each track stopped, its own patch was correlated against
the next frame over a 160 px radius:

| trk | died | best NCC anywhere | at distance |
|---|---|---|---|
| P00 | 13 | 0.721 | 62 px |
| P01 | 16 | 0.596 | 69 px |
| P02 | 10 | 0.680 | 167 px |
| P04 | 23 | 0.640 | 136 px |
| P06 | 47 | 0.528 | 43 px |

Every one is under the 0.75 floor. The feature genuinely stops looking like itself in a
single frame — a 28 px pattern travelling 47 px per frame is mostly motion blur, smeared
differently each time.

**And this is why the floor must not be lowered.** Those sub-threshold peaks sit 136-167 px
away where the plate moves 43-48 px/frame. They are false matches. Dropping `min_match` to
0.65 would not lengthen a track; it would plant it 136 px off the feature — longer, and
wrong, and wrong in a way that looks fine in a span count. The earlier note in this file that
0.40 "buys +34 % tracked frames" is exactly the trap: it buys frames, not tracks.

**Raising `rounds` does not help either.** Re-acquire refuses too, and correctly: 0 resumes,
5 misses, best 0.45-0.54 against the 0.60 floor. Those foreground points swept off the plate;
there is nothing to come back to, and refusing beats planting a marker on a different clod of
dirt.

### What the shot actually has

Same shipped config, spans out of 303 frames, by region:

| region | measured motion | spans |
|---|---|---|
| foreground (y .72-.88) | 43-48 px/frame | 13, 16, 10, 7, 23 |
| midground (y .45-.62) | 17-32 px/frame | 32, 47, **154**, 57, 63 |
| background (y .25-.40) | 0-8 px/frame | **206**, 128, 20, 39, **157** |

The plate tracks perfectly well. The near-ground of a 59.94 fps chase does not hold a feature
for long, and no setting changes that — it is the footage, not the tracker. A matchmove here
comes from the midground and background, plus many short foreground tracks, which is what an
artist would hand-track anyway.

### What shipped

A warning, not a tuning change. Any seed sitting where the plate moves faster than
`FAST_SEED_PX = 35` px/frame now says so before tracking starts, per seed in the console and
once in the status bar. Nothing else changed: the box re-fit stays, the correlation floor
stays, `LocScale` stays.

That threshold is where the measured falloff happens (43-48 -> short, 17-32 -> long), not a
limit in the tracker. Telling the artist which seeds cannot pay off is worth more here than
another round of tuning that measurably does nothing.

### What this does not settle

* Everything above is SPAN. No accuracy number exists for SH013 at any setting — there is no
  reference on this plate.
* Whether CoTracker alone could follow that foreground is untested. It produces a path
  regardless; the pattern check refuses it, and refusing is right without evidence the path
  is correct. This is the one remaining idea with a plausible route to foreground continuity,
  and it needs a hand-tracked foreground feature to judge.
* `rounds` is capped at 10 and is not exposed in the panel at all. It did not matter here,
  but an artist cannot raise it on footage where it might.
* `min_resume_len = 12` abandons a track whose resumed segment is shorter than 12 frames.
  On this plate 7-frame segments are normal, so the give-up rule is tuned against footage
  like this one. Untested either way.

### The scale watch was deleting correct tracks (2026-08-24, SH013)

Reported three times, and I kept measuring the wrong thing: *"still the track is not fully
tracked"*. Every experiment above ran `track_core.track_job` directly. **The operator runs
more than that**, and the difference is the whole answer.

Same seed, same plate, same shipped settings, one flag:

```
watch_scale = 1     LIVE_01 live markers    5   (f1..f5)
watch_scale = 0     LIVE_01 live markers  302   (f1..f302)
```

```
[assist] LIVE_01 f54 lost: no match at any size -- 49 frame(s) from f6 dropped
```

A track that runs the entire 303-frame shot was cut to five markers by its own quality check.

### Why

`classify_drift` returned `lost` when neither the seed patch nor its resized copy reached
`min_match`, and the operator's `lost` branch deleted every frame back to the swell onset. It
was the only destructive verdict in the addon, and its justification was that a patch which
is not there at any size is evidence the box grew onto nothing.

**It is not evidence. It is the absence of evidence**, and the two causes are
indistinguishable from inside that function:

* the tracker slid off the feature, or
* the feature stopped looking like its seed frame.

On SH004, where the watch was built and validated, the second case does not arise — the seed
patch stays findable for all 160 frames, so a low score really did mean the first. On SH013 it
is routine: measured earlier in this file, a foreground patch scores **0.53-0.72 against the
very next frame**, let alone against a seed fifty frames back. A perfectly tracked feature
reads exactly like a lost one, and the destructive branch fires on both.

The three `lost` verdicts that validated this on SH004 were real. They were also the only
evidence, from the one plate where the failure mode cannot occur.

### The fix

`lost` becomes `unknown`, and `unknown` is handled the way the addon already handles a
question it cannot ask — the `not chk.get("ok")` branch, whose comment already said the right
thing: *"refusing to answer is not evidence against the track."* Keep every measured frame,
stop watching that track, carry on from where it stopped.

Nothing else about the watch changed. `bad-box` still resets a box that swelled onto the
surroundings, `grown` still keeps a box that followed a feature approaching camera, `clean`
still does nothing. Those three have positive evidence behind them; only the destructive one
did not.

`_rewind` is deleted with its last caller — it was the one function in the addon that removes
markers, and nothing may reach it any more.

### After, watch ON, as shipped

| seed | region | markers |
|---|---|---|
| 0.50, 0.55 | midground | **302** of 303 (was 5) |
| 0.20, 0.30 | background | 247, two `bad-box` repairs, one re-acquire at f254 |
| 0.35, 0.62 | mid/fore | 32 — this one genuinely dies |

The repairs still happen and still help: *"box was 0.62x yours and the feature was not (match
0.85) -- box reset"*.

### The lesson worth keeping

A quality check validated on one plate had never met footage where its core assumption fails.
It did not merely miss problems, it **destroyed correct work**, silently, and reported the
destruction as a repair. Both defects found in 2026-08 were metrics measuring something other
than what they read as; this is a third, and the most expensive, because a metric that only
reports can be ignored while one that deletes cannot.

Three earlier sessions of measurement on this complaint went into `track_job` in isolation and
found nothing, because the fault was in the layer the harness did not exercise. **Test the
thing the artist runs.**

### What this does not settle

* A track that genuinely slides onto the background is now kept rather than cut back. That is
  the deliberate trade: the artist confirms re-acquires anyway, and no automatic deletion is
  worth the case above. Unmeasured on SH004 whether those 3 tracks now survive as bad data.
* Everything here is span. No accuracy number exists for SH013.

### A refused resume killed the track; now it asks (2026-08-24)

The pattern gate is the only thing standing between a CoTracker path and a resumed track, and
when it says no the track ends for good. Two measurements say that is too final:

* on the SH004 known-answer cases the gate refuses **7 % of resumes that are correct**
  (seed patch at the track's own box size, scored at a position known to be right);
* on SH013 it refused **every** foreground resume at 0.45-0.54 while CoTracker produced a
  path each time, and every one of those tracks stayed dead.

So a NEAR MISS -- within `UNVERIFIED_MARGIN = 0.15` of the gate -- is now offered as a resume
marked `verified: False`, on one extra condition: **CoTracker must independently call the
feature visible at that frame**. Two signals have to agree that something is there before the
artist is asked about it.

An unverified resume is **always** confirmed. It ignores `confirm_only_occluded`, and the
prompt says what it is rather than showing a score that would read as endorsement:

```
<track> found again at frame N   (pattern only reached 0.52 -- NOT verified, your call)
```

No gate moved. `min_match` still decides what counts as verified; what changed is that
falling short of it produces a question instead of a dead track.

### It correctly declines, too

On SH013 the near miss did NOT fire, and the log now says why:

```
USER_01: best 0.52 at f26, but CoTracker calls the feature NOT visible there -- not offered
```

Those foreground points had swept off the plate. The patch half-matched something; CoTracker
knew the feature was gone; the two signals disagreed and nothing was offered. The old message
("nothing reached 0.60") read like a tracker giving up, which is a different and misleading
statement.

Forced positive case, SH004 with the gate raised to 0.99 so correct resumes fall under it:

```
ref014  back at f94   first over the line fNone   match 0.94
ref048  back at f136  first over the line fNone   match 0.98
ref052  back at f131  first over the line fNone   match 0.98
final spans [131, 133, 147, 160, 160] of 160
```

Three tracks that would have ended, proposed instead.

### The leash idea, measured and dropped

The artist reported that **native Blender cannot track SH013's foreground by hand either** --
independent confirmation of the NCC measurement. The obvious response was to let the guide
carry the track (`track_job` has a leash; the operator sets it to 0). That needed CoTracker to
be good on that content, so closure was measured -- track forward, track back, see where it
lands:

| point | span | closure px | frames CoTracker calls visible |
|---|---|---|---|
| FG_a | 1-60 | 6565 | 12/60 |
| FG_c | 1-60 | 5838 | 23/60 |
| MID | 60-180 | **3.4** | 93/121 |
| BG | 60-180 | **2.7** | 121/121 |

**The foreground rows are meaningless and must not be read as "CoTracker is bad there."**
Those points leave the frame — measured separately, they exit in 13-47 frames — and a round
trip through a point that left the plate returns nonsense. CoTracker reporting them visible
for only 12-24 frames is it being *right*.

What the valid rows say: on features that stay in frame CoTracker closes to 2.7-3.4 px over
121 frames. Respectable, and beside the point — Blender already tracks those same features for
154-302 frames at sub-pixel precision since the watch fix.

So the leash has no case on this shot: the foreground cannot be carried by anything because it
exits, and the midground does not need carrying. Not built. The assumption it rested on —
that foreground features persist and merely tracked badly — was never true.

### What this does not settle

* The 7 % figure is from SH004 known-answer cases; how often a near miss is CORRECT on real
  footage is unmeasured, and every one still costs the artist a decision.
* `UNVERIFIED_MARGIN = 0.15` is a judgement, not a measurement. Too wide and it manufactures
  questions; too narrow and it changes nothing.
* An unverified resume that the artist accepts is indistinguishable downstream from a verified
  one. The end report counts them separately; the track file does not.

### What a seed has to hold, measured (2026-08-24, SH013)

Reported: *"the track stopped at frame 12, feature still mid-frame."* Not the frame-edge case
the earlier fixes explained, so a 30-seed grid was run across the plate in the shipped
config. 11 of 30 died before frame 25, most with the feature well inside frame.

The predictor is **contrast**, not motion:

| patch std | seeds | median span |
|---|---|---|
| < 4 | 11 | 26 |
| 4-8 | 9 | 23 |
| 8-15 | 10 | **113** |

Seeds dying before frame 25: median std **4.4**. Seeds running 100+ frames: median **10.6**.
For scale, artist-placed features on SH004 measure **std 40-64**. SH013 is uniform brown dirt
at 59.94 fps and nothing on it reads above 15.

The motion pre-flight now also reads each seed's own patch and reports **both** numbers for
**every** seed, not only the ones that trip a threshold — because the case that prompted this
tripped neither. A seed at 33 px/frame (under the 35 warning) with std 9.8 (over the 8
warning) died at frame 11. Neither number was extreme; the combination was hopeless. Two
binary thresholds tuned on one shot cannot say that, and the numbers themselves can.

The soft warning remains for genuinely empty patches and quotes what it is worth. It is not a
gate: an artist may have reason to track a soft feature, and refusing would be the tool
overruling them on their own plate.

**What this does not settle:** the rule is one shot, n=30, spans only. Whether std predicts
span on footage that is not uniform dirt is untested, and the two numbers are reported rather
than combined precisely because there is not enough evidence to combine them honestly.

### The box shrank and the track drifted, with the span still looking fine (2026-08-24)

The artist's diagnosis, and it is correct: *"the pattern box goes smaller and drifts away."*
They proposed ending the track when the box hits the frame border. Both halves were measured
and only one of them is where the damage is.

**The border rule changes nothing here.** Implemented (`edge_stop`, on by default) and run over
30 seeds: **0 of 30 tracks stopped earlier**, including four whose markers end OFF the plate
(-18, -2, -23, -49 px). Blender already stops at or just past the border on this footage, so
the rule fires on the frame the track was ending anyway. It is kept because a pattern box
straddling the frame edge is correlating against nothing and that is worth refusing on
principle -- but it earned no frames, and saying otherwise would be inventing a result.

**The collapse is everywhere, not at the border.** Pattern box over each track's life, 30
seeds, `LocScale`, shipped config:

| trk | span | seed px | end px | end/seed | first under 60 % |
|---|---|---|---|---|---|
| P04 | 251 | 28.0 | **0.3** | 0.01 | f173 |
| P07 | 303 | 28.0 | **0.7** | 0.03 | f38 |
| P00 | 302 | 28.0 | **1.5** | 0.06 | f78 |
| P12 | 71 | 28.0 | **0.6** | 0.02 | f40 |
| P16 | 58 | 28.0 | 237.4 | **8.47** | - |

A 0.3 px pattern is not tracking anything, and the track does not die -- it keeps returning
positions. **This invalidates span as a success measure, including numbers reported earlier in
this file.** P07 "ran the full 303-frame shot" with its box collapsed by frame 38; roughly 265
of those frames are drift with a marker attached.

Totals over the same 30 seeds:

| config | tracked frames | degenerate boxes |
|---|---|---|
| `Loc` | 1660 | 0 |
| `LocScale` (shipped) | 2535 | **13 of 30** |
| `LocScale` + clamp | **2037** | **0** |

`LocScale`'s apparent 875-frame advantage over `Loc` was mostly collapsed boxes. Clamping the
box to within `scale_ratio` of the size the artist set removes every degenerate case and still
beats `Loc` by 377 honest frames, so scale tracking keeps its value rather than being switched
off.

`clamp_pattern` touches the box only, never the position -- the same rule as the search-box
re-fit and the watch's `bad-box` repair: geometry is what the evidence supports correcting.
It reuses `scale_ratio`, the watch's own "too far from your box" number, so one setting bounds
both what is flagged and what is possible. `Opts.scale_clamp` defaults to 0 (off), so the
headless path and the parity gate are unchanged.

**What this does not settle:** still spans, not accuracy -- there is no reference on SH013, so
"honest frames" means "the box was a plausible size", not "the marker was on the feature". The
degenerate-box threshold (0.6-1.8x of seed) used to count failures is a reading aid, not a
measured boundary. And the earlier scale-watch fix (`lost` -> `unknown`) means a collapsing box
no longer gets cut back; the clamp is now what prevents the degenerate state instead, which is
a better place to stop it but was not the reasoning at the time.

### The addon's settings never reached the artist's own markers (2026-08-25, SH012)

Found from a diagnostic report written by the artist -- the first time in this whole
investigation that their actual scene could be read rather than approximated.

The report showed a selected track carrying **`pattern_match = KEYFRAME`** while the clip's
defaults said `PREV_FRAME`.

`track_core.apply_settings` writes the clip's `default_*` settings, and Blender reads those
when a track is **created**. A marker the artist placed earlier keeps whatever it was made
with. The operator overrode `motion_model` per track -- with a comment making exactly the
right argument, *"an artist's existing markers carry whatever model they were made with"* --
and stopped there. `pattern_match`, `correlation_min`, `use_brute`, `use_normalization` and
`frames_limit` were never applied.

This project had already measured the cost. From `track_core.Opts`: KEYFRAME dies **2.6-2.9x
more often** than PREV_FRAME on real plates, because it matches the seed patch forever while
appearance drifts. The addon's whole measured configuration was being skipped on precisely
the tracks an artist places by hand.

Reproduced from the report's own numbers -- seed (1750, 731), 21 px pattern, 71 px search,
SH012:

| pattern_match | span | ends at | why it stopped |
|---|---|---|---|
| KEYFRAME (theirs) | **158** of 328 | f158, (1838, 725) | **82 px inside frame** -- died |
| PREV_FRAME | **224** of 328 | f224, (1909, 708) | 11 px from edge -- left frame |

`f158` and `82 px from edge` match the artist's report exactly, so this is their track, not
something resembling it. **+42 % span, and the failure mode changes from dying mid-shot to
running until the feature leaves the plate.**

Every setting is now applied per track, and each change is printed:
`[assist] Track: pattern_match KEYFRAME->PREV_FRAME`. Taking over an artist's settings
silently is its own kind of wrong even when the change is right.

The same report also showed **scene frame_end 250 against a 328-frame clip**. Not a tracking
limit -- the operator uses `clip.frame_duration` -- but it decides what can be scrubbed, so a
track running to f300 looks like it stopped at the end of the range. Now a panel warning.

**What this does not settle:** the fix asserts the addon's configuration over the artist's on
tracks they made themselves. That is consistent with `apply_settings`' stated intent
("nothing here may be inherited") and it is now announced rather than silent, but an artist
who deliberately chose KEYFRAME for a reason will be overruled.

### The occluder captured the track and nothing asked whether it was still the feature (2026-08-25, SH006)

The first real occlusion any measurement in this project has had. From the artist's own
diagnostic report: seed at frame 1, occluded at 14, reappearing at 25 -- and the track ran
**continuously to frame 22** and was never re-acquired.

`live runs [[1, 22]]`. It never died. That is the whole failure: Blender tracks with
PREV_FRAME, each frame matched against the one before it, so an occluder sliding in over a
few frames never produces a step that looks wrong. Correlation stays satisfied, the track
stays alive, nothing downstream asks for a re-acquire, and the drift is written to the file
as though it were data.

Reproduced from the report's own numbers -- seed (3559, 197), 41.2 px pattern, 113.1 px
search -- and scored against the artist's own patch at every position the track claimed:

| frame | seed NCC | | frame | seed NCC |
|---|---|---|---|---|
| 13 | 0.985 | | 15 | **0.218** |
| 14 | 0.908 | | 16 | 0.210 |
| | | | 22 | 0.178 |

**0.91 to 0.22 in one frame.** The reproduction ends at (3814, 72) -- the same position the
artist's report records for frame 22, so this is their track, not something resembling it.

The signal was always there. Nobody was asking the question.

### What shipped: hold the feature, or stop

`/jobs/hold` scores the artist's seed patch at every position a track claims, in frame order,
with a 3 px radius -- this is not a search, the position is Blender's and is not in question;
it asks what is AT that position. `first_loss` finds where the track stopped being that
feature, and the operator **deletes the drift** and hands the track to re-acquire from the
last frame that was genuinely the feature.

Two conditions, and the second is what makes it safe:

* the score is below an absolute floor (0.5), **and**
* below half of what that track was holding before it -- its own median.

An absolute floor alone would condemn every track on SH013, where patches score 0.53-0.72
against the very NEXT frame while tracking perfectly well. **A score that was never high
cannot fall.** Two frames must agree, so a grain hit or a lighting step does not cut a good
track.

This deletes markers, which the addon otherwise refuses to do, and the difference from the
`lost` verdict removed earlier is *evidence*. There, a patch that could not be found anywhere
was treated as proof a track was lost -- it is not, and a healthy track on poor footage reads
identically. Here the patch is scored at the position the track claims, frame by frame, and
the finding is a **fall** against a baseline the track itself set.

Live through the real operator on the artist's seed:

```
LIVE_01 stopped being your feature at f15 (was 1.00, became 0.34)
        -- 10 frame(s) of drift removed, re-acquire takes it from here
LIVE_01 died f14 -> back at f17, match 0.84
live markers 14 (f1..f14)
```

14 clean frames and a proposal to judge, instead of 22 frames of which 8 are wrong.

`tests/test_scale_drift.py` pins six cases including both measured plates: SH006 cuts at 15,
SH013 never cuts, one bad frame does not cut, two in a row does, a gentle defocus decline does
not, and an occlusion at f100 of a long healthy track does.

### What this does not settle

* The resume it then proposed was frame 17, while the artist reports the feature returning at
  25. Whether f17 is a partial reappearance or a wrong landing is unknown -- it is a proposal
  the artist confirms, and occluded resumes always ask, but it has not been judged.
* The floor (0.5) and the fall ratio (0.5) come from one measured occlusion with an enormous
  margin. Footage where a feature genuinely changes appearance while staying itself -- a face
  turning, a light change -- has not been tested and is exactly where this would cut wrongly.
* The check runs once per pass, over the whole track, so drift is tracked before it is
  removed. Frame-exact detection during tracking would need the check inside the loop.

### One answer was not enough: N cycles the candidates (2026-08-25, SH006)

The artist checked the resume the loop proposed -- frame 17 -- and confirmed by eye that it
was **the wrong feature**, while the real reappearance was frame 25. Their suggestion, and it
is the right shape: snap to the best match, then let them press a key for the next one.

The alternatives existed and were being discarded twice over.

**First, the sweep stopped looking.** `find_reappearance` resolves at the first frame over
`min_match` plus `settle`, then marks the job done. On this track that was frames 17-21, so
frame 25 was never scored at all. It now keeps SCORING for `collect` frames past the
crossing -- those frames cannot change the answer, only fill the candidate list.

**Second, every score was thrown away.** The sweep computed a correlation for every frame it
looked at and kept only the best. `top_candidates` now keeps the best few, each at least
`min_gap` frames from the others so the list spans the window instead of returning six frames
of the same peak.

Measured on the artist's own seed, through the real operator:

```
LIVE_01: 6 candidate(s) to cycle with N --
   f17(0.84), f26(0.99), f29(0.98), f45(0.95), f42(0.95), f48(0.94)
```

The wrong landing they found is candidate 1. **f26 at 0.99 is one keypress away**, against
the frame 25 they report by eye.

In the confirm phase: `ENTER` accepts, **`N` cycles to the next candidate** (wrapping, so
cycling past the right one comes back to it), `D` drops, `A` accepts the rest, `ESC` stops.
The prompt shows `[2/6]` so the artist knows where they are in the list. Choosing a candidate
re-plants the marker at that frame and position, keeping the artist's own box.

### A rule that was nearly changed by accident

Scoring past the crossing initially let a later, higher-scoring frame win the resume itself --
the proposal moved from f17/0.84 to f26/0.97, which is the *right answer* on this shot. It is
also **best-over-the-window**, a different rule from the documented "first over the line, not
best", whose stated reason is that skipping frames costs the artist hand-tracking.

Reverted: only the `settle` frames may improve the answer, and the collect frames feed
candidates only. Getting a better result is not a reason to change a decided rule silently,
and one shot is not enough to overturn it -- but the evidence is now recorded, because it is
the only measurement this project has that bears on it:

| | frame | score | artist's verdict |
|---|---|---|---|
| first over the line (shipped) | 17 | 0.84 | **wrong feature** |
| best over 45 frames | 26 | 0.97 | matches their observed f25 |

If that holds on more shots, the default is worth revisiting. It needs more than one.

### What this does not settle

* One artist verdict on one occlusion. Whether the best-scoring candidate is usually right is
  unmeasured, and the match score is already known NOT to predict landing accuracy.
* `collect = 40` frames of extra scoring costs decode time on every re-acquire, unmeasured.
* Candidates are ranked by score. Given the score does not predict accuracy, an artist may
  still have to walk the whole list -- the ordering is a convenience, not a claim.

### The candidates were right; the rule that chose between them was not (2026-08-25, SH006)

The artist hand-tracked the occluded feature. **The first reference this project has for an
occlusion** -- everything before it was scored against Blender's own output, a synthetic
plate, or Lucas-Kanade, and none of those can say whether a resume landed on the right thing.

`reacquretracke_manual.txt`, runs `[[1, 14], [25, 32], [40, 64]]`. It ends at f14 and resumes
at f25.

That alone confirms two things shipped earlier on thinner evidence:

* the hold check cut the track at **f15** -- the frame after the hand track's first run ends;
* CoTracker queried at the artist's seed reported the feature back at **f25** -- the frame the
  hand track resumes on.

Then the measurement that mattered. Every candidate scored against the hand track:

| chosen by | frame | score | error vs hand track |
|---|---|---|---|
| **"first over the line" (was shipped)** | f17 | 0.84 | **no hand sample -- inside the occlusion** |
| | f26 | 0.99 | **1.7 px** |
| | f31 | 0.98 | 1.4 px |
| | f45 | 0.95 | 1.2 px |
| | f42 | 0.95 | 1.3 px |
| | f48 | 0.94 | 2.4 px |

**Every alternative was on the feature within 2.4 px.** The sweep, the localisation patch and
the candidate list were all working. The single wrong thing was which candidate got picked.

The old rule -- first frame over `min_match` -- was defending against skipping past a good
return for a marginally sharper frame later, at the cost of frames the artist then tracks by
hand. On this shot the "good return" it was defending was the occluder.

Now: **the earliest frame scoring within `band` (0.04) of the best in that sweep.** It still
refuses to skip to f45 when f26 is as good -- it picks f26 over f31 -- and never considers
f17. Measured after the change:

```
the loop proposes f26 (match 0.973)  ->  1.7 px from the hand track, ON the feature
```

A caution kept in the code: a score is comparable **within one sweep**, where every candidate
is the same patch against the same track. It is NOT comparable across tracks -- measured on
the SH004 known-answer set, the worst landings scored 0.85-0.98. This rule only ever compares
within a sweep, which is why it is sound here and would not be as a global threshold.

`tests/eval_reacquire.py` makes the reference a gate: the resume must land on a frame the hand
track actually has, and within 5 px of it. Both conditions matter and they are different
claims -- landing inside the occlusion is the failure the reference was brought in to catch.

### What this does not settle

* One occlusion, one track, one shot. The band (0.04) is a judgement; the evidence says only
  that 0.84 must lose to 0.99.
* The resume lands at f26 while the artist resumed at f25. One frame of the return is left to
  the gap. `gap = 3` starts the search at f17 and f25 did not survive candidate spacing.
* The hand track has a second gap at f33-39 that nothing here has been tested against.

### N occlusions, not one (2026-08-25, SH006)

The artist: *"there might be n number of occlusions on the same track not one or two."* Their
hand track has two in 64 frames -- runs `[[1,14],[25,32],[40,64]]`.

`tests/eval_track_vs_manual.py` drives the whole loop from the reference's own seed and scores
every frame against it. Three numbers, because they are different questions: **on the feature**,
**off the feature** (frames the artist must hunt down and delete -- worse than missing, a
confident wrong number solves), and **missed** (an honest gap, which 3DE solves across).

First run, 2 occlusions:

```
round 0: resumed at f26          first occlusion crossed
round 1: cut at f54              should have been f33
ON 21 (45%)   OFF 21   MISSED 12
```

It crossed the first occlusion and tracked straight through the second, producing 21 frames of
drift.

### The baseline was defined by the drift

`first_loss` asks whether a score has fallen below what the track "was holding", and took that
baseline as the **median of all the track's scores**. The check runs after the pass -- so by
then the drift is IN the scores:

```
baseline from the whole track   0.61      <- half the track was already drift
baseline from the first 14      0.99
```

At 0.61 the threshold is 0.30 and the occluded frames, scoring 0.35, sail through. **The wrong
frames were defining what normal looked like.** The baseline now comes from the head of the
track, which is the part anchored to the frame the artist seeded and the only part known to be
their feature.

That alone over-corrected and began cutting a gentle defocus decline, so a loss must also be a
**fall**: at least 0.20 below the recent level. Measured, the second occlusion reads
`0.86 -> 0.61 -> 0.35` across two frames; a defocus slides about 0.02 a frame. Only frames that
were ACCEPTED update the recent level, so a drift cannot become the new normal one frame at a
time.

After:

```
round 0: resumed at f26
round 1: cut at f33              the second occlusion, correctly
round 1: resumed at f41
ON 45 (96%)   OFF 0   MISSED 2
constant offset +2.5, -2.2 px    scatter 2.2 px rms
```

The two missed frames are f25 and f40 -- the first frame of each return, lost to `gap = 3`.

The 3.3 px constant offset is the artist and the correlator settling on slightly different
points OF THE SAME FEATURE; it is the same point every frame and harmless to a solve. The
scatter, 2.2 px, is the number to compare against Blender's own 2.20 px vs hand tracks.

### A threshold that was measuring the wrong thing

The gate first counted anything over 5 px as WRONG, and reported 3 failures -- all at exactly
5.0 px, with the best "recovered" frame also at 5.0 px. The same population, split by a line
invented before the reference existed. Landing on the wrong feature is hundreds of px;
disagreeing with a human click on a 3840-wide plate is a few. The gate now fails only on
**off the feature** (>25 px, or a frame the hand track has no sample for) and reports precision
separately, split into offset and scatter so one cannot hide the other.

### Two defaults that assumed one occlusion

* `rounds` was 3, capped at 10, and not exposed in the panel at all. Now 8, cap 50, and in
  Options. A shot with five occluders was not configurable.
* `min_resume_len = 12` abandons a track whose resumed segment is shorter than 12 frames. **The
  reference's middle run is EIGHT frames.** The rule exists for a real failure -- a point that
  died to blur re-acquires and dies again immediately, so the loop crawls forward a few frames
  per round forever -- but length alone cannot tell that from a short window between two
  occluders. It now gives up only on lack of PROGRESS: a short segment AND two consecutive
  resumes that barely advanced.

### What this does not settle

* One track, one shot, two occlusions. `min_resume_len`'s new progress rule is reasoned, not
  measured -- the reference never triggers it.
* The 0.20 fall and the 10-frame head are judgements fitted to one reference.
* f25 and f40 are lost to `gap = 3`. A smaller gap would recover them and has not been tested.
* Nothing here measures what happens with several tracks at once, where the sidecar batches.

### Motion does not lie: cutting jumps and slides without the plate (2026-08-25, SH006)

The artist supplied both files for the same feature -- what they hand-tracked, and what the
assistant produced beside it -- and named the two faults: *"the unwanted jumps and the
unwanted slides."*

Compared frame by frame, they are **the same signal at different sizes**:

| frame | assist step | what it is |
|---|---|---|
| f15 | **39.5 px** against a recent median of 9 | the occluder arriving |
| f20-23 | 28-34 px sustained | sliding along it |
| f24 | **70.4 px** | |
| f25-26 | -- | 120-129 px OFF the feature |
| f27 | **116.5 px** | snapping back onto the feature by luck |

**The hand track never steps more than 16 px.**

None of this is visible to a correlation score. An occluder that resembles the feature keeps
Blender satisfied the whole way -- which is precisely how the track survived to f33 without
ever dying. The appearance check catches it only when the occluder looks *different*.

`track_core.first_jump` reads the track's own positions. No plate, no sidecar, no GPU, so it
runs even when the appearance check cannot. Each step is judged against **the track's own
recent median**, not an absolute speed: a plate moving 40 px a frame everywhere is not
jumping. Steps across a gap are skipped -- a resume is a new head. Six samples must exist
first, or a track that starts slow and accelerates is cut on its own acceleration (measured:
the artist's hand track trips at f5 without it).

Chosen `k = 3.0`, floor 12 px, 6 samples. The separation on this reference is 39 px against a
9 px median -- 4.3x -- so every parameter set in a sweep from k=2.5 to 4.0 gave the same
answer, which means this shot does not discriminate between them. Conservative values were
taken for that reason, not because they were fitted.

False-positive check on SH013, where motion is genuinely 20-50 px/frame and varies: **1 of 20
tracks** would be cut, and that one goes from a 1 px/frame median to a 17 px step, which is a
real discontinuity.

`tests/test_jump.py` pins both artist files: the hand track is never cut, the assist output is
cut at f15, a gap is not a jump, a steady 40 px/frame track is not a jump, and a 400 px step
on a 40 px/frame track is.

### What this does not settle

* The jump rule fires at f67 on the reference run -- past the hand track's end at f64 -- so on
  this shot it never changed the scored result. It is proven against the artist's earlier
  output, not against the current loop's.
* One plate for the false-positive rate. 1 in 20 is not a measured rate, it is one observation.
* A slow drift that never exceeds 3x the median is invisible to it. That case belongs to the
  appearance check, and the two are deliberately independent.

### Every resume landed at its own offset (2026-08-25, SH006 v002)

The artist, on a run where the re-acquire finally worked: *"the track is placed offset in the
pattern when compared to the first frame."* Measured against their hand track, per RUN:

| run | offset vs hand track | scatter inside the run |
|---|---|---|
| f1-15 | 2.34 px | 0.38 px |
| f29-33 | **7.22 px** | 0.31 px |
| f41-64 | **5.06 px** | 0.23 px |

Each run is internally excellent -- a fifth of a pixel of scatter. But each one sits at its
own constant bias, and Blender carries that bias for the whole run once it starts.

Run 1's 2.34 px is the artist's click against the correlator's idea of the same feature, and
is unavoidable. The extra 3-5 px on the resumed runs is ours.

**Cause: the resume is planted at the peak of the LAST-GOOD patch.** That patch is cut
mid-track, and its correlation peak sits at a different sub-position within the feature than
the artist's original seed. The seed patch is already correlated at that position immediately
afterwards, to verify identity -- and its peak POSITION was being discarded, only the score
kept.

Taking the position too, within `VERIFY_RADIUS` and only when the seed patch is confident:

```
f29   7.14 px -> 3.69 px   (seed NCC 0.96)
f41   4.75 px -> 4.08 px   (seed NCC 0.93)
```

Whole-track, against the hand track:

| | before | after |
|---|---|---|
| constant offset | +2.5, -2.2 (3.3 px) | **-0.8, +2.3 (2.4 px)** |
| within 5 px | 93 % | **100 %** |
| p50 | 4.6 px | 4.0 px |

2.4 px is run 1's own baseline, so the resume-introduced bias is gone rather than reduced.

This is deliberately NOT the peak-offset metric removed earlier. That one proposed MOVING an
existing tracked marker onto a stale patch's peak, and was measured to drag healthy tracks
tens of pixels. This places a NEW marker that has no position yet, within 6 px, only when the
artist's own patch is confident there.

### Search radius: dynamic in magnitude, never in direction

Recorded because it was asked and the answer is not in one place:

* **Tracking** -- dynamic per region AND per frame. `motion.measure` reports p95 motion on a
  6x4 grid; the box is `2 * (p95 * 1.5 + pattern/2)`, capped at a quarter of plate width, and
  `refit_search` re-fits it every frame from the cell the marker has reached. SH013: 55 -> 213
  px. SH004's slow plate: stays 55.
* **Re-acquire sweep** -- `clamp(search_px / 2, 8, 96)`, inherited from the marker's own box.
* **Verify** -- fixed 6 px, deliberately: a second opinion on one spot, not a search.

**None of them uses the motion VECTOR.** Magnitude only. A box centred on the last position
must therefore cover motion in every direction while the feature travels in one, and a large
box is exactly what lets a lookalike win. Predicting the next position from the track's own
velocity would allow a much smaller box centred where the feature is going. `track_job`
already has the clamping machinery (`opts.leash`) and `first_jump` already computes per-frame
steps. Not built, not measured.

### What this does not settle

* Two resumes on one track. The refinement is bounded by `VERIFY_RADIUS = 6 px`; a resume
  landing further out than that keeps its bias.
* 2.4 px of offset remains and is attributed to the artist's click vs the correlator. That is
  an inference from run 1, not a measurement of either.

### A wire crossed the feature and a high score meant nothing (2026-08-25, SH006 Track.002)

The artist: *"a thin wire across the street crosses the pattern and the track drifts along
with the wire, but the pattern I gave is still in the frame."* Then they hand-tracked the same
feature for **250 frames** — the longest reference this project has, and the first that
reaches past f64 at all.

Measured, assist against hand track:

```
f1-65    err ~4.3 px      f85   1.48 px
f70-85   err 1.2-1.5 px   f90   6.66 px
                          f93  10.64 px     first over 10
                          f95  12.64 px
```

The divergence starts around **f88**, not f77 — the wire crosses at 77, the consequence
arrives ten frames later.

**Nothing already built could see it.** Three probes, all clean at f77:

| test | at f77 | what it would show |
|---|---|---|
| seed patch NCC | 0.976 | wrong feature -> low |
| correlation-surface shape (PREV_FRAME) | ratio 0.55, localised | a wire ridge -> <0.15 |
| motion jump | steps 9-12 px | a jump -> 3x the median |

The aperture hypothesis was wrong: neither the seed template nor the PREV_FRAME template
produces a ridge. The score then declines gradually, which `first_loss` deliberately exempts
as the defocus case.

**What did move: a lookalike 52 px away.** The margin between the best match and the best
other match collapses — 0.132 at f77, 0.073 at f85, **0.006 at f90**. On repeating texture a
high score means nothing, because there are two equally good answers.

### The margin alone cannot condemn a track

Checked against the artist's own hand track, and this is why their file mattered: at f91
**both** tracks see a margin of 0.006. It is a property of the PLATE, not of the tracker.
Cutting on it alone would have cut their own correct work — and would also have fired at f10
of the other hand-tracked feature, where the margin is 0.048 at a perfectly good position.

What separates them is the score at the position each track CLAIMS:

| frame | assist NCC | hand-track NCC | margin |
|---|---|---|---|
| f91 | **0.797** | 0.906 | 0.006 |
| f93 | **0.708** | 0.931 | 0.032 |
| f95 | **0.498** | 0.886 | 0.037 |

So the trigger is both or neither: `margin < 0.05` **and** score below `0.85x` the baseline
the track set at its own head. That cuts the drifting track at f91 and leaves the artist's
alone. Both are pinned in `tests/test_scale_drift.py`.

### Result on both references

| | reference 1 (2 occlusions, 47 frames) | reference 2 (wire, 250 frames) |
|---|---|---|
| on the feature | 43 (91 %) | **235 (94 %)** |
| off the feature | **0** | **0** |
| gaps | 4 | 15 |
| precision p50 | 4.1 px | **0.8 px** |
| constant offset | 2.5 px | **0.8 px** |

The wire track now reaches f255. The artist's own assist output for the same feature died at
f95 with 12.6 px of error.

Reference 1 lost two frames to gaps (45 -> 43) when the ambiguity trigger went in. Frames
turning into honest gaps rather than wrong data is the trade that was asked for.

### Two bugs found on the way, both mine

* `peak_margin` used `math.hypot` and `patmatch.py` never imported `math`. `hold_check` threw,
  the sidecar returned an error, the operator's fallback carried on WITHOUT cutting, and
  reference 1 silently regressed to 29 off-feature frames. **Every unit test still passed**,
  because they feed `first_loss` synthetic tuples and never call `hold_check` — the same shape
  as the panel test that never reached the branch holding a bad icon. The check existed and
  did not touch the code path.
* The harness counted frames past the reference's LAST frame as off-feature. A hand track that
  stops at f250 says nothing about f251; that measured where the artist stopped clicking.
  Inside the range a missing sample still counts — that is a frame deliberately left out
  because the feature was hidden, so a marker there is on something else.

### What this does not settle

* One wire, one track. `ambig_margin = 0.05` and `ambig_drop = 0.85` separate cleanly here —
  0.797 against 0.906 — but that is one crossing.
* The margin costs a wide correlation per frame per track, on top of the existing one. Not
  profiled.
* 15 gaps on reference 2 are unexamined; some may be recoverable with a smaller `gap`.

### QC: does the track still end on the pattern it started with? (2026-08-25)

The artist: *"always compare the pattern that the track ends at with the pattern on which the
user seeds, to check if the track is correctly tracked. Use this as one of the QC methods."*

The one question asked of a FINISHED track, and the addon had no way to answer it. Every other
check here judges a track while the assist loop is building it — the scale watch, the hold
check, the jump check. A track tracked by hand, imported from 3DE, or made before any of this
existed got nothing at all.

### The obvious version is wrong, and the artist's own files prove it

First attempt: correlate the seed patch at the last frame, compare against what it scored over
the track's opening, fail below 75 %.

```
NOT ON IT  Track.002: ends at 0.49 (49 % of its own)     the drifting assist track
NOT ON IT  Track.003: ends at 0.72 (73 % of its own)     the ARTIST'S OWN HAND TRACK
```

Their 250-frame hand track is on the feature the whole way, and still ends at 0.72 — because
over 250 frames a feature legitimately changes: perspective, light, scale. **An end-vs-start
ratio punishes long tracks for the plate doing what plates do.** Any threshold that passes
0.72 and fails 0.49 is being fitted to two numbers.

### The question that survives appearance change

Not *"does it still look the same"* but *"is there somewhere better it should be"*. A track
sitting on its feature is at the local optimum even when its score has fallen. A track that
slid off has a much better answer a short distance away:

| track | last frame | at the claimed position | best within 60 px | gain |
|---|---|---|---|---|
| Track.002 (drifted) | f95 | 0.501 | **0.979, 23 px away** | **0.478** |
| Track.003 (hand track) | f250 | 0.731 | 0.786 | **0.055** |

An order of magnitude apart, and the reading does not care that the feature looks different.
`PROBE_GAIN = 0.10` sits between them with room on both sides.

The end-vs-opening ratio is still REPORTED — it is what an artist reads to see how much the
feature has changed — but it is not the verdict.

```
NOT ON IT  Track.002: f1-95,  ends at 0.49 (49 % of its own) -- 0.97 sits 23 px away
on it      Track.003: f1-250, ends at 0.72 (73 % of its own), and nothing better is nearby
```

### Deliberately read-only

It reports and edits nothing. A QC pass that modifies what it is checking is not a QC pass —
and earlier in this same week a check that DID edit deleted correct work and reported the
deletion as a repair.

Runs on any track in the clip, selected or all. **Assist ▸ 3DE tracks ▸ Check ends on my
pattern.**

`tests/test_qc_ends.py` drives the real operator on the artist's two files, imported into
Blender from 3DE. The pair is what makes it a test rather than a threshold: both tracks meet
the SAME ambiguous texture at the SAME frames, so anything separating them must be reading the
track and not the plate.

### What this does not settle

* Two tracks, one shot. The gain separates 0.478 from 0.055 here; that is one comparison.
* `PROBE_RADIUS = 60 px` is a judgement. A track that slid further than that reads as "nothing
  better nearby" and passes.
* It checks the LAST frame only. A track that wanders off and comes back would pass.
* The probe costs one extra wide correlation per track per checked frame. Not profiled.

### A feature approaching camera is not a feature lost (2026-08-25)

The artist: *"when the track comes close to the camera and the feature perspective changes,
the track is cut until where the track maintained the perspective shape."*

The check working as written, and being wrong. A feature approaching camera stops resembling
the patch taken when it was small and far away. Its score collapses in the same shape drift
does, and nothing in `first_loss` could tell them apart — so it ended good tracks precisely
where they start to matter.

**What separates them is whether somewhere better exists.** A track still on its feature sits
at the local optimum however far its score has fallen; a track that slid off has a much better
answer a short distance away. Measured on the artist's SH006 pair, at each track's last frame:

| | claimed position | best within 60 px | gain |
|---|---|---|---|
| drifted assist track, f95 | 0.501 | **0.979, 23 px away** | **0.478** |
| artist's hand track, f250 | 0.731 | 0.786 | **0.055** |

`hold_check` now probes for that better match — one extra correlation per frame — and **every
cut requires it**. Two pinned test rows differ ONLY in that field, with identical scores and
identical margins:

```
approaching camera: gain 0.02  ->  no cut
drift:              gain 0.45  ->  cut at f44
```

### It closed a gap nothing else covered

A **slow** slide onto a neighbour: gradual enough never to trip the fall test, on texture
distinct enough never to trip the ambiguity test. Invisible to everything built so far.
"A much better match sits 20 px away" needs no rate and no margin, so it is now its own
trigger — guarded by the score having given way at all, so a healthy track with a marginally
better neighbour is left alone.

Reference 1 improved as a side effect: **43/47 -> 45/47 on feature (96 %), 0 off, 2 gaps.**
The probe recovered two frames the ambiguity trigger had been over-cutting.

### The CoTracker model, recorded because it was asked

`weights/cotracker3_scaled_offline.pth` — **CoTracker3 scaled, offline**. 102 MB, 25.5 M
parameters, 188 tensors, loaded as `CoTrackerPredictor(offline=True, v2=False, window_len=60)`.
Not a lite variant, and none is installed.

Offline is load-bearing rather than a preference: it sees the whole window at once, which is
what produces the per-frame visibility that identified the occlusion at f13-24 and the return
at f25. The online model streams and cannot. The CC-BY-NC licence noted in `cotrack.py` applies
to this model.

### What this does not settle

* `PROBE_RADIUS = 60 px`. A track that slides further than that reads as "nothing better
  nearby" and survives.
* The probe costs one wide correlation per frame per track on top of the existing one. Still
  not profiled.
* The perspective case is pinned as a synthetic pair, not against footage of a feature actually
  approaching camera. The artist reported it; no reference for it exists.

### Cut too early, and never coming back (2026-08-25)

Two reports: *"tracks are not tracked completely even though the tracks are still inside the
frame"*, and *"CoTracker is not acquiring the tracks if they leave the frame and enter again
after a few frames."* Neither was CoTracker. Both were assumptions written here this week.

### Re-entry was made impossible on purpose

Three commits earlier, `edge_stop` gained this:

```python
# It did not fail, it left. Every sweep for it would be looking off-plate.
self._gave_up.add(tr.name)
```

True only while the feature is outside. A feature that leaves and returns a few frames later
was structurally unrecoverable, which is exactly what got reported. Removed. Off-plate
positions cost nothing to sweep — `reference_patch` refuses a box that does not fit, so the
correlation is skipped until the feature is back inside; measured directly:

```
well inside              -> score 0.695
just off the right edge  -> no match (skipped)
off the top              -> no match (skipped)
```

### The early cuts: three causes, compounding

**1. `settle` was fitted to a synthetic case.** The artist's 250-frame hand track — every frame
correct by construction — contains a **four-frame** run where the frame-1 patch scores as low
as **0.132** at a position they tracked by hand, then recovers to 0.731:

```
f150 0.979   f200 0.939   f216 0.692   f231 0.132   f250 0.731
```

A 230-frame-old patch stops describing the feature for a moment. At `settle = 2` that cut
their track at f230 — and at f216 once the probe trigger was added. Their drift on the same
shot runs **five** frames and never recovers, so 5 is the separation, taken from footage
rather than taste.

**2. Rejected frames were defining the "recent level".** Bad frames were appended to `recent`,
so by the third one the drop was no longer a fall against it, the run reset, and a long
failure could never reach the settle count. Invisible at 2; fatal at 5 — three tests failed
the moment it was raised, which is how it surfaced.

**3. A cut track was re-acquired with the patch that caused the cut.** After the hold check
cuts at f91, the resume localised with the last-good patch from f90 — but the track was
already **5.65 px** onto the lookalike at f87. It landed where the artist's own patch scores
**0.64**, the identity gate correctly refused it, and the track died with 160 frames still to
run. At the right position that same patch scores **0.931**.

Every component behaved as designed and the outcome was still wrong: the reference was
poisoned by the drift the cut existed to remove. A hold or jump cut now sends no `last_box`,
so the sidecar localises with the artist's own patch. Normally last-good is far better —
0.46 px against 3.87 px on the SH004 known-answer set — but that assumes the recent frames
were ON the feature, and a cut asserts the opposite.

**4. And that silently disabled the position refinement.** The verify step was gated on
`localised`, so a resume searched with the seed patch skipped refinement entirely. Cost p50
0.8 px -> 3.4 px. The gate looked like it was guarding correctness and was actually guarding
an unrelated code path. It now runs whenever a seed patch exists.

### Both references, best yet

| | reference 1 (2 occlusions, 47 frames) | reference 2 (wire, 250 frames) |
|---|---|---|
| on the feature | **46 (98 %)** | **243 (97 %)** |
| off the feature | **0** | **0** |
| gaps | **1** | **7** |

Reference 1 was 45/47 with 2 gaps; reference 2 was 235/250 with 15.

### One number that got worse, stated plainly

Reference 2's precision loosened: p50 0.8 px -> 3.4 px, of which 2.3 px is a constant offset.
That offset is the artist's click against where the correlator centres on the same feature —
identical to reference 1's 2.5 px baseline, constant within a run, and benign to a solve. The
earlier 0.8 px was partly flattering: that run tracked THROUGH the drift region, so its
scored frames were the easy ones. More coverage of harder frames is not a free improvement
and should not be presented as one.

### What this does not settle

* `settle = 5` comes from one hand track's one transient. A correct track with a six-frame dip
  would still be cut.
* The seed-patch-after-a-cut rule is reasoned from one measurement (0.64 against 0.931). It has
  not been scored on a clean death, where last-good is known to be better.
* The 2.3 px offset is attributed to click-vs-correlator by inference from two references, not
  measured directly.

### CoTracker driving the track itself (2026-08-26)

Asked: *"what if we use CoTracker directly for tracking, and Blender tracking as two
separate modes?"* Worth a number rather than an opinion, and the number is already cheap to
get: `cotrack.track_points` returns a position for EVERY frame on every re-acquire, and
`resume_path` keeps one of them. "CoTracker as its own mode" is not a new capability, it is
a decision to keep what is already computed.

`tests/eval_cotracker_direct.py` scores that path against a hand track with the same rules
as `eval_track_vs_manual.py`, so the two sit side by side.

| | ref 1 — 2 occlusions (47 fr) | ref 2 — wire (250 fr) |
|---|---|---|
| assist loop | **46 (98 %)**, 0 off, 1 gap | 243 (97 %), 0 off, 7 gaps, **p50 3.4 px** |
| CoTracker direct @768 | 14 (30 %), **50 off**, to 140 px | **250 (100 %)**, **0 off**, p50 **13.0 px** |
| CoTracker direct @1920/1536 | 14 (30 %), 50 off | 243 (97 %), p50 14.4 px |

**Identity is what it is good at.** 250/250 on the wire shot — the shot where the assist
loop cuts 7 frames, and where the artist reported a slide at f77 and again at f90. It never
left the feature once.

**Precision is what it is bad at, and resolution does not fix it.** 13 px at 4K, and raising
the long edge from 768 to 1536 made it WORSE — 13.0 -> 14.4 px, and it lost 7 frames it had
at 768. CoTracker3 is trained near 512; a 4K plate fed in at 1536 is out of distribution.
The instinct to "run it at full res for accuracy" is measurably backwards here.

**Raw, it fails occlusions harder than the loop does.** 30 % on ref 1, ending 140 px out.
It glides through the occluder and then tracks the occluder — there is nothing in the model
that knows the artist's pattern. It works in the current design only because NCC verifies
where it lands and refuses it when wrong, which is the whole reason the loop beats it 98 %
to 30 % on the same footage.

So as a replacement it is worse in both directions. What the numbers actually argue for is
neither mode: keep the full path and use it as a **leash** — CoTracker says where the
feature is within ~13 px on every frame, Blender's NCC refines to sub-pixel inside that
window. A 13 px prior makes a 130 px slide structurally impossible, which turns the drift
class from something detected after the fact into something that cannot happen.

Two constraints on anything built from this:

* **Licence.** CoTracker is CC-BY-NC — it restricts USE, not only redistribution. It is
  already in the loop for re-acquisition; making it the tracker of record for delivered
  tracks deepens that exposure. TAPNext++ (Apache-2.0) is vendored with weights present.
* A genuine direct mode still earns its place on plates where Blender tracks nothing at all
  — the SH013 motocross FG. 13 px beats no track.

### What this does not settle

* Two references, one shot, one feature each. Ref 2 has no true occlusion and ref 1 has two;
  no reference here has fast FG motion, which is the case a direct mode is FOR.
* The leash is reasoned from these numbers, not measured. 13 px is a p50 — the p90 is 22 px
  and the max 24 px, so a leash sized on the median would be too tight on one frame in ten.
* Chained windows re-query at the previous window's last position, so error compounds across
  window seams. Ref 2 crossed two seams; a 600-frame shot would cross more.

### The leash: a guide for every frame, and knowing when to ignore it (2026-08-26)

Asked for after the measurement in the previous section: use CoTracker's whole path, not the
one number the re-acquire keeps. **The version that was pitched does not work, and the
reason is worth more than the feature.**

### Steering with it drags correct tracks off their feature

The pitch was per-frame steering: CoTracker says where the feature is within ~13 px,
Blender's NCC refines inside that window, and a 130 px slide becomes impossible. Re-anchored
to the last confirmed frame the guide is genuinely tight -- its displacement over a short gap
agrees with a hand track to 2.02 px at gap 1 and 7.20 px at gap 30, sublinear, so a `sqrt`
tolerance fits it.

On the 250-frame shot. On the 64-frame shot with two real occlusions the same model walks
onto the OCCLUDER at the first cover and never returns: 4.54 px error at gap 1, 266 px by
f64. A leash built from that path demanded a cut at **f27 of the artist's own hand track**.

| | ref 1 (2 occlusions) | ref 2 (250 frames) |
|---|---|---|
| guide displacement error, gap 1 | **4.54 px** | 1.08 px |
| guide displacement error, gap 30 | **127.90 px** | 4.03 px |

Same model, same plate, same artist. A leash cannot be trusted unconditionally, and
CoTracker's own visibility head does not separate the two cases -- it is a threshold on
covered-ness, not on identity.

### What does separate them: closure

Track forward from the seed, then track BACK from where the forward pass ended, and compare.
Using only what exists at runtime -- the backward query is the forward pass's own endpoint,
never a known-good position:

| shot | closure p50 / p90 / max | the guide's TRUE error p50 / max |
|---|---|---|
| 2 occlusions | 21.5 / **62.1** / 103.6 | 152.5 / 264.7 |
| 250 frames | 8.5 / **11.1** / 12.7 | 5.1 / 12.2 |

5.6x on p90. Closure badly UNDERSTATES the true error (21.5 against 152.5) because both
passes share a model and therefore share its faults -- the same limitation
`tools/make_lk_reference.py` records for round-trip closure. It is a detector, not a
measurement, and it is only ever asked yes or no.

### Trust has to be LOCAL, or it condemns a guide that is fine where it is used

The guide's reliability falls off with distance from its query, and a fill only ever reaches
a few frames. Over the whole 151-frame window after f90 the closure is 66.0 px --
untrustworthy -- while over the five frames actually being filled it is 15.6 px and the leash
is accurate to 2 px there. A verdict taken over the window throws away the frames it was
computed to protect.

### So the leash does not steer. It bridges.

Where the loop cuts and re-acquires a few frames later, the frames in between are simply
missing from the exported track. On the artist's 250-frame reference the gaps were f91-95 and
f211-212 -- **and their hand track has a sample on every one of them.** The feature was
visible the whole time; the loop cut as a precaution. That is 7 frames of work handed back to
the artist by a tool that exists to remove it.

Four gates, each earning its place on those two shots:

* **closure over this gap** -- as above;
* **CoTracker must call the feature visible.** Visibility is a report and not a gate
  everywhere else in this codebase, and for good reason: using it to gate the SEARCH killed
  whole tracks. A fill is optional, so a false negative costs an empty gap that was already
  empty. Same signal, opposite cost;
* **the peak, probed WIDER than the tolerance, must come back inside it.** Searching only as
  far as the tolerance guarantees an in-range answer and proves nothing -- the same "is there
  somewhere better" reasoning the hold check and the QC pass already use. This is the gate
  that does the work: across a real occlusion the peak lands 3.7-19.8 px from the prediction,
  across a precautionary cut 1.8-4.0 px;
* **contiguity.** A fill that skipped a frame and carried on would be an island across an
  occlusion -- markers on the occluder with correct-looking neighbours either side.

### Filled from BOTH ends, because neither end is reliably the good one

Working forward from the cut fails on exactly the case that matters. When a track dies to
DRIFT, its last good frame is already on the wrong feature, so a guide queried there follows
the wrong feature:

```
guide queried at the artist's hand-tracked f90   -> closure 15.2 px over the gap
guide queried where Blender actually was at f90  -> closure 132.9 px
```

Same gap, same frames. The trust gate correctly refuses the second -- and correctly refuses
to fill anything.

The resume end has no such problem: it was correlated against the artist's own pattern and
scored 0.96 before anything was planted. So the return pass is queried at the **verified
resumes** rather than at the forward pass's endpoint, which costs nothing extra
(`track_points` takes queries on different frames, so a group of tracks that came back at
different moments is still one pass) and anchors the second opinion on evidence instead of on
another guess.

The closure gate then applies only to the cut-anchored guide -- the one with no independent
support. Refusing the resume-anchored guide for disagreeing with it would be refusing the
trustworthy end for disagreeing with the untrustworthy one, and closure cannot say which side
is wrong. Per-frame evidence rules on that one, and the occlusion reference is the check that
it is enough.

### Both references, and the occlusion reference is the one that matters

| | ref 1 (2 occlusions) | ref 2 (wire, 250 frames) |
|---|---|---|
| on the feature | **46 (98 %)** | **246 (98 %)** — was 243 |
| off the feature | **0** | **0** |
| gaps | **1** | **4** — was 7 |

Every occlusion gap on ref 1 was refused, including on the un-gated backward walk. That is
the result the whole design hangs on: a bridge across a real occlusion is a marker on the
occluder, which is worse than the hole it fills.

### What this does not settle

* `TRUST_P90_PX = 25` sits between 11.1 and 62.1 -- two shots, one plate.
* The closest call in the whole measurement is f33 on ref 1: NCC 0.881, peak 3.66 px from the
  prediction against a 3.30 px tolerance. It is refused by 0.36 px. A slightly wider
  tolerance plants a marker on an occluder.
* Closure cannot see a guide that fails identically in both directions. Nothing here
  overrides the pattern check for exactly that reason.
* The fill costs one extra CoTracker pass per group of deaths. Not profiled against a real
  batch of 20 tracks; measured only on single-track references.
* Both references are the same plate and the same artist.

### A new marker that is the same size on screen at any zoom (2026-08-26)

Reported: Ctrl-clicking on a 4K plate makes a pattern box so small it has to be zoomed into,
dragged bigger, and zoomed back out -- once per track.

A unit mismatch, not a bad default. Blender sizes a new track from
`clip.tracking.settings.default_pattern_size`, which counts PLATE pixels. The default is 21.
That is a sixth of the width of a 128-px proxy and 0.55 % of a 3840-px plate, and fitting 4K
into a clip editor puts the zoom near 26 %:

```
21 plate px at 26 % zoom  ->  5.5 pixels on screen
```

Which is what was being clicked into existence.

### Kept in step with the zoom instead

Plate pixels and screen pixels are related by the zoom alone, so the setting Blender already
reads can simply be kept correct:

    plate_px = screen_px / (zoom_percentage / 100)

| zoom | plate px for a 40 px box | was |
|---|---|---|
| 25 % | 160 | 21 |
| 50 % | 80 | 21 |
| 100 % | 40 | 21 |
| 400 % | 16 (the floor) | 21 |

Nothing intercepts the click. `clip.add_marker` stays Blender's own operator with its own
drag-to-place behaviour, and it reads a default that happens to be right by the time it runs.
Two clamps: never under 16 plate px, because a smaller patch holds too little texture to
correlate however big it looks; never over a quarter of the short edge, for the
zoomed-far-out case where the arithmetic would ask for a box bigger than the frame.

### A timer, not a draw handler

The obvious place to read the zoom is a draw callback -- it runs on every zoom, for free. But
a draw callback must not write data and this writes two RNA properties. `bpy.app.timers` runs
where that is allowed. 0.2 s is a latency budget, not a sampling rate: it only has to beat
the gap between the artist stopping a scroll and clicking. The tick writes only on a real
change, because writing the same value back tags the ID modified and asks the artist to save
work that did not happen.

### The setting still belongs to the artist

`default_pattern_size` is a real preference someone may have set deliberately, and this
overwrites it. The value in force when a clip is first touched is remembered and put back
when the option is switched off or the addon is unregistered. Taking a setting over for the
session is defensible; keeping it afterwards is not.

### The side effect it had to not have

Importing 3DE tracks also creates tracks from that default, and `ops_qc` correlates using the
resulting pattern box. Left alone, a track imported at 25 % zoom would carry a 160 px pattern
and QC would be reading a patch six times the size of the feature -- a change to how a FILE
is read, caused by how zoomed in someone happened to be. Import now pins the box to
`click_size.artist_default(clip)`, the artist's own value, so reading a file off disk behaves
exactly as it did before any of this.

`tests/test_click_size.py` drives the INSTALLED extension: the loose module on `sys.path` has
no addon preferences, so testing against it would exercise the arithmetic twice and never run
the reconciler at all. It also asserts `register()` actually scheduled the timer -- every
other check calls `_apply` by hand and would pass just as well if nothing were ever scheduled
to call it -- and that the restore works without the test planting the saved value first.

### What this does not settle

* 40 screen px is a guess at what reads well, exposed as a preference rather than measured.
* `zoom_percentage` is taken as the whole story. On a HiDPI display where Blender applies a UI
  scale, "screen pixels" and device pixels may differ; not checked on such a monitor.
* A clip open in two editors at different zooms gets whichever the loop reaches last.
  Harmless -- the size only matters at the click -- but it is not defined behaviour.
