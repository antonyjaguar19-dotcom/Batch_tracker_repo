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
