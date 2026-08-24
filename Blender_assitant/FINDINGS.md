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
