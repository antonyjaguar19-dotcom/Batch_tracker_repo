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
