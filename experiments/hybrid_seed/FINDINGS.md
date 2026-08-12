# Hybrid seeding: TAPNext seeds, SynthEyes tracks

Question asked: can one engine pick the features and the other track them?

Answer: **yes, the mechanism works and is now measured end to end.** The accuracy question
is currently unanswerable on this machine, because SynthEyes is running as **Pro Demo**,
which stops tracking every tracker after 10 frames.

Nothing in `app/` was modified. Everything here imports the bot rather than changing it.

## What is proven

Measured on SynthEyes 2026.2.4679, plate `D:\Jefrin\IN\SH004.mp4` (2560x1440, 160 frames).

| Step | Result |
|---|---|
| Inject external seed points | works — `tk = new ob.trk`, `tk.key = Point(u,v)` |
| Seed lands where asked | **max round-trip error 0.001px** over 122 seeds (`check_roundtrip.py`) |
| Track them | works — `tk.Run()` once per frame |
| Export | works — the engine's existing Sizzle export, 122/122 tracks |
| Speed | **unknown — see the correction below** |

### Correction: the throughput number here was wrong

An earlier version of this file claimed ~1,400 tracker-frames/s (122 trackers x 160 frames
in 13.1s). **That figure is not real.** Under the Demo cap most trackers were dead by frame
10, and `tk.Run()` on a dead tracker is a no-op, so the timer was mostly measuring nothing
happening. The true cost of tracking has not been measured on this box.

Two related things came out of chasing it, and both are now handled in the code:

- **Tracker geometry must be specified in pixels, not normalised units.** `size` / `srchu` /
  `srchv` are fractions of the plate, so a fixed set of numbers makes every tracker twice as
  wide on a 4K plate as on a 2K one, and area-match cost grows with the square of that.
  `KIND_GEOM_PX` in `run_hybrid.py` is in pixels and converted per plate.
- **A whole shot in one `RunScriptFile` call is a bad idea.** On SH016 (4096x2160, 127
  frames) SynthEyes went to the Windows "Ghost" not-responding class and then dropped the
  socket (`WinError 10054`). Tracking is now done in bounded chunks (`--frames-per-call`),
  every call is time-limited (`--chunk-timeout`), and a hang is reported with the frame
  range in flight instead of blocking forever.

The whole path is Sizzle plus SyPy3. It never enters the Features room, so it does **not**
need `_click_panel_button`, does not take the mouse pointer, and is not subject to the
`Maximum tracker count` default of 120 — the tracker count is whatever the seeder hands
over, because we create the trackers ourselves.

## Three traps, all measured

**1. `Scene.RunTrackersFwd()` does nothing under `RunScriptFile`.**
It is the idiom in SynthEyes' own `scripts/Trackers/trackbyx.szl`, it raises no error, and
it leaves every tracker valid on its key frame only. The call that actually advances a
tracker one frame is **`tk.Run()`**. `trackbyx.szl` works because it is an attached,
interactive tool script; a script run through `RunScriptFile` is a different context.

**2. `Scene.SetFrame` is not needed, and the Sizzle `frame` global is enough.**
Verified directly (`szl/img_changes.szl`): sampling `tk.AvgImgColor()` at frames 0..152
gives identical values whether or not `Scene.SetFrame` is called, and the values change
frame to frame, so the plate really is advancing. Dropping `SetFrame` costs nothing.

**3. Any SyPy3 call that desyncs the socket makes the next `RunScriptFile` a silent no-op.**
`hlev.Validate()` does this. The symptom is a snippet that "did nothing" with no error —
which sends you hunting the wrong bug. `sylab.py` now calls `_resync_socket()` before
every run.

## The blocker: SynthEyes is in Demo mode

```
Unnamed - SynthEyes Pro Demo 2026 Build 4679 64-bit - Camera01 - D:\Jefrin\IN\SH004.mp4
```

Every injected tracker tracks correctly for exactly 10 frames and then stops:

```
f= 8 valid=1 u=-0.07792 v=0.32242
f= 9 valid=1 u=-0.07808 v=0.32210
f=10 valid=0 u=0.00000  v=0.00000     <- and never recovers
```

Of 122 trackers: 80 go invalid at frame 10, 41 jump to a fixed position and hold it
unchanged for the remaining 150 frames, 1 stops at frame 8.

This was **not** caused by the injection method. Ruled out by measurement:

- not the coordinates — round-trip is exact to 0.001px
- not the image — `AvgImgColor` keeps changing past frame 10, at every frame sampled
- not the plate — a full inter-frame difference scan of SH004 shows no cut or flash at
  frame 10 (diff 0.75 there against a 1.95 peak at frame 59)
- not the RAM cache — `hlev.Validate()` on the shot changes nothing
- not tracker parameters — a sweep over `size`, `srchu/srchv`, `smooth`, `autokey` and
  `kind` (`szl/sweep.szl`) gives `lastvalid=9` on **every** variant
- not the enable channel — `tk.isEnabled` reads 1 at every frame, before and after
- not key spacing — re-keying at the tracker's own position every 8 frames still dies at 9

The held-coordinate tail matches what this repo already documents about the Demo build:

> *"The Demo build's 'frozen tail' truncation was removed for the Pro license: Pro produces
> real full-length tracks with no held-coord padding"* — `app/syntheyes_engine.py:1952`

Strictly: the Demo mode is confirmed from the window title; that it is the *cause* of the
10-frame cap is inference, but it is consistent with the licence state, with the observed
held-coord tail, and with everything else being ruled out. The clean control — run the
bot's normal blip/peel path and see whether it is also capped — has not been run.

**This affects the shipping bot, not only this experiment.** If the licence has lapsed,
the SynthEyes backend is currently producing 10 good frames and a frozen tail on every
shot, and the repo comment above assumes a Pro licence that is not in effect.

## Second finding: mid-shot seed creation does not take

A tracker created at frame 80 never becomes valid, while the identical tracker created at
the shot start does. Tried and failed: `tk.mainFrame = 80`, and `tkgrid.szl`'s documented
mid-shot pattern of keying `isEnabled = 0` at start then `= 1` at the creation frame.

Consequence: the hybrid forces `seed_stagger = 1` (all seeds on the first frame) rather
than silently creating dead trackers. That is the bot's own legacy setting, not an
invention, but it does mean the hybrid cannot yet cover content that enters mid-shot.
Worth revisiting — it may share a cause with the Demo cap.

## Per-seed tracker geometry

Because we create each tracker, the seed class from `app/track_meta.py` can be turned into
SynthEyes tracker fields — a corner gets a small tight patch, an edge point a wider search
across the edge (`KIND_GEOM` in `run_hybrid.py`). **Neither engine does this today**, and it
is the strongest argument for the hybrid. It is untested: with tracking capped at 10 frames
there is no honest way to measure whether it helps. `--flat-geom` runs the control.

## Scoring

Run against `refs/SH004_lk`, and read only as a demonstration that the harness is wired —
the numbers describe the demo cap, not the method:

```
ref014  corner  13.46px      ref017/048/052/069  MISSED
scored 1/5
```

Four of five references are missed because 122 seeds on frame 0 do not land near those
features. Both that and the 13.46px are dominated by the frozen tail.

## Files

| File | What it is |
|---|---|
| `run_hybrid.py` | the e2e: bot seeder -> inject -> track -> 3DE export |
| `check_roundtrip.py` | proves the injected seeds land where asked |
| `render_overlay.py` | burns tracks over the plate; frozen points drawn hollow grey |
| `sylab.py` | runs a `.szl` against the live SynthEyes; ~1s iteration |
| `probe_inject.py` | the original minimal probe |
| `szl/` | the diagnostics, each one answering a single question |
| `out/SH004__hybrid.txt` | the export |
| `out/SH004__hybrid_overlay.mp4` | the video |

## How to re-run

```bat
runtime\python311\python.exe experiments\hybrid_seed\run_hybrid.py --seeds 400
runtime\python311\python.exe experiments\hybrid_seed\check_roundtrip.py
runtime\python311\python.exe experiments\hybrid_seed\render_overlay.py
runtime\python311\python.exe tools\eval_refs.py refs\SH004_lk --bot experiments\hybrid_seed\out\SH004__hybrid.txt
```

Expected today: 122 tracks, ~13s, round-trip max 0.001px, and an overlay that tracks for
10 frames then goes grey.

Expected on a licensed SynthEyes: tracks running the full 160 frames, at which point
`eval_refs` becomes meaningful and the per-kind geometry can be tested against
`--flat-geom`.

## Third finding: this dev box's SynthEyes stopped working entirely

Late in the session, after several force-kills following the 4K hang, **every** Sizzle script
began hanging on this machine — including `szl/inspect.szl`, a two-line read-only script that
had been completing in 0.0s all day, and including on a freshly launched instance. A fresh
launch leaves a visible `SplashPopup` child window that never clears, so SynthEyes is not
finishing its own start-up.

That is a machine/licence state problem, not a code one, but it has a direct consequence
worth being explicit about:

**What is verified, and what is not.** The Python side of the experiment suite is verified
— the seeder, plate handling, probe verdict logic, round-trip comparison and the licence
check were all run and behaved correctly (`check_licence.py` correctly returned DEMO/exit 2;
the mid-shot probe correctly returned FAIL with a passing control; the re-acquire probe
correctly returned INCONCLUSIVE rather than a false FAIL). The **rewritten SynthEyes path**
— chunked tracking, the bounded-call hang guard, the shot-load verification, the splash
wait, and the whole `--reacquire` resume path — could **not** be re-run end to end after the
rewrite, because by then this box would not execute any Sizzle at all. Its first real test
will be the prod run.

The code is defensive about exactly this: every call is time-bounded and the plate load is
verified, so a bad state should produce a named error rather than a hang or a silent wrong
answer.

## Next step

Run `RunHybridExperiment.bat` on the licensed box (see `README.md`). It hard-stops if the
licence is not active, and answers the mid-shot-creation and re-acquisition questions before
producing any numbers.
