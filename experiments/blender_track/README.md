# Blender hybrid experiment — how to run it

TAPNext picks the features and says where they reappear after an occlusion; **Blender**
does every per-frame measurement in between.

Everything here is an experiment. **`app/` is not modified**, so the bot is unaffected by
anything in this folder. Results and plates land in `out/`, which is gitignored.

Unlike the SynthEyes hybrid next door, this needs **no licence** and **never takes the
mouse** — Blender tracks fully headless.

## Setup

Blender is a portable unzip, not an installed app, so its location is a setting:

```bat
set BTR_BLENDER_EXE=C:\path\to\blender.exe
```

Default is the 5.2.0 portable build under `Downloads\`. Nothing else to install — the
Blender side runs on Blender's own Python and imports nothing from this repo.

## Run it

### Against synthetic ground truth (no GPU, ~30s)

The fastest honest measurement. Seeds come from an export already on disk, so nothing has
to run TAPNext:

```bat
runtime\python311\python.exe experiments\blender_track\run_blender_hybrid.py ^
    --plate bench\synth\lab02\plate --name lab02 ^
    --reuse-tapnext bench\synth\lab02\runs\base\lab02__tapnext.txt ^
    --tag kind --out bench\synth\lab02\runs\bl_kind\lab02__tapnext.txt

runtime\python311\python.exe bench\score_synth.py bench\synth\lab02 --run bl_kind --baseline base
```

### Against a real plate

The plate must be a **folder of frames**. If yours is a movie, extract it first — Blender
and the bot then read identical pixels, instead of two different decoders:

```bat
runtime\python311\python.exe experiments\blender_track\extract_frames.py ^
    --mp4 D:\Jefrin\IN\SH004.mp4 --out experiments\blender_track\out\SH004
```

Then the bot for the guide, then Blender. **Use these two `--set` flags** — they are what
the measurements in `FINDINGS.md` were taken with:

```bat
REM ~139s. Refines OFF because Blender does that job, and spacing opened up so the
REM guide is dense: at the default 60 this shot exports 28 tracks, at 15 it exports 122.
runtime\python311\python.exe bench\run_bench.py --shot experiments\blender_track\out\SH004 ^
    --tag dense_raw --set track_spacing_px=15 --set moving_tile=0 --set pattern_refine=0

REM ~26s
runtime\python311\python.exe experiments\blender_track\run_blender_hybrid.py ^
    --plate experiments\blender_track\out\SH004\plate --name SH004 ^
    --reuse-tapnext experiments\blender_track\out\SH004\runs\dense_raw\SH004__tapnext.txt ^
    --tag dense
```

`tools\eval_refs.py refs\SH004_lk` will run against the result, but read `first_step`
before believing any row: at 122 tracks the proximity pairing starts matching a reference
to a *neighbouring* feature, and a 12px first-step means the pairing is wrong, not the
track. See `FINDINGS.md`.

Drop `--reuse-tapnext` to have the runner call the bot itself. That builds a `RunnerConfig`
directly rather than going through the shipping `AppState` mapping, so the two-step form
above is preferred when the numbers matter.

### Tune it

```bat
runtime\python311\python.exe experiments\blender_track\sweep.py --shot bench\synth\lab02
```

Ten Blender configurations against exact ground truth, ~3 min, no GPU. As measured, the
**defaults already win** — Affine, Perspective, `PREV_FRAME` and bigger pattern boxes are
all worse. Re-run it after changing `KIND_GEOM` in `bl_track.py`.

### Check the reappearances

The thing to distrust. A track that resumes on the wrong feature is *longer*, so every span
number gets better while the output gets worse:

```bat
runtime\python311\python.exe experiments\blender_track\check_replants.py ^
    --plate experiments\blender_track\out\SH004\plate ^
    --tracks experiments\blender_track\out\SH004__dense__blender.txt --sheet
```

`--sheet` writes a contact sheet — before the gap | after the gap | best match nearby, worst
first — so a bad reappearance can be seen rather than inferred.

**Run `--selfcheck 4` on any new plate before believing the numbers.** It punches synthetic
gaps into *continuous* tracks, where the correct answer is known to be ~0px, so anything
below ~95% "on the feature" means the checker is wrong on that footage, not the tracker.
The first version of this metric failed exactly that test at 45%, because it was measuring
repetitive texture rather than the replant.

### Watch it

```bat
runtime\python311\python.exe experiments\blender_track\render_overlay.py ^
    --plate experiments\blender_track\out\SH004\plate ^
    --bot   experiments\blender_track\out\SH004__repl__blender.txt ^
    --guide experiments\blender_track\out\SH004\runs\guide\SH004__tapnext.txt
```

Green = Blender's measurement, orange = the TAPNext guide it was seeded from, with a line
between them so disagreement is visible. A hollow grey ring is a track inside a gap,
waiting for its replant — that is what the replant stage exists for.

## What to look for

**`seed round-trip : max <n>px  PASS` on every run.** This is the one that matters. It
proves each Blender tracker started on the pixel the seeder asked for. If it FAILS, stop:
the trackers are on the wrong features and every other number is meaningless. It exists
because exactly that happened, and the synthetic bench scored it as a *good* run — its
ground truth is anchored per track, so a track planted in the wrong place still looks
clean. See `FINDINGS.md`.

Then:

- **mean_err / p90 in `score_synth`** — the accuracy claim.
- **median track length** — what replant is worth. Compare against `--no-replant`.
- **replant rounds** in the Blender log — zero means nothing died, so nothing was tested.

## Controls worth running

| Flag | What it isolates |
|---|---|
| `--flat-geom` | one pattern/search size for every seed, instead of per seed class |
| `--no-replant` | how much of the track span comes from resuming dead tracks |
| `--no-backward` | whether late-seeded tracks are covering the head of the shot |
| `--source seeder` | the raw seeder with no guide at all — no TAPNext pass needed |
| `--motion-model`, `--pattern-scale`, `--search-scale`, `--pattern-match` | what `sweep.py` sweeps |
| `--max-guide-dev PX` | truncate where Blender and TAPNext disagree. **Measured as harmful** against a raw guide — it trims the better of the two. Off by default; see `FINDINGS.md` |

## If it fails

The runner prints Blender's last 40 lines on a non-zero exit, so the exception is in the
output. Common ones:

1. **`blender.exe not found`** — set `BTR_BLENDER_EXE`.
2. **`clip is AxB but seeds are for CxD`** — the plate handed to Blender is not the plate
   the seeds were measured on. Usually a movie/sequence mix-up; pass the frames folder.
3. **`No such file or directory`** on the plate — pass an absolute path; Blender resolves
   relative paths against the .blend, not the shell.

## Files

| File | What it is |
|---|---|
| `run_blender_hybrid.py` | the whole thing: seeds/guide -> Blender -> 3DE export |
| `bl_track.py` | the inside-Blender worker (seed, track, replant) |
| `blio.py` | plate access, coordinates, the Blender call |
| `extract_frames.py` | movie -> PNG frames laid out as a bench shot |
| `check_replants.py` | did the disappeared tracks come back on the same feature |
| `sweep.py` | tunes the Blender side against exact ground truth |
| `render_overlay.py` | the video |
| `FINDINGS.md` | what has actually been measured, and what has not |
