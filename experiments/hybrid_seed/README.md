# Hybrid tracking experiment — how to run it in prod

TAPNext's seeder picks the features, SynthEyes tracks them.

Everything here is an experiment. **`app/` is not modified**, so the bot is unaffected by
anything in this folder.

## Run it

On the machine with the **Pro licence**:

```bat
git pull
experiments\hybrid_seed\RunHybridExperiment.bat "D:\path\to\SHOT.mp4"
```

The plate can be a movie file **or** a folder of frames (EXR/DPX sequences are built into an
IFL the same way `process_shot` does it):

```bat
experiments\hybrid_seed\RunHybridExperiment.bat "\\liv1\shows\SHOW\SH010\in\plates\v001"
```

Add a reference folder as a second argument to get scored numbers instead of just videos:

```bat
experiments\hybrid_seed\RunHybridExperiment.bat "D:\plates\SH004.mp4" refs\SH004_lk
```

If SynthEyes is installed somewhere other than
`C:\Program Files\BorisFX\SynthEyes 2026\SynthEyes64.exe`, either set `BTR_SYNTHEYES_EXE`
and `BTR_SYPY3_DIR` first, or edit the two marked lines near the top of the bat.

**Close SynthEyes before starting.** The experiment launches its own instance, and a warm
one left in an odd state is the single most common cause of a hang.

## What it does, and what each step tells you

| # | Step | Meaning |
|---|---|---|
| 1 | Licence check | **Hard-stops on a Demo build.** A Demo tracks ~10 frames per tracker and holds a frozen coordinate for the rest, so everything below would describe the licence, not the tracker |
| 2 | Mid-shot creation | If it PASSES, staggered seeding is switched on for steps 4–5, and the wired mode can seed content that appears part-way through a shot |
| 3 | Re-acquisition | Can a *live* tracker be re-keyed mid-shot and carry on? This is what "replant the same seed when the feature comes back" needs. If it FAILS, step 5 is skipped |
| 4 | Baseline run | Full hybrid, seeds on the first frame |
| 5 | Run with re-acquisition | Same seeds, dead tracks resumed as **the same track with a gap** |
| 6 | Round-trip check | Proves the seeds landed exactly where the seeder asked. Must say PASS — if it fails, every tracker started on the wrong feature and all other numbers are meaningless |
| 7 | Videos + scoring | An overlay mp4 per run, plus `eval_refs` if you passed a reference folder |

Results land in `experiments\hybrid_seed\out\` — exports, the generated `.szl` scripts, and
the videos. That folder is gitignored, so prod results stay on the prod box.

## What to look for

- Step 1 must say **LICENSED**.
- Step 6 must say **PASS**.
- In the videos, points should keep moving for the **whole shot**. If they go grey around
  frame 10, the licence is not actually active.
- The per-chunk lines print real throughput. Watch them: if tracker-frames/s collapses, the
  patch/search sizes are too big for that plate's resolution.

## If it hangs or fails

The runner bounds every SynthEyes call (`--chunk-timeout`, default 300s) and reports which
frame range was in flight rather than blocking forever. If a chunk times out:

1. Close SynthEyes completely and re-run — a wedged instance is the usual cause.
2. Lower the load: `--frames-per-call 10 --seeds 150`.
3. On very large plates, try `--no-validate` (skips the RAM cache).

Individual steps can be run on their own, which is the fastest way to narrow a problem:

```bat
runtime\python311\python.exe experiments\hybrid_seed\check_licence.py
runtime\python311\python.exe experiments\hybrid_seed\probes.py midshot   --plate "<plate>"
runtime\python311\python.exe experiments\hybrid_seed\probes.py reacquire --plate "<plate>"
runtime\python311\python.exe experiments\hybrid_seed\run_hybrid.py --plate "<plate>" --seeds 400
runtime\python311\python.exe experiments\hybrid_seed\check_roundtrip.py --plate "<plate>"
runtime\python311\python.exe experiments\hybrid_seed\render_overlay.py --plate "<plate>"
```

`--flat-geom` on `run_hybrid.py` gives every tracker the same patch and search size. That is
the control for the per-seed-class geometry, which is **on by default and untested** — worth
one extra run so you find out whether it helps rather than assuming it does.

## Files

| File | What it is |
|---|---|
| `RunHybridExperiment.bat` | the whole suite; the only thing you need to run |
| `check_licence.py` | step 1 |
| `probes.py` | steps 2 and 3 |
| `run_hybrid.py` | the hybrid itself: seed, inject, track, export, optional re-acquisition |
| `check_roundtrip.py` | step 6 |
| `render_overlay.py` | overlay videos |
| `plate_io.py` | movie / image-sequence handling, pixel↔SynthEyes coordinate conversion |
| `sylab.py` | run a `.szl` against a live SynthEyes; ~1s iteration when debugging |
| `bench_run.py` | times `tk.Run()` at several patch sizes |
| `szl/` | the diagnostics, each answering one question |
| `FINDINGS.md` | what has actually been measured, and what has not |
