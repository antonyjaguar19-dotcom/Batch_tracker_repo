# Hybrid tracking: TAPNext seeds and replants, Blender tracks

Question asked: the bot already seeds with TAPNext and re-acquires after occlusion. What
if the per-frame *measurement* in between were done by Blender's movie-clip tracker
instead of the bot's own refine chain?

Answer: **the mechanism works end to end, headless, and the numbers are in.** On exact
synthetic ground truth the hybrid is slightly *better* than the shipping bot; on the one
usable real-plate reference it is a tie. Nothing in `app/` was modified.

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

- **The hybrid is additive, not a replacement.** It needs a full bot TAPNext run first, for
  seeds and for the guide. On SH004 that is 580s of bot plus 18s of Blender — the 18s is
  not a speed win over the 580s, it is 18s spent on top. A Blender pass that *replaced* the
  two refine stages would be the interesting version, and is not what was measured.
- **Occlusion accuracy.** Replant demonstrably extends spans; nothing here shows the
  resumed segments land on the right feature.
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

REM real plate, from frames
runtime\python311\python.exe bench\run_bench.py --shot experiments\blender_track\out\SH004 --tag guide
runtime\python311\python.exe experiments\blender_track\run_blender_hybrid.py ^
    --plate experiments\blender_track\out\SH004\plate --name SH004 ^
    --reuse-tapnext experiments\blender_track\out\SH004\runs\guide\SH004__tapnext.txt --tag repl
runtime\python311\python.exe tools\eval_refs.py refs\SH004_lk ^
    --bot experiments\blender_track\out\SH004__repl__blender.txt ^
    --baseline experiments\blender_track\out\SH004\runs\guide\SH004__tapnext.txt
```

Every run must print `seed round-trip : max <n>px PASS`. If it says FAIL, stop — the
trackers are not on the features they were given and no other number means anything.

## Files

| File | What it is |
|---|---|
| `run_blender_hybrid.py` | the e2e: bot seeder/guide -> Blender -> 3DE export, with the round-trip check |
| `bl_track.py` | runs inside Blender; seeds, tracks, replants. Imports nothing from this repo |
| `blio.py` | plate access, pixel<->clip coordinates, the Blender subprocess call |
| `render_overlay.py` | burns hybrid + guide over the plate; in-gap points drawn hollow |
| `README.md` | how to run it |
