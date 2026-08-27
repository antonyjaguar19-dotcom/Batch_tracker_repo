# Can a tracker match a hand track 100 %? — measured on SH006

Asked: a deep comparison of CoTracker, DINO-Tracker and hybrids, and a solid answer on
whether 100 % agreement with the artist's own tracks is achievable or a waste of time.

Every number here came off this machine, on SH006, against `track3 manual tracked.txt` and
`reacquretracke_manual.txt`. Nothing is quoted from a paper.

**The short answer is in three parts, and only one of them is bad news.**

1. 100 % agreement in ABSOLUTE POSITION is impossible, and not because of the tracker.
2. Holding the pattern between occlusions is ALREADY better than a hand track.
3. Re-acquiring after an occlusion is the real problem, and on this plate it is **not a
   model problem at all** — neither CoTracker nor DINO can fix it, because the blocker is
   the footage.

---

## 1. What "100 %" can mean: the artist against themselves

The artist tracked ONE feature by hand TWICE — `reacquretracke_manual.txt` and `track3`
Track.001, 46 shared frames, same runs, same feature.

| | |
|---|---|
| frames identical | **0 of 46** |
| difference, p50 / max | **2.13 px / 2.57 px** |
| constant offset between the attempts | 0.43 px |
| scatter about that offset, p50 / max | **1.78 px / 3.00 px** |
| frame-to-frame MOTION disagreement, p50 / max | **0.05 px / 1.39 px** |

So a hand track is not reproducible in absolute position **by the person who made it**. Two
careful attempts wander up to 3 px apart.

What IS reproducible is the motion — to a twentieth of a pixel. And motion is what a solve
consumes: a constant offset is a different but equally valid point on the same feature and
costs a solve nothing.

**Chasing absolute agreement below ~2 px is chasing the reference's own noise.** That part of
the goal is a waste of time, and it is worth knowing before spending a week on it.

---

## 2. Holding the pattern: already past hand-track quality

Every method run over the frames before the first occlusion, seeded on the artist's own point:

| method | on the feature | absolute p50 | motion p50 |
|---|---|---|---|
| CoTracker raw | 14/14 | 7.7 px | 3.65 px |
| **pattern only** (seed patch, previous-frame prior, warped) | 14/14 | **0.1 px** | **0.12 px** |
| **CoTracker + pattern pin** | 14/14 | **0.1 px** | **0.12 px** |
| *artist vs artist* | *46/46* | *2.13 px* | *0.05 px* |

Two things fall out.

**Raw CoTracker is not a matchmove tracker** — 7.7 px absolute, 3.65 px of motion error. It
knows *which* feature; it does not know where it is to a pixel. That is why the pattern pin
exists and why it matters more than the choice of model.

**Pinned to the artist's pattern, the tool reproduces their hand track to 0.1 px.** That is
twenty times tighter than the artist reproduces themselves. On this axis the answer is not
"100 % is impossible", it is "we are already past it, and the reference is the limit".

Note that pattern-only and CoTracker+pin are *identical* here to two decimal places. Between
occlusions the guide contributes nothing — the pattern is doing all the work.

---

## 3. Re-acquisition: the real problem, and it is the plate

Track.001 has two occlusions: hidden f15–24 (back at f25), hidden f33–40 (back at f41).

### The pattern is findable — that is not the issue

Scored at the artist's own positions after the first occlusion:

| frame | seed patch (f1) | last-good patch (f14) |
|---|---|---|
| f25 | 0.948 | 0.949 |
| f26 | 0.976 | 0.981 |
| f28 | 0.962 | 0.984 |
| f31 | 0.938 | 0.974 |

And a 56 px sweep around the guide's prediction lands **1–4 px from the artist's track on
every one of f25–f32**. So the machinery works.

### But the pattern is just as findable where the feature is NOT

Same sweep, same patch, on the frames the artist deliberately left EMPTY because the feature
is behind something:

```
f15  0.913   f16  0.912   f17  0.904   f18  0.923   f19  0.957
f20  0.931   f21  0.938   f22  0.899   f23  0.922   f24  0.865
```

**0.86 to 0.96 on frames where the feature is definitively hidden** — higher than f32, where
it is genuinely visible (0.92). A sweep with any threshold resumes at **f16, on the
occluder**, and every frame after that is wrong.

This is a property of the FOOTAGE. The texture repeats, so within 56 px of the right answer
there is always a convincing wrong one. No score threshold separates these two tables, and
no tracker model changes that: the pattern is what is being matched, and it matches.

### CoTracker's visibility head gets the first half right and the second half wrong

| frames | visibility says | truth | |
|---|---|---|---|
| f16–24 | **not visible** | hidden | **correct** |
| f25–45 | **not visible** | plainly back, matching 0.98 | **wrong** |

It agrees with the artist on 18 of 32 frames (56 %). It can tell you when to **stop**. It
cannot tell you when to **resume** — it never flips back.

### Motion consistency is a weak discriminator, not a clean one

How far the sweep's answer jumps frame to frame:

```
during the occlusion   27-36 px  (one jump of 110)
after the return       9-17 px
the return frame f25   82 px
```

Real separation, heavy overlap. Usable as evidence, not as a gate.

### How far the feature travels while hidden

99 px between f14 and f25. So the search cannot simply be widened: a sweep large enough to
reach the feature is large enough to contain many convincing wrong answers, and this plate
supplies them.

### The guide is violently sensitive to where it is re-queried from

Distance between the guide's prediction and the truth at f25, by the frame the guide was
re-queried at:

| re-queried at | f25 | f41 |
|---|---|---|
| f14 | **44 px** | 60 px |
| f12 | 89 px | 70 px |
| f10 | 24 px | 134 px |

A two-frame difference in where the track died moves the prediction from 24 px to 89 px. The
current loop re-queries at the last good frame and has no way to know that was a good choice.

---

## 4. What this says about CoTracker vs DINO

**It says the choice barely matters for this failure.** The blocker is that the artist's
pattern matches the occluder. That is true whichever model supplies the neighbourhood.

Where the two were measured head to head — same seeds, same 20-frame window, 16 of the
artist's own points:

| | on the feature | median p50 |
|---|---|---|
| CoTracker | 277/306 (91 %) | 5.7 px |
| DINO-Tracker (trained on this plate) | 240/273 (88 %) | 6.0 px |

A tie, and both are sitting at **one model pixel** — DINO runs at 854×476, CoTracker at 768
wide, so one model pixel is ~4.5 plate pixels either way. Neither is limited by how good the
model is. Both are fixed by the same pattern pin.

Across full spans, raw CoTracker gives 1644/1839 (89 %) between occlusions, median 7.2 px —
and 71/109 (65 %) after gaps, which is entirely down to *which* gap: Track.008's 38-frame gap
came back 71/76, Track.001's occluder came back **0/33**.

---

## 5. Verdict

**Not a waste of time:**

- *Holding the pattern.* Already at 0.1 px against a reference that is only self-consistent
  to 2 px. Done, and provably so.
- *Detecting that a track has left the feature.* The slide detector and the hold check work,
  and are measured elsewhere in `FINDINGS.md`.

**A waste of time:**

- *Chasing absolute agreement below ~2 px.* The reference is not that good.
- *Choosing between CoTracker and DINO to fix re-acquisition.* They tie on tracking, and the
  thing that fails is neither of them. DINO additionally costs hours per shot with no reuse.
- *Widening the search, or tuning the match threshold.* Both tables above sit in the same
  score range. There is no threshold between them.

**The one thing that would actually close it**, and it is not a tracker at all: knowing
**where the occluder is**. If the shot is analysed first — the pipeline next door already runs
Qwen2.5-VL for movers and occluders and SAM3 for per-frame mattes — then "the feature is
behind that" is a fact rather than an inference from a correlation score. The rule becomes
trivial and needs no threshold: do not resume while the feature's predicted position is inside
an occluder matte; resume on the first frame it is not.

That was offered as a direction earlier and passed over for coverage-and-solve work. On this
evidence it is the only remaining route to 100 % re-acquisition on plates like this one.

## What this does not settle

- **One plate, one artist, four occlusion events.** SH006 has repeating texture; a plate with
  distinctive features may separate on appearance perfectly well.
- The artist's self-consistency is measured from **one** pair of hand tracks over 46 frames.
- DINO's re-acquisition is still unmeasured — training on f1–65 was running when this was
  written. It cannot change the conclusion about the pattern matching the occluder, but it
  could change how far the guide carries the point across it.
- Blender's own tracker was not run in this matrix; it is measured against the same references
  in `FINDINGS.md` and is the weakest of the three at crossing anything.
