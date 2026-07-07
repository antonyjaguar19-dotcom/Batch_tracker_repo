# Decision Log

Chronological record of notable engineering decisions and why they were made.
Newest first. Dates are absolute.

---

## 2026-07-07

### Replaced CoTracker3 with TAPNext++ as the secondary (GPU) tracking backend
- **What**: removed all CoTracker3 components (`app/cotracker_engine.py`, vendored
  `pipeline/co-tracker-main/`, `scaled_offline.pth`, `*__cotracker3_bidir.txt`) and replaced them
  with **TAPNext++** (google-deepmind/tapnet) in new `app/tapnext_engine.py`. SynthEyes stays the
  **primary** backend; TAPNext++ is the secondary GPU fallback (`track_backend` value `cotracker`
  → `tapnext`; legacy value auto-migrates). Output files now `<shot>__tapnext.txt`. `TapNextEngine`
  keeps the exact `track_grid`/`track_queries` interface, so `tracker_core.py` seeding / 4-pass merge /
  SAM3 gating / filtering / 3DE export is unchanged.
- **Why**: **CoTracker3 is CC-BY-NC 4.0 (non-commercial)** — a blocker for paid VFX work. TAPNext++
  is **Apache-2.0** (commercial OK), ships a native PyTorch checkpoint (`tapnextpp_ckpt.pt`), and is
  stronger for matchmove (40× longer stable tracks, tracks through occlusion, re-detection).
- **How / gotchas**: TAPNext++ is a **fixed 256×256** model and **causal/streaming** (next-token).
  The wrapper absorbs both: resize frames to 256² in and rescale predicted coords back to the caller's
  frame space; seed queries at block frame 0 then feed one frame at a time carrying the recurrent
  `state`. `tracker_core` already reverses frames for BWD/MID passes, so a single forward stream covers
  every pass. TAPNext has no grid mode → `track_grid` synthesizes a uniform grid gated by the SAM3
  inclusion mask. Checkpoint resolver: `pipeline/tapnext-main/checkpoints/` → `<repo>/checkpoints/` →
  env `BTR_TAPNEXT_CKPT`. **Open verify item**: 256² on 4K plates — confirm sub-pixel precision is
  acceptable on the GPU box, and confirm video normalization ([0,1] RGB per the torch demo).

---

## 2026-07-06

### SynthEyes 2026 automation: SyPy3 API is broken → drive it via Win32 UI + Sizzle
- **What**: proved a fully-automated, background-capable SynthEyes 2026 tracking pipeline that
  does NOT use the SyPy3 return-value API (it is broken on build 2026.2.4679 — see below):
  1. **SyPy3** only for the calls that work (launch, `NewSceneAndShot` via IFL, `SetRoom`).
  2. **Win32 `PostMessage`** clicks on SynthEyes' `Butt` child windows (found by `GetWindowText`)
     to run **Blips all frames** + **Peel all** — works even with the window at `HWND_BOTTOM`
     (backgrounded), and is resize-robust (buttons located by name, not pixels).
  3. **Export** via a Sizzle `//SIZZLET` tool script run with `RunScriptFile` using
     **`openout(path)` + `printf`** to write the 3D Equalizer 2D-track `.txt` DIRECTLY (the
     export menu is custom-drawn and unreachable via the API).
  4. **SAM3 gating in Python** (post-export): sample each mask at each tracker point, drop points
     on movers. Chosen over SynthEyes roto automation (fragile file-dialog + polarity guessing).
- **Why**: SyPy3's `Sync()`/`RecvSzl()` return-value protocol desyncs against 2026.2.4679 —
  attribute reads, `ActionID`, `Actions()` enumeration, `Call()` returns all fail intermittently
  ("list index out of range"). Only plain `Run()`-based calls work. This blocked every normal
  automation path (blip/peel dispatch, export, tracker read-back), so we route around it entirely.
  Full detail + reference scripts in the session memory `syntheyes-2026-headless-status`.
- **Note**: test build was **SynthEyes Pro DEMO 2026** — the demo caps real track length at
  ~10 frames (uniform, artificial; the "AUTO" full pipeline incl. coalesce gives the identical
  cap). This is a **demo limitation, not the pipeline** — re-verify on the full license.
  Reference impl lives in scratchpad (`se_final.py`, `sam3_filter.py`); not yet wired into
  `app/worker_track` as a `track_backend="syntheyes_ui"` path (TODO, post-license).

### SAM3 mask filenames renamed `####_alpha.png` → `mask_####.png` (SynthEyes-readable)
- **What**: `pipeline/sam3/sam3_runner.py` now writes `mask_<stem>.png` / `preview_<stem>.png`
  (frame number **trailing**) instead of `<stem>_alpha.png` / `<stem>_preview.png`.
- **Why**: SynthEyes detects a numbered image sequence only when the varying digits are the
  trailing token (`name_####`, e.g. its own `DCP_####.JPG`); the old digits-first form couldn't
  be loaded as a sequence via `+Alpha`. Polarity was already correct (white=track, black=exclude,
  matching SynthEyes) — no invert needed.
- **Note**: safe because downstream readers use **sorted order**, not the filename
  (`tracker_core.get_global_mask_idx` indexes a `files.sort()`ed list; the reuse check is an
  existence glob). Both naming schemes sort identically by zero-padded frame number.

### SAM3 `mask_dilation_px` — Mask ML-style edge dilation
- **What**: new `RunnerConfig.mask_dilation_px` (default `0` = off). When >0, the final merged
  mask's EXCLUDE (black/mover) region is grown by N px via `cv2.erode(alpha_keep)` before writing.
- **Why**: mirrors SynthEyes Mask ML's "Mask Dilation" — keeps auto-trackers off soft edges
  (hair, motion-blur fringes). SAM edges are tight; ~8–15 px matches SynthEyes tracking behavior.
  Independent of the existing `motion_dilate_px` (which dilates only the CV-motion-backstop movers).

## 2026-06-23

### VRAM/RAM-safe CoTracker — chained chunking + OOM retry + streamed decode
- **What**: long and/or high-res shots no longer OOM. New levers in `app/tracker_core.py`:
  - **Temporal chunking** — a shot is tracked in N overlapping windows whose track IDs are
    **chained across the seams** (overlap-surviving points become queries for the next chunk),
    so a feature crossing a seam stays **one continuous track**. FWD-chain + BWD-chain.
  - **Auto chunk count** — sized from free VRAM (`torch.cuda.mem_get_info`) with a factor that
    self-calibrates off the measured CUDA peak after the first chunk; UI override wins
    (`AppState.track_chunks` → `RunnerConfig.chunks`; `0` = auto).
  - **OOM retry** — on CUDA OOM a chunk downscales (×0.7, floor 0.25), rescales its carried
    queries, and retries; export rescales coords back to original resolution.
  - **Streamed decode** — when a full-clip decode would exceed `host_ram_frac` of available RAM,
    frames are decoded **per window** (`video_io.FrameSource`/`read_window_bgr_scaled`) so host
    RAM is bounded too; small clips keep the fast full-decode path.
- **Why**: CoTracker3 offline runs the whole clip in one forward per pass → VRAM ∝ frames×res,
  and the whole clip was also loaded into host RAM. Both walls hit on long 4K. Chosen over a
  naive split because naive chunking fragments tracks at seams (weaker 3DE solve).
- **Note**: single-chunk shots keep the exact prior 4-pass behavior (no regression). Chunked
  shots use FWD/BWD chains only (mid passes dropped — chaining already preserves long tracks).
  Calibration + the analytic VRAM estimate are approximate; the OOM-retry net covers misses.

### Repo reorganized: entries in `app/`, stages under `pipeline/`
- **What**: `app.py` + `app_nicegui.py` moved into `app/`. The three AI stages moved to
  `pipeline/qwen`, `pipeline/llama/core`, `pipeline/sam3` (flattened — `sam3_mask_tool`
  gone); vendored CoTracker moved to `pipeline/co-tracker-main` and `thirdparty/`
  removed. Root now holds only `launch_nicegui.bat`, `requirements.txt`, and the docs.
- **Why**: a clear, conventional layout — code under `app/`, the swappable AI stages
  grouped under `pipeline/`.
- **How it stays working**: the backend loaders rglob by filename, so dir renames are
  tolerated. Fixed the few path-dependent spots: `app/app.py` `_HERE` = `parents[1]`
  (repo root, since it now lives in `app/`) and `DEFAULT_SAM3_WEIGHTS` →
  `pipeline/sam3/weights`; Qwen model base → the run script's own `models/` dir;
  `cotracker_engine` searches `pipeline/co-tracker-main` first; `app_nicegui.py` loads
  the backend from `app/app.py`; `launch_nicegui.bat` runs `app\app_nicegui.py`. The
  `core/` and `app/` package names are load-bearing and were left unchanged. Verified by
  importing the backend: all loaders + `core.*` imports + CoTracker path resolve.

### Stripped to NiceGUI-only; deleted unused files
- **What**: removed everything not needed by the NiceGUI tool — the Gradio launcher,
  the standalone solo-stage UIs (`app/ui*.py`, the PySide reasoner + SAM apps) and their
  per-stage `.bat`/README/requirements, the dead `core/llama_backend.py`, and the unused
  vendored `RAFT-master` + `tapnet-main`.
- **Why**: one front-end, one launcher; less to maintain and ship. The Gradio UI code
  still lives inside `app/app.py` but no launcher ships for it.
- **Note**: deletions are permanent; confirmed the keep/remove list before executing.
  `app/syntheyes_runner.py` (alt SynthEyes engine) is also gone; the tracker loader falls
  back to `tracker_core.py` (CoTracker), so nothing broke.

### CoTracker checkpoint fetched
- **What**: downloaded `scaled_offline.pth` to `co-tracker-main/checkpoints/` from the
  public `facebook/cotracker3` HF repo.
- **Why**: it was missing — tracking errored with "Missing checkpoint scaled_offline.pth".

### Reuse-existing-masks prompt
- **What**: at Generate-masks, if selected shots already have masks in OUT, NiceGUI asks
  Reuse / Regenerate / Cancel. Reuse writes a filtered `*_run.json` guide so SAM3 only
  generates the missing shots; `state.reuse_existing_masks` drives it.
- **Why**: avoid silently re-masking (slow) or silently skipping; let the artist decide.

### 0-track shots now explain themselves
- **What**: a shot exporting 0 tracks logs a `ZERO …` line naming the stage that killed
  them (seeding / post-filter / mask gating / too-short) with per-stage counts.
- **Why**: "0 tracks" with no reason was undebuggable.

### Frame range restored on Scan
- **What**: scanning an Output with a previous guide now restores per-shot
  `frame_start`/`frame_end` (alongside the include/exclude prompts it already restored),
  reading the newest of `mask_guidance.json`/`overdrive_guide.json`.
- **Why**: reused masks were produced for a specific range; tracking must use the same.

---

## 2026-06-22

### Switched the front-end from Gradio to NiceGUI
- **What**: New `app_nicegui.py` (primary, `launch_nicegui.bat`); Gradio `app.py` kept
  as a fallback. Entire backend reused unchanged.
- **Why**: Gradio's `Dataframe` made shot selection unreliable — bool cells did not
  render as usable checkboxes, and `.select` on an interactive dataframe was flaky
  (cell-edit vs row-select; the 1 s refresh timer could cancel clicks). NiceGUI's
  `ui.table` provides native per-row checkboxes, selection highlight, and row-click.
- **Note**: `app_nicegui.py` loads `app.py` by file path under its own `sys.modules`
  name. Reasons: the repo has an `app/` package that shadows `import app`, and the
  embeddable Python `._pth` isolation does not auto-add the script dir to `sys.path`.

### SAM3 masking now honors the per-shot frame range
- **What**: `frame_start/frame_end` flow `worker_mask` → guide JSON →
  `sam3_runner.normalize_shots` → frame slice in `run_sam3_batch`. The tracker's mask
  gating detects a sub-range mask set (fewer masks than the full clip) and maps masks
  1:1 to the tracked frames; full-clip masks still map proportionally.
- **Why**: Bug report — SAM3 was masking past the user's frame range. Previously the
  range only affected tracking; SAM3 processed every frame.

### CV motion backstop made a user toggle
- **What**: "CV motion backstop" switch in the UI → `AppState.motion_backstop` →
  `SamConfig.motion_backstop`; also `BTR_MOTION_BACKSTOP=0`.
- **Why**: The CV layer masks anything moving independently of the camera, ignoring the
  Exclude list. It correctly leaves the first frame alone (no previous frame for flow),
  but on later frames it re-masked subjects the user had deliberately removed from
  Exclude. Users need to turn it off for full manual control.

### Shot-edit prompts are authoritative
- **What**: `worker_mask` now writes Include/Exclude verbatim (empty list clears the
  old value) instead of only adding when non-empty.
- **Why**: Removing a term in the UI did nothing — the old guarded code only ever
  appended, never cleared.

### VRAM staging between stages
- **What**: Qwen freed after Analyze; the Ollama LLM kept resident through Analyze and
  unloaded (`keep_alive=0`) at Mask; CUDA cache cleared before SAM3 and before/after
  CoTracker. Helpers `_free_vram` / `_free_ollama`.
- **Why**: Fit each heavy model on a 16 GB RTX A4000 without OOM; keep the LLM available
  only while it is needed (through Analyze), as requested.

### Four "mover" backstops for camera solves
- **What**, in order of authority:
  1. VLM `moving_things`; 2. deterministic `dynamic_subjects` (animals/people/vehicles/
  fx) scanned from the **raw** detected-things list; 3. scene-text heuristic
  (`moving_foreground_terms`); 4. CV optical-flow residual after global-motion
  subtraction (masks pixels moving independently of the camera, no name needed).
- **Why**: A running dog was tracked into the camera solve. Root cause: it was detected
  but not in the scene prose, so `filter_things_by_scene` dropped it before the mover
  heuristic. Layers 2–4 each remove a different dependency (prose mention, motion
  judgement, detection at all).

### Qwen2.5-VL: added matchmove analysis fields + GUI display
- **What**: One extra structured-JSON prompt per shot extracts `moving_things`,
  `bad_track_regions`, `foreground_occluders`, `quality_flags`, `depth_layers`,
  `parallax`. Carried through the parser/bridge; camera excludes use them; shown in the
  UI (Quality column + per-shot analysis panel).
- **Why**: The VLM was already loaded; these signals directly improve mask prompts and
  give the artist pre-flight QC. Also fixed the bridge so the guide lists every shot
  even with no client brief.

### Qwen2.5-VL frame sampling fix
- **What**: Sample 6–8 frames spread across the **whole clip** (not the first N), at
  768 px. Frame count plateaus by design.
- **Why**: The old code sampled the first ~6 frames; higher FPS made coverage *worse*,
  so a subject appearing later (e.g. a dog) was never seen. Too many frames also dilute
  the VLM, so the count is capped.

### Upgraded Qwen 3B → 7B, loaded int4
- **What**: Model resolver prefers 7B (AWQ or full); full 7B loaded via bitsandbytes
  NF4 (~5.5 GB VRAM). `app.py` requests int4.
- **Why**: 3B was the weakest link — missed/hallucinated objects, which propagate to
  masks and tracks. 7B int4 fits the GPU with headroom.

### SAM3 weights sourced from `facebook/sam3`
- **What**: Downloaded `sam3.pt` (gated HF repo) to `SAM3/weights/`; path wired via
  `BTR_SAM3_WEIGHTS`.
- **Why**: Weights were missing; not auto-hosted by Ultralytics. Verified the
  checkpoint has the expected `detector.`/`tracker.` key structure before use.

### Portable, project-local Python runtime
- **What**: Embeddable Python 3.11.9 at `runtime/python311/` with tkinter restored;
  all pip installs/caches/temp pinned inside `runtime/`. GPU PyTorch is the cu121 build.
- **Why**: Hard constraint — keep every install/download inside the project folder and
  off the C: drive. No system Python existed.

### Default shot selection is OFF
- **What**: New shots start unticked.
- **Why**: All-selected-by-default meant a run touched every shot; users wanted to pick
  one (or a few) deliberately.
