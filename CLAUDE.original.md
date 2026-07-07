# CLAUDE.md

Guide for Claude Code (claude.ai/code) on this repo. Human-facing docs: `README.md` (how to run, requirements) + `DECISIONS.md` (why things are the way they are, newest-first).

## What this is

Windows/CUDA VFX matchmove pipeline tool ("Batch Tracker"). Chains four AI/CV stages, raw shot footage → 2D point tracks importable into 3D Equalizer:

1. **Qwen** (`pipeline/qwen/`) — local offline Qwen2.5-VL vision-language model, describes each shot (`scene_elements`, `camera_movement`, `things`) into `shot_descriptions.json`. Model resolver (`pipeline/qwen/run_qwen2_shot_describer.py:_resolve_model_dir`) prefers `Qwen2.5-VL-7B-Instruct-AWQ` → `Qwen2.5-VL-7B-Instruct` → `Qwen2.5-VL-3B-Instruct` (override `QWEN2_MODEL_DIR`); weights sit under `pipeline/qwen/models/` (base = the run script's own dir, location-independent). Full 7B loaded **int4** via bitsandbytes NF4 (~5.5 GB VRAM). Extra structured-JSON pass per shot (`_analyze_for_matchmove`) extracts matchmove signals: `moving_things`, `bad_track_regions`, `foreground_occluders`, `quality_flags`, `depth_layers`, `parallax`. Frames sampled 6–8 spread across **whole clip** at 768px (not first N) — `_sample_frames`.
2. **LLaMA** (`pipeline/llama/`) — deterministic heuristics (`core/heuristics.py`) decide camera/object track strategy + mask prompts; Ollama-hosted `llama3.1:8b` (`core/ollama_backend.py`) only consulted for ambiguous client intent. Output: "masking guide" JSON (`mask_guidance.json`), schema `shots[].tasks[]` (task_id `camera`/`object`/`other`, `track_mode`, `mask_includes`/`mask_excludes`, `sam3_include_prompt`/`sam3_exclude_prompt`).
3. **SAM3** (`pipeline/sam3/`) — `sam3_runner.py` reads masking guide JSON + `sam3.pt` weights, runs Ultralytics SAM3, makes per-shot alpha masks (`<OUT>/<SHOT>/masks/*.png`, white=keep/track, black=ignore) + previews.
4. **CoTracker3 batch tracker** (`app/tracker_core.py` + `app/cotracker_engine.py`) — runs vendored `pipeline/co-tracker-main` CoTracker3 model per shot, 4-pass strategy (forward, backward, mid-forward, mid-backward), gates/filters tracks vs SAM3 masks, exports 3D Equalizer-format track `.txt` via `app/export_3de.py`.

## Front-end + backend

Single front-end now: **NiceGUI**. Both entry files live in `app/`:

- **`app/app_nicegui.py`** (UI entry) — NiceGUI app (`launch_nicegui.bat`, http://localhost:8080). Computes `_APP_DIR` = its own dir, `_HERE` = repo root (parent of `app/`), inserts repo root on `sys.path`, then loads the backend `app/app.py` **by file path** under `sys.modules["btr_backend"]` (NOT `import app` — the `app/` package shadows it; also embeddable Python `._pth` doesn't auto-add the script dir). Uses `ui.table(selection=...)` native checkboxes for shot selection.
- **`app/app.py`** (shared backend) — holds workers, model loaders, `AppState`/`ShotData` dataclasses, the `importlib` stage loaders, AND a still-present Gradio UI (no launcher ships for it; it only builds under `__main__`). `_HERE = Path(__file__).resolve().parents[1]` = repo root (app.py lives in `app/`). Does NOT statically import other stages; dynamically locates + `importlib`-loads them via **recursive `rglob`** from candidate roots (`_HERE`, cwd, `_LEGACY_ROOT`): `core/io_parsers.py` (+ `core/ollama_backend.py`/`core/bridge.py`), `app/tracker_core.py`, `pipeline/sam3/sam3_runner.py`, `pipeline/qwen/run_qwen2_shot_describer.py` (see `_bootstrap_paths`, `_load_tracker_direct`, `_load_sam3_direct`, `_load_qwen_robustly` near top). Loaders are rename/move tolerant (rglob by filename) but **fail silent** into `..._IMPORT_ERROR` globals — they don't raise at import time, so check those globals if a stage goes missing.

Everything in `app/app.py` (`worker_analyze`/`worker_mask`/`worker_track`, `_free_vram`, `_free_ollama`, `STOP_EVENT`, loaders) is the one backend — edit once.

## Path / loader rules (important)

- **`core/` and `app/` names are load-bearing.** `app/app.py` imports `from app.tracker_core …` (package on repo root) and the LLaMA stage is imported as `from core.X` (bootstrap rglob finds `core/io_parsers.py`, adds its parent — `pipeline/llama` — to `sys.path`). `bridge.py` uses relative `.ollama_backend`/`.heuristics`. Don't rename `core/` or `app/`.
- Subsystem loaders **rglob by filename**, so the `pipeline/*` dirs can be renamed/moved as long as the filenames + `core/` subdir survive. Three hardcoded paths must track moves: `app/app.py` `DEFAULT_SAM3_WEIGHTS` (= `_HERE/"pipeline"/"sam3"/"weights"/"sam3.pt"`), the Qwen model base (`<run script dir>/models`), and `launch_nicegui.bat` `BTR_SAM3_WEIGHTS`.
- `app/cotracker_engine.py:_add_cotracker_to_path` searches, in order, `pipeline/co-tracker-main`, `pipeline/co-tracker`, then `thirdparty/co-tracker-main`, `thirdparty/co-tracker`. `tool_root` = repo root (computed from `tracker_core.py` location). Checkpoint resolved at `<repo>/checkpoints` (the cotracker repo's own `checkpoints/`).
- **Legacy fallback**: project once lived at `D:\Jefrin\BTr\batch_tracker_v001_starter`. `_LEGACY_ROOT` kept only as a last-resort rglob root in `app/app.py`; don't "fix"/delete it.

## Running things

Portable, project-local **embeddable Python 3.11.9** at `runtime/python311/` (`._pth` isolation: no auto sys.path for script dir / `-m`, ignores PYTHONPATH). HARD CONSTRAINT: keep every install/download inside the project folder, off C: drive — pip cache/temp pinned to `runtime/` via env vars set in `launch_nicegui.bat`. GPU PyTorch is the cu121 build. Frozen deps in root `requirements.txt` (incl. `bitsandbytes==0.49.2`).

- Ollama = separate local server, not a pip dep — `OllamaConfig.base_url` defaults `http://localhost:11434`, model `llama3.1:8b`, override `BTR_OLLAMA_URL`.
- SAM3 needs `ultralytics` + `sam3.pt` weights at `pipeline/sam3/weights/sam3.pt` (or `BTR_SAM3_WEIGHTS`).
- Qwen needs `torch`, `transformers` (Qwen2.5-VL), `bitsandbytes` (int4), `pillow`, `opencv-python`; weights under `pipeline/qwen/models/<model>` — `qwen2_shot_describer_core.py` loads `local_files_only=True`, errors instead of download.
- CoTracker needs `scaled_offline.pth` under `pipeline/co-tracker-main/checkpoints/` (or `<repo_root>/checkpoints/`).

No test suite, linter, or build step — verify manually (run the NiceGUI UI, check output files).

Launcher: **`launch_nicegui.bat`** (root) → runs `app\app_nicegui.py` on `runtime/python311`; portable `%~dp0` path; sets `BTR_SAM3_WEIGHTS` + cache/temp env inside the repo. It's the only launcher.

## Key data contracts between stages

- **Qwen → LLaMA**: `shot_descriptions.json` w/ `shots[].{name, scene_elements, camera_movement, things}` + matchmove signals `{moving_things, bad_track_regions, foreground_occluders, quality_flags, depth_layers, parallax}` — loaded by `core/io_parsers.load_qwen2_v1_scene_cam_things` (carries all extra fields), keyed by normalized shot name (`core/io_parsers._norm_shot` strips spaces/dashes/underscores, case-insensitive).
- **LLaMA → SAM3**: masking guide JSON, schema `"2.1-dual-tasks"` — `shots[].tasks[]` authoritative; top-level `mask_includes`/`mask_excludes`/`track_mode` per shot kept only for backward compat (mirror `tasks[0]`). `pipeline/sam3/sam3_runner.normalize_shots` accepts both schemas, explodes multi-task shots into one normalized entry per task (split by `mask_subdir`/`preview_subdir`, e.g. `masks_camera`, `masks_object`).
- **SAM3 → CoTracker**: masks live at `<OUT>/<SHOT_NAME>/<mask_subdir>/*.png` (default `masks`), white=keep/track, black=ignore. `RunnerConfig.mask_mode` (`"inside"`/`"outside"`) + `mask_polarity` (`"auto"`/`"white"`/`"black"`) control how `app/tracker_core.py` reads them — `"auto"` infers polarity via majority-white-pixel heuristic per mask.
- **Tracking targets vocabulary** (`core/heuristics.py`): `track_mode` ∈ `track_inside_mask` | `track_outside_mask` | `no_mask_needed`. Camera tasks default `track_outside_mask` excluding `MOVING_TERMS` (people, vehicles, fluids, reflections etc, capped 4 via `cap_excludes`); object tasks always single-subject `track_inside_mask` (see `pick_object_subject`). `BAD_MASK_TERMS` / `ENV_BLACKLIST` filter verbs + env nouns that'd confuse SAM3 prompts.
- **Four "mover" backstops** for camera-solve excludes, in authority order: (1) VLM `moving_things`; (2) deterministic `dynamic_subjects(things)` in `heuristics.py` — scans the **raw** detected-things list (animals/people/vehicles/fx), catches subjects not in scene prose; (3) scene-text heuristic (`moving_foreground_terms`); (4) CV optical-flow residual after global-motion subtraction (`pipeline/sam3/sam3_runner.compute_motion_keep_mask`, Farneback) — masks pixels moving independently of camera, no name needed. Layer 4 is a UI toggle (NiceGUI "CV motion backstop" → `AppState.motion_backstop` → `SamConfig.motion_backstop`) / env `BTR_MOTION_BACKSTOP=0`. `core/bridge._build_camera_task` merges layers 1–3; layer 4 applied at SAM3 per-frame loop via `np.minimum`.
- **Per-shot frame range** (`frame_start`/`frame_end`, 1-based, `0`=full): flows `worker_mask` → guide JSON → `sam3_runner.normalize_shots` → frame slice in `run_sam3_batch`, AND into `RunnerConfig.frame_start/frame_end` for the tracker. Honored by **both** SAM3 + CoTracker. Tracker mask gating (`get_global_mask_idx`) detects a sub-range mask set (`M < orig_total`) → maps masks 1:1; full-clip masks map proportionally. Exported frame numbers stay aligned to the original shot (`t+1+_frame_offset`). On Scan, the newest previous guide (`mask_guidance.json`/`overdrive_guide.json`) restores include/exclude prompts **and** frame range.
- **CoTracker → 3DE**: output `.txt` written by `app/export_3de.py:write_tracks_txt`, classic 3D Equalizer 2D-track ASCII format (`<N tracks>` header, then per track: name / color-id `0` / point-count / lines `frame x y`). Files named `<shot>__cotracker3_bidir.txt` or, tagged multi-task runs, `<shot>__<task_id>__cotracker3_bidir.txt`.

## CoTracker tracker internals worth knowing before touching `app/tracker_core.py`

- **Two tracking paths, chosen per shot by `_decide_chunks(T,Ws,Hs)`** (free VRAM via `torch.cuda.mem_get_info`, factor self-calibrated from `torch.cuda.max_memory_allocated` after the first chunk; `RunnerConfig.chunks` override, `0`=auto). `n<=1` → the original single-block path below; `n>1` → **chunked** path (`_track_chunked`).
- **Single-block path** = **4 passes**: full-forward, full-backward (reversed frame order), mid-forward (midpoint to end), mid-backward (midpoint back to start) — merged per-track. Track IDs prefixed `FWD_`/`BWD_`/`MID_F_`/`MID_B_`.
- **Chunked path** (`_chain_core`) = FWD-chain + BWD-chain over overlapping windows (`chunk_overlap`, default 24). Points still visible in a window's overlap become **queries for the next window keeping their global ID** → tracks stay continuous across seams (no mid passes). Per-chunk **OOM retry**: catch CUDA OOM → `empty_cache` → `cv2.resize` block ×`oom_scale_step` (floor `oom_scale_floor`) + rescale carried queries → retry. Coords assembled at **original** resolution (so `_merge_filter_export` is called with `inv=1.0`); gating runs at `(W0,H0)` via `_gate_assembled`.
- **Frame source** (`app/video_io.FrameSource`): full-decode-to-RAM when the clip fits (`stream_decode="auto"` + `host_ram_frac`), else per-window decode (`read_window_bgr_scaled`) so host RAM is bounded for long 4K. `RunnerConfig` carries all of these; UI exposes only **Track chunks** (`AppState.track_chunks`).
- Both paths feed the shared `_merge_filter_export(passes, T, W0, H0, diag, inv)` (filter → gate → smooth → 0-track diagnostics → original-coord export).
- Seeding feature-based by default (`cv2.goodFeaturesToTrack` gated by inverse SAM3 mask), falls back to uniform grid (`engine.track_grid`) only if fewer than 5 features found.
- Two independent filter stages can drop tracks: `_post_filter_tracks` (motion-residual/jitter/jump outlier rejection vs frame diagonal) and `_apply_per_frame_mask_gating` (drops tracks that ever enter mask region in "outside" mode, or don't stay inside it ≥`inside_ratio` of time in "inside" mode).
- **0-track diagnostics**: when a shot exports 0 tracks, `_run_impl` logs a `ZERO …` line naming the stage that killed them — `seeded=` (no features + empty grid), `after_filter=` (post-filter rejected all), `after_gate=` (mask gating dropped all — outside: every track entered mask; inside: none stayed ≥`inside_ratio`), or `short=` (survived but <2 visible points).
- Coords exported at **original** (unscaled) resolution even when frames processed at downscaled `selected_scales` factor — rescaled by `1/scale` before write. Y flipped (`H0-1-y`) by default for 3DE (`flip_y_for_3de`).
- `CoTracker3Engine` (`app/cotracker_engine.py`) loads model FP32 but feeds FP16 video tensors through `torch.autocast("cuda")` — deliberate VRAM/precision tradeoff (see inline "FIX 1"/"FIX 2"); don't "simplify" to plain `.half()` without checking VRAM headroom.

## Conventions

- **VRAM staging** (16 GB RTX A4000 budget): Qwen freed after Analyze; Ollama LLM stays resident through Analyze, unloaded at Mask (`_free_ollama`, `keep_alive=0`); CUDA cache cleared before SAM3 + before/after CoTracker (`_free_vram`). Don't reorder model loads without re-checking OOM headroom.
- **Reuse existing masks**: at Mask, `shots_with_existing_masks` checks `<OUT>/<shot>/masks*/*.png`. NiceGUI asks Reuse/Regenerate/Cancel; `state.reuse_existing_masks` → `worker_mask` writes a filtered `*_run.json` guide so SAM3 only generates the missing shots.
- **Stop / single-session**: module-global `STOP_EVENT` (threading.Event) checked between shots/passes in `worker_track` (no mid-pass interrupt — CoTracker pass atomic); cleared at job start. Job state via module globals (`JOB_QUEUE`, `CURRENT_JOB_*`, `LAST_PROGRESS`) — single-user assumption; two tabs interleave.
- Heavy dynamic `importlib.util.spec_from_file_location` loading instead of normal package imports — subsystems live in non-package dirs under `pipeline/`. New cross-subsystem imports → follow the rglob loader pattern, don't `import` the dirs directly.
- Shot identity = filename stem (or folder name), must match across Qwen JSON, masking guide JSON, SAM3 output folder, input video — case-insensitive + whitespace/dash/underscore-insensitive (`_norm_shot`) on LLaMA side, case-insensitive-only (`_find_child_dir_case_insensitive`) on SAM3/tracker side.
- `pipeline/co-tracker-main` — vendored upstream code (not tracked here); treat as a read-only dep, not code to refactor. (Earlier vendored `RAFT-master`/`tapnet-main` were removed — unused by the CoTracker path.)
