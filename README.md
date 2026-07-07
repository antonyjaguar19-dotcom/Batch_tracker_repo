# Batch Tracker

![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6?logo=windows&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![last commit](https://img.shields.io/github/last-commit/antonyjaguar19-dotcom/Batch_tracker_repo)
![repo size](https://img.shields.io/github/repo-size/antonyjaguar19-dotcom/Batch_tracker_repo)

![Qwen2.5-VL](https://img.shields.io/badge/Qwen2.5--VL-VLM-1E90FF)
![LLaMA 3.1](https://img.shields.io/badge/LLaMA_3.1-Ollama-8A2BE2)
![SAM3](https://img.shields.io/badge/SAM3-masks-FF8C00)
![SynthEyes](https://img.shields.io/badge/SynthEyes-tracking-008080)
![TAPNext++](https://img.shields.io/badge/TAPNext++-tracking-purple)

Windows/CUDA VFX matchmove pipeline. Turns raw shot footage into 2D point tracks
importable into 3D Equalizer, by chaining four AI/CV stages:

1. **Qwen2.5-VL (vision-language)** — describes each shot: scene, camera movement,
   objects, and matchmove signals (moving things, bad-track regions, foreground
   occluders, quality flags, depth layers, parallax).
2. **LLaMA 3.1 (Ollama) + deterministic heuristics** — decide per-shot camera/object
   track strategy and the SAM3 include/exclude mask prompts.
3. **SAM3** — turns those prompts into per-frame keep/ignore alpha masks.
4. **Tracking** — SynthEyes (primary, over SyPy3) or **TAPNext++** (secondary GPU
   fallback, Apache-2.0) tracks points per shot, filters against the masks, and
   exports 3D Equalizer `.txt` tracks.

## Fresh-machine setup (clone + weights + runtime)

The weights and the portable Python runtime are **not in git** (too big). On a bare
Windows box, download and double-click **`clone_and_setup.bat`** — it clones the repo,
bootstraps `runtime/python311/`, `pip install`s the cu121 deps, and downloads the model
weights (Qwen2.5-VL, SAM3, TAPNext++). Prereqs it does *not* install: **Git**, an NVIDIA
driver new enough for CUDA 12.1, and **Ollama** (optional, for the LLaMA stage). SAM3 is a
**gated** HF repo — the script prompts for a Hugging Face token with access to `facebook/sam3`.

Notes:
- Weights are fetched with the `huggingface_hub` **Python API** (`hf_hub_download` /
  `snapshot_download`), not the `hf` CLI — the embeddable interpreter omits the `venv`
  stdlib module the CLI imports on startup.
- `QWEN_REPO` in the bat's CONFIG block picks the Qwen model — default 7B (~16 GB, matches
  the int4/bitsandbytes loader); switch to `Qwen/Qwen2.5-VL-3B-Instruct` (~7 GB) for a lighter box.
- **tkinter/tcl** is not part of the embeddable Python, so the UI's folder-browse dialogs
  won't work until Tcl/Tk is copied in (paste `tcl/`, `DLLs/tk*`, `DLLs/tcl*` and
  `Lib/tkinter/` from a full CPython 3.11.9 install into `runtime/python311/`). Not scripted —
  you can type paths into the UI fields directly meanwhile.

## Running it

Double-click **`app\BotBatchTracker.bat`**. It uses the project-local Python interpreter,
keeps all caches/temp inside the repo, and opens the NiceGUI app at http://localhost:8080.

Workflow in the UI:

1. **Scan inputs** — set Input/Output folders, scan to list shots (resolution, frames).
   Re-scanning an Output that holds a previous run restores its include/exclude prompts
   and frame range.
2. Tick the shot(s) to run (table checkboxes; active rows are highlighted). Use the
   search box to filter, and the **Edit shot** picker to tune one shot.
3. **Analyze (AI)** — runs Qwen2.5-VL then LLaMA/Ollama, writes a masking guide.
4. (Optional) edit per-shot Include/Exclude prompts, downscale, and **frame range**.
5. **Generate masks** — SAM3 makes the alpha masks (honors the frame range). If masks
   already exist for the selected shots, you're asked to **Reuse** (skip) or
   **Regenerate** (overwrite).
6. **Start tracking** — pick the backend under **Settings → Tracking backend**
   (`syntheyes` default, `tapnext` fallback). SynthEyes exports `<shot>__syntheyes.txt`;
   the TAPNext++ fallback exports `<shot>__tapnext.txt`. A shot that yields 0 tracks logs
   the reason (no features / filtered out / mask-gated / too short). On the TAPNext++ path,
   long / high-res shots are auto-split into chunks to fit GPU memory (see below).

## Layout

```
batch_tracker_v001_starter/
  launch_nicegui.bat        the only launcher (root)
  requirements.txt          frozen deps (torch/torchvision are cu121 builds)
  README.md  DECISIONS.md  CLAUDE.md
  app/
    app_nicegui.py          NiceGUI UI entry
    app.py                  shared backend (workers, loaders, state)
    tracker_core.py  tapnext_engine.py  syntheyes_engine.py  export_3de.py
    video_io.py  video_meta.py  reformat_plate_core.py  __init__.py
  pipeline/
    qwen/                   run_qwen2_shot_describer.py, qwen2_shot_describer_core.py, models/
    llama/core/             bridge, heuristics, io_parsers, ollama_backend
    sam3/                   sam3_runner.py, weights/
    tapnext-main/           vendored google-deepmind/tapnet + checkpoints/
  runtime/                  project-local Python + caches (not committed)
```

## Requirements (placed locally, offline by design)

- **Python runtime**: portable interpreter at `runtime/python311/` (set up once; see
  DECISIONS). All pip installs/caches stay inside `runtime/`.
- **Qwen2.5-VL weights** under `pipeline/qwen/models/` — resolver prefers, in order:
  `Qwen2.5-VL-7B-Instruct-AWQ` → `Qwen2.5-VL-7B-Instruct` → `Qwen2.5-VL-3B-Instruct`
  (override with `QWEN2_MODEL_DIR`). 7B is loaded in **int4** (bitsandbytes NF4), ~5.5 GB VRAM.
- **SAM3 weights**: `pipeline/sam3/weights/sam3.pt` (or set `BTR_SAM3_WEIGHTS`).
- **TAPNext++ checkpoint**: `tapnextpp_ckpt.pt` under
  `pipeline/tapnext-main/checkpoints/` (or `<repo>/checkpoints/`, or set `BTR_TAPNEXT_CKPT`).
  Apache-2.0 — commercial use OK.
- **Ollama** running locally with `llama3.1:8b` (default `http://localhost:11434`,
  override with `BTR_OLLAMA_URL`). Only used for ambiguous client intent.

## Tracking backends

Two interchangeable tracking backends, selected in the UI (**Settings → Tracking backend**)
or `AppState.track_backend`:

- **SynthEyes** (default) — drives a local SynthEyes instance over the SyPy3 socket +
  Win32 UI automation; SAM3 masks applied as a Python post-filter. Needs a SynthEyes install
  (commercial app). Output: `<shot>__syntheyes.txt`.
- **TAPNext++** (fallback) — GPU neural point tracker (`google-deepmind/tapnet`, **Apache-2.0,
  commercial-safe**). Fixed 256×256 causal model; the engine wrapper handles resize + coord
  rescale + streaming internally. 4-pass (fwd/bwd/mid) + SAM3 mask gating + chunked OOM-safe
  path. Output: `<shot>__tapnext.txt`. Falls back automatically if SynthEyes is unavailable.

Both export classic 3D Equalizer 2D-track ASCII, frame numbers aligned to the original shot.

## Key features

- **Shot selection**: per-row checkboxes; active shots highlighted; search filter;
  "X of N selected" count; Mask/Track abort if nothing is selected.
- **Per-shot frame range** (1-based, `0` = full). Honored by **both** SAM3 masking and
  the tracker; exported track frame numbers stay aligned to the original shot.
  Restored from the previous run on Scan.
- **Reuse existing masks**: at Mask, if the selected shots already have masks, choose
  Reuse (skip) or Regenerate (overwrite).
- **VRAM staging**: Qwen freed after Analyze; the Ollama LLM stays resident through
  Analyze and is unloaded at Mask; SAM3 freed before Track. Watch the `VRAM freed …`
  log lines.
- **Four "mover" backstops** for camera solves (exclude moving subjects):
  1. VLM `moving_things`, 2. deterministic dynamic-subjects (animals/people/vehicles),
  3. scene-text heuristic, 4. **CV optical-flow** (masks pixels moving independently of
  the camera — even objects never named). The CV layer is a UI toggle
  (**CV motion backstop**) / env `BTR_MOTION_BACKSTOP=0`.
- **0-track diagnostics**: a shot that exports nothing logs which stage dropped the
  tracks (seeding / post-filter / mask gating / too-short).
- **VRAM/RAM-safe tracking** (long / high-res shots, TAPNext++ path): tracks the shot in
  **overlapping chunks whose track IDs are chained across seams** (tracks stay continuous).
  Chunk count is auto-sized from free VRAM (override via **Settings → Track chunks**, `0`=auto);
  on a GPU OOM the chunk auto-downscales and retries; if the whole clip is too big for host RAM,
  frames are decoded per-window. Exported coords are always at original resolution.
- **Busy state**: step buttons disable while a job runs; live status + progress; Stop.

## Environment variables

| Var | Purpose | Default |
|-----|---------|---------|
| `BTR_SAM3_WEIGHTS` | SAM3 `.pt` path | `<repo>/pipeline/sam3/weights/sam3.pt` |
| `BTR_TAPNEXT_CKPT` | TAPNext++ checkpoint `.pt` path | (auto-resolve under `pipeline/tapnext-main/checkpoints/`) |
| `BTR_OLLAMA_URL` | Ollama base URL | `http://localhost:11434` |
| `QWEN2_MODEL_DIR` | Force a specific Qwen model folder | (auto-resolve under `pipeline/qwen/models/`) |
| `BTR_MOTION_BACKSTOP` | `0` disables the CV motion mask | `1` |

See `DECISIONS.md` for why things are the way they are, and `CLAUDE.md` for the
internal architecture / loader rules.
