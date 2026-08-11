# -*- coding: utf-8 -*-
"""TAPNext++ tracking engine (Apache-2.0, commercial-safe).

Drop-in replacement for the old CoTracker3 engine: exposes the SAME two methods
`track_grid` / `track_queries` with the SAME numpy shapes so `tracker_core.py`
(seeding, 4-pass merge, mask gating, filtering, 3DE export) is untouched.

Key differences from CoTracker, absorbed entirely inside this wrapper:
  * TAPNext++ is a 256x256 model in practice. Frames are resized to the model's
    input size on the way in and predicted coordinates are rescaled back to the
    caller's frame space, so tracker_core keeps working in its (Ws,Hs) scaled-pixel
    space. The input size is parameterised (BTR_TAPNEXT_IMG) but 256 is the only
    value that works -- see the measurement by _IMG_ENV below before changing it.
  * TAPNext is CAUSAL/STREAMING (next-token): seed query points at frame 0 of the
    block, then feed one frame at a time carrying the recurrent state. tracker_core
    already reverses frames for the BWD / MID passes, so every pass seeds at the
    block's local frame 0 -> a single forward stream covers it.
  * TAPNext has no native grid mode, so `track_grid` synthesizes a uniform grid of
    query points (optionally gated by the SAM3 inclusion mask) and streams them.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import cv2  # type: ignore

# Help PyTorch manage memory fragmentation (kept from the old engine).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# TAPNext++ published checkpoint is trained/served at 256x256.
_IMG = 256

# Optional higher inference resolution, from BTR_TAPNEXT_IMG (must be a multiple of the
# patch size, i.e. 256 / 384 / 512).
#
# There is NO 512 checkpoint. Probed 2026-08-11: tapnextpp_512.ckpt, tapnextpp_512.pt and
# tapnextpp_ckpt_512.pt all return 404 from storage.googleapis.com/dm-tapnet/tapnextpp/;
# only tapnextpp_ckpt.pt (256) exists, and the repo's own table lists TAPNext at 256x256
# only. So running at 512 means running the 256 WEIGHTS at 512, with the learned image
# position embedding bicubically resized 32x32 -> 64x64.
#
# That is the step the TAPNext++ paper performs BEFORE fine-tuning at 512, and the fine-tune
# is the part we cannot reproduce.
#
# MEASURED 2026-08-11 and the answer is NO. Same 12-frame synthetic pan, three points,
# ground-truth motion of (+2.0, +1.0) px/frame:
#
#     img=256   mean err   0.553px   max   1.250px   1.49s   1.35 GB
#     img=512   mean err 135.790px   max 761.660px   5.62s   3.19 GB
#
# 245x worse -- on a 960px-wide frame a 761px error is not a degraded track, it is noise.
# The interpolated table puts the transformer in a positional space it never trained on and
# the model simply does not work there. The fine-tune is not a refinement of this idea, it is
# the whole thing.
#
# Kept, env-gated and defaulting to 256, for two reasons: the measurement above is one
# command away from being reproduced, and if a real 512 checkpoint is ever published the
# plumbing (coordinate scaling, chunk sizing) is already correct. Do NOT raise this expecting
# accuracy from the current weights.
_IMG_ENV = "BTR_TAPNEXT_IMG"


def _resolve_img_size() -> int:
    try:
        v = int(os.environ.get(_IMG_ENV, "") or _IMG)
    except ValueError:
        return _IMG
    if v % 8 or v < 128 or v > 1024:
        return _IMG
    return v


def _add_tapnext_to_path(tool_root: str) -> str:
    """Locate the vendored google-deepmind/tapnet repo and put it on sys.path.

    Mirrors the CoTracker loader: search a few conventional locations for a repo
    that contains the `tapnet/tapnext/tapnext_torch.py` module.
    """
    thirdparty = os.path.join(tool_root, "thirdparty")
    pipeline = os.path.join(tool_root, "pipeline")
    candidates = [
        os.path.join(pipeline, "tapnext-main"),
        os.path.join(pipeline, "tapnext"),
        os.path.join(pipeline, "tapnet-main"),
        os.path.join(pipeline, "tapnet"),
        os.path.join(thirdparty, "tapnext-main"),
        os.path.join(thirdparty, "tapnet-main"),
    ]
    repo = None
    for c in candidates:
        if os.path.isfile(os.path.join(c, "tapnet", "tapnext", "tapnext_torch.py")):
            repo = c
            break
    if repo is None:
        raise RuntimeError(
            "TAPNext repo not found. Clone google-deepmind/tapnet into one of:\n"
            + "\n".join(f"  {c}" for c in candidates)
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


def _resolve_ckpt(tool_root: str, repo_root: str) -> str:
    """Find tapnextpp_ckpt.pt. Env override wins, then repo/local checkpoints dirs."""
    env = os.environ.get("BTR_TAPNEXT_CKPT", "").strip()
    if env and os.path.isfile(env):
        return env
    for base in (repo_root, tool_root):
        for name in ("tapnextpp_ckpt.pt", "tapnext_ckpt.pt"):
            p = os.path.join(base, "checkpoints", name)
            if os.path.isfile(p):
                return p
    raise RuntimeError(
        "Missing TAPNext++ checkpoint 'tapnextpp_ckpt.pt'. Download:\n"
        "  https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt\n"
        f"into {os.path.join(repo_root, 'checkpoints')} (or set BTR_TAPNEXT_CKPT)."
    )


class TapNextEngine:
    def __init__(self, tool_root: str, device: str = "cuda"):
        self.tool_root = tool_root
        self.repo_root = _add_tapnext_to_path(tool_root)

        import torch  # type: ignore
        from tapnet.tapnext.tapnext_torch import TAPNext  # type: ignore

        self.torch = torch
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

        self.checkpoint = _resolve_ckpt(tool_root, self.repo_root)
        self.img = _resolve_img_size()
        model = TAPNext(image_size=(self.img, self.img))
        ckpt = torch.load(self.checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # Published checkpoint prefixes weights with "tapnext." — strip it.
        state_dict = {k.replace("tapnext.", "", 1): v for k, v in state_dict.items()}
        if self.img != _IMG:
            state_dict = self._fit_state_dict_to_img(state_dict, model)
        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()

    # ---- running the 256 weights at another resolution -------------------------
    def _fit_state_dict_to_img(self, sd: dict, model) -> dict:
        """Resize the checkpoint's position tables to this model's token grid.

        Two entries are resolution-bound and neither can be loaded as-is:

          image_pos_emb   a LEARNED (1, 32*32, 768) table. Bicubically resampled to the new
                          patch grid -- the same operation the TAPNext++ paper applies before
                          fine-tuning at 512. Nothing here reproduces that fine-tune, so the
                          transformer is being asked to read positions it never trained on.
          query_pos_embed a DETERMINISTIC sincos buffer the model already rebuilt for its own
                          image_size in __init__. The checkpoint's copy is simply dropped;
                          keeping it would only overwrite a correct table with a stale one.
        """
        torch = self.torch
        out = dict(sd)

        want = model.state_dict()
        # Substitute, don't drop: load_state_dict stays STRICT, so a genuinely missing weight
        # is still an error rather than silently leaving a randomly-initialised layer in a
        # model that then tracks plausible-looking nonsense.
        if "query_pos_embed" in want:
            out["query_pos_embed"] = want["query_pos_embed"]

        key = "image_pos_emb"
        if key in out and key in want and out[key].shape != want[key].shape:
            src, dst = out[key], want[key]
            n_src, c = int(src.shape[1]), int(src.shape[2])
            n_dst = int(dst.shape[1])
            g_src, g_dst = int(round(n_src ** 0.5)), int(round(n_dst ** 0.5))
            if g_src * g_src != n_src or g_dst * g_dst != n_dst:
                raise RuntimeError(f"{key}: non-square token grid ({n_src} -> {n_dst})")
            grid = src.reshape(1, g_src, g_src, c).permute(0, 3, 1, 2).float()
            grid = torch.nn.functional.interpolate(
                grid, size=(g_dst, g_dst), mode="bicubic", align_corners=False)
            out[key] = grid.permute(0, 2, 3, 1).reshape(1, n_dst, c).to(src.dtype)
            print(f"TAPNext: image_pos_emb {g_src}x{g_src} -> {g_dst}x{g_dst} (bicubic); "
                  f"running 256-trained weights at {self.img}px")
        return out

    def _autocast(self, fp16: bool):
        """Half-precision inference, unless BTR_TAPNEXT_FP32=1.

        The tile model is the dominant cost of a 4K run: moving_tile_refine re-runs it per
        track per window, which on SH016 was 9.3 of an 18 min pipeline. Two things it is NOT,
        both measured before this was written:

          * launch-bound. Batching tiles gives nothing -- per-track cost is flat at ~1.2s
            whether B is 1, 4 or 16, i.e. a single 256px stream already saturates the GPU.
          * cheap to pack differently. The win has to come from less work per frame.

        fp16 autocast, 16-frame tile, measured on an RTX A4000:

            fp32     1337.1 ms   (reference)
            fp16      816.7 ms   1.64x   max coordinate change 0.0023 px
            bfloat16  806.8 ms   1.66x   max coordinate change 0.0266 px

        0.0023px is ~40x below the bench's own measurement floor (0.06px). bfloat16 is no
        faster and 10x less precise -- fewer mantissa bits -- so fp16 is the choice, not
        "whatever half is available".

        Confirmed on the real plate, whole moving-tile stage, SH016 4096x2160 / 38 tracks:
        6.24 min -> 3.58 min (1.74x). The accuracy story there is NOT just the mean, and the
        tail is the honest part:

            coords 4718   mean 0.00522   p50 0.00136   p99 0.0413   max 1.5354 px
            over 0.1px: 38/4718     over 1.0px: 3/4718
            tracks with any coord >0.1px: 4 of 38   (worst 1.535, next 0.255)

        One track moved 1.5px. That is not rounding -- moving_tile_refine is a FEEDBACK loop
        (each window's result places the next window's tile) with hard branches in it: the
        `tx <= 2.0` tile-edge tests and the `if not vs[k]: break` visibility stop. A 0.002px
        perturbation can flip one of those, end a window a frame earlier, and send that track
        down a different path from there. Neither result is the correct one; both are valid
        outcomes of a knife-edge decision, and the quality gates downstream judge them the
        same way. p99 = 0.04px is the number that describes the other 34 tracks.

        SCOPE: this is an explicit per-CALL argument, and only moving_tile_refine passes it.
        That is not caution, it is where the measurement applies. Enabling it globally was
        tried and measured on a full SH016 pipeline run against the fp32 run:

            isolated moving-tile   p99 0.041px   max 1.54px    4/38 tracks >0.1px
            whole pipeline         p99 9.14px    max 14.65px  26/34 tracks >0.1px, 38 -> 37

        The second number is not fp16 tracking 14px worse. The main FWD/BWD/MID passes feed
        seeding, the quality gate, spread selection and defragmentation -- so perturbing them
        changes WHICH tracks survive and what they are called, and the diff is then comparing
        different tracks. Chaotic, not wrong, but it makes a run irreproducible against an
        earlier delivery for no measured gain: those passes are ~2 min of an 18 min run, while
        moving-tile is 9.3.

        It is an argument rather than a flag on the engine on purpose: a sticky flag left on
        by an exception mid-stage (an OOM in the tile loop is the obvious one) would silently
        put every LATER shot's coarse passes in fp16, and nothing would report it.

        So the coarse passes stay fp32 -- deterministic, and identical to every export made
        before this existed -- and the half-precision speed is taken on the stage that
        actually costs the time. BTR_TAPNEXT_FP32=1 disables it there too.
        """
        torch = self.torch
        if (self.device == "cpu" or not fp16
                or os.environ.get("BTR_TAPNEXT_FP32", "").strip() == "1"):
            return torch.autocast("cuda", enabled=False)
        return torch.autocast("cuda", dtype=torch.float16)

    # ---- video / coordinate plumbing (256x256 <-> caller frame space) ----------
    def _prep_video(self, frames_bgr: np.ndarray):
        """(T,H,W,3) BGR -> (1,T,256,256,3) float[-1,1] RGB tensor. Returns (video, H, W).

        TAPNext expects TAP-Vid normalization frames/255*2-1 -> [-1,1] (canonical across
        tapnet: tapvid/evaluation_datasets, pytorch_live_demo, training). Feeding [0,1]
        makes tracks ignore image content -> uniform sliding.
        """
        torch = self.torch
        T, H, W = frames_bgr.shape[0], frames_bgr.shape[1], frames_bgr.shape[2]
        S = self.img
        out = np.empty((T, S, S, 3), dtype=np.float32)
        for t in range(T):
            r = cv2.resize(frames_bgr[t], (S, S), interpolation=cv2.INTER_AREA)
            out[t] = r[..., ::-1]  # BGR -> RGB
        out = out / 255.0 * 2.0 - 1.0  # -> [-1, 1]
        video = torch.from_numpy(out)[None].to(self.device)  # (1,T,S,S,3)
        return video, H, W

    def _q_to_model(self, queries: np.ndarray, H: int, W: int) -> np.ndarray:
        """Caller queries are [frame, x, y] in (W,H) pixels. TAPNext wants [t, y, x] in model
        px (see tapnet/tapvid/evaluation_datasets: query_points = [t, y, x]). Swap + scale."""
        qin = queries.astype(np.float32)
        sx, sy = float(self.img) / max(1, W), float(self.img) / max(1, H)
        out = np.empty_like(qin)
        out[..., 0] = qin[..., 0]           # t
        out[..., 1] = qin[..., 2] * sy      # y -> row (256)
        out[..., 2] = qin[..., 1] * sx      # x -> col (256)
        return out

    def _tracks_from_model(self, tracks_m: np.ndarray, H: int, W: int) -> np.ndarray:
        """TAPNext returns (T,N,2) as [y, x] in model px. Back to [x, y] in (W,H) pixels."""
        out = np.empty_like(tracks_m, dtype=np.float32)
        out[..., 0] = tracks_m[..., 1] * (float(W) / self.img)   # x = col
        out[..., 1] = tracks_m[..., 0] * (float(H) / self.img)   # y = row
        return out

    # ---- core streaming inference ---------------------------------------------
    def _stream(self, video, query_points, fp16: bool = False):
        """Run TAPNext causally over the whole block.

        video: (1,T,256,256,3) float tensor. query_points: (1,N,3) [t,x,y] tensor (256 space).
        Returns tracks (T,N,2) numpy [x,y] and visibility (T,N) bool numpy.
        """
        torch = self.torch
        T = int(video.shape[1])
        tr_all, vis_all = [], []
        with torch.no_grad(), self._autocast(fp16):
            # Frame 0 seeds the queries and initializes the recurrent state.
            tracks, _tlog, vlog, state = self.model(
                video=video[:, :1], query_points=query_points
            )
            tr_all.append(tracks)
            vis_all.append(vlog)
            for f in range(1, T):
                tracks, _tlog, vlog, state = self.model(
                    video=video[:, f:f + 1], state=state
                )
                tr_all.append(tracks)
                vis_all.append(vlog)
        tracks = torch.cat(tr_all, dim=1)[0]           # (T,N,2)
        vlog = torch.cat(vis_all, dim=1)[0]            # (T,N) or (T,N,1)
        if vlog.ndim == 3 and vlog.shape[-1] == 1:
            vlog = vlog.squeeze(-1)                     # drop trailing singleton -> (T,N)
        vis = (vlog > 0)                               # bool  (visible_logits > 0)
        return tracks.float().cpu().numpy(), vis.cpu().numpy().astype(bool)

    # ---- public API (matches the old CoTracker3Engine) -------------------------
    def track_queries(self, frames_bgr: np.ndarray, queries: np.ndarray, fp16: bool = False):
        """queries: (1,N,3) [frame,x,y] in caller frame space (frame index expected 0)."""
        video, H, W = self._prep_video(frames_bgr)
        if queries.ndim == 2:
            queries = queries[None]
        N = int(queries.shape[1])
        if N == 0:
            T = int(frames_bgr.shape[0])
            return np.zeros((T, 0, 2), np.float32), np.zeros((T, 0), bool)
        q = self._q_to_model(queries, H, W)
        q_t = self.torch.from_numpy(q).float().to(self.device)
        tracks256, vis = self._stream(video, q_t, fp16=fp16)
        return self._tracks_from_model(tracks256, H, W), vis

    def track_grid(self, frames_bgr: np.ndarray, grid_size: int,
                   grid_query_frame: int = 0, segm_mask: np.ndarray | None = None):
        """Synthesize a uniform grid of query points (gated by segm_mask) and stream them.

        segm_mask (H,W): seed only where nonzero (matches CoTracker's inclusion semantics).
        """
        T, H, W = frames_bgr.shape[0], frames_bgr.shape[1], frames_bgr.shape[2]
        g = max(1, int(grid_size))
        # Interior grid with a half-cell margin so points aren't on the frame edge.
        xs = (np.arange(g) + 0.5) * (W / g)
        ys = (np.arange(g) + 0.5) * (H / g)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)  # (g*g,2) [x,y]

        if segm_mask is not None:
            m = segm_mask
            if m.shape[:2] != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            keep = m[np.clip(pts[:, 1].astype(int), 0, H - 1),
                     np.clip(pts[:, 0].astype(int), 0, W - 1)] > 0
            pts = pts[keep]

        N = pts.shape[0]
        if N == 0:
            return np.zeros((T, 0, 2), np.float32), np.zeros((T, 0), bool)
        queries = np.zeros((1, N, 3), dtype=np.float32)
        queries[0, :, 0] = float(grid_query_frame)
        queries[0, :, 1] = pts[:, 0]
        queries[0, :, 2] = pts[:, 1]
        return self.track_queries(frames_bgr, queries)
