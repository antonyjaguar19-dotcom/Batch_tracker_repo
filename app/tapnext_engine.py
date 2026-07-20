# -*- coding: utf-8 -*-
"""TAPNext++ tracking engine (Apache-2.0, commercial-safe).

Drop-in replacement for the old CoTracker3 engine: exposes the SAME two methods
`track_grid` / `track_queries` with the SAME numpy shapes so `tracker_core.py`
(seeding, 4-pass merge, mask gating, filtering, 3DE export) is untouched.

Key differences from CoTracker, absorbed entirely inside this wrapper:
  * TAPNext++ is a FIXED 256x256 model. Frames are resized to 256x256 on the way
    in and predicted coordinates are rescaled back to the caller's frame space,
    so tracker_core keeps working in its (Ws,Hs) scaled-pixel space.
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
        model = TAPNext(image_size=(_IMG, _IMG))
        ckpt = torch.load(self.checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # Published checkpoint prefixes weights with "tapnext." — strip it.
        state_dict = {k.replace("tapnext.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()

    # ---- video / coordinate plumbing (256x256 <-> caller frame space) ----------
    def _prep_video(self, frames_bgr: np.ndarray):
        """(T,H,W,3) BGR -> (1,T,256,256,3) float[-1,1] RGB tensor. Returns (video, H, W).

        TAPNext expects TAP-Vid normalization frames/255*2-1 -> [-1,1] (canonical across
        tapnet: tapvid/evaluation_datasets, pytorch_live_demo, training). Feeding [0,1]
        makes tracks ignore image content -> uniform sliding.
        """
        torch = self.torch
        T, H, W = frames_bgr.shape[0], frames_bgr.shape[1], frames_bgr.shape[2]
        out = np.empty((T, _IMG, _IMG, 3), dtype=np.float32)
        for t in range(T):
            r = cv2.resize(frames_bgr[t], (_IMG, _IMG), interpolation=cv2.INTER_AREA)
            out[t] = r[..., ::-1]  # BGR -> RGB
        out = out / 255.0 * 2.0 - 1.0  # -> [-1, 1]
        video = torch.from_numpy(out)[None].to(self.device)  # (1,T,256,256,3)
        return video, H, W

    @staticmethod
    def _q_to_256(queries: np.ndarray, H: int, W: int) -> np.ndarray:
        """Caller queries are [frame, x, y] in (W,H) pixels. TAPNext wants [t, y, x] in 256px
        (see tapnet/tapvid/evaluation_datasets: query_points = [t, y, x]). Swap axes + scale."""
        qin = queries.astype(np.float32)
        sx, sy = float(_IMG) / max(1, W), float(_IMG) / max(1, H)
        out = np.empty_like(qin)
        out[..., 0] = qin[..., 0]           # t
        out[..., 1] = qin[..., 2] * sy      # y -> row (256)
        out[..., 2] = qin[..., 1] * sx      # x -> col (256)
        return out

    @staticmethod
    def _tracks_from_256(tracks256: np.ndarray, H: int, W: int) -> np.ndarray:
        """TAPNext returns (T,N,2) as [y, x] in 256px. Swap back to [x, y] in (W,H) pixels."""
        out = np.empty_like(tracks256, dtype=np.float32)
        out[..., 0] = tracks256[..., 1] * (float(W) / _IMG)   # x = col
        out[..., 1] = tracks256[..., 0] * (float(H) / _IMG)   # y = row
        return out

    # ---- core streaming inference ---------------------------------------------
    def _stream(self, video, query_points):
        """Run TAPNext causally over the whole block.

        video: (1,T,256,256,3) float tensor. query_points: (1,N,3) [t,x,y] tensor (256 space).
        Returns tracks (T,N,2) numpy [x,y] and visibility (T,N) bool numpy.
        """
        torch = self.torch
        T = int(video.shape[1])
        tr_all, vis_all = [], []
        with torch.no_grad():
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
    def track_queries(self, frames_bgr: np.ndarray, queries: np.ndarray):
        """queries: (1,N,3) [frame,x,y] in caller frame space (frame index expected 0)."""
        video, H, W = self._prep_video(frames_bgr)
        if queries.ndim == 2:
            queries = queries[None]
        N = int(queries.shape[1])
        if N == 0:
            T = int(frames_bgr.shape[0])
            return np.zeros((T, 0, 2), np.float32), np.zeros((T, 0), bool)
        q = self._q_to_256(queries, H, W)
        q_t = self.torch.from_numpy(q).float().to(self.device)
        tracks256, vis = self._stream(video, q_t)
        return self._tracks_from_256(tracks256, H, W), vis

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
