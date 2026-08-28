"""SpatialTracker V2 (offline) on a plate directory, exported for Blender.

SpaTrackV2 is a different animal from everything else measured in this repo. TAPNext, MFT
and the Blender tracker all answer "where did this pixel go" in 2D. SpaTrackV2 answers it
in 3D: a VGGT-derived front end predicts per-frame camera pose, intrinsics and a metric
point map, and the tracker then follows query points through that reconstructed space. So
its output is not a 2D track file -- it is a camera and a moving 3D point cloud, which is
why the deliverable here is a .blend rather than a 3DE ASCII.

This script does NOT touch the bot. It vendors the upstream repo (vendor/SpaTrackerV2),
keeps its extra dependencies in a private `pydeps/` so the app runtime stays as it was, and
reads weights from `weights/` (downloaded once, never fetched at run time).

    runtime\\python311\\python.exe "Spatial tracker\\run_spatracker.py" ^
        --plate experiments\\blender_track\\out\\SH006\\plate ^
        --name SH006 --start 1 --end 48 --grid 12

Frames are fed at 518 px wide (the front end's native input; anything else is resized to it
inside the model anyway), and the 2D side of the result is scaled back to plate resolution
on export so a track file stays comparable with the rest of the tooling.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VENDOR = os.path.join(HERE, "vendor", "SpaTrackerV2")
PYDEPS = os.path.join(HERE, "pydeps")
# pydeps first: the embeddable runtime is in isolated mode, so PYTHONPATH is ignored and
# the only way these land on the path is from inside the process.
for _p in (PYDEPS, VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # weights are local; never phone home

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_BLENDER = os.environ.get(
    "BTR_BLENDER_EXE",
    r"C:\Users\jefrin\Downloads\blender-5.2.0-windows-x64\blender-5.2.0-windows-x64\blender.exe")

TARGET_W = 518          # VGGT patch grid is 14 px; 518 = 37 patches
VIS_THRESH = 0.5        # vis_pred is a score in roughly 0.2..1.2, not a flag


def load_plate(plate: str, start: int, end: int, stride: int):
    """Frames at the model's working width, plus the original plate size for the export.

    Loading 4K frames and letting the model downscale them would cost ~6 GB of host RAM for
    a 50 frame window and change nothing: the front end resizes to 518 px wide regardless.
    """
    files = sorted(glob.glob(os.path.join(plate, "*.png")) +
                   glob.glob(os.path.join(plate, "*.jpg")) +
                   glob.glob(os.path.join(plate, "*.exr")))
    if not files:
        raise SystemExit(f"no frames in {plate}")
    end = end or len(files)
    picked = files[start - 1:end:stride]
    if not picked:
        raise SystemExit(f"frame range {start}-{end} step {stride} selected nothing")

    probe = cv2.imread(picked[0], cv2.IMREAD_COLOR)
    plate_h, plate_w = probe.shape[:2]
    new_h = int(round(plate_h * (TARGET_W / plate_w) / 14) * 14)

    frames = np.empty((len(picked), new_h, TARGET_W, 3), dtype=np.float32)
    for i, f in enumerate(picked):
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (TARGET_W, new_h), interpolation=cv2.INTER_AREA)
        frames[i] = im[:, :, ::-1]                       # BGR -> RGB, still 0..255
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    return video, (plate_w, plate_h), picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--name", default="shot")
    ap.add_argument("--start", type=int, default=1, help="1-based, inclusive")
    ap.add_argument("--end", type=int, default=0, help="1-based inclusive, 0 = last")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--grid", type=int, default=12, help="grid_size: grid**2 query points")
    ap.add_argument("--vo-points", type=int, default=756)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--no-blend", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t_all = time.time()

    video, (plate_w, plate_h), picked = load_plate(args.plate, args.start, args.end, args.stride)
    T, _, H, W = video.shape
    print(f"[plate] {T} frames  {plate_w}x{plate_h} -> {W}x{H}  "
          f"({os.path.basename(picked[0])}..{os.path.basename(picked[-1])})")

    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.utils import get_points_on_a_grid
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

    # ---------------------------------------------------------------- front end
    # Camera poses, intrinsics and a metric point map for the whole window at once. This is
    # the 4.6 GB half of the download and the part that decides whether the 3D is any good.
    front_dir = os.path.join(HERE, "weights", "SpatialTrackerV2_Front")
    print(f"[front] loading {front_dir}")
    front = VGGT4Track.from_pretrained(front_dir).eval().to("cuda")
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = front(video.cuda()[None] / 255)
        extrinsic, intrinsic = pred["poses_pred"], pred["intrs"]
        depth_map, depth_conf = pred["points_map"][..., 2], pred["unc_metric"]
    depth = depth_map.squeeze().float().cpu().numpy()
    extrs = extrinsic.squeeze().float().cpu().numpy()
    intrs = intrinsic.squeeze().float().cpu().numpy()
    unc_metric = depth_conf.squeeze().float().cpu().numpy() > 0.5
    print(f"[front] {time.time() - t0:.1f}s   depth {depth.shape}  "
          f"conf>0.5 on {100 * unc_metric.mean():.1f}% of pixels")
    del front, pred, depth_map, depth_conf
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- tracker
    off_dir = os.path.join(HERE, "weights", "SpatialTrackerV2-Offline")
    print(f"[track] loading {off_dir}")
    model = Predictor.from_pretrained(off_dir)
    model.spatrack.track_num = args.vo_points
    model.eval()
    model.to("cuda")

    grid_pts = get_points_on_a_grid(args.grid, (H, W), device="cpu")
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    print(f"[track] {len(query_xyt)} query points on frame 0")

    t0 = time.time()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (c2w_traj, intrs_out, point_map, conf_depth,
         track3d_pred, track2d_pred, vis_pred, conf_pred, video_out) = model.forward(
            video, depth=depth, intrs=intrs, extrs=extrs, queries=query_xyt,
            fps=1, full_point=False, iters_track=args.iters,
            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
            support_frame=T - 1, replace_ratio=0.2)
    print(f"[track] {time.time() - t0:.1f}s")

    # world-space 3D tracks: the tracker predicts in camera space, c2w puts them in world
    coords = (torch.einsum("tij,tnj->tni", c2w_traj[:, :3, :3], track3d_pred[:, :, :3].cpu())
              + c2w_traj[:, :3, 3][:, None, :]).numpy()
    # vis_pred is a score, not a flag -- it runs past 1.0 and never reaches 0, so anything
    # that treats it as a boolean marks every sample visible and the occlusions vanish.
    vis = (vis_pred.cpu().numpy().reshape(T, -1) > VIS_THRESH)
    track2d = track2d_pred[..., :2].cpu().numpy()

    # ---------------------------------------------------------------- self-check
    # The 3D branch and the 2D branch can disagree, and when the reconstruction degenerates
    # it does so silently: points end up BEHIND the camera and the export still writes. So
    # reproject the 3D tracks through the predicted camera and compare against the 2D ones
    # before believing any of it. On a healthy window this sits at a few pixels.
    K = intrs_out.cpu().numpy()
    w2c = torch.inverse(c2w_traj).cpu().numpy()
    errs, behind = [], 0
    for t in range(T):
        P = (w2c[t][:3, :3] @ coords[t].T).T + w2c[t][:3, 3]
        behind += int((P[:, 2] <= 1e-2).sum())
        uv = (K[t] @ P.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        m = vis[t]
        if m.any():
            errs.append(float(np.median(np.linalg.norm(uv[m] - track2d[t][m], axis=1))))
    med = float(np.median(errs)) if errs else float("nan")
    print(f"[check] 3D->2D reprojection median {med:.2f} px at {W}x{H} "
          f"({med * plate_w / W:.1f} px at plate res); "
          f"{100.0 * behind / (T * coords.shape[1]):.1f}% of samples behind the camera")
    if med > 10 or behind > 0.05 * T * coords.shape[1]:
        print("[check] DEGENERATE -- the reconstruction did not hold over this window. "
              "Use fewer frames (or a larger --stride) and re-run; the .blend below is not "
              "trustworthy.")

    npz = os.path.join(args.out, f"{args.name}__spatrack.npz")
    np.savez(npz,
             coords=coords,
             c2w=c2w_traj.cpu().numpy(),
             extrinsics=torch.inverse(c2w_traj).cpu().numpy(),
             intrinsics=intrs_out.cpu().numpy(),
             visibs=vis,
             confs=conf_pred.cpu().numpy().reshape(T, -1),
             track3d_cam=track3d_pred[:, :, :3].cpu().numpy(),
             track2d=track2d,
             work_size=np.array([W, H]),
             plate_size=np.array([plate_w, plate_h]),
             frames=np.array([os.path.abspath(p) for p in picked]),
             start=args.start, stride=args.stride)
    print(f"[save] {npz}")

    # 2D side, at plate resolution, in the repo's 3DE convention (y-up) -- free to write and
    # it is the only part of this that is comparable with the other trackers.
    scale = plate_w / float(W)
    txt = os.path.join(args.out, f"{args.name}__spatrack_2d.txt")
    write_3de(txt, track2d, vis, scale, plate_h, args.start, args.stride)
    print(f"[save] {txt}")

    if not args.no_blend:
        blend = os.path.join(args.out, f"{args.name}__spatrack.blend")
        rc = run_blender(args.blender, npz, blend)
        print(f"[blender] rc={rc}  {blend}")

    print(f"[done] {time.time() - t_all:.1f}s total")
    return 0


def write_3de(path: str, track2d, vis, scale: float, plate_h: int, start: int, stride: int):
    """3DE ASCII, y-up, per-point frame numbers -- occluded samples are dropped, not faked."""
    T, N, _ = track2d.shape
    lines = [str(N)]
    for n in range(N):
        rows = []
        for t in range(T):
            if not vis[t, n]:
                continue
            x = float(track2d[t, n, 0]) * scale
            y = (plate_h - 1.0) - float(track2d[t, n, 1]) * scale
            rows.append(f"{start + t * stride} {x:.7f} {y:.7f}")
        lines += [f"SPT_{n:04d}", "0", str(len(rows))] + rows
    with open(path, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")


def run_blender(exe: str, npz: str, blend: str) -> int:
    if not os.path.isfile(exe):
        print(f"[blender] not found: {exe}  (set BTR_BLENDER_EXE) -- skipping .blend")
        return -1
    script = os.path.join(HERE, "to_blender.py")
    # --python-exit-code, or a script that raises still exits 0 and the run reports success
    # with no .blend written -- which is exactly how the first version of this failed.
    cmd = [exe, "--background", "--factory-startup", "--python-exit-code", "1",
           "--python", script, "--", "--npz", npz, "--out", blend]
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in (p.stdout or "").strip().splitlines():
        if line.startswith(("[blend]", "[plate]", "[warn]", "Error", "Traceback")):
            print("   " + line)
    if p.returncode:
        print((p.stdout or "").strip()[-2000:])
        print((p.stderr or "").strip()[-2000:])
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
