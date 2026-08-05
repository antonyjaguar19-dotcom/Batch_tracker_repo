# -*- coding: utf-8 -*-
"""Per-track identity, measurements and policy -- the thing this pipeline never had.

Every track has been tracked with ONE parameter set. RunnerConfig is built once per shot,
_auto_tune mutates that single object, and every seeded point inherits it: the crisp corner
on a bolt head and the soft blob on a wall both get a 31px pattern box, translation-only
motion and the same NCC thresholds. Those two features want opposite settings.

The information to do better is already measured and then discarded one line later --
_drop_edge_points computes a structure tensor at every seed and reduces it to a boolean;
_refine_segment builds real per-track state (anchor, patch, certs) as locals and drops it
on return. This module is where that survives.

Two pieces:

  TrackRegistry   one per shot. Maps track id -> TrackMeta, and follows the id through the
                  two places it mutates (refine split, defragment split).
  _TrackCfg       a read-through view of the shot RunnerConfig with a handful of per-track
                  overrides on top.

_TrackCfg is what makes this cheap. _refine_segment reads nearly every parameter off cfg,
so handing it a view instead of the shot config makes it per-track adaptive without a line
changing inside it. And when a track has no overrides, view() returns the shot config
ITSELF -- so with the feature off the behaviour is not merely similar to the old path, it
is the same object taking the same branches.

No cv2 and no torch here: this is a data structure, and the seed measurements arrive
already computed from tracker_core.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

# Ids gain a suffix when a track is split: "_b", "_c" ... at the refine stage and "_f1",
# "_f2" ... at defragment. If a future split site forgets to call derive(), get() strips
# these and retries the parent rather than silently losing the metadata.
_SPLIT_SUFFIX = re.compile(r"(_[b-z]|_f\d+)$")


@dataclass
class SeedFeat:
    """What the frame looked like where this track was seeded.

    lmin/lmax are the structure-tensor eigenvalues cv2.cornerEigenValsAndVecs already
    produces at every seed; theta is the eigenvector angle, which says which direction an
    edge-like feature is UNCONSTRAINED in. scale_px is the block size at which the corner
    response peaks -- the feature's own natural size, and the number that should be setting
    the pattern box.
    """
    lmin: float = 0.0
    lmax: float = 0.0
    aniso: float = 0.0          # lmin/lmax: 1.0 = isotropic corner, ~0 = pure edge
    theta: float = 0.0          # radians, direction of the weaker eigenvector
    scale_px: float = 0.0       # block size where cornerMinEigenVal response peaks
    local_density: float = 0.0  # fraction of strong neighbours: rival-peak risk
    local_contrast: float = 0.0 # patch std


@dataclass
class SeedStats:
    """The frame's own distribution, so every threshold is relative.

    Absolute cornerness thresholds do not survive a change of exposure, codec or plate --
    shot_profile learned this already and works in percentiles for the same reason.
    """
    lmin_p25: float = 0.0
    lmin_p75: float = 0.0
    density_p90: float = 1.0
    scale_med: float = 0.0


@dataclass
class TrackMeta:
    """Everything known about one track, from seed to export."""
    # identity
    tid: str = ""
    seed_frame: int = 0
    seed_xy: Tuple[float, float] = (0.0, 0.0)
    pass_name: str = ""
    parent: Optional[str] = None

    # what it was seeded on
    feat: SeedFeat = field(default_factory=SeedFeat)
    kind: str = ""              # corner | blob | edge | dense-*  (see classify_seed)

    # what the AI said about it (phase 2; empty until then)
    sam_obj: int = -1           # SAM3 instance id at the seed, 0 = background/static world
    depth_class: str = ""       # fg | mg | bg
    risk: str = ""              # glass | water | reflection | screen | mirror
    rigid: Optional[bool] = None

    # how it should be tracked: RunnerConfig field name -> value
    policy: Dict[str, Any] = field(default_factory=dict)

    # measured during tracking
    certainty: float = 0.0
    escalation: int = 0
    geo_residual: float = float("nan")
    reasons: List[str] = field(default_factory=list)

    def note(self, why: str) -> None:
        if why not in self.reasons:
            self.reasons.append(why)


def classify_seed(feat: SeedFeat, stats: SeedStats, min_aniso: float = 0.08) -> str:
    """Name what this seed is sitting on.

    The split that matters is corner vs edge vs blob, because each fails differently. A
    corner is pinned in both directions and wants a small tight box. An edge is pinned in
    one, so NCC slides it along -- it needs a stricter ambiguity test, not a bigger box. A
    blob is pinned weakly in both, and a box sized for a corner sees mostly featureless
    pixels, so it needs a LARGER one.

    "dense-" prefixes any of those when the neighbourhood is full of near-identical rivals
    (bolts, rivets, window grids, tiles) -- the case match_ambiguity_ratio exists for.
    """
    if feat.aniso < min_aniso or feat.lmax <= 0.0:
        return "flat"
    if feat.aniso < 0.25:
        kind = "edge"
    elif feat.lmin >= stats.lmin_p75:
        kind = "corner"
    elif feat.scale_px > max(stats.scale_med, 1.0):
        kind = "blob"
    else:
        kind = "corner"
    if feat.local_density > stats.density_p90:
        kind = "dense-" + kind
    return kind


def _odd(v: float, lo: int, hi: int) -> int:
    """Pattern boxes must be odd (they are centred on the point)."""
    n = int(round(v))
    n = max(lo, min(hi, n))
    return n if n % 2 == 1 else n + 1


def policy_for(kind: str, feat: SeedFeat, base) -> Dict[str, Any]:
    """Map a seed class to the parameters that class actually wants.

    Only fields that DIFFER from the shot config are returned: an empty dict means this
    track is ordinary and should run on the shot settings untouched.
    """
    pol: Dict[str, Any] = {}
    base_patch = int(getattr(base, "refine_patch_px", 31))
    scale = float(feat.scale_px) if feat.scale_px > 0 else 0.0
    core = kind.split("-")[-1]

    if core == "corner":
        # Tight box: a corner is fully constrained, and every extra pixel of box is
        # background that dilutes the correlation peak certainty is read off.
        if scale > 0:
            pol["refine_patch_px"] = _odd(2.0 * scale + 1, 21, 41)

    elif core == "blob":
        # A soft feature never reaches high NCC, so judging it by a corner's bar throws
        # away a track that was tracking fine. Bigger box, more template averaging, and
        # euclidean motion: the global affine->translation revert was about affine's extra
        # freedom adding wobble on translation-dominant plates, which is an argument about
        # corners. A blob has no orientation for the extra DoF to corrupt.
        pol["refine_patch_px"] = _odd(3.0 * scale + 1 if scale > 0 else base_patch * 1.5, 31, 61)
        pol["refine_motion"] = "euclidean"
        pol["refine_ncc_lost"] = round(float(getattr(base, "refine_ncc_lost", 0.60)) - 0.05, 3)
        pol["template_frames"] = int(getattr(base, "template_frames", 5)) + 2

    elif core == "edge":
        # An edge cannot be pinned along its own direction, so a wider search or a looser
        # threshold makes it slide FASTER. Tighten instead: demand a more distinctive match
        # and a tighter forward-backward return, and raise the bar to keep it at all.
        if scale > 0:
            pol["refine_patch_px"] = _odd(2.0 * scale + 1, 25, 45)
        pol["refine_ncc_lost"] = round(float(getattr(base, "refine_ncc_lost", 0.60)) + 0.05, 3)
        pol["match_ambiguity_ratio"] = min(
            0.80, float(getattr(base, "match_ambiguity_ratio", 0.90)))
        fb = float(getattr(base, "refine_fb_max_px", 0.0) or 0.0)
        if fb > 0.0:
            pol["refine_fb_max_px"] = min(0.8, fb)

    if kind.startswith("dense-"):
        # Repetitive detail: the rival peak is the whole danger. Reject harder, and refuse
        # to widen the search box, because a wider box is what lets the rival in.
        pol["match_ambiguity_ratio"] = min(
            0.75, float(pol.get("match_ambiguity_ratio",
                                getattr(base, "match_ambiguity_ratio", 0.90))))
        pol["refine_search_max"] = int(getattr(base, "refine_search_px", 24))

    return pol


class _TrackCfg:
    """Read-through view of the shot config with per-track overrides.

    __getattr__ fires only when normal lookup fails, so an override wins and anything else
    falls through to the shot config -- including raising AttributeError for a name neither
    has, which is what makes the many getattr(cfg, name, default) reads in pattern_refine
    still land on their defaults.
    """
    __slots__ = ("_base", "_over")

    def __init__(self, base, over: Dict[str, Any]):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_over", dict(over))

    def __getattr__(self, name: str):
        over = object.__getattribute__(self, "_over")
        if name in over:
            return over[name]
        return getattr(object.__getattribute__(self, "_base"), name)

    def __setattr__(self, name: str, value) -> None:
        # Writes land on the view, never on the shot config -- one track must not be able
        # to change the settings of every other track in the shot.
        object.__getattribute__(self, "_over")[name] = value

    def __repr__(self) -> str:
        return f"_TrackCfg({object.__getattribute__(self, '_over')})"


class TrackRegistry:
    """One per shot: track id -> TrackMeta, and the id-mutation hooks."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._meta: Dict[str, TrackMeta] = {}
        self.orphans: int = 0        # ids resolved only via the parent fallback

    # -- population ------------------------------------------------------------------
    def register(self, tid: str, meta: TrackMeta) -> TrackMeta:
        meta.tid = tid
        self._meta[tid] = meta
        return meta

    def derive(self, parent: str, child: str) -> Optional[TrackMeta]:
        """A split produced a new id. The piece is the same feature: carry the meta over."""
        if child == parent:
            return self._meta.get(parent)
        src = self.get(parent)
        if src is None:
            return None
        meta = replace(src, tid=child, parent=parent,
                       policy=dict(src.policy), reasons=list(src.reasons))
        self._meta[child] = meta
        return meta

    # -- lookup ----------------------------------------------------------------------
    def get(self, tid: str) -> Optional[TrackMeta]:
        m = self._meta.get(tid)
        if m is not None:
            return m
        # Fallback for a split site that forgot to call derive(). Better a slightly stale
        # parent record than a silently empty one -- and orphans is counted so it shows up.
        stem = tid
        while True:
            stripped = _SPLIT_SUFFIX.sub("", stem)
            if stripped == stem:
                return None
            stem = stripped
            m = self._meta.get(stem)
            if m is not None:
                self.orphans += 1
                return m

    def view(self, tid: str, base):
        """The config this track should be tracked with.

        Returns the shot config ITSELF when the track has no overrides, so the untouched
        path is not just equivalent to the old one, it is identical.
        """
        if not self.enabled:
            return base
        m = self.get(tid)
        if m is None or not m.policy:
            return base
        return _TrackCfg(base, m.policy)

    # -- reporting -------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._meta)

    def summary(self) -> str:
        if not self._meta:
            return "no per-track metadata"
        kinds: Dict[str, int] = {}
        tuned = 0
        for m in self._meta.values():
            kinds[m.kind or "?"] = kinds.get(m.kind or "?", 0) + 1
            if m.policy:
                tuned += 1
        bits = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        out = f"seed classes: {bits} ({tuned}/{len(self._meta)} tuned)"
        if self.orphans:
            out += f" [{self.orphans} id lookups needed the parent fallback]"
        return out

    def row(self, tid: str) -> Dict[str, Any]:
        """Flat fields for the track report CSV."""
        m = self.get(tid)
        if m is None:
            return {}
        return {
            "kind": m.kind,
            "patch_px": m.policy.get("refine_patch_px", ""),
            "motion": m.policy.get("refine_motion", ""),
            "sam_obj": m.sam_obj if m.sam_obj >= 0 else "",
            "depth_class": m.depth_class,
            "risk": m.risk,
            "escalation": m.escalation,
            "geo_residual": ("" if m.geo_residual != m.geo_residual
                             else f"{m.geo_residual:.3f}"),
        }

    def dump(self, path: str) -> None:
        """Full record beside the CSV, for offline analysis of a whole batch."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({tid: asdict(m) for tid, m in self._meta.items()}, f, indent=1)
