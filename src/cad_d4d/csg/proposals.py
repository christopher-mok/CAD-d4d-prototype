"""Residual-guided topology proposals.

Evidence (``residuals.analyze``) -> a bounded shortlist of grammar operations.
The no-edit baseline (pure boundary deformation) always competes in the trials,
so a residual that deformation can explain is not forced into an edit.

Missing-material region R
  * inside an existing void feature (>= ``unsupported`` of that feature's carved
    volume) -> RemoveCavity(feature) if the void it carves is enclosed, else
    CloseTunnel(feature)
  * touches >= 2 material components, or one component in >= 2 separate
    patches -> BridgeBodies between the two largest contact patches
  * otherwise -> AddBody (sphere at the inscribed center, and an oriented box from
    the region's principal axes)
Excess-material region R
  * holds >= ``unsupported`` of a material feature's exclusive volume -> RemoveBody
  * removing it would split the material -> PinchBody (slab normal = R's thinnest axis)
  * no contact with any void -> AddCavity (sphere and box)
  * touches a cavity and the exterior -> OpenCavity
  * touches the exterior in >= 2 separate patches -> BridgeVoid between them
Sizes come from R (inscribed radius, PCA half-extents) at ``scales``; nothing
reads target topology, labels or construction history.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage

from . import grammar as G
from .measure import S26, voxel_topology
from .primitives import DTYPE
from .residuals import Evidence, Region
from .solid import VOID, CSGSolid, Grid


@dataclass
class ProposalConfig:
    max_candidates: int = 8
    scales: tuple = (0.8, 1.15)
    unsupported: float = 0.5
    r_min: float = 0.06


@dataclass
class Candidate:
    op: G.TopoOp
    priority: float
    evidence: dict = field(default_factory=dict)


def _sdf(solid: CSGSolid, P) -> np.ndarray:
    with torch.no_grad():
        return solid.sdf(torch.as_tensor(np.atleast_2d(P), dtype=DTYPE)).cpu().numpy()


def _push_out(solid: CSGSolid, p, direction, step: float, max_steps: int = 40):
    """Move p along ``direction`` until it leaves material (for void-channel endpoints)."""
    d = np.asarray(direction, float)
    d /= max(np.linalg.norm(d), 1e-12)
    p = np.asarray(p, float).copy()
    for _ in range(max_steps):
        if _sdf(solid, p)[0] > step:
            return p
        p += step * d
    return p


def _pull_in(solid: CSGSolid, p, direction, step: float, max_steps: int = 40):
    """Move p along ``direction`` until it is inside material (for bridge endpoints)."""
    d = np.asarray(direction, float)
    d /= max(np.linalg.norm(d), 1e-12)
    p = np.asarray(p, float).copy()
    for _ in range(max_steps):
        if _sdf(solid, p)[0] < 0:
            return p
        p += step * d
    return p


def _n_components(occ: np.ndarray, min_voxels: int = 8) -> int:
    lab, n = ndimage.label(occ, structure=S26)
    return int(np.sum(np.bincount(lab.ravel(), minlength=n + 1)[1:] >= min_voxels)) if n else 0


def propose(solid: CSGSolid, ev: Evidence, grid: Grid, cfg: ProposalConfig | None = None) -> list[Candidate]:
    cfg = cfg or ProposalConfig()
    out: list[Candidate] = []
    n_tunnels = voxel_topology(ev.hard)["tunnels"] if ev.add else 0

    def add(op, region: Region, rule: str, weight: float = 1.0):
        out.append(Candidate(op, weight * region.mass, {"region": region.describe(), "rule": rule}))

    def r_of(x):
        return float(max(cfg.r_min, x))

    for R in ev.add:
        # missing material inside a void feature: that feature is unsupported
        for fid, frac in sorted(R.feature_overlap.items(), key=lambda kv: -kv[1]):
            if frac < cfg.unsupported or not solid.has(fid) or solid.get(fid).role != VOID:
                continue
            carved = ev.feature_masks[fid] & ~ev.hard
            labs = set(np.unique(ev.void_labels[carved]).tolist()) - {0}
            enclosed = bool(labs) and labs <= ev.cavity_labels
            op = G.RemoveCavity(fid=fid) if enclosed else G.CloseTunnel(fid=fid)
            add(op, R, "missing material inside void feature", 2.0)
        # missing material in a cavity / hole formed by material arrangement (no void feature to remove)
        labs = set(np.unique(ev.void_labels[R.mask]).tolist()) - {0}
        half = np.maximum(R.half_extents, cfg.r_min)
        if labs and labs <= ev.cavity_labels:
            add(G.RemoveCavity(point=R.c_in, axes=R.axes, half_extents=half * 1.1), R,
                "missing material filling an enclosed void", 1.5)
        elif n_tunnels > 0 and R.half_extents[2] < 0.5 * R.half_extents[1] and len(R.material_contacts) == 1:
            add(G.CloseTunnel(center=R.centroid, axes=R.axes, half_extents=half), R,
                "missing material spanning a hole (thin region with one rim contact)", 1.5)
        comps = {}
        for p in R.material_contacts:
            comps.setdefault(p.label, []).append(p)
        patches = sorted(R.material_contacts, key=lambda p: -p.voxels)
        pair = None
        if len(comps) >= 2:
            a, b = (max(v, key=lambda p: p.voxels) for v in sorted(comps.values(), key=lambda v: -sum(p.voxels for p in v))[:2])
            pair = (a, b)
        elif len(patches) >= 2:
            pair = (patches[0], patches[1])
        if pair is not None:
            a, b = pair
            ab = b.centroid - a.centroid
            pa = _pull_in(solid, a.centroid, -ab, grid.h)
            pb = _pull_in(solid, b.centroid, ab, grid.h)
            for s in cfg.scales:
                add(G.BridgeBodies(pa, pb, r_of(s * R.r_in)), R, "missing material between material patches", 1.5)
        if R.r_in >= cfg.r_min:
            for s in cfg.scales:
                add(G.AddBody(R.c_in, radius=r_of(s * R.r_in)), R, "missing material region (sphere)")
            add(G.AddBody(R.centroid, axes=R.axes, half_extents=np.maximum(R.half_extents, cfg.r_min)), R,
                "missing material region (box)")

    for R in ev.remove:
        for fid, frac in sorted(R.feature_overlap.items(), key=lambda kv: -kv[1]):
            if frac >= cfg.unsupported and solid.has(fid):
                add(G.RemoveBody(fid), R, "excess material holds most of a body", 2.0)
        if R.disconnects:
            # slab normals: the principal axes whose slab (on the voxel grid) actually separates material
            w = float(np.max(R.half_extents)) * 1.5 + grid.h
            P = grid.points.numpy()
            base = _n_components(ev.hard)
            for k in range(3):
                n = R.axes[:, k]
                for s in cfg.scales:
                    t = r_of(s * R.r_in)
                    rel = P - R.centroid
                    along = rel @ n
                    across = np.linalg.norm(rel - along[:, None] * n, axis=1)
                    slab = ((np.abs(along) < t) & (across < w)).reshape(grid.shape)
                    if _n_components(ev.hard & ~slab) > base:
                        add(G.PinchBody(R.centroid, n, t, w), R, "excess material in a neck", 1.5)
                        # split variant: the material feature that spans the neck, if one does
                        for fid, mask in ev.feature_masks.items():
                            if solid.has(fid) and solid.get(fid).role != VOID and _n_components(mask & ~slab) >= 2:
                                add(G.PinchBody(R.centroid, n, t, w, split=fid), R,
                                    "excess material in a neck (split the spanning body)", 1.6)
        enclosed = not R.exterior_contacts and not R.cavity_contacts
        if enclosed:
            for s in cfg.scales:
                add(G.AddCavity(R.c_in, radius=r_of(s * R.r_in)), R, "enclosed excess material (sphere)", 1.5)
            add(G.AddCavity(R.centroid, axes=R.axes, half_extents=np.maximum(R.half_extents, cfg.r_min)), R,
                "enclosed excess material (box)", 1.5)
        if R.cavity_contacts and R.exterior_contacts:
            c = max(R.cavity_contacts, key=lambda p: p.voxels).centroid
            e = max(R.exterior_contacts, key=lambda p: p.voxels).centroid
            for s in cfg.scales:
                add(G.OpenCavity(c, _push_out(solid, e, e - c, grid.h), r_of(s * R.r_in)), R,
                    "excess material between a cavity and the exterior", 1.5)
        if len(R.exterior_contacts) >= 2:
            ps = sorted(R.exterior_contacts, key=lambda p: -p.voxels)[:2]
            a, b = ps[0].centroid, ps[1].centroid
            pa = _push_out(solid, a, a - b, grid.h)
            pb = _push_out(solid, b, b - a, grid.h)
            for s in cfg.scales:
                add(G.BridgeVoid(pa, pb, r_of(s * R.r_in)), R, "excess material between exterior patches", 1.5)

    out.sort(key=lambda c: -c.priority)
    seen, unique = set(), []
    for c in out:
        key = repr(c.op)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[: cfg.max_candidates]
