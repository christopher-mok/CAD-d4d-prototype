"""Volumetric residual analysis: where is material missing or in excess, and what touches it.

With rho the current soft occupancy and rho_t the target's (same smoothing):

    r_add    = max(rho_t - rho, 0)     missing material
    r_remove = max(rho - rho_t, 0)     excess material

Cells with r > ``threshold`` (default 0.5: the hard occupancies disagree) are
grouped into 6-connected regions; regions smaller than ``min_voxels`` are
dropped. Per region we estimate size (volume, residual mass), position
(centroid, the inscribed-ball center = distance-transform maximum),
principal directions and half-extents (PCA; a solid ellipsoid with variance
s^2 along an axis has half-extent sqrt(5) s), and adjacency evidence measured
on the *current* hard occupancy, with the conventions of ``measure.py``
(material 26-connected, void 6-connected):

* ``material_contacts``  contact patches with current material, per material
  component (add regions: one component touched in two separate patches, or
  two components, is bridge evidence);
* ``exterior_contacts``  contact patches with the exterior void (remove regions:
  none = enclosed -> cavity evidence; two or more separate patches = channel
  evidence);
* ``cavity_contacts``    contact with existing enclosed voids;
* ``disconnects``        removing the region would split current material;
* ``feature_overlap``    share of each feature's exclusive volume inside the region (material:
  cells only it fills; void: cells only it carves out of material).

Nothing here reads the target's topology: only rho_t on the grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage

from .measure import S6, S26
from .solid import MATERIAL, VOID, CSGSolid, Grid


@dataclass
class ResidualConfig:
    threshold: float = 0.5
    min_voxels: int = 6
    max_regions: int = 8      # per sign, largest residual mass first


@dataclass
class Patch:
    centroid: np.ndarray
    voxels: int
    label: int = 0            # component / cavity label it belongs to


@dataclass
class Region:
    id: int
    sign: str                 # "add" | "remove"
    voxels: int
    volume: float
    mass: float               # sum of residual * cell volume
    centroid: np.ndarray
    axes: np.ndarray          # columns: principal directions, largest extent first
    half_extents: np.ndarray
    r_in: float               # inscribed radius (world units)
    c_in: np.ndarray          # inscribed-ball center
    material_contacts: list = field(default_factory=list)
    exterior_contacts: list = field(default_factory=list)
    cavity_contacts: list = field(default_factory=list)
    disconnects: bool = False
    feature_overlap: dict = field(default_factory=dict)
    mask: np.ndarray | None = None

    @property
    def n_material_components(self) -> int:
        return len({p.label for p in self.material_contacts})

    def describe(self) -> dict:
        return {"id": self.id, "sign": self.sign, "voxels": self.voxels, "volume": round(self.volume, 5),
                "mass": round(self.mass, 5), "centroid": np.round(self.centroid, 3).tolist(),
                "half_extents": np.round(self.half_extents, 3).tolist(), "r_in": round(self.r_in, 4),
                "material_contacts": [(p.label, p.voxels) for p in self.material_contacts],
                "exterior_contacts": len(self.exterior_contacts), "cavity_contacts": len(self.cavity_contacts),
                "disconnects": self.disconnects,
                "feature_overlap": {k: round(v, 3) for k, v in self.feature_overlap.items()}}


@dataclass
class Evidence:
    add: list
    remove: list
    material_labels: np.ndarray       # current material components (26-conn), 0 = void
    void_labels: np.ndarray           # current void components (6-conn), 0 = material
    exterior: set
    cavity_labels: set
    rho: np.ndarray
    rho_t: np.ndarray
    hard: np.ndarray
    feature_masks: dict               # fid -> own hard occupancy of the feature's primitive
    exclusive: dict                   # fid -> cells where this feature alone decides the occupancy


def _patches(mask: np.ndarray, labels: np.ndarray, grid: Grid) -> list[Patch]:
    lab, n = ndimage.label(mask, structure=S26)
    out = []
    for i in range(1, n + 1):
        idx = np.argwhere(lab == i)
        lv = labels[tuple(idx.T)]
        out.append(Patch(grid.world(idx.mean(0)), len(idx), int(np.bincount(lv).argmax()) if len(lv) else 0))
    return out


def analyze(solid: CSGSolid, rho_t: torch.Tensor, grid: Grid, eps: float, cfg: ResidualConfig | None = None) -> Evidence:
    cfg = cfg or ResidualConfig()
    X = grid.points
    with torch.no_grad():
        d = solid.sdf(X)
        rho = torch.sigmoid(-d / eps)
        own = {f.id: (f.prim.sdf(X) < 0).cpu().numpy().reshape(grid.shape) for f in solid.features}
    rho_np, rt_np = grid.reshape(rho), grid.reshape(rho_t)
    hard = grid.reshape(d < 0)
    pad = 1
    hp = np.pad(hard, pad)
    mlab_p, _ = ndimage.label(hp, structure=S26)
    vlab_p, _ = ndimage.label(~hp, structure=S6)
    exterior = set(np.unique(np.concatenate([vlab_p[0].ravel(), vlab_p[-1].ravel(), vlab_p[:, 0].ravel(),
                                            vlab_p[:, -1].ravel(), vlab_p[:, :, 0].ravel(), vlab_p[:, :, -1].ravel()])).tolist())
    exterior.discard(0)
    mlab = mlab_p[1:-1, 1:-1, 1:-1]
    vlab = vlab_p[1:-1, 1:-1, 1:-1]
    cavities = set(np.unique(vlab).tolist()) - exterior - {0}
    # cells where a single feature decides the occupancy (removing it would flip them)
    exclusive = {}
    mats = [f for f in solid.features if f.role == MATERIAL]
    voids = [f for f in solid.features if f.role == VOID]
    in_void = np.zeros(grid.shape, bool)
    for v in voids:
        in_void |= own[v.id]
    for f in mats:
        others = np.zeros(grid.shape, bool)
        for g in mats:
            if g.id != f.id:
                others |= own[g.id]
        exclusive[f.id] = own[f.id] & ~others & ~in_void
    mat_any = np.zeros(grid.shape, bool)
    for f in mats:
        mat_any |= own[f.id]
    for v in voids:
        others = np.zeros(grid.shape, bool)
        for g in voids:
            if g.id != v.id:
                others |= own[g.id]
        exclusive[v.id] = own[v.id] & mat_any & ~others

    def regions(sign: str) -> list[Region]:
        r = np.clip(rt_np - rho_np, 0, None) if sign == "add" else np.clip(rho_np - rt_np, 0, None)
        lab, n = ndimage.label(r > cfg.threshold, structure=S6)
        if n == 0:
            return []
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        masses = ndimage.sum(r, lab, index=np.arange(1, n + 1)) * grid.cell_volume
        order = [i + 1 for i in np.argsort(-masses) if sizes[i + 1] >= cfg.min_voxels][: cfg.max_regions]
        out = []
        for li in order:
            m = lab == li
            idx = np.argwhere(m)
            P = grid.world(idx)
            c = P.mean(0)
            cov = np.cov((P - c).T) if len(P) > 1 else np.zeros((3, 3))
            ev, evec = np.linalg.eigh(cov + 1e-12 * np.eye(3))
            order_ax = np.argsort(-ev)
            half = np.sqrt(5.0 * np.maximum(ev[order_ax], 0)) + 0.5 * grid.h
            dt = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1, 1:-1]
            imax = np.unravel_index(np.argmax(dt), dt.shape)
            reg = Region(len(out), sign, int(m.sum()), float(m.sum() * grid.cell_volume), float(masses[li - 1]),
                         c, evec[:, order_ax], half, float(dt[imax] * grid.h), grid.world(np.array(imax)), mask=m)
            ring = ndimage.binary_dilation(m, structure=S6) & ~m
            if sign == "add":
                reg.material_contacts = _patches(ring & hard, mlab, grid)
                reg.disconnects = False
                for f in voids:  # share of the volume this void actually carves out of material
                    vol = exclusive[f.id].sum()
                    if vol:
                        reg.feature_overlap[f.id] = float((exclusive[f.id] & m).sum() / vol)
            else:
                vring = ring & ~hard
                ext = vring & np.isin(vlab, list(exterior))
                cav = vring & np.isin(vlab, list(cavities)) if cavities else np.zeros_like(vring)
                reg.exterior_contacts = _patches(ext, vlab, grid)
                reg.cavity_contacts = _patches(cav, vlab, grid)
                before = ndimage.label(hard, structure=S26)[1]
                after_lab, after = ndimage.label(hard & ~m, structure=S26)
                big = np.bincount(after_lab.ravel())[1:] >= cfg.min_voxels
                reg.disconnects = bool(int(big.sum()) > before)
                for f in mats:
                    vol = exclusive[f.id].sum()
                    if vol:
                        reg.feature_overlap[f.id] = float((exclusive[f.id] & m).sum() / vol)
            out.append(reg)
        return out

    return Evidence(regions("add"), regions("remove"), mlab, vlab, exterior, cavities, rho_np, rt_np, hard, own,
                    exclusive)
