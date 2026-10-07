"""CSG solids in a canonical two-level form, and sampling grids.

Canonical expression
--------------------
    S = (M_1 u M_2 u ... u M_m) \\ (V_1 u V_2 u ... u V_v)

``M_i`` are *material* features, ``V_j`` *void* features, each one primitive.
With ``d_X`` the signed distance of X (negative inside):

    union        d_{A u B}  = min(d_A, d_B)
    difference   d_{A \\ B} = max(d_A, -d_B)
    S            d_S        = max(min_i d_{M_i}, -min_j d_{V_j})

Hard occupancy is ``d_S < 0``; soft occupancy is ``sigmoid(-d_S / eps)`` with
``eps`` in world units (so it does not depend on the sampling resolution).
``d_S`` is exact outside unions and a lower bound on |distance| inside
differences (the usual CSG caveat); signs are always exact.

The flat form keeps edits local and the expression size equal to the number of
features: every rewrite adds or removes features, it never nests expressions.
Its limitation: material cannot be placed *inside* a void feature (voids are
subtracted last), so nested shells (a ball floating in a cavity) are outside
the representation. ``canonicalize`` drops features that no longer change the
hard occupancy on a check grid, so dead features cannot accumulate.

Features carry a ``kind`` (body, bridge, plug, cavity, channel, slab) and a
provenance record. Kinds are bookkeeping for the grammar (which features an
inverse edit may target); topology is always *measured*, never read from kinds.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from .primitives import DTYPE, Primitive

MATERIAL, VOID = "material", "void"


@dataclass
class Feature:
    id: int
    role: str          # MATERIAL | VOID
    kind: str          # body | bridge | plug | cavity | channel | slab
    prim: Primitive
    provenance: dict = field(default_factory=dict)

    def copy(self) -> "Feature":
        return Feature(self.id, self.role, self.kind, self.prim.copy(), copy.deepcopy(self.provenance))


class CSGSolid:
    def __init__(self, features: list[Feature] | None = None, next_id: int | None = None):
        self.features: list[Feature] = list(features or [])
        self.next_id = next_id if next_id is not None else (max((f.id for f in self.features), default=-1) + 1)

    # -- construction -----------------------------------------------------
    def add(self, role: str, kind: str, prim: Primitive, **provenance) -> Feature:
        f = Feature(self.next_id, role, kind, prim, dict(provenance))
        self.next_id += 1
        self.features.append(f)
        return f

    def remove(self, fid: int) -> Feature:
        f = self.get(fid)
        self.features.remove(f)
        return f

    def get(self, fid: int) -> Feature:
        for f in self.features:
            if f.id == fid:
                return f
        raise KeyError(fid)

    def has(self, fid: int) -> bool:
        return any(f.id == fid for f in self.features)

    def copy(self) -> "CSGSolid":
        return CSGSolid([f.copy() for f in self.features], self.next_id)

    @property
    def material(self) -> list[Feature]:
        return [f for f in self.features if f.role == MATERIAL]

    @property
    def voids(self) -> list[Feature]:
        return [f for f in self.features if f.role == VOID]

    def complexity(self) -> float:
        """Representation complexity: one unit per feature (each is one primitive)."""
        return float(len(self.features))

    # -- evaluation -------------------------------------------------------
    def sdf(self, X: torch.Tensor) -> torch.Tensor:
        mats = self.material
        if not mats:
            return torch.full((len(X),), 1e3, dtype=DTYPE)
        d = torch.stack([f.prim.sdf(X) for f in mats]).min(dim=0).values
        if self.voids:
            v = torch.stack([f.prim.sdf(X) for f in self.voids]).min(dim=0).values
            d = torch.maximum(d, -v)
        return d

    def occupancy(self, X: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sigmoid(-self.sdf(X) / eps)

    def hard(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.sdf(X) < 0

    def parameters(self) -> list[torch.Tensor]:
        return [t for f in self.features for t in f.prim.tensors()]

    def project_(self, r_min: float) -> None:
        for f in self.features:
            f.prim.project_(r_min)

    def describe(self) -> list[dict]:
        return [{"id": f.id, "role": f.role, "kind": f.kind, **f.prim.describe(), "provenance": f.provenance}
                for f in self.features]

    # -- canonical form ---------------------------------------------------
    def canonicalize(self, grid: "Grid", tol_cells: int = 0) -> list[int]:
        """Drop features whose removal changes at most ``tol_cells`` cells of the hard occupancy
        (dead voids outside material, bodies swallowed by others or by voids). Returns removed ids."""
        X = grid.points
        removed = []
        base = self.hard(X)
        for f in list(self.features):
            if f.role == MATERIAL and len(self.material) == 1:
                continue
            trial = CSGSolid([g for g in self.features if g.id != f.id], self.next_id)
            if int((trial.hard(X) != base).sum()) <= tol_cells:
                self.features.remove(f)
                removed.append(f.id)
        return removed


class Grid:
    """Cell-center samples of the box [lo, hi]^3 with n cells per axis (optionally offset)."""

    def __init__(self, n: int, lo: float = -1.2, hi: float = 1.2, offset: float = 0.0):
        self.n, self.lo, self.hi = int(n), float(lo), float(hi)
        self.h = (self.hi - self.lo) / self.n
        self.offset = float(offset)
        ax = self.lo + (np.arange(self.n) + 0.5 + offset) * self.h
        self.axis = ax
        g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
        self.points = torch.as_tensor(g, dtype=DTYPE)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n, self.n, self.n)

    @property
    def cell_volume(self) -> float:
        return self.h ** 3

    def reshape(self, v) -> np.ndarray:
        v = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        return v.reshape(self.shape)

    def world(self, ijk) -> np.ndarray:
        """World coordinates of (fractional) cell indices."""
        return self.lo + (np.asarray(ijk, float) + 0.5 + self.offset) * self.h
