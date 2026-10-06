"""Physics objective terms for ``ObjectiveConfig.physics``."""
from __future__ import annotations

import torch

from ..occupancy.field import OccupancyGrid
from .fem import FixedGridFEM


class ComplianceTerm:
    """Compliance of the explicit solid on a fixed FEM grid, normalized by its first value.

        control points -> proxy surface -> soft occupancy rho (per cell) -> SIMP moduli -> C = f . u

    ``grid`` and ``fem`` must describe the same cells (same lo, h, dims).
    """

    name = "compliance"

    def __init__(self, fem: FixedGridFEM, grid: OccupancyGrid, weight: float = 1.0, eps: float | None = None,
                 normalize: bool = True):
        if tuple(grid.dims) != tuple(fem.dims) or abs(grid.h - fem.h) > 1e-12:
            raise ValueError("occupancy grid and FEM grid differ")
        self.fem, self.grid, self.weight, self.eps = fem, grid, float(weight), eps
        self.normalize = normalize
        self.C0: float | None = None
        self.last: dict = {}

    def density(self, terms: dict, disc) -> torch.Tensor:
        return self.grid.occupancy(terms["X"], disc.sampler.tri_t, eps=self.eps).reshape(-1)

    def __call__(self, state, P, disc, terms) -> torch.Tensor:
        rho = self.density(terms, disc)
        C = self.fem.compliance(rho)
        if self.C0 is None:
            self.C0 = float(C.detach())
        self.last = {"compliance": float(C.detach()), "material": float(rho.detach().mean()),
                     **getattr(self.fem, "last_solve", {})}
        return C / self.C0 if self.normalize else C


def fem_on_grid(grid: OccupancyGrid, supports, loads, cfg=None) -> FixedGridFEM:
    """FEM over exactly the cells of an occupancy grid."""
    return FixedGridFEM(grid.lo.tolist(), grid.h, grid.dims, supports, loads, cfg)
