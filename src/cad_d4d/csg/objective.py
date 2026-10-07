"""Objective and continuous fitting for CSG reconstructions.

    F = w_volume * L_volume + w_surface * L_surface + lambda_complex * C

    L_volume  = mean_cells w * (rho - rho_target)^2
    L_surface = sum_cells delta * ((phi_target - d) / L_ref)^2 / sum_cells delta,   delta = rho (1 - rho)
    C         = number of features (primitives)

* ``rho = sigmoid(-d / eps)`` is the soft occupancy of the reconstruction and
  ``rho_target`` the same smoothing of the target SDF ``phi_target``, so an exact
  reconstruction has zero loss. ``eps`` is in world units (default 0.03 =
  1/80 of the domain width), not in cells.
* ``L_volume`` is a mean over grid cells, i.e. (mismatched volume) / (domain
  volume) up to smoothing: the same number at any grid resolution. Weights
  ``w`` are 1 (uniform).
* ``L_surface`` compares the target SDF with the reconstruction's SDF ``d`` on the
  reconstruction's soft surface band (``delta`` peaks at its zero level set, where
  the term is the target distance of the surface). It vanishes for an exact
  reconstruction (comparing ``phi_target`` alone would not: the band has width),
  it is a weighted mean, so it is resolution independent, and ``L_ref = 0.1``
  makes it dimensionless.
* ``C`` counts primitives; ``lambda_complex = 1e-3`` means a feature must
  explain about 0.1% of the domain volume to pay for itself.

Defaults: ``w_volume = 1``, ``w_surface = 0.05``. Continuous fitting uses Adam
with a fixed step count, then projects every primitive back to its valid set
(minimum size ``r_min``). The same optimizer and step budget are used for every
topology trial and for the no-edit baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .primitives import DTYPE
from .solid import CSGSolid, Grid


@dataclass
class ObjectiveConfig:
    eps: float = 0.03
    w_volume: float = 1.0
    w_surface: float = 0.05
    l_ref: float = 0.1
    lambda_complex: float = 1e-3


@dataclass
class FitConfig:
    lr: float = 0.02
    lr_final: float = 0.004   # linear decay within one fit call
    r_min: float = 0.06       # minimum primitive radius / half-extent (world units)


class Target:
    """A target solid given by an SDF (negative inside). ``recipe`` (optional) builds the
    correct CSG structure; it is used ONLY by the labeled oracle baseline."""

    def __init__(self, name: str, sdf_fn, params: dict | None = None, recipe=None, family: str = ""):
        self.name, self.sdf_fn, self.params, self.recipe, self.family = name, sdf_fn, params or {}, recipe, family

    def sdf(self, X: torch.Tensor) -> torch.Tensor:
        return self.sdf_fn(X)


class TopoObjective:
    def __init__(self, target: Target, grid: Grid, cfg: ObjectiveConfig | None = None):
        self.target, self.grid, self.cfg = target, grid, cfg or ObjectiveConfig()
        with torch.no_grad():
            self.phi_t = target.sdf(grid.points).to(DTYPE)
        self.rho_t = torch.sigmoid(-self.phi_t / self.cfg.eps)

    def terms(self, solid: CSGSolid) -> dict:
        cfg = self.cfg
        d = solid.sdf(self.grid.points)
        rho = torch.sigmoid(-d / cfg.eps)
        L_vol = ((rho - self.rho_t) ** 2).mean()
        delta = rho * (1.0 - rho)
        L_surf = (delta * ((self.phi_t - d) / cfg.l_ref) ** 2).sum() / delta.sum().clamp(min=1e-12)
        C = solid.complexity()
        total = cfg.w_volume * L_vol + cfg.w_surface * L_surf + cfg.lambda_complex * C
        return {"total": total, "volume": L_vol, "surface": L_surf, "complexity": C, "rho": rho, "d": d}

    def value(self, solid: CSGSolid) -> dict:
        with torch.no_grad():
            t = self.terms(solid)
        return {k: (float(v) if k not in ("rho", "d") else v) for k, v in t.items()}


def fit(solid: CSGSolid, objective: TopoObjective, steps: int, cfg: FitConfig | None = None) -> dict:
    """Optimize ``solid``'s primitive parameters in place for exactly ``steps`` Adam steps."""
    cfg = cfg or FitConfig()
    params = solid.parameters()
    if steps <= 0 or not params:
        return {"steps": 0, **{k: v for k, v in objective.value(solid).items() if k not in ("rho", "d")}}
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.lr)
    for it in range(steps):
        for g in opt.param_groups:
            g["lr"] = cfg.lr + (cfg.lr_final - cfg.lr) * it / max(steps - 1, 1)
        opt.zero_grad()
        objective.terms(solid)["total"].backward()
        opt.step()
        solid.project_(cfg.r_min)
    for p in params:
        p.requires_grad_(False)
        p.grad = None
    return {"steps": steps, **{k: v for k, v in objective.value(solid).items() if k not in ("rho", "d")}}
