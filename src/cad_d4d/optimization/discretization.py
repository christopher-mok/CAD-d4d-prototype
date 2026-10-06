"""Per-structure evaluation operators (cached on the state until a rewrite).

All operators are uploaded to the compute device once per structure:

* ``sampler``   loss quadrature proxy (SDF term, area weights, mass matrix)
* ``coverage``  finer proxy used only for the coverage distance (reduces the
                chordal bias of point-to-triangle distances)
* ``check``     coarse proxy for Jacobian / self-intersection validity checks
* ``F``         fairness (bending + split-line crease) quadratic form
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..device import get_device
from ..geometry.state import CADState
from ..geometry.tessellation import SamplingConfig, SurfaceSampler, _torch_sparse
from ..geometry.validity import self_intersections
from ..losses.fairness import fairness_operator


@dataclass
class DiscretizationConfig:
    loss: SamplingConfig = field(default_factory=lambda: SamplingConfig(min_res=17, per_span=4, max_res=33))
    coverage: SamplingConfig | None = field(default_factory=lambda: SamplingConfig(min_res=25, per_span=6, max_res=49))
    check: SamplingConfig = field(default_factory=lambda: SamplingConfig(min_res=9, per_span=3, max_res=17))
    fairness: str = "bending"  # "bending" (refinement invariant) | "control_net"
    # C1 penalty across split lines, relative to the bending energy. Off by default: it keeps
    # unused splits mergeable and runs more consistent, but in multi-seed sweeps it cost
    # compactness and median fit (see README). 1e4 is a reasonable value when enabled.
    crease_weight: float = 0.0


class Discretization:
    """Operators are built lazily: scoring a candidate rewrite needs the loss and
    coverage proxies and the fairness form, but not the validity-check proxy or
    the face adjacency."""

    def __init__(self, state: CADState, cfg: DiscretizationConfig):
        self.cfg = cfg
        self._state = state
        self.sampler = SurfaceSampler(state, cfg.loss)
        self.coverage = SurfaceSampler(state, cfg.coverage) if cfg.coverage is not None else self.sampler
        F = fairness_operator(state, cfg.fairness, cfg.crease_weight)
        self.F_np = F
        self.F = _torch_sparse(F)
        self.G_abs = torch.abs(self.sampler.G)
        self._check = None
        self._adjacency = None

    @property
    def check(self) -> SurfaceSampler:
        if self._check is None:
            self._check = SurfaceSampler(self._state, self.cfg.check)
        return self._check

    @property
    def face_adjacency(self) -> torch.Tensor:
        if self._adjacency is None:
            order = self.sampler.face_order
            verts = [self._state.cx.face_vertices(fid) for fid in order]
            n = len(order)
            adj = np.zeros((n, n), bool)
            for a in range(n):
                for b in range(a + 1, n):
                    if verts[a] & verts[b]:
                        adj[a, b] = adj[b, a] = True
            self._adjacency = torch.as_tensor(adj, device=get_device())
        return self._adjacency

    def count_self_intersections(self, X_check: torch.Tensor, return_hits: bool = False):
        c = self.check
        return self_intersections(X_check, c.tri_t, c.tri_face_t, c.tri_cell_t, self.face_adjacency,
                                  c.sample_face_t, c.sample_boundary_t, return_hits=return_hits)


def get_discretization(state: CADState, cfg: DiscretizationConfig | None = None) -> Discretization:
    cfg = cfg or DiscretizationConfig()
    key = ("disc", repr(cfg))
    disc = state.cache.get(key)
    if disc is None:
        disc = Discretization(state, cfg)
        state.cache[key] = disc
    return disc
