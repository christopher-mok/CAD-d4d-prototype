"""Rewrite proposal sampling: uniform (ablation) or residual-guided.

Residual field on proxy samples:

    e(x_s) = |phi_target(x_s)| + c(x_s)

where c splats each target point's coverage distance onto its nearest loss
sample (max). Locations are drawn with probability

    P(s)  proportional to  w_s * (e_s + alpha * ||grad e||_s)

(w_s: area weights; grad e by finite differences on each face's sample grid,
in physical units). Without a shape target (physics-only objectives) the
residual is zero and sampling falls back to area-uniform. Refinement kinds:
LocalRefine (optionally refining its window's carriers), FaceRefine,
KnotInsert, CarrierKnotInsert, SplitFace. Simplification candidates
(KnotRemove -- pre-filtered by invariant I1 --, MergeFace, CarrierKnotRemove)
are enumerated and subsampled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..device import to_numpy
from ..geometry import bspline_basis as bb
from ..losses.coverage import nearest_centroids
from ..geometry.state import CADState
from ..losses.objective import ShapeObjective
from ..rewrites.carrier_knots import CarrierKnotInsert, carrier_knot_remove_candidates
from ..rewrites.knot_insert import KnotInsert
from ..rewrites.knot_remove import knot_remove_candidates
from ..rewrites.local_refine import FaceRefine, LocalRefine
from ..rewrites.merge_face import merge_face_candidates
from ..rewrites.split_face import SplitFace


@dataclass
class ProposalConfig:
    mode: str = "residual"  # "residual" | "uniform"
    n_refine: int = 8
    n_simplify: int = 12
    alpha_grad: float = 0.1
    kind_weights: dict = field(default_factory=lambda: {"LocalRefine": 1.0, "KnotInsert": 0.3, "SplitFace": 0.0,
                                                        "CarrierKnotInsert": 0.3, "FaceRefine": 0.3})
    local_widths: tuple = (0.25, 0.4)  # in root-face parameter units
    local_refine_knots: tuple = (1, 2)
    p_refine_boundary: float = 0.5  # share of LocalRefine proposals that also refine the window's carriers
    # Coarse-to-fine schedule: for the first N discrete phases only *global* refinements
    # (FaceRefine, CarrierKnotInsert) are proposed. Local windows have the highest immediate
    # gain per DOF, so a myopic score commits to them early even when the misfit is global.
    global_first_rounds: int = 0
    global_kinds: tuple = ("FaceRefine", "CarrierKnotInsert")
    # Residual-adaptive scale (default on): if the residual is *spread* -- the smallest
    # area fraction holding half of the residual energy exceeds ``spread_threshold`` --
    # the phase proposes global refinements only; concentrated residuals get local windows.
    # Threshold 0.04 = geometric mean of the two tuning targets' spreads (0.013 grammar /
    # 0.127 analytic) after the first continuous phase.
    adaptive_scale: bool = True
    spread_threshold: float = 0.04
    min_knot_gap: float = 0.04
    eps_remove: float = 5e-3
    eps_merge: float = 5e-3


def residual_spread(rf: dict) -> float:
    """Smallest area fraction holding half of the residual energy sum_s w_s e_s^2 (0 if no residual)."""
    en = rf["w"] * rf["e"] ** 2
    if en.sum() <= 0:
        return 0.0
    order = np.argsort(-en)
    k = int(np.searchsorted(np.cumsum(en[order]) / en.sum(), 0.5))
    return float(rf["w"][order[: k + 1]].sum() / rf["w"].sum())


def residual_field(objective: ShapeObjective, state: CADState, terms: dict) -> dict:
    """Per-sample residual e, its gradient magnitude, and sampling weights."""
    disc = objective.disc(state)
    sm = disc.sampler
    if "phi" not in terms:  # no shape target (e.g. physics-only objective): no residual field
        w = to_numpy(terms["w"])
        return {"e": np.zeros_like(w), "grad": np.zeros_like(w), "w": w}
    phi = to_numpy(terms["phi"])
    e = np.abs(phi)
    if "cov_d" in terms:
        # each target point's coverage distance is splatted onto its nearest loss sample
        cov = np.zeros(sm.n_samples)
        d = to_numpy(terms["cov_d"])
        nearest = to_numpy(nearest_centroids(objective.target.points, terms["X"].detach(), 1))[:, 0]
        np.maximum.at(cov, nearest, d)
        e = e + cov
    X = to_numpy(terms["X"])
    grad = np.zeros_like(e)
    for fid in sm.face_order:
        off, nu, nv = sm.face_slices[fid]
        E = e[off: off + nu * nv].reshape(nu, nv)
        P = X[off: off + nu * nv].reshape(nu, nv, 3)
        gi = np.gradient(E, axis=0) / np.maximum(np.linalg.norm(np.gradient(P, axis=0), axis=-1), 1e-12)
        gj = np.gradient(E, axis=1) / np.maximum(np.linalg.norm(np.gradient(P, axis=1), axis=-1), 1e-12)
        grad[off: off + nu * nv] = np.sqrt(gi**2 + gj**2).ravel()
    w = to_numpy(terms["w"])
    return {"e": e, "grad": grad, "w": w}


class ProposalSampler:
    def __init__(self, cfg: ProposalConfig | None = None, seed: int = 0):
        self.cfg = cfg or ProposalConfig()
        self.rng = np.random.default_rng(seed)

    def location_probabilities(self, objective, state, terms) -> np.ndarray:
        rf = residual_field(objective, state, terms)
        self.last_spread = residual_spread(rf)
        if self.cfg.mode == "uniform":
            p = rf["w"].copy()
        elif self.cfg.mode == "residual":
            p = rf["w"] * (rf["e"] + self.cfg.alpha_grad * rf["grad"])
            if p.sum() <= 0:  # no residual information: fall back to uniform
                p = rf["w"].copy()
        else:
            raise ValueError(self.cfg.mode)
        p = np.maximum(p, 0)
        return p / p.sum()

    def refinements(self, objective: ShapeObjective, state: CADState, terms: dict, phase: int = 10**9) -> list:
        """Refinement proposals; ``phase`` counts discrete phases (for the coarse-to-fine schedule)."""
        cfg = self.cfg
        sm = objective.disc(state).sampler
        prob = self.location_probabilities(objective, state, terms)
        kinds = [k for k, w in cfg.kind_weights.items() if w > 0]
        global_phase = phase < cfg.global_first_rounds or (
            cfg.adaptive_scale and "phi" in terms and self.last_spread > cfg.spread_threshold)
        if global_phase:
            kinds = [k for k in kinds if k in cfg.global_kinds] or kinds
        kw = np.array([cfg.kind_weights[k] for k in kinds], float)
        kw /= kw.sum()
        out = []
        for s in self.rng.choice(len(prob), size=cfg.n_refine, p=prob):
            fid = sm.face_order[sm.sample_face[s]]
            u, v = sm.sample_uv[s]
            f = state.cx.faces[fid]
            kind = kinds[self.rng.choice(len(kinds), p=kw)]
            if kind == "LocalRefine":
                root_w = float(self.rng.choice(cfg.local_widths))
                size = np.sqrt((f.domain[1] - f.domain[0]) * (f.domain[3] - f.domain[2]))
                width = min(1.0, root_w / size)
                out.append(LocalRefine(fid, float(u), float(v), width=width,
                                       refine_knots=int(self.rng.choice(cfg.local_refine_knots)),
                                       refine_boundary=bool(self.rng.random() < cfg.p_refine_boundary)))
            elif kind == "FaceRefine":
                out.append(FaceRefine(fid))
            elif kind == "CarrierKnotInsert":
                rw = self.carrier_proposal(state, fid, float(u), float(v))
                if rw is not None:
                    out.append(rw)
            else:
                axis = "u" if self.rng.random() < 0.5 else "v"
                t = float(u if axis == "u" else v)
                knots = f.knots_u if axis == "u" else f.knots_v
                if t < cfg.min_knot_gap or t > 1 - cfg.min_knot_gap or \
                        np.min(np.abs(bb.interior_knots(knots, f.degree_u if axis == "u" else f.degree_v) - t),
                               initial=1.0) < cfg.min_knot_gap:
                    continue
                out.append(KnotInsert(fid, axis, t) if kind == "KnotInsert" else SplitFace(fid, axis, t))
        return out

    def carrier_proposal(self, state: CADState, fid: int, u: float, v: float):
        """Refine the carrier of the face side nearest to (u, v) at the corresponding parameter."""
        cx = state.cx
        f = cx.faces[fid]
        side = min((("v0", v), ("u1", 1 - u), ("v1", 1 - v), ("u0", u)), key=lambda x: x[1])[0]
        t = u if side in ("v0", "v1") else v
        for use in f.sides[side]:
            if use.a - 1e-12 <= t <= use.b + 1e-12:
                sa, sb = cx.use_carrier_params(use)
                s = sa + (t - use.a) * (sb - sa) / (use.b - use.a)
                c = cx.carriers[cx.edges[use.edge].carrier]
                existing = np.concatenate([[0.0, 1.0], bb.interior_knots(c.knots, c.degree)])
                if np.min(np.abs(existing - s)) < self.cfg.min_knot_gap:
                    return None
                return CarrierKnotInsert(c.id, float(s))
        return None

    def simplifications(self, state: CADState) -> list:
        cfg = self.cfg
        cands = (knot_remove_candidates(state, cfg.eps_remove) + merge_face_candidates(state, cfg.eps_merge)
                 + carrier_knot_remove_candidates(state, cfg.eps_remove))
        if len(cands) > cfg.n_simplify:
            idx = self.rng.choice(len(cands), size=cfg.n_simplify, replace=False)
            cands = [cands[i] for i in sorted(idx)]
        return cands
