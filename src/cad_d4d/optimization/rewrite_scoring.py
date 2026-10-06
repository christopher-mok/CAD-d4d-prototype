"""Rewrite scoring, including exact-refinement-aware marginal-descent scores.

Objective used for discrete decisions:

    F(x) = L_smooth(x) + lambda_complex * C_eff(s)

Modes
-----
A  "naive":           score = F(x) - F(x')  (immediate). For an exact
                      refinement L_smooth(x') = L_smooth(x), so the score is
                      -lambda_complex * dC < 0: refinements are always rejected.
B  "marginal":        exact refinements scored by the first-order descent
                      capability they unlock, with no birth cost:
                          D = g^T M^{-1} g            (same preconditioner M)
                          B_refine = D_new - D_old    (no-rewrite counterfactual:
                                                      old state + one step)
                          score = horizon * eta * B_refine
C  "marginal_birth":  score = horizon * eta * B_refine - lambda_birth * dC,
                      plus a grace period: newborn structure cannot be
                      simplified during its hold period, then its complexity
                      cost ramps from lambda_birth to lambda_complex. (default)

Inexact rewrites (KnotRemove, MergeFace, CarrierKnotRemove) always use the immediate difference
F(x) - F(x'), with effective (grace-discounted) complexity in mode C.
"""
from __future__ import annotations

import copy

from dataclasses import dataclass, field

import numpy as np

from ..geometry.state import CADState
from ..losses.complexity import complexity_delta, effective_complexity
from ..losses.objective import ShapeObjective
from ..rewrites.base import Rewrite, RewriteOutcome
from .continuous import ContinuousOptimizer, GradientInfo

MODES = ("naive", "marginal", "marginal_birth")


@dataclass
class ScoringConfig:
    mode: str = "marginal_birth"
    eta: float | None = None  # nominal step for predicted improvement (default: continuous eta)
    horizon: float = 1.0
    log_naive_descent: bool = False  # naive mode: also compute D_new for logging (one extra backward pass)
    rank_by: str = "ratio"  # "ratio" (refinements by score per added complexity unit) | "score"


@dataclass
class ScoringContext:
    state: CADState
    gi: GradientInfo
    D_old: float
    smooth: float
    C_eff: float


@dataclass
class ScoredRewrite:
    rewrite: Rewrite
    outcome: RewriteOutcome
    score: float
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome.ok and np.isfinite(self.score)


def consumed_faces(rewrite: Rewrite, state: CADState) -> set[int]:
    """Faces whose structure a simplifying rewrite removes."""
    if rewrite.kind == "KnotRemove":
        return {rewrite.face}
    if rewrite.kind == "MergeFace":
        return {rewrite.face_a, rewrite.face_b}
    if rewrite.kind == "CarrierKnotRemove":
        return rewrite.touched_faces(state)
    return set()


def consumed_lineage(state: CADState, rewrite: Rewrite) -> set[int]:
    """Birth-record ids carried by the faces a simplifying rewrite consumes."""
    tags = set()
    for fid in consumed_faces(rewrite, state):
        if fid in state.cx.faces:
            tags |= set(state.cx.faces[fid].lineage)
    return tags


def mark_simplified(before: CADState, after: CADState, rewrite: Rewrite, removed: float) -> list[int]:
    """Charge ``removed`` complexity against the birth records of the consumed structure.

    Records touched by the simplification are flagged ``simplified``; their
    still-discounted ``remaining`` complexity shrinks by the removed amount,
    youngest record first. Mutates ``after.birth_records``: call it once, on
    acceptance (scoring uses a scratch copy).
    """
    tags = consumed_lineage(before, rewrite)
    hit = sorted((r for r in after.birth_records if r.id in tags), key=lambda r: r.age)
    left = max(0.0, removed)
    for r in hit:
        r.simplified = True
        take = min(r.remaining, left)
        r.remaining -= take
        left -= take
    return [r.id for r in hit]


class RewriteScorer:
    def __init__(self, objective: ShapeObjective, optimizer: ContinuousOptimizer, cfg: ScoringConfig | None = None):
        self.obj = objective
        self.opt = optimizer
        self.cfg = cfg or ScoringConfig()
        if self.cfg.mode not in MODES:
            raise ValueError(self.cfg.mode)

    @property
    def eta(self) -> float:
        return self.cfg.eta if self.cfg.eta is not None else self.opt.cfg.eta

    def prepare(self, state: CADState) -> ScoringContext:
        gi = self.opt.gradient(state)
        return ScoringContext(state, gi, gi.descent_capacity, float(gi.terms["smooth"]),
                              effective_complexity(state, self.obj.cfg.complexity))

    def grace_protected(self, state: CADState, rewrite: Rewrite) -> list[int]:
        """Birth records (still in their hold period) whose structure ``rewrite`` would remove."""
        cc = self.obj.cfg.complexity
        if not cc.use_grace:
            return []
        tags = consumed_lineage(state, rewrite)
        return [r.id for r in state.birth_records
                if r.id in tags and r.remaining > 0 and r.age <= cc.hold_steps]

    def score(self, ctx: ScoringContext, rewrite: Rewrite) -> ScoredRewrite:
        protected = self.grace_protected(ctx.state, rewrite)
        if protected:
            return ScoredRewrite(rewrite, RewriteOutcome(None, "protected by grace period"), -np.inf,
                                 {"reason": "grace", "protected_records": protected})
        out = rewrite.apply(ctx.state)
        if not out.ok:
            return ScoredRewrite(rewrite, out, -np.inf, {"reason": out.reason, "deviation": out.deviation})
        return self.score_outcome(ctx, rewrite, out)

    def score_outcome(self, ctx: ScoringContext, rewrite: Rewrite, out: RewriteOutcome) -> ScoredRewrite:
        cc = self.obj.cfg.complexity
        new = out.state
        dC = complexity_delta(ctx.state, new, cc)
        # effective complexity after the rewrite, charging a scratch copy of the birth records
        # (the real records are charged once, when the rewrite is accepted)
        records = new.birth_records
        new.birth_records = copy.deepcopy(records)
        mark_simplified(ctx.state, new, rewrite, -dC)
        C_eff_new = effective_complexity(new, cc)
        new.birth_records = records
        exact_refinement = rewrite.exact and rewrite.refinement
        need_gradient = exact_refinement and (self.cfg.mode != "naive" or self.cfg.log_naive_descent)
        gi_new = self.opt.gradient(new) if need_gradient else None
        smooth_new = float(gi_new.terms["smooth"]) if gi_new is not None else self.obj.smooth_value(new)
        dC_eff = C_eff_new - ctx.C_eff
        immediate = (ctx.smooth - smooth_new) - cc.lambda_complex * dC_eff
        info = {"delta_complexity": dC, "delta_complexity_eff": dC_eff, "immediate_gain": immediate,
                "delta_smooth": smooth_new - ctx.smooth, "deviation": out.deviation,
                "n_cp_before": ctx.state.n_control_points, "n_cp_after": new.n_control_points}
        if exact_refinement and self.cfg.mode != "naive":
            D_new = gi_new.descent_capacity
            B = D_new - ctx.D_old
            predicted = self.cfg.horizon * self.eta * B
            birth = cc.lambda_birth * dC if self.cfg.mode == "marginal_birth" else 0.0
            score = predicted - birth
            info.update(D_old=ctx.D_old, D_new=D_new, B_refine=B, predicted_improvement=predicted,
                        birth_cost=birth, scoring="marginal")
        else:
            score = immediate
            info.update(scoring="immediate")
            if gi_new is not None:  # naive mode: descent capacity logged for comparison only
                info.update(D_old=ctx.D_old, D_new=gi_new.descent_capacity,
                            B_refine=gi_new.descent_capacity - ctx.D_old)
        info["score"] = score
        new.cache.clear()  # release per-candidate operators (dense preconditioner factors etc.)
        return ScoredRewrite(rewrite, out, score, info)
