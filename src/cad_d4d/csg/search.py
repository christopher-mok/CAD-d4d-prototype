"""Residual-guided topology search with matched optimization trials.

Each round:
  1. residual analysis of the current state -> bounded shortlist of grammar edits
     (edits blocked by inverse-edit protection are logged as ``protected``);
  2. baseline trial: a copy of the current state, optimized for K steps;
  3. candidate trials: each edit applied to its own copy of the *current* state
     (never to the baseline or to another candidate), then optimized for the same
     K steps with the same optimizer settings;
  4. score_i = F(baseline after K) - F(candidate_i after K), with F the full
     objective (volume + surface + complexity);
  5. accept the best candidate iff score > accept_abs + accept_rel * F(baseline);
     commit its *optimized* state, otherwise commit the optimized baseline.
Accepted features get provenance (op, round, evidence rule); inverse edits of an
accepted edit (same feature, or within the edit's radius) are blocked for
``protect_rounds`` rounds. The search stops after ``patience`` rounds without an
acceptance (or ``rounds`` rounds), then runs ``final_steps`` of fitting.

This trial scoring is the only scoring used for topology edits; it is separate
from the gradient-based marginal scoring of exact B-spline refinements in
``cad_d4d.optimization.rewrite_scoring``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .grammar import INVERSE, OpContext, topo
from .objective import FitConfig, TopoObjective, fit
from .proposals import ProposalConfig, propose
from .residuals import ResidualConfig, analyze
from .solid import CSGSolid, Grid


@dataclass
class SearchConfig:
    rounds: int = 6
    trial_steps: int = 60          # K: steps per trial (baseline and every candidate)
    initial_steps: int = 0         # fit before the first residual analysis (0: analyze the initial state,
                                   # so unsupported initial features are seen before fitting can hide them)
    final_steps: int = 150
    accept_abs: float = 2e-4       # absolute improvement threshold on F
    accept_rel: float = 0.02       # ... plus this fraction of the baseline F
    protect_rounds: int = 2
    patience: int = 2
    check_n: int = 40              # check grid for operator effect measurement
    fit: FitConfig = field(default_factory=FitConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    proposals: ProposalConfig = field(default_factory=ProposalConfig)


@dataclass
class Protection:
    op: str           # the accepted op
    created: list
    location: np.ndarray
    radius: float
    until_round: int

    def blocks(self, op, rnd: int) -> bool:
        if rnd > self.until_round or INVERSE.get(self.op) != op.name:
            return False
        fid = op.target_feature()
        if fid is not None and fid in self.created:
            return True
        try:
            loc = np.asarray(op.location(), float)
        except NotImplementedError:
            return False
        return bool(np.linalg.norm(loc - self.location) < self.radius + op.radius())


@dataclass
class SearchResult:
    solid: CSGSolid
    rounds: list
    events: list           # accepted edits
    trials: list           # every evaluated candidate (accepted, rejected, failed, protected)
    snapshots: list        # (round, solid copy) after each round, for the viewer
    costs: dict
    wall_time: float
    canonicalized: list = field(default_factory=list)   # features dropped as no-ops, per round


class TopologySearch:
    def __init__(self, objective: TopoObjective, cfg: SearchConfig | None = None):
        self.obj = objective
        self.cfg = cfg or SearchConfig()
        self.cfg.proposals.r_min = self.cfg.fit.r_min
        self.check = Grid(self.cfg.check_n, objective.grid.lo, objective.grid.hi, offset=0.37)

    def F(self, solid: CSGSolid) -> float:
        return self.obj.value(solid)["total"]

    def run(self, solid: CSGSolid, record: bool = True) -> SearchResult:
        cfg = self.cfg
        t_start = time.perf_counter()
        state = solid.copy()
        rounds, events, trials, snapshots, canonicalized = [], [], [], [], []
        protections: list[Protection] = []
        costs = {"proposal_s": 0.0, "trial_s": 0.0, "baseline_steps": 0, "candidate_steps": 0,
                 "committed_steps": 0, "n_candidates": 0, "n_failed": 0, "n_protected": 0}
        fit(state, self.obj, cfg.initial_steps, cfg.fit)
        costs["committed_steps"] += cfg.initial_steps
        if record:
            snapshots.append((-1, state.copy()))
        idle = 0
        for rnd in range(cfg.rounds):
            ctx = OpContext(self.check, round=rnd)
            t0 = time.perf_counter()
            ev = analyze(state, self.obj.rho_t, self.obj.grid, self.obj.cfg.eps, cfg.residual)
            cands = propose(state, ev, self.obj.grid, cfg.proposals)
            t_prop = time.perf_counter() - t0
            costs["proposal_s"] += t_prop
            t1 = time.perf_counter()
            base = state.copy()
            fit(base, self.obj, cfg.trial_steps, cfg.fit)
            costs["baseline_steps"] += cfg.trial_steps
            F_base = self.F(base)
            results = []
            for c in cands:
                rec = {"round": rnd, **c.op.describe(), "rule": c.evidence.get("rule"), "priority": c.priority}
                if any(p.blocks(c.op, rnd) for p in protections):
                    rec.update(status="protected")
                    costs["n_protected"] += 1
                    trials.append(rec)
                    continue
                out = c.op.apply(state, ctx, provenance={"rule": c.evidence.get("rule")})
                if not out.ok:
                    rec.update(status="failed", reason=out.reason, **{k: v for k, v in out.info.items()
                                                                      if k in ("before", "after")})
                    costs["n_failed"] += 1
                    trials.append(rec)
                    continue
                cand = out.solid
                fit(cand, self.obj, cfg.trial_steps, cfg.fit)
                costs["candidate_steps"] += cfg.trial_steps
                costs["n_candidates"] += 1
                F_c = self.F(cand)
                rec.update(status="rejected", score=F_base - F_c, F_candidate=F_c, F_baseline=F_base,
                           trial_steps=cfg.trial_steps, **out.info)
                trials.append(rec)
                results.append((F_base - F_c, c, cand, out, rec))
            costs["trial_s"] += time.perf_counter() - t1
            threshold = cfg.accept_abs + cfg.accept_rel * F_base
            accepted = None
            if results:
                score, c, cand, out, rec = max(results, key=lambda r: r[0])
                if score > threshold:
                    accepted = (score, c, cand, out, rec)
            if accepted:
                score, c, cand, out, rec = accepted
                rec["status"] = "accepted"
                state = cand
                after = topo(state, ctx)
                ev_rec = {"round": rnd, **c.op.describe(), "score": score, "threshold": threshold,
                          "rule": c.evidence.get("rule"), "region": c.evidence.get("region"),
                          "created": out.info.get("created"), "removed": out.info.get("removed"),
                          "measured_before": out.info.get("before"),
                          "measured_after": out.info.get("after"),
                          "after_fit": f"components {after['components']}, cavities {after['cavities']}, "
                                       f"tunnels {after['tunnels']}"}
                events.append(ev_rec)
                protections.append(Protection(c.op.name, list(out.info.get("created") or []),
                                              np.asarray(c.op.location(), float), c.op.radius(),
                                              rnd + cfg.protect_rounds))
                idle = 0
            else:
                state = base
                idle += 1
            costs["committed_steps"] += cfg.trial_steps
            # canonical form: drop features that no longer change the occupancy (e.g. a void the fit
            # pushed out of the material, a body swallowed by another one); logged, never silent
            dead = state.canonicalize(self.check)
            if dead:
                canonicalized.append({"round": rnd, "removed": dead})
            t = topo(state, ctx)
            rounds.append({"round": rnd, "F_baseline": F_base, "threshold": threshold,
                           "best_score": max((r[0] for r in results), default=None),
                           "accepted": accepted[1].op.name if accepted else None, "n_proposals": len(cands),
                           "n_trials": len(results), "proposal_s": t_prop, "F": self.F(state),
                           "n_features": len(state.features), "add_regions": len(ev.add),
                           "remove_regions": len(ev.remove), "topology": {k: t[k] for k in ("components", "cavities", "tunnels")}})
            if record:
                snapshots.append((rnd, state.copy()))
            if idle >= cfg.patience:
                break
        fit(state, self.obj, cfg.final_steps, cfg.fit)
        costs["committed_steps"] += cfg.final_steps
        if record:
            snapshots.append((len(rounds), state.copy()))
        dead = state.canonicalize(self.check)
        if dead:
            canonicalized.append({"round": "final", "removed": dead})
        return SearchResult(state, rounds, events, trials, snapshots, costs, time.perf_counter() - t_start,
                            canonicalized)
