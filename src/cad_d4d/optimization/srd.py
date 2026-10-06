"""Stochastic Rewrite Descent over B-spline patch complexes.

    repeat:
        continuous phase: preconditioned descent on p (structure s fixed)
        discrete phase:
            sample refinements (residual-guided) + simplifications
            score (exact refinements: marginal descent; others: immediate)
            [optional lookahead on a shortlist]
            accept compatible improving rewrites (greedy, conflict-free)
            birth records / grace bookkeeping

Conflicts: two rewrites conflict when their touched-face sets intersect
(touched = modified faces plus faces whose wires change). Accepted rewrites are
applied sequentially; each later one is re-applied to the updated state.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field


from ..device import to_tensor
from ..geometry.state import BirthRecord, CADState
from ..losses.complexity import complexity_delta, effective_complexity, structural_complexity
from ..losses.objective import ShapeObjective
from .continuous import ContinuousConfig, ContinuousOptimizer
from .proposal_sampling import ProposalConfig, ProposalSampler
from .rewrite_scoring import RewriteScorer, ScoredRewrite, ScoringConfig, mark_simplified


@dataclass
class SRDConfig:
    rounds: int = 16
    steps_per_round: int = 40
    warmup_rounds: int = 1
    discrete: bool = True
    max_refine_per_round: int = 1
    max_simplify_per_round: int = 2
    accept_threshold: float = 0.0
    lookahead_steps: int = 0
    lookahead_shortlist: int = 3
    track_counterfactual: int = 0  # number of accepted refinements to compare against no-rewrite runs
    polish_steps: int = 0  # final continuous steps against the exact narrow-band SDF (no rewrites)
    # Preconditioner M of the descent capacity D = g^T M^{-1} g used to *score* refinements
    # (None: the continuous optimizer's). Default: the consistent L2 mass. The lumped mass
    # understates the descent available in fine-scale (oscillatory) control modes -- exactly
    # the modes an exact refinement adds -- so lumped scores under-predict useful refinements.
    # Steps keep the lumped/semi-implicit metric, which is more local (better on small features).
    scoring_preconditioner: str | None = "consistent_mass"
    seed: int = 0
    continuous: ContinuousConfig = field(default_factory=ContinuousConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    proposals: ProposalConfig = field(default_factory=ProposalConfig)


@dataclass
class SRDResult:
    state: CADState
    history: list
    rounds: list
    proposals: list
    events: list
    counterfactuals: list
    runtime: float


def polish_continuous(obj: ShapeObjective, state: CADState, steps: int, cont: ContinuousConfig,
                      callback=None) -> list:
    """Final continuous phase against the exact narrow-band SDF.

    The trilinear SDF is the better objective while the surface is still far from
    sharp target features (the exact field's kinks then pull boundary rows into
    folds); near convergence the exact field removes the trilinear bias.
    """
    pcfg = dataclasses.replace(obj.cfg, sdf_eval="exact")  # shallow: physics terms are shared, not copied
    return ContinuousOptimizer(ShapeObjective(obj.target, pcfg), cont).run(state, steps, callback=callback)


def rank_candidates(scored: list, rank_by: str, threshold: float) -> list[int]:
    """Indices of acceptable candidates (ok and score > threshold), best first.

    ``score``: by score. ``ratio``: refinements by score per added complexity unit
    (greedy knapsack: prefer the most gain per new DOF), simplifications by score;
    refinements and simplifications have separate per-round caps anyway.
    """
    ok = [i for i, s in enumerate(scored) if s.ok and s.score > threshold]
    if rank_by == "score":
        return sorted(ok, key=lambda i: -scored[i].score)
    if rank_by == "ratio":
        def key(i):
            s = scored[i]
            dC = s.info.get("delta_complexity", 0.0)
            return -(s.score / dC if s.rewrite.refinement and dC > 0 else s.score)
        return sorted(ok, key=key)
    raise ValueError(rank_by)


class SRD:
    def __init__(self, objective: ShapeObjective, cfg: SRDConfig | None = None):
        self.cfg = cfg or SRDConfig()
        # Grace periods are part of scoring mode C only. Shallow config copy: physics terms
        # (FEM operators, warm starts) are shared with the caller, not duplicated.
        ocfg = dataclasses.replace(objective.cfg, complexity=dataclasses.replace(
            objective.cfg.complexity, use_grace=self.cfg.scoring.mode == "marginal_birth"))
        self.obj = ShapeObjective(objective.target, ocfg)
        self.opt = ContinuousOptimizer(self.obj, self.cfg.continuous)
        score_opt = self.opt
        if self.cfg.scoring_preconditioner not in (None, self.cfg.continuous.preconditioner):
            score_opt = ContinuousOptimizer(self.obj, dataclasses.replace(
                self.cfg.continuous, preconditioner=self.cfg.scoring_preconditioner))
        self.scorer = RewriteScorer(self.obj, score_opt, self.cfg.scoring)
        self.sampler = ProposalSampler(self.cfg.proposals, self.cfg.seed)

    # ------------------------------------------------------------------
    def snapshot(self, state: CADState) -> dict:
        r = self.obj.report(state)
        r["n_birth_young"] = sum(1 for b in state.birth_records if b.remaining > 0 and b.age < self.obj.cfg.complexity.grace_steps)
        return r

    def run(self, state: CADState, callback=None, recorder=None) -> SRDResult:
        """Run SRD. ``recorder`` (visualization.recording.Recorder) captures frames/events for replay."""
        cfg = self.cfg
        t0 = time.perf_counter()
        state = state.copy()
        state.meta.setdefault("next_record", 1)
        history, rounds, proposals, events, cfs = [], [], [], [], []
        pending_cf: list[tuple[int, CADState, int]] = []
        step = 0
        cc = self.obj.cfg.complexity
        if recorder is not None:
            recorder.frame(state, state.values(), 0, 0, self.obj.report(state))
        for rnd in range(cfg.rounds):
            hook = recorder.step_hook(state, step, rnd) if recorder is not None else None
            logs = self.opt.run(state, cfg.steps_per_round, callback=hook)
            C = structural_complexity(state, cc)
            C_eff = effective_complexity(state, cc)
            for lg in logs:
                step += 1
                history.append({"step": step, "round": rnd, "loss": lg["loss"], "sdf": lg["sdf"],
                                "coverage": lg["coverage"], "fair": lg["fair"], "eta": lg["eta"], "D": lg["D"],
                                "n_faces": state.n_faces, "n_cp": state.n_control_points,
                                "complexity": C, "complexity_eff": C_eff,
                                "total": lg["loss"] + cc.lambda_complex * C,
                                "total_eff": lg["loss"] + cc.lambda_complex * C_eff})
            for rid, cf_state, at_step in pending_cf:
                ContinuousOptimizer(self.obj, cfg.continuous).run(cf_state, len(logs))
                cfs.append({"record": rid, "accepted_at_step": at_step, "steps": len(logs),
                            "fit_with_rewrite": self.obj.report(state)["fit"],
                            "fit_without_rewrite": self.obj.report(cf_state)["fit"]})
            pending_cf = []
            snap = self.snapshot(state)
            snap.update(round=rnd, step=step)
            if cfg.discrete and rnd >= cfg.warmup_rounds and rnd < cfg.rounds - 1:
                state, rnd_props, rnd_events = self.discrete_phase(state, rnd, step)
                if recorder is not None:
                    recorder.discrete(step, rnd, rnd_props, rnd_events)
                    if rnd_events:  # structure changed: snapshot the exact (unchanged) geometry
                        recorder.frame(state, state.values(), step, rnd, self.obj.report(state))
                proposals += rnd_props
                events += rnd_events
                n_cf_done = sum(1 for e in events[: len(events) - len(rnd_events)] if e["refinement"])
                for e in rnd_events:
                    if e["refinement"] and n_cf_done < cfg.track_counterfactual:
                        pending_cf.append((e["record"], e.pop("_before"), step))
                        n_cf_done += 1
                    e.pop("_before", None)
                snap["n_proposals"] = len(rnd_props)
                snap["n_accepted"] = len(rnd_events)
            rounds.append(snap)
            if callback:
                callback(rnd, state, snap)
        if cfg.polish_steps > 0:
            hook = recorder.step_hook(state, step, cfg.rounds) if recorder is not None else None
            polish_continuous(self.obj, state, cfg.polish_steps, cfg.continuous, callback=hook)
        if recorder is not None:
            recorder.frame(state, state.values(), step + cfg.polish_steps, cfg.rounds, self.obj.report(state))
        return SRDResult(state, history, rounds, proposals, events, cfs, time.perf_counter() - t0)

    # ------------------------------------------------------------------
    def discrete_phase(self, state: CADState, rnd: int, step: int):
        cfg = self.cfg
        ctx = self.scorer.prepare(state)
        props = (self.sampler.refinements(self.obj, state, ctx.gi.terms, phase=rnd - cfg.warmup_rounds)
                 + self.sampler.simplifications(state))
        scored: list[ScoredRewrite] = [self.scorer.score(ctx, rw) for rw in props]
        if cfg.lookahead_steps > 0:
            self.lookahead(ctx, scored)
        logs = []
        for sc in scored:
            loc = sc.rewrite.location(state)
            logs.append({"round": rnd, "step": step, **sc.rewrite.describe(),
                         "location": None if loc is None else loc.tolist(),
                         "ok": sc.outcome.ok, "accepted": False, "status": "rejected",
                         **{k: v for k, v in sc.info.items()}})
        order = rank_candidates(scored, cfg.scoring.rank_by, cfg.accept_threshold)
        touched: set[int] = set()
        n_ref = n_simp = 0
        events = []
        current = state
        n_cur = None  # self-intersections of ``current`` (lazily computed)
        for i in order:
            sc = scored[i]
            rw = sc.rewrite
            if rw.refinement and n_ref >= cfg.max_refine_per_round:
                logs[i]["status"] = "cap"
                continue
            if not rw.refinement and n_simp >= cfg.max_simplify_per_round:
                logs[i]["status"] = "cap"
                continue
            t_faces = rw.touched_faces(state)
            if t_faces & touched:
                logs[i]["status"] = "conflict"
                continue
            out = sc.outcome if current is state else rw.apply(current)
            if not out.ok:
                logs[i]["status"] = "reapply_failed"
                continue
            new = out.state
            # Validity gate: the new structure's check tessellation may reveal (or a refit may
            # create) self-intersections; never accept a rewrite that adds any.
            if n_cur is None:
                n_cur = self.opt.self_intersections(self.obj.disc(current), to_tensor(current.values()))
            n_new = self.opt.self_intersections(self.obj.disc(new), to_tensor(new.values()))
            if n_new > n_cur:
                logs[i]["status"] = "invalid"
                continue
            dC = complexity_delta(current, new, self.obj.cfg.complexity)
            event = {"round": rnd, "step": step, "kind": rw.kind, "refinement": rw.refinement,
                     "exact": rw.exact, "score": sc.score, "delta_complexity": dC,
                     "n_cp": new.n_control_points, "n_faces": new.n_faces, "rewrite": repr(rw)}
            if rw.refinement:
                rid = new.meta["next_record"]
                new.meta["next_record"] = rid + 1
                new.birth_records.append(BirthRecord(rid, rw.kind, dC, info={"round": rnd, "rewrite": repr(rw)}))
                for fid in out.info.get("born_faces", []):
                    if fid in new.cx.faces:
                        new.cx.faces[fid].lineage = new.cx.faces[fid].lineage | {rid}
                event["record"] = rid
                event["_before"] = current.copy()
                n_ref += 1
            else:
                event["simplified_records"] = mark_simplified(current, new, rw, -dC)
                n_simp += 1
            touched |= t_faces
            new.meta["eta"] = cfg.continuous.eta  # conditioning changes with structure: restart the step size
            current, n_cur = new, n_new
            logs[i]["accepted"] = True
            logs[i]["status"] = "accepted"
            events.append(event)
        return current, logs, events

    def lookahead(self, ctx, scored: list[ScoredRewrite]) -> None:
        """Replace shortlisted scores by realized gains after a few local steps,
        measured against the no-rewrite counterfactual with the same steps."""
        cfg = self.cfg
        cc = self.obj.cfg.complexity
        short = sorted([s for s in scored if s.ok], key=lambda s: -s.score)[: cfg.lookahead_shortlist]
        if not short:
            return
        base = ctx.state.copy()
        ContinuousOptimizer(self.obj, cfg.continuous).run(base, cfg.lookahead_steps)
        F_base = self.obj.smooth_value(base) + cc.lambda_complex * effective_complexity(base, cc)
        for s in short:
            cand = s.outcome.state.copy()
            if s.rewrite.refinement:
                dC = complexity_delta(ctx.state, cand, cc)
                cand.birth_records.append(BirthRecord(-1, s.rewrite.kind, dC))
            ContinuousOptimizer(self.obj, cfg.continuous).run(cand, cfg.lookahead_steps)
            F_c = self.obj.smooth_value(cand) + cc.lambda_complex * effective_complexity(cand, cc)
            s.info["lookahead_gain"] = F_base - F_c
            s.info["score_before_lookahead"] = s.score
            s.score = F_base - F_c
            s.info["score"] = s.score
