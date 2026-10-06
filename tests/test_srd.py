"""Milestone 8: the full SRD loop."""
import numpy as np
import pytest

from cad_d4d.geometry.state import BirthRecord, root_deviation, watertightness_error
from cad_d4d.losses.complexity import complexity_delta
from cad_d4d.losses.objective import ObjectiveConfig, ShapeObjective
from cad_d4d.optimization.continuous import ContinuousOptimizer
from cad_d4d.optimization.proposal_sampling import ProposalConfig
from cad_d4d.optimization.rewrite_scoring import ScoringConfig
from cad_d4d.optimization.srd import SRD, SRDConfig
from cad_d4d.rewrites.local_refine import LocalRefine
from cad_d4d.targets.synthetic import TargetConfig, localized_bump_target


@pytest.fixture(scope="module")
def bump(coarse_sphere):
    target = localized_bump_target(coarse_sphere, face_index=0, height=0.25,
                                   cfg=TargetConfig(sdf_res=48, n_coverage=1500))
    return target, ShapeObjective(target, ObjectiveConfig())


def run(obj, state, mode, rounds=6, steps=30, **kw):
    cfg = SRDConfig(rounds=rounds, steps_per_round=steps, scoring=ScoringConfig(mode=mode), **kw)
    return SRD(obj, cfg).run(state)


def test_srd_adaptive_refinement_beats_fixed_coarse(bump, coarse_sphere):
    target, obj = bump
    res = run(obj, coarse_sphere, "marginal_birth")
    coarse = coarse_sphere.copy()
    ContinuousOptimizer(obj).run(coarse, 6 * 30)
    fit_srd, fit_coarse = obj.report(res.state)["fit"], obj.report(coarse)["fit"]
    assert fit_srd < 0.3 * fit_coarse
    refinements = [e for e in res.events if e["refinement"]]
    assert refinements, "at least one refinement must be accepted"
    # refinement concentrates at the localized feature (root face of the bump). Small
    # cheap refinements elsewhere may be accepted when their predicted gain exceeds the
    # birth cost; they are subject to removal after their grace period.
    bump_root = sorted(coarse_sphere.cx.faces)[0]
    first = refinements[0]["rewrite"]
    assert f"face={bump_root}" in first or "Carrier" in first
    s = res.state
    live = {r.id: r.remaining for r in s.birth_records}
    on_feature = {r for f in s.cx.faces.values() if f.root == bump_root for r in f.lineage}
    total = sum(live.values())
    assert sum(v for r, v in live.items() if r in on_feature) >= 0.75 * total
    s.cx.check_invariants()
    assert watertightness_error(s) < 1e-12
    assert len(s.birth_records) == len(refinements)
    # history is consistent: loss is monotone inside every continuous phase
    for r in range(6):
        losses = [h["loss"] for h in res.history if h["round"] == r]
        assert all(b <= a + 1e-15 for a, b in zip(losses, losses[1:]))


def test_srd_naive_scoring_never_refines(bump, coarse_sphere):
    _, obj = bump
    res = run(obj, coarse_sphere, "naive", rounds=4)
    assert not [e for e in res.events if e["refinement"]]
    assert res.state.n_faces == 6
    refine_props = [p for p in res.proposals if p["refinement"] and p["ok"]]
    assert refine_props and all(p["score"] < 0 for p in refine_props)


def test_unused_refinement_is_eventually_removed(bump, coarse_sphere):
    """A refinement in a region that needs no extra capacity is protected during
    its grace period and later simplified away by KnotRemove/MergeFace."""
    target, obj = bump
    s = coarse_sphere.copy()
    ContinuousOptimizer(obj).run(s, 60)
    far = sorted(s.cx.faces)[1]
    out = LocalRefine(far, 0.5, 0.5, width=0.4, refine_knots=1).apply(s)
    refined = out.state
    dC = complexity_delta(s, refined, obj.cfg.complexity)
    refined.birth_records.append(BirthRecord(1, "LocalRefine", dC))
    refined.meta["next_record"] = 2
    for fid in out.info["born_faces"]:
        refined.cx.faces[fid].lineage = refined.cx.faces[fid].lineage | {1}
    assert refined.n_faces == 10  # 4 exact splits: 1 face -> 5
    no_refine = ProposalConfig(n_refine=0, n_simplify=40)
    res = run(obj, refined, "marginal_birth", rounds=10, steps=25, warmup_rounds=0,
              max_simplify_per_round=4, proposals=no_refine)
    simplifications = [e for e in res.events if not e["refinement"]]
    first = min(e["step"] for e in simplifications)
    assert first > obj.cfg.complexity.hold_steps, "nothing may be removed during the hold period"
    assert res.state.n_faces == s.n_faces
    assert res.state.n_control_points == s.n_control_points
    assert res.state.birth_records[0].simplified
    assert watertightness_error(res.state) < 1e-12
