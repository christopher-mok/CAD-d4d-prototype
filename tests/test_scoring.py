"""Milestone 7: proposals, exact-refinement scoring, birth/grace complexity."""
import dataclasses

import numpy as np
import pytest

from cad_d4d.geometry.state import BirthRecord, root_deviation
from cad_d4d.losses.complexity import ComplexityConfig, complexity_delta
from cad_d4d.losses.objective import ObjectiveConfig, ShapeObjective
from cad_d4d.optimization.continuous import ContinuousOptimizer
from cad_d4d.optimization.proposal_sampling import ProposalConfig, ProposalSampler
from cad_d4d.optimization.rewrite_scoring import RewriteScorer, ScoringConfig
from cad_d4d.rewrites.knot_remove import knot_remove_candidates
from cad_d4d.rewrites.local_refine import LocalRefine
from cad_d4d.targets.synthetic import TargetConfig, localized_bump_target


@pytest.fixture(scope="module")
def bump_setup(coarse_sphere):
    target = localized_bump_target(coarse_sphere, face_index=0, height=0.25,
                                   cfg=TargetConfig(sdf_res=48, n_coverage=1500))
    obj = ShapeObjective(target, ObjectiveConfig())
    opt = ContinuousOptimizer(obj)
    stalled = coarse_sphere.copy()
    opt.run(stalled, 120)  # coarse fitting until progress stalls
    bump_face = sorted(coarse_sphere.cx.faces)[0]
    far_face = sorted(coarse_sphere.cx.faces)[1]  # opposite side of the cube (-x vs +x)
    return target, obj, opt, stalled, bump_face, far_face


def scorer(obj, opt, mode):
    return RewriteScorer(obj, opt, ScoringConfig(mode=mode))


def test_coarse_representation_stalls(bump_setup):
    target, obj, opt, stalled, _, _ = bump_setup
    s = stalled.copy()
    before = obj.report(s)["fit"]
    opt.run(s, 40)
    after = obj.report(s)["fit"]
    assert after > 0.8 * before  # < 20% progress over 40 more steps
    assert after > 1e-4  # large residual remains (the bump)


def test_exact_refinement_changes_geometry_by_zero(bump_setup):
    _, obj, _, stalled, bump_face, _ = bump_setup
    out = LocalRefine(bump_face, 0.5, 0.5, width=0.4, refine_knots=2).apply(stalled)
    assert root_deviation(stalled, out.state) < 1e-13
    d_fit = abs(obj.report(out.state)["fit"] - obj.report(stalled)["fit"])
    assert d_fit < 0.02 * obj.report(stalled)["fit"]  # only quadrature differences


def test_naive_rejects_but_marginal_accepts_useful_refinement(bump_setup):
    _, obj, opt, stalled, bump_face, _ = bump_setup
    rw = LocalRefine(bump_face, 0.5, 0.5, width=0.4, refine_knots=2)
    res = {}
    for mode in ("naive", "marginal", "marginal_birth"):
        sc = scorer(obj, opt, mode)
        res[mode] = sc.score(sc.prepare(stalled), rw)
    assert res["naive"].score < 0, "naive immediate score must reject the exact refinement"
    assert res["marginal"].score > 0
    c = res["marginal_birth"]
    assert c.score > 0
    assert c.info["B_refine"] == pytest.approx(c.info["D_new"] - c.info["D_old"])
    assert c.info["birth_cost"] == pytest.approx(obj.cfg.complexity.lambda_birth * c.info["delta_complexity"])
    assert c.info["birth_cost"] < obj.cfg.complexity.lambda_complex * c.info["delta_complexity"]
    # The marginal score dwarfs the naive penalty: the refinement unlocks real descent.
    assert c.info["predicted_improvement"] > 5 * abs(res["naive"].score)


def test_marginal_birth_rejects_useless_refinement(bump_setup):
    _, obj, opt, stalled, _, far_face = bump_setup
    sc = scorer(obj, opt, "marginal_birth")
    out = sc.score(sc.prepare(stalled), LocalRefine(far_face, 0.5, 0.5, width=0.4, refine_knots=2))
    assert out.score < 0


def test_no_credit_for_ordinary_gradient_progress(bump_setup, coarse_sphere):
    """Far from convergence D_new is large, but so is D_old: B_refine compares
    against the no-rewrite counterfactual instead of the new gradient norm."""
    _, obj, opt, _, _, far_face = bump_setup
    fresh = coarse_sphere.copy()
    sc = scorer(obj, opt, "marginal_birth")
    ctx = sc.prepare(fresh)
    out = sc.score(ctx, LocalRefine(far_face, 0.5, 0.5, width=0.4, refine_knots=2))
    assert out.info["D_new"] > 1e-4
    assert abs(out.info["B_refine"]) < 0.05 * out.info["D_new"]


def test_new_control_points_realize_the_predicted_benefit(bump_setup):
    _, obj, opt, stalled, bump_face, _ = bump_setup
    refined = LocalRefine(bump_face, 0.5, 0.5, width=0.4, refine_knots=2).apply(stalled).state
    counterfactual = stalled.copy()
    opt_a, opt_b = ContinuousOptimizer(obj), ContinuousOptimizer(obj)
    opt_a.run(refined, 40)
    opt_b.run(counterfactual, 40)
    assert obj.report(refined)["fit"] < 0.5 * obj.report(counterfactual)["fit"]


def test_residual_guided_proposals_concentrate_on_feature(bump_setup):
    _, obj, opt, stalled, bump_face, _ = bump_setup
    gi = opt.gradient(stalled)
    sm = obj.disc(stalled).sampler
    on_face = sm.sample_face == sm.face_order.index(bump_face)
    mass = {}
    for mode in ("residual", "uniform"):
        p = ProposalSampler(ProposalConfig(mode=mode)).location_probabilities(obj, stalled, gi.terms)
        mass[mode] = p[on_face].sum()
    assert mass["uniform"] == pytest.approx(1 / 6, abs=0.03)
    assert mass["residual"] > 2 * mass["uniform"]
    props = ProposalSampler(ProposalConfig(mode="residual", n_refine=30), seed=1).refinements(obj, stalled, gi.terms)
    def on_feature(p):
        if p.kind == "CarrierKnotInsert":  # boundary refinement: feature face must be incident
            return bump_face in p.touched_faces(stalled)
        return p.face == bump_face
    assert sum(on_feature(p) for p in props) > 0.4 * len(props)


def test_grace_period_protects_newborn_refinement(bump_setup):
    """Right after birth, removing the new structure looks profitable at full
    complexity cost but not under the grace discount."""
    target, _, _, stalled, bump_face, _ = bump_setup
    cc = ComplexityConfig(lambda_complex=1e-4, lambda_birth=1e-6, hold_steps=40, ramp_steps=40)
    obj = ShapeObjective(target, ObjectiveConfig(complexity=cc))
    opt = ContinuousOptimizer(obj)
    out = LocalRefine(bump_face, 0.5, 0.5, width=0.4, refine_knots=2).apply(stalled)
    born = out.state
    dC = complexity_delta(stalled, born, cc)
    born.birth_records.append(BirthRecord(1, "LocalRefine", dC))
    for fid in out.info["born_faces"]:
        born.cx.faces[fid].lineage = born.cx.faces[fid].lineage | {1}
    opt.run(born, 3)  # newborn: only a few steps of adaptation
    refined = out.info["refined_face"]
    removals = [r for r in knot_remove_candidates(born, eps=0.05) if r.face == refined]
    assert removals

    def best_removal(state, use_grace):
        o = ShapeObjective(target, ObjectiveConfig(complexity=dataclasses.replace(cc, use_grace=use_grace)))
        sc = RewriteScorer(o, ContinuousOptimizer(o), ScoringConfig(mode="marginal_birth"))
        ctx = sc.prepare(state)
        return max((sc.score(ctx, r) for r in removals), key=lambda x: x.score)

    assert best_removal(born, use_grace=False).score > 0, "without grace the young DOFs would be removed"
    protected = best_removal(born, use_grace=True)
    assert protected.score == -np.inf and protected.info["reason"] == "grace"
    # Survives long enough to improve: after the hold period the new control
    # points have grown into the bump, so they are geometrically necessary and
    # removal is epsilon-gated out at the default tolerance.
    opt.run(born, 60)
    assert born.birth_records[0].age > cc.hold_steps
    assert obj.report(born)["fit"] < 0.1 * obj.report(stalled)["fit"]
    later = [r for r in knot_remove_candidates(born, eps=2e-3) if r.face == refined]
    assert later and all(not r.apply(born).ok for r in later)


def test_birth_discount_schedule():
    from cad_d4d.losses.complexity import birth_discount
    cc = ComplexityConfig(lambda_complex=1e-6, lambda_birth=1e-7, hold_steps=10, ramp_steps=10)
    assert birth_discount(0, cc) == pytest.approx(0.9)
    assert birth_discount(10, cc) == pytest.approx(0.9)
    assert birth_discount(15, cc) == pytest.approx(0.45)
    assert birth_discount(25, cc) == 0.0
    assert birth_discount(0, dataclasses.replace(cc, use_grace=False)) == 0.0
