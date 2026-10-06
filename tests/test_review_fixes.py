"""Regression tests for fixes from the code review."""
import numpy as np

from cad_d4d.geometry import bspline_basis as bb
from cad_d4d.geometry.fitting import dense_params, fit_face_interior
from cad_d4d.geometry.state import BirthRecord, DofMap
from cad_d4d.losses.complexity import complexity_delta
from cad_d4d.losses.objective import ObjectiveConfig, ShapeObjective
from cad_d4d.optimization.continuous import ContinuousOptimizer
from cad_d4d.optimization.rewrite_scoring import RewriteScorer, ScoringConfig
from cad_d4d.rewrites.knot_remove import knot_remove_candidates
from cad_d4d.rewrites.local_refine import LocalRefine

from helpers import perturbed_state


def test_separable_refit_matches_dense_least_squares():
    s = perturbed_state(41, n_knots=2)
    fid = sorted(s.cx.faces)[0]
    f = s.cx.faces[fid]
    us, vs = dense_params(f.knots_u, 3, 5), dense_params(f.knots_v, 3, 5)
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(len(us), len(vs), 3))
    dm = DofMap(s.cx)
    net = (dm.face_rows(fid) @ dm.values).reshape(-1, 3)
    B = np.kron(bb.basis_matrix(f.knots_u, 3, us), bb.basis_matrix(f.knots_v, 3, vs))
    mask = f.interior_mask()
    X_ref, *_ = np.linalg.lstsq(B[:, mask], Y.reshape(-1, 3) - B[:, ~mask] @ net[~mask], rcond=None)
    fit_face_interior(s.cx, fid, us, vs, Y, dm)
    assert np.allclose(f.interior.reshape(-1, 3), X_ref, atol=1e-10)


def test_scoring_does_not_charge_birth_records_of_the_candidate(reachable_target):
    s = perturbed_state(42)
    out = LocalRefine(sorted(s.cx.faces)[0], 0.5, 0.5, width=0.4, refine_knots=2).apply(s)
    born = out.state
    obj = ShapeObjective(reachable_target, ObjectiveConfig())
    dC = complexity_delta(s, born, obj.cfg.complexity)
    born.birth_records.append(BirthRecord(1, "LocalRefine", dC, age=500))  # past its grace period
    for fid in out.info["born_faces"]:
        born.cx.faces[fid].lineage = born.cx.faces[fid].lineage | {1}
    sc = RewriteScorer(obj, ContinuousOptimizer(obj), ScoringConfig())
    ctx = sc.prepare(born)
    removal = [r for r in knot_remove_candidates(born, eps=10.0) if r.face == out.info["refined_face"]][0]
    scored = sc.score(ctx, removal)
    assert scored.ok
    rec = scored.outcome.state.birth_records[0]
    assert rec.remaining == dC and not rec.simplified  # charged only on acceptance (in SRD)


def test_copied_state_reuses_a_consistent_dof_map():
    s = perturbed_state(43)
    _ = s.dof_map.E
    c = s.copy()
    assert c._dof_map is not None and c._dof_map is not s._dof_map
    fresh = DofMap(c.cx)
    assert abs(c.dof_map.E - fresh.E).max() == 0
    c.set_values(c.values() + 1.0)
    assert not np.allclose(s.values(), c.values())  # values are independent


def test_monotone_self_intersection_rule(reachable_target, coarse_sphere, monkeypatch):
    """Steps may not add self-intersections, but a state that already has some (e.g. a fold
    revealed when a rewrite re-samples the check tessellation) must not freeze the optimizer."""
    from cad_d4d.optimization.discretization import Discretization
    obj = ShapeObjective(reachable_target, ObjectiveConfig())
    opt = ContinuousOptimizer(obj)

    s = coarse_sphere.copy()
    monkeypatch.setattr(Discretization, "count_self_intersections", lambda self, X, return_hits=False: 3)
    logs = opt.run(s, 5)  # pre-existing, not worsened
    assert all(l["eta"] > 0 for l in logs)

    import torch
    from cad_d4d.device import to_tensor
    s = coarse_sphere.copy()
    P0 = to_tensor(s.values())
    disc = obj.disc(s)
    X_cur = disc.check.evaluate(P0)[0]

    def growing(self, X, return_hits=False):  # every trial looks worse than the current state
        return 3 if torch.equal(X, X_cur) else 4

    monkeypatch.setattr(Discretization, "count_self_intersections", growing)
    P_new, log = opt.step(s, P0, 0.5, disc)
    assert P_new is None and log["invalid"] > 0
