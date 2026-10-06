"""Milestone 4: synthetic targets, losses, continuous fitting, validity."""
import numpy as np
import pytest
import torch

from cad_d4d.device import to_numpy, to_tensor
from cad_d4d.geometry.state import watertightness_error
from cad_d4d.geometry.tessellation import SamplingConfig, SurfaceSampler
from cad_d4d.geometry.validity import jacobian_check, self_intersections
from cad_d4d.losses.coverage import coverage_loss
from cad_d4d.losses.objective import ObjectiveConfig, ShapeObjective
from cad_d4d.optimization.continuous import ContinuousOptimizer
from cad_d4d.optimization.discretization import get_discretization


def test_sdf_matches_sphere_distance(fine_sphere_target):
    sdf = fine_sphere_target.sdf
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = rng.uniform(0.3, 1.6, size=(2000, 1))
    x = torch.tensor(d * r)
    phi = to_numpy(sdf(x))
    exact = r[:, 0] - 1.0
    # tolerance: surface approximation of the bicubic sphere + interpolation (~h)
    assert np.max(np.abs(phi - exact)) < 0.5 * sdf.h + 0.01
    assert sdf(torch.zeros(1, 3)).item() < -0.9
    assert sdf(torch.tensor([[1.5, 0, 0.0]])).item() > 0.4


def test_sdf_is_small_on_target_surface(reachable_target):
    gt = reachable_target.ground_truth
    X, _, _ = SurfaceSampler(gt).evaluate_np(gt.values())
    phi = to_numpy(reachable_target.sdf(torch.tensor(X)))
    assert np.sqrt(np.mean(phi**2)) < 0.05 * reachable_target.sdf.h + 0.002


def test_coverage_zero_on_target_and_differentiable(reachable_target, coarse_sphere):
    gt = reachable_target.ground_truth
    sm = SurfaceSampler(gt, SamplingConfig(min_res=17))
    X = to_tensor(sm.evaluate_np(gt.values())[0])
    L, d, _ = coverage_loss(reachable_target.points, X, sm.tri_t)
    assert float(d.max()) < 0.01  # only chordal error of the proxy remains
    sm2 = SurfaceSampler(coarse_sphere)
    P = to_tensor(coarse_sphere.values()).requires_grad_(True)
    L2, _, _ = coverage_loss(reachable_target.points, torch.sparse.mm(sm2.G, P), sm2.tri_t)
    L2.backward()
    assert float(L2.detach()) > 10 * float(L)
    assert torch.isfinite(P.grad).all() and P.grad.abs().sum() > 0


def test_continuous_optimization_decreases_loss_and_stays_watertight(reachable_target, coarse_sphere):
    obj = ShapeObjective(reachable_target, ObjectiveConfig())
    opt = ContinuousOptimizer(obj)
    s = coarse_sphere.copy()
    L0 = obj.report(s)["fit"]
    logs = opt.run(s, 80)
    L1 = obj.report(s)["fit"]
    assert L1 < 0.05 * L0
    losses = [l["loss"] for l in logs]
    assert all(b <= a + 1e-15 for a, b in zip(losses, losses[1:])), "Armijo steps must be monotone"
    assert watertightness_error(s) < 1e-13


def test_preconditioned_descent_capacity_positive(reachable_target, coarse_sphere):
    opt = ContinuousOptimizer(ShapeObjective(reachable_target))
    gi = opt.gradient(coarse_sphere.copy())
    assert gi.descent_capacity > 0
    assert torch.all(gi.m > 0)
    from cad_d4d.optimization.preconditioner import lumped_mass
    disc = opt.obj.disc(coarse_sphere)
    m = lumped_mass(disc, gi.terms["w"])
    # lumped masses sum to ~1 (partition of unity x normalized area weights)
    assert abs(float(m.sum()) - 1.0) < 1e-9
    # the default semi-implicit operator M' = diag(m) + tau H is SPD and only damps:
    # g^T M'^{-1} g <= g^T diag(m)^{-1} g
    assert torch.allclose(gi.m, m)
    D_lumped = float((gi.g * gi.g / m[:, None]).sum())
    assert 0 < gi.descent_capacity <= D_lumped * (1 + 1e-12)


def test_jacobian_check_detects_flip_and_degeneracy():
    Xu = to_tensor([[1.0, 0, 0], [1.0, 0, 0]])
    Xv = to_tensor([[0, 1.0, 0], [0, 1.0, 0]])
    ref = torch.linalg.cross(Xu, Xv, dim=1)
    face = to_tensor([0, 0], dtype=torch.long)
    ok, _ = jacobian_check(Xu, Xv, face, ref, 1e-3)
    assert ok
    ok, info = jacobian_check(Xu, -Xv, face, ref, 1e-3)
    assert not ok and info["n_flipped"] == 2
    ok, info = jacobian_check(Xu, 1e-6 * Xv, face, ref, 1e-3)
    assert not ok and info["n_degenerate"] == 2


def test_self_intersection_detected_when_face_pushed_through(coarse_sphere):
    s = coarse_sphere.copy()
    disc = get_discretization(s)
    X, _, _ = disc.check.evaluate(to_tensor(s.values()))
    assert disc.count_self_intersections(X) == 0
    fid = disc.check.face_order[0]
    P = s.values()
    idx = s.dof_map.face_dof[fid].ravel()
    P[idx] *= -4.0  # push the face interior through the opposite face
    X2, _, _ = disc.check.evaluate(to_tensor(P))
    assert disc.count_self_intersections(X2) > 0


def test_validity_backtracking_rejects_folding_step(reachable_target, coarse_sphere):
    obj = ShapeObjective(reachable_target)
    opt = ContinuousOptimizer(obj)
    s = coarse_sphere.copy()
    disc = obj.disc(s)
    P = to_tensor(s.values())
    P_new, log = opt.step(s, P, eta=2000.0, disc=disc)  # absurd step: folds the surface
    assert log["invalid"] > 0
    assert P_new is not None and log["eta"] < 2000.0
    ok, _ = opt.is_valid(disc, P_new, opt.reference_normals(disc, P))
    assert ok


def test_exact_narrow_band_removes_trilinear_floor(reachable_target):
    """Near the surface the SDF is the exact distance to the dense target tessellation."""
    gt = reachable_target.ground_truth
    rng = np.random.default_rng(5)
    X = np.concatenate([gt.evaluate(f, rng.random((300, 2))) for f in gt.cx.faces])
    sdf = reachable_target.sdf
    band = np.abs(to_numpy(sdf(X)))
    sdf.exact_band = False
    try:
        tri = np.abs(to_numpy(sdf(X)))
    finally:
        sdf.exact_band = True
    assert np.sqrt(np.mean(band**2)) < 3e-4
    assert np.sqrt(np.mean(band**2)) < 0.3 * np.sqrt(np.mean(tri**2))
    # offset points: signed distance with the right sign and magnitude
    n_pts = X + 0.02 * (X / np.linalg.norm(X, axis=1, keepdims=True))
    assert np.all(to_numpy(sdf(n_pts)) > 0)
