"""Occupancy fields: winding number, signed distance, soft occupancy, volume."""
import math

import numpy as np
import pytest
import torch

from cad_d4d.device import to_numpy, to_tensor
from cad_d4d.geometry.builders import build_cube_complex
from cad_d4d.geometry.tessellation import SamplingConfig, SurfaceSampler
from cad_d4d.losses.objective import ObjectiveConfig, ShapeObjective
from cad_d4d.occupancy import OccupancyGrid, enclosed_volume, signed_distance, winding_number
from cad_d4d.optimization.continuous import ContinuousOptimizer
from cad_d4d.rewrites.local_refine import LocalRefine


@pytest.fixture(scope="module")
def fine_sphere():
    s = build_cube_complex(n_interior_knots=3)
    sm = SurfaceSampler(s, SamplingConfig(min_res=33, per_span=8, max_res=65))
    return s, sm


def proxy(state, sm, P=None):
    P = to_tensor(state.values()) if P is None else P
    return torch.sparse.mm(sm.G, P)


def test_winding_number_inside_outside(fine_sphere):
    s, sm = fine_sphere
    X = proxy(s, sm)
    rng = np.random.default_rng(0)
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    inside = to_tensor(d * rng.uniform(0.0, 0.9, (500, 1)))
    outside = to_tensor(d * rng.uniform(1.1, 3.0, (500, 1)))
    w_in, w_out = winding_number(inside, X, sm.tri_t), winding_number(outside, X, sm.tri_t)
    assert torch.all((w_in - 1).abs() < 1e-6) and torch.all(w_out.abs() < 1e-6)


def test_enclosed_volume_and_gradient(fine_sphere):
    s, sm = fine_sphere
    P = to_tensor(s.values()).requires_grad_(True)
    V = enclosed_volume(proxy(s, sm, P), sm.tri_t)
    assert abs(float(V.detach()) - 4 / 3 * math.pi) < 0.01 * 4 / 3 * math.pi
    (g,) = torch.autograd.grad(V, P)
    # finite-difference check along a random direction
    rng = np.random.default_rng(1)
    dP = to_tensor(rng.normal(size=s.values().shape))
    h = 1e-6
    with torch.no_grad():
        fd = (enclosed_volume(proxy(s, sm, P + h * dP), sm.tri_t)
              - enclosed_volume(proxy(s, sm, P - h * dP), sm.tri_t)) / (2 * h)
    assert abs(float(fd) - float((g * dP).sum())) < 1e-6 * max(1.0, abs(float(fd)))
    # uniform scaling by (1 + t) scales the volume by (1 + t)^3: dV/dt = 3 V
    assert abs(float((g * P.detach()).sum()) - 3 * float(V.detach())) < 1e-8


def test_signed_distance_on_grid(fine_sphere):
    s, sm = fine_sphere
    X = proxy(s, sm)
    Q = to_tensor(np.random.default_rng(2).uniform(-1.5, 1.5, (2000, 3)))
    sd = to_numpy(signed_distance(Q, X, sm.tri_t))
    exact = np.linalg.norm(to_numpy(Q), axis=1) - 1.0
    assert np.max(np.abs(sd - exact)) < 0.01  # bicubic sphere approximation + chord error


def test_soft_occupancy_volume_and_gradient_consistency(fine_sphere):
    s, sm = fine_sphere
    grid = OccupancyGrid([-1.3] * 3, [1.3] * 3, res=48, device=sm.G.device)
    P = to_tensor(s.values()).requires_grad_(True)
    X = proxy(s, sm, P)
    rho = grid.occupancy(X, sm.tri_t)
    V_grid = grid.volume(rho)
    V_exact = enclosed_volume(X, sm.tri_t)
    assert abs(float(V_grid.detach()) - float(V_exact.detach())) < 0.01 * float(V_exact.detach())
    # the occupancy volume responds to the control points like the exact volume
    g_grid = torch.autograd.grad(V_grid, P, retain_graph=True)[0]
    g_exact = torch.autograd.grad(V_exact, P)[0]
    cos = float((g_grid * g_exact).sum() / (g_grid.norm() * g_exact.norm()))
    assert cos > 0.98
    assert 0.0 <= float(rho.detach().min()) and float(rho.detach().max()) <= 1.0


def test_occupancy_unchanged_by_exact_rewrite():
    s = build_cube_complex()
    s2 = LocalRefine(sorted(s.cx.faces)[0], 0.5, 0.5, width=0.4, refine_knots=2).apply(s).state
    cfg = SamplingConfig(min_res=33, per_span=8, max_res=65)
    grid_pts = to_tensor(np.random.default_rng(3).uniform(-1.2, 1.2, (3000, 3)))
    sd = [to_numpy(signed_distance(grid_pts, proxy(st, sm), sm.tri_t))
          for st, sm in ((s, SurfaceSampler(s, cfg)), (s2, SurfaceSampler(s2, cfg)))]
    assert np.max(np.abs(sd[0] - sd[1])) < 2e-3  # equal up to proxy chord differences


def test_volume_loss_term_drives_volume_to_target(reachable_target, coarse_sphere):
    obj = ShapeObjective(reachable_target, ObjectiveConfig(lambda_sdf=0.0, lambda_coverage=0.0,
                                                           lambda_volume=1.0, lambda_fair=0.0))
    s = coarse_sphere.copy()
    before = float(obj.terms(s, to_tensor(s.values()))["volume_err"])
    ContinuousOptimizer(obj).run(s, 30)
    after = float(obj.terms(s, to_tensor(s.values()))["volume_err"])
    assert after < 0.01 * before
