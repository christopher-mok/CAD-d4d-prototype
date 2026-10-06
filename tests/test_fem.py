"""Fixed-grid FEM: element stiffness, exact bar solution, compliance gradients, coupling to the B-spline solid."""
import numpy as np
import torch

from cad_d4d.device import to_tensor
from cad_d4d.geometry.builders import build_cube_complex, sphere_map
from cad_d4d.geometry.tessellation import SamplingConfig, SurfaceSampler
from cad_d4d.occupancy import OccupancyGrid
from cad_d4d.physics.fem import FEMConfig, FixedGridFEM, hex8_stiffness


def test_element_stiffness_symmetric_with_six_rigid_modes():
    K = hex8_stiffness(0.5, 0.3)
    w = np.linalg.eigvalsh(K)
    assert np.abs(K - K.T).max() < 1e-14
    assert np.sum(np.abs(w) < 1e-10) == 6 and w.min() > -1e-10


def bar(dims=(16, 4, 4), h=0.25, sigma=0.01, nu=0.0):
    L = dims[0] * h

    def loads(n):
        F = np.zeros_like(n)
        right = np.abs(n[:, 0] - L) < 1e-9
        wgt = np.ones(right.sum())
        for c, extent in ((n[right, 1], dims[1] * h), (n[right, 2], dims[2] * h)):
            wgt *= np.where((np.abs(c) < 1e-9) | (np.abs(c - extent) < 1e-9), 0.5, 1.0)
        F[right, 0] = sigma * h * h * wgt  # consistent nodal loads of a uniform traction
        return F

    return FixedGridFEM([0, 0, 0], h, dims, lambda n: n[:, 0] < 1e-9, loads, FEMConfig(nu=nu)), L


def test_bar_in_tension_matches_analytic_solution():
    fem, L = bar()
    E = fem.modulus(torch.ones(int(np.prod(fem.dims)), dtype=torch.float64, device=fem.f.device))
    u, it, res = fem.solve(E)
    ux = u.reshape(-1, 3)[:, 0].cpu().numpy()
    tip = ux[np.abs(fem.nodes[:, 0] - L) < 1e-9]
    assert res < 1e-8
    assert np.allclose(tip, 0.01 * L, atol=1e-10)  # sigma L / E, exact for trilinear elements (nu = 0)


def test_compliance_gradient_matches_finite_differences():
    fem, _ = bar(dims=(6, 3, 3), h=0.5, nu=0.3)
    fem.cfg.cg_tol = 1e-13
    rng = np.random.default_rng(0)
    rho = torch.tensor(rng.uniform(0.3, 1.0, int(np.prod(fem.dims))), device=fem.f.device, requires_grad=True)
    C = fem.compliance(rho)
    (g,) = torch.autograd.grad(C, rho)
    d = torch.tensor(rng.normal(size=rho.shape), device=rho.device)
    eps = 1e-6
    with torch.no_grad():
        fd = (fem.compliance(rho + eps * d) - fem.compliance(rho - eps * d)) / (2 * eps)
    assert abs(float(fd) - float((g * d).sum())) < 1e-6 * abs(float(fd))
    assert torch.all(g <= 0)  # adding material never increases compliance


def test_compliance_differentiable_through_bspline_occupancy():
    """Control points -> proxy surface -> soft occupancy -> moduli -> compliance.

    The body is off-center in the grid: for a symmetric setup many cell centers lie on
    the medial axis (two closest points tie), where the distance is not differentiable
    and central differences average the two one-sided derivatives.
    """
    s = build_cube_complex(sphere_map(0.9, center=(0.031, -0.017, 0.023)), n_interior_knots=1)
    sm = SurfaceSampler(s, SamplingConfig(min_res=17, per_span=4, max_res=33))
    grid = OccupancyGrid([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], res=10, device=sm.G.device)
    fem = FixedGridFEM([-1.0, -1.0, -1.0], grid.h, grid.dims, lambda n: n[:, 0] < -1 + 1e-9,
                       lambda n: np.where((np.abs(n[:, 0] - 1) < 1e-9)[:, None], [[0.0, 0.0, -1e-3]], 0.0),
                       FEMConfig(cg_tol=1e-12, E_min=1e-3))
    P = to_tensor(s.values()).requires_grad_(True)

    def C_of(P):
        rho = grid.occupancy(torch.sparse.mm(sm.G, P), sm.tri_t, eps=0.15)
        return fem.compliance(rho.reshape(-1))

    C = C_of(P)
    (g,) = torch.autograd.grad(C, P)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0
    d = torch.zeros_like(P)
    d[:, 0] = 1.0  # stretch-free translation along x changes how much material sits near the supports
    eps = 1e-5
    with torch.no_grad():
        fd = (C_of(P + eps * d) - C_of(P - eps * d)) / (2 * eps)
    assert abs(float(fd) - float((g * d).sum())) < 2e-3 * abs(float(fd)) + 1e-12
    # growing the body (scaling up) adds material -> lower compliance
    assert float((g * P.detach()).sum()) < 0


def test_compliance_objective_optimizes_shape():
    """ShapeObjective without a target, driven by compliance + volume: J decreases, validity holds."""
    from cad_d4d.losses.objective import ObjectiveConfig, ShapeObjective
    from cad_d4d.optimization.continuous import ContinuousOptimizer
    from cad_d4d.physics import ComplianceTerm, fem_on_grid
    s = build_cube_complex(sphere_map(0.8))
    grid = OccupancyGrid([-1.0, -0.6, -0.6], [1.0, 0.6, 0.6], res=12, device=to_tensor([0.0]).device)
    fem = fem_on_grid(grid, lambda n: n[:, 0] < -0.75,
                      lambda n: np.where(((n[:, 0] > 0.6) & (np.abs(n[:, 1]) < 0.2) & (np.abs(n[:, 2]) < 0.2))[:, None],
                                         [[0.0, 0.0, -1e-4]], 0.0), FEMConfig(E_min=1e-3, cg_tol=1e-9))
    term = ComplianceTerm(fem, grid, eps=0.5 * grid.h)
    obj = ShapeObjective(None, ObjectiveConfig(lambda_sdf=0, lambda_coverage=0, lambda_fair=0, lambda_volume=10.0,
                                               volume_target=4 / 3 * np.pi * 0.8**3, physics=[term]))
    logs = ContinuousOptimizer(obj).run(s, 8)
    assert logs[-1]["new_loss"] < 0.9 * logs[0]["loss"]
    assert all(b <= a + 1e-12 for a, b in zip([l["loss"] for l in logs], [l["new_loss"] for l in logs]))
