"""Milestone 2: one differentiable bicubic patch."""
import numpy as np
import torch

from cad_d4d.geometry import bspline_basis as bb
from cad_d4d.geometry.patch import SplinePatch, normals


def make_patch(rng):
    ku = bb.clamped_uniform_knots(2)
    kv = np.array([0, 0, 0, 0, 0.4, 1, 1, 1, 1.0])
    patch = SplinePatch(ku, kv)
    nu, nv = patch.shape
    grid = np.stack(np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij"), -1)
    net = np.concatenate([grid, 0.2 * rng.normal(size=(nu, nv, 1))], axis=-1)
    return patch, net


def de_boor_point(patch, net, u, v):
    """Independent evaluation: curve-of-curves via 1D basis evaluation."""
    rows = np.array([bb.evaluate_curve(patch.knots_v, 3, net[a], [v])[0] for a in range(net.shape[0])])
    return bb.evaluate_curve(patch.knots_u, 3, rows, [u])[0]


def test_grid_eval_matches_pointwise():
    rng = np.random.default_rng(1)
    patch, net = make_patch(rng)
    us, vs = np.linspace(0, 1, 9), np.linspace(0, 1, 7)
    S, _, _ = patch.evaluate_grid(torch.tensor(net), us, vs)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            assert np.allclose(S[i, j].cpu().numpy(), de_boor_point(patch, net, u, v), atol=1e-14)


def test_gradient_reaches_control_points_exactly():
    rng = np.random.default_rng(2)
    patch, net = make_patch(rng)
    P = torch.tensor(net, requires_grad=True)
    u, v = 0.37, 0.61
    S, _, _ = patch.evaluate_grid(P, [u], [v])
    S[0, 0, 2].backward()
    Nu = bb.basis_matrix(patch.knots_u, 3, [u])[0]
    Mv = bb.basis_matrix(patch.knots_v, 3, [v])[0]
    expected = np.outer(Nu, Mv)
    assert np.allclose(P.grad[..., 2].numpy(), expected, atol=1e-15)
    assert np.all(P.grad[..., :2].numpy() == 0)


def test_gradcheck():
    rng = np.random.default_rng(3)
    patch, net = make_patch(rng)
    ops = patch.grid_operators(np.linspace(0, 1, 5), np.linspace(0, 1, 4))
    P = torch.tensor(net, requires_grad=True)

    def f(P):
        S, Su, Sv = ops.evaluate(P)
        n = normals(Su, Sv)
        return (S**2).sum() + (n[..., 2] * S[..., 0]).sum()

    assert torch.autograd.gradcheck(f, (P,))


def test_derivatives_and_normals_fd():
    rng = np.random.default_rng(4)
    patch, net = make_patch(rng)
    us, vs = np.array([0.2, 0.55, 0.8]), np.array([0.15, 0.6])
    h = 1e-6
    _, Su, Sv = patch.evaluate_grid(torch.tensor(net), us, vs)
    Sp, _, _ = patch.evaluate_grid(torch.tensor(net), us + h, vs)
    Sm, _, _ = patch.evaluate_grid(torch.tensor(net), us - h, vs)
    assert torch.allclose(Su, (Sp - Sm) / (2 * h), atol=1e-7)
    Sp, _, _ = patch.evaluate_grid(torch.tensor(net), us, vs + h)
    Sm, _, _ = patch.evaluate_grid(torch.tensor(net), us, vs - h)
    assert torch.allclose(Sv, (Sp - Sm) / (2 * h), atol=1e-7)
    # A flat patch has unit normal +z.
    flat = net.copy()
    flat[..., 2] = 0
    _, Su, Sv = patch.evaluate_grid(torch.tensor(flat), us, vs)
    n = normals(Su, Sv)
    assert torch.allclose(n, torch.tensor([0, 0, 1.0], dtype=n.dtype, device=n.device).expand_as(n), atol=1e-14)


def test_basis_cache_reuse():
    patch = SplinePatch(bb.clamped_uniform_knots(1), bb.clamped_uniform_knots(1))
    a = patch.basis_u(np.linspace(0, 1, 5))
    b = patch.basis_u(np.linspace(0, 1, 5))
    assert a is b
