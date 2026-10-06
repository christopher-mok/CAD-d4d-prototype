"""Fixed-grid linear elasticity (ersatz material) with differentiable compliance.

The explicit B-spline solid is mapped to a per-cell density rho in [0, 1] on a
fixed hexahedral grid (``occupancy.OccupancyGrid`` soft occupancy). Each cell is
a trilinear 8-node brick with SIMP-interpolated Young's modulus

    E_e = E_min + rho_e^p (E0 - E_min)

The static problem K(E) u = f is solved matrix-free by Jacobi-preconditioned
conjugate gradients on the compute device. Compliance  C = f . u  is
differentiable w.r.t. the cell moduli through the adjoint identity (the problem
is self-adjoint):

    dC / dE_e = - u_e^T K0 u_e        (K0: unit-modulus element stiffness)

so gradients reach the B-spline control points via rho(x; p) without a second
solve. Supports are clamped nodes; loads are nodal forces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ..device import DTYPE, get_device


def hex8_stiffness(h: float, nu: float) -> np.ndarray:
    """Unit-modulus stiffness (24 x 24) of a cubic trilinear brick of edge h (2x2x2 Gauss)."""
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1.0 / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    corners = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], float) * 2 - 1  # node order
    g = 1 / np.sqrt(3)
    K = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                dN = np.zeros((8, 3))  # derivatives w.r.t. reference coords
                for a, (cx, cy, cz) in enumerate(corners):
                    dN[a] = 0.125 * np.array([cx * (1 + cy * eta) * (1 + cz * zeta),
                                              cy * (1 + cx * xi) * (1 + cz * zeta),
                                              cz * (1 + cx * xi) * (1 + cy * eta)])
                dN *= 2.0 / h  # d/dx = (2/h) d/dxi
                B = np.zeros((6, 24))
                for a in range(8):
                    bx, by, bz = dN[a]
                    B[0, 3 * a], B[1, 3 * a + 1], B[2, 3 * a + 2] = bx, by, bz
                    B[3, 3 * a], B[3, 3 * a + 1] = by, bx
                    B[4, 3 * a + 1], B[4, 3 * a + 2] = bz, by
                    B[5, 3 * a], B[5, 3 * a + 2] = bz, bx
                K += B.T @ D @ B * (h / 2) ** 3
    return K


@dataclass
class FEMConfig:
    E0: float = 1.0
    E_min: float = 1e-6
    nu: float = 0.3
    penal: float = 3.0
    cg_tol: float = 1e-8
    cg_max_iter: int = 4000


class FixedGridFEM:
    """Linear elasticity on a regular grid of ``dims`` cells of edge ``h`` starting at ``lo``.

    ``supports(nodes) -> bool mask`` clamps nodes; ``loads(nodes) -> (n_nodes, 3)`` nodal forces.
    """

    def __init__(self, lo, h: float, dims, supports: Callable, loads: Callable, cfg: FEMConfig | None = None):
        self.cfg = cfg or FEMConfig()
        dev = get_device()
        self.h, self.dims = float(h), tuple(int(d) for d in dims)
        nx, ny, nz = self.dims
        lo = np.asarray(lo, float)
        ix, iy, iz = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
        self.nodes = lo + self.h * np.stack([ix.ravel(), iy.ravel(), iz.ravel()], 1)
        nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
        ei, ej, ek = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        ei, ej, ek = ei.ravel(), ej.ravel(), ek.ravel()
        conn = np.stack([nid(ei + di, ej + dj, ek + dk) for dk in (0, 1) for dj in (0, 1) for di in (0, 1)], 1)
        self.cell_centers = lo + self.h * (np.stack([ei, ej, ek], 1) + 0.5)
        edof = (3 * conn[:, :, None] + np.arange(3)[None, None, :]).reshape(len(conn), 24)
        self.edof = torch.as_tensor(edof, device=dev)
        self.n_dof = 3 * len(self.nodes)
        self.K0 = torch.as_tensor(hex8_stiffness(self.h, self.cfg.nu), dtype=DTYPE, device=dev)
        fixed_nodes = np.asarray(supports(self.nodes), bool)
        free = np.ones(self.n_dof, bool)
        free[(3 * np.flatnonzero(fixed_nodes)[:, None] + np.arange(3)).ravel()] = False
        self.free = torch.as_tensor(free, device=dev)
        self.f = torch.as_tensor(np.asarray(loads(self.nodes), float).ravel(), dtype=DTYPE, device=dev) * self.free
        self.diagK0 = torch.diagonal(self.K0)

    # -- operators -------------------------------------------------------------
    def modulus(self, rho: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return c.E_min + rho.clamp(0, 1) ** c.penal * (c.E0 - c.E_min)

    def matvec(self, u: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        ue = u[self.edof]                                   # (ne, 24)
        fe = (ue @ self.K0) * E[:, None]                    # K0 symmetric
        out = torch.zeros_like(u).index_add_(0, self.edof.reshape(-1), fe.reshape(-1))
        return out * self.free

    def solve(self, E: torch.Tensor, u0: torch.Tensor | None = None):
        """Jacobi-PCG for K(E) u = f on the free DOFs. Returns (u, iterations, relative residual)."""
        c = self.cfg
        diag = torch.zeros(self.n_dof, dtype=DTYPE, device=E.device).index_add_(
            0, self.edof.reshape(-1), (E[:, None] * self.diagK0[None, :]).reshape(-1))
        Minv = torch.where(self.free, 1.0 / diag.clamp(min=1e-30), torch.zeros_like(diag))
        u = torch.zeros_like(self.f) if u0 is None else u0.clone()
        r = self.f - self.matvec(u, E)
        z = Minv * r
        p = z.clone()
        rz = torch.dot(r, z)
        fnorm = torch.linalg.norm(self.f).clamp(min=1e-300)
        it = 0
        for it in range(1, c.cg_max_iter + 1):
            Ap = self.matvec(p, E)
            alpha = rz / torch.dot(p, Ap)
            u = u + alpha * p
            r = r - alpha * Ap
            if it % 10 == 0 and float(torch.linalg.norm(r) / fnorm) < c.cg_tol:
                break
            z = Minv * r
            rz_new = torch.dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return u, it, float(torch.linalg.norm(r) / fnorm)

    def compliance(self, rho: torch.Tensor) -> torch.Tensor:
        """Differentiable compliance f . u(rho) (rho: per-cell densities, flattened in cell order)."""
        return _Compliance.apply(self.modulus(rho.reshape(-1)), self)


class _Compliance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, E, fem: FixedGridFEM):
        with torch.no_grad():
            u, it, res = fem.solve(E, getattr(fem, "_warm", None))
        fem._warm = u.detach()
        fem.last_solve = {"iterations": it, "residual": res}
        ctx.fem = fem
        ctx.save_for_backward(u)
        return torch.dot(fem.f, u)

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        fem = ctx.fem
        ue = u[fem.edof]
        energy = ((ue @ fem.K0) * ue).sum(1)  # u_e^T K0 u_e
        return -grad_out * energy, None
