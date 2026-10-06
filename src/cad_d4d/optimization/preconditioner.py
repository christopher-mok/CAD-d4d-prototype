"""Preconditioners  d = M^{-1} g  for the continuous step  p <- p - eta d.

``lumped_mass``
    The loss is an area-weighted integral, so the gradient w.r.t. a control
    point scales with the surface area that control point influences. Dividing
    by the lumped mass

        m_k = sum_s |G_sk| w_s        (w_s: normalized area weight of sample s)

    turns  M^{-1} g  into a local *average* of the pointwise loss gradient,
    comparable between coarse (large support) and refined (small support)
    control points.

``lumped_mass_fair``
    Lumped mass plus a Gershgorin bound of the fairness Hessian on the diagonal,
    m'_k = m_k + tau sum_j |H_kj|. Stable but crude: it damps *whole* DOFs that
    any stiff term touches (e.g. every control point next to a split line under
    the crease penalty), which slows the fit there.

``semi_implicit`` (default)
    M' = diag(m) + tau H,  H = 2 lambda_fair F^T F  (the constant Hessian of the
    fairness + crease quadratic form). One semi-implicit (Sobolev-type) step:
    only the stiff *directions* are damped, the fit directions keep their
    lumped-mass scaling. eig(M'^{-1} H) < 1/tau, so steps of size tau are stable
    for the quadratic terms. The system is small (canonical DOFs), so it is
    factored densely (Cholesky) on the device every step.

``consistent_mass``
    M' = G^T W G + tau H: the *consistent* (unlumped) L2 mass of the sample
    operator G with area weights W, plus the semi-implicit fairness term.
    The lumped mass is exactly the row-sum lumping of G^T W G (B-spline bases
    are non-negative and sum to one), and for bicubic tensor bases that lumping
    is a weak approximation: the mass matrix has eigenvalues far below its row
    sums for oscillatory control-point modes, so lumped steps are too short
    in exactly the directions a refined fit needs. The consistent mass makes
    M^{-1} g the L2 Riesz representative of the gradient, so the step (and the
    descent capacity D) does not depend on how the space is parametrized.
    ``mass_blend`` = beta uses (1 - beta) G^T W G + beta diag(m) instead: beta = 1 is
    the lumped mass; small beta keeps most of the speed-up while bounding the
    normalized eigenvalues below by beta (shorter, more local oscillatory steps).
"""
from __future__ import annotations

import numpy as np
import torch

from .discretization import Discretization


def lumped_mass(disc: Discretization, w: torch.Tensor, floor_rel: float = 1e-8) -> torch.Tensor:
    m = torch.sparse.mm(disc.G_abs.t(), w.detach()[:, None])[:, 0]
    return torch.clamp(m, min=floor_rel * float(m.max()))


def consistent_mass(disc: Discretization, w: torch.Tensor) -> torch.Tensor:
    """Dense G^T diag(w) G on the device."""
    G = disc.sampler.G.coalesce()
    idx, val = G.indices(), G.values()
    Gw = torch.sparse_coo_tensor(idx, val * w.detach()[idx[0]], G.shape)
    return torch.sparse.mm(G.t(), Gw).to_dense()


def fairness_stiffness(disc: Discretization) -> torch.Tensor:
    """Row sums of |F^T F| (cached per structure)."""
    if getattr(disc, "_fair_rowsum", None) is None:
        H = (disc.F_np.T @ disc.F_np).tocsr()
        disc._fair_rowsum = torch.as_tensor(np.asarray(abs(H).sum(axis=1)).ravel(), dtype=torch.float64,
                                            device=disc.G_abs.device)
    return disc._fair_rowsum


def scaled_fairness_hessian(disc: Discretization, scale: float) -> torch.Tensor:
    """Dense ``scale * F^T F`` on the device (cached per structure and scale)."""
    cached = getattr(disc, "_fair_dense", None)
    if cached is None or cached[0] != scale:
        H = (disc.F_np.T @ disc.F_np).toarray() * scale
        disc._fair_dense = (scale, torch.as_tensor(H, dtype=torch.float64, device=disc.G_abs.device))
    return disc._fair_dense[1]


class Preconditioner:
    """Applies M^{-1} for one state of one structure."""

    def __init__(self, disc: Discretization, w: torch.Tensor, kind: str = "semi_implicit",
                 lambda_fair: float = 0.0, tau: float = 0.5, mass_blend: float = 0.0):
        self.kind = kind
        m = lumped_mass(disc, w)
        self.m = m
        self._chol = None
        self._A = None  # dense operator behind ``_chol`` (for restricted solves)
        if kind == "lumped_mass":
            self.diag = m
        elif kind == "lumped_mass_fair":
            self.diag = m + tau * 2.0 * lambda_fair * fairness_stiffness(disc)
        elif kind == "identity":
            self.diag = torch.ones_like(m)
        elif kind == "consistent_mass":
            self.diag = None
            A = consistent_mass(disc, w)
            if mass_blend > 0:
                A = (1.0 - mass_blend) * A
                A.diagonal().add_(mass_blend * m)
            if lambda_fair > 0:
                A = A + scaled_fairness_hessian(disc, tau * 2.0 * lambda_fair)
            A.diagonal().add_(1e-8 * float(m.max()))  # DOFs no sample sees (should not happen)
            self._A, self._chol = A, torch.linalg.cholesky(A)
        elif kind == "semi_implicit":
            self.diag = None
            if lambda_fair > 0:
                A = scaled_fairness_hessian(disc, tau * 2.0 * lambda_fair).clone()
                A.diagonal().add_(m)
                self._A, self._chol = A, torch.linalg.cholesky(A)
            else:
                self.diag = m
        else:
            raise ValueError(kind)

    def solve(self, g: torch.Tensor) -> torch.Tensor:
        if self._chol is not None:
            return torch.cholesky_solve(g, self._chol)
        return g / self.diag[:, None]

    def solve_restricted(self, g: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
        """M_ff^{-1} g_f on the ``free`` DOFs, zero on the others: the preconditioned step with
        some DOFs frozen. Re-solving (instead of masking M^{-1} g) keeps it a descent direction
        for a non-diagonal M."""
        out = torch.zeros_like(g)
        idx = free.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return out
        if self._A is not None:
            out[idx] = torch.cholesky_solve(g[idx], torch.linalg.cholesky(self._A[idx][:, idx]))
        else:
            out[idx] = g[idx] / self.diag[idx, None]
        return out
