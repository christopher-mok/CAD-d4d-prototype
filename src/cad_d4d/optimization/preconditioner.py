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
"""
from __future__ import annotations

import numpy as np
import torch

from .discretization import Discretization


def lumped_mass(disc: Discretization, w: torch.Tensor, floor_rel: float = 1e-8) -> torch.Tensor:
    m = torch.sparse.mm(disc.G_abs.t(), w.detach()[:, None])[:, 0]
    return torch.clamp(m, min=floor_rel * float(m.max()))


def fairness_stiffness(disc: Discretization) -> torch.Tensor:
    """Row sums of |F^T F| (cached per structure)."""
    if getattr(disc, "_fair_rowsum", None) is None:
        H = (disc.F_np.T @ disc.F_np).tocsr()
        disc._fair_rowsum = torch.as_tensor(np.asarray(abs(H).sum(axis=1)).ravel(), dtype=torch.float64,
                                            device=disc.G_abs.device)
    return disc._fair_rowsum


def fairness_hessian_dense(disc: Discretization) -> torch.Tensor:
    """Dense F^T F on the device (cached per structure)."""
    if getattr(disc, "_fair_dense", None) is None:
        H = (disc.F_np.T @ disc.F_np).toarray()
        disc._fair_dense = torch.as_tensor(H, dtype=torch.float64, device=disc.G_abs.device)
    return disc._fair_dense


class Preconditioner:
    """Applies M^{-1} for one state of one structure."""

    def __init__(self, disc: Discretization, w: torch.Tensor, kind: str = "semi_implicit",
                 lambda_fair: float = 0.0, tau: float = 0.5):
        self.kind = kind
        m = lumped_mass(disc, w)
        self.m = m
        self._chol = None
        if kind == "lumped_mass":
            self.diag = m
        elif kind == "lumped_mass_fair":
            self.diag = m + tau * 2.0 * lambda_fair * fairness_stiffness(disc)
        elif kind == "identity":
            self.diag = torch.ones_like(m)
        elif kind == "semi_implicit":
            self.diag = None
            if lambda_fair > 0:
                A = tau * 2.0 * lambda_fair * fairness_hessian_dense(disc)
                A = A + torch.diag(m)
                self._chol = torch.linalg.cholesky(A)
            else:
                self.diag = m
        else:
            raise ValueError(kind)

    def solve(self, g: torch.Tensor) -> torch.Tensor:
        if self._chol is not None:
            return torch.cholesky_solve(g, self._chol)
        return g / self.diag[:, None]


def preconditioner_diag(disc: Discretization, w: torch.Tensor, kind: str = "lumped_mass_fair",
                        lambda_fair: float = 0.0, tau: float = 0.5) -> torch.Tensor:
    """Diagonal variants only (kept for diagnostics)."""
    p = Preconditioner(disc, w, kind if kind != "semi_implicit" else "lumped_mass_fair", lambda_fair, tau)
    return p.diag
