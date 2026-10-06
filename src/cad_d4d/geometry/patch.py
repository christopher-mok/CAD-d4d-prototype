"""Tensor-product B-spline patch evaluation (differentiable, PyTorch float64).

A patch is pure structure (degrees + knot vectors). Control points are passed
in, so the same structure can be evaluated for any parameter vector and
autograd flows to whatever tensor the control net was built from.

    S(u, v) = sum_a sum_b N_a(u) M_b(v) P[a, b]

For a fixed structure and fixed sample grid the basis matrices are constant
and are cached; evaluation is then a pair of matrix products.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from . import bspline_basis as bb

from ..device import DTYPE, get_device  # noqa: E402


@lru_cache(maxsize=4096)
def _cached_basis(knots_key: tuple, degree: int, params_key: tuple, deriv: int) -> np.ndarray:
    B = bb.basis_matrix(np.array(knots_key), degree, np.array(params_key), deriv)
    B.setflags(write=False)
    return B


def basis(knots, degree: int, params, deriv: int = 0) -> np.ndarray:
    """Cached dense basis matrix (read-only)."""
    return _cached_basis(tuple(np.asarray(knots, float).tolist()), degree,
                         tuple(np.asarray(params, float).tolist()), deriv)


@dataclass
class SplinePatch:
    """Structural description of a tensor-product B-spline patch.

    ``weights`` is reserved for a future rational (NURBS) extension and must
    currently be ``None`` (all weights equal to one).
    """

    knots_u: np.ndarray
    knots_v: np.ndarray
    degree_u: int = 3
    degree_v: int = 3
    weights: np.ndarray | None = None

    def __post_init__(self):
        self.knots_u = np.asarray(self.knots_u, dtype=float)
        self.knots_v = np.asarray(self.knots_v, dtype=float)
        if self.weights is not None:
            raise NotImplementedError("rational weights are not active in the baseline")

    @property
    def shape(self) -> tuple[int, int]:
        return (bb.num_basis(self.knots_u, self.degree_u), bb.num_basis(self.knots_v, self.degree_v))

    def basis_u(self, us, deriv=0):
        return basis(self.knots_u, self.degree_u, us, deriv)

    def basis_v(self, vs, deriv=0):
        return basis(self.knots_v, self.degree_v, vs, deriv)

    # -- evaluation --------------------------------------------------------
    def grid_operators(self, us, vs) -> "GridOperators":
        return GridOperators.build(self, us, vs)

    def evaluate_grid(self, net, us, vs):
        """Evaluate on the tensor grid ``us x vs``; returns (S, S_u, S_v), each (nu_s, nv_s, 3)."""
        return self.grid_operators(us, vs).evaluate(net)

    def evaluate_points(self, net: np.ndarray, uv: np.ndarray, deriv=(0, 0)) -> np.ndarray:
        """NumPy evaluation at scattered parameter pairs ``uv`` (N, 2)."""
        uv = np.atleast_2d(uv)
        Bu = bb.basis_matrix(self.knots_u, self.degree_u, uv[:, 0], deriv[0])
        Bv = bb.basis_matrix(self.knots_v, self.degree_v, uv[:, 1], deriv[1])
        return np.einsum("ia,ib,abk->ik", Bu, Bv, np.asarray(net))


@dataclass
class GridOperators:
    """Cached torch basis matrices for one patch structure and one sample grid."""

    Bu: torch.Tensor
    Bv: torch.Tensor
    dBu: torch.Tensor
    dBv: torch.Tensor
    us: np.ndarray = field(repr=False)
    vs: np.ndarray = field(repr=False)

    @staticmethod
    def build(patch: SplinePatch, us, vs) -> "GridOperators":
        t = lambda a: torch.as_tensor(np.array(a), dtype=DTYPE, device=get_device())
        return GridOperators(
            Bu=t(patch.basis_u(us)), Bv=t(patch.basis_v(vs)),
            dBu=t(patch.basis_u(us, 1)), dBv=t(patch.basis_v(vs, 1)),
            us=np.asarray(us), vs=np.asarray(vs),
        )

    def evaluate(self, net: torch.Tensor):
        net = torch.as_tensor(net, dtype=DTYPE, device=get_device())
        S = torch.einsum("ia,jb,abk->ijk", self.Bu, self.Bv, net)
        Su = torch.einsum("ia,jb,abk->ijk", self.dBu, self.Bv, net)
        Sv = torch.einsum("ia,jb,abk->ijk", self.Bu, self.dBv, net)
        return S, Su, Sv


def normals(Su: torch.Tensor, Sv: torch.Tensor, normalize: bool = True, eps: float = 1e-300):
    n = torch.linalg.cross(Su, Sv, dim=-1)
    if normalize:
        n = n / torch.clamp(torch.linalg.norm(n, dim=-1, keepdim=True), min=eps)
    return n
