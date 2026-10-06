"""Occupancy of the explicit B-spline solid on a fixed grid (bridge to fixed-grid FEM).

The B-spline shell stays the source of truth. For a fixed background grid we
derive, from the current proxy tessellation:

* ``winding_number(Q)``      generalized winding number (Jacobson et al. 2013),
                             sum of signed solid angles / 4 pi: ~1 inside,
                             ~0 outside, robust for the (closed) proxy;
* ``signed_distance(Q)``     exact distance to the proxy (k nearest triangles by
                             centroid, exact closest points, differentiable in
                             the control points), signed by the winding number;
* ``soft_occupancy(Q)``      rho = sigmoid(-sd / eps): a smooth density whose
                             derivative w.r.t. the control points is supported
                             in an eps-band around the surface -- what a
                             fixed-grid (ersatz material) FEM solver consumes;
* ``enclosed_volume(X, tri)`` exact proxy volume by the divergence theorem.

Everything runs on the compute device; heavy loops are chunked.
"""
from __future__ import annotations

import math

import torch

from ..geometry.distance import closest_point_on_triangle
from ..losses.coverage import nearest_centroids


def solid_angles(Q: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Signed solid angle of every triangle seen from every query: (len(Q), len(A)).

    Van Oosterom & Strackee: tan(Omega/2) = a.(b x c) / (|a||b||c| + (a.b)|c| + (a.c)|b| + (b.c)|a|).
    """
    a = A[None] - Q[:, None]
    b = B[None] - Q[:, None]
    c = C[None] - Q[:, None]
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    det = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
    den = la * lb * lc + (a * b).sum(-1) * lc + (a * c).sum(-1) * lb + (b * c).sum(-1) * la
    return 2.0 * torch.atan2(det, den)


def winding_number(Q: torch.Tensor, X: torch.Tensor, tri: torch.Tensor, q_chunk: int = 2048,
                   t_chunk: int = 8192) -> torch.Tensor:
    """Generalized winding number of the oriented mesh (X, tri) at queries Q (no gradient)."""
    with torch.no_grad():
        A, B, C = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
        out = torch.zeros(len(Q), dtype=X.dtype, device=X.device)
        for s in range(0, len(Q), q_chunk):
            q = Q[s: s + q_chunk]
            acc = torch.zeros(len(q), dtype=X.dtype, device=X.device)
            for t in range(0, len(A), t_chunk):
                acc += solid_angles(q, A[t: t + t_chunk], B[t: t + t_chunk], C[t: t + t_chunk]).sum(1)
            out[s: s + q_chunk] = acc / (4.0 * math.pi)
        return out


def unsigned_distance(Q: torch.Tensor, X: torch.Tensor, tri: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Distance from Q to the mesh, differentiable w.r.t. X (candidate search is detached)."""
    A, B, C = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    nn = nearest_centroids(Q, ((A + B + C) / 3).detach(), k)
    Qk = Q[:, None, :].expand(-1, nn.shape[1], -1)
    cp = closest_point_on_triangle(Qk, A[nn], B[nn], C[nn])
    return torch.sqrt(((cp - Qk) ** 2).sum(-1).min(dim=1).values + 1e-30)


def signed_distance(Q: torch.Tensor, X: torch.Tensor, tri: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Negative inside: sign from the winding number, magnitude from exact closest points."""
    sign = torch.where(winding_number(Q, X, tri) > 0.5, -1.0, 1.0).to(X.dtype)
    return sign * unsigned_distance(Q, X, tri, k)


def soft_occupancy(Q: torch.Tensor, X: torch.Tensor, tri: torch.Tensor, eps: float, k: int = 8) -> torch.Tensor:
    return torch.sigmoid(-signed_distance(Q, X, tri, k) / eps)


def enclosed_volume(X: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    """Exact volume of the closed, outward-oriented proxy (divergence theorem)."""
    A, B, C = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    return (A * torch.linalg.cross(B, C, dim=1)).sum() / 6.0


class OccupancyGrid:
    """A fixed axis-aligned background grid (cell centers), e.g. for ersatz-material FEM."""

    def __init__(self, lo, hi, res: int, device=None, dtype=torch.float64):
        lo = torch.as_tensor(lo, dtype=dtype, device=device)
        hi = torch.as_tensor(hi, dtype=dtype, device=device)
        self.h = float((hi - lo).max()) / res
        dims = torch.ceil((hi - lo) / self.h).long().clamp(min=1)
        self.dims = tuple(int(d) for d in dims)
        axes = [lo[i] + self.h * (torch.arange(self.dims[i], dtype=dtype, device=device) + 0.5) for i in range(3)]
        self.centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        self.cell_volume = self.h**3

    def occupancy(self, X: torch.Tensor, tri: torch.Tensor, eps: float | None = None, k: int = 8) -> torch.Tensor:
        """Soft occupancy per cell (eps defaults to half a cell)."""
        eps = 0.5 * self.h if eps is None else eps
        return soft_occupancy(self.centers, X, tri, eps, k).reshape(self.dims)

    def volume(self, rho: torch.Tensor) -> torch.Tensor:
        return rho.sum() * self.cell_volume
