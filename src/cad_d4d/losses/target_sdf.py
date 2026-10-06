"""Precomputed target signed-distance field with differentiable trilinear lookup.

Construction (offline, once per target):
  * exact unsigned distance from every grid node to a dense target
    tessellation (k nearest triangles by centroid, exact point-triangle test);
  * sign by flood fill: nodes farther than h/2 from the surface cannot be
    separated from their 6-neighbours by the surface, so connected components of
    such nodes touching the grid boundary are outside. Nodes within h/2 take
    the sign of (x - c) . n(c) at their closest surface point c.

Evaluation: trilinear interpolation of the grid, except inside a narrow band
around the surface (cells within ``band`` grid spacings), where every query
uses the *exact* signed distance to the dense target tessellation: each band
cell stores its k nearest triangles (precomputed), the closest point is found
exactly on the device, and the sign comes from the interpolated vertex normal
at the closest point. Trilinear interpolation cannot represent the kinks of a
distance field (patch creases, medial axis), which otherwise sets an error
floor of O(h) right where the optimizer converges.

    L_sdf = sum_i w_i phi_target(x_i)^2   (area-weighted)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from ..device import DTYPE, get_device
from ..geometry.distance import closest_point_on_triangle


def mesh_closest_points(Q: np.ndarray, V: np.ndarray, T: np.ndarray, k: int = 12, chunk: int = 200000):
    """Closest points on a triangle mesh: returns (distance, closest point, triangle index).

    Candidates: k nearest triangle centroids (KD-tree, all CPU cores); exact
    point-triangle closest points are evaluated in float64 on the device.
    """
    dev = get_device()
    cent = V[T].mean(axis=1)
    k = min(k, len(T))
    _, NN = cKDTree(cent).query(Q, k=k, workers=-1)
    NN = np.asarray(NN).reshape(len(Q), k)
    dist = np.empty(len(Q))
    closest = np.empty((len(Q), 3))
    tri_idx = np.empty(len(Q), dtype=np.int64)
    Vt = torch.as_tensor(V, dtype=DTYPE, device=dev)
    Tt = torch.as_tensor(T, dtype=torch.long, device=dev)
    for s in range(0, len(Q), chunk):
        nn = torch.as_tensor(NN[s: s + chunk], device=dev)
        q = torch.as_tensor(Q[s: s + chunk], dtype=DTYPE, device=dev)
        tri = Tt[nn]  # (n, k, 3)
        qt = q[:, None, :].expand(-1, k, -1)
        cp = closest_point_on_triangle(qt, Vt[tri[..., 0]], Vt[tri[..., 1]], Vt[tri[..., 2]])
        d = torch.linalg.norm(cp - qt, dim=-1)
        j = torch.argmin(d, dim=1)
        r = torch.arange(len(q), device=dev)
        dist[s: s + chunk] = d[r, j].cpu().numpy()
        closest[s: s + chunk] = cp[r, j].cpu().numpy()
        tri_idx[s: s + chunk] = nn[r, j].cpu().numpy()
    return dist, closest, tri_idx


def trilinear(values: torch.Tensor, origin: torch.Tensor, h: float, x: torch.Tensor) -> torch.Tensor:
    """Trilinear interpolation of a (nx, ny, nz[, c]) grid; points outside are clamped."""
    n = torch.tensor(values.shape[:3], dtype=DTYPE, device=x.device)
    g = (x - origin) / h
    g = torch.minimum(torch.clamp(g, min=0.0), n - 1 - 1e-9)
    i0 = torch.floor(g).long()
    t = g - i0
    out = 0.0
    for dx in (0, 1):
        wx = t[:, 0] if dx else 1 - t[:, 0]
        for dy in (0, 1):
            wy = t[:, 1] if dy else 1 - t[:, 1]
            for dz in (0, 1):
                wz = t[:, 2] if dz else 1 - t[:, 2]
                v = values[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
                w = wx * wy * wz
                out = out + (w[:, None] * v if v.dim() == 2 else w * v)
    return out


class SDFGrid:
    def __init__(self, values: np.ndarray, origin: np.ndarray, h: float):
        self.values_np = values
        dev = get_device()
        self.values = torch.as_tensor(values, dtype=DTYPE, device=dev)
        self.origin = torch.as_tensor(origin, dtype=DTYPE, device=dev)
        self.h = float(h)
        grad = np.stack(np.gradient(values, h), axis=-1)
        self.grad = torch.as_tensor(grad, dtype=DTYPE, device=dev)
        self.lo = self.origin
        self.hi = self.origin + h * (torch.tensor(values.shape, dtype=DTYPE, device=dev) - 1)

    @staticmethod
    def from_mesh(V: np.ndarray, T: np.ndarray, vertex_normals: np.ndarray, res: int = 64, pad: float = 0.3):
        lo, hi = V.min(0), V.max(0)
        ext = (hi - lo).max()
        lo = lo - pad * ext
        hi = hi + pad * ext
        h = (hi - lo).max() / (res - 1)
        dims = np.ceil((hi - lo) / h).astype(int) + 1
        axes = [lo[i] + h * np.arange(dims[i]) for i in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        dist, closest, tri_idx = mesh_closest_points(G, V, T)
        dist = dist.reshape(dims)
        far = dist > 0.5 * h
        labels, _ = ndimage.label(far)
        border = np.unique(np.concatenate([
            labels[[0, -1], :, :].ravel(), labels[:, [0, -1], :].ravel(), labels[:, :, [0, -1]].ravel()]))
        border = border[border > 0]
        outside = np.isin(labels, border)
        sign = np.where(outside, 1.0, -1.0)
        # Near-surface nodes: sign from the interpolated normal at the closest point.
        near = ~far.ravel()
        if near.any():
            tri = T[tri_idx[near]]
            a, b, c = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
            bary = _barycentric(closest[near], a, b, c)
            nrm = (bary[:, :1] * vertex_normals[tri[:, 0]] + bary[:, 1:2] * vertex_normals[tri[:, 1]]
                   + bary[:, 2:] * vertex_normals[tri[:, 2]])
            s_near = np.sign(np.einsum("ij,ij->i", G[near] - closest[near], nrm))
            s_near[s_near == 0] = 1.0
            sign_flat = sign.ravel()
            sign_flat[near] = s_near
            sign = sign_flat.reshape(dims)
        grid = SDFGrid(sign * dist, lo, h)
        grid.attach_band(V, T, vertex_normals)
        return grid

    def attach_band(self, V: np.ndarray, T: np.ndarray, vertex_normals: np.ndarray, band: float = 2.0,
                    k: int | None = None) -> None:
        """Precompute per-cell candidate triangles for exact near-surface evaluation.

        The closest triangle of any point in a cell lies within (cell half-diagonal +
        a few triangle sizes) of the cell center; ``k`` defaults to the expected
        number of triangle centroids in that disk (times a safety factor of 2).
        """
        if k is None:
            e = float(np.median(np.linalg.norm(V[T[:, 1]] - V[T[:, 0]], axis=1)))
            r = 0.87 * self.h + 2.0 * e
            k = int(np.clip(np.ceil(2.0 * np.pi * r**2 / (0.5 * e**2)), 24, 256))
        vals = self.values_np
        dims = np.array(vals.shape)
        corners = [np.abs(vals[i:dims[0] - 1 + i, j:dims[1] - 1 + j, l:dims[2] - 1 + l])
                   for i in (0, 1) for j in (0, 1) for l in (0, 1)]
        near = np.minimum.reduce(corners) <= band * self.h
        cells = np.argwhere(near)
        centers = self.origin.cpu().numpy() + (cells + 0.5) * self.h
        cent = V[T].mean(axis=1)
        k = min(k, len(T))
        _, nn = cKDTree(cent).query(centers, k=k, workers=-1)
        lookup = np.full(dims - 1, -1, dtype=np.int64)
        lookup[tuple(cells.T)] = np.arange(len(cells))
        dev = self.values.device
        self.band_lookup = torch.as_tensor(lookup, device=dev)
        self.band_tris = torch.as_tensor(np.asarray(nn).reshape(len(cells), k), device=dev)
        self.mesh_V = torch.as_tensor(V, dtype=DTYPE, device=dev)
        self.mesh_T = torch.as_tensor(T, dtype=torch.long, device=dev)
        self.mesh_N = torch.as_tensor(vertex_normals, dtype=DTYPE, device=dev)
        self.exact_band = True

    def __call__(self, x: torch.Tensor, exact: bool | None = None) -> torch.Tensor:
        """Signed distance. ``exact=None`` uses the narrow band if attached.

        The exact band is the right *measurement*; as an optimization objective the
        trilinear field is preferable: it blurs the kinks of the distance field at
        sharp target creases, which otherwise pull near-boundary control rows into
        folds (see README, "SDF for optimization vs. measurement").
        """
        x = torch.as_tensor(x, dtype=DTYPE).to(self.values.device)
        phi = trilinear(self.values, self.origin, self.h, x)
        outside = torch.linalg.norm(x - torch.minimum(torch.maximum(x, self.lo), self.hi), dim=1)
        phi = phi + outside
        if (getattr(self, "exact_band", False) if exact is None else exact) and hasattr(self, "band_lookup"):
            phi = self._exact_in_band(x, phi)
        return phi

    def _exact_in_band(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        dims = torch.tensor(self.band_lookup.shape, device=x.device)
        cell = torch.floor((x.detach() - self.origin) / self.h).long()
        inside = ((cell >= 0) & (cell < dims)).all(dim=1)
        cell = torch.minimum(torch.clamp(cell, min=0), dims - 1)
        row = self.band_lookup[cell[:, 0], cell[:, 1], cell[:, 2]]
        sel = (inside & (row >= 0)).nonzero(as_tuple=True)[0]
        if len(sel) == 0:
            return phi
        tri = self.mesh_T[self.band_tris[row[sel]]]  # (n, k, 3)
        k = tri.shape[1]
        q = x[sel][:, None, :].expand(-1, k, -1)
        V = self.mesh_V
        a, b, c = V[tri[..., 0]], V[tri[..., 1]], V[tri[..., 2]]
        cp = closest_point_on_triangle(q, a, b, c)
        d2 = ((q - cp) ** 2).sum(-1)
        j = d2.argmin(dim=1)
        r = torch.arange(len(sel), device=x.device)
        cpj, trij = cp[r, j], tri[r, j]
        # barycentric interpolation of vertex normals at the closest point (sign only)
        with torch.no_grad():
            A, B, C = V[trij[:, 0]], V[trij[:, 1]], V[trij[:, 2]]
            v0, v1, v2 = B - A, C - A, cpj - A
            d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
            d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
            den = torch.clamp(d00 * d11 - d01 * d01, min=1e-300)
            bv = ((d11 * d20 - d01 * d21) / den).clamp(0, 1)
            bw = ((d00 * d21 - d01 * d20) / den).clamp(0, 1)
            n = ((1 - bv - bw)[:, None] * self.mesh_N[trij[:, 0]] + bv[:, None] * self.mesh_N[trij[:, 1]]
                 + bw[:, None] * self.mesh_N[trij[:, 2]])
            sgn = torch.sign(((x[sel].detach() - cpj.detach()) * n).sum(1))
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        dist = torch.sqrt(d2[r, j] + 1e-30)
        out = phi.clone()
        out[sel] = sgn * dist
        return out

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=DTYPE).to(self.values.device)
        return trilinear(self.grad, self.origin, self.h, x)


def _barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-300, 1e-300, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.clip(np.stack([1 - v - w, v, w], 1), 0, 1)
