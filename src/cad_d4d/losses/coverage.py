"""Coverage: target samples y_j must be close to the current surface.

    L_cov = mean_j d(y_j, current_proxy)^2

d is the exact point-to-triangle distance to the current proxy tessellation,
minimized over a candidate set of triangles found on detached positions:
the k nearest triangle centroids *and* the triangles incident to the nearest
proxy vertices. Centroids alone are not enough: with triangles of very
different sizes (e.g. after merges), a large triangle can be the closest one
while its centroid is far away; missing it makes the loss discontinuous (the
candidate set changes under tiny motions) and can trap the line search.
The search runs in float64 (a float32 search resolves near-ties by rounding
noise, which made the candidate set -- and the loss -- jump).
Gradients flow to the closest points and hence to the DOFs.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..geometry.distance import closest_point_on_triangle


def nearest_centroids(Y: torch.Tensor, cent: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    """Indices (len(Y), k) of the k nearest points of ``cent`` to every query (float64, exact)."""
    k = min(k, len(cent))
    if cent.is_cuda:
        c = cent.detach()
        out = []
        for s in range(0, len(Y), chunk):
            d = torch.cdist(Y[s: s + chunk].detach(), c)
            out.append(d.topk(k, dim=1, largest=False).indices)
        return torch.cat(out)
    _, nn = cKDTree(cent.detach().cpu().numpy()).query(Y.detach().cpu().numpy(), k=k, workers=-1)
    return torch.as_tensor(np.asarray(nn).reshape(len(Y), k), device=Y.device)


def vertex_star(tri: torch.Tensor, n_vertices: int, max_valence: int = 8) -> torch.Tensor:
    """(n_vertices, max_valence) incident triangle indices per vertex, padded by repetition."""
    t = tri.cpu().numpy()
    verts = t.ravel()
    tris = np.repeat(np.arange(len(t)), 3)
    order = np.argsort(verts, kind="stable")
    verts, tris = verts[order], tris[order]
    starts = np.searchsorted(verts, np.arange(n_vertices))
    counts = np.bincount(verts, minlength=n_vertices)
    slot = np.arange(max_valence)[None, :]
    # j-th incident triangle, repeating the last one when a vertex has fewer than max_valence
    idx = starts[:, None] + np.minimum(slot, np.maximum(counts, 1)[:, None] - 1)
    out = tris[np.minimum(idx, len(tris) - 1)]
    out[counts == 0] = 0
    return torch.as_tensor(out, device=tri.device)


def candidate_triangles(Y: torch.Tensor, X: torch.Tensor, tri: torch.Tensor, k: int = 8, k_vertices: int = 2,
                        star: torch.Tensor | None = None) -> torch.Tensor:
    """Candidate triangles per query: k nearest centroids + stars of the k_vertices nearest vertices."""
    A, B, C = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    by_centroid = nearest_centroids(Y, (A + B + C) / 3, k)
    star = vertex_star(tri, len(X)) if star is None else star
    nv = nearest_centroids(Y, X, k_vertices)                    # nearest proxy vertices
    by_vertex = star[nv].reshape(len(Y), -1)
    return torch.cat([by_centroid, by_vertex], 1)


def coverage_distances(Y: torch.Tensor, X: torch.Tensor, tri: torch.Tensor, k: int = 8,
                       star: torch.Tensor | None = None):
    """Per-target-point distance to the proxy mesh (X, tri). Returns (d, nearest triangle)."""
    tri = torch.as_tensor(tri, device=X.device)
    A, B, C = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    nn = candidate_triangles(Y, X, tri, k, star=star)
    m = nn.shape[1]
    Yk = Y[:, None, :].expand(-1, m, -1)
    cp = closest_point_on_triangle(Yk, A[nn], B[nn], C[nn])
    d2 = ((cp - Yk) ** 2).sum(-1)
    d2min, j = d2.min(dim=1)
    return torch.sqrt(d2min + 1e-30), nn[torch.arange(len(Y), device=Y.device), j]


def coverage_loss(Y: torch.Tensor, X: torch.Tensor, tri: torch.Tensor, k: int = 8, star: torch.Tensor | None = None):
    d, nearest = coverage_distances(Y, X, tri, k, star)
    return (d**2).mean(), d, nearest
