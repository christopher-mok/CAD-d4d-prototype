"""Coverage: target samples y_j must be close to the current surface.

    L_cov = mean_j d(y_j, current_proxy)^2

d is the exact point-to-triangle distance to the current proxy tessellation,
restricted to the k triangles with nearest centroids (found on detached
positions: brute force in float32 on the GPU, a KD-tree on the CPU).
Gradients flow to the closest points and hence to the DOFs.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..geometry.distance import closest_point_on_triangle


def nearest_centroids(Y: torch.Tensor, cent: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    """Indices (len(Y), k) of the k nearest centroids to every query point."""
    k = min(k, len(cent))
    if cent.is_cuda:
        c32 = cent.detach().float()
        out = []
        for s in range(0, len(Y), chunk):
            d = torch.cdist(Y[s: s + chunk].detach().float(), c32)
            out.append(d.topk(k, dim=1, largest=False).indices)
        return torch.cat(out)
    _, nn = cKDTree(cent.detach().cpu().numpy()).query(Y.detach().cpu().numpy(), k=k, workers=-1)
    return torch.as_tensor(np.asarray(nn).reshape(len(Y), k), device=Y.device)


def coverage_distances(Y: torch.Tensor, X: torch.Tensor, tri: torch.Tensor, k: int = 6):
    """Per-target-point distance to the proxy mesh (X, tri). Returns (d, nearest triangle)."""
    tri = torch.as_tensor(tri, device=X.device)
    A, B, C = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    nn = nearest_centroids(Y, (A + B + C) / 3, k)
    k = nn.shape[1]
    Yk = Y[:, None, :].expand(-1, k, -1)
    cp = closest_point_on_triangle(Yk, A[nn], B[nn], C[nn])
    d2 = ((cp - Yk) ** 2).sum(-1)
    d2min, j = d2.min(dim=1)
    return torch.sqrt(d2min + 1e-30), nn[torch.arange(len(Y), device=Y.device), j]


def coverage_loss(Y: torch.Tensor, X: torch.Tensor, tri: torch.Tensor, k: int = 6):
    d, nearest = coverage_distances(Y, X, tri, k)
    return (d**2).mean(), d, nearest
