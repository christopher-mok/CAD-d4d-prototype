"""Geometric validity of a candidate configuration (torch, runs on the compute device).

Watertightness is structural; these checks catch what it does not:

* degenerate Jacobian:  |S_u x S_v| <= eps_rel * (reference mean of that face)
* orientation flips:    n_new . n_ref <= 0 at any sample (foldover)
* nonlocal self-intersection of the temporary tessellation: brute-force AABB
  broad phase (float32, chunked) and exact segment/triangle narrow phase
  (float64). Grid neighbours inside one face are local and skipped.
  Intersections between two adjacent faces are discarded only when the hit
  point lies on their common seam (within one local edge length of the
  boundary samples of *both* faces) -- these are artifacts of non-matching
  boundary sampling, not fold-throughs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ValidityConfig:
    eps_jacobian_rel: float = 1e-3
    check_orientation: bool = True
    check_self_intersection: bool = True


def jacobian_check(Xu: torch.Tensor, Xv: torch.Tensor, sample_face: torch.Tensor,
                   ref_normals: torch.Tensor | None, eps_rel: float, check_orientation: bool = True):
    """Returns (ok, info) for per-sample Jacobians of the patch complex."""
    n = torch.linalg.cross(Xu, Xv, dim=1)
    norm = torch.linalg.norm(n, dim=1)
    nf = int(sample_face.max()) + 1
    src = torch.linalg.norm(ref_normals, dim=1) if ref_normals is not None else norm
    cnt = torch.bincount(sample_face, minlength=nf).clamp(min=1).to(norm.dtype)
    scale = torch.zeros(nf, dtype=norm.dtype, device=norm.device).index_add_(0, sample_face, src) / cnt
    degenerate = norm <= eps_rel * scale[sample_face]
    flipped = torch.zeros_like(degenerate)
    if check_orientation and ref_normals is not None:
        flipped = (n * ref_normals).sum(1) <= 0
    n_deg, n_flip = int(degenerate.sum()), int(flipped.sum())
    return n_deg == 0 and n_flip == 0, {"n_degenerate": n_deg, "n_flipped": n_flip,
                                        "min_jacobian": float(norm.min()) if len(norm) else 0.0}


def _segment_hits_triangle(p, q, v0, v1, v2, eps=1e-10):
    """Strict segment/triangle intersection (Moller-Trumbore). Returns (hit mask, hit point)."""
    d = q - p
    e1, e2 = v1 - v0, v2 - v0
    h = torch.linalg.cross(d, e2, dim=-1)
    a = (e1 * h).sum(-1)
    ok = a.abs() > 1e-14
    f = torch.where(ok, 1.0 / torch.where(ok, a, torch.ones_like(a)), torch.zeros_like(a))
    s = p - v0
    u = f * (s * h).sum(-1)
    qv = torch.linalg.cross(s, e1, dim=-1)
    v = f * (d * qv).sum(-1)
    t = f * (e2 * qv).sum(-1)
    hit = ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)
    return hit, p + t[:, None] * d


def candidate_pairs(A, B, C, chunk: int = 2048) -> torch.Tensor:
    """Pairs (i < j) of triangles with overlapping (padded) bounding boxes."""
    lo = torch.minimum(torch.minimum(A, B), C).float()
    hi = torch.maximum(torch.maximum(A, B), C).float()
    pad = 1e-6 * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    T = len(A)
    out = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        ov = ((lo[s:e, None, :] <= hi[None, :, :]) & (hi[s:e, None, :] >= lo[None, :, :])).all(-1)
        i, j = ov.nonzero(as_tuple=True)
        i = i + s
        keep = j > i
        out.append(torch.stack([i[keep], j[keep]], 1))
    return torch.cat(out) if out else torch.zeros((0, 2), dtype=torch.long, device=A.device)


def self_intersections(X: torch.Tensor, tri: torch.Tensor, tri_face: torch.Tensor, tri_cell: torch.Tensor,
                       adjacent_faces: torch.Tensor | None, sample_face: torch.Tensor | None = None,
                       sample_boundary: torch.Tensor | None = None, return_hits: bool = False):
    """Number of intersecting nonlocal triangle pairs in the tessellation."""
    A, B, C = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    pairs = candidate_pairs(A, B, C)
    if len(pairs) == 0:
        return (0, []) if return_hits else 0
    i, j = pairs[:, 0], pairs[:, 1]
    same = tri_face[i] == tri_face[j]
    near_cell = (tri_cell[i] - tri_cell[j]).abs().max(dim=1).values <= 1
    keep = ~(same & near_cell)
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return (0, []) if return_hits else 0
    hit = torch.zeros(len(i), dtype=torch.bool, device=X.device)
    point = torch.zeros((len(i), 3), dtype=X.dtype, device=X.device)
    Ti, Tj = (A[i], B[i], C[i]), (A[j], B[j], C[j])
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for P, Q in ((Ti, Tj), (Tj, Ti)):
            h, pt = _segment_hits_triangle(P[a], P[b], *Q)
            new = h & ~hit
            point[new] = pt[new]
            hit |= h
    idx = hit.nonzero(as_tuple=True)[0]
    if len(idx) and adjacent_faces is not None and sample_boundary is not None:
        fi, fj = tri_face[i[idx]], tri_face[j[idx]]
        cand = (fi != fj) & adjacent_faces[fi, fj]
        if cand.any():
            ck = cand.nonzero(as_tuple=True)[0]
            ti, tj = i[idx[ck]], j[idx[ck]]
            edges = torch.stack([A[ti] - B[ti], B[ti] - C[ti], C[ti] - A[ti],
                                 A[tj] - B[tj], B[tj] - C[tj], C[tj] - A[tj]], 1).norm(dim=2).max(dim=1).values
            bpts = X[sample_boundary]
            bface = sample_face[sample_boundary]
            D = torch.cdist(point[idx[ck]], bpts)  # (hits, boundary samples)
            inf = torch.tensor(float("inf"), dtype=D.dtype, device=D.device)
            dA = torch.where(bface[None, :] == fi[ck][:, None], D, inf).min(dim=1).values
            dB = torch.where(bface[None, :] == fj[ck][:, None], D, inf).min(dim=1).values
            seam = (dA <= edges) & (dB <= edges)  # artifact of non-matching seam sampling
            real = torch.ones(len(idx), dtype=torch.bool, device=X.device)
            real[ck[seam]] = False
            idx = idx[real]
    n = int(len(idx))
    if return_hits:
        return n, [(int(i[k]), int(j[k])) for k in idx.tolist()]
    return n
