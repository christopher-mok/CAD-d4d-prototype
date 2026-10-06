"""Point-to-triangle closest points (vectorized, differentiable in torch)."""
from __future__ import annotations

import torch


def _dot(a, b):
    return (a * b).sum(-1)


def _safe(den, eps=1e-30):
    return torch.where(den.abs() < eps, torch.full_like(den, eps), den)


def closest_point_on_triangle(p, a, b, c):
    """Closest point on triangles (a, b, c) to points p; all (..., 3) tensors.

    Ericson, Real-Time Collision Detection, 5.1.5, vectorized with region masks
    applied in reverse priority order. Unused branches use safe denominators so
    gradients stay finite.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp = p - b
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    cp = p - c
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom = _safe(va + vb + vc)
    v_in, w_in = vb / denom, vc / denom
    result = a + ab * v_in[..., None] + ac * w_in[..., None]

    w_bc = (d4 - d3) / _safe((d4 - d3) + (d5 - d6))
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    result = torch.where(m[..., None], b + (c - b) * w_bc[..., None], result)

    w_ac = d2 / _safe(d2 - d6)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    result = torch.where(m[..., None], a + ac * w_ac[..., None], result)

    m = (d6 >= 0) & (d5 <= d6)
    result = torch.where(m[..., None], c, result)

    v_ab = d1 / _safe(d1 - d3)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    result = torch.where(m[..., None], a + ab * v_ab[..., None], result)

    m = (d3 >= 0) & (d4 <= d3)
    result = torch.where(m[..., None], b, result)

    m = (d1 <= 0) & (d2 <= 0)
    result = torch.where(m[..., None], a, result)
    return result
