"""Least-squares fitting helpers with fixed (carrier-derived) boundaries.

Used to (a) build initial complexes from analytic shapes and (b) refit face
interiors after inexact simplifying rewrites (KnotRemove, MergeFace). Boundary
rows are never fitted here: they always come from the carriers, so a refit can
never break watertightness.
"""
from __future__ import annotations

import numpy as np

from . import bspline_basis as bb
from .state import DofMap
from .topology import Carrier, PatchComplex


def fit_carrier_interior(c: Carrier, p_start, p_end, s, points) -> float:
    """LS fit of a carrier's interior control points to ``points`` at params ``s``."""
    B = bb.basis_matrix(c.knots, c.degree, s)
    rhs = np.asarray(points) - np.outer(B[:, 0], p_start) - np.outer(B[:, -1], p_end)
    X, *_ = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)
    c.interior = X
    full = np.vstack([p_start, X, p_end])
    return float(np.max(np.linalg.norm(B @ full - points, axis=1)))


def fit_face_interior(cx: PatchComplex, fid: int, us, vs, target_grid: np.ndarray,
                      dof_map: DofMap | None = None) -> float:
    """LS fit of face ``fid``'s interior control points to ``target_grid`` (len(us), len(vs), 3).

    The boundary rows are taken from the carriers through the DOF map. Returns
    the max deviation of the fitted surface from the targets at the samples.
    """
    f = cx.faces[fid]
    dm = dof_map or DofMap(cx)
    nu, nv = f.shape
    Bu = bb.basis_matrix(f.knots_u, f.degree_u, us)
    Bv = bb.basis_matrix(f.knots_v, f.degree_v, vs)
    B = np.kron(Bu, Bv)  # (len(us) * len(vs), nu * nv), flattening (i, j) -> i * nv + j
    net = (dm.face_rows(fid) @ dm.values).reshape(-1, 3)
    mask = f.interior_mask()
    Y = np.asarray(target_grid).reshape(-1, 3)
    rhs = Y - B[:, ~mask] @ net[~mask]
    if mask.any():
        X, *_ = np.linalg.lstsq(B[:, mask], rhs, rcond=None)
        net[mask] = X
        f.interior = X.reshape(nu - 2, nv - 2, 3)
    return float(np.max(np.linalg.norm(B @ net - Y, axis=1)))


def dense_params(knots: np.ndarray, degree: int, per_span: int = 6) -> np.ndarray:
    """Parameter samples that resolve every knot span (used for refits/deviation)."""
    brk = np.unique(np.round(knots, 12))
    pts = [np.linspace(a, b, per_span + 1)[:-1] for a, b in zip(brk[:-1], brk[1:])]
    return np.concatenate(pts + [[brk[-1]]])
