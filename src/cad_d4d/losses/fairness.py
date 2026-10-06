"""Fairness (bending) energy, kept separate from structural complexity.

Two discretizations, both a fixed quadratic form ``|F p|^2`` per structure:

``bending`` (default)
    Thin-plate bending energy of the actual surface, integrated with 4-point
    Gauss quadrature on every knot span (exact for the polynomial integrand of
    a bicubic patch), with derivatives taken w.r.t. the *root* face parameter:

        E = sum_roots int |S_uu|^2 + 2 |S_uv|^2 + |S_vv|^2  dU dV

    It depends only on geometry, so it is exactly invariant under exact
    rewrites (KnotInsert, SplitFace, LocalRefine). This matters for
    marginal-descent scoring, which compares gradients across
    parameterizations of the same surface.

``control_net``
    Greville-spaced divided second differences of the control net (a classic
    control-net fairness). Cheap, but not refinement invariant; kept for
    ablations. Near-coincident Greville abscissae are skipped.

Crease penalty across split lines (both kinds, weight ``crease_weight``)
    Per-face energies cannot see a tangent discontinuity *between* faces, so
    children of a SplitFace could crease apart freely (and then never merge
    back). For every edge whose two faces share a root face (an isoparametric
    split line), we add

        crease_weight * int_edge | dS_A/dN - dS_B/dN |^2 ds,

    N being the root parameter across the line. Right after an exact split the
    surface is C2 there, so the term is zero (refinement invariant); it grows
    only if the children develop a crease.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..geometry import bspline_basis as bb
from ..geometry.state import CADState

_GX, _GW = np.polynomial.legendre.leggauss(4)


def _gauss_points(knots):
    brk = np.unique(np.round(knots, 12))
    pts, wts = [], []
    for a, b in zip(brk[:-1], brk[1:]):
        pts.append(0.5 * (a + b) + 0.5 * (b - a) * _GX)
        wts.append(0.5 * (b - a) * _GW)
    return np.concatenate(pts), np.concatenate(wts)


def _bending_rows(f, R):
    su = f.domain[1] - f.domain[0]
    sv = f.domain[3] - f.domain[2]
    gu, wu = _gauss_points(f.knots_u)
    gv, wv = _gauss_points(f.knots_v)
    B = {d: bb.basis_matrix(f.knots_u, f.degree_u, gu, d) for d in (0, 1, 2)}
    C = {d: bb.basis_matrix(f.knots_v, f.degree_v, gv, d) for d in (0, 1, 2)}
    W = np.outer(wu, wv).ravel() * su * sv  # root-domain area weights
    mats = []
    for (du, dv), coef, scale in (((2, 0), 1.0, su**-2), ((1, 1), 2.0, 1.0 / (su * sv)), ((0, 2), 1.0, sv**-2)):
        M = sp.csr_matrix(np.kron(B[du], C[dv]) * scale)
        mats.append(sp.diags(np.sqrt(coef * W)) @ M)
    return sp.vstack(mats) @ R


def _second_diff_1d(knots, degree, scale, min_gap=1e-3):
    xi = bb.greville(knots, degree) * scale
    n = len(xi)
    rows, cell = [], []
    for i in range(1, n - 1):
        h0, h1 = xi[i] - xi[i - 1], xi[i + 1] - xi[i]
        if h0 <= min_gap * scale or h1 <= min_gap * scale:
            continue
        c = 2.0 / (h0 + h1)
        r = np.zeros(n)
        r[i - 1], r[i], r[i + 1] = c / h0, -c * (1 / h0 + 1 / h1), c / h1
        rows.append(r)
        cell.append(0.5 * (h0 + h1))
    return np.array(rows).reshape(-1, n), np.array(cell)


def _cell_widths(knots, degree, scale):
    xi = bb.greville(knots, degree) * scale
    w = np.zeros(len(xi))
    w[1:] += 0.5 * np.diff(xi)
    w[:-1] += 0.5 * np.diff(xi)
    return w


def _control_net_rows(f, R):
    su = f.domain[1] - f.domain[0]
    sv = f.domain[3] - f.domain[2]
    nu, nv = f.shape
    Du, wu = _second_diff_1d(f.knots_u, f.degree_u, su)
    Dv, wv = _second_diff_1d(f.knots_v, f.degree_v, sv)
    cu = _cell_widths(f.knots_u, f.degree_u, su)
    cv = _cell_widths(f.knots_v, f.degree_v, sv)
    mats = []
    if len(Du):
        mats.append(sp.diags(np.sqrt(np.outer(wu, cv)).ravel()) @ sp.kron(sp.csr_matrix(Du), sp.eye(nv)))
    if len(Dv):
        mats.append(sp.diags(np.sqrt(np.outer(cu, wv)).ravel()) @ sp.kron(sp.eye(nu), sp.csr_matrix(Dv)))
    return (sp.vstack(mats) @ R) if mats else None


def _crease_rows(state: CADState, weight: float):
    cx, dm = state.cx, state.dof_map
    rows = []
    for eid, uses in cx.edge_incidence().items():
        if len(uses) != 2:
            continue
        (fa, sa, ka), (fb, sb, kb) = uses
        A, B = cx.faces[fa], cx.faces[fb]
        if A.root != B.root:
            continue
        intervals, mats = [], []
        for f, side, k in ((A, sa, ka), (B, sb, kb)):
            use = f.sides[side][k]
            ax = 0 if side in ("v0", "v1") else 2  # root coordinate along the side
            d0, d1 = f.domain[ax], f.domain[ax + 1]
            intervals.append((d0 + use.a * (d1 - d0), d0 + use.b * (d1 - d0)))
        (a0, a1), (b0, b1) = intervals
        if abs(a0 - b0) > 1e-9 or abs(a1 - b1) > 1e-9 or a1 - a0 <= 1e-12:
            continue  # not a shared isoparametric line in root coordinates
        X = 0.5 * (a0 + a1) + 0.5 * (a1 - a0) * _GX
        W = 0.5 * (a1 - a0) * _GW
        for f, side in ((A, sa), (B, sb)):
            along_u = side in ("v0", "v1")
            ax = 0 if along_u else 2
            t = (X - f.domain[ax]) / (f.domain[ax + 1] - f.domain[ax])
            uv = f.side_uv(side, np.clip(t, 0.0, 1.0))
            if along_u:  # cross derivative d/dV_root = (1 / sv) d/dv
                Bu = bb.basis_matrix(f.knots_u, f.degree_u, uv[:, 0])
                Bv = bb.basis_matrix(f.knots_v, f.degree_v, uv[:, 1], 1) / (f.domain[3] - f.domain[2])
            else:        # d/dU_root = (1 / su) d/du
                Bu = bb.basis_matrix(f.knots_u, f.degree_u, uv[:, 0], 1) / (f.domain[1] - f.domain[0])
                Bv = bb.basis_matrix(f.knots_v, f.degree_v, uv[:, 1])
            D = np.einsum("ia,ib->iab", Bu, Bv).reshape(len(X), -1)
            mats.append(sp.csr_matrix(D) @ dm.face_rows(f.id))
        rows.append(sp.diags(np.sqrt(weight * W)) @ (mats[0] - mats[1]))
    return rows


def fairness_operator(state: CADState, kind: str = "bending", crease_weight: float = 0.0) -> sp.csr_matrix:
    """Sparse F with fairness energy = ||F @ P||_F^2."""
    dm = state.dof_map
    blocks = []
    for fid in dm.face_order:
        f = state.cx.faces[fid]
        R = dm.face_rows(fid)
        if kind == "bending":
            blocks.append(_bending_rows(f, R))
        elif kind == "control_net":
            rows = _control_net_rows(f, R)
            if rows is not None:
                blocks.append(rows)
        else:
            raise ValueError(kind)
    if crease_weight > 0:
        blocks += _crease_rows(state, crease_weight)
    if not blocks:
        return sp.csr_matrix((0, dm.n_dof))
    return sp.vstack(blocks).tocsr()
