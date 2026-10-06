"""LocalRefine(face, u, v): genuinely local capacity via an exact composite.

Tensor-product knot insertion is not point-local: a u-knot refines an entire
parametric strip. LocalRefine instead

    split the face at u_lo and u_hi   (exact SplitFace)
    split the middle child at v_lo and v_hi
    -> isolated child covering [u_lo, u_hi] x [v_lo, v_hi]
    insert ``refine_knots`` uniform knots per direction in that child only
    [refine_boundary] also refine the carriers bounding that child at the
                      child's new knots (exact CarrierKnotInsert; neighbours
                      sharing those carriers receive matching knots)

Every step preserves geometry exactly. Splits whose cut would fall within
``min_frac`` of the face boundary are skipped (the window is extended to the
boundary). Without ``refine_boundary`` neighbouring faces gain hanging vertices
but no knots or DOFs -- the refined child's boundary then stays in the coarse
carrier space, which leaves a coarse seam around every window. With
``refine_boundary`` the seam gets the same resolution as the interior.
"""
from __future__ import annotations

import numpy as np

from ..geometry import bspline_basis as bb
from ..geometry.state import CADState
from ..geometry.topology import SIDES, TOL
from .base import Rewrite, RewriteOutcome, face_center_point
from .carrier_knots import carrier_faces, insert_carrier_knot_in_place
from .knot_insert import insert_knot_in_place
from .split_face import split_face_in_place


class LocalRefine(Rewrite):
    kind = "LocalRefine"
    exact = True
    refinement = True

    def __init__(self, face: int, u: float, v: float, width: float = 0.3, refine_knots: int = 1,
                 min_frac: float = 0.08, min_root_size: float = 0.05, refine_boundary: bool = False):
        self.face, self.u, self.v = face, float(u), float(v)
        self.width, self.refine_knots, self.min_frac = float(width), int(refine_knots), float(min_frac)
        self.min_root_size = float(min_root_size)
        self.refine_boundary = bool(refine_boundary)

    def touched_faces(self, state):
        out = {self.face} | state.cx.neighbors(self.face)
        if self.refine_boundary and self.face in state.cx.faces:
            cx = state.cx
            for side in SIDES:
                for use in cx.faces[self.face].sides[side]:
                    out |= carrier_faces(state, cx.edges[use.edge].carrier)
        return out

    def location(self, state):
        return face_center_point(state, self.face, (self.u, self.v)) if self.face in state.cx.faces else None

    def window(self, domain=None):
        """Split window in face-local parameters.

        Cuts closer than ``min_frac`` (local) or ``min_root_size`` (root-face
        units) to the face boundary are dropped, which avoids sliver faces.
        """
        out = []
        for k, c in enumerate((self.u, self.v)):
            size = 1.0 if domain is None else domain[2 * k + 1] - domain[2 * k]
            m = max(self.min_frac, self.min_root_size / size)
            lo, hi = c - self.width / 2, c + self.width / 2
            lo = 0.0 if lo < m else lo
            hi = 1.0 if hi > 1 - m else hi
            if 0.0 < lo and hi < 1.0 and hi - lo < m:
                lo, hi = max(0.0, c - m / 2), min(1.0, c + m / 2)
            out.append((lo, hi))
        return out

    def _apply(self, state: CADState) -> RewriteOutcome:
        if self.face not in state.cx.faces:
            return RewriteOutcome(None, "face missing")
        (u_lo, u_hi), (v_lo, v_hi) = self.window(state.cx.faces[self.face].domain)
        target, born = self.face, []
        for axis, lo, hi in (("u", u_lo, u_hi), ("v", v_lo, v_hi)):
            if lo > 0.0:
                if hi - lo < self.min_frac:
                    return RewriteOutcome(None, "window too small")
                a, b = split_face_in_place(state, target, axis, lo)
                born += [a, b]
                target = b
                hi = (hi - lo) / (1.0 - lo) if hi < 1.0 else 1.0
            if hi < 1.0:
                if hi < self.min_frac:
                    return RewriteOutcome(None, "window too small")
                a, b = split_face_in_place(state, target, axis, hi)
                born += [a, b]
                target = a
        n = self.refine_knots
        inserted = 0
        for axis in ("u", "v"):
            for k in range(1, n + 1):
                t = k / (n + 1)
                f = state.cx.faces[target]
                existing = bb.interior_knots(f.knots_u if axis == "u" else f.knots_v, 3)
                if len(existing) and np.min(np.abs(existing - t)) < 0.25 / (n + 1):
                    continue  # avoid near-duplicate knots on already refined faces
                reason = insert_knot_in_place(state, target, axis, t)
                if reason:
                    return RewriteOutcome(None, reason)
                inserted += 1
        if inserted == 0 and target == self.face:
            return RewriteOutcome(None, "nothing to refine")
        if self.refine_boundary:
            born += refine_boundary_carriers(state, target)
        born = [f for f in dict.fromkeys(born + [target]) if f in state.cx.faces]
        return RewriteOutcome(state, info={"born_faces": born, "refined_face": target,
                                           "window": [(u_lo, u_hi), (v_lo, v_hi)]})


def refine_boundary_carriers(state: CADState, fid: int, min_gap: float = 0.02) -> list[int]:
    """Insert the face's interior side knots into the carriers along its sides (exact).

    Returns the faces touched by the carrier refinements.
    """
    cx = state.cx
    f = cx.faces[fid]
    todo = []
    for side in SIDES:
        knots = f.side_knots(side)
        for run in cx.side_runs(f, side):
            for t in np.unique(np.round(bb.interior_knots(knots, 3), 12)):
                if run.a + TOL < t < run.b - TOL:
                    sc = run.s_a + (t - run.a) * (run.s_b - run.s_a) / (run.b - run.a)
                    todo.append((run.carrier, float(sc)))
    touched = []
    for cid, sc in todo:
        c = cx.carriers[cid]
        existing = np.concatenate([[0.0, 1.0], bb.interior_knots(c.knots, c.degree)])
        if np.min(np.abs(existing - sc)) < min_gap:
            continue
        touched += sorted(carrier_faces(state, cid))
        insert_carrier_knot_in_place(state, cid, sc)  # a refusal (multiplicity) leaves geometry intact
    return touched


class FaceRefine(Rewrite):
    """Bisect every knot span of one face (both directions) and refine its
    bounding carriers at the same knots: the per-face analogue of uniform
    refinement, for broad smooth residuals where small windows leave seams."""

    kind = "FaceRefine"
    exact = True
    refinement = True

    def __init__(self, face: int, axes: str = "uv", refine_boundary: bool = True, min_root_span: float = 0.03):
        self.face, self.axes, self.refine_boundary = face, axes, bool(refine_boundary)
        self.min_root_span = float(min_root_span)

    def touched_faces(self, state):
        return LocalRefine(self.face, 0.5, 0.5, refine_boundary=self.refine_boundary).touched_faces(state)

    def location(self, state):
        return face_center_point(state, self.face) if self.face in state.cx.faces else None

    def _apply(self, state: CADState) -> RewriteOutcome:
        if self.face not in state.cx.faces:
            return RewriteOutcome(None, "face missing")
        f = state.cx.faces[self.face]
        inserted = 0
        for axis in self.axes:
            knots = f.knots_u if axis == "u" else f.knots_v
            size = (f.domain[1] - f.domain[0]) if axis == "u" else (f.domain[3] - f.domain[2])
            brk = np.unique(np.round(knots, 12))
            for a, b in zip(brk[:-1], brk[1:]):
                if (b - a) * size < 2 * self.min_root_span:
                    continue
                reason = insert_knot_in_place(state, self.face, axis, 0.5 * (a + b))
                if reason:
                    return RewriteOutcome(None, reason)
                inserted += 1
                f = state.cx.faces[self.face]
        if inserted == 0:
            return RewriteOutcome(None, "spans already at minimum size")
        born = [self.face]
        if self.refine_boundary:
            born += refine_boundary_carriers(state, self.face)
        return RewriteOutcome(state, info={"born_faces": list(dict.fromkeys(born)), "refined_face": self.face})
