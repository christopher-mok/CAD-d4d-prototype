"""KnotInsert(face, axis, t): exact refinement of one face.

    P_new = A P_old  (Boehm insertion along one parameter direction)

The face's boundary rows along the refined direction get the knot too; they
remain exact refinements of their carriers, so invariant I1 still holds. The
new interior control points become fresh DOFs initialized to their exact
values, so the geometry is unchanged and the continuous dimension grows by
(n_other - 2).

Note: tensor-product insertion refines a whole parametric strip of the face;
use LocalRefine for point-local capacity.
"""
from __future__ import annotations

import numpy as np

from ..geometry import bspline_basis as bb
from ..geometry.state import CADState
from .base import Rewrite, RewriteOutcome, apply_along_axis, face_center_point


class KnotInsert(Rewrite):
    kind = "KnotInsert"
    exact = True
    refinement = True

    def __init__(self, face: int, axis: str, t: float, times: int = 1):
        self.face, self.axis, self.t, self.times = face, axis, float(t), int(times)

    def touched_faces(self, state):
        return {self.face}

    def location(self, state):
        uv = (self.t, 0.5) if self.axis == "u" else (0.5, self.t)
        return face_center_point(state, self.face, uv) if self.face in state.cx.faces else None

    def _apply(self, state: CADState) -> RewriteOutcome:
        cx = state.cx
        if self.face not in cx.faces:
            return RewriteOutcome(None, "face missing")
        reason = insert_knot_in_place(state, self.face, self.axis, self.t, self.times)
        if reason:
            return RewriteOutcome(None, reason)
        return RewriteOutcome(state, info={"born_faces": [self.face], "modified_faces": [self.face]})


def insert_knot_in_place(state: CADState, fid: int, axis: str, t: float, times: int = 1) -> str:
    """Exact knot insertion on one face of ``state``; returns a failure reason or ''."""
    f = state.cx.faces[fid]
    knots = f.knots_u if axis == "u" else f.knots_v
    p = f.degree_u if axis == "u" else f.degree_v
    if not (bb.KNOT_TOL < t < 1 - bb.KNOT_TOL):
        return "knot outside open domain"
    if bb.multiplicity(knots, t) + times > p:
        return "multiplicity would exceed degree"
    net = state.dof_map.net(fid)
    A = np.eye(bb.num_basis(knots, p))
    k = knots
    for _ in range(times):
        Ai, k = bb.insertion_matrix(k, p, t)
        A = Ai @ A
    new_net = apply_along_axis(net, A, axis)
    if axis == "u":
        f.knots_u = k
    else:
        f.knots_v = k
    f.interior = new_net[1:-1, 1:-1].copy()
    state.structure_changed()
    return ""
