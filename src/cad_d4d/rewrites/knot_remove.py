"""KnotRemove(face, axis, t): epsilon-gated simplification (inverse of KnotInsert).

Removes one occurrence of an interior knot, refits the face interior by least
squares with the carrier-derived boundary fixed, and accepts only if

    max ||S_before - S_after|| < eps_remove

over a dense validation grid. Fails if the knot is required by invariant I1
(i.e. by a carrier or a vertex on the face boundary).
"""
from __future__ import annotations

import copy

import numpy as np

from ..geometry import bspline_basis as bb
from ..geometry.fitting import dense_params, fit_face_interior
from ..geometry.state import CADState
from ..geometry.topology import TopologyError
from .base import Rewrite, RewriteOutcome, face_center_point


def deviation_on_grid(old: CADState, old_fid: int, new: CADState, new_fid: int, us_new, vs_new, map_uv) -> float:
    """Max distance between new face at (u, v) and old face at map_uv(u, v)."""
    U, V = np.meshgrid(us_new, vs_new, indexing="ij")
    uv_new = np.stack([U.ravel(), V.ravel()], 1)
    a = new.evaluate(new_fid, uv_new)
    b = old.evaluate(old_fid, map_uv(uv_new))
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def validation_params(knots, degree) -> np.ndarray:
    d = dense_params(knots, degree, 6)
    return np.unique(np.concatenate([d, 0.5 * (d[1:] + d[:-1])]))


class KnotRemove(Rewrite):
    kind = "KnotRemove"
    exact = False
    refinement = False

    def __init__(self, face: int, axis: str, t: float, eps: float = 1e-3):
        self.face, self.axis, self.t, self.eps = face, axis, float(t), float(eps)

    def touched_faces(self, state):
        return {self.face}

    def location(self, state):
        uv = (self.t, 0.5) if self.axis == "u" else (0.5, self.t)
        return face_center_point(state, self.face, uv) if self.face in state.cx.faces else None

    def _apply(self, state: CADState) -> RewriteOutcome:
        old = state.copy()
        cx = state.cx
        if self.face not in cx.faces:
            return RewriteOutcome(None, "face missing")
        f = cx.faces[self.face]
        knots = f.knots_u if self.axis == "u" else f.knots_v
        if not (bb.KNOT_TOL < self.t < 1 - bb.KNOT_TOL) or bb.multiplicity(knots, self.t) == 0:
            return RewriteOutcome(None, "not an interior knot")
        i = int(np.argmax(np.abs(knots - self.t) <= bb.KNOT_TOL))
        new_knots = np.delete(knots, i)
        us_old = validation_params(f.knots_u, f.degree_u)
        vs_old = validation_params(f.knots_v, f.degree_v)
        if self.axis == "u":
            f.knots_u = new_knots
        else:
            f.knots_v = new_knots
        nu, nv = f.shape
        f.interior = np.zeros((nu - 2, nv - 2, 3))
        sides = ("v0", "v1") if self.axis == "u" else ("u0", "u1")
        try:
            for side in sides:
                cx.check_side_refinement(f, side)
        except TopologyError:
            return RewriteOutcome(None, "knot required by boundary carrier/vertex")
        U, V = np.meshgrid(us_old, vs_old, indexing="ij")
        Y = old.evaluate(self.face, np.stack([U.ravel(), V.ravel()], 1)).reshape(len(us_old), len(vs_old), 3)
        state.structure_changed()
        fit_face_interior(cx, self.face, us_old, vs_old, Y, state.dof_map)
        state.structure_changed()
        uu = validation_params(f.knots_u, f.degree_u) * 0.999 + 0.0005
        vv = validation_params(f.knots_v, f.degree_v) * 0.999 + 0.0005
        dev = max(deviation_on_grid(old, self.face, state, self.face, us_old, vs_old, lambda x: x),
                  deviation_on_grid(old, self.face, state, self.face, uu, vv, lambda x: x))
        if dev > self.eps:
            return RewriteOutcome(None, f"deviation {dev:.3e} > eps {self.eps:.1e}", deviation=dev)
        return RewriteOutcome(state, deviation=dev, info={"modified_faces": [self.face]})


def removal_allowed(cx, f, axis: str, t: float) -> bool:
    """Whether removing one copy of knot t keeps invariant I1 on the sides along ``axis``."""
    trial = copy.copy(f)
    knots = f.knots_u if axis == "u" else f.knots_v
    reduced = np.delete(knots, int(np.argmax(np.abs(knots - t) <= bb.KNOT_TOL)))
    if axis == "u":
        trial.knots_u = reduced
    else:
        trial.knots_v = reduced
    try:
        for side in (("v0", "v1") if axis == "u" else ("u0", "u1")):
            cx.check_side_refinement(trial, side)
    except TopologyError:
        return False
    return True


def knot_remove_candidates(state: CADState, eps: float) -> list[KnotRemove]:
    """Interior knots whose removal is structurally allowed (I1); the epsilon gate is checked on apply."""
    out = []
    for fid, f in state.cx.faces.items():
        for axis, knots, p in (("u", f.knots_u, f.degree_u), ("v", f.knots_v, f.degree_v)):
            for t in np.unique(np.round(bb.interior_knots(knots, p), 12)):
                if removal_allowed(state.cx, f, axis, float(t)):
                    out.append(KnotRemove(fid, axis, float(t), eps))
    return out
