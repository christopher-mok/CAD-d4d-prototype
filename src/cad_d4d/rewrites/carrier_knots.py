"""Knot insertion / removal on carriers (canonical edge curves).

Face KnotInsert can only add *interior* DOFs: boundary rows are derived from
the carriers, so the shared boundary itself stays in the carrier's coarse
spline space. CarrierKnotInsert refines the boundary curve itself:

    1. every face side that follows the carrier across s gets the mapped knot
       (exact face knot insertion; keeps invariant I1: face side knots must
       contain the carrier knots),
    2. the carrier gets the knot (exact Boehm insertion).

Geometry is unchanged; the carrier gains one interior DOF and the incident faces
gain the strips required by step 1. Hanging vertices hosted on the carrier keep
their parameter and therefore their position.

CarrierKnotRemove is the epsilon-gated inverse: it removes one occurrence of a
carrier knot, refits the carrier interior by least squares (end points fixed)
and measures the deviation of *every* face, since vertices hosted on the
carrier (and curves ending at them) move with it.
"""
from __future__ import annotations

import numpy as np

from ..geometry import bspline_basis as bb
from ..geometry.fitting import dense_params, fit_carrier_interior
from ..geometry.state import CADState
from ..geometry.topology import SIDES, TOL
from .base import Rewrite, RewriteOutcome
from .knot_insert import insert_knot_in_place


def carrier_faces(state: CADState, cid: int) -> set[int]:
    """Faces whose geometry depends on carrier ``cid`` (directly or via hosted vertices)."""
    cx = state.cx
    carriers = {cid}
    hosted = {v.id for v in cx.vertices.values() if v.host is not None and v.host[0] == cid}
    for c in cx.carriers.values():
        if c.v_start in hosted or c.v_end in hosted:
            carriers.add(c.id)
    out = set()
    for f in cx.faces.values():
        for side in SIDES:
            if any(cx.edges[u.edge].carrier in carriers for u in f.sides[side]):
                out.add(f.id)
                break
    return out


def carrier_point(state: CADState, cid: int, s: float) -> np.ndarray:
    dm = state.dof_map
    c = state.cx.carriers[cid]
    return bb.evaluate_curve(c.knots, c.degree, dm.carrier_rows(cid) @ dm.values, [s])[0]


def insert_carrier_knot_in_place(state: CADState, cid: int, s: float, times: int = 1) -> str:
    cx = state.cx
    c = cx.carriers[cid]
    p = c.degree
    if not (TOL < s < 1 - TOL):
        return "knot outside open carrier domain"
    required = bb.multiplicity(c.knots, s) + times
    if required > p - 1:
        return "carrier knot multiplicity would reach the degree (use SplitEdge for a C0 break)"
    # 1. incident face sides must contain the mapped knot with at least the same multiplicity
    todo = []
    for f in cx.faces.values():
        for side in SIDES:
            for run in cx.side_runs(f, side):
                if run.carrier != cid:
                    continue
                lo, hi = sorted((run.s_a, run.s_b))
                if lo + TOL < s < hi - TOL:
                    t = run.a + (s - run.s_a) * (run.b - run.a) / (run.s_b - run.s_a)
                    axis = "u" if side in ("v0", "v1") else "v"
                    todo.append((f.id, axis, float(t)))
    for fid, axis, t in todo:
        f = cx.faces[fid]
        knots = f.knots_u if axis == "u" else f.knots_v
        missing = required - bb.multiplicity(knots, t)
        if missing > 0:
            reason = insert_knot_in_place(state, fid, axis, t, missing)
            if reason:
                return reason
    # 2. refine the carrier
    dm = state.dof_map
    cps = dm.carrier_rows(cid) @ dm.values
    A, k = bb.insert_times(c.knots, p, s, times)
    new = A @ cps
    c.knots = k
    c.interior = new[1:-1].copy()
    state.structure_changed()
    return ""


def face_grid_snapshot(state: CADState, faces, n: int = 21) -> dict[int, np.ndarray]:
    t = np.linspace(0, 1, n)
    U, V = np.meshgrid(t, t, indexing="ij")
    uv = np.stack([U.ravel(), V.ravel()], 1)
    return {fid: state.evaluate(fid, uv) for fid in faces}


class CarrierKnotInsert(Rewrite):
    kind = "CarrierKnotInsert"
    exact = True
    refinement = True

    def __init__(self, carrier: int, s: float, times: int = 1):
        self.carrier, self.s, self.times = carrier, float(s), int(times)

    def touched_faces(self, state):
        return carrier_faces(state, self.carrier) if self.carrier in state.cx.carriers else set()

    def location(self, state):
        return carrier_point(state, self.carrier, self.s) if self.carrier in state.cx.carriers else None

    def _apply(self, state: CADState) -> RewriteOutcome:
        if self.carrier not in state.cx.carriers:
            return RewriteOutcome(None, "carrier missing")
        faces = sorted(carrier_faces(state, self.carrier))
        reason = insert_carrier_knot_in_place(state, self.carrier, self.s, self.times)
        if reason:
            return RewriteOutcome(None, reason)
        return RewriteOutcome(state, info={"born_faces": faces, "modified_faces": faces})


class CarrierKnotRemove(Rewrite):
    kind = "CarrierKnotRemove"
    exact = False
    refinement = False

    def __init__(self, carrier: int, s: float, eps: float = 5e-3):
        self.carrier, self.s, self.eps = carrier, float(s), float(eps)

    def touched_faces(self, state):
        return carrier_faces(state, self.carrier) if self.carrier in state.cx.carriers else set()

    def location(self, state):
        return carrier_point(state, self.carrier, self.s) if self.carrier in state.cx.carriers else None

    def _apply(self, state: CADState) -> RewriteOutcome:
        cx = state.cx
        if self.carrier not in cx.carriers:
            return RewriteOutcome(None, "carrier missing")
        c = cx.carriers[self.carrier]
        if bb.multiplicity(c.knots, self.s) == 0 or not (TOL < self.s < 1 - TOL):
            return RewriteOutcome(None, "not an interior carrier knot")
        affected = sorted(carrier_faces(state, c.id))  # incl. faces bounded by curves ending on hosted vertices
        before = face_grid_snapshot(state, affected)
        dm = state.dof_map
        old_cps = dm.carrier_rows(c.id) @ dm.values
        s_fit = dense_params(c.knots, c.degree, 8)
        target = bb.evaluate_curve(c.knots, c.degree, old_cps, s_fit)
        i = int(np.argmax(np.abs(c.knots - self.s) <= TOL))
        c.knots = np.delete(c.knots, i)
        c.interior = np.zeros((c.n - 2, 3))
        fit_carrier_interior(c, old_cps[0], old_cps[-1], s_fit, target)
        state.structure_changed()
        after = face_grid_snapshot(state, affected)
        dev = max(float(np.max(np.linalg.norm(after[f] - before[f], axis=1))) for f in before)
        if dev > self.eps:
            return RewriteOutcome(None, f"deviation {dev:.3e} > eps {self.eps:.1e}", deviation=dev)
        return RewriteOutcome(state, deviation=dev, info={"modified_faces": sorted(carrier_faces(state, c.id))})


def carrier_knot_remove_candidates(state: CADState, eps: float) -> list[CarrierKnotRemove]:
    out = []
    for c in state.cx.carriers.values():
        for t in np.unique(np.round(bb.interior_knots(c.knots, c.degree), 12)):
            out.append(CarrierKnotRemove(c.id, float(t), eps))
    return out
