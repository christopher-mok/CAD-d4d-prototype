"""SplitFace(face, axis, t): exact split along an isoparametric curve.

    insert t to full multiplicity -> extract the two child control nets
    -> new internal carrier = the shared row (its interior becomes DOFs)
    -> split the two perpendicular boundary edges at t (hanging vertices)
    -> two child faces with provenance (parent, axis, t)

The geometry before and after is identical. Neighbours keep their structure;
their wires only gain the hanging vertices.
"""
from __future__ import annotations

import numpy as np

from ..geometry import bspline_basis as bb
from ..geometry.state import CADState
from ..geometry.topology import TOL, EdgeUse, Face, PatchComplex, Provenance
from .base import Rewrite, RewriteOutcome, apply_along_axis, face_center_point
from .split_edge import split_edge_in_place


def vertex_at_side_param(cx: PatchComplex, fid: int, side: str, t: float) -> int:
    """Vertex on a face side at side parameter t, splitting an edge if needed."""
    for use in cx.faces[fid].sides[side]:
        va, vb = cx.use_vertices(use)
        if abs(use.a - t) <= TOL:
            return va
        if abs(use.b - t) <= TOL:
            return vb
        if use.a < t < use.b:
            sa, sb = cx.use_carrier_params(use)
            s = sa + (t - use.a) * (sb - sa) / (use.b - use.a)
            return split_edge_in_place(cx, use.edge, s)
    raise ValueError("side parameter not covered by wire")


def restrict_wire(uses: list, lo: float, hi: float) -> list:
    out = []
    for u in uses:
        if u.a >= lo - TOL and u.b <= hi + TOL:
            a = (u.a - lo) / (hi - lo)
            b = (u.b - lo) / (hi - lo)
            a = 0.0 if abs(a) <= TOL else (1.0 if abs(a - 1) <= TOL else a)
            b = 0.0 if abs(b) <= TOL else (1.0 if abs(b - 1) <= TOL else b)
            out.append(EdgeUse(u.edge, a, b, u.reversed))
    return out


def split_face_in_place(state: CADState, fid: int, axis: str, t: float) -> tuple[int, int]:
    cx = state.cx
    f = cx.faces[fid]
    knots = f.knots_u if axis == "u" else f.knots_v
    p = f.degree_u if axis == "u" else f.degree_v
    if not (TOL < t < 1 - TOL):
        raise ValueError("split parameter outside open domain")
    net = state.dof_map.net(fid)
    A_l, K_l = bb.segment_matrix(knots, p, 0.0, t)
    A_r, K_r = bb.segment_matrix(knots, p, t, 1.0)
    net_l = apply_along_axis(net, A_l, axis)
    net_r = apply_along_axis(net, A_r, axis)

    if axis == "u":
        s0, s1 = "v0", "v1"
        shared = net_l[-1]
        other_knots = f.knots_v
    else:
        s0, s1 = "u0", "u1"
        shared = net_l[:, -1]
        other_knots = f.knots_u
    X0 = vertex_at_side_param(cx, fid, s0, t)
    X1 = vertex_at_side_param(cx, fid, s1, t)
    C = cx.add_carrier(other_knots.copy(), X0, X1, shared[1:-1].copy())
    E = cx.add_edge(C.id, 0.0, 1.0, X0, X1)
    inner = [EdgeUse(E.id, 0.0, 1.0, False)]

    d = f.domain
    if axis == "u":
        cut = d[0] + t * (d[1] - d[0])
        dom1, dom2 = np.array([d[0], cut, d[2], d[3]]), np.array([cut, d[1], d[2], d[3]])
        sides1 = {"v0": restrict_wire(f.sides["v0"], 0, t), "v1": restrict_wire(f.sides["v1"], 0, t),
                  "u0": list(f.sides["u0"]), "u1": inner}
        sides2 = {"v0": restrict_wire(f.sides["v0"], t, 1), "v1": restrict_wire(f.sides["v1"], t, 1),
                  "u0": [EdgeUse(E.id, 0.0, 1.0, False)], "u1": list(f.sides["u1"])}
        k1 = (K_l, f.knots_v.copy())
        k2 = (K_r, f.knots_v.copy())
    else:
        cut = d[2] + t * (d[3] - d[2])
        dom1, dom2 = np.array([d[0], d[1], d[2], cut]), np.array([d[0], d[1], cut, d[3]])
        sides1 = {"u0": restrict_wire(f.sides["u0"], 0, t), "u1": restrict_wire(f.sides["u1"], 0, t),
                  "v0": list(f.sides["v0"]), "v1": inner}
        sides2 = {"u0": restrict_wire(f.sides["u0"], t, 1), "u1": restrict_wire(f.sides["u1"], t, 1),
                  "v0": [EdgeUse(E.id, 0.0, 1.0, False)], "v1": list(f.sides["v1"])}
        k1 = (f.knots_u.copy(), K_l)
        k2 = (f.knots_u.copy(), K_r)

    children = []
    for (ku, kv), n, dom, sides in ((k1, net_l, dom1, sides1), (k2, net_r, dom2, sides2)):
        child = Face(-1, ku, kv, n[1:-1, 1:-1].copy(), sides, f.degree_u, f.degree_v, dom, f.root,
                     Provenance("SplitFace", (fid,), axis, float(t)), f.lineage)
        children.append(cx.add_face(child).id)
    for c in children:
        cx.faces[c].provenance.siblings = tuple(children)
    del cx.faces[fid]
    state.structure_changed()
    return children[0], children[1]


class SplitFace(Rewrite):
    kind = "SplitFace"
    exact = True
    refinement = True

    def __init__(self, face: int, axis: str, t: float, min_frac: float = 0.05, min_root_size: float = 0.05):
        self.face, self.axis, self.t, self.min_frac = face, axis, float(t), float(min_frac)
        self.min_root_size = float(min_root_size)

    def touched_faces(self, state):
        return {self.face} | state.cx.neighbors(self.face)

    def location(self, state):
        uv = (self.t, 0.5) if self.axis == "u" else (0.5, self.t)
        return face_center_point(state, self.face, uv) if self.face in state.cx.faces else None

    def _apply(self, state: CADState) -> RewriteOutcome:
        if self.face not in state.cx.faces:
            return RewriteOutcome(None, "face missing")
        d = state.cx.faces[self.face].domain
        size = d[1] - d[0] if self.axis == "u" else d[3] - d[2]
        if not (self.min_frac <= self.t <= 1 - self.min_frac) or                 min(self.t, 1 - self.t) * size < self.min_root_size:
            return RewriteOutcome(None, "child too small")
        a, b = split_face_in_place(state, self.face, self.axis, self.t)
        return RewriteOutcome(state, info={"born_faces": [a, b], "children": [a, b]})
