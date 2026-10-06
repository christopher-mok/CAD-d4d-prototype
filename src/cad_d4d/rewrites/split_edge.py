"""SplitEdge / MergeEdge: topological boundary subdivision on a fixed carrier.

SplitEdge(e, s): A ------ B   ->   A --- X --- B
  X is a hanging vertex constrained to the carrier at parameter s. Every face
  wire using e is updated to use the two subedges. The carrier (canonical
  geometry) is untouched, so no geometry and no DOF changes. Neighbouring
  faces do not have to be split; they simply see one more vertex on a side.

MergeEdge(X): inverse, allowed when X is a hanging vertex between two subedges
  of the same carrier, is no carrier endpoint, and is not a corner of any face.
"""
from __future__ import annotations

from ..geometry.state import CADState
from ..geometry.topology import TOL, EdgeUse, PatchComplex
from .base import Rewrite, RewriteOutcome


def side_param_of(cx: PatchComplex, use: EdgeUse, s: float) -> float:
    sa, sb = cx.use_carrier_params(use)
    return use.a + (s - sa) * (use.b - use.a) / (sb - sa)


def split_edge_in_place(cx: PatchComplex, eid: int, s: float) -> int:
    e = cx.edges[eid]
    if not (e.s0 + TOL < s < e.s1 - TOL):
        raise ValueError("split parameter outside edge interior")
    X = cx.add_vertex(host=(e.carrier, float(s)))
    ea = cx.add_edge(e.carrier, e.s0, s, e.v0, X.id)
    eb = cx.add_edge(e.carrier, s, e.s1, X.id, e.v1)
    for fid, side, k in cx.edge_incidence()[eid]:
        wire = cx.faces[fid].sides[side]
        use = wire[k]
        t = side_param_of(cx, use, s)
        if not use.reversed:
            new = [EdgeUse(ea.id, use.a, t, False), EdgeUse(eb.id, t, use.b, False)]
        else:
            new = [EdgeUse(eb.id, use.a, t, True), EdgeUse(ea.id, t, use.b, True)]
        wire[k: k + 1] = new
    del cx.edges[eid]
    return X.id


def can_merge_edges_at(cx: PatchComplex, vid: int) -> tuple[int, int] | None:
    """Return (edge ending at X, edge starting at X) if MergeEdge(X) is valid."""
    v = cx.vertices.get(vid)
    if v is None or v.free:
        return None
    if any(vid in (c.v_start, c.v_end) for c in cx.carriers.values()):
        return None
    edges = cx.vertex_edges(vid)
    if len(edges) != 2:
        return None
    e1, e2 = (cx.edges[e] for e in edges)
    if e1.v1 != vid:
        e1, e2 = e2, e1
    if e1.v1 != vid or e2.v0 != vid or e1.carrier != e2.carrier:
        return None
    inc = cx.edge_incidence()
    for eid, other in ((e1.id, e2.id), (e2.id, e1.id)):
        for fid, side, k in inc[eid]:
            wire = cx.faces[fid].sides[side]
            neigh = [wire[j].edge for j in (k - 1, k + 1) if 0 <= j < len(wire)]
            if other not in neigh:
                return None  # X is a face corner
    return e1.id, e2.id


def merge_edge_in_place(cx: PatchComplex, vid: int) -> int:
    pair = can_merge_edges_at(cx, vid)
    if pair is None:
        raise ValueError("MergeEdge not applicable")
    e1, e2 = cx.edges[pair[0]], cx.edges[pair[1]]
    e = cx.add_edge(e1.carrier, e1.s0, e2.s1, e1.v0, e2.v1)
    for f in cx.faces.values():
        for side, wire in f.sides.items():
            idx = [k for k, u in enumerate(wire) if u.edge in (e1.id, e2.id)]
            if not idx:
                continue
            k0, k1 = idx
            assert k1 == k0 + 1
            wire[k0: k1 + 1] = [EdgeUse(e.id, wire[k0].a, wire[k1].b, wire[k0].reversed)]
    del cx.edges[e1.id], cx.edges[e2.id], cx.vertices[vid]
    return e.id


class SplitEdge(Rewrite):
    kind = "SplitEdge"
    exact = True
    refinement = False

    def __init__(self, edge: int, s: float):
        self.edge, self.s = edge, float(s)

    def touched_faces(self, state):
        return state.cx.faces_of_edge(self.edge)

    def _apply(self, state: CADState) -> RewriteOutcome:
        if self.edge not in state.cx.edges:
            return RewriteOutcome(None, "edge missing")
        vid = split_edge_in_place(state.cx, self.edge, self.s)
        return RewriteOutcome(state, info={"vertex": vid})


class MergeEdge(Rewrite):
    kind = "MergeEdge"
    exact = True
    refinement = False

    def __init__(self, vertex: int):
        self.vertex = vertex

    def touched_faces(self, state):
        out = set()
        for e in state.cx.vertex_edges(self.vertex):
            out |= state.cx.faces_of_edge(e)
        return out

    def _apply(self, state: CADState) -> RewriteOutcome:
        if can_merge_edges_at(state.cx, self.vertex) is None:
            return RewriteOutcome(None, "not mergeable")
        eid = merge_edge_in_place(state.cx, self.vertex)
        return RewriteOutcome(state, info={"edge": eid})
