"""Minimal B-rep-like topology for a watertight complex of untrimmed B-spline patches.

Objects
-------
Vertex   free (owns a 3D DOF) or *constrained*: lies on a carrier at parameter
         ``s`` (a hanging vertex created by splitting an edge).
Carrier  canonical boundary geometry: a clamped B-spline curve on [0, 1]. Its
         end control points are its endpoint vertices; its interior control
         points are DOFs. Every face boundary row is derived from carriers.
Edge     topological edge = parameter interval [s0, s1] of one carrier.
         Splitting an edge never changes its carrier's geometry.
EdgeUse  occurrence of an edge in a face side wire, with the side-parameter
         interval [a, b] it covers and its orientation relative to the carrier.
Face     an untrimmed tensor-product patch: knot vectors + interior control
         points + four side wires ('v0', 'u1', 'v1', 'u0').

Parameterization conventions
----------------------------
Side 'v0' is v=0 traversed with increasing u; 'v1' is v=1 (increasing u);
'u0' is u=0 and 'u1' is u=1 (increasing v). The outward normal of every face
is ``S_u x S_v``. The counter-clockwise boundary loop traverses v0 and u1
forward, v1 and u0 backward (``SIDE_LOOP_SIGN``).

Watertightness invariant (I1)
-----------------------------
Consecutive uses on a side that belong to the same carrier, are contiguous on
it and share one affine side<->carrier map form a *run*. For each run:
  * the face side knot vector has full multiplicity (= degree) at the run's
    interior end points, and
  * the face side knots restricted to the run contain the carrier knots
    restricted to the run.
Then each run's boundary control points are an exact linear function of the
carrier's control points, so both faces incident to an edge reproduce the same
curve for every value of the continuous parameters.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import bspline_basis as bb
from .patch import SplinePatch

SIDES = ("v0", "u1", "v1", "u0")
SIDE_PARAM_AXIS = {"v0": "u", "v1": "u", "u0": "v", "u1": "v"}
SIDE_LOOP_SIGN = {"v0": +1, "u1": +1, "v1": -1, "u0": -1}
TOL = bb.KNOT_TOL


@dataclass
class Vertex:
    id: int
    position: np.ndarray | None = None  # free vertex DOF value
    host: tuple[int, float] | None = None  # (carrier id, s) if constrained

    @property
    def free(self) -> bool:
        return self.host is None


@dataclass
class Carrier:
    id: int
    knots: np.ndarray
    v_start: int
    v_end: int
    interior: np.ndarray  # (n - 2, 3)
    degree: int = 3

    @property
    def n(self) -> int:
        return bb.num_basis(self.knots, self.degree)


@dataclass
class Edge:
    id: int
    carrier: int
    s0: float
    s1: float
    v0: int  # vertex at s0
    v1: int  # vertex at s1


@dataclass
class EdgeUse:
    edge: int
    a: float
    b: float
    reversed: bool  # True if the side parameter runs from s1 to s0


@dataclass
class Provenance:
    op: str
    parents: tuple = ()
    axis: str | None = None
    param: float | None = None
    siblings: tuple = ()


@dataclass
class Face:
    id: int
    knots_u: np.ndarray
    knots_v: np.ndarray
    interior: np.ndarray  # (nu - 2, nv - 2, 3)
    sides: dict = field(default_factory=dict)  # side -> list[EdgeUse]
    degree_u: int = 3
    degree_v: int = 3
    # Sub-rectangle of the root face's parameter domain this face covers.
    domain: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0, 1.0]))
    root: int = -1
    provenance: Provenance | None = None
    lineage: frozenset = frozenset()

    @property
    def patch(self) -> SplinePatch:
        return SplinePatch(self.knots_u, self.knots_v, self.degree_u, self.degree_v)

    @property
    def shape(self) -> tuple[int, int]:
        return (bb.num_basis(self.knots_u, self.degree_u), bb.num_basis(self.knots_v, self.degree_v))

    def side_knots(self, side: str) -> np.ndarray:
        return self.knots_u if SIDE_PARAM_AXIS[side] == "u" else self.knots_v

    def side_degree(self, side: str) -> int:
        return self.degree_u if SIDE_PARAM_AXIS[side] == "u" else self.degree_v

    def side_row_indices(self, side: str) -> np.ndarray:
        """Flat net indices (i * nv + j) of the boundary row, ordered by side parameter."""
        nu, nv = self.shape
        if side == "v0":
            return np.arange(nu) * nv
        if side == "v1":
            return np.arange(nu) * nv + nv - 1
        if side == "u0":
            return np.arange(nv)
        if side == "u1":
            return (nu - 1) * nv + np.arange(nv)
        raise KeyError(side)

    def side_uv(self, side: str, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        fixed = {"v0": (None, 0.0), "v1": (None, 1.0), "u0": (0.0, None), "u1": (1.0, None)}[side]
        if fixed[0] is None:
            return np.stack([t, np.full_like(t, fixed[1])], -1)
        return np.stack([np.full_like(t, fixed[0]), t], -1)

    def interior_mask(self) -> np.ndarray:
        nu, nv = self.shape
        m = np.zeros((nu, nv), bool)
        m[1:-1, 1:-1] = True
        return m.reshape(-1)


@dataclass
class Run:
    """Maximal stretch of a face side that follows one carrier with one affine map."""
    carrier: int
    a: float
    b: float
    s_a: float  # carrier parameter at side parameter a
    s_b: float
    uses: list

    @property
    def reversed(self) -> bool:
        return self.s_b < self.s_a


class TopologyError(RuntimeError):
    pass


class PatchComplex:
    """Container for vertices, carriers, edges and faces of one (or more) shells."""

    def __init__(self):
        self.vertices: dict[int, Vertex] = {}
        self.carriers: dict[int, Carrier] = {}
        self.edges: dict[int, Edge] = {}
        self.faces: dict[int, Face] = {}
        self._next_id = 0

    # -- construction ------------------------------------------------------
    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def add_vertex(self, position=None, host=None) -> Vertex:
        v = Vertex(self.new_id(), None if position is None else np.asarray(position, float), host)
        self.vertices[v.id] = v
        return v

    def add_carrier(self, knots, v_start, v_end, interior) -> Carrier:
        c = Carrier(self.new_id(), np.asarray(knots, float), v_start, v_end, np.asarray(interior, float).reshape(-1, 3))
        if c.interior.shape[0] != c.n - 2:
            raise TopologyError("carrier interior size mismatch")
        self.carriers[c.id] = c
        return c

    def add_edge(self, carrier, s0, s1, v0, v1) -> Edge:
        e = Edge(self.new_id(), carrier, float(s0), float(s1), v0, v1)
        self.edges[e.id] = e
        return e

    def add_face(self, face: Face) -> Face:
        if face.id < 0 or face.id in self.faces:
            face.id = self.new_id()
        if face.root < 0:
            face.root = face.id
        self.faces[face.id] = face
        return face

    def copy(self) -> "PatchComplex":
        return copy.deepcopy(self)

    # -- queries -------------------------------------------------------------
    def edge_incidence(self) -> dict[int, list[tuple[int, str, int]]]:
        """edge id -> list of (face id, side, index in side wire)."""
        inc: dict[int, list] = {e: [] for e in self.edges}
        for f in self.faces.values():
            for side in SIDES:
                for k, use in enumerate(f.sides[side]):
                    inc.setdefault(use.edge, []).append((f.id, side, k))
        return inc

    def faces_of_edge(self, edge_id: int) -> set[int]:
        return {f for f, _, _ in self.edge_incidence().get(edge_id, [])}

    def face_edges(self, face_id: int) -> set[int]:
        f = self.faces[face_id]
        return {u.edge for s in SIDES for u in f.sides[s]}

    def face_vertices(self, face_id: int) -> set[int]:
        out = set()
        for e in self.face_edges(face_id):
            out |= {self.edges[e].v0, self.edges[e].v1}
        return out

    def neighbors(self, face_id: int) -> set[int]:
        """Faces sharing an edge with ``face_id``."""
        mine = self.face_edges(face_id)
        return {g.id for g in self.faces.values() if g.id != face_id and mine & self.face_edges(g.id)}

    def vertex_edges(self, vertex_id: int) -> list[int]:
        return [e.id for e in self.edges.values() if vertex_id in (e.v0, e.v1)]

    def use_carrier_params(self, use: EdgeUse) -> tuple[float, float]:
        """Carrier parameter at the side-parameter ends (a, b) of a use."""
        e = self.edges[use.edge]
        return (e.s1, e.s0) if use.reversed else (e.s0, e.s1)

    def use_vertices(self, use: EdgeUse) -> tuple[int, int]:
        """Vertices at side-parameter ends (a, b) of a use."""
        e = self.edges[use.edge]
        return (e.v1, e.v0) if use.reversed else (e.v0, e.v1)

    def side_runs(self, face: Face, side: str) -> list[Run]:
        runs: list[Run] = []
        for use in face.sides[side]:
            e = self.edges[use.edge]
            s_a, s_b = self.use_carrier_params(use)
            if runs:
                r = runs[-1]
                slope_r = (r.s_b - r.s_a) / (r.b - r.a)
                slope_u = (s_b - s_a) / (use.b - use.a)
                if (r.carrier == e.carrier and abs(r.b - use.a) <= TOL and abs(r.s_b - s_a) <= TOL
                        and abs(slope_r - slope_u) <= 1e-7 * max(1.0, abs(slope_r))):
                    r.b, r.s_b = use.b, s_b
                    r.uses.append(use)
                    continue
            runs.append(Run(e.carrier, use.a, use.b, s_a, s_b, [use]))
        return runs

    # -- invariants ----------------------------------------------------------
    def check_invariants(self, closed: bool = True) -> None:
        """Raise ``TopologyError`` if the complex violates a structural invariant."""
        for v in self.vertices.values():
            if v.free == (v.position is None):
                raise TopologyError(f"vertex {v.id}: free vertices must (only) carry a position")
            if not v.free and v.host[0] not in self.carriers:
                raise TopologyError(f"vertex {v.id}: missing host carrier")
        for c in self.carriers.values():
            if c.v_start not in self.vertices or c.v_end not in self.vertices:
                raise TopologyError(f"carrier {c.id}: missing endpoint vertex")
            for vid in (c.v_start, c.v_end):
                h = self.vertices[vid].host
                if h is not None and h[0] == c.id:
                    raise TopologyError(f"carrier {c.id}: endpoint hosted on itself")
        for e in self.edges.values():
            if e.carrier not in self.carriers:
                raise TopologyError(f"edge {e.id}: missing carrier")
            if not (0 - TOL <= e.s0 < e.s1 <= 1 + TOL):
                raise TopologyError(f"edge {e.id}: bad carrier interval")
            c = self.carriers[e.carrier]
            for vid, s in ((e.v0, e.s0), (e.v1, e.s1)):
                v = self.vertices[vid]
                if abs(s) <= TOL and vid != c.v_start or abs(s - 1) <= TOL and vid != c.v_end:
                    raise TopologyError(f"edge {e.id}: end vertex mismatch with carrier")
                if TOL < s < 1 - TOL and (v.host is None or v.host[0] != c.id or abs(v.host[1] - s) > TOL):
                    raise TopologyError(f"edge {e.id}: interior vertex {vid} not hosted on carrier at {s}")
        inc = self.edge_incidence()
        for eid, uses in inc.items():
            if eid not in self.edges:
                raise TopologyError(f"use of missing edge {eid}")
            if closed and len(uses) != 2:
                raise TopologyError(f"edge {eid} has {len(uses)} uses (closed shell requires 2)")
            if len(uses) == 2:
                dirs = []
                for fid, side, k in uses:
                    use = self.faces[fid].sides[side][k]
                    dirs.append(SIDE_LOOP_SIGN[side] * (-1 if use.reversed else 1))
                if dirs[0] == dirs[1]:
                    raise TopologyError(f"edge {eid}: inconsistent face orientation")
        for f in self.faces.values():
            nu, nv = f.shape
            if f.interior.shape != (nu - 2, nv - 2, 3):
                raise TopologyError(f"face {f.id}: interior shape {f.interior.shape} != {(nu - 2, nv - 2, 3)}")
            for side in SIDES:
                uses = f.sides[side]
                if not uses or abs(uses[0].a) > TOL or abs(uses[-1].b - 1) > TOL:
                    raise TopologyError(f"face {f.id} side {side}: wire does not cover [0, 1]")
                for u0, u1 in zip(uses[:-1], uses[1:]):
                    if abs(u0.b - u1.a) > TOL or self.use_vertices(u0)[1] != self.use_vertices(u1)[0]:
                        raise TopologyError(f"face {f.id} side {side}: wire not contiguous")
                self.check_side_refinement(f, side)
            # Corners: consecutive sides meet at a common vertex.
            corner_pairs = [("v0", -1, "u1", 0), ("u1", -1, "v1", -1), ("v1", 0, "u0", -1), ("u0", 0, "v0", 0)]
            for s1, k1, s2, k2 in corner_pairs:
                end1 = self.use_vertices(f.sides[s1][k1])[1 if k1 == -1 else 0]
                end2 = self.use_vertices(f.sides[s2][k2])[1 if k2 == -1 else 0]
                if end1 != end2:
                    raise TopologyError(f"face {f.id}: corner mismatch between {s1} and {s2}")

    def check_side_refinement(self, f: Face, side: str) -> None:
        """Invariant I1 for one face side (raises on violation)."""
        knots = f.side_knots(side)
        p = f.side_degree(side)
        for run in self.side_runs(f, side):
            for t in (run.a, run.b):
                if TOL < t < 1 - TOL and bb.multiplicity(knots, t) != p:
                    raise TopologyError(f"face {f.id} side {side}: run end {t} lacks full multiplicity")
            run_knots_required(self, f, side, run)  # raises if not a refinement


def run_knots_required(cx: PatchComplex, f: Face, side: str, run: Run):
    """Return (carrier segment matrix, carrier seg knots, face seg knots); raise if I1 fails."""
    c = cx.carriers[run.carrier]
    lo, hi = sorted((run.s_a, run.s_b))
    A_c, K_c = bb.segment_matrix(c.knots, c.degree, lo, hi)
    if run.reversed:
        A_c, K_c = A_c[::-1], bb.reverse_knots(K_c)
    knots = f.side_knots(side)
    p = f.side_degree(side)
    inner = [k for k in knots if run.a + TOL < k < run.b - TOL]
    K_f = bb.affine_map_knots(np.concatenate([np.full(p + 1, run.a), inner, np.full(p + 1, run.b)]), run.a, run.b)
    if not bb.is_knot_subset(K_c, K_f):
        raise TopologyError(f"face {f.id} side {side}: side knots do not refine carrier {c.id} knots")
    return A_c, K_c, K_f


_RUN_CACHE: dict = {}


def run_matrix(cx: PatchComplex, f: Face, side: str, run: Run):
    """Return (row positions along the side, T) with side cps = T @ carrier cps.

    T depends only on the carrier knots, the face side knots and the run
    parameters, so it is cached across states (rewrites rebuild DOF maps often).
    """
    c = cx.carriers[run.carrier]
    knots = f.side_knots(side)
    p = f.side_degree(side)
    key = (tuple(np.round(c.knots, 12)), c.degree, tuple(np.round(knots, 12)), p,
           round(run.a, 12), round(run.b, 12), round(run.s_a, 12), round(run.s_b, 12))
    hit = _RUN_CACHE.get(key)
    if hit is not None:
        return hit
    A_c, K_c, K_f = run_knots_required(cx, f, side, run)
    ia = bb.interpolating_index(knots, p, run.a)
    ib = bb.interpolating_index(knots, p, run.b)
    T = bb.refinement_matrix(K_c, K_f, p) @ A_c
    if T.shape[0] != ib - ia + 1:
        raise TopologyError("run matrix size mismatch")
    T.setflags(write=False)
    if len(_RUN_CACHE) > 50_000:
        _RUN_CACHE.clear()
    _RUN_CACHE[key] = out = (np.arange(ia, ib + 1), T)
    return out
