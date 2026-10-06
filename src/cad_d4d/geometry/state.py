"""CAD state x = (s, p): structure + canonical continuous parameters.

The canonical DOFs (``p``) are 3D points of three kinds:

* ``('v', vid)``          free vertex positions,
* ``('c', cid, k)``       interior control points of carrier curves,
* ``('f', fid, i, j)``    interior control points of faces.

Every face control net is a sparse linear function of ``p``::

    net_f = E_f @ p          (rows: flattened (i, j) net indices)

Boundary rows of ``E_f`` are exact knot-insertion/extraction maps of carrier
rows, and hanging vertices are basis evaluations of their host carriers, so
gradients from all incident faces accumulate on the same shared DOFs and the
complex is watertight for every ``p``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from . import bspline_basis as bb
from .topology import SIDES, PatchComplex, TopologyError, run_matrix


class DofMap:
    """Index of canonical DOFs and the linear map to every face control net."""

    def __init__(self, cx: PatchComplex):
        self.cx = cx
        owners: list[tuple] = []
        values: list[np.ndarray] = []
        self.vertex_dof: dict[int, int] = {}
        self.carrier_dof: dict[int, np.ndarray] = {}
        self.face_dof: dict[int, np.ndarray] = {}
        for vid in sorted(cx.vertices):
            v = cx.vertices[vid]
            if v.free:
                self.vertex_dof[vid] = len(owners)
                owners.append(("v", vid))
                values.append(v.position)
        for cid in sorted(cx.carriers):
            c = cx.carriers[cid]
            start = len(owners)
            for k in range(c.n - 2):
                owners.append(("c", cid, k))
                values.append(c.interior[k])
            self.carrier_dof[cid] = np.arange(start, len(owners))
        for fid in sorted(cx.faces):
            f = cx.faces[fid]
            nu, nv = f.shape
            start = len(owners)
            for i in range(nu - 2):
                for j in range(nv - 2):
                    owners.append(("f", fid, i, j))
                    values.append(f.interior[i, j])
            self.face_dof[fid] = np.arange(start, len(owners)).reshape(nu - 2, nv - 2)
        self.owners = owners
        self.n_dof = len(owners)
        self.values = np.array(values, dtype=float).reshape(-1, 3)
        self._carrier_rows: dict[int, sp.csr_matrix] = {}
        self._vertex_rows: dict[int, sp.csr_matrix] = {}
        self._face_rows: dict[int, sp.csr_matrix] = {}
        self.face_order = sorted(cx.faces)
        self._E = None

    # -- row construction --------------------------------------------------
    def _unit(self, idx) -> sp.csr_matrix:
        idx = np.atleast_1d(idx)
        return sp.csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), idx)), shape=(len(idx), self.n_dof))

    def vertex_row(self, vid: int) -> sp.csr_matrix:
        if vid not in self._vertex_rows:
            v = self.cx.vertices[vid]
            if v.free:
                row = self._unit(self.vertex_dof[vid])
            else:
                cid, s = v.host
                c = self.cx.carriers[cid]
                B = sp.csr_matrix(bb.basis_matrix(c.knots, c.degree, [s]))
                row = (B @ self.carrier_rows(cid)).tocsr()
            self._vertex_rows[vid] = row
        return self._vertex_rows[vid]

    def carrier_rows(self, cid: int) -> sp.csr_matrix:
        if cid not in self._carrier_rows:
            c = self.cx.carriers[cid]
            rows = [self.vertex_row(c.v_start), self._unit(self.carrier_dof[cid]), self.vertex_row(c.v_end)]
            self._carrier_rows[cid] = sp.vstack(rows).tocsr()
        return self._carrier_rows[cid]

    def face_rows(self, fid: int, check: bool = False, tol: float = 1e-11) -> sp.csr_matrix:
        """Sparse (nu * nv, n_dof) map from DOFs to the flattened face control net."""
        if fid in self._face_rows and not check:
            return self._face_rows[fid]
        f = self.cx.faces[fid]
        nu, nv = f.shape
        n_rows = nu * nv
        assigned = np.zeros(n_rows, bool)
        blocks_r, blocks_c, blocks_v = [], [], []

        def put(rows_idx: np.ndarray, M: sp.spmatrix):
            M = sp.csr_matrix(M)
            keep = ~assigned[rows_idx]
            if check and np.any(~keep):
                prev = self._assemble(blocks_r, blocks_c, blocks_v, n_rows)[rows_idx[~keep]]
                diff = abs(prev - M[np.flatnonzero(~keep)]).max() if prev.nnz or M.nnz else 0.0
                if diff > tol:
                    raise TopologyError(f"face {fid}: inconsistent shared boundary rows (diff {diff:.2e})")
            sel = np.flatnonzero(keep)
            if len(sel) == 0:
                return
            sub = M[sel].tocoo()
            blocks_r.append(rows_idx[sel][sub.row])
            blocks_c.append(sub.col)
            blocks_v.append(sub.data)
            assigned[rows_idx[sel]] = True

        # interior rows
        interior_idx = np.flatnonzero(f.interior_mask())
        put(interior_idx, self._unit(self.face_dof[fid].reshape(-1)))
        for side in SIDES:
            side_idx = f.side_row_indices(side)
            for run in self.cx.side_runs(f, side):
                pos, T = run_matrix(self.cx, f, side, run)
                put(side_idx[pos], sp.csr_matrix(T) @ self.carrier_rows(run.carrier))
        if not assigned.all():
            raise TopologyError(f"face {fid}: unassigned control-net rows")
        R = self._assemble(blocks_r, blocks_c, blocks_v, n_rows)
        self._face_rows[fid] = R
        return R

    def _assemble(self, r, c, v, n_rows) -> sp.csr_matrix:
        if not r:
            return sp.csr_matrix((n_rows, self.n_dof))
        M = sp.coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(n_rows, self.n_dof))
        M = M.tocsr()
        M.eliminate_zeros()
        return M

    @property
    def E(self) -> sp.csr_matrix:
        """Stacked map for all faces in ``face_order``."""
        if self._E is None:
            self._E = sp.vstack([self.face_rows(fid) for fid in self.face_order]).tocsr()
        return self._E

    def face_row_offsets(self) -> dict[int, int]:
        out, off = {}, 0
        for fid in self.face_order:
            out[fid] = off
            nu, nv = self.cx.faces[fid].shape
            off += nu * nv
        return out

    # -- value helpers -----------------------------------------------------
    def net(self, fid: int, P=None) -> np.ndarray:
        P = self.values if P is None else P
        nu, nv = self.cx.faces[fid].shape
        return (self.face_rows(fid) @ P).reshape(nu, nv, 3)

    def scatter(self, P: np.ndarray) -> None:
        """Write DOF values back into the complex."""
        P = np.asarray(P, dtype=float)
        for k, owner in enumerate(self.owners):
            if owner[0] == "v":
                self.cx.vertices[owner[1]].position = P[k].copy()
            elif owner[0] == "c":
                self.cx.carriers[owner[1]].interior[owner[2]] = P[k]
            else:
                self.cx.faces[owner[1]].interior[owner[2], owner[3]] = P[k]
        self.values = P.copy()


@dataclass
class BirthRecord:
    """Bookkeeping for a structural refinement (complexity grace period).

    ``remaining`` is the part of ``delta_complexity`` not yet removed by later
    simplifications; only that part receives the grace discount.
    """
    id: int
    kind: str
    delta_complexity: float
    age: int = 0
    remaining: float | None = None
    simplified: bool = False
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = max(0.0, self.delta_complexity)


class CADState:
    """x = (s, p). Rewrites return new states; continuous steps update values."""

    def __init__(self, cx: PatchComplex, birth_records: list | None = None, meta: dict | None = None):
        self.cx = cx
        self.birth_records: list[BirthRecord] = birth_records or []
        self.meta: dict = meta or {}
        self._dof_map: DofMap | None = None
        self._cache: dict = {}

    def copy(self) -> "CADState":
        return CADState(self.cx.copy(), copy.deepcopy(self.birth_records), copy.deepcopy(self.meta))

    def structure_changed(self) -> None:
        self._dof_map = None
        self._cache = {}

    @property
    def dof_map(self) -> DofMap:
        if self._dof_map is None:
            self._dof_map = DofMap(self.cx)
        return self._dof_map

    @property
    def cache(self) -> dict:
        """Per-structure cache (discretizations, operators). Cleared on structural change."""
        return self._cache

    def values(self) -> np.ndarray:
        return self.dof_map.values.copy()

    def set_values(self, P) -> None:
        self.dof_map.scatter(np.asarray(P, dtype=float))

    def nets(self) -> dict[int, np.ndarray]:
        dm = self.dof_map
        return {fid: dm.net(fid) for fid in dm.face_order}

    @property
    def n_faces(self) -> int:
        return len(self.cx.faces)

    @property
    def n_control_points(self) -> int:
        """Number of canonical (independent) control points."""
        return self.dof_map.n_dof

    @property
    def n_raw_control_points(self) -> int:
        return int(sum(np.prod(f.shape) for f in self.cx.faces.values()))

    def evaluate(self, fid: int, uv: np.ndarray, deriv=(0, 0)) -> np.ndarray:
        f = self.cx.faces[fid]
        return f.patch.evaluate_points(self.dof_map.net(fid), uv, deriv)


def watertightness_error(state: CADState, n: int = 33) -> float:
    """Max distance between the two incident faces along every shared edge.

    Evaluates the actual patches (not the carriers) at corresponding parameters.
    """
    cx = state.cx
    worst = 0.0
    for eid, uses in cx.edge_incidence().items():
        e = cx.edges[eid]
        s = np.linspace(e.s0, e.s1, n)
        pts = []
        for fid, side, k in uses:
            use = cx.faces[fid].sides[side][k]
            sa, sb = cx.use_carrier_params(use)
            t = use.a + (s - sa) * (use.b - use.a) / (sb - sa)
            pts.append(state.evaluate(fid, cx.faces[fid].side_uv(side, t)))
        c = cx.carriers[e.carrier]
        dm = state.dof_map
        carrier_pts = bb.evaluate_curve(c.knots, c.degree, dm.carrier_rows(c.id) @ dm.values, s)
        for P in pts:
            worst = max(worst, float(np.max(np.linalg.norm(P - carrier_pts, axis=1))))
    return worst


def evaluate_root(state: CADState, root: int, UV: np.ndarray) -> np.ndarray:
    """Evaluate the surface at root-face parameters (U, V) regardless of subdivision.

    Every face covers a rectangle ``domain`` of its root face's parameter
    square; this allows comparing states that differ by splits/merges.
    """
    UV = np.atleast_2d(UV)
    out = np.full((len(UV), 3), np.nan)
    for f in state.cx.faces.values():
        if f.root != root:
            continue
        d = f.domain
        m = ((UV[:, 0] >= d[0] - 1e-12) & (UV[:, 0] <= d[1] + 1e-12)
             & (UV[:, 1] >= d[2] - 1e-12) & (UV[:, 1] <= d[3] + 1e-12) & np.isnan(out[:, 0]))
        if m.any():
            loc = np.stack([(UV[m, 0] - d[0]) / (d[1] - d[0]), (UV[m, 1] - d[2]) / (d[3] - d[2])], 1)
            out[m] = state.evaluate(f.id, np.clip(loc, 0.0, 1.0))
    return out


def root_deviation(a: CADState, b: CADState, n: int = 61) -> float:
    """Max distance between two states sampled on a dense root-parameter grid."""
    t = np.linspace(0, 1, n)
    U, V = np.meshgrid(t, t, indexing="ij")
    UV = np.stack([U.ravel(), V.ravel()], 1)
    roots = {f.root for f in a.cx.faces.values()} | {f.root for f in b.cx.faces.values()}
    worst = 0.0
    for r in roots:
        pa, pb = evaluate_root(a, r, UV), evaluate_root(b, r, UV)
        worst = max(worst, float(np.max(np.linalg.norm(pa - pb, axis=1))))
    return worst
