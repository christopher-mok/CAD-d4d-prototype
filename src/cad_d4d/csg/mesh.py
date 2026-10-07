"""Surface extraction: marching tetrahedra on a Freudenthal (Kuhn) split of a node lattice.

Each lattice cube is split into the 6 tetrahedra around its main diagonal. On
a tetrahedral mesh the zero set of the piecewise-linear interpolant of the
SDF samples is a combinatorial 2-manifold when no sample is exactly zero (we
classify ``f < 0`` as inside and ``f >= 0`` as outside, so ties are broken
consistently). Surface vertices are created once per *lattice edge*
(keyed by its two node ids), so the mesh is welded by construction -- no
vertex merging by position, no triangle soup. ``mesh_topology`` still checks
closedness and manifoldness explicitly before it reports a genus.
"""
from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# tetrahedra of the unit cube: paths 0 -> e_a -> e_a + e_b -> (1,1,1)
_E = np.eye(3, dtype=int)
_TETS = [np.array([[0, 0, 0], _E[p[0]], _E[p[0]] + _E[p[1]], [1, 1, 1]]) for p in permutations(range(3))]


def node_lattice(lo: float, hi: float, n: int) -> np.ndarray:
    """(n+1)^3 x 3 node positions of a lattice with n cells per axis."""
    ax = np.linspace(lo, hi, n + 1)
    return np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)


def marching_tetrahedra(f: np.ndarray, P: np.ndarray, m: int):
    """Zero set of the nodal values ``f`` ((m^3,), inside: f < 0) on the lattice ``P`` with m nodes
    per axis. Returns (V, F, touches_boundary): vertices, outward-oriented triangles, and whether
    the inside region reaches the lattice boundary (then the surface is not closed)."""
    f = np.asarray(f, float).reshape(m, m, m)
    inside_all = f < 0
    touches = bool(inside_all[0].any() or inside_all[-1].any() or inside_all[:, 0].any() or inside_all[:, -1].any()
                   or inside_all[:, :, 0].any() or inside_all[:, :, -1].any())
    n = m - 1
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    base = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
    # keep only cubes with a sign change
    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    cube_in = inside_all[base[:, None, 0] + corners[None, :, 0], base[:, None, 1] + corners[None, :, 1],
                         base[:, None, 2] + corners[None, :, 2]]
    mixed = cube_in.any(1) & ~cube_in.all(1)
    base = base[mixed]
    fr = f.ravel()
    tets = []
    for T in _TETS:
        idx = base[:, None, :] + T[None]                      # (cubes, 4, 3)
        tets.append(idx[..., 0] * m * m + idx[..., 1] * m + idx[..., 2])
    tets = np.concatenate(tets)                               # (T, 4) node ids
    vals = fr[tets]
    inside = vals < 0
    code = (inside * (1 << np.arange(4))).sum(1)
    tri_edges, tri_src = [], []                               # per triangle group: edges (node pairs); source tets
    for c in range(1, 15):
        sel = code == c
        if not sel.any():
            continue
        tv = tets[sel]
        ins = [i for i in range(4) if c >> i & 1]
        outs = [i for i in range(4) if not c >> i & 1]
        if len(ins) in (1, 3):
            lone, others = (ins[0], outs) if len(ins) == 1 else (outs[0], ins)
            e = [(lone, o) for o in others]
            tri_edges.append(np.stack([np.stack([tv[:, a], tv[:, b]], 1) for a, b in e], 1))
            tri_src.append((sel, ins, outs))
        else:
            a, b = ins
            cc, d = outs
            quad = [(a, cc), (a, d), (b, d), (b, cc)]
            for t in ((0, 1, 2), (0, 2, 3)):
                e = [quad[i] for i in t]
                tri_edges.append(np.stack([np.stack([tv[:, x], tv[:, y]], 1) for x, y in e], 1))
                tri_src.append((sel, ins, outs))
    if not tri_edges:
        return np.zeros((0, 3)), np.zeros((0, 3), int), touches
    # inside/outside centroids per triangle (for orientation)
    in_c, out_c = [], []
    for sel, ins, outs in tri_src:
        tv = tets[sel]
        in_c.append(P[tv[:, ins]].mean(1))
        out_c.append(P[tv[:, outs]].mean(1))
    E = np.concatenate(tri_edges)                             # (F, 3, 2)
    in_c, out_c = np.concatenate(in_c), np.concatenate(out_c)
    lo_, hi_ = np.minimum(E[..., 0], E[..., 1]), np.maximum(E[..., 0], E[..., 1])
    keys = lo_.astype(np.int64) * (m ** 3) + hi_
    uniq, inv = np.unique(keys.ravel(), return_inverse=True)
    a_id, b_id = uniq // (m ** 3), uniq % (m ** 3)
    fa, fb = fr[a_id], fr[b_id]
    t = np.where(np.abs(fa - fb) > 0, fa / np.where(np.abs(fa - fb) > 0, fa - fb, 1.0), 0.5)
    V = P[a_id] + t[:, None] * (P[b_id] - P[a_id])
    F = inv.reshape(-1, 3)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = (nrm * (out_c - in_c)).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V, F, touches


def mesh_topology(F: np.ndarray, n_vertices: int) -> dict:
    """Closedness, manifoldness, shells, Euler characteristic and genus per shell of a welded mesh.

    ``genus`` is reported only when the mesh is closed and manifold (else None)."""
    F = np.asarray(F)
    if len(F) == 0:
        return {"closed": True, "manifold": True, "n_shells": 0, "chi": [], "genus": [], "genus_total": 0,
                "n_vertices": 0, "n_faces": 0}
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    tri_of_edge = np.tile(np.arange(len(F)), 3)
    es = np.sort(e, 1)
    ekey = es[:, 0].astype(np.int64) * n_vertices + es[:, 1]
    uk, einv, cnt = np.unique(ekey, return_inverse=True, return_counts=True)
    closed = bool(np.all(cnt == 2))
    used = np.unique(F)
    # vertex manifoldness: the corners of each vertex must form a single fan (one connected component)
    manifold = closed
    if closed:
        order = np.argsort(einv, kind="stable")
        pairs = order.reshape(-1, 2)                          # the two half-edges of every edge
        h0, h1 = pairs[:, 0], pairs[:, 1]
        t0, t1 = tri_of_edge[h0], tri_of_edge[h1]
        v0, v1 = es[h0, 0], es[h0, 1]
        def cid(t, v):
            return t * 3 + np.argmax(F[t] == v[:, None], axis=1)
        a = np.concatenate([cid(t0, v0), cid(t0, v1)])
        b = np.concatenate([cid(t1, v0), cid(t1, v1)])
        G = coo_matrix((np.ones(len(a)), (a, b)), shape=(3 * len(F), 3 * len(F)))
        n_cc, _ = connected_components(G, directed=False)
        manifold = n_cc == len(used)
    # shells: connected components over vertices
    G = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_vertices, n_vertices))
    _, vlab = connected_components(G, directed=False)
    shell_of_tri = vlab[F[:, 0]]
    shells = np.unique(vlab[used])
    chi, genus = [], []
    edge_shell = vlab[es[np.unique(einv, return_index=True)[1], 0]]
    for s in shells:
        Vs = int(np.sum(vlab[used] == s))
        Fs = int(np.sum(shell_of_tri == s))
        Es = int(np.sum(edge_shell == s))
        x = Vs - Es + Fs
        chi.append(x)
        genus.append((2 - x) // 2 if closed and manifold and (2 - x) % 2 == 0 and x <= 2 else None)
    valid = closed and manifold and all(g is not None for g in genus)
    return {"closed": closed, "manifold": manifold, "n_shells": int(len(shells)), "chi": chi,
            "genus": genus if valid else None, "genus_total": int(sum(genus)) if valid else None,
            "n_vertices": int(len(used)), "n_faces": int(len(F))}


def export_obj(path, V: np.ndarray, F: np.ndarray) -> Path:
    path = Path(path)
    with open(path, "w") as fh:
        fh.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V))
        fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F))
    return path
