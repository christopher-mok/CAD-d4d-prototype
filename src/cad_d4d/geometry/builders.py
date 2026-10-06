"""Construction of initial watertight patch complexes.

``build_cube_complex`` creates the 6-face / 12-edge / 8-vertex quad layout of a
cube and fits every carrier and face to ``shape_map(point_on_cube)``. With the
default map (radial projection) this is the classic cube-sphere.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import bspline_basis as bb
from .fitting import dense_params, fit_carrier_interior, fit_face_interior
from .state import CADState, DofMap
from .topology import EdgeUse, Face, PatchComplex


def sphere_map(radius: float = 1.0, center=(0.0, 0.0, 0.0)) -> Callable:
    center = np.asarray(center, float)

    def f(x):
        x = np.atleast_2d(x)
        return center + radius * x / np.linalg.norm(x, axis=1, keepdims=True)
    return f


def ellipsoid_map(radii=(1.0, 0.8, 0.7)) -> Callable:
    radii = np.asarray(radii, float)

    def f(x):
        x = np.atleast_2d(x)
        return radii * x / np.linalg.norm(x, axis=1, keepdims=True)
    return f


def cube_quads():
    """Cube corners and outward-oriented quads (a, b, c, d) with S(0,0)=a, S(1,0)=b, S(1,1)=c, S(0,1)=d."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    quads = []
    for axis in range(3):
        for sign in (-1, 1):
            n = np.zeros(3)
            n[axis] = sign
            t1 = np.zeros(3)
            t1[(axis + 1) % 3] = 1
            t2 = np.cross(n, t1)
            c = n
            pts = [c - t1 - t2, c + t1 - t2, c + t1 + t2, c - t1 + t2]
            idx = [int(np.argmin(np.linalg.norm(corners - p, axis=1))) for p in pts]
            quads.append(idx)
    return corners, quads


def build_cube_complex(shape_map: Callable | None = None, n_interior_knots: int = 0,
                       carrier_interior_knots: int | None = None) -> CADState:
    """Watertight 6-patch bicubic complex fitted to ``shape_map`` of the cube surface."""
    shape_map = shape_map or sphere_map()
    n_ck = n_interior_knots if carrier_interior_knots is None else carrier_interior_knots
    corners, quads = cube_quads()
    cx = PatchComplex()
    vids = [cx.add_vertex(shape_map(c)[0]).id for c in corners]
    face_knots = bb.clamped_uniform_knots(n_interior_knots)
    carrier_knots = bb.clamped_uniform_knots(n_ck)
    n_c = bb.num_basis(carrier_knots, 3)

    carriers: dict[tuple, tuple] = {}  # (lo, hi) corner pair -> (carrier, edge)
    for q in quads:
        for a, b in zip(q, q[1:] + q[:1]):
            key = (min(a, b), max(a, b))
            if key in carriers:
                continue
            lo, hi = key
            c = cx.add_carrier(carrier_knots, vids[lo], vids[hi], np.zeros((n_c - 2, 3)))
            s = dense_params(carrier_knots, 3, 12)
            pts = shape_map(corners[lo][None] * (1 - s[:, None]) + corners[hi][None] * s[:, None])
            fit_carrier_interior(c, cx.vertices[vids[lo]].position, cx.vertices[vids[hi]].position, s, pts)
            e = cx.add_edge(c.id, 0.0, 1.0, vids[lo], vids[hi])
            carriers[key] = (c, e)

    def use(a, b):
        c, e = carriers[(min(a, b), max(a, b))]
        return [EdgeUse(e.id, 0.0, 1.0, reversed=a > b)]

    n_f = bb.num_basis(face_knots, 3)
    face_targets = {}
    for q in quads:
        a, b, c, d = q
        sides = {"v0": use(a, b), "u1": use(b, c), "v1": use(d, c), "u0": use(a, d)}
        f = cx.add_face(Face(-1, face_knots.copy(), face_knots.copy(), np.zeros((n_f - 2, n_f - 2, 3)), sides))
        face_targets[f.id] = (corners[a], corners[b], corners[d])

    dm = DofMap(cx)
    for fid, (pa, pb, pd) in face_targets.items():
        us = dense_params(face_knots, 3, 8)
        U, V = np.meshgrid(us, us, indexing="ij")
        flat = pa + U[..., None] * (pb - pa) + V[..., None] * (pd - pa)
        target = shape_map(flat.reshape(-1, 3)).reshape(len(us), len(us), 3)
        fit_face_interior(cx, fid, us, us, target, dm)
    cx.check_invariants()
    return CADState(cx)
