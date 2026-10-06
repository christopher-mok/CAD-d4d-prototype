"""MergeFace(face_a, face_b): epsilon-gated inverse of SplitFace.

Provenance (same root face, adjacent root-domain rectangles, one shared full
side) identifies plausible candidates, but the merge is NOT assumed exact:
after independent optimization the children generally leave the parent's
spline space. The merge therefore

  1. concatenates the children into one patch with a C0 knot at tau
     (exact), with tau given by the children's root-domain sizes;
  2. removes as many copies of tau as epsilon allows: all of them (the
     parent's spline space) or, failing that, all but one/two (a C2/C1 knot
     remains; KnotRemove may take it later);
  3. refits the interior by least squares with carrier boundaries fixed;
  4. accepts only if  max ||S_before - S_after|| < eps_merge.

The shared carrier and edge are deleted; hanging vertices that become plain
subdivision points are merged away (MergeEdge).
"""
from __future__ import annotations

import numpy as np

from ..geometry import bspline_basis as bb
from ..geometry.fitting import fit_face_interior
from ..geometry.state import CADState
from ..geometry.topology import TOL, EdgeUse, Face, PatchComplex, Provenance, TopologyError
from .base import Rewrite, RewriteOutcome, face_center_point
from .knot_remove import validation_params
from .split_edge import can_merge_edges_at, merge_edge_in_place


def _scaled(uses, lo, hi):
    return [EdgeUse(u.edge, lo + u.a * (hi - lo), lo + u.b * (hi - lo), u.reversed) for u in uses]


def merge_layout(cx: PatchComplex, fa: int, fb: int):
    """Return (axis, first, second) if fa/fb can be concatenated, else None."""
    A, B = cx.faces[fa], cx.faces[fb]
    for axis, s_hi, s_lo in (("u", "u1", "u0"), ("v", "v1", "v0")):
        for first, second in ((A, B), (B, A)):
            w1, w2 = first.sides[s_hi], second.sides[s_lo]
            if len(w1) != 1 or len(w2) != 1 or w1[0].edge != w2[0].edge or w1[0].reversed != w2[0].reversed:
                continue
            if first.root != second.root:
                continue
            d1, d2 = first.domain, second.domain
            if axis == "u":
                ok = abs(d1[1] - d2[0]) < 1e-9 and abs(d1[2] - d2[2]) < 1e-9 and abs(d1[3] - d2[3]) < 1e-9
                other_ok = len(first.knots_v) == len(second.knots_v) and np.allclose(first.knots_v, second.knots_v, atol=TOL)
            else:
                ok = abs(d1[3] - d2[2]) < 1e-9 and abs(d1[0] - d2[0]) < 1e-9 and abs(d1[1] - d2[1]) < 1e-9
                other_ok = len(first.knots_u) == len(second.knots_u) and np.allclose(first.knots_u, second.knots_u, atol=TOL)
            if ok and other_ok:
                return axis, first.id, second.id
    return None


class MergeFace(Rewrite):
    kind = "MergeFace"
    exact = False
    refinement = False

    def __init__(self, face_a: int, face_b: int, eps: float = 1e-3, max_keep: int = 2):
        self.face_a, self.face_b, self.eps = face_a, face_b, float(eps)
        self.max_keep = int(max_keep)  # copies of the C0 knot a partial merge may keep

    def touched_faces(self, state):
        cx = state.cx
        out = {self.face_a, self.face_b}
        for f in (self.face_a, self.face_b):
            if f in cx.faces:
                out |= cx.neighbors(f)
        return out

    def location(self, state):
        if self.face_a in state.cx.faces:
            return face_center_point(state, self.face_a)
        return None

    def _apply(self, state: CADState) -> RewriteOutcome:
        old = state.copy()
        cx = state.cx
        if self.face_a not in cx.faces or self.face_b not in cx.faces:
            return RewriteOutcome(None, "face missing")
        lay = merge_layout(cx, self.face_a, self.face_b)
        if lay is None:
            return RewriteOutcome(None, "faces not mergeable (layout/provenance)")
        axis, f1_id, f2_id = lay
        f1, f2 = cx.faces[f1_id], cx.faces[f2_id]
        ax = 0 if axis == "u" else 2
        w1 = f1.domain[ax + 1] - f1.domain[ax]
        w2 = f2.domain[ax + 1] - f2.domain[ax]
        tau = w1 / (w1 + w2)
        p = f1.degree_u if axis == "u" else f1.degree_v
        k1 = f1.knots_u if axis == "u" else f1.knots_v
        k2 = f2.knots_u if axis == "u" else f2.knots_v
        inner1 = bb.interior_knots(k1, p) * tau
        inner2 = tau + bb.interior_knots(k2, p) * (1 - tau)
        merged_knots = bb.snap_knots(np.concatenate([np.zeros(p + 1), inner1, inner2, np.ones(p + 1)]))

        shared_edge = f1.sides["u1" if axis == "u" else "v1"][0].edge
        C = cx.edges[shared_edge].carrier
        if any(v.host is not None and v.host[0] == C for v in cx.vertices.values()):
            return RewriteOutcome(None, "shared carrier hosts vertices")
        if axis == "u":
            sides = {"v0": _scaled(f1.sides["v0"], 0, tau) + _scaled(f2.sides["v0"], tau, 1),
                     "v1": _scaled(f1.sides["v1"], 0, tau) + _scaled(f2.sides["v1"], tau, 1),
                     "u0": list(f1.sides["u0"]), "u1": list(f2.sides["u1"])}
            ku, kv = merged_knots, f1.knots_v.copy()
            dom = np.array([f1.domain[0], f2.domain[1], f1.domain[2], f1.domain[3]])
        else:
            sides = {"u0": _scaled(f1.sides["u0"], 0, tau) + _scaled(f2.sides["u0"], tau, 1),
                     "u1": _scaled(f1.sides["u1"], 0, tau) + _scaled(f2.sides["u1"], tau, 1),
                     "v0": list(f1.sides["v0"]), "v1": list(f2.sides["v1"])}
            ku, kv = f1.knots_u.copy(), merged_knots
            dom = np.array([f1.domain[0], f1.domain[1], f1.domain[2], f2.domain[3]])
        merged = Face(-1, ku, kv, np.zeros((1, 1, 3)), sides, f1.degree_u, f1.degree_v, dom, f1.root,
                      Provenance("MergeFace", (f1_id, f2_id), axis, float(tau)), f1.lineage | f2.lineage)
        nu, nv = merged.shape
        merged.interior = np.zeros((nu - 2, nv - 2, 3))
        X0, X1 = cx.carriers[C].v_start, cx.carriers[C].v_end
        del cx.faces[f1_id], cx.faces[f2_id]
        del cx.edges[shared_edge], cx.carriers[C]
        merged = cx.add_face(merged)
        for vid in (X0, X1):
            if can_merge_edges_at(cx, vid) is not None:
                merge_edge_in_place(cx, vid)
        knots_full = merged.knots_u if axis == "u" else merged.knots_v

        def old_eval(uv):
            t = uv[:, 0] if axis == "u" else uv[:, 1]
            out = np.empty((len(uv), 3))
            left = t <= tau
            for mask, fid, lo, hi in ((left, f1_id, 0.0, tau), (~left, f2_id, tau, 1.0)):
                if mask.any():
                    loc = uv[mask].copy()
                    j = 0 if axis == "u" else 1
                    loc[:, j] = np.clip((loc[:, j] - lo) / (hi - lo), 0, 1)
                    out[mask] = old.evaluate(fid, loc)
            return out

        # Remove as many copies of the C0 knot tau as epsilon allows: all p (the
        # parent's space), else leave a C1 (p-1 removed) or C... knot behind. Even a
        # partial merge removes a face, a carrier and DOFs; leftover copies can be
        # simplified later by KnotRemove.
        best_dev, reason = np.inf, "C0 knot required by boundary (cannot remove)"
        for keep in range(0, min(self.max_keep, p - 1) + 1):
            reduced = np.sort(np.concatenate([knots_full[np.abs(knots_full - tau) > TOL], [tau] * keep]))
            if axis == "u":
                merged.knots_u = reduced
            else:
                merged.knots_v = reduced
            nu, nv = merged.shape
            merged.interior = np.zeros((nu - 2, nv - 2, 3))
            try:
                for side in ("v0", "u1", "v1", "u0"):
                    cx.check_side_refinement(merged, side)
            except TopologyError:
                continue
            # validation breakpoints: the merged knots plus the removed C0 knot tau
            us = validation_params(np.sort(np.concatenate([merged.knots_u, [tau] if axis == "u" else []])),
                                   merged.degree_u)
            vs = validation_params(np.sort(np.concatenate([merged.knots_v, [tau] if axis == "v" else []])),
                                   merged.degree_v)
            U, V = np.meshgrid(us, vs, indexing="ij")
            Y = old_eval(np.stack([U.ravel(), V.ravel()], 1)).reshape(len(us), len(vs), 3)
            state.structure_changed()
            fit_face_interior(cx, merged.id, us, vs, Y, state.dof_map)
            state.structure_changed()
            uu, vv = 0.0005 + 0.999 * us, 0.0005 + 0.999 * vs
            dev = 0.0
            for a, b in ((us, vs), (uu, vv)):
                A, B = np.meshgrid(a, b, indexing="ij")
                uv = np.stack([A.ravel(), B.ravel()], 1)
                dev = max(dev, float(np.max(np.linalg.norm(state.evaluate(merged.id, uv) - old_eval(uv), axis=1))))
            if dev <= self.eps:
                return RewriteOutcome(state, deviation=dev, info={"merged_face": merged.id, "kept_multiplicity": keep,
                                                                  "removed_faces": [f1_id, f2_id]})
            best_dev = min(best_dev, dev)
            reason = f"deviation {dev:.3e} > eps {self.eps:.1e}"
        return RewriteOutcome(None, reason, deviation=best_dev if np.isfinite(best_dev) else 0.0)


def merge_face_candidates(state: CADState, eps: float) -> list[MergeFace]:
    cx = state.cx
    out, seen = [], set()
    inc = cx.edge_incidence()
    for f in cx.faces.values():
        for side in ("u1", "v1"):
            w = f.sides[side]
            if len(w) != 1:
                continue
            others = [g for g, _, _ in inc.get(w[0].edge, []) if g != f.id]
            for g in others:
                key = tuple(sorted((f.id, g)))
                if key not in seen and merge_layout(cx, f.id, g) is not None:
                    seen.add(key)
                    out.append(MergeFace(key[0], key[1], eps))
    return out
