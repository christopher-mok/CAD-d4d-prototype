"""Topology grammar: material-side and void-side rewrites of a CSG solid.

Every operation
  1. checks explicit preconditions on the current solid,
  2. performs its stated edit on a private copy (adds or removes features),
  3. measures the voxel topology before and after on a check grid, and
  4. fails -- with the measured counts in the reason -- unless the measured
     change is the operation's intended effect.
Failures are reported, never replaced by a different edit.

Material side
  AddBody(center, size)         add a material primitive in missing-material space.
                                Pre: center outside material; < 50% of the new
                                primitive already material. Effect: material volume grows.
  RemoveBody(feature)           delete an existing material feature (any: original,
                                added, bridge or plug). Pre: another material
                                feature remains. Effect: material volume shrinks.
  BridgeBodies(p, q, r)         material capsule p-q. Pre: p, q within r of material,
                                the segment leaves material somewhere. Effect:
                                components decrease OR a handle appears (tunnels +1).
  PinchBody(point, n, t, w)     separate a material neck at the plane (point, n):
                                * slab variant: subtract a void slab (thickness 2t,
                                  half-width w). Pre: point inside material.
                                * split variant (``split=feature``): replace that body
                                  by two bodies, each a moment-matched box of the
                                  body's cells on one side of the slab. Pre: point
                                  inside that body; the slab cuts it into >= 2 pieces.
                                Effect (both): components increase.
Void side
  AddCavity(center, size)       enclosed void primitive. Pre: center deeper than the
                                cavity's bounding radius + a wall of 2 check cells.
                                Effect: cavities +1.
  RemoveCavity(feature | point) delete the void feature forming a cavity, or (cavity
                                formed by material arrangement) fill it with a
                                material plug at ``point``. Effect: cavities -1.
  BridgeVoid(p, q, r)           void capsule p-q between two void regions.
                                Pre: p, q outside material, the segment crosses
                                material. Effect: tunnels +1 (exterior to exterior)
                                or cavities -1 (a cavity is connected to another void).
  OpenCavity(p, q, r)           BridgeVoid with one endpoint inside an enclosed
                                cavity. Effect: cavities -1.
  CloseTunnel(feature | plug)   delete a void channel feature, or add a material plug
                                across the hole. Pre: the solid has a tunnel or the
                                feature is a void. Effect: tunnels -1, or cavities +1
                                (re-sealing a cavity the channel had opened).

Inverses (used for temporary protection): AddBody <-> RemoveBody,
BridgeBodies <-> PinchBody, AddCavity <-> RemoveCavity,
BridgeVoid / OpenCavity <-> CloseTunnel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage

from .measure import S6, voxel_topology
from .primitives import DTYPE, Capsule, Primitive, Sphere, box_from_axes
from .solid import MATERIAL, VOID, CSGSolid, Grid

INVERSE = {"AddBody": "RemoveBody", "RemoveBody": "AddBody", "BridgeBodies": "PinchBody",
           "PinchBody": "BridgeBodies", "AddCavity": "RemoveCavity", "RemoveCavity": "AddCavity",
           "BridgeVoid": "CloseTunnel", "OpenCavity": "CloseTunnel", "CloseTunnel": "BridgeVoid"}


@dataclass
class OpContext:
    check: Grid
    min_voxels: int = 8
    round: int = 0

    @property
    def wall(self) -> float:
        return 2.0 * self.check.h


@dataclass
class OpOutcome:
    solid: CSGSolid | None
    reason: str = ""
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.solid is not None


def topo(solid: CSGSolid, ctx: OpContext) -> dict:
    occ = solid.hard(ctx.check.points).cpu().numpy().reshape(ctx.check.shape)
    return voxel_topology(occ, ctx.min_voxels)


def _sdf_at(solid: CSGSolid, p) -> float:
    with torch.no_grad():
        return float(solid.sdf(torch.as_tensor(np.asarray(p, float)[None], dtype=DTYPE))[0])


def _segment_sdf(solid: CSGSolid, p, q, n: int = 48) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    P = np.asarray(p, float) * (1 - t) + np.asarray(q, float) * t
    with torch.no_grad():
        return solid.sdf(torch.as_tensor(P, dtype=DTYPE)).cpu().numpy()


def _counts(t: dict) -> str:
    return f"components {t['components']}, cavities {t['cavities']}, tunnels {t['tunnels']}"


class TopoOp:
    name = "op"
    removes = False   # targets an existing feature for removal

    def params(self) -> dict:
        return {}

    def location(self) -> np.ndarray:
        raise NotImplementedError

    def radius(self) -> float:
        return 0.0

    def target_feature(self) -> int | None:
        return getattr(self, "fid", None)

    def _apply(self, s: CSGSolid, ctx: OpContext, before: dict) -> tuple[str, list[int], list[int]]:
        raise NotImplementedError

    def intended(self, before: dict, after: dict) -> bool:
        raise NotImplementedError

    expected = ""

    def apply(self, solid: CSGSolid, ctx: OpContext, provenance: dict | None = None) -> OpOutcome:
        s = solid.copy()
        before = topo(s, ctx)
        self._prov = {"op": self.name, "round": ctx.round, **(provenance or {})}
        reason, created, removed = self._apply(s, ctx, before)
        if reason:
            return OpOutcome(None, f"{self.name}: precondition failed: {reason}",
                             {"before": _counts(before)})
        after = topo(s, ctx)
        info = {"before": _counts(before), "after": _counts(after), "created": created, "removed": removed,
                "volume_change": after["material_voxels"] - before["material_voxels"]}
        if not self.intended(before, after):
            return OpOutcome(None, f"{self.name}: no {self.expected} (measured {_counts(before)} -> {_counts(after)})",
                             info)
        return OpOutcome(s, "", info)

    def describe(self) -> dict:
        return {"op": self.name, **self.params()}

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params().items())})"


def _r(x):
    return np.round(np.asarray(x, float), 3).tolist()


# ---------------------------------------------------------------- material side
class AddBody(TopoOp):
    name, expected = "AddBody", "material increase"

    def __init__(self, center, radius: float | None = None, axes=None, half_extents=None):
        self.center = np.asarray(center, float)
        self.r, self.axes, self.half = radius, axes, None if half_extents is None else np.asarray(half_extents, float)

    def prim(self) -> Primitive:
        if self.half is not None:
            return box_from_axes(self.center, self.axes, self.half)
        return Sphere(self.center, self.r)

    def params(self):
        return {"center": _r(self.center), **({"radius": round(self.r, 3)} if self.half is None else
                                              {"half_extents": _r(self.half)})}

    def location(self):
        return self.center

    def radius(self):
        return float(self.r if self.half is None else np.linalg.norm(self.half))

    def _apply(self, s, ctx, before):
        if _sdf_at(s, self.center) <= 0:
            return "center lies inside existing material", [], []
        p = self.prim()
        X = ctx.check.points
        with torch.no_grad():
            mine = p.sdf(X) < 0
            if int(mine.sum()) == 0:
                return "body smaller than the check grid resolution", [], []
            if float((s.hard(X) & mine).sum()) / float(mine.sum()) > 0.5:
                return "body would lie mostly inside existing material", [], []
        f = s.add(MATERIAL, "body", p, **self._prov)
        return "", [f.id], []

    def intended(self, before, after):
        return after["material_voxels"] > before["material_voxels"]


class RemoveBody(TopoOp):
    name, expected, removes = "RemoveBody", "material decrease", True

    def __init__(self, fid: int, center=None):
        self.fid, self.center = int(fid), center

    def params(self):
        return {"feature": self.fid}

    def location(self):
        return np.asarray(self.center if self.center is not None else np.zeros(3), float)

    def _apply(self, s, ctx, before):
        if not s.has(self.fid):
            return f"feature {self.fid} does not exist", [], []
        f = s.get(self.fid)
        if f.role != MATERIAL:
            return f"feature {self.fid} is not material", [], []
        if len(s.material) <= 1:
            return "cannot remove the last material feature", [], []
        self.center = f.prim.center.detach().numpy().copy()
        s.remove(self.fid)
        return "", [], [self.fid]

    def intended(self, before, after):
        return after["material_voxels"] < before["material_voxels"]


class BridgeBodies(TopoOp):
    name, expected = "BridgeBodies", "connection (components did not decrease, no handle appeared)"

    def __init__(self, p, q, r: float):
        self.p, self.q, self.r = np.asarray(p, float), np.asarray(q, float), float(r)

    def params(self):
        return {"p": _r(self.p), "q": _r(self.q), "radius": round(self.r, 3)}

    def location(self):
        return 0.5 * (self.p + self.q)

    def radius(self):
        return 0.5 * float(np.linalg.norm(self.q - self.p)) + self.r

    def _apply(self, s, ctx, before):
        if _sdf_at(s, self.p) > self.r or _sdf_at(s, self.q) > self.r:
            return "an endpoint is farther than the bridge radius from material", [], []
        if _segment_sdf(s, self.p, self.q).max() <= 0:
            return "endpoints are already connected through material along the segment", [], []
        f = s.add(MATERIAL, "bridge", Capsule(self.p, self.q, self.r), **self._prov)
        return "", [f.id], []

    def intended(self, before, after):
        return after["components"] < before["components"] or after["tunnels"] > before["tunnels"]


class PinchBody(TopoOp):
    name, expected = "PinchBody", "separation (components did not increase)"

    def __init__(self, point, normal, half_thickness: float, half_width: float, split: int | None = None):
        self.point = np.asarray(point, float)
        n = np.asarray(normal, float)
        self.normal = n / np.linalg.norm(n)
        self.t, self.w = float(half_thickness), float(half_width)
        self.split = None if split is None else int(split)

    def params(self):
        out = {"point": _r(self.point), "normal": _r(self.normal), "half_thickness": round(self.t, 3),
               "half_width": round(self.w, 3)}
        if self.split is not None:
            out["split"] = self.split
        return out

    def target_feature(self):
        return None

    def location(self):
        return self.point

    def radius(self):
        return self.w

    def _slab(self):
        n = self.normal
        a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        return box_from_axes(self.point, np.stack([n, a, b], 1), [self.t, self.w, self.w], rounding=0.2)

    def _apply(self, s, ctx, before):
        if _sdf_at(s, self.point) >= 0:
            return "point is not inside material", [], []
        if self.split is None:
            f = s.add(VOID, "slab", self._slab(), **self._prov)
            return "", [f.id], []
        if not s.has(self.split) or s.get(self.split).role != MATERIAL:
            return f"feature {self.split} is not an existing material feature", [], []
        body = s.get(self.split)
        X = ctx.check.points
        with torch.no_grad():
            if float(body.prim.sdf(torch.as_tensor(self.point[None], dtype=DTYPE))[0]) >= 0:
                return f"point is not inside feature {self.split}", [], []
            cells = ((body.prim.sdf(X) < 0) & (self._slab().sdf(X) >= 0)).numpy().reshape(ctx.check.shape)
        lab, k = ndimage.label(cells, structure=np.ones((3, 3, 3), bool))
        sizes = np.bincount(lab.ravel(), minlength=k + 1)[1:]
        pieces = [i + 1 for i in np.argsort(-sizes)[:2] if sizes[i] >= ctx.min_voxels]
        if len(pieces) < 2:
            return f"the slab does not cut feature {self.split} into two pieces", [], []
        s.remove(self.split)
        created = []
        P = X.numpy().reshape(ctx.check.shape + (3,))
        for i in pieces:
            Q = P[lab == i]
            c = Q.mean(0)
            ev, evec = np.linalg.eigh(np.cov((Q - c).T))
            half = np.maximum(np.sqrt(3.0 * np.maximum(ev, 0)), ctx.check.h)
            created.append(s.add(MATERIAL, "body", box_from_axes(c, evec, half, rounding=0.5),
                                 split_from=self.split, **self._prov).id)
        return "", created, [self.split]

    def intended(self, before, after):
        return after["components"] > before["components"]


# ---------------------------------------------------------------- void side
class AddCavity(TopoOp):
    name, expected = "AddCavity", "new enclosed cavity"

    def __init__(self, center, radius: float | None = None, axes=None, half_extents=None):
        self.center = np.asarray(center, float)
        self.r, self.axes, self.half = radius, axes, None if half_extents is None else np.asarray(half_extents, float)

    def prim(self):
        if self.half is not None:
            return box_from_axes(self.center, self.axes, self.half, rounding=0.5)
        return Sphere(self.center, self.r)

    def params(self):
        return {"center": _r(self.center), **({"radius": round(self.r, 3)} if self.half is None else
                                              {"half_extents": _r(self.half)})}

    def location(self):
        return self.center

    def radius(self):
        return float(self.r if self.half is None else np.linalg.norm(self.half))

    def _apply(self, s, ctx, before):
        p = self.prim()
        depth = -_sdf_at(s, self.center)
        if depth < p.bounding_radius() + ctx.wall:
            return (f"not enclosed: center depth {depth:.3f} < cavity radius {p.bounding_radius():.3f} "
                    f"+ wall {ctx.wall:.3f}"), [], []
        f = s.add(VOID, "cavity", p, **self._prov)
        return "", [f.id], []

    def intended(self, before, after):
        return after["cavities"] > before["cavities"]


def _cavity_label_at(s: CSGSolid, ctx: OpContext, point) -> int:
    occ = s.hard(ctx.check.points).cpu().numpy().reshape(ctx.check.shape)
    vlab, _ = ndimage.label(np.pad(~occ, 1), structure=S6)
    border = set(np.unique(np.concatenate([vlab[0].ravel(), vlab[-1].ravel(), vlab[:, 0].ravel(), vlab[:, -1].ravel(),
                                           vlab[:, :, 0].ravel(), vlab[:, :, -1].ravel()])).tolist())
    ijk = np.clip(np.round((np.asarray(point) - ctx.check.lo) / ctx.check.h - 0.5 - ctx.check.offset).astype(int),
                  0, ctx.check.n - 1)
    lab = int(vlab[tuple(ijk + 1)])
    return lab if lab and lab not in border else 0


class RemoveCavity(TopoOp):
    name, expected, removes = "RemoveCavity", "cavity removal (cavities did not decrease)", True

    def __init__(self, fid: int | None = None, point=None, radius: float | None = None, axes=None,
                 half_extents=None):
        self.fid = None if fid is None else int(fid)
        self.point = None if point is None else np.asarray(point, float)
        self.r = radius
        self.axes, self.half = axes, None if half_extents is None else np.asarray(half_extents, float)

    def params(self):
        if self.fid is not None:
            return {"feature": self.fid}
        size = {"radius": round(self.r, 3)} if self.half is None else {"half_extents": _r(self.half)}
        return {"plug_at": _r(self.point), **size}

    def location(self):
        return self.point if self.point is not None else np.zeros(3)

    def radius(self):
        return float(np.linalg.norm(self.half)) if self.half is not None else float(self.r or 0.0)

    def _apply(self, s, ctx, before):
        if before["cavities"] == 0:
            return "the solid has no enclosed cavity", [], []
        if self.fid is not None:
            if not s.has(self.fid) or s.get(self.fid).role != VOID:
                return f"feature {self.fid} is not an existing void", [], []
            self.point = s.get(self.fid).prim.center.detach().numpy().copy()
            s.remove(self.fid)
            return "", [], [self.fid]
        if _cavity_label_at(s, ctx, self.point) == 0:
            return "point is not inside an enclosed cavity", [], []
        plug = Sphere(self.point, self.r) if self.half is None else box_from_axes(self.point, self.axes, self.half)
        f = s.add(MATERIAL, "plug", plug, **self._prov)
        return "", [f.id], []

    def intended(self, before, after):
        return after["cavities"] < before["cavities"]


class BridgeVoid(TopoOp):
    name, expected = "BridgeVoid", "void connection (no new tunnel, no cavity opened)"

    def __init__(self, p, q, r: float):
        self.p, self.q, self.r = np.asarray(p, float), np.asarray(q, float), float(r)

    def params(self):
        return {"p": _r(self.p), "q": _r(self.q), "radius": round(self.r, 3)}

    def location(self):
        return 0.5 * (self.p + self.q)

    def radius(self):
        return 0.5 * float(np.linalg.norm(self.q - self.p)) + self.r

    def _pre(self, s, ctx):
        if _sdf_at(s, self.p) <= 0 or _sdf_at(s, self.q) <= 0:
            return "an endpoint lies inside material"
        if _segment_sdf(s, self.p, self.q).min() >= 0:
            return "the segment does not cross material"
        return ""

    def _apply(self, s, ctx, before):
        reason = self._pre(s, ctx)
        if reason:
            return reason, [], []
        f = s.add(VOID, "channel", Capsule(self.p, self.q, self.r), **self._prov)
        return "", [f.id], []

    def intended(self, before, after):
        return after["tunnels"] > before["tunnels"] or after["cavities"] < before["cavities"]


class OpenCavity(BridgeVoid):
    name, expected = "OpenCavity", "cavity opening (cavities did not decrease)"

    def _pre(self, s, ctx):
        reason = super()._pre(s, ctx)
        if reason:
            return reason
        if not (_cavity_label_at(s, ctx, self.p) or _cavity_label_at(s, ctx, self.q)):
            return "neither endpoint lies in an enclosed cavity"
        return ""

    def intended(self, before, after):
        return after["cavities"] < before["cavities"]


class CloseTunnel(TopoOp):
    name, expected, removes = "CloseTunnel", "closure (tunnels did not decrease, no cavity re-sealed)", True

    def __init__(self, fid: int | None = None, center=None, axes=None, half_extents=None):
        self.fid = None if fid is None else int(fid)
        self.center = None if center is None else np.asarray(center, float)
        self.axes, self.half = axes, None if half_extents is None else np.asarray(half_extents, float)

    def params(self):
        return {"feature": self.fid} if self.fid is not None else {"plug_at": _r(self.center),
                                                                    "half_extents": _r(self.half)}

    def location(self):
        return self.center if self.center is not None else np.zeros(3)

    def radius(self):
        return float(np.linalg.norm(self.half)) if self.half is not None else 0.0

    def _apply(self, s, ctx, before):
        if self.fid is not None:
            if not s.has(self.fid) or s.get(self.fid).role != VOID:
                return f"feature {self.fid} is not an existing void", [], []
            self.center = s.get(self.fid).prim.center.detach().numpy().copy()
            s.remove(self.fid)
            return "", [], [self.fid]
        if before["tunnels"] < 1:
            return "the solid has no tunnel", [], []
        f = s.add(MATERIAL, "plug", box_from_axes(self.center, self.axes, self.half), **self._prov)
        return "", [f.id], []

    def intended(self, before, after):
        return after["tunnels"] < before["tunnels"] or after["cavities"] > before["cavities"]


ALL_OPS = (AddBody, RemoveBody, BridgeBodies, PinchBody, AddCavity, RemoveCavity, BridgeVoid, OpenCavity, CloseTunnel)
__all__ = [c.__name__ for c in ALL_OPS] + ["INVERSE", "OpContext", "OpOutcome", "topo"]
