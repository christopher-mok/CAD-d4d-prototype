"""Parameterized solid primitives with exact signed distance functions.

Sign convention (used everywhere in ``cad_d4d.csg``): ``sdf < 0`` inside the
primitive, ``> 0`` outside, ``|sdf|`` = Euclidean distance to its surface.

Every primitive owns a dict of float64 tensors (its trainable parameters) and
a ``project_()`` that restores validity after a gradient step (minimum sizes,
rounding below the half-extents). Minimum sizes keep features from shrinking
below what the measurement grids resolve: a primitive never becomes a
near-zero sliver that only exists between samples.
"""
from __future__ import annotations

import math

import torch

DTYPE = torch.float64


def as_t(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=DTYPE).clone()


def rotation_matrix(w: torch.Tensor) -> torch.Tensor:
    """Rotation for the axis-angle vector ``w`` (Rodrigues; smooth at w = 0)."""
    th2 = (w * w).sum()
    th = torch.sqrt(th2 + 1e-30)
    zero = torch.zeros((), dtype=w.dtype)
    K = torch.stack([torch.stack([zero, -w[2], w[1]]), torch.stack([w[2], zero, -w[0]]),
                     torch.stack([-w[1], w[0], zero])])
    small = th2 < 1e-12
    a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th)) / torch.where(small, torch.ones_like(th2), th2))
    return torch.eye(3, dtype=w.dtype) + a * K + b * (K @ K)


class Primitive:
    kind = "primitive"

    def __init__(self, **params):
        self.params = {k: as_t(v) for k, v in params.items()}

    # -- interface ---------------------------------------------------------
    def sdf(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def project_(self, r_min: float) -> None:
        """Restore validity in place (no autograd)."""

    @property
    def center(self) -> torch.Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> float:
        raise NotImplementedError

    # -- helpers -------------------------------------------------------------
    def tensors(self) -> list[torch.Tensor]:
        return list(self.params.values())

    def copy(self) -> "Primitive":
        out = type(self).__new__(type(self))
        out.params = {k: v.detach().clone() for k, v in self.params.items()}
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, **{k: [round(float(x), 5) for x in v.detach().reshape(-1)] for k, v in self.params.items()}}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.describe()})"


class Sphere(Primitive):
    kind = "sphere"

    def __init__(self, c, r):
        super().__init__(c=c, r=r)

    def sdf(self, X):
        return (X - self.params["c"]).norm(dim=-1) - self.params["r"]

    def project_(self, r_min):
        with torch.no_grad():
            self.params["r"].clamp_(min=r_min)

    @property
    def center(self):
        return self.params["c"]

    def bounding_radius(self):
        return float(self.params["r"])


class Box(Primitive):
    """Oriented rounded box: center c, half-extents h (outer), axis-angle w, edge rounding k."""

    kind = "box"

    def __init__(self, c, h, w=(0.0, 0.0, 0.0), k=0.0):
        super().__init__(c=c, h=h, w=w, k=k)

    def local(self, X):
        R = rotation_matrix(self.params["w"])
        return (X - self.params["c"]) @ R  # rows: R^T (x - c)

    def sdf(self, X):
        h, k = self.params["h"], self.params["k"]
        q = self.local(X).abs() - (h - k)
        outside = q.clamp(min=0).norm(dim=-1)
        inside = q.max(dim=-1).values.clamp(max=0)
        return outside + inside - k

    def project_(self, r_min):
        with torch.no_grad():
            self.params["h"].clamp_(min=r_min)
            self.params["k"].clamp_(min=0.0)
            self.params["k"].clamp_(max=0.9 * float(self.params["h"].min()))

    @property
    def center(self):
        return self.params["c"]

    def bounding_radius(self):
        return float(self.params["h"].norm())

    def axes(self) -> torch.Tensor:
        """Columns: the box axes in world coordinates."""
        return rotation_matrix(self.params["w"].detach())


class Capsule(Primitive):
    """Segment a-b swept by a sphere of radius r (bridges, channels, plugs)."""

    kind = "capsule"

    def __init__(self, a, b, r):
        super().__init__(a=a, b=b, r=r)

    def sdf(self, X):
        a, b, r = self.params["a"], self.params["b"], self.params["r"]
        ab = b - a
        t = (((X - a) @ ab) / (ab @ ab).clamp(min=1e-18)).clamp(0.0, 1.0)
        return (X - (a + t[:, None] * ab)).norm(dim=-1) - r

    def project_(self, r_min):
        with torch.no_grad():
            self.params["r"].clamp_(min=r_min)

    @property
    def center(self):
        return 0.5 * (self.params["a"] + self.params["b"])

    def bounding_radius(self):
        return float(0.5 * (self.params["b"] - self.params["a"]).norm() + self.params["r"])


def box_from_axes(c, axes: torch.Tensor, h, rounding: float = 0.3) -> Box:
    """Box with given center, orthonormal axes (columns) and half-extents; rounding as a fraction of min(h)."""
    R = torch.as_tensor(axes, dtype=DTYPE)
    if torch.det(R) < 0:
        R = R.clone()
        R[:, 2] = -R[:, 2]
    w = axis_angle(R)
    h = as_t(h)
    return Box(c, h, w, rounding * float(h.min()))


def axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector of a rotation matrix (inverse of ``rotation_matrix``)."""
    R = torch.as_tensor(R, dtype=DTYPE)
    cos = ((torch.trace(R) - 1.0) / 2.0).clamp(-1.0, 1.0)
    th = float(torch.acos(cos))
    if th < 1e-9:
        return torch.zeros(3, dtype=DTYPE)
    if math.pi - th < 1e-6:
        # 180 degrees: axis from the diagonal of (R + I) / 2
        M = (R + torch.eye(3, dtype=DTYPE)) / 2.0
        i = int(torch.argmax(torch.diagonal(M)))
        v = M[:, i] / torch.sqrt(M[i, i].clamp(min=1e-18))
        return v / v.norm() * th
    v = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return v * th
