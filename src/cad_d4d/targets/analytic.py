"""Analytic (out-of-grammar) star-shaped targets.

A shape is a radial function  x(d) = r(d) d  over unit directions d. It is
meshed densely on a cube-sphere layout (6 grids, outward orientation) and frozen
with ``Target.from_mesh``. These shapes are generally *not* exactly
representable by the patch grammar, unlike the synthetic grammar targets.

Building blocks (combine additively in r):
    gaussian bumps / dents     h * exp(-(1 - d.c) / s^2)     (angular width ~ s)
    superellipsoid             (|dx|^n + |dy|^n + |dz|^n)^(-1/n)  (n=2 sphere, n>2 boxy)
    ridge around an axis       h * exp(-((d.a) / s)^2)
    low-frequency blob         sum_j a_j cos(f_j . d + phi_j)
    anisotropic scaling        applied to the final points
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..geometry.builders import cube_quads
from .synthetic import Target, TargetConfig

RadialFn = Callable[[np.ndarray], np.ndarray]  # (N, 3) unit directions -> (N,) radii


def mesh_radial(r: RadialFn, res: int = 97, scale=(1.0, 1.0, 1.0)):
    """Dense mesh of x(d) = scale * r(d) d on the cube-sphere layout. Returns (V, T, N)."""
    corners, quads = cube_quads()
    scale = np.asarray(scale, float)
    t = np.linspace(0.0, 1.0, res)
    U, W = np.meshgrid(t, t, indexing="ij")
    h = 1e-5

    def surf(p):
        d = p / np.linalg.norm(p, axis=-1, keepdims=True)
        return scale * (r(d.reshape(-1, 3)).reshape(d.shape[:-1])[..., None] * d)

    Vs, Ts, Ns = [], [], []
    off = 0
    for a, b, c, dd in quads:
        pa, pb, pd = corners[a], corners[b], corners[dd]
        P = pa + U[..., None] * (pb - pa) + W[..., None] * (pd - pa)
        X = surf(P)
        Xu = (surf(P + h * (pb - pa)) - surf(P - h * (pb - pa))) / (2 * h)
        Xv = (surf(P + h * (pd - pa)) - surf(P - h * (pd - pa))) / (2 * h)
        n = np.cross(Xu, Xv)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        Vs.append(X.reshape(-1, 3))
        Ns.append(n.reshape(-1, 3))
        I, J = np.meshgrid(np.arange(res - 1), np.arange(res - 1), indexing="ij")
        I, J = I.ravel(), J.ravel()
        va = off + I * res + J
        vb = off + (I + 1) * res + J
        vc = off + (I + 1) * res + J + 1
        vd = off + I * res + J + 1
        Ts.append(np.concatenate([np.stack([va, vb, vc], 1), np.stack([va, vc, vd], 1)]))
        off += res * res
    return np.concatenate(Vs), np.concatenate(Ts), np.concatenate(Ns)


# ---- radial building blocks ---------------------------------------------------

def sphere(radius: float = 1.0) -> RadialFn:
    return lambda d: np.full(len(d), radius)


def superellipsoid(n: float = 4.0, radius: float = 1.0) -> RadialFn:
    return lambda d: radius * (np.abs(d) ** n).sum(1) ** (-1.0 / n)


def gaussian_bumps(centers, heights, widths) -> RadialFn:
    C = np.asarray(centers, float)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    H, S = np.asarray(heights, float), np.asarray(widths, float)
    return lambda d: (H[None] * np.exp(-(1.0 - d @ C.T) / S[None] ** 2)).sum(1)


def ridge(axis, height: float, width: float) -> RadialFn:
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    return lambda d: height * np.exp(-((d @ a) / width) ** 2)


def blob(rng: np.random.Generator, n_terms: int = 6, amplitude: float = 0.08, freq=(1.5, 3.0)) -> RadialFn:
    F = rng.normal(size=(n_terms, 3))
    F *= (rng.uniform(*freq, n_terms) / np.linalg.norm(F, axis=1))[:, None]
    A = amplitude * rng.uniform(0.5, 1.0, n_terms)
    phi = rng.uniform(0, 2 * np.pi, n_terms)
    return lambda d: (A[None] * np.cos(d @ F.T + phi[None])).sum(1)


def add(*fns: RadialFn) -> RadialFn:
    return lambda d: sum(f(d) for f in fns)


def random_bumps(rng: np.random.Generator, k: int, height=(0.08, 0.25), width=(0.18, 0.45),
                 dents: bool = True) -> RadialFn:
    C = rng.normal(size=(k, 3))
    H = rng.uniform(*height, k) * (np.where(rng.random(k) < 0.4, -1.0, 1.0) if dents else 1.0)
    S = rng.uniform(*width, k)
    return gaussian_bumps(C, H, S)


@dataclass
class AnalyticSpec:
    name: str
    r: RadialFn
    scale: tuple = (1.0, 1.0, 1.0)
    res: int = 97
    cfg: TargetConfig = field(default_factory=TargetConfig)

    def build(self) -> Target:
        V, T, N = mesh_radial(self.r, self.res, self.scale)
        return Target.from_mesh(V, T, N, self.cfg)
