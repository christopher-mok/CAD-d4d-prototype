"""Target solids and initial states for topology search.

Targets are SDFs. Most are built from CSG recipes (so a correct structure exists
for the labeled *oracle* baseline); ``torus`` is an exact torus SDF that no
finite set of these primitives represents exactly. The search never reads a
target's recipe, family, or parameters -- only its SDF sampled on the grid.

``CASES``: six deterministic end-to-end cases (tests and experiment).
``family_target(family, seed)``: parameterized families for the benchmark.
Seeds 0-99 are for development/tuning, seeds >= 100 are held out for reporting.
"""
from __future__ import annotations

import numpy as np
import torch

from .objective import Target
from .primitives import Box, Capsule, Sphere, box_from_axes, rotation_matrix
from .solid import MATERIAL, VOID, CSGSolid, Grid


def solid_of(*specs) -> CSGSolid:
    """CSGSolid from (role, kind, primitive) tuples."""
    s = CSGSolid()
    for role, kind, prim in specs:
        s.add(role, kind, prim, op="init")
    return s


def target_from_solid(name: str, build, family: str = "", params: dict | None = None) -> Target:
    ref = build()
    return Target(name, lambda X, ref=ref: ref.sdf(X), params, recipe=build, family=family)


def rbox(c, h, w=(0, 0, 0), k=0.12):
    return Box(c, h, w, k)


# ------------------------------------------------------------ end-to-end cases
def case_cavity():
    tgt = target_from_solid("cavity", lambda: solid_of(
        (MATERIAL, "body", rbox((0, 0, 0), (0.7, 0.6, 0.5))),
        (VOID, "cavity", Sphere((0.05, -0.03, 0.0), 0.28))), family="cavity")
    init = solid_of((MATERIAL, "body", rbox((0.02, 0, 0), (0.65, 0.65, 0.55), k=0.1)))
    return tgt, init, {"components": 1, "cavities": 1, "tunnels": 0}


def case_tunnel():
    tgt = target_from_solid("tunnel", lambda: solid_of(
        (MATERIAL, "body", rbox((0, 0, 0), (0.75, 0.7, 0.35))),
        (VOID, "channel", Capsule((0.05, 0.0, -1.0), (0.05, 0.0, 1.0), 0.26))), family="tunnel")
    init = solid_of((MATERIAL, "body", rbox((0, 0, 0), (0.7, 0.7, 0.4), k=0.1)))
    return tgt, init, {"components": 1, "cavities": 0, "tunnels": 1}


def case_two_bodies():
    tgt = target_from_solid("two_bodies", lambda: solid_of(
        (MATERIAL, "body", Sphere((-0.52, 0.0, 0.0), 0.36)),
        (MATERIAL, "body", Sphere((0.52, 0.05, 0.0), 0.34))), family="two_bodies")
    init = solid_of((MATERIAL, "body", rbox((0, 0, 0), (0.85, 0.36, 0.36), k=0.2)))
    return tgt, init, {"components": 2, "cavities": 0, "tunnels": 0}


def case_bridge():
    tgt = target_from_solid("bridge", lambda: solid_of(
        (MATERIAL, "body", Sphere((-0.55, 0.0, 0.0), 0.36)),
        (MATERIAL, "body", Sphere((0.55, 0.0, 0.0), 0.36)),
        (MATERIAL, "bridge", Capsule((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), 0.16))), family="bridge")
    init = solid_of((MATERIAL, "body", Sphere((-0.55, 0.0, 0.0), 0.34)),
                    (MATERIAL, "body", Sphere((0.55, 0.0, 0.0), 0.34)))
    return tgt, init, {"components": 1, "cavities": 0, "tunnels": 0}


def case_spurious_cavity():
    tgt = target_from_solid("solid", lambda: solid_of((MATERIAL, "body", rbox((0, 0, 0), (0.7, 0.6, 0.5)))),
                            family="solid")
    init = solid_of((MATERIAL, "body", rbox((0, 0, 0), (0.7, 0.6, 0.5))),
                    (VOID, "cavity", Sphere((0.0, 0.0, 0.0), 0.25)))
    return tgt, init, {"components": 1, "cavities": 0, "tunnels": 0}


def case_spurious_tunnel():
    tgt = target_from_solid("solid", lambda: solid_of((MATERIAL, "body", rbox((0, 0, 0), (0.7, 0.6, 0.4)))),
                            family="solid")
    init = solid_of((MATERIAL, "body", rbox((0, 0, 0), (0.7, 0.6, 0.4))),
                    (VOID, "channel", Capsule((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), 0.22)))
    return tgt, init, {"components": 1, "cavities": 0, "tunnels": 0}


def case_extra_body():
    tgt = target_from_solid("one_body", lambda: solid_of((MATERIAL, "body", Sphere((-0.2, 0.0, 0.0), 0.55))),
                            family="solid")
    init = solid_of((MATERIAL, "body", Sphere((-0.2, 0.0, 0.0), 0.5)),
                    (MATERIAL, "body", Sphere((0.75, 0.55, 0.3), 0.22)))
    return tgt, init, {"components": 1, "cavities": 0, "tunnels": 0}


CASES = {"cavity": case_cavity, "tunnel": case_tunnel, "two_bodies": case_two_bodies, "bridge": case_bridge,
         "spurious_cavity": case_spurious_cavity, "spurious_tunnel": case_spurious_tunnel,
         "extra_body": case_extra_body}


# ------------------------------------------------------------ benchmark families
FAMILIES = ("cavity", "tunnel", "two_bodies", "bridge", "torus", "solid")


def _rot(rng, scale=0.35):
    return rng.normal(0, scale, 3)


def family_target(family: str, seed: int) -> Target:
    rng = np.random.default_rng(seed)
    if family == "cavity":
        h = rng.uniform(0.5, 0.75, 3)
        w = _rot(rng)
        r = rng.uniform(0.18, 0.28)
        off = rng.uniform(-1, 1, 3) * np.maximum(h - r - 0.22, 0)
        def build(h=h, w=w, r=r, off=off):
            body = rbox((0, 0, 0), h, w)
            R = rotation_matrix(torch.as_tensor(w)).numpy()
            return solid_of((MATERIAL, "body", body), (VOID, "cavity", Sphere(R @ off, r)))
        params = {"h": h, "w": w, "r": r, "offset": off}
    elif family == "tunnel":
        h = np.array([rng.uniform(0.55, 0.8), rng.uniform(0.55, 0.8), rng.uniform(0.28, 0.42)])
        w = _rot(rng, 0.25)
        r = rng.uniform(0.18, 0.28)
        off = np.array([rng.uniform(-1, 1) * (h[0] - r - 0.25), rng.uniform(-1, 1) * (h[1] - r - 0.25), 0.0])
        def build(h=h, w=w, r=r, off=off):
            R = rotation_matrix(torch.as_tensor(w)).numpy()
            a, b = R @ (off + [0, 0, -1.0]), R @ (off + [0, 0, 1.0])
            return solid_of((MATERIAL, "body", rbox((0, 0, 0), h, w)), (VOID, "channel", Capsule(a, b, r)))
        params = {"h": h, "w": w, "r": r, "offset": off}
    elif family == "two_bodies":
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        ra, rb = rng.uniform(0.28, 0.4, 2)
        gap = rng.uniform(0.15, 0.3)
        sep = ra + rb + gap
        ca, cb = -d * sep * rb / (ra + rb), d * sep * ra / (ra + rb)
        def build(ca=ca, cb=cb, ra=ra, rb=rb):
            return solid_of((MATERIAL, "body", Sphere(ca, ra)), (MATERIAL, "body", Sphere(cb, rb)))
        params = {"ca": ca, "cb": cb, "ra": ra, "rb": rb}
    elif family == "bridge":
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        ra, rb = rng.uniform(0.28, 0.38, 2)
        gap = rng.uniform(0.3, 0.5)
        rr = rng.uniform(0.12, 0.18)
        sep = ra + rb + gap
        ca, cb = -d * sep / 2, d * sep / 2
        def build(ca=ca, cb=cb, ra=ra, rb=rb, rr=rr):
            return solid_of((MATERIAL, "body", Sphere(ca, ra)), (MATERIAL, "body", Sphere(cb, rb)),
                            (MATERIAL, "bridge", Capsule(ca, cb, rr)))
        params = {"ca": ca, "cb": cb, "ra": ra, "rb": rb, "r_bridge": rr}
    elif family == "torus":
        R0, r0 = rng.uniform(0.5, 0.65), rng.uniform(0.2, 0.27)
        w = _rot(rng, 0.5)
        Rm = rotation_matrix(torch.as_tensor(w))
        def sdf(X, R0=R0, r0=r0, Rm=Rm):
            Y = X @ Rm
            q = torch.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2) - R0
            return torch.sqrt(q ** 2 + Y[:, 2] ** 2) - r0
        def build(R0=R0, r0=r0, Rm=Rm):  # oracle approximation: flat box minus a central channel
            ax = Rm.numpy()
            return solid_of((MATERIAL, "body", box_from_axes((0, 0, 0), ax, [R0 + r0, R0 + r0, r0], rounding=0.9)),
                            (VOID, "channel", Capsule(ax @ np.array([0, 0, -1.0]), ax @ np.array([0, 0, 1.0]), R0 - r0)))
        return Target(f"torus_s{seed}", sdf, {"R": R0, "r": r0, "w": w}, recipe=build, family="torus")
    elif family == "solid":
        h = rng.uniform(0.45, 0.75, 3)
        w = _rot(rng)
        def build(h=h, w=w):
            return solid_of((MATERIAL, "body", rbox((0, 0, 0), h, w)))
        params = {"h": h, "w": w}
    else:
        raise ValueError(family)
    return target_from_solid(f"{family}_s{seed}", build, family=family,
                             params={k: np.round(np.asarray(v, float), 4).tolist() for k, v in params.items()})


def moment_box(target: Target, grid: Grid) -> CSGSolid:
    """Generic initialization: one rounded box matching the target occupancy's centroid and
    second moments (uses only the target occupancy on the grid, no structure)."""
    with torch.no_grad():
        occ = (target.sdf(grid.points) < 0).numpy()
    P = grid.points.numpy()[occ]
    c = P.mean(0)
    ev, evec = np.linalg.eigh(np.cov((P - c).T))
    half = np.sqrt(3.0 * np.maximum(ev, 1e-6))   # a solid box of half-extent a has variance a^2 / 3
    return solid_of((MATERIAL, "body", box_from_axes(c, evec, half, rounding=0.3)))


def oracle_init(target: Target, seed: int = 0, noise: float = 0.05) -> CSGSolid:
    """ORACLE: the target's own construction with perturbed parameters (correct structure supplied)."""
    s = target.recipe()
    rng = np.random.default_rng(10_000 + seed)
    for f in s.features:
        for k, v in f.prim.params.items():
            if k in ("c", "a", "b"):
                f.prim.params[k] = v + torch.as_tensor(rng.normal(0, noise, 3))
            elif k in ("r", "h"):
                f.prim.params[k] = v * float(rng.uniform(0.9, 1.1))
    return s
