"""Topology measurements of solids on independent grids.

Voxel measurement (``voxel_topology``)
    Material voxels are treated as *closed* unit cubes, so two material voxels
    touching at a face, edge or corner are connected: material uses
    26-connectivity and, as its complementary convention, void uses
    6-connectivity (a void path may not squeeze through an edge or corner
    contact of material). With these conventions
        b0 = material components (26-conn.)
        b2 = enclosed cavities: void components (6-conn.) not reaching the
             (padded) grid boundary; the exterior void is excluded
        chi = V - E + F - C of the cubical complex (exact, by counting)
        b1 = b0 + b2 - chi  (tunnels/handles; = total genus of the boundary shells)
    Components smaller than ``min_voxels`` are counted separately as
    ``tiny_*`` and excluded from the main counts, so single-voxel sampling
    debris does not count as a body or a cavity.

Mesh measurement (``mesh_topology`` in mesh.py)
    Shells, Euler characteristic and genus of the marching-tetrahedra surface
    of the same SDF on a node lattice; genus is reported only for closed,
    manifold meshes.

``measure`` combines both and flags ``consistent`` when the mesh shells equal
b0 + b2 and the mesh genus equals b1. Inconsistent or invalid results are
reported as such (``genus = None`` / ``consistent = False``), never guessed.
All counts depend on resolution: features thinner than ~2 cells may be lost
or merged; compare two resolutions (``measure_multi``) before trusting them.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from .mesh import marching_tetrahedra, mesh_topology, node_lattice

S26 = np.ones((3, 3, 3), bool)
S6 = ndimage.generate_binary_structure(3, 1)


def _count_components(mask: np.ndarray, structure, min_voxels: int):
    lab, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return lab, np.zeros(0, int), n
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    return lab, sizes, n


def cubical_euler(occ: np.ndarray) -> int:
    """Euler characteristic of the union of closed unit cubes at the True voxels."""
    A = np.pad(np.asarray(occ, bool), 1)
    C = int(A.sum())
    F = sum(int(np.logical_or(A.take(range(0, A.shape[ax] - 1), ax), A.take(range(1, A.shape[ax]), ax)).sum())
            for ax in range(3))
    E = 0
    for ax in range(3):
        o = [a for a in range(3) if a != ax]
        B = np.moveaxis(A, (ax, o[0], o[1]), (0, 1, 2))
        E += int((B[:, :-1, :-1] | B[:, 1:, :-1] | B[:, :-1, 1:] | B[:, 1:, 1:]).sum())
    V = int((A[:-1, :-1, :-1] | A[1:, :-1, :-1] | A[:-1, 1:, :-1] | A[:-1, :-1, 1:] |
             A[1:, 1:, :-1] | A[1:, :-1, 1:] | A[:-1, 1:, 1:] | A[1:, 1:, 1:]).sum())
    return V - E + F - C


def voxel_topology(occ: np.ndarray, min_voxels: int = 8) -> dict:
    occ = np.pad(np.asarray(occ, bool), 1)              # the exterior always touches the padded border
    _, msizes, _ = _count_components(occ, S26, min_voxels)
    vlab, vsizes, nv = _count_components(~occ, S6, min_voxels)
    border = np.unique(np.concatenate([vlab[0].ravel(), vlab[-1].ravel(), vlab[:, 0].ravel(), vlab[:, -1].ravel(),
                                       vlab[:, :, 0].ravel(), vlab[:, :, -1].ravel()]))
    enclosed = np.array([i + 1 not in set(border.tolist()) for i in range(nv)], bool)
    cav_sizes = vsizes[enclosed]
    chi = cubical_euler(occ)
    b0 = int(np.sum(msizes >= min_voxels))
    b2 = int(np.sum(cav_sizes >= min_voxels))
    b0_all, b2_all = int(len(msizes)), int(len(cav_sizes))
    out = {"components": b0, "cavities": b2, "tiny_components": b0_all - b0, "tiny_cavities": b2_all - b2,
           "chi": chi, "material_voxels": int(occ.sum()),
           "component_voxels": sorted(msizes.tolist(), reverse=True),
           "cavity_voxels": sorted(cav_sizes.tolist(), reverse=True)}
    # b1 from the full counts (chi counts every piece, tiny ones included)
    out["tunnels"] = b0_all + b2_all - chi
    return out


def measure(sdf_fn, n: int = 64, lo: float = -1.2, hi: float = 1.2, min_voxels: int = 8,
            with_mesh: bool = True) -> dict:
    """Voxel (cell centers of an n^3 grid) and mesh (node lattice of the same grid) topology of ``sdf_fn``.

    ``sdf_fn`` maps an (N, 3) float64 tensor to signed distances (negative inside)."""
    h = (hi - lo) / n
    ax = lo + (np.arange(n) + 0.5) * h
    C = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    with torch.no_grad():
        dc = sdf_fn(torch.as_tensor(C, dtype=torch.float64)).cpu().numpy()
    vox = voxel_topology((dc < 0).reshape(n, n, n), min_voxels)
    out = {"n": n, "voxel": vox, "volume": float((dc < 0).sum() * h ** 3)}
    if with_mesh:
        P = node_lattice(lo, hi, n)
        with torch.no_grad():
            dn = sdf_fn(torch.as_tensor(P, dtype=torch.float64)).cpu().numpy()
        V, F, touches = marching_tetrahedra(dn, P, n + 1)
        mt = mesh_topology(F, len(V))
        mt["touches_boundary"] = touches
        out["mesh"] = mt
        out["_mesh"] = (V, F)
        valid = mt["genus_total"] is not None and not touches
        out["consistent"] = bool(valid and vox["tiny_components"] == 0 and vox["tiny_cavities"] == 0
                                 and mt["n_shells"] == vox["components"] + vox["cavities"]
                                 and mt["genus_total"] == vox["tunnels"])
        out["genus"] = mt["genus_total"] if out["consistent"] else None
    return out


def summary(m: dict) -> dict:
    """Compact topology record (no mesh arrays)."""
    v = m["voxel"]
    return {"components": v["components"], "cavities": v["cavities"], "tunnels": v["tunnels"],
            "genus": m.get("genus"), "consistent": m.get("consistent"),
            "shells": m.get("mesh", {}).get("n_shells"), "tiny_components": v["tiny_components"],
            "tiny_cavities": v["tiny_cavities"], "n": m["n"]}


def measure_multi(sdf_fn, ns=(64, 96), **kw) -> dict:
    """Measure at several resolutions; ``stable`` when the topology agrees across all of them."""
    ms = [summary(measure(sdf_fn, n=n, **kw)) for n in ns]
    keys = ("components", "cavities", "tunnels", "genus")
    stable = all(all(m[k] == ms[0][k] for k in keys) for m in ms) and all(m["consistent"] for m in ms)
    return {"per_resolution": ms, "stable": stable, **{k: ms[-1][k] for k in keys}}
