"""Interactive viewer for topology-search runs (single self-contained HTML, three.js from a CDN).

Per run: target vs. reconstruction after every round (slider), material residuals
(missing material / excess material voxels of the shown state), and the edit log
(accepted, rejected, failed, protected candidates with their measured topology).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

from .mesh import marching_tetrahedra, node_lattice
from .solid import CSGSolid, Grid

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def mesh_of(sdf_fn, n: int = 30, lo: float = -1.2, hi: float = 1.2) -> dict:
    """Mesh as base64 typed arrays: vertices quantized to uint16 over [lo, hi], indices uint16/uint32."""
    P = node_lattice(lo, hi, n)
    with torch.no_grad():
        f = sdf_fn(torch.as_tensor(P, dtype=torch.float64)).cpu().numpy()
    V, F, _ = marching_tetrahedra(f, P, n + 1)
    q = np.round((V - lo) / (hi - lo) * 65535).clip(0, 65535).astype("<u2")
    idx = F.astype("<u2" if len(V) < 65536 else "<u4")
    return {"v": base64.b64encode(q.tobytes()).decode(), "f": base64.b64encode(idx.tobytes()).decode(),
            "i32": idx.dtype.itemsize == 4, "lo": lo, "hi": hi}


def residual_points(solid: CSGSolid, target_sdf, n: int = 32, max_points: int = 1500) -> dict:
    g = Grid(n)
    with torch.no_grad():
        cur = solid.sdf(g.points) < 0
        tgt = target_sdf(g.points) < 0
    P = g.points.numpy()
    out = {}
    for key, m in (("missing", (tgt & ~cur).numpy()), ("excess", (cur & ~tgt).numpy())):
        Q = P[m]
        if len(Q) > max_points:
            Q = Q[np.linspace(0, len(Q) - 1, max_points).astype(int)]
        out[key] = np.round(Q, 2).ravel().tolist()
    return out


def run_record(name: str, target, result, metrics: dict | None = None, mesh_n: int = 34) -> dict:
    """Viewer record of one search result (``result``: csg.search.SearchResult). Snapshots: the
    initial state, the state after each round that accepted an edit, and the final state."""
    snaps = []
    accepted = {e["round"] for e in result.events}
    for rnd, solid in result.snapshots:
        if rnd not in accepted and rnd != -1 and rnd != len(result.rounds):
            continue
        label = "initial" if rnd == -1 else ("final" if rnd == len(result.rounds) else f"after round {rnd}")
        snaps.append({"label": label, "mesh": mesh_of(solid.sdf, mesh_n), "residual": residual_points(solid, target.sdf),
                      "features": [{k: v for k, v in f.items() if k != "provenance"} | {"op": f["provenance"].get("op")}
                                   for f in solid.describe()]})
    trials = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in t.items()
               if k in ("round", "op", "status", "score", "reason", "rule", "before", "after", "feature", "split")}
              for t in result.trials]
    return {"name": name, "target": mesh_of(target.sdf, mesh_n), "snapshots": snaps, "events": result.events,
            "trials": trials, "metrics": metrics or {}, "canonicalized": result.canonicalized}


def write_html(path, runs: list[dict], title: str = "Topology search") -> Path:
    payload = json.dumps({"title": title, "runs": runs}, separators=(",", ":"), default=float).replace("</", "<\\/")
    path = Path(path)
    path.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__TOPO_DATA__", payload), encoding="utf-8")
    return path
