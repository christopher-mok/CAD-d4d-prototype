"""Recording SRD runs for the interactive replay viewer.

A ``Recorder`` is attached to ``SRD.run(..., recorder=...)``. It stores

* ``structures``  one entry per structure the run passes through: per-face
                  viewer grid sizes, root / refined flags, control-net shapes,
                  and the kind of every canonical DOF (vertex / carrier / face);
* ``frames``      every ``every`` continuous steps (and at every structure
                  change): surface samples, |phi| residual, canonical control
                  points, flattened control nets and scalar metrics;
* ``events``      every scored proposal of every discrete phase (kind,
                  location, scores, status) and every accepted rewrite.

Arrays are float32, base64-encoded. ``to_dict()`` is JSON-serializable and is
what ``visualization.web.build_html`` embeds into the viewer page.
"""
from __future__ import annotations

import base64

import numpy as np

from ..device import to_numpy
from ..geometry.state import CADState
from ..geometry.tessellation import SamplingConfig, SurfaceSampler


def b64(a: np.ndarray, dtype=np.float32) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


class Recorder:
    def __init__(self, target, every: int = 5, sampling: SamplingConfig | None = None, name: str = "run",
                 max_target_vertices: int = 120_000):
        self.target = target
        self.every = max(1, int(every))
        self.sampling = sampling or SamplingConfig(min_res=9, per_span=3, max_res=25)
        self.name = name
        self.structures: list[dict] = []
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self._struct_key = None
        self._sampler: SurfaceSampler | None = None
        self.max_target_vertices = max_target_vertices

    # -- hooks called by SRD -------------------------------------------------
    def structure(self, state: CADState) -> int:
        """Register the state's structure (deduplicated); returns its index."""
        dm = state.dof_map
        if dm is self._struct_key:  # strong reference: ids can be reused after garbage collection
            return len(self.structures) - 1
        self._struct_key = dm
        self._sampler = SurfaceSampler(state, self.sampling)
        sm, dm, cx = self._sampler, state.dof_map, state.cx
        faces = []
        for fid in sm.face_order:
            off, nu, nv = sm.face_slices[fid]
            f = cx.faces[fid]
            cu, cv = f.shape
            faces.append({"id": fid, "root": f.root, "refined": bool(f.lineage), "offset": off, "nu": nu, "nv": nv,
                          "net_u": cu, "net_v": cv, "domain": [float(x) for x in f.domain]})
        kinds = "".join(o[0] for o in dm.owners)  # 'v' vertex, 'c' carrier, 'f' face
        self.structures.append({"faces": faces, "dof_kinds": kinds, "n_samples": sm.n_samples,
                                "n_faces": state.n_faces, "n_cp": state.n_control_points})
        return len(self.structures) - 1

    def frame(self, state: CADState, P, step: int, round_: int, metrics: dict) -> None:
        P = np.asarray(to_numpy(P), dtype=float)
        s_idx = self.structure(state)
        X = self._sampler.G_np @ P
        phi = np.abs(to_numpy(self.target.sdf(X, exact=False)))
        nets = state.dof_map.E @ P
        self.frames.append({"step": int(step), "round": int(round_), "structure": s_idx, "X": b64(X),
                            "resid": b64(phi), "P": b64(P), "nets": b64(nets),
                            "metrics": {k: float(v) for k, v in metrics.items() if np.isscalar(v)}})

    def step_hook(self, state: CADState, step: int, round_: int):
        """Callback for ``ContinuousOptimizer.run``: records every ``every`` steps."""
        def cb(it, log, P):
            g = step + it + 1
            if g % self.every == 0:
                self.frame(state, P, g, round_, {"loss": log["loss"], "sdf": log["sdf"], "coverage": log["coverage"],
                                                 "fair": log["fair"], "eta": log.get("eta", 0.0)})
        return cb

    def discrete(self, step: int, round_: int, proposals: list[dict], events: list[dict]) -> None:
        keep = ("kind", "status", "accepted", "score", "D_old", "D_new", "B_refine", "predicted_improvement",
                "birth_cost", "immediate_gain", "delta_complexity", "deviation", "reason", "location", "face", "scoring")
        for p in proposals:
            self.events.append({"type": "proposal", "step": int(step), "round": int(round_),
                                **{k: _jsonable(p.get(k)) for k in keep if k in p}})
        for e in events:
            self.events.append({"type": "accepted", "step": int(step), "round": int(round_),
                                "kind": e["kind"], "rewrite": e["rewrite"], "score": float(e["score"]),
                                "delta_complexity": float(e["delta_complexity"])})

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict:
        V, T = self.target.V, self.target.T
        if len(V) > self.max_target_vertices:
            pts = to_numpy(self.target.points)
            target = {"points": b64(pts)}
        else:
            target = {"V": b64(V), "T": b64(T, np.int32)}
        return {"name": self.name, "every": self.every, "target": target, "structures": self.structures,
                "frames": self.frames, "events": self.events}


def _jsonable(v):
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return _jsonable(v.item())
    return str(v)
