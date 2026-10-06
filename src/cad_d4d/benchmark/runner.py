"""Running methods on benchmark targets.

Methods
-------
``fixed``  continuous optimization only, on a cube-sphere with ``k`` uniform
           interior knots per direction (faces and carriers); k=0 is the coarse model.
``srd``    adaptive SRD from the coarse model (mode C unless overridden).

All methods start from a cube-sphere whose radius matches the target volume and
get the same number of continuous steps.

Metric
------
``fit`` = area-weighted mean of the *exact* narrow-band target SDF^2 at dense
samples of the result + mean squared coverage distance of the target samples to
the dense result mesh. Independent of the optimizer's own quadrature.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..device import to_numpy, to_tensor
from ..geometry.builders import build_cube_complex, sphere_map
from ..geometry.state import CADState, watertightness_error
from ..geometry.tessellation import SamplingConfig, SurfaceSampler
from ..losses.complexity import ComplexityConfig
from ..losses.coverage import coverage_distances
from ..losses.objective import ObjectiveConfig, ShapeObjective
from ..optimization.continuous import ContinuousConfig, ContinuousOptimizer
from ..optimization.proposal_sampling import ProposalConfig
from ..optimization.rewrite_scoring import ScoringConfig
from ..optimization.srd import SRD, SRDConfig, polish_continuous


@dataclass
class Budget:
    rounds: int = 20
    steps_per_round: int = 40
    polish_steps: int = 0  # extra final steps against the exact SDF, identical for all methods

    @property
    def steps(self) -> int:
        return self.rounds * self.steps_per_round


@dataclass
class MethodSpec:
    name: str
    kind: str                      # "fixed" | "srd"
    k: int = 0                     # fixed: interior knots per direction
    seed: int = 0
    mode: str = "marginal_birth"
    lambda_complex: float | None = None
    srd: dict = field(default_factory=dict)        # SRDConfig overrides
    proposals: dict = field(default_factory=dict)  # ProposalConfig overrides
    disc: dict = field(default_factory=dict)       # DiscretizationConfig overrides (e.g. crease_weight)

    def key(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)


def exact_metrics(state: CADState, target) -> dict:
    sm = SurfaceSampler(state, SamplingConfig(min_res=33, per_span=8, max_res=65))
    X, Xu, Xv = sm.evaluate_np(state.values())
    area = np.linalg.norm(np.cross(Xu, Xv), axis=1) * to_numpy(sm.quad_w)
    w = area / area.sum()
    phi = to_numpy(target.sdf(X, exact=True))
    d = to_numpy(coverage_distances(target.points, to_tensor(X), sm.tri_t)[0])
    return {"fit": float(np.sum(w * phi**2) + np.mean(d**2)), "sdf_rms": float(np.sqrt(np.sum(w * phi**2))),
            "cov_rms": float(np.sqrt(np.mean(d**2))), "cov_max": float(d.max())}


def initial_radius(target) -> float:
    return float((3.0 * target.volume / (4.0 * math.pi)) ** (1.0 / 3.0))


def objective_for(target, method: MethodSpec, base_cfg: ObjectiveConfig | None = None) -> ShapeObjective:
    cfg = copy.deepcopy(base_cfg or ObjectiveConfig())
    if method.disc:
        cfg.discretization = dataclasses.replace(cfg.discretization, **method.disc)
    if method.lambda_complex is not None:
        cfg.complexity = dataclasses.replace(cfg.complexity, lambda_complex=method.lambda_complex,
                                             lambda_birth=0.1 * method.lambda_complex)
    return ShapeObjective(target, cfg)


def run_method(target, method: MethodSpec, budget: Budget, base_cfg: ObjectiveConfig | None = None) -> dict:
    obj = objective_for(target, method, base_cfg)
    r0 = initial_radius(target)
    t0 = time.perf_counter()
    events = []
    if method.kind == "fixed":
        state = build_cube_complex(sphere_map(r0), n_interior_knots=method.k)
        ContinuousOptimizer(obj, ContinuousConfig()).run(state, budget.steps)
        if budget.polish_steps:
            polish_continuous(obj, state, budget.polish_steps, ContinuousConfig())
    elif method.kind == "srd":
        state = build_cube_complex(sphere_map(r0))
        pcfg = dataclasses.replace(ProposalConfig(), **method.proposals)
        scfg = SRDConfig(rounds=budget.rounds, steps_per_round=budget.steps_per_round, seed=method.seed,
                         polish_steps=budget.polish_steps,
                         scoring=ScoringConfig(mode=method.mode), proposals=pcfg, **method.srd)
        res = SRD(obj, scfg).run(state)
        state, events = res.state, res.events
    else:
        raise ValueError(method.kind)
    runtime = time.perf_counter() - t0
    out = {"method": method.name, "kind": method.kind, "seed": method.seed, "runtime_s": runtime,
           "n_cp": state.n_control_points, "n_faces": state.n_faces,
           "refinements": sum(1 for e in events if e["refinement"]),
           "simplifications": sum(1 for e in events if not e["refinement"]),
           "watertight_err": watertightness_error(state)}
    out.update(exact_metrics(state, target))
    out["_state"] = state
    return out


class ResultStore:
    """Append-only JSONL cache keyed by (target, method spec, budget, tag)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if self.path.exists():
            self.rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    @staticmethod
    def key(target: str, method: MethodSpec, budget: Budget, tag: str) -> str:
        return json.dumps([target, method.key(), dataclasses.asdict(budget), tag])

    def get(self, key: str):
        for r in self.rows:
            if r.get("key") == key:
                return r
        return None

    def add(self, row: dict) -> None:
        self.rows.append(row)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
