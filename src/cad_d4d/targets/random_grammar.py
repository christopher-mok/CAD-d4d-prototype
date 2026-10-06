"""Random targets generated from the grammar (reachable by construction).

Features (all built with exact rewrites, then a validity-checked displacement):

* ``interior``  LocalRefine at a random interior (u, v), then the inner control
                points of the refined child move along the face normal (bump or
                dent, C1 to the rest of the face).
* ``near_edge`` same, with the window touching a face boundary.
* ``edge``      CarrierKnotInsert (2 knots) on a random edge curve, then its new
                interior control points move outward/inward: a ridge or groove
                along part of a patch boundary (C0 there).
* ``corner``    a cube-corner vertex moves radially (affects 3 faces).

The generating program is discarded when the target is frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.builders import build_cube_complex, sphere_map
from ..geometry.state import CADState
from ..rewrites.carrier_knots import insert_carrier_knot_in_place
from ..rewrites.local_refine import LocalRefine
from .synthetic import Target, TargetConfig, is_valid_state


@dataclass
class GrammarSpec:
    name: str
    seed: int
    interior: int = 2
    near_edge: int = 0
    edge: int = 0
    corner: int = 0
    height: tuple = (0.12, 0.28)
    width: tuple = (0.25, 0.45)
    global_scale: tuple = (1.0, 1.0, 1.0)
    dents: bool = True
    cfg: TargetConfig = field(default_factory=TargetConfig)

    def build(self) -> Target:
        return random_grammar_target(self)


def _face_normal(state: CADState, fid: int, uv=(0.5, 0.5)) -> np.ndarray:
    su = state.evaluate(fid, np.array([uv]), (1, 0))[0]
    sv = state.evaluate(fid, np.array([uv]), (0, 1))[0]
    n = np.cross(su, sv)
    return n / np.linalg.norm(n)


def _signed_height(rng, spec) -> float:
    h = rng.uniform(*spec.height)
    return -h if (spec.dents and rng.random() < 0.4) else h


def random_grammar_state(spec: GrammarSpec, max_tries: int = 20) -> CADState:
    rng = np.random.default_rng(spec.seed)
    s = build_cube_complex(sphere_map())
    s.set_values(s.values() * np.asarray(spec.global_scale))
    root_faces = sorted(s.cx.faces)
    plan = (["interior"] * spec.interior + ["near_edge"] * spec.near_edge + ["edge"] * spec.edge
            + ["corner"] * spec.corner)
    used_roots: list[int] = []
    for kind in plan:
        for _ in range(max_tries):
            trial = s.copy()
            if kind in ("interior", "near_edge"):
                free = [r for r in root_faces if r not in used_roots] or root_faces
                root = int(rng.choice(free))
                if kind == "interior":
                    u, v = rng.uniform(0.3, 0.7, 2)
                else:
                    u, v = rng.uniform(0.3, 0.7, 2)
                    if rng.random() < 0.5:
                        u = rng.choice([0.12, 0.88])
                    else:
                        v = rng.choice([0.12, 0.88])
                fid = next(f.id for f in trial.cx.faces.values() if f.root == root)
                if trial.cx.faces[fid].shape != (4, 4):
                    continue  # one feature per untouched root face keeps windows simple
                out = LocalRefine(fid, u, v, width=rng.uniform(*spec.width), refine_knots=2).apply(trial)
                if not out.ok:
                    continue
                trial = out.state
                child = out.info["refined_face"]
                n = _face_normal(trial, child)
                P = trial.values()
                idx = trial.dof_map.face_dof[child][1:-1, 1:-1].ravel()
                P[idx] += _signed_height(rng, spec) * n
                trial.set_values(P)
                used_roots.append(root)
            elif kind == "edge":
                cid = int(rng.choice(sorted(trial.cx.carriers)))
                if len(trial.cx.carriers[cid].knots) != 8:
                    continue
                if any(insert_carrier_knot_in_place(trial, cid, t) for t in (1 / 3, 2 / 3)):
                    continue  # a non-empty reason means the insertion failed
                k = trial.dof_map.carrier_dof[cid]
                P = trial.values()
                mid = P[k].mean(0)
                P[k[1:-1]] += _signed_height(rng, spec) * 0.7 * mid / np.linalg.norm(mid)
                trial.set_values(P)
            elif kind == "corner":
                vids = [v for v in sorted(trial.cx.vertices) if trial.cx.vertices[v].free]
                vid = int(rng.choice(vids))
                k = trial.dof_map.vertex_dof[vid]
                P = trial.values()
                P[k] *= 1.0 + 0.6 * _signed_height(rng, spec)
                trial.set_values(P)
            if is_valid_state(trial):
                s = trial
                break
        else:
            raise RuntimeError(f"could not place feature {kind} for {spec.name}")
    s.cx.check_invariants()
    return s


def random_grammar_target(spec: GrammarSpec) -> Target:
    return Target.from_state(random_grammar_state(spec), spec.cfg)
