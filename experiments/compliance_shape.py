"""Compliance-driven shape optimization of an explicit B-spline solid (FEM demo).

    python experiments/compliance_shape.py [--steps 150] [--srd-rounds 0]

Cantilever: a rounded beam is clamped at its left end and loaded downward at the
right tip. The objective is normalized compliance plus a volume penalty asking
for 80% of the initial volume:

    J = C(rho(p)) / C0 + lambda_vol ((V - 0.8 V0) / (0.8 V0))^2

C is computed by fixed-grid linear elasticity on the soft occupancy of the
B-spline solid, with adjoint gradients flowing to the control points. The result
is compared with the naive way to remove 20% of the volume (uniform scaling).
With --srd-rounds > 0, exact refinements are proposed and scored by the same
marginal-descent rule as in shape fitting (objective-agnostic).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cad_d4d.device import get_device, to_tensor  # noqa: E402
from cad_d4d.geometry.builders import build_cube_complex  # noqa: E402
from cad_d4d.losses.objective import ObjectiveConfig, ShapeObjective  # noqa: E402
from cad_d4d.occupancy import OccupancyGrid, enclosed_volume  # noqa: E402
from cad_d4d.optimization.continuous import ContinuousConfig, ContinuousOptimizer  # noqa: E402
from cad_d4d.optimization.proposal_sampling import ProposalConfig  # noqa: E402
from cad_d4d.optimization.srd import SRD, SRDConfig  # noqa: E402
from cad_d4d.physics import ComplianceTerm, FEMConfig, fem_on_grid  # noqa: E402

HALF = np.array([1.4, 0.45, 0.45])  # beam half-extents


def beam_map(n=4.0):
    def f(x):
        x = np.atleast_2d(x)
        d = x / np.linalg.norm(x, axis=1, keepdims=True)
        r = (np.abs(d) ** n).sum(1) ** (-1.0 / n)
        return HALF * (r[:, None] * d)
    return f


def setup(res: int):
    lo, hi = -HALF - 0.15, HALF + 0.15
    grid = OccupancyGrid(lo, hi, res=res, device=get_device())
    supports = lambda n: n[:, 0] < -HALF[0] + 0.12

    def loads(n):
        tip = (np.abs(n[:, 0] - HALF[0] + 0.1) < 0.1) & (np.abs(n[:, 1]) < 0.15) & (np.abs(n[:, 2]) < 0.15)
        F = np.zeros_like(n)
        F[tip, 2] = -1e-3 / max(1, tip.sum())
        return F

    fem = fem_on_grid(grid, supports, loads, FEMConfig(E_min=1e-3, cg_tol=1e-7))
    return grid, fem


def volume(state, obj):
    disc = obj.disc(state)
    X, _, _ = disc.sampler.evaluate(to_tensor(state.values()))
    return float(enclosed_volume(X, disc.sampler.tri_t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--res", type=int, default=30, help="FEM cells along the beam axis")
    ap.add_argument("--srd-rounds", type=int, default=0)
    args = ap.parse_args()
    print(f"device {get_device()}")
    grid, fem = setup(args.res)
    print(f"FEM grid {grid.dims} cells, {fem.n_dof} DOFs")
    state = build_cube_complex(beam_map(), n_interior_knots=1)

    term = ComplianceTerm(fem, grid, weight=1.0, eps=0.5 * grid.h)
    obj0 = ShapeObjective(None, ObjectiveConfig(lambda_sdf=0, lambda_coverage=0, lambda_fair=0, physics=[term]))
    V0 = volume(state, obj0)
    cfg = ObjectiveConfig(lambda_sdf=0, lambda_coverage=0, lambda_fair=1e-7, lambda_volume=20.0,
                          volume_target=0.8 * V0, physics=[term])
    obj = ShapeObjective(None, cfg)
    with torch.no_grad():
        obj.terms(state, to_tensor(state.values()))
    C0 = term.C0
    print(f"initial: volume {V0:.4f}, compliance {C0:.4e}")

    # baseline: uniform scaling to 80% volume
    scaled = state.copy()
    scaled.set_values(scaled.values() * 0.8 ** (1 / 3))
    with torch.no_grad():
        obj.terms(scaled, to_tensor(scaled.values()))
    C_scaled = term.last["compliance"]
    print(f"uniform scaling to 80% volume: compliance {C_scaled:.4e} ({C_scaled / C0:.3f} x C0)")

    t0 = time.perf_counter()
    opt = ContinuousOptimizer(obj, ContinuousConfig(eta=0.5))
    if args.srd_rounds > 0:
        res = SRD(obj, SRDConfig(rounds=args.srd_rounds, steps_per_round=max(1, args.steps // args.srd_rounds),
                                 proposals=ProposalConfig(mode="uniform", n_refine=6, n_simplify=4))).run(state)
        state = res.state
        print(f"SRD: {sum(e['refinement'] for e in res.events)} refinements, "
              f"{sum(not e['refinement'] for e in res.events)} simplifications")
    else:
        opt.run(state, args.steps)
    with torch.no_grad():
        obj.terms(state, to_tensor(state.values()))
    V1, C1 = volume(state, obj), term.last["compliance"]
    print(f"optimized ({time.perf_counter() - t0:.0f}s): volume {V1:.4f} ({V1 / V0:.3f} x V0), "
          f"compliance {C1:.4e} ({C1 / C0:.3f} x C0), faces {state.n_faces}, control points {state.n_control_points}")
    print(f"vs. uniform scaling at {0.8:.2f} V0: compliance ratio optimized / scaled = {C1 / C_scaled:.3f}")
    out = ROOT / "experiments" / "output"
    out.mkdir(parents=True, exist_ok=True)
    from cad_d4d.visualization.viewer import save_state_figure  # noqa: E402
    save_state_figure(state, None, str(out / "compliance_beam.png"), title="compliance-optimized beam", view=(20, -60))
    print(f"figure: {out / 'compliance_beam.png'}")


if __name__ == "__main__":
    main()
