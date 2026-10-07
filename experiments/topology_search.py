"""Residual-guided topology search on CSG solids: end-to-end cases and held-out benchmark.

    python experiments/topology_search.py cases                 # 7 deterministic cases + viewer
    python experiments/topology_search.py bench --split test    # held-out benchmark (seeds 100-102)
    python experiments/topology_search.py bench --split tune    # development seeds (0-1)

Methods (identical continuous budget on the committed path; all CPU, float64):
  fixed   one moment-matched box (from the target occupancy only), continuous fitting only,
          FIXED_STEPS steps.
  search  same initialization, residual-guided topology search (SearchConfig defaults:
          <= 6 rounds, K = 60 steps per trial, 150 final steps); its candidate trials
          cost extra steps, reported as ``candidate_steps``.
  oracle  ORACLE: the target's own construction (correct structure) with perturbed
          parameters, FIXED_STEPS steps. Not available to the search; an upper reference.
          For the torus family (not exactly representable) the oracle structure is an
          approximation (flat rounded box minus a central channel).

Evaluation (independent of the 32^3 optimization grid): IoU and occupancy error at 64^3
(offset grid), two-sided surface error from marching-tetrahedra meshes at 64^3, topology at
56^3 and 80^3 (``stable`` when both agree and the mesh/voxel measurements are consistent).
Outputs: <out>/results.jsonl, report.md, meshes/*.obj, viewer.html.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cad_d4d.csg import Grid, SearchConfig, TopoObjective, TopologySearch, fit, measure_multi  # noqa: E402
from cad_d4d.csg.mesh import export_obj, marching_tetrahedra, node_lattice  # noqa: E402
from cad_d4d.csg.targets import CASES, FAMILIES, family_target, moment_box, oracle_init  # noqa: E402
from cad_d4d.csg.viewer import run_record, write_html  # noqa: E402

OPT_N = 32
SEEDS = {"tune": (0, 1), "test": (100, 101, 102)}


def committed_budget(cfg: SearchConfig) -> int:
    return cfg.initial_steps + cfg.rounds * cfg.trial_steps + cfg.final_steps


def surface_errors(recon_sdf, target_sdf, n: int = 64) -> dict:
    P = node_lattice(-1.2, 1.2, n)
    Pt = torch.as_tensor(P, dtype=torch.float64)
    out = {}
    with torch.no_grad():
        fr, ft = recon_sdf(Pt).numpy(), target_sdf(Pt).numpy()
    Vr, _, _ = marching_tetrahedra(fr, P, n + 1)
    Vt, _, _ = marching_tetrahedra(ft, P, n + 1)
    with torch.no_grad():
        a = np.abs(target_sdf(torch.as_tensor(Vr, dtype=torch.float64)).numpy()) if len(Vr) else np.array([np.inf])
        b = np.abs(recon_sdf(torch.as_tensor(Vt, dtype=torch.float64)).numpy()) if len(Vt) else np.array([np.inf])
    out["surface_mean"] = float(0.5 * (a.mean() + b.mean()))
    out["surface_max"] = float(max(a.max(), b.max()))
    return out


def evaluate(solid, target, eval_n: int = 64) -> dict:
    g = Grid(eval_n, offset=0.21)
    with torch.no_grad():
        r, t = solid.sdf(g.points) < 0, target.sdf(g.points) < 0
    iou = float((r & t).sum()) / max(float((r | t).sum()), 1.0)
    occ_err = float((r ^ t).double().mean())
    m = measure_multi(solid.sdf, ns=(56, 80))
    tm = measure_multi(target.sdf, ns=(56, 80))
    keys = ("components", "cavities", "tunnels", "genus")
    return {"iou": iou, "occupancy_error": occ_err, **surface_errors(solid.sdf, target.sdf),
            "topology": {k: m[k] for k in keys}, "topology_stable": m["stable"],
            "target_topology": {k: tm[k] for k in keys}, "target_stable": tm["stable"],
            "topology_match": bool(m["stable"] and all(m[k] == tm[k] for k in keys)),
            "complexity": len(solid.features)}


def run_one(method: str, target, seed: int, cfg: SearchConfig):
    obj = TopoObjective(target, Grid(OPT_N))
    t0 = time.perf_counter()
    res = None
    if method == "search":
        res = TopologySearch(obj, cfg).run(moment_box(target, obj.grid))
        solid = res.solid
    else:
        solid = moment_box(target, obj.grid) if method == "fixed" else oracle_init(target, seed)
        fit(solid, obj, committed_budget(cfg), cfg.fit)
    wall = time.perf_counter() - t0
    row = {"target": target.name, "family": target.family, "method": method, "seed": seed, "wall_s": wall,
           "objective": obj.value(solid)["total"], **evaluate(solid, target)}
    if res is not None:
        st = [t["status"] for t in res.trials]
        row.update(accepted=st.count("accepted"), rejected=st.count("rejected"), failed=st.count("failed"),
                   protected=st.count("protected"), edits=[e["op"] for e in res.events],
                   candidate_steps=res.costs["candidate_steps"], committed_steps=res.costs["committed_steps"],
                   baseline_steps=res.costs["baseline_steps"], proposal_s=res.costs["proposal_s"],
                   rounds=len(res.rounds), canonicalized=res.canonicalized)
    else:
        row.update(accepted=0, rejected=0, failed=0, protected=0, edits=[], candidate_steps=0,
                   committed_steps=committed_budget(cfg))
    return row, solid, res


def fmt_topo(t: dict) -> str:
    g = "?" if t["genus"] is None else t["genus"]
    return f"{t['components']}/{t['cavities']}/{t['tunnels']}/g{g}"


def report(rows: list[dict], title: str) -> str:
    lines = [f"# {title}", "",
             "Topology as components/cavities/tunnels/genus (genus '?' = mesh not valid or measurements inconsistent).",
             "", "| target | method | IoU | occ. err | surface mean / max | topology (target) | match | features | edits "
             "(acc/rej/fail) | cand. steps | wall s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['target']} | {r['method']} | {r['iou']:.3f} | {r['occupancy_error']:.4f} | "
                     f"{r['surface_mean']:.4f} / {r['surface_max']:.3f} | {fmt_topo(r['topology'])} "
                     f"({fmt_topo(r['target_topology'])}) | {'yes' if r['topology_match'] else 'NO'} | "
                     f"{r['complexity']} | {', '.join(r['edits']) or '-'} ({r['accepted']}/{r['rejected']}/{r['failed']}) | "
                     f"{r['candidate_steps']} | {r['wall_s']:.1f} |")
    lines += ["", "## Summary by method", "", "| method | runs | topology match | median IoU | median surface mean | "
              "median wall s | median candidate steps |", "|---|---|---|---|---|---|---|"]
    for m in ("fixed", "search", "oracle"):
        rs = [r for r in rows if r["method"] == m]
        if rs:
            lines.append(f"| {m} | {len(rs)} | {sum(r['topology_match'] for r in rs)}/{len(rs)} | "
                         f"{np.median([r['iou'] for r in rs]):.3f} | {np.median([r['surface_mean'] for r in rs]):.4f} | "
                         f"{np.median([r['wall_s'] for r in rs]):.1f} | {np.median([r['candidate_steps'] for r in rs]):.0f} |")
    fams = sorted({r["family"] for r in rows})
    lines += ["", "## Topology match by family (search / fixed / oracle)", "", "| family | search | fixed | oracle |",
              "|---|---|---|---|"]
    for f in fams:
        cell = []
        for m in ("search", "fixed", "oracle"):
            rs = [r for r in rows if r["family"] == f and r["method"] == m]
            cell.append(f"{sum(r['topology_match'] for r in rs)}/{len(rs)}")
        lines.append(f"| {f} | " + " | ".join(cell) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["cases", "bench"])
    ap.add_argument("--split", default="test", choices=list(SEEDS))
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--methods", nargs="*", default=["fixed", "search", "oracle"])
    ap.add_argument("--out", default=str(ROOT / "experiments" / "topology_out"))
    args = ap.parse_args()
    torch.manual_seed(0)
    out = Path(args.out) / (args.mode if args.mode == "cases" else args.split)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    cfg = SearchConfig()
    rows, viewer_runs = [], []
    if args.mode == "cases":
        jobs = [(name, None) for name in CASES]
    else:
        jobs = [(fam, seed) for fam in args.families for seed in SEEDS[args.split]]
    print(f"budget: committed {committed_budget(cfg)} steps, K = {cfg.trial_steps}, rounds <= {cfg.rounds}", flush=True)
    for name, seed in jobs:
        if args.mode == "cases":
            target, init, _ = CASES[name]()
            methods = [("search", init)]
        else:
            target = family_target(name, seed)
            methods = [(m, None) for m in args.methods]
        for method, init in methods:
            if init is not None:   # explicit initial state of an end-to-end case
                obj = TopoObjective(target, Grid(OPT_N))
                t0 = time.perf_counter()
                res = TopologySearch(obj, cfg).run(init)
                solid = res.solid
                st = [t["status"] for t in res.trials]
                row = {"target": name, "family": name, "method": "search", "seed": 0,
                       "wall_s": time.perf_counter() - t0, "objective": obj.value(solid)["total"],
                       **evaluate(solid, target), "initial": evaluate(init, target),
                       "accepted": st.count("accepted"), "rejected": st.count("rejected"),
                       "failed": st.count("failed"), "protected": st.count("protected"),
                       "edits": [e["op"] for e in res.events], "candidate_steps": res.costs["candidate_steps"],
                       "committed_steps": res.costs["committed_steps"], "canonicalized": res.canonicalized}
            else:
                row, solid, res = run_one(method, target, seed, cfg)
            rows.append(row)
            print(f"{row['target']:16s} {method:7s} IoU {row['iou']:.3f} surf {row['surface_mean']:.4f} "
                  f"topo {fmt_topo(row['topology'])} target {fmt_topo(row['target_topology'])} "
                  f"match {row['topology_match']} edits {row['edits']} ({row['wall_s']:.1f}s)", flush=True)
            P = node_lattice(-1.2, 1.2, 64)
            with torch.no_grad():
                f = solid.sdf(torch.as_tensor(P, dtype=torch.float64)).numpy()
            V, F, _ = marching_tetrahedra(f, P, 65)
            export_obj(out / "meshes" / f"{row['target']}_{method}.obj", V, F)
            if res is not None:
                metrics = {k: row[k] for k in ("iou", "occupancy_error", "surface_mean", "topology",
                                                "target_topology", "topology_match", "complexity", "wall_s")}
                viewer_runs.append(run_record(f"{row['target']} ({method})", target, res, metrics))
        with torch.no_grad():
            P = node_lattice(-1.2, 1.2, 64)
            V, F, _ = marching_tetrahedra(target.sdf(torch.as_tensor(P, dtype=torch.float64)).numpy(), P, 65)
        export_obj(out / "meshes" / f"{target.name if args.mode == 'bench' else name}_target.obj", V, F)
    with open(out / "results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=float) + "\n")
    title = "Topology search: end-to-end cases" if args.mode == "cases" else f"Topology search benchmark ({args.split} seeds)"
    (out / "report.md").write_text(report(rows, title), encoding="utf-8")
    write_html(out / "viewer.html", viewer_runs, title=title)
    print(f"wrote {out / 'report.md'}, {out / 'viewer.html'}, {len(rows)} runs")


if __name__ == "__main__":
    main()
