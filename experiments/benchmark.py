"""Benchmark: adaptive SRD vs. fixed uniform refinement on many targets.

    python experiments/benchmark.py --split tune            # development targets
    python experiments/benchmark.py --split test --report   # held-out targets + report

Results are cached in ``<out>/results.jsonl`` (resumable). For every target the
fixed uniform models k = 0..5 trace a fit-vs-control-points curve; each adaptive
run is compared against that curve at its own control-point count:

    efficiency = fit_adaptive / fit_uniform(n_cp_adaptive)     (< 1: adaptive better)
    cp_saving  = n_cp_uniform(fit_adaptive) / n_cp_adaptive    (> 1: adaptive smaller)

both by log-log interpolation along the uniform curve.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cad_d4d.benchmark.analysis import compare, uniform_curve  # noqa: E402
from cad_d4d.benchmark.runner import Budget, MethodSpec, ResultStore, run_method  # noqa: E402
from cad_d4d.benchmark.suite import catalog  # noqa: E402
from cad_d4d.device import get_device  # noqa: E402


def method_registry() -> dict[str, MethodSpec]:
    """All named methods; ``--methods`` selects from these (default: DEFAULT_METHODS)."""
    m = [MethodSpec(f"uniform_k{k}", "fixed", k=k) for k in range(6)]
    m += [MethodSpec(f"srd_C_s{s}", "srd", seed=s) for s in (0, 1, 2)]
    m += [MethodSpec(f"srd_C_lam{lam:g}", "srd", seed=0, lambda_complex=lam) for lam in (3e-8, 1e-7, 3e-7, 3e-6)]
    m += [MethodSpec(f"srd_C_lam{lam:g}_s{s}", "srd", seed=s, lambda_complex=lam) for lam in (1e-7, 3e-8)
          for s in (0, 1)]
    # throughput variant: up to 3 compatible refinements per round
    m += [MethodSpec(f"srd_C3_lam{lam:g}", "srd", seed=0, lambda_complex=lam, srd={"max_refine_per_round": 3})
          for lam in (3e-8, 1e-7, 3e-7, 1e-6)]
    # seam treatments: no boundary refinement (previous default) / crease penalty
    for lam in (3e-8, 1e-7, 3e-7):
        m.append(MethodSpec(f"srd_C_nobnd_lam{lam:g}", "srd", seed=0, lambda_complex=lam,
                            proposals={"p_refine_boundary": 0.0}))
        m.append(MethodSpec(f"srd_C_crease_lam{lam:g}", "srd", seed=0, lambda_complex=lam,
                            disc={"crease_weight": 1e4}))
    # ranking variant: refinements by predicted gain per added control point
    for lam in (1e-8, 3e-8, 1e-7, 3e-7):
        m.append(MethodSpec(f"srd_C_ratio_lam{lam:g}", "srd", seed=0, lambda_complex=lam, rank_by="ratio"))
        m.append(MethodSpec(f"srd_C_score_lam{lam:g}", "srd", seed=0, lambda_complex=lam, rank_by="score"))
    for w in (1e2, 1e3):
        for lam in (3e-8, 1e-7):
            m.append(MethodSpec(f"srd_C_crease{w:g}_lam{lam:g}", "srd", seed=0, lambda_complex=lam,
                                disc={"crease_weight": w}))
    return {x.name: x for x in m}


DEFAULT_METHODS = ([f"uniform_k{k}" for k in range(6)] + [f"srd_C_s{s}" for s in (0, 1, 2)]
                   + [f"srd_C_lam{lam:g}_s{s}" for lam in (1e-7, 3e-8) for s in (0, 1)])


def default_methods() -> list[MethodSpec]:
    reg = method_registry()
    return [reg[n] for n in DEFAULT_METHODS]


def report(store: ResultStore, targets: list[str], tag: str, out_dir: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cad_d4d.visualization.viewer import INK_2, SERIES, SURFACE

    lines = [f"# Benchmark report ({tag})", ""]
    eff_all, save_all = [], []
    beyond = 0
    n = len(targets)
    cols = 4
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4.2 * cols, 3.4 * ((n + cols - 1) // cols)),
                             squeeze=False)
    for i, t in enumerate(targets):
        rows = [r for r in store.rows if r["target"] == t and r["tag"] == tag]
        if not rows:
            continue
        cps, fits = uniform_curve(rows)
        lines += [f"## {t}", "", "| method | fit | control pts | faces | refine | simplify | efficiency | cp saving | runtime (s) |",
                  "|" + "---|" * 9]
        for r in sorted(rows, key=lambda r: (r["kind"], r["method"])):
            eff, save = compare(r, cps, fits) if r["kind"] == "srd" else (None, None)
            if r["kind"] == "srd" and r["method"].startswith("srd_C_s"):
                if eff is not None:
                    eff_all.append(eff)
                if save is not None:
                    save_all.append(save)
                if r["fit"] < fits.min():
                    beyond += 1
            fmt = lambda v: "-" if v is None else f"{v:.2f}"
            lines.append(f"| {r['method']} | {r['fit']:.2e} | {r['n_cp']} | {r['n_faces']} | {r['refinements']} | "
                         f"{r['simplifications']} | {fmt(eff)} | {fmt(save)} | {r['runtime_s']:.0f} |")
        lines.append("")
        ax = axes[i // cols][i % cols]
        ax.plot(cps, fits, "-o", color=INK_2, ms=4, lw=1.5, label="uniform k=0..5")
        for r in rows:
            if r["kind"] != "srd":
                continue
            seedrun = r["method"].startswith("srd_C_s")
            ax.scatter(r["n_cp"], r["fit"], s=40 if seedrun else 28, zorder=3, edgecolors=SURFACE, linewidths=1.5,
                       color=SERIES[0] if seedrun else SERIES[1], label=None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("control points", fontsize=8)
        ax.set_ylabel("fit", fontsize=8)
    for j in range(n, axes.size):
        axes.flat[j].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK_2, marker="o", ms=4, label="fixed uniform, k = 0..5"),
               Line2D([], [], color=SERIES[0], marker="o", ls="", label="adaptive C (seeds)"),
               Line2D([], [], color=SERIES[1], marker="o", ls="", label="adaptive C, lambda sweep")]
    fig.legend(handles=handles, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"pareto_{tag}.png", dpi=110)
    plt.close(fig)

    summary = ["## Summary (adaptive C seed runs vs. the uniform curve of the same target)", ""]
    if eff_all:
        summary.append(f"* efficiency (fit / uniform fit at same #cp): median {np.median(eff_all):.2f}, "
                       f"range {min(eff_all):.2f} - {max(eff_all):.2f}, better in {sum(e < 1 for e in eff_all)}"
                       f"/{len(eff_all)} runs")
    if save_all:
        summary.append(f"* cp saving (uniform #cp needed for the same fit / adaptive #cp): median "
                       f"{np.median(save_all):.2f}, range {min(save_all):.2f} - {max(save_all):.2f}")
    summary.append(f"* runs below the best uniform fit (k=5): {beyond}")
    summary.append("")
    text = "\n".join(lines[:2] + summary + lines[2:])
    (out_dir / f"report_{tag}.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tune", choices=["tune", "test", "all"])
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="subset of method names")
    ap.add_argument("--tag", default="v1", help="label separating benchmark versions in the cache")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--polish", type=int, default=200, help="final exact-SDF steps (all methods)")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "benchmark_out"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dashboard", action="store_true", help="write the interactive dashboard (viewer.html)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    store = ResultStore(out_dir / "results.jsonl")
    cat = catalog()
    names = args.targets or [n for n, t in cat.items() if args.split in ("all", t["split"])]
    reg = method_registry()
    methods = [reg[n] for n in args.methods] if args.methods else default_methods()
    budget = Budget(args.rounds, args.steps, args.polish)
    print(f"device {get_device()}; targets {names}; {len(methods)} methods; {budget.steps} steps", flush=True)
    for t in names:
        target = None
        for m in methods:
            key = ResultStore.key(t, m, budget, args.tag)
            if store.get(key):
                continue
            target = target or cat[t]["spec"].build()
            res = run_method(target, m, budget)
            res.pop("_state")
            res.update(target=t, tag=args.tag, key=key, split=cat[t]["split"], family=cat[t]["family"],
                       time=time.strftime("%Y-%m-%d %H:%M:%S"))
            store.add(res)
            print(f"{t:18s} {m.name:16s} fit {res['fit']:.3e} cp {res['n_cp']:4d} faces {res['n_faces']:3d} "
                  f"ref {res['refinements']:2d} simp {res['simplifications']:2d} ({res['runtime_s']:.0f}s)", flush=True)
    if args.report:
        print(report(store, names, args.tag, out_dir))
    if args.dashboard:
        from cad_d4d.visualization.web import benchmark_rows, write_html
        out = write_html(out_dir / "dashboard.html", [], benchmark_rows(store.path), title="D4D benchmark",
                         default_tag=args.tag)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
