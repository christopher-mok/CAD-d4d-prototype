"""Figures in docs/images.

    python experiments/make_figures.py targets                 # gallery of all benchmark targets
    python experiments/make_figures.py fits a_super_bumps:1 ... # target | uniform | SRD v3 | SRD v4
    python experiments/make_figures.py efficiency               # per-target efficiency (needs benchmark results)

fits runs SRD twice (v3 = lumped-metric scoring, v4 = default) and the uniform model with
the closest control-point count; GPU runs are not bit-reproducible, so figures vary by run.
efficiency reads experiments/benchmark_out/results.jsonl (tag v3) and
experiments/benchmark_out/v4_lumped/results.jsonl (tag v4_lumped).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cad_d4d.benchmark.analysis import annotate  # noqa: E402
from cad_d4d.benchmark.runner import Budget, MethodSpec, run_method  # noqa: E402
from cad_d4d.benchmark.suite import catalog  # noqa: E402
from cad_d4d.geometry.tessellation import SamplingConfig, SurfaceSampler  # noqa: E402
from cad_d4d.visualization.viewer import INK, INK_2, MUTED, SEQ, SERIES, _shade, boundary_curves, knot_lines  # noqa: E402

OUT = ROOT / "docs" / "images"


VIEW = (22, 35)


def view_dir(view):
    e, a = np.radians(view[0]), np.radians(view[1])
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def visible_segments(curves, center, view, margin=0.05):
    """Segments of polylines on the camera side of a star-shaped surface around ``center``."""
    v = view_dir(view)
    out = []
    for c in curves:
        r = c - center
        vis = (r / np.linalg.norm(r, axis=1, keepdims=True)) @ v > margin
        for a, b, ok in zip(c[:-1], c[1:], vis[:-1] & vis[1:]):
            if ok:
                out.append(np.stack([a, b]))
    return out


def setup(ax, pts, zoom=1.35):
    lo, hi = pts.min(0), pts.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    for set_lim, k in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        set_lim(c[k] - r, c[k] + r)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.set_axis_off()


def draw_state(ax, state, target, vmax, view=VIEW, title=""):
    sm = SurfaceSampler(state, SamplingConfig(min_res=25, per_span=6, max_res=49))
    X, _, _ = sm.evaluate_np(state.values())
    tris = X[sm.tri]
    phi = np.abs(target.sdf(torch.as_tensor(X, device=target.sdf.values.device), exact=True).cpu().numpy())
    colors = _shade(SEQ(np.clip(phi[sm.tri].mean(1) / vmax, 0, 1)), tris, view)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, edgecolors="none", zorder=1))
    center = X.mean(0)
    ax.add_collection3d(Line3DCollection(visible_segments(knot_lines(state, 40), center, view),
                                         colors="#3d3d3a", linewidths=0.45, alpha=0.75, zorder=2))
    ax.add_collection3d(Line3DCollection(visible_segments(boundary_curves(state, 40), center, view),
                                         colors=INK, linewidths=1.4, zorder=3))
    setup(ax, X)
    ax.view_init(*view)
    ax.set_title(title, fontsize=10, color=INK)


def draw_target(ax, target, view=VIEW, title=""):
    tris = target.V[target.T]
    base = np.tile(np.array([[0.80, 0.86, 0.95, 1.0]]), (len(tris), 1))
    ax.add_collection3d(Poly3DCollection(tris, facecolors=_shade(base, tris, view), edgecolors="none", zorder=1))
    gt = target.ground_truth
    if gt is not None:
        ax.add_collection3d(Line3DCollection(visible_segments(boundary_curves(gt, 40), target.V.mean(0), view),
                                             colors=INK, linewidths=1.0, zorder=3))
    setup(ax, target.V)
    ax.view_init(*view)
    ax.set_title(title, fontsize=10, color=INK)


def fig_targets():
    cat = catalog()
    names = list(cat)
    fig = plt.figure(figsize=(15, 9.6))
    for i, name in enumerate(names):
        ax = fig.add_subplot(3, 5, i + 1, projection="3d", computed_zorder=False)
        draw_target(ax, cat[name]["spec"].build(), view=(25, 35), title=f"{name}  ({cat[name]['split']})")
    fig.suptitle("Benchmark targets: g_* are reachable by the shape grammar, a_* are analytic (outside it)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "targets.png", dpi=90)


def fig_fits(pairs: list[str]):
    cat = catalog()
    CP = {0: 56, 1: 98, 2: 152, 3: 218, 4: 296, 5: 386}
    for pair in pairs:
        tname, seed = pair.split(":")
        tg = cat[tname]["spec"].build()
        new = run_method(tg, MethodSpec("srd", "srd", seed=int(seed)), Budget(polish_steps=200))
        old = run_method(tg, MethodSpec("srd_old", "srd", seed=int(seed), srd={"scoring_preconditioner": None}),
                         Budget(polish_steps=200))
        k = min(CP, key=lambda k: abs(np.log(CP[k] / new["n_cp"])))
        uni = run_method(tg, MethodSpec(f"u{k}", "fixed", k=k), Budget(polish_steps=200))
        vmax = 0.004
        fig = plt.figure(figsize=(19, 5.6))
        ax = fig.add_subplot(1, 4, 1, projection="3d", computed_zorder=False)
        draw_target(ax, tg, title=f"target: {tname}")
        for i, (r, lab) in enumerate(((uni, f"uniform refinement, k={k}"), (old, "adaptive SRD, lumped-metric scoring (v3)"),
                                      (new, "adaptive SRD, L2-metric scoring (v4)"))):
            ax = fig.add_subplot(1, 4, i + 2, projection="3d", computed_zorder=False)
            draw_state(ax, r["_state"], tg, vmax,
                       title=f"{lab}\n{r['n_cp']} control points, {r['n_faces']} faces, fit {r['fit']:.1e}")
        fig.subplots_adjust(left=0, right=0.94, wspace=0.0)
        cax = fig.add_axes([0.95, 0.2, 0.008, 0.55])
        cb = fig.colorbar(ScalarMappable(Normalize(0, vmax), SEQ), cax=cax, extend="max")
        cb.set_label("distance to target (exact SDF), shape radius ~1")
        fig.text(0.47, 0.03, "black: patch (face) boundaries   grey: knot lines inside faces   "
                 "fit = mean squared SDF + coverage distance", ha="center", fontsize=9, color="#52514e")
        fig.savefig(OUT / f"fit_{tname}.png", dpi=80)
        plt.close(fig)
        print(tname, f"new {new['n_cp']} {new['fit']:.2e}  old {old['n_cp']} {old['fit']:.2e}  uniform k{k} {uni['n_cp']} {uni['fit']:.2e}", flush=True)


def fig_efficiency():
    B = str(ROOT / "experiments" / "benchmark_out") + "/"
    rows = [json.loads(l) for l in open(B + "results.jsonl")] + [json.loads(l) for l in open(B + "v4_lumped/results.jsonl")]
    rows = annotate([r for r in rows if r["tag"] in ("v3", "v4_lumped")])
    GROUPS = [("v3 (lumped-metric scoring)", "v3", "srd_C_s", MUTED),
              ("v4 default (L2-metric scoring)", "v4_lumped", "srd_C_scorecm_s", SERIES[0]),
              ("v4 + ResidualRefine", "v4_lumped", "srd_C_rr_s", SERIES[1])]
    targets = sorted({r["target"] for r in rows if r["tag"] == "v4_lumped"}, key=lambda t: (t[0] != "g", t))
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    ax.axvline(1.0, color=INK_2, lw=1.2, zorder=1)
    ax.text(1.03, len(targets) - 0.35, "uniform refinement\n(same #control points)", fontsize=8, color=INK_2, va="top")
    meds = []
    for gi, (lab, tag, pre, col) in enumerate(GROUPS):
        ys, xs, allx = [], [], []
        for ti, t in enumerate(targets):
            for s in (0, 1):
                r = [r for r in rows if r["tag"] == tag and r["target"] == t and r["method"] == f"{pre}{s}"]
                if r and r[0].get("efficiency"):
                    xs.append(r[0]["efficiency"]); ys.append(len(targets) - 1 - ti + (1 - gi) * 0.22)
        allx = np.array(xs)
        meds.append((lab, np.median(allx), int((allx < 1).sum()), len(allx)))
        ax.scatter(xs, ys, s=34, color=col, edgecolors="#fcfcfb", linewidths=1.2, zorder=3,
                   label=f"{lab}: median {np.median(allx):.2f}, {int((allx < 1).sum())}/{len(allx)} runs < 1")
    ax.set_xscale("log")
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(list(reversed(targets)), fontsize=9)
    ax.set_ylim(-0.6, len(targets) - 0.2)
    ax.axhline(len(targets) - 4 - 0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("efficiency = adaptive fit / uniform fit at the same #control points   (left of 1: adaptive wins)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Adaptive SRD vs. uniform refinement, 13 held-out targets x 2 seeds", fontsize=11, color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.11), fontsize=8.5, frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "efficiency_v3_v4.png", dpi=100)
    for m in meds:
        print(m)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "targets"
    if cmd == "targets":
        fig_targets()
    elif cmd == "fits":
        fig_fits(sys.argv[2:] or ["a_super_bumps:1"])
    elif cmd == "efficiency":
        fig_efficiency()
    else:
        raise SystemExit(__doc__)
