"""Matplotlib debug views for patch complexes and SRD runs.

* ``plot_state``      target + current surface, patch boundaries, control nets,
                      canonical control points, residual heatmap, refined faces,
                      accepted / rejected rewrite proposals.
* ``plot_history``    loss terms, structure size and objective over steps, with
                      accepted rewrites marked.
* ``plot_comparison`` final fit vs. complexity of several runs.

Colors: categorical slots in fixed order (blue, orange, aqua, yellow); a
single-hue blue ramp for residual magnitude; neutral inks for structure.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

from ..geometry import bspline_basis as bb  # noqa: E402
from ..geometry.state import CADState  # noqa: E402
from ..geometry.tessellation import SamplingConfig, SurfaceSampler  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9a9893"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
REFINED = "#eb6834"  # categorical slot 2: faces born from refinements

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2.0,
    "legend.frameon": False,
})


def _shade(rgba: np.ndarray, tris: np.ndarray, view) -> np.ndarray:
    """Lambert shading toward the camera so curvature (e.g. a bump) is visible."""
    elev, azim = np.radians(view[0]), np.radians(view[1])
    light = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    k = 0.55 + 0.45 * np.clip(n @ light, 0, 1)
    out = rgba.copy()
    out[:, :3] *= k[:, None]
    return out


def view_toward(point) -> tuple[float, float]:
    """(elev, azim) of a camera looking at ``point`` from outside the origin."""
    p = np.asarray(point, float)
    r = np.linalg.norm(p)
    return float(np.degrees(np.arcsin(p[2] / r))) + 15.0, float(np.degrees(np.arctan2(p[1], p[0])))


def _set_equal(ax, pts):
    lo, hi = pts.min(0), pts.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def boundary_curves(state: CADState, n: int = 24) -> list[np.ndarray]:
    t = np.linspace(0, 1, n)
    out = []
    for f in state.cx.faces.values():
        for side in ("v0", "u1", "v1", "u0"):
            out.append(state.evaluate(f.id, f.side_uv(side, t)))
    return out


def knot_lines(state: CADState, n: int = 24) -> list[np.ndarray]:
    """Iso-parameter curves at every interior knot of every face (the structure's mesh lines)."""
    t = np.linspace(0, 1, n)
    out = []
    for f in state.cx.faces.values():
        for k in np.unique(np.round(bb.interior_knots(f.knots_u, f.degree_u), 12)):
            out.append(state.evaluate(f.id, np.stack([np.full(n, k), t], 1)))
        for k in np.unique(np.round(bb.interior_knots(f.knots_v, f.degree_v), 12)):
            out.append(state.evaluate(f.id, np.stack([t, np.full(n, k)], 1)))
    return out


def plot_state(ax, state: CADState, target=None, residual: bool = True, show_net: bool = True,
               show_points: bool = True, highlight_refined: bool = True, proposals=None,
               title: str | None = None, view=(20, 35), sampling: SamplingConfig | None = None,
               show_knots: bool = False, vmax: float | None = None):
    sm = SurfaceSampler(state, sampling or SamplingConfig(min_res=13, per_span=3, max_res=25))
    X, _, _ = sm.evaluate_np(state.values())
    tris = X[sm.tri]
    if residual and target is not None:
        phi = np.abs(target.sdf(torch.as_tensor(X, device=target.sdf.values.device)).cpu().numpy())
        vals = phi[sm.tri].mean(1)
        vmax = max(float(np.quantile(phi, 0.99)), 1e-6) if vmax is None else vmax
        colors = SEQ(np.clip(vals / vmax, 0, 1))
    else:
        colors = np.tile(np.array([[0.80, 0.86, 0.95, 1.0]]), (len(tris), 1))
    if highlight_refined:
        born = np.array([bool(state.cx.faces[sm.face_order[k]].lineage) for k in sm.tri_face])
        if born.any() and not residual:
            tint = 0.55 * np.array(matplotlib.colors.to_rgba(REFINED)) + 0.45 * np.array([0.80, 0.86, 0.95, 1.0])
            colors[born] = tint
    colors = _shade(np.asarray(colors, float), tris, view)
    poly = Poly3DCollection(tris, facecolors=colors, edgecolors="none", alpha=0.95)
    ax.add_collection3d(poly)
    lines = boundary_curves(state)
    ax.add_collection3d(Line3DCollection(lines, colors=INK, linewidths=0.9))
    if highlight_refined:
        refined_lines = []
        t = np.linspace(0, 1, 24)
        for f in state.cx.faces.values():
            if f.lineage:
                refined_lines += [state.evaluate(f.id, f.side_uv(s, t)) for s in ("v0", "u1", "v1", "u0")]
        if refined_lines:
            ax.add_collection3d(Line3DCollection(refined_lines, colors=REFINED, linewidths=1.6))
    if show_knots:
        ax.add_collection3d(Line3DCollection(knot_lines(state), colors=INK_2, linewidths=0.35, alpha=0.8))
    if show_net:
        segs = []
        for net in state.nets().values():
            for i in range(net.shape[0]):
                segs.append(net[i])
            for j in range(net.shape[1]):
                segs.append(net[:, j])
        ax.add_collection3d(Line3DCollection(segs, colors=MUTED, linewidths=0.4, alpha=0.7))
    if show_points:
        dm = state.dof_map
        P = dm.values
        for kind, color, label in (("v", SERIES[0], "vertex DOF"), ("c", SERIES[1], "carrier DOF"),
                                   ("f", SERIES[2], "face DOF")):
            idx = [k for k, o in enumerate(dm.owners) if o[0] == kind]
            if idx:
                ax.scatter(*P[idx].T, s=6, color=color, depthshade=False, label=label)
    if target is not None and not residual:
        sub = target.V[:: max(1, len(target.V) // 3000)]
        ax.scatter(*sub.T, s=0.5, color=MUTED, alpha=0.4, depthshade=False)
    if proposals:
        acc = np.array([p["location"] for p in proposals if p.get("location") and p.get("accepted")])
        rej = np.array([p["location"] for p in proposals if p.get("location") and not p.get("accepted")])
        if len(rej):
            ax.scatter(*rej.T, marker="x", s=18, color=INK_2, depthshade=False, label="rejected proposal")
        if len(acc):
            ax.scatter(*acc.T, marker="o", s=40, facecolors="none", edgecolors=INK, linewidths=1.5,
                       depthshade=False, label="accepted rewrite")
    _set_equal(ax, X)
    ax.view_init(*view)
    if title:
        ax.set_title(title, fontsize=9, color=INK)
    if residual and target is not None:
        ax._residual_vmax = vmax
    return ax


def plot_target(ax, target, title="target", view=(20, 35)):
    tris = target.V[target.T]
    base = np.tile(np.array([[0.80, 0.86, 0.95, 1.0]]), (len(tris), 1))
    ax.add_collection3d(Poly3DCollection(tris, facecolors=_shade(base, tris, view), edgecolors="none"))
    gt = target.ground_truth
    if gt is not None:
        ax.add_collection3d(Line3DCollection(boundary_curves(gt), colors=INK, linewidths=0.8))
    _set_equal(ax, target.V)
    ax.view_init(*view)
    ax.set_title(title, fontsize=9, color=INK)


def plot_history(runs: dict, path: str, events: dict | None = None):
    """runs: name -> history list (per step dicts). Small multiples sharing the step axis."""
    keys = [("fit", "fit loss (SDF + coverage)", True), ("fair", "fairness energy", False),
            ("n_cp", "control points", False), ("total", "total objective", True)]
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 10), sharex=True)
    for k, (key, label, logy) in enumerate(keys):
        ax = axes[k]
        for i, (name, hist) in enumerate(runs.items()):
            steps = [h["step"] for h in hist]
            if key == "fit":
                vals = [h["sdf"] + h["coverage"] for h in hist]
            else:
                vals = [h[key] for h in hist]
            # naive drawn dashed: it coincides with the coarse run when nothing is accepted
            ax.plot(steps, vals, color=SERIES[i], label=name, ls="--" if "naive" in name else "-")
        if events:
            for name, evs in events.items():
                for e in evs:
                    if e["refinement"]:
                        ax.axvline(e["step"], color=REFINED, lw=0.8, alpha=0.6)
                    else:
                        ax.axvline(e["step"], color=MUTED, lw=0.8, ls=":", alpha=0.8)
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(label)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("continuous step  (orange: accepted refinement, dotted: simplification; adaptive run)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_comparison(results: dict, path: str, highlight: str | None = None):
    """Final fit vs. canonical control points. Identity by direct labels; the highlighted
    method in categorical slot 1, everything else neutral. Coincident runs share one label."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    groups: dict[tuple, list[str]] = {}
    for name, m in results.items():
        groups.setdefault((m["n_cp"], round(float(np.log10(m["fit"])), 4)), []).append(name)
    for (n_cp, _), names in groups.items():
        m = results[names[0]]
        main = highlight in names
        ax.scatter(m["n_cp"], m["fit"], s=70 if main else 48, color=SERIES[0] if main else INK_2,
                   edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(" = ".join(names), (m["n_cp"], m["fit"]), xytext=(8, 3), textcoords="offset points",
                    fontsize=8, color=INK if main else INK_2, fontweight="bold" if main else "normal")
    ax.set_yscale("log")
    ax.set_xlabel("canonical control points")
    ax.set_ylabel("final fit loss (lower is better)")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_state_figure(state: CADState, target, path: str, proposals=None, title: str = "", view=(20, 35)):
    fig = plt.figure(figsize=(15, 5.4))
    fig.suptitle(title, fontsize=11, color=INK)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    if target is not None:
        plot_target(ax1, target, "target (ground-truth patch layout)", view=view)
    else:
        plot_state(ax1, state, None, residual=False, show_net=False, show_points=False,
                   highlight_refined=False, title="surface", view=view)
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    plot_state(ax2, state, target, residual=False, show_net=True, show_points=True, proposals=proposals,
               title="patches + control net (orange: refined faces)", view=view)
    ax2.legend(loc="lower left", fontsize=7)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    plot_state(ax3, state, target, residual=target is not None, show_net=False, show_points=False,
               title="residual |phi_target| on the surface" if target is not None else "refined faces", view=view)
    if target is not None:
        sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, getattr(ax3, "_residual_vmax", 1.0)))
        fig.colorbar(sm, ax=ax3, shrink=0.6, pad=0.02, label="|phi| (capped at 99th percentile)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
