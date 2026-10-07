# cad_d4d: Design for Descent over explicit B-spline CAD

A research prototype for **Stochastic Rewrite Descent (SRD)** over an explicit, watertight
complex of untrimmed bicubic B-spline patches. The design state is `x = (s, p)`, where `s` is
the discrete structure (faces, carriers/edges, vertices, knot vectors, split ancestry) and `p`
holds the canonical control-point positions. Continuous optimization moves `p` with `s` fixed.
Discrete rewrites change `s`, and many of them are **exact**: they add degrees of freedom
without changing the geometry.

```
coarse patch --LocalRefine/KnotInsert--> same surface, more DOFs --descent--> better fit
                                     \__ scored by the descent capability it unlocks
```

This is milestone 1 of a longer program (CSG -> fixed-grid FEM -> AddBody/AddCavity ->
Bridge/Pinch). Beyond shape fitting, the explicit solid now drives fixed-grid linear
elasticity: a differentiable occupancy feeds an ersatz-material FEM, and compliance gradients
flow back to the control points. Out of scope for now: trimmed NURBS, rational weights, STEP
import, booleans, and topology surgery (AddBody/AddCavity, Bridge/Pinch).

**Device.** All per-step work (proxy, losses, gradients, validity checks, occupancy) runs in
float64 on CUDA when available. Set `CAD_D4D_DEVICE=cpu` to force the CPU. Structure
(knots, sparse DOF maps, rewrites) stays in NumPy/SciPy, and per-structure operators are
uploaded once.

## Quick start

```bash
pip install -e .[test]            # numpy, scipy, torch, matplotlib, pyyaml, pytest
python -m pytest                  # 98 tests, ~50 s on GPU (CAD_D4D_DEVICE=cpu: ~3 min)
python experiments/adaptive_refinement.py [--record]   # single-target baseline, 3 SRD seeds (~10 min)
python experiments/benchmark.py --split test --tag v4 --report --dashboard   # 13-target benchmark
python experiments/make_figures.py targets|fits a_super_bumps:1|efficiency  # PNGs in docs/images
python experiments/compliance_shape.py                 # FEM: compliance-driven cantilever (~1 min)
python -m cad_d4d.visualization.web --recording run.json --benchmark experiments/benchmark_out/results.jsonl -o viewer.html
```

Outputs go to `experiments/output/` (baseline: `metrics.md/json`, plots, `exact_refinement_log.csv`,
`events_C.json`, and with `--record` an interactive `viewer.html`) and `experiments/benchmark_out/`
(`results.jsonl` cache, `report_<tag>.md`, `pareto_<tag>.png`, `dashboard.html`).

**Interactive viewer.** `viewer.html` is a single self-contained page (data embedded; three.js and
Plotly load from a CDN):
- *Run replay*: orbit the target and the fitted surface; toggle patch boundaries, control nets,
  control points (vertex / edge-curve / face), and the residual heatmap or refined faces. A
  timeline replays the run (play, step, jump to the next rewrite). The panel shows every
  proposal of the current discrete phase with its score and B_refine, the accepted rewrites,
  and loss / control-point curves.
- *Benchmark*: per-target Pareto charts (fit vs. control points) of adaptive runs against the
  uniform-refinement curve, summary statistics, and a sortable table of all runs.

## Repository layout

```
configs/adaptive_refinement.yaml   experiment configuration
experiments/adaptive_refinement.py baseline experiment (6 methods, single target)
experiments/benchmark.py           13 held-out + 2 tuning targets vs. uniform refinement (resumable)
experiments/compliance_shape.py    FEM demo: compliance-driven shape optimization
src/cad_d4d/
  geometry/     what the CAD object is
    bspline_basis.py  Cox-de Boor basis/derivatives, Boehm insertion, refinement and
                      segment-extraction matrices (all as explicit linear maps)
    patch.py          SplinePatch (degrees, knots, NURBS-weight placeholder), cached torch evaluation
    topology.py       Vertex, Carrier, Edge, EdgeUse, Face, PatchComplex, invariant I1, run matrices
    state.py          DofMap (canonical DOFs -> every face net), CADState, BirthRecord,
                      watertightness and root-parameter deviation checks
    tessellation.py   SurfaceSampler: X = G P, X_u = G_u P, X_v = G_v P (cached sparse proxy)
    validity.py       Jacobian/orientation checks and nonlocal self-intersection
    fitting.py, builders.py, distance.py
  rewrites/     transform one valid representation into another (no policy)
    knot_insert.py  KnotInsert (exact)          knot_remove.py  KnotRemove (epsilon-gated LS)
    carrier_knots.py CarrierKnotInsert (exact, refines a shared boundary curve) / CarrierKnotRemove
    split_face.py   SplitFace (exact)           merge_face.py   MergeFace (provenance + epsilon)
    split_edge.py   SplitEdge / MergeEdge (exact, topological)
    local_refine.py LocalRefine = exact splits isolating a window + refinement of that child only
                    (optionally also refining the window's carriers); FaceRefine = bisect all spans
  losses/       evaluate geometry
    target_sdf.py (SDF grid: trilinear + exact narrow band), coverage.py, normals.py, fairness.py,
    complexity.py, objective.py (shape terms + volume + pluggable physics terms)
  optimization/ how to move through the continuous and discrete design space
    continuous.py (preconditioned GD, Armijo and validity backtracking),
    preconditioner.py (lumped mass / semi-implicit with the fairness Hessian),
    proposal_sampling.py, rewrite_scoring.py (modes A/B/C), srd.py, discretization.py
  occupancy/field.py     winding number, signed distance to the shell, soft occupancy on a fixed grid,
                         exact enclosed volume (all differentiable in the control points)
  physics/fem.py         fixed-grid trilinear elasticity, matrix-free PCG, adjoint compliance
  physics/terms.py       ComplianceTerm: control points -> occupancy -> compliance
  targets/               synthetic.py (frozen targets), random_grammar.py (reachable multi-feature
                         targets), analytic.py (out-of-grammar radial shapes)
  benchmark/             suite.py (target catalog), runner.py (methods, exact metric, cache),
                         analysis.py (comparison against the uniform curve)
  device.py              compute device / dtype
  visualization/         viewer.py (matplotlib), recording.py (Recorder), web.py +
                         viewer_template.html (interactive replay + dashboard)
  csg/                   topology search prototype on CSG solids (separate representation):
    primitives.py (Sphere, oriented rounded Box, Capsule; exact SDFs), solid.py (canonical
    union-minus-union form, Grid), mesh.py (welded marching tetrahedra), measure.py (voxel +
    mesh topology), objective.py (volume/surface/complexity, Adam fit), residuals.py,
    grammar.py (8 rewrites + OpenCavity), proposals.py, search.py (matched trials),
    targets.py (cases, families, oracle), viewer.py + viewer_template.html
experiments/topology_search.py     topology search: 7 end-to-end cases, held-out benchmark
experiments/make_figures.py        PNGs in docs/images
tests/                   one file per milestone (test_csg_* for the topology prototype)
```

## Key design decisions

**Canonical DOFs and watertightness by construction.** The only continuous parameters are
(i) free vertex positions, (ii) the interior control points of *carriers* (the canonical
B-spline curve of each edge), and (iii) the interior control points of faces. A face's
boundary rows are never stored. They are derived from carriers through exact
knot-insertion/extraction matrices, so every face net is a sparse linear map
`net_f = E_f p`. **Invariant I1** requires that, along each run of a side that follows one
carrier, the face knots contain the carrier knots and have full multiplicity at the run's
ends. Under I1, both faces incident to an edge reproduce the *same* curve for **every** `p`.
The tests check this for random `p`, and after optimization. Gradients from both incident
faces accumulate on the shared DOFs automatically.

**Hanging vertices instead of mesh T-junctions.** A topological edge is an interval of a
carrier. SplitEdge inserts a vertex *constrained to the carrier* (its position is a basis
evaluation of the carrier, not a DOF). Both incident wires reference the two subedges, and the
neighbor is not refined. This is what makes LocalRefine genuinely local: neighbors keep their
knots and DOFs.

**Exact rewrites are linear maps.** KnotInsert is Boehm insertion. SplitFace inserts the
split knot to full multiplicity, extracts both child nets, and turns the shared row into a new
carrier. LocalRefine composes these. Geometry is unchanged to about 1e-14, checked on dense
root-parameter grids: every face records its rectangle in its root face's parameter square,
so states with different subdivisions can be compared point for point.

**Inexact simplifications are gated.** KnotRemove and MergeFace refit the face interior by
least squares with carrier-derived boundaries fixed (so watertightness is untouchable), then
accept only if the max deviation over a dense validation grid is below epsilon. Provenance
(same root, adjacent root domains, one shared full side) only nominates merge candidates.

**Preconditioned descent.** `p <- p - eta M^{-1} grad L`, where `M` is the lumped mass
`m_k = sum_s |G_sk| w_s` with area weights `w_s`. `M^{-1} grad L` is then a local average of
the pointwise loss gradient, so it is comparable between coarse and refined control points.
Each step uses Armijo backtracking plus validity backtracking (degenerate Jacobian,
orientation flip, nonlocal self-intersection). No Adam.

**Refinement-invariant fairness.** The default fairness is the thin-plate bending energy
`int |S_uu|^2 + 2|S_uv|^2 + |S_vv|^2` in root-face parameters, integrated with Gauss
quadrature per knot span. It depends only on geometry, so exact rewrites leave it unchanged.
A control-net second-difference fairness is also available (`fairness="control_net"`), but it
is *not* refinement invariant. With it, refining at coincident Greville abscissae produced
spurious descent capacity that corrupted marginal-descent scores.

**Exact-refinement scoring (modes).** `D = g^T M^{-1} g` is evaluated at the same geometry in
both parameterizations (the no-rewrite counterfactual: old state plus one step vs. refined state
plus one step), and `B_refine = D_new - D_old`.
- A, `naive`: immediate `F(x) - F(x')`. Exact refinements always score `-lambda_complex * dC`.
- B, `marginal`: `eta * B_refine`.
- C, `marginal_birth` (default): `eta * B_refine - lambda_birth * dC`, plus a grace period.
  Structure born from an accepted refinement cannot be simplified during `hold_steps`. Its
  effective complexity cost then ramps from `lambda_birth` to `lambda_complex` over
  `ramp_steps`. After that, unused structure is removed by KnotRemove/MergeFace.
  Simplifications charge removed complexity against the youngest affected birth record.

Inexact rewrites always use the immediate objective difference. Optional lookahead
(`lookahead_steps`) replaces shortlisted scores with gains realized against the counterfactual.

**Scoring metric vs. step metric.** Refinements are scored with `D` in the *consistent* L2
metric, `M = G^T W G` (`SRDConfig.scoring_preconditioner = "consistent_mass"`, the default),
while continuous steps keep the semi-implicit lumped metric. The lumped mass is the row-sum
lumping of `G^T W G`; for bicubic patches it overstates the mass of oscillatory control modes by
up to ~1000x (smallest eigenvalue of `diag(m)^{-1/2} M diag(m)^{-1/2}` is about 1e-3). Those are
exactly the modes an exact refinement adds, so the lumped `D_new - D_old` under-predicted useful
refinements: they lost to their birth cost, and SRD stalled on shapes with many spread
features. Consistent-mass *steps* (`ContinuousConfig.preconditioner = "consistent_mass"`) are
available too: they converge uniform fits ~3x faster but make steps less local and lost on the
grammar targets, so they are not the default.

**Boundary refinement.** Face KnotInsert only adds interior DOFs, since boundaries come from
carriers. CarrierKnotInsert refines the carrier itself: it first inserts the mapped knot into
every face side that follows the carrier, then into the carrier, exactly. CarrierKnotRemove is
the epsilon-gated inverse and measures deviation on all faces, because hanging vertices move
with their carrier.

**Partial merges and the split-line crease penalty.** MergeFace removes as many copies of the
split knot as epsilon allows (3, else 2 or 1). C1-joined children therefore still merge exactly,
leaving a knot that KnotRemove may take later. Optionally (`crease_weight`, default 0), the
fairness operator adds `crease_weight * int |dS_A/dN - dS_B/dN|^2` across split lines; it is zero
right after an exact split. In a 3-seed sweep it made runs more consistent and kept unused splits
mergeable (10x smaller merge deviation at 1e4), but it raised the median fit (3.6e-6 vs. 1.65e-6)
and control-point count (114 vs. 85), so it is opt-in.

**SDF for optimization vs. measurement.** Trilinear interpolation cannot represent the kinks of
a distance field (sharp target creases, medial axis), which gives an O(h) error. Each grid cell
within 2h of the surface stores candidate triangles. With `exact=True`, queries there use the
exact signed point-triangle distance on the device: about 1e-4 rms on the target surface,
independent of grid resolution and 13x better than trilinear at 64^3. All reported fits use this
exact measurement. As an *optimization objective*, however, the exact field is worse. At sharp
target creases it pulls near-boundary control rows into folds that validity checks must then
block: uniform k=3 reached 1.09e-5 when optimized against the exact field vs. 5.2e-6 when
optimized against trilinear, both measured exactly. The objective therefore uses the trilinear
field by default (`ObjectiveConfig.sdf_eval`).

**Semi-implicit preconditioner.** `d = (diag(m) + eta H)^{-1} g`, where `H = 2 lambda_fair F^T F`
is the constant Hessian of the fairness/crease form. It is factored densely (Cholesky) on the
device each step; the canonical DOF count is small. Only stiff directions are damped, and a
step of size eta stays stable on small refined faces, where bending is stiff (~h^-4). Without
this, the first GPU experiment stalled: about 60% of a run's steps failed backtracking. A
diagonal (Gershgorin) variant `lumped_mass_fair` is kept; it damps whole DOFs and slowed fitting
near split lines when the crease penalty was on. Scoring uses the same operator.

**Validity on the device.** Brute-force AABB broad phase (float32), exact narrow phase
(float64). Intersections between adjacent faces are discarded only when the hit point lies on
their common seam.

**Sliver prevention.** Split children must be at least `min_root_size` wide in root-face
units. Bending energy (and its preconditioned gradient) on thin slivers is stiff, which made
first-order marginal scores meaningless there.

**Refinement policy for complex shapes.** These additions came out of the multi-target
benchmark:
- *Boundary refinement*: LocalRefine can also refine the carriers that bound its window,
  removing the coarse seam around each window.
- *FaceRefine*: bisects every knot span of a face, carriers included.
- *Ratio ranking*: candidates are ranked by predicted gain per added complexity unit.
- *Exact-SDF polish*: a final continuous phase against the exact narrow-band SDF.
- *Residual-adaptive scale*: the first-order score is myopic. A local window has the highest
  immediate gain per DOF, so on a globally smooth misfit (a superellipsoid fitted from a
  sphere) SRD committed to windows early and plateaued about 10x above FaceRefine-only. It is
  path dependence, not an optimization pathology: an exact split applied to a converged model
  keeps improving. Each discrete phase therefore measures the *residual spread*, the smallest
  surface fraction holding half of the residual energy. If it exceeds 0.04, only global
  refinements are proposed (FaceRefine, CarrierKnotInsert). Concentrated residuals get local
  windows. The threshold is the geometric mean of the two tuning targets' spreads (0.013 vs.
  0.127); the 13 test targets split at 0.021 (grammar) / 0.054 (analytic).

**Robust closest-triangle search.** Coverage and occupancy distances minimize over candidate
triangles: the k nearest centroids plus the stars of the nearest proxy vertices, searched in
float64. Centroids alone miss large triangles next to small ones, and a float32 search resolves
near-ties by rounding noise. Together these made the loss discontinuous, which trapped the
line search at step sizes around 1e-13.

**Monotone validity.** A trial step is rejected if it *adds* self-intersections, not if any
exist. A rewrite that re-samples the check tessellation can reveal a pre-existing small fold;
under the absolute rule every later step inherited it and the optimizer froze for the rest of
the run. SRD also refuses rewrites whose new structure shows more intersections than the
parent. The continuous step size is reset after accepted rewrites.

**Contact freezing.** If self-intersections force a step to backtrack, a second line search runs
with the DOFs of the intersecting faces frozen (the step re-solved on the free DOFs), and the
better step is taken (`ContinuousConfig.freeze_contact`). Otherwise one face approaching a fold
shrinks the step of the whole model to ~1e-14.

**ResidualRefine (optional rule).** Inserts as many knots per direction as a face has spans, at
quantiles of the face's residual mass (with a 20% uniform floor), and refines the bounding
carriers there: FaceRefine's resolution increase, with spans placed by the residual instead of
bisected. Exact. Enable it via `ProposalConfig.kind_weights` / `global_kinds`.

**FEM (fixed-grid compliance).** `physics/fem.py` solves linear elasticity on a regular grid
of trilinear bricks, with SIMP-interpolated moduli `E_min + rho^p (E0 - E_min)` taken from the
soft occupancy of the B-spline solid. It is matrix-free with Jacobi-preconditioned CG on the
device, warm-started between evaluations. Compliance `C = f.u` has adjoint gradients
`dC/dE_e = -u_e^T K0 u_e` (self-adjoint, so no second solve) that chain through occupancy to
the control points. `ComplianceTerm` plugs into `ObjectiveConfig.physics`, so the continuous
optimizer, SRD scoring and the viewer work unchanged. Validation: a bar in tension matches
`sigma L / E` to machine precision, and compliance gradients match finite differences both with
respect to densities and with respect to control points. Note: on symmetric setups, grid
points on the shape's medial axis have non-differentiable distances (tied closest points).

## Tests (by milestone)

| file | invariant |
|---|---|
| test_bspline_basis | Cox-de Boor vs. recursive reference, Bernstein, partition of unity, derivatives vs. FD, exact insertion/extraction |
| test_patch | grid vs. pointwise eval, `dS/dP = N_a M_b` exactly, gradcheck, normals |
| test_topology | watertight for arbitrary `p`, shared DOFs get gradients from both faces, orientation checks |
| test_continuous | SDF vs. analytic sphere, coverage, monotone fitting on a reachable target, watertight after optimization, validity backtracking, self-intersection detection |
| test_knot_ops | KnotInsert exact, KnotRemove round trip, epsilon gating, I1-required knots refused |
| test_split_merge | SplitFace/SplitEdge/MergeEdge/MergeFace exact, merge epsilon gating, LocalRefine exact and local, full LocalRefine undo |
| test_fairness | bending energy invariant under exact rewrites |
| test_scoring | coarse stalls; naive rejects / marginal accepts useful refinement; useless refinement rejected; no credit for ordinary gradient progress; realized benefit vs. counterfactual; residual-guided proposals; grace protection |
| test_srd | SRD beats fixed coarse; naive never refines; an unused refinement is removed after the grace period |
| test_carrier_and_crease | CarrierKnotInsert exact (also with hanging vertices), round trip and gating; crease penalty zero after split, keeps unused splits mergeable |
| test_occupancy | winding number, exact volume and its gradient, signed distance, soft-occupancy volume/gradient consistency, invariance under exact rewrites, volume loss |
| test_fem | element stiffness (symmetric, 6 rigid modes), exact bar solution, compliance gradient vs. FD (densities and control points), compliance-driven shape optimization |
| test_benchmark | analytic/grammar target generators, comparison math |
| test_viewer | recorder frames/structures/events, safe HTML embedding |
| test_review_fixes | separable refit = dense LS, no double birth-record charge, DOF map reuse on copy, monotone self-intersection rule, contact freezing |
| test_csg_measure | SDF signs, union/difference, copy isolation, canonical form; components/cavities/tunnels/genus of known solids at two resolutions; cavity vs. exterior-connected void; voxel conventions, tiny components; welded mesh, triangle soup and non-manifold meshes report no genus; open surfaces |
| test_csg_grammar | every rewrite's effect and failing preconditions, inverses on original (not just recent) features, plug variants, split pinch, inputs never mutated, failures reported not substituted |
| test_csg_search | residual direction (missing vs. excess), cavity vs. channel evidence, resolution-independent objective, trial isolation and equal budgets, rejection keeps the optimized baseline, acceptance with provenance, inverse-edit protection |
| test_csg_e2e | 7 end-to-end reconstructions with topology at 2 resolutions, offset robustness (+-0.03), IoU, and a measured grammar edit |

## Topology search prototype (CSG)

The B-spline patch complex keeps its topology fixed: its grammar refines and simplifies
geometry, but cannot add a hole, a cavity or a second body. `cad_d4d.csg` is a first milestone
for **residual-guided topology search**, on a separate CSG/implicit representation. Its
results are **CSG reconstructions, not editable B-spline CAD**; conversion is future work (below).

```bash
python -m pytest tests/test_csg_measure.py tests/test_csg_grammar.py tests/test_csg_search.py tests/test_csg_e2e.py  # ~45 s, CPU
python experiments/topology_search.py cases               # 7 end-to-end cases (~30 s)
python experiments/topology_search.py bench --split test  # held-out benchmark (~3 min)
python experiments/topology_search.py bench --split tune  # development seeds
python experiments/make_figures.py topology               # docs/images/topology_cases.png
```

Outputs go to `experiments/topology_out/<cases|test|tune>/`: `results.jsonl`, `report.md`,
`meshes/*.obj` (target and every reconstruction), and `viewer.html`. The viewer shows target vs.
reconstruction after each accepted edit, missing/excess-material voxels, and every candidate
with its score or failure reason.

**Representation.** `S = (M_1 u ... u M_m) \ (V_1 u ... u V_v)`: material and void features, one
primitive each (sphere, oriented rounded box, capsule; trainable position, size, orientation).
`sdf < 0` inside; union = min, difference = `max(d, -d_void)`. Hard occupancy `sdf < 0`; soft
occupancy `sigmoid(-sdf / eps)`, eps = 0.03 in world units. The flat two-level form is the
canonical expression: edits add or remove features and never nest, and features that no
longer change the occupancy are dropped every round (logged). Limitation: no material inside a
void, so nested shells are not representable.

**Grammar.** Material side: AddBody, RemoveBody, BridgeBodies, PinchBody (slab or split
variant). Void side: AddCavity, RemoveCavity (remove the void feature, or plug a cavity formed
by material), BridgeVoid / OpenCavity, CloseTunnel (remove the channel feature, or plug a
handle formed by material). Each operation checks preconditions and edits a copy. It then
*measures* topology before and after on a check grid, and fails with the measured counts
unless its stated effect happened. Semantics and preconditions are in the `grammar.py`
docstring. Inverses target any existing feature or measured void, not only the latest edit.

**Proposals.** From `r_add = max(rho_t - rho, 0)` and `r_remove = max(rho - rho_t, 0)`: connected
regions (6-connected, residual > 0.5). For each: volume, centroid, PCA axes and half-extents,
inscribed ball, contact patches with material components, the exterior void and cavities,
whether removing it disconnects material, and overlap with existing features. Rules map this
evidence to edits at two scales (`proposals.py`), capped at 8 per round. Nothing reads target
genus, shell counts, labels or construction history.

**Scoring.** Matched trials. The baseline (no edit) and every candidate are optimized for the same
K = 60 Adam steps from independent copies; `score = F(baseline) - F(candidate)` with
`F = L_volume + 0.05 L_surface + 1e-3 * #features`. The best candidate is accepted only if
`score > 2e-4 + 0.02 F(baseline)`; its optimized state is committed, otherwise the optimized
baseline. Accepted features carry provenance; inverse edits near or on them are blocked for 2
rounds. `L_volume` is a cell mean and `L_surface` a band-weighted mean, so neither depends on
grid resolution (tested). This trial scoring is separate from the gradient-based marginal
scoring of exact B-spline refinements.

**Topology measurement.** Voxels: material 26-connected (closed cubes), void 6-connected.
Components; cavities = void components not reaching the padded border; Euler characteristic of
the cubical complex; tunnels `b1 = b0 + b2 - chi`. Components below 8 voxels are reported
separately as tiny. Meshes: marching tetrahedra on a Freudenthal split, welded by lattice edge;
closedness and vertex-manifoldness are checked before shells, chi and genus are reported. A
result counts as `stable` only if mesh and voxel measurements agree at two resolutions; otherwise
genus is reported as unknown, not guessed. Counts depend on resolution: features thinner than
~2 cells can vanish or merge.

**Toward B-spline CAD (future work).** A CSG result fixes the *topology* a B-spline model must
have: components, cavities and handles, plus where they are. A converter could
1. extract a quad-dominant surface layout per shell, for example from a cross field on the
   reconstruction's mesh, with the genus fixing the number of irregular vertices;
2. build an untrimmed patch complex on that layout;
3. fit it to the CSG SDF;
4. hand it to the existing SRD loop for adaptive refinement.

A tighter integration would run the topology trials directly on patch complexes, using
handle-attachment and shell-splitting rewrites that keep watertightness. None of this is
implemented.

## Results

`RESULTS.md` summarizes the single-target baseline, the 13-target held-out benchmark, the FEM
demo and the performance checks.
