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
Bridge/Pinch), plus the first bridge piece: a differentiable occupancy of the explicit solid
on a fixed grid. Out of scope for now: trimmed NURBS, rational weights, STEP import,
booleans, FEM, and topology surgery.

**Device.** All per-step work (proxy, losses, gradients, validity checks, occupancy) runs in
float64 on CUDA when available. Set `CAD_D4D_DEVICE=cpu` to force the CPU. Structure
(knots, sparse DOF maps, rewrites) stays in NumPy/SciPy, and per-structure operators are
uploaded once.

## Quick start

```bash
pip install -e .[test]            # numpy, scipy, torch, matplotlib, pyyaml, pytest
python -m pytest                  # 74 tests, ~50 s on GPU (CAD_D4D_DEVICE=cpu: ~2 min)
python experiments/adaptive_refinement.py          # full baseline, 3 SRD seeds (~10 min on GPU)
python experiments/adaptive_refinement.py --quick  # smoke run (~1.5 min on GPU)
```

Outputs go to `experiments/output/`: `metrics.md/json`, `history.png`,
`fit_vs_control_points.png`, per-method `state_*.png`, `exact_refinement_log.csv`,
`events_C.json` and `proposals_C.json`.

## Repository layout

```
configs/adaptive_refinement.yaml   experiment configuration
experiments/adaptive_refinement.py baseline experiment (6 methods)
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
  losses/       evaluate geometry
    target_sdf.py (SDF grid, trilinear), coverage.py, normals.py, fairness.py, complexity.py, objective.py
  optimization/ how to move through the continuous and discrete design space
    continuous.py (preconditioned GD, Armijo and validity backtracking), preconditioner.py (lumped mass),
    proposal_sampling.py, rewrite_scoring.py (modes A/B/C), srd.py, discretization.py
  occupancy/field.py     winding number, signed distance to the shell, soft occupancy on a fixed grid,
                         exact enclosed volume (all differentiable in the control points)
  targets/synthetic.py   grammar-generated targets (program discarded after freezing)
  device.py              compute device / dtype
  visualization/viewer.py
tests/                   one file per milestone
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

## Results

See `experiments/output/metrics.md` after running the experiment (summarized in RESULTS.md).
