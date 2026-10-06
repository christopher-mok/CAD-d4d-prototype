# Baseline experiment: adaptive exact refinement

`python experiments/adaptive_refinement.py` (config `configs/adaptive_refinement.yaml`), on an
RTX 4080 in float64. Every method gets the same continuous budget: 14 rounds x 40 steps = 560
preconditioned steps. The SRD variants are stochastic (proposal sampling), so each runs with
seeds 0, 1 and 2 and the table reports medians. The fixed baselines are deterministic.

**Target.** A cube-sphere with an anisotropic global deformation, which the coarse model can
represent, plus a localized bump, which it cannot. The bump is a LocalRefine on face 0 with its
inner control points displaced by 0.25 along the normal. The generating program is discarded;
the optimizer only sees the SDF grid (64^3) and 3000 coverage samples.

**Fit metric.** Exact narrow-band SDF^2 (area-weighted) + coverage^2, on a dense sampling
independent of the optimizer. This is stricter than the trilinear-SDF numbers in earlier
versions of this file.

| method | fit (median) | fit range | control pts | faces | complexity | total objective | refinements | simplifications | runtime / run (s) |
|---|---|---|---|---|---|---|---|---|---|
| fixed coarse | 1.54e-04 | - | 56 | 6 | 62 | 2.19e-04 | - | - | 10.0 |
| fixed uniform k=1 | 1.24e-04 | - | 98 | 6 | 104 | 2.32e-04 | - | - | 10.1 |
| fixed uniform k=3 | 5.16e-06 | - | 218 | 6 | 224 | 2.34e-04 | - | - | 10.4 |
| adaptive, naive scoring (A) | 1.54e-04 | 1.54e-04 - 1.54e-04 | 56 | 6 | 62 | 2.19e-04 | 0 | 0 | 21.8 |
| adaptive, marginal, no birth cost (B) | 1.61e-06 | 1.33e-06 - 1.37e-05 | 174 | 21 | 195 | 2.00e-04 | 9 | 20 | 49.2 |
| **adaptive, marginal + birth (C)** | **1.65e-06** | 1.61e-06 - 1.24e-05 | **85** | 10 | **95** | **1.01e-04** | 1 | 2 | 29.0 |
| adaptive C, uniform proposals | 6.88e-06 | 1.55e-06 - 1.07e-05 | 95 | 11 | 106 | 1.23e-04 | 2 | 4 | 32.7 |

`total = fit + lambda_fair * fairness + lambda_complex * C(s)`, with
`C = #faces + #control points` and `lambda_complex = 1e-6`.

## Representative run C (the median-fit seed; `events_C.json`, `exact_refinement_log.csv`)

1. **Coarse stall.** By step 80 the fit has plateaued near 2e-4 and `D_old` = 4.9e-6.
2. **Step 80: LocalRefine on the bump face, accepted.** The geometry is unchanged (exact).
   `D_new` = 6.0e-4, so `B_refine` = 6.0e-4 and the predicted improvement `eta * B` is 3.0e-4,
   against a birth cost of 3.3e-6. The naive immediate score for the same rewrite is -3.3e-5,
   a rejection.
3. **Realized vs. counterfactual.** 40 steps later the fit is 1.31e-5 with the rewrite and
   1.90e-4 without it (same state and steps, no rewrite): 14x better.
4. **Step 160: one KnotRemove** deletes a knot the bump did not need (85 -> 82 control points).
5. **Proposal statistics.** Of 94 exact refinement proposals scored, all 94 naive immediate gains
   were negative, and 89 got a negative score under C.
6. **Final state.** A 3x better fit than uniform k=3, with 38% of its control points.

## Observations and caveats

* **A (naive)** never accepts a refinement and ends identical to fixed coarse on every seed.
* **B (no birth cost)** matches C's median fit but churns (9 refinements, 20 simplifications)
  and ends with 2x the control points. Its total objective is twice C's.
* **Seed sensitivity.** Seed 0 is an outlier for B and C (fit ~1.3e-5). Hypothesis, not
  verified for this exact run: in earlier seed-0 runs the first accepted LocalRefine window was
  centered off the bump (u ~ 0.75 vs. the bump at 0.5), and later refinements only partly
  compensated. More rounds, or lookahead on the shortlist, are the obvious levers.
* **Uniform proposal sampling** is more variable (1.6e-6 to 1.1e-5) than residual-guided
  sampling.
* Three seeds and one target demonstrate the mechanism. They do not establish significance.
