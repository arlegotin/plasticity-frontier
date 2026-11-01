# Source-only B04 branch-switch diagnosis

The retained unresolved group follows the declared B04 procedure. Its large validation gap is explained by selection of different F/Z metrics across the final sampled interval. This targeted trace found no lost ordered basis, solver-comparison defect, or incorrect selected-metric correspondence. The existing unresolved tuning result remains valid under `min_marginal_loss_se_v1`; it is not converted to resolved.

The controller has directed the campaign's `INSENSITIVE_OR_UNSUITABLE` tuning off-ramp. No cap increase, policy change, candidate removal, branch-aware redesign, fresh fit, or dependent protected study was performed. These source-validation observations make no population-effect claim.

## Evidence and scope

- Source revision: `536a4003ca402d2e67d51b3bf24da256c3829314`, clean before diagnosis.
- Retained file: `runs/campaign/2026-11-01-main-objective-v1/source-cost-probe-cap8/pilot-04993ecbd02f2de0c6c9/preparation-12.json`.
- File size: 91,390,523 bytes; SHA256: `aa41390497dd9df729d92e375fa08f684bc359aa497572ff6505d8915bc7ecf6`.
- Outer preparation 12; 256 training and 256 validation source tasks. All indices below are zero-based positions in `candidate_records`.
- Parent asset: `20f47fffef06f02c8168dad7ccca59303df524777e0d409df8de51744273a6a9`.
- Estimator: `moment_directions`, rank 3, `y_squared_xx` basis provenance, explicitly labelled equal-variance surrogate. Ordered moment eigenvalues are `[3.35589853137993, 2.458140143536636, 2.220213733770468, 2.084800329474237, 1.8773559198348657, 1.8015092034155877]`.
- Estimated noise variance: `0.9753560491702808`; task SE `0.08880234294291621`. Source-task hash: `9b3a040d747bc7496e3075befd4c42120ef1f2c19a83f9780f3658d8f1023f52`.
- Preparation: dimension 6, target context 4, depth 2, norm cap 0.6, parameter cap 21, learning rate 1. Thus the retained frontier has width 3.

Only source-validation candidate fields, their prepared states and diagnostics, source code, and approved procedure documentation were inspected. No evaluator-truth or primary-contrast values were accessed or used. The file was decoded with the Python standard library; no project or NumPy imports, tests, fitting, or simulation occurred. Light metadata analysis overlapped the controller-reported independent four-control cost probe; this diagnosis supplies no timing estimate. The JSON companion retains exact indices, candidate IDs, source hashes, prepared/asset/metric hashes, branch diagnostics, and all nine rounds of interval checks.

## Actual crossing and validation gap

| B04 candidate index | Gamma | F prior objective | Z prior objective | Selected | Source-validation mean loss |
|---|---:|---:|---:|---|---:|
| 2930 | 0 | 0.7094846880677399 | 0.7092606814751085 | Z | 1.8724447180951755 |
| 2931 | 0.00390625 | 0.7098878758962006 | 0.7099442752251085 | F | 1.762242583818109 |

At zero, Z has gains approximately `(0.2, 0.2, 0.2, 0, 0, 0)` in the preserved ordered basis. At `1/256`, F has isotropic floor `0.1398012958994618` and two increments `0.2259074678296111`, hence approximately `(0.36570876, 0.36570876, 0.13980130, 0.13980130, 0.13980130, 0.13980130)` in that same basis. These are substantially different metrics.

The final left interval check is exactly `candidate_records[2931].tuning_checks[0]`: interval `[0, 0.00390625]`, absolute mean loss gap `0.11020213427706649`, tolerance `0.07032554806569007`, paired-difference SE `0.025151435736343715`, decision `refine_objective_gap`. Its proposed midpoint `0.001953125` was not executed because the declared cap was reached. The adjacent right check is already resolved: gap `0.00009074915586504417` versus tolerance `0.07032220432470453` on `[0.00390625, 0.0078125]`.

The same retained cap-8 ledger records this trajectory; no second large ledger was needed:

| Round | Zero / nearest-positive indices | Upper gamma | Left gap | Minimum marginal SE |
|---:|---|---:|---:|---:|
| 5 | 1436 / 1437 | 0.03125 | 0.10952880907825246 | 0.07030536997883023 |
| 6 | 1882 / 1883 | 0.015625 | 0.10992444573209204 | 0.07031597803666564 |
| 7 | 2380 / 2381 | 0.0078125 | 0.11011138512120144 | 0.07032220432470453 |
| 8 | 2930 / 2931 | 0.00390625 | 0.11020213427706649 | 0.07032554806569007 |

The label `unresolved` copied onto candidate records is the shared selector outcome; it does not mean every B02/B03 interval independently failed.

## Within-branch controls and preserved representation

At the same parent, round, and gamma, B03 retains the F-only metric and B02 retains the separately solved Z metric:

| Method | Gamma-0 index / loss | Gamma-1/256 index / loss | Change |
|---|---|---|---:|
| B03, F only | 2622 / 1.7621536487603353 | 2623 / 1.762242583818109 | 0.00008893505777374244 |
| B02, Z only | 3062 / 1.8724447180951755 | 3063 / 1.8724447180951755 | 0 |

All 61 B04 rows in rounds 0–8 have the identical full ordered-basis diagnostic hash `2035d872bea62636f704d73b5fe5e9838cc3e8a318ae6d42789957093eb4ab1f`. This hash uses compact sorted standard-library JSON, as specified in the JSON companion. For all 61 rows, direct comparisons confirm:

- Prepared prior equals the recorded asset; F uses its first two ordered basis columns and Z its full basis.
- Prepared metric equals the branch selected by the smaller recorded prior objective; both branch statuses are completed and every branch comparison is selected.
- F metric equals the paired B03 metric; selected-F initial layers equal B03 initial layers.
- B04's observed validation mean equals the corresponding selected B03/B02 control mean.
- Repeated gamma values across rounds retain exactly the same asset hash, prepared hash, validation mean and marginal SE.

The maximum Z-versus-B02 gain discrepancy across these rows is `5.551115123125783e-17`. The minimum branch objective separation is more than `5.63e8` times the implemented comparison guard `10 * max(abs(dual gaps), 1e-14)`. At the two final interval endpoints, F stationarity and complementarity residuals are zero; Z stationarity is zero, complementarity at most `3.36e-17`, and dual gap at most `1.11e-16`. These are retained float64 diagnostics, not exact certificates.

## Code correspondence and interpretation

The approved design's method table defines B03 as F alone and B04 as the selected joint frontier. `priors.py:52–66` retains the ordered moment basis through surrogate construction and shrinkage; `selection.py` passes that basis and analytic variances explicitly through noise attachment. This matches the Task 8 preservation correction. `frontier.py:20–47` consumes the supplied basis directly and selects by the estimated-prior F/Z objectives. `methods.py:51–66` uses that selected branch metric. `metrics.py:71–79` preserves the basis during canonical normalization, and `geometry.py:25–52` constructs network layers from those directions without an eigendecomposition. The actual retained comparisons above agree with these paths. Relevant source-file SHA256 values are in the JSON companion.

For this positive-noise prior, each branch has a strictly positive diagonal quadratic and a fixed compact feasible set. Its coefficients vary continuously with gamma, so the branch optima and optimum values vary continuously. The observed sign change of `F objective - Z objective` places at least one prior-objective crossing inside `(0, 1/256)`. The chosen estimator can switch between different branch metrics there. Equality of their estimated-prior objectives does not require equality of their finite source-validation losses. The controls show that branch selection accounts for the large observed gap, while the within-F change is small and within-Z change is zero over this sampled interval.

This is an interior branch crossing bracket, not evidence of a singularity exactly at gamma zero. The records do not determine the exact crossing location or prove that every possible future grid would remain unresolved. They do establish a valid failure to satisfy this campaign's finite tuning gate at its declared cap. No basis or solver defect was found in the scoped causal trace, and no additional scientific execution is justified by this diagnosis alone.
