# Actual descriptive G8 robustness execution

All 26 predeclared arms executed with top-level status `completed`; the public CLI returned 0 in 0.380632209 seconds. Its gate is `INSENSITIVE_OR_UNSUITABLE`: these fixed small diagnostics provide no protected confirmation, retained amendment, transfer CI, or latency claim. The campaign's unresolved tuning off-ramp still stops dependent full G6 and protected G7.

Source remained clean at `536a4003ca402d2e67d51b3bf24da256c3829314` before and after execution. Runtime, code, dependency-lock, source-manifest and selection fingerprints matched. No source or dependency edits occurred. The full 438,594-byte scientific artifact is `robustness/robustness.json`; the exact command, stdout, stderr, return code and UTC/perf-counter timing are in `raw/g8-robustness.process.json` and its stdout/stderr files. The CLI-created `robustness-process.json` agrees byte-for-byte on captured stdout/stderr and on return code. Stderr is empty. `g8-summary.json` indexes every arm, nested adverse status and retained artifact hash, and supplies the complete-artifact replay descriptor.

The public command was:

```sh
.venv/bin/python -m prepared_learning robustness --manifest runs/campaign/2026-11-01-main-objective-v1/robustness-manifest.json --output runs/campaign/2026-11-01-main-objective-v1/robustness --process-record runs/campaign/2026-11-01-main-objective-v1/robustness-process.json
```

The capture records the actual absolute-path argv. The base was a standard JSON copy of `fit-config.json`, with exploration root `202611011009`, unchanged population seed `2026110106`, and fitting-cap metadata 8. Manifest creation used committed `load_config` and `make_manifest(base, list(ARM_NAMES))`; all arm declarations and numerical interventions are unchanged public output. Only the top-level purpose text was corrected from the generator's generic fixture wording to actual descriptive execution, with a campaign note. The original wording is retained in `g8-input-identity.json`.

- Base file SHA256: `1cda1d8d8e80e4b42846ece449ad50e207b8f99e29a8f913f892c278130efbbe`.
- Loaded base content hash: `4b330e25aba6f3198ec1edaa1232d4c103fcb36d4cee326efdc45eb6ce90b241`.
- Manifest file SHA256: `58aadb6c25f8c178e14670d2ea03573ddcf95c916f8f004abbfe615b4285dc57`.
- Manifest content hash, also matched in the scientific artifact: `f43cc98382cc83295b38bff58fc71c28e5c5dd4ab0fccd9f59683d7adca6a2e5`.

Every following arm completed. Values describe this execution, without confidence intervals or population-effect conclusions. Exact arrays, failures, branch diagnostics and original qualifications remain in the full artifact.

| Index | Arm | Retained outcome or scope |
|---:|---|---|
| 0 | no_floor | Independently optimized F/Z retained. Unchanged-gain floor removal is infeasible: norm 0.7155684 exceeds cap 0.6. |
| 1 | isotropy | Optimized scalar 0.2352941; diagnostic conditional risk 0.7647059. |
| 2 | actual_counts | At norm cap 0.6, F/Z counts 21/21; at 0.4, 21/14. Padding gives 21/21 with zero map error. |
| 3 | same_map | Canonical/head-zero first-step errors 1.37e-17/2.22e-17; counts 21/42, versus 15 stored scalars for the compact map. The 42-parameter realization exceeds the base cap. |
| 4 | wrong_basis | Same-spectrum rotated map has diagnostic risk change +0.1406367; original/rotated resources retained. |
| 5 | prior_uncertainty | Three independent 8-task source preparations; all shrinkage 0/0.5/1 F/Z and empirical F/Z candidates retained, known shared noise, no test-risk selection or CI. |
| 6 | prior_rotation | 24 paired tasks; 12 negative Z-minus-F coefficient-loss differences; actual rotated population and both losses retained. |
| 7 | leakage | 24 paired tasks; 5 negative differences; actual 0.2 complement leakage retained. |
| 8 | mean_shift | 24 paired tasks; 11 negative differences; nonzero-mean second-moment change explicitly retained. |
| 9 | anisotropic | 24 paired tasks; 7 negative differences; sampled coefficient/prediction losses under changed input covariance, not the isotropic risk formula. |
| 10 | student10 | 24 paired tasks; 7 negative differences; finite eighth input moment; descriptive only. |
| 11 | student3 | 24 paired tasks; 7 negative differences, minimum -6.1226002; infinite fourth input moment, no ordinary mean-risk CI. |
| 12 | orthogonal_coordinates | Tangent error 1.48e-16 and norm change 1.11e-16. |
| 13 | nonorthogonal_coordinates | Norm changes from 0.6 to 0.8380298; tangent discrepancy 0.4359535. Euclidean resource cost is not invariant. |
| 14 | free_eta | Scale 2, eta 0.25, initial norm and tangent ratios 4; first coefficient unchanged. Norm cap deliberately relaxed. |
| 15 | depth_one | Zero-norm six-parameter linear predictor, explicitly outside the deep architecture. |
| 16 | fixed_feature | Three trainable head parameters plus stored features: 21 total stored scalars; direct error 6.25e-18; outside the all-trainable class. |
| 17 | approximate_width | Widths 1 and 2 each retain truncation and independent F/Z reoptimization, losses, counts and solver diagnostics. |
| 18 | float32 | Literal float32 layers/gradients/updates, float64 evaluation; coefficient error 9.16e-9. No exact multiplicity certificate. |
| 19 | extreme_scale | All ten dtype/scale combinations completed from 1e-30 through 1e30; large absolute errors retained, including 8.50e21 at float32 scale 1e30. No overflow failure occurred in this execution. |
| 20 | second_step | Fixed-initial-metric recurrence disagrees by 0.0168983. Executed depth 3, norm 1.5491933 and count 22 violate the original depth-2/norm-0.6/count-21 comparison contract; all eligibility flags retained. |
| 21 | noncanonical | Rational coefficient 0.64 versus tangent prediction 0.6, plus all four signed random disagreements retained. |
| 22 | degeneracy | Float64 near-minimum gap 2.78e-17 collapses to zero in float32: retained `precision_support_loss`. |
| 23 | information | Oracle/estimated/no-prior references share known noise and caps; all source candidates retained, no validation or test-risk winner promoted. |
| 24 | shared_optimizations | Ridge primal/dual discrepancy 1.63e-16; compact/dense and collapsed-network discrepancies retained for F/Z. Numerical equivalence only, no timing claim. |
| 25 | resume_error | One recovered pair retains one deliberately injected failed method row; duplicate pair and changed config both rejected. |

There are zero top-level failed arms and zero extreme-scale sensitivity failures. Nested adverse status records are `/arms/22/near_minimum_execution/status = precision_support_loss` and `/arms/25/replayed_rows/1/status = failed`; the latter is the declared storage diagnostic's injected failure, not a failed arm. Infeasible controls, outside-class comparisons, noncanonical and second-step discrepancies, negative sampled differences and large absolute precision errors are retained regardless of status labels.

The two source arms each repeat the same three 8-task schedules: 48 recorded source-key occurrences represent 24 unique source tasks and 192 unique labels across both arms, not six independent cohorts. The six boundary arms each contain 24 paired tasks; shared random roots/interventions prevent treating their concatenation as 144 independent observations. Changing the base T/V=32 or cap-8 metadata did not enlarge these fixed public schedules. Both coefficient and prediction losses are retained. Gaussian, finite-eighth-moment Student, and infinite-fourth-moment Student scopes remain distinct.

All 26 amendment records and both W1/W2 transfer records retain the public schema status `unperformed`, meaning not run. `old_protected_keys=[]` is truthful: no protected keys were allocated for this campaign, and no current primary result exists. The generic output `original_primary_result="unchanged"` does not establish a primary result. No G6/G7 reopening or new study follows from these descriptive results.
