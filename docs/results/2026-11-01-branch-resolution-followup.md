# Reused-cohort branch-resolution verification — 2026-11-01

The versioned branch-envelope procedure resolves all **18/18 reused exploratory preparations**, compared with 17/18 under v1. This verifies the local stopping repair on development observations. **Independent new preparations: 0. G6 remains `INSENSITIVE_OR_UNSUITABLE`.** No protected keys, confirmation, full G6 pilot or primary-advantage claim are produced.

## Executed identity and capture

The completed public pilot used clean committed source `5ee684002ea34da80b124f62df2c21dac985f942`, policy `min_marginal_loss_se_branch_envelope_v2`, code SHA `dc9f60de9b357830cfb151e0a464a670730453e3e37fd40cd5fc55edaeb36d51`, runtime SHA `811ac6ab70956520cb018d53719f84c374cf8e1504961395d9eaf278b8fb7a7a` and lock SHA `6e648a5a51d603b8a9fcd4ea85c2d7316fd5b022ab6dcd48b80b021685507fe6`. Historical v1 remains source `536a4003ca402d2e67d51b3bf24da256c3829314` / code SHA `04806bbd50355d1c16986541e6986c37e8f1c8f5b28fff5f2ba8e249772ce89d` / `min_marginal_loss_se_v1`; its original unresolved verdict is retained. Runtime and lock identities match the historical run.

The exact copied config SHA is `b539b1ffa4c828148dfd9871d3fa0f9984611c55e919857271d9ec705594225a`. Source/validation levels [32,128,256], two blocks × three preparations per level, W0, population seed 2026110106, exploration root 202611011007, max refinement rounds 8, max seconds 2400 and all six method IDs are unchanged. The config's legacy `exact_fixture_path` string is retained byte-for-byte; `specs.load_config` does not consume it, and neither execution nor comparison imports test fixtures.

```sh
.venv/bin/python -m prepared_learning pilot --config runs/campaign/2026-11-01-main-branch-envelope-v2/reused-cohort-validation.json --output runs/campaign/2026-11-01-main-branch-envelope-v2/reused-cohort-recovery
```

A synchronous parent retained complete file-backed stdout/stderr and captured actual CLI exit **0**. Start `2026-11-01T22:50:17.995213+00:00`; finish `2026-11-01T22:53:59.540358+00:00`; elapsed 221.545167 seconds. The actual completed store is `runs/campaign/2026-11-01-main-branch-envelope-v2/reused-cohort-recovery/pilot-02b3f8ea13c322e1c024`; public stdout, `run.json` and verified terminal all say `completed`. That execution status does not pass G6. Terminal SHA: `c6caeb83d55ad9337bdb5a2d3c4eb1cc78bc44964eac18cd2e7a68d80496f357`. All 37 journal receipts, block manifests, records and asset hashes were checked without repairing or rewriting the store.

The first capture wrapper failed with `PermissionError: [Errno 1] Operation not permitted: 'ps'` after launching the CLI. Its store `reused-cohort/pilot-0563a0d90f52266c0971` contains preparations 0–11, no terminal and empty stdout/stderr. A narrowly scoped process-state check established that its child PID no longer existed. Wrapper exit was 1; actual child OS exit is unknown. It is an interrupted incomplete attempt, with no cohort verdict. It and its wrapper/error metadata remain intact. The completed recovery repeated exactly the same observations with unchanged source/config/caps; the incomplete attempt is never pooled as additional evidence.

The recovery launcher measured 17,851,678,720 available RAM bytes and reserved half (8,925,839,360); the public pilot independently measured 17,782,013,952 and enforced its half-memory budget 8,891,006,976. Start disk free was 34,698,403,840 bytes with half (17,349,201,920) reserved; finish disk free was 34,324,226,048. Public time/RAM guards remained active. Reported whole-process cumulative peak RSS was 485,834,752 bytes; this is neither a per-method maximum nor full-G6 capacity evidence. All inherited thread controls and installed runtime fingerprints are retained in the capture.

## Observation and estimator identity

One old/new preparation pair was loaded at a time. All 18 ordered source-key lists, source keys, training/validation observation hashes, population second-moment/noise hashes and level/block/replication identities match. Both cohorts contain exactly **4,992 unique source keys**. All **3,536 common logical candidate points** match exactly in asset, ordered basis, noise, metric, initial layers, prior-selected branch, options, prepared-state hash and validation mean/task losses. Repeated attempts within each run show no pointwise drift. Attempt IDs, grids, timing IDs and final selected gamma are procedure-dependent and need not match.

The complete historical 16,532-file inventory is unchanged before/after execution and comparison. Comparison SHA `4a76b4707b631a702b14c15560336f00189eda68f4d5ea782e939848e7e9c5ad`; raw preparation hashes, all common-point checks, full outer-12 envelopes, complete failures/refusals and per-method final checks are retained in `runs/campaign/2026-11-01-main-branch-envelope-v2/cohort-comparison.json`.

## Local resolution and measured costs

| Procedure | Resolved preparations | Candidate attempts | Preparation seconds, sum | Tuning seconds, sum | Largest raw preparation bytes |
|---|---:|---:|---:|---:|---:|
| Historical v1 | 17/18 | 10,244 | 311.440428 | 303.234060 | 91,390,523 |
| Recovered v2 | 18/18 | 7,228 | 196.794289 | 190.618203 | 25,642,925 |

The final v2 round has 1,015 ordinary resolved neighbor checks and 2 checks resolved by the branch envelope, with no remaining refinement request. By method: B01 432 ordinary; B02 93; B03 101; B04 101 ordinary plus 2 envelope; B11:F 144; B11:Z 144. No candidate, paired-role or restoration failures occurred in either complete cohort. Final refinement iterations are 1 for 8 preparations, 2 for 9 and 3 for 1, within the unchanged cap 8.

All **33** actual envelope calls remain in raw evidence: **2 resolved, 23 unresolved intermediate checks, 8 unsupported**. All 8 refusals report `unresolved prior branch comparison`; they retained ordinary refinement. The 70 retained probe records comprise 44 F and 26 Z records; every probe is noneligible, and each envelope declares zero additional labels/independent tasks. Envelope elapsed work is 0.635560 seconds; nested completed-probe work is 0.010946 seconds. These are charged within tuning/preparation times, not added again. The second successful interval is outer 13 B04 [0,0.125], bound 0.03464128682416279, tau 0.06566524919474218, margin 0.03102396236401728. Earlier unresolved/refused checks are not suppressed.

Outer 12 drops from 3,172 to 520 candidate attempts and from 127.887542 to 22.540495 preparation seconds. These same-data timings are descriptive engineering observations without thermal/power controls, replicated benchmark comparisons or operational-superiority inference. Total raw preparation bytes are 296,241,342 old / 211,498,339 new. The selected-level full-record execution envelope is 15,805,577 preparation bytes and 2,548,948 task bytes under `paired_chunks_v1`; compact-summary bytes were not substituted.

## Retained outer-12 switch

For B04 base `20f47fffef06f02c8168dad7ccca59303df524777e0d409df8de51744273a6a9`, the old final interval [0,1/256] had gap 0.11020213427706649 and tau 0.070325548065690066; the old result remains unresolved. The new procedure resolves the enclosing **[0,0.25]** interval, with eligible incumbent gamma 0.25 and these recorded quantities:

| Quantity | Actual value |
|---|---:|
| F lower bound | 1.707438130011568 |
| Z lower bound | 1.725909738814186 |
| Lower envelope | 1.707438130011568 |
| Possible improvement bound | 0.064075590189603959 |
| Endpoint marginal-SE tau | 0.070416578202313312 |
| Comparison padding | 6.646127014026746e-12 |
| Numerical margin | 0.006340988006063226 |
| Full-gain sensitivity K | 0.82668294788813612 |
| rho lower | 0.49383901229254917 |
| Incumbent loss padding | 3.9813633943905664e-08 |

Tau continues to be `min(left_marginal_loss_se,right_marginal_loss_se)`; its numerical value differs from the historical narrower interval because the right endpoint is 0.25. It was not enlarged by a setting change or probe loss. The gamma-0 selected branch remains Z and gamma-0.25 selected branch remains F at common points. The unselected gamma-0 F probe has loss 1.762153648760335, below the eligible incumbent's 1.7715136803875378; it remains noneligible and does not replace that incumbent.

The `uniform_original_unit_residual_complementarity_feasibility_v1` contract includes both anchor and unsampled successful-solver errors. F anchor/uniform error allowances are 0.00006396910917096702 / 0.00012224401680618343; Z anchor allowance is at most 0.00006377687972654803 and uniform allowance 0.0001029514272406314. Both branches require nonnegative multipliers, stationarity/complementarity residuals below 1e-9 and feasibility at most 6e-11; recorded rounding inflation, multiplicities, loss/gradient/map/distance padding, endpoint residuals and witnesses remain in the raw diagnostics. Width is 3, gain cap 0.6, estimated noise 0.9753560491702808; basis orthogonality error is 1.5142969544082355e-15. This is a **local `float64_diagnostic` conditional on accepted successful solves**, not a rigorous enclosure, confidence statement, global optimum or promise of termination. Solver success over unevaluated points is not asserted.

## Source accuracy and continuation

| T=V | Relative covariance error mean | Upper diagnostic | Eigenspace error mean | Validation stability | Preparation seconds mean |
|---:|---:|---:|---:|---|---:|
| 32 | 0.700932767 | 0.798078813 | 1.252256594 | stable | 2.065090 |
| 128 | 0.484284819 | 0.559369346 | 0.513312736 | stable | 8.940172 |
| 256 | 0.357659531 | 0.396779433 | 0.563916379 | stable | 21.793786 |

The public source-only accuracy/cost decision selects T=V256 at target 0.5. Primary effect signs were not used to choose settings, source levels or continuation. Public G6 remains `INSENSITIVE_OR_UNSUITABLE` because final outer count, sufficient independent-block calibration, final precision planning and local resource feasibility are unresolved. There are only three reused preparations per block at each level, below the required 30; W1/W2 scaling and a final J are absent.

Proceed only to the separately reviewed bounded-retention/capacity work, then a separately configured genuinely fresh full-G6 pilot if its gates permit. Preserve all negative/unsupported raw checks and historical verdicts. Do not allocate protected roots, freeze, confirm, extend caps or claim a primary advantage from this repeated cohort. Cap5, cap8, interrupted v2 and recovered v2 remain **the same 18 development preparation units, with zero newly independent preparations**.

Raw capture/artifact verification: `runs/campaign/2026-11-01-main-branch-envelope-v2/recovery-capture-metadata.json`, `recovery-stdout.txt`, `recovery-stderr.txt`, `artifact-verification.json`, `completed-run-inventory.json`, `interrupted-run-inventory.json` and both historical inventories. Raw captures are ignored local evidence; this document is the selected durable finding.
