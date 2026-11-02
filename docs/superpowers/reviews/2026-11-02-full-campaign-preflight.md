# Full staged campaign preflight — readiness review

Planning only: no tests, fits, scientific calculations, keys or campaign directories were created. Only this document was written. `FINAL_EXECUTION_SHA` means the future reviewed, clean commit containing all required repairs; no current revision is nominated.

Authority: approved `docs/superpowers/specs/2026-11-01-plasticity-frontier-design.md` §§5–11; `docs/blueprint.md` §§5.2, 5.5–5.6, WP06–WP10; actual `docs/results/2026-11-01-branch-resolution-followup.md`; retained `runs/campaign/2026-11-01-main-objective-v1/followup/continuation-runbook.md`. Earlier prospective statements in that runbook remain historical proposals where the actual follow-up supersedes them.

## 1. Source and upstream readiness

Finish, test, review and commit compact-pilot and report/replay repairs before the definitive full G6 pilot. Resolve real G9/report/replay capacity on this same source before protected freeze. Preserve full candidates, noneligible probes, failures, pairing and hash-bound assets; retain existing reserves and indivisible-unit refusals.

Bind each new execution to `FINAL_EXECUTION_SHA`, code/runtime/lock/source-manifest hashes, exact inputs, thread/BLAS policy and actual returned artifact paths. Benchmark worker runtime strata may differ from primary execution only when explicitly recorded and internally consistent; never substitute one runtime hash for another.

Minimum same-source supporting reruns are the existing eight-command G0–G4 set plus the retained descriptive G8 manifest:

- `doctor --output NEW_DOCTOR`; `check-theory --config configs/exploration/exact_cases.json`; `check-geometry --suite adversarial`.
- `reproduce --suite source_and_baselines --run-source --output NEW_REPRODUCTION`; retain six-budget, direct-map and singular ridge/posterior checks and supplied-script adverse output.
- `frontier --config configs/exploration/frontier.json --certify`; `export-exact --config configs/exploration/frontier.json --branch F`, then `--branch Z`.
- `frontier --config NEW_GRID_CONFIG --output NEW_GRID --process-record NEW_GRID_SIDECAR`; preserve the declared grid/tolerances and unresolved intervals.
- `robustness --manifest NEW_ROBUSTNESS_MANIFEST --output NEW_G8 --process-record NEW_G8_SIDECAR`; preserve all declared arms and their descriptive/inference labels. Reusing historical diagnostic settings is allowed only as labelled repeated diagnostics, never independent confirmation.

All commands use `.venv/bin/python -m prepared_learning`. Capture argv, status, stdout/stderr, exact inputs and before/after identity. Current-source G5 access, fitting/B11, accounting and selector validity must also be demonstrated by reviewed implementation and actual fresh pilot records. Old `536a4003ca402d2e67d51b3bf24da256c3829314` artifacts remain SOURCE/history inputs, never rebound as current-source G0–G9 evidence or finalized calibration. Same-source reruns establish the report/replay chain; they do not erase old failures.

## 2. Full G9 scope and workflow

The required full operational set for the approved six-network learned comparison is **14 descriptors**:

| Networks (6) | Matching compact equality controls (6) |
|---|---|
| `B01` | `B05:B01` |
| `B02` | `B05:B02` |
| `B03` | `B05:B03` |
| `B04` | `B05:B04` |
| `B11:F` | `B05:B11:F` |
| `B11:Z` | `B05:B11:Z` |

Add `B08:ridge` and `B10:empirical_bayes`. The design explicitly requires B05 for B01–B04 and both B11 gain-fitting families; WP09 compares networks with structured maps. Extending equality coverage to both trainable B11 realizations yields six pairs and satisfies the public benchmark's every-network control requirement. The spec does not literally state “14”; this derived full scope also matches the retained `predeclared-decisions.json` descriptor list. B00/B06/B07/B08:gd/B09/oracle-posterior remain required baseline/attribution evidence, not silently counted among these 14 operational descriptors.

The actual old `operational-cost-probe-manifest.json` had only `B01`, `B05:B01`, ridge and EB; four balanced blocks produced 16 workers. Its source input is historical exploratory evidence, not a full G9 or a new-source calibration.

Use `workflow=generated`, first `phase=exploratory`, with common source observations/options, actual selected T/V, immutable selected assets, identical precision/optimization rules and measured conditions. Full position balance requires a multiple of 14 blocks: the smallest cycle is **14 blocks × 14 methods = 196 fresh-interpreter workers per phase**, at least **392 calibration + fixed workers**, before technical replay. Per-worker pilot samples/repetitions are additional work, not independent source replications. Measure cold/warm preparation/adaptation/prediction, end-to-end and serialization, RSS, bytes/scalars and amortization; charge coefficient collapse and share compact/small-space optimizations.

Predeclare future G9 `settings.checkpoint_tasks=1` via the allowed manifest field `checkpoint_tasks: 1` (there is no CLI flag), so each indivisible raw block contains one worker. The default 16 can put 16 workers in one block; Task4 streams an existing block as one indivisible unit, which could exceed half live RAM under the unchanged 256× allowance. The controller resolves batch size separately. This operational choice preserves methods/candidates, precision and frozen timing/source contracts; it is not a capacity pass.

Run `benchmark --manifest NEW_CALIBRATION_MANIFEST`; inspect uncapped, resolved timing calibration and complete raw blocks. Then call public Python API `prepared_learning.benchmark.finalize_benchmark_manifest(calibration_path, manifest_path, campaign_output)` and run `benchmark --manifest FINALIZED_MANIFEST`. There is no finalize CLI. Finalization requires the exact clean calibration source/runtime identity and hashes; fixed execution preserves selected execution hashes, observations, order and measured repetition schedule. Never rebind the old four-control calibration. Timing repetitions support conditional operational claims only. Require every direct equality, risk/control and baseline gate; report direct-map domination honestly, with raw network norm inapplicable for software direct/ridge/EB.

## 3. Fresh full G6 and protected eligibility

Start one public full-pilot invocation with two independent blocks of **30 complete paired preparations per block at each T=V level [32,128,256]**, hence **180 W0 preparations**; `workloads=["W0","W1","W2"]`. The 18 reused preparations contribute zero new units and must not be appended. Keep population seed `2026110106`, richer d6/n4/depth2/norm0.6/parameter21/eta1 contract, estimated prior/noise, source block size4 and original six-method selectors/candidates. Use a controller-audited genuinely fresh exploration root; no root is allocated here.

`sample_plan.analysis_mode=conditional_risk_over_preparations`; initial T=V32; outer J and inner N initially null. Baseline selector is B01/B02/B11:Z; joint selector is B03/B04/B11:F/B11:Z. Use the versioned branch-envelope policy emitted by final source, not an invented config switch; local float64 resolution does not certify global optimality or guarantee future termination.

Require a measured poor source regime, a complete accuracy-qualified stable level, resolved tuning/failures and cost-based choice. Initial relative-error target remains .5. Source levels/brackets may change only from preparation accuracy/stability/cost, never D signs, p-values or favorable roots. Unresolved brackets remain unsuitable. Preserve every failed paired unit.

Require two positive block variance estimates with ratio ≤4 and stable precision planning. Keep h=`Fraction(1266883,39703750)/4`, primary alpha=.05 and the declared approximate variance allowance. No final J exists yet. Zero/unstable variance calls for documented pilot extension or `INSENSITIVE_OR_UNSUITABLE`, never zero uncertainty. Evaluate Student/outer-bootstrap assumptions, extreme observations, numerical error and the bounded-map policy; freeze the supported interval and its approximate-coverage qualification.

Calibrate at **actual planned J**, not pilot length: two ≥30-unit blocks, matching endpoint/interval policy, ≥500 resamples (initial proposal 1,000), adequate centered-reference calibration and stability (public coverage acceptance .90–1.00 is diagnostic, not a coverage proof). Primary analytic risk has no inner simulation N. Network-update checks and X-dependent ridge/EB use separately fixed schedules; the current primary protocol's operational simulations are descriptive-only. Nested endpoints require their own J/N and variance components. Prespecified secondary confirmation families need Holm or simultaneous intervals. Student-input stress with df>8 can support finite-moment inference; infinite-fourth-moment stress is an assumption-failure diagnostic without an ordinary Gaussian mean-risk CI.

The full prescribed workload grid is W1 d64/r8/n4,8,32/widths1,8,9,64 and W2 d64/trace-normalized i^(-3/2)/n8/widths4,8,16. That broader workload/attribution scope is distinct from current `run_pilot` resource scaling: exactly one W1 and one W2 preparation, each d64/n8/width8/norm.6/P520, using `source_levels[0]` source tasks (initially32) and source block size8. These two measured cost records do not cover the full grid or establish transfer. Pilot scaling and exploratory G8 remain descriptive; retained W1/W2 transfer claims need separately frozen endpoint/count/selector/multiplicity contracts and fresh independent source/test cohorts. Exploratory robustness changes require a recorded amendment and fresh confirmation for every retained claim.

## 4. Capacity ledger before freeze

Actual postfix reused-cohort evidence: selected-level **15,805,577 B full preparation envelope**, **2,548,948 B task envelope**, **25,642,925 B largest pretty preparation file**, **485,834,752 B cumulative whole-process RSS**, and **221.545 s for all 18 reused preparations**. These are observed local values, not W1/W2 bounds, per-method peaks or assurance that fresh 180-unit/refinement histories fit.

Conditional planning arithmetic only: if the same mix/cost persisted, W0180 is about 2,215.45 s (36.9 min), excluding extra calibration, W1/W2, G9 and replay. Repeating the selected-level envelope180 times is 2,845,003,860 B; repeating the observed largest pretty file180 times is 4,615,726,500 B. These are different serialization scenarios, not additive totals or proven upper bounds. Compact retained summaries must never replace raw sizes in resource admission. A proposed 10,800-second allowance needs complete-work measurements first.

Controller must measure live available RAM, free and allocated disk, current raw maxima, process/isolation overhead and runtime projection. Reserve at least half live available RAM and the established half-free-disk allowance. Account for largest indivisible fit/parse, current compact state, serialization copies, report summaries/risk vectors/index and old/new replay pair plus projections under unchanged admission factors. At 256× expansion the observed largest pretty file alone is 6,564,588,800 B before baseline; two such live units are 13,129,177,600 B before projections. This is a warning scenario for the repaired guards to measure, not a measured resident size.

Disk ledger must include retained history + fresh full pilot raw/assets + protected original and replay stores + G9 calibration/fixed raw/assets and technical replay + complete report worker references/copies + reports/figures/logs/partial recovery files + clean clone/venv/dependencies. Hash-bound worker references reduce RAM retention but do not make the retained complete workers free on disk. Count logical and allocated bytes, temporary publication overlap and copy/hardlink behavior.

Let S be the sum of JSON bytes in **both** protected original/replay stores. The unchanged confirmation comparison index needs **8*S + 64 MiB** temporary disk within half currently free temporary-disk space, in addition to those stores and all other allocations. Benchmark replay does not use that SQLite index. With an assumed unchanged per-preparation pretty-file maximum, S alone would scale as about `2*J*25,642,925` plus tasks/metadata/assets; J and actual maxima remain unknown. Do not treat disk admission as passed before measuring them.

Time ledger includes final-source upstream reruns, G6 generation/fits/publication/calibration, representative W1/W2, every G9 calibration/fixed worker and repetition, protected J plus fixed simulations, G8/transfer arms if retained, interrupted/resumed verification, clean technical replay and final reconstruction. Require `project_runtime`, `validate_execution_envelope` and `execution_overhead` to match actual final J and schedules. Keep launch overhead, timeout/cap failures and isolation costs. The old four-control worker mix cannot forecast ten unmeasured descriptors as established capacity. Task4 owns implementation details; this preflight requires its measured capacity evidence before freeze.

## 5. Exclusion union, freeze and delivery

Refresh a provenance-bearing registry with exact arbitrary-precision Python integers (`type(value) is int`, nonnegative, unique; reject bool/float/string coercion). Union retained `runs/campaign/2026-11-01-main-3e7d329/exploration-roots.json` and its `root-inventory-provenance.json`, all subsequent known configs/run metadata/protected development roots, and every new exploratory/root reservation. Do not parse large roots through JavaScript numbers. The original referenced provisional workspace is absent; its preserved provenance is authoritative history, not evidence of a new scan.

Read-only audit of the retained provenance and `main-objective-v1/predeclared-decisions.json` found prior reservations `202611011001`–`202611011005`, later `202611011006`–`202611011011`, actual reused root `202611011007`, fit `202611011006`, G8 `202611011009` and cost `202611011011`. Full-pilot **202611011008** and full-G9 **202611011010** were unexecuted reservations in the retained campaign. Preserve them conservatively in exclusion history; neither is allocated or declared fresh here. Later controller inventory must distinguish reservation from execution and retain the limitation that deleted/unretained random development roots cannot be reconstructed. No old root is automatically reusable because its execution was incomplete or absent.

After genuine G6 pass and whole-pipeline feasibility, copy actual sealed `pilot.json`/`protocol_draft.json` into a new ignored handoff and union this registry into **`evidence.excluded_roots`**; preserve sealed pilot bytes. There is no `--exclude-root` flag. Freeze all scenarios/counts/selectors/grids, endpoint/inference/multiplicity, code/dependencies/assets, key mapping and analysis before protected access. Only successful public freeze allocates protected keys.

Future paths, with actual controller-selected filenames and returned RunStore paths:

```sh
.venv/bin/python -m prepared_learning pilot --config NEW_FULL_PILOT_CONFIG --output NEW_PILOT_OUTPUT
.venv/bin/python -m prepared_learning freeze --protocol NEW_HANDOFF_DRAFT --output NEW_CONFIRMATION_MANIFEST
.venv/bin/python -m prepared_learning confirm --manifest NEW_CONFIRMATION_MANIFEST --output NEW_CONFIRMATION_OUTPUT --checkpoint-tasks 16
.venv/bin/python -m prepared_learning analyze --manifest NEW_CONFIRMATION_MANIFEST --run ACTUAL_CONFIRMATION_RUN
.venv/bin/python -m prepared_learning report --manifest NEW_REPORT_MANIFEST --inputs ACTUAL_INPUT_DESCRIPTORS --output NEW_REPORT
.venv/bin/python -m prepared_learning replay --manifest NEW_REPLAY_MANIFEST --output NEW_REPLAY
```

Any intervening code/dependency/scientific change requires a new confirmation identity; even a docs-only commit changes strict Git identity. Finish commits before definitive pilot, then defer code/docs/evidence commits until confirmation, report and same-source clean replay finish. Keep intermediate inputs/results ignored; later evidence commits retain the executed source SHA. Operational checkpoint/batch recovery preserves identities only under the existing logged recovery contract. Validate actual G10 report inputs, complete raw reconstruction, interrupted/replayed equivalence and claim/source links; technical replay is never pooled as independent evidence.

## 6. Conditional outcomes and optional scope

G6 tuning/bracket/variance/calibration failure => `INSENSITIVE_OR_UNSUITABLE`; indivisible RAM/disk/time failure => `LOCAL_RESOURCE_LIMIT`; preserve raw evidence and do not freeze. Any extension must be effect-independent, documented and separately feasible. Invalid implementation or verified counterexample blocks affected descendants with `IMPLEMENTATION_INVALID` or `MATHEMATICAL_COUNTEREXAMPLE`; solver failure alone is not a counterexample.

Valid fixed confirmation with no supported benefit is reported as inconclusive or `NO_COMPARATIVE_ADVANTAGE` according to interval/sensitivity, without outcome-driven extension. A supported result is only `SUPPORTED_IN_SCOPE`. G9 may proceed independently of a positive G7 if its baseline/equality/calibration/resource gates hold; report practical domination or tradeoffs even when the theorem remains valid. Do not promise either G6 passage or a positive result.

Optional original-scope natural arm remains UCI white Wine Quality covariates with quality discarded, official asset/license/hash verification, identical-row grouping before deterministic60/20/20 split, development-only transforms and artificial tasks/labels. It is conditional semi-synthetic robustness, not wine prediction; retained claims require their own protocol and fresh protected pool/cohorts. No new download here. Omit and label unrun if asset/license/split gates fail or the arm resolves no remaining input-distribution question.

Load-bearing unknowns: final source, fresh resolution/source bracket/variance/J/calibration/tails, representative W1/W2 costs, all14-method G9 costs and indivisible records, and measured simultaneous RAM/disk/time through protected comparison and clean replay. The local18/18 result settles none of those by itself.
