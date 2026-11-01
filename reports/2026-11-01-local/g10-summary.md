# Actual combined report and clean replay

The public final report returned `SUPPORTED_IN_SCOPE` for **G10 report reconstruction and clean replay only**. Source is unchanged and clean at `536a4003ca402d2e67d51b3bf24da256c3829314`. No source/dependency edits, implementation changes, tests, commits, or new scientific studies were performed in this task.

The approved scientific off-ramp remains `INSENSITIVE_OR_UNSUITABLE`: the full at-least-30-preparation G6 pilot, protected G7, fixed/nested secondary studies, retained G8 amendments, W1/W2 confirmations and finalized full14-method G9 are not run. Natural-input, GPU and physical-energy work are also explicitly not run. No learned population effect, practical superiority, natural transfer, novelty or global-theorem claim is made.

## Authoritative outputs

- Final report: `report-final/results.md`.
- Final ordered manifest, including the actual receipt: `report-inputs-durable.json`.
- Frozen replay input: `replay-durable.json`.
- Actual receipt: `clean-replay/replay.json`.
- Complete execution/index summary: `g10-summary.json`.
- Original-byte preservation receipt: `combined-preservation-after-replay.json`.
- Exact argv, stdout, stderr, return codes, UTC and perf-counter captures: `combined-command-captures/`.

Input-order hash: `cf237b4192651839303837c632d9762cc0a09835c85f13c1f750ec5061512720`.

Final manifest file SHA256: `f35ace08cc1b6bd8eb734b7f91911ec20a2766e625d884e951d7e965ac37ff60`.

Clean replay receipt SHA256: `9fde6c8855fe8cde7a76f0805e00362c6f2a24cb5da63c6db8407d4305d88547`.

## Execution and verification

| Public command | Return code | Elapsed seconds | Result |
|---|---:|---:|---|
| `report --manifest report-inputs-durable.json --output report-before-replay-durable` | 0 | 20.300823 | Reconstructed; replay not yet attached |
| `replay --manifest replay-durable.json --output clean-replay` | 0 | 26.996194 | Matched in fresh locked environment |
| `report --manifest report-inputs-durable.json --output report-final` | 0 | 21.165118 | `SUPPORTED_IN_SCOPE`, matched replay attached |

The table abbreviates paths; captures contain the actual absolute argv. Replay created a fresh local clone at the executed revision and a fresh virtual environment, verified all 13 manifest-listed wheels, installed the lock without network access, and completed `pip check` successfully. All setup/worker subprocesses returned 0. It restored 105 assets and retained complete process logs. The receipt has `clean_environment=true` and `fixture=false`; exact deductions still carry their finite-fixture qualifications.

Seven actual executable inputs matched: theory, adversarial geometry, single frontier point, exact F export, exact Z export, full frontier grid and all26-arm robustness artifact. The two rational exports matched with zero tolerance. Other numerical comparisons used the declared `rtol=1e-10`, `atol=1e-12`; grid/robustness comparison includes the complete output file plus process summary/status. The other 23 inputs were retained without execution. No benchmark worker or protected study was invented or replayed, and replay adds zero independent observations.

After replay, streaming SHA256/size checks matched all **144 unique report-inventory files** and **81,560 archived original files**, totaling **2,043,974,526 original bytes**, with zero mismatches. Both old/current archives and their original authorities remain unchanged. Only `replay.receipt` was added to the durable report manifest; input order, inventories, claims and challenge responses did not change.

## Retained scope and limits

The 30 ordered inputs comprise 11 source records, five diagnostics, two exact exports, eleven explicit not-run records and one separately labelled forecast. The complete current/old exploratory archives retain every original candidate/worker/failed source record. Cap5 and cap8 use the same18 preparation cohort, not36 independent units. The B04 source-validation mean-query-MSE gap is distinct from the unperformed conditional coefficient-risk population comparison. The four-control16-worker cost probe remains SOURCE provenance; engineering projections are forecasts from the same measurements, not full-scale measurements or operational proof.

The claim ledger contains only two available `exact_rational_fixture_kkt_verified` claims with all five contract fields copied from the actual F/Z exports. All twenty public challenge keys have actual input IDs and scoped limitations.

Standard CSV/SVG/PDF outputs were rebuilt. `frontier.csv` contains 507 rows: 502 branch rows for251 grid points across widths1/2/3, all three unresolved intervals, and two single-point branches. All26 robustness arms and nested adverse outcomes remain in full diagnostics. Protected comparison, resource and Pareto panels correctly show not run/no applicable observations.

The public frontier renderer emits a constrained-layout warning because the plot has dense labels. The complete CSV, SVG and PDF remain available; no renderer changes were made. The public `adverse_records.json` filter indexes status strings and contains one injected failed-method record here. Historical `success=false` results (supplied4/6 and developer5/6 full-matrix failures), unresolved grid intervals, float32 `precision_support_loss`, infeasible controls and outside-class resource accounting remain in the complete diagnostics/CSV/archive evidence even when that narrow filter does not list them.

## Obsolete first reconstruction

The first manifest/report (`report-inputs.json`, `report-before-replay/`) was built before the controller-directed durable relocation of a historical implementation report. It was never replayed and remains explicitly obsolete. The document's bytes are unchanged; only inputIDs6/7 differ in the fresh durable manifest. The two initially copied descriptor snapshots were already post-relocation. Byte-exact original descriptors were recovered by the controller and independently verified against the original frozen manifest hashes; the precise sequence is retained in `obsolete-pre-durable-relocation/disposition.json` and `history/historical-adverse/process-report-durability-relocation.json`.

No live input, asset, raw or inventory path in the durable manifest depends on the temporary SDD workspace. No cleanup or evidence commit was performed; the controller's final evidence audit remains pending.
