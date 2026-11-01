# Prepared learning research

This repository preserves the supplied research inputs and builds a local CPU float64 research environment. The original ZIP and PDFs are inputs: do not modify them or run the source scripts in place. `doctor` stages a validated read-only source snapshot under `source_bundle/`; experiments must use copies and write outputs under `runs/`. Optional source assets do not gate equation-based fixtures.

Bootstrap on a native arm64 macOS Python compatible with the locked wheels (the recorded environment used Python 3.14.6):

```sh
python3 -m venv .venv
.venv/bin/python -m pip download --only-binary=:all: --dest wheelhouse -r requirements.lock.txt
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -v
.venv/bin/python -m prepared_learning doctor --output runs/environment
```

Verify the downloaded wheels against `wheel_manifest.json`; the wheel files themselves are local and ignored. `installer-report.json` records initial installation, and `licenses/` retains wheel notices. Lock and wheel hashes describe this machine's bootstrap; a different Python/platform needs a newly resolved, recorded environment. Hardware discovery records non-identifying command outcomes; unavailable fields remain unknown. The read-only scalar commands `sysctl -n hw.memsize`, `hw.logicalcpu`, and `hw.physicalcpu` may require sandbox escalation. No serial number or hardware UUID is queried. The doctor checks required CPU operations and emits a nonzero status on failure. Each doctor invocation creates a unique attempt directory beneath the requested output root; the JSON response includes `attempt_path` for its reports, gate decision, and source manifest. Earlier attempts and existing evidence are preserved. Protected scientific runs require committed code and a separately frozen scientific identity; these smoke tests are not a scientific campaign.

Learned candidate tuning declares the fixed, source-versioned policy `min_marginal_loss_se_v1` in `public_method_config.tuning_resolution_policy` and the candidate ledger. Candidates are ranked by mean validation query MSE on the same independent source tasks. Each task contributes the mean of its A-to-B and B-to-A query MSEs. For neighboring candidates a and b around a sampled group minimum, define `SE_a = sample_sd(task_losses_a) / sqrt(V)` and `tau = min(SE_a, SE_b)`. The local objective check passes when `abs(mean_loss_a - mean_loss_b) <= tau`. Both marginal SEs and tau have query-MSE units. Paired loss-difference SE remains a ranking diagnostic; it does not determine this objective-level tolerance. This is an explicit implementation convention under blueprint §5.7, not its uniquely prescribed uncertainty statistic or a user-adjustable stopping shortcut.

Every shrinkage group includes gamma 0 and 1 and must evaluate an interior candidate before local resolution. An initial grid containing an interior candidate satisfies that probe; an endpoints-only grid requires evaluating .5 in the next round. Each valid neighboring interval of every method/base/tuning/rank group's sampled minimum receives the same check, including endpoint minima and losing groups. Local ridge refinement uses the same tolerance, while a best finite ridge-grid boundary still requires extension. Fewer than two tasks or missing/nonfinite uncertainty cannot resolve tuning; a positive gap with zero tolerance remains unresolved. A zero sampled gap with valid zero uncertainty can pass its local check after the required probe. Exhausting `max_refinement_rounds` never supplies a resolution certificate.

The smaller marginal SE is stricter than the larger marginal SE, but **is not generally conservative relative to paired precision**. Adding the same task-varying, candidate-independent component to every candidate loss can change stopping while leaving rankings, paired gaps and paired SEs unchanged. Common task noise can permit objective resolution despite a detectable ranking difference. Keep the loss definition fixed; do not add or subtract such components to obtain resolution. Adaptive reuse of the same validation tasks creates no additional independent observations. This finite local convention provides neither a simultaneous confidence guarantee nor proof of global optimality, and can miss an unsampled valley or discontinuity. Independent preparations and protected evaluation remain necessary for scientific inference.

Completed candidate rows store `validation_loss_se` and a policy version. Small `tuning_checks` on the relevant group-best row retain pair IDs, interval, marginal and paired SEs, absolute gap, tolerance, inward-probe evidence, and decision; `ridge_boundary_check` records finite-boundary extensions separately. All attempted and failed candidates and their costs remain in the ledger. The changed selection procedure requires a new committed source identity and fresh pilot evidence; historical fits, costs and unresolved statuses retain their executed revision.

Report reconstruction consumes explicit, ordered input descriptors. It does not discover a favorable subset of `runs/`, reuse `gate_decision.json`, or read benchmark coordinator summaries as measurements. Each confirmation descriptor names the original frozen protocol and its immutable RunStore. Paths inside a descriptor resolve relative to that descriptor. For example, save this structure as `runs/campaign/confirmation-input.json`, substituting the **actual** run directory printed by `confirm`:

```json
{
  "schema_version": 1,
  "kind": "confirmation",
  "label": "declared primary comparison",
  "run_path": "../confirmation/ACTUAL_RUN_ID",
  "protocol_path": "../../protocol/confirmation.json"
}
```

Other descriptor kinds are `benchmark` (`run_path` containing `inputs.json`, `requests/` and `blocks/`), `diagnostic`, `exact`, `source`, `forecast` (each with `raw_path`), and `not_run` (with `reason`). A diagnostic records its CLI argument list in `command` and lists every input file in `assets`. Use the configuration actually executed. Stdout-only commands use their complete stdout JSON as `raw_path`; robustness and frontier grids use the full-artifact route below. Include every failed source attempt, every family/shift arm, both F/Z branches, and unsuccessful controls. An unperformed natural arm must explicitly state that no official data/acquisition receipt exists. Source records and forecasts are retained but are not re-executed or promoted into evidence. `fixture: true` permanently prevents a test descriptor or replay receipt from satisfying production G10. A fixture can test clean replay only when its actual frozen protocol, committed source, runtime and assets pass the same strict checks.

The public exact export produces an independently verifiable rational QP, active set, executed configuration and fixture hashes, and full claim contract. Run it separately for F and Z using the committed normative fixture configuration:

```sh
.venv/bin/python -m prepared_learning export-exact --config configs/exploration/frontier.json --branch F > runs/campaign/exact-F.json
.venv/bin/python -m prepared_learning export-exact --config configs/exploration/frontier.json --branch Z > runs/campaign/exact-Z.json
```

Each `exact` descriptor names that raw JSON, records `command: ["export-exact", "--config", "configs/exploration/frontier.json", "--branch", "F"]` (or Z), and inventories both `configs/exploration/frontier.json` and `tests/fixtures/exact_cases.json` in `assets`. Descriptor file paths are relative to its directory; command arguments are relative to the repository root. For an explicit claim, set `evidence_level` to `rigorous_certificate`, `statement` to `exact_rational_fixture_kkt_verified`, `input_ids` to the actual descriptor index, and copy the exported `contract` fields into that claim. Admission rebuilds the exact envelope from the inventoried configuration and committed normative fixture, then rechecks the received integer-pair rational KKT; altered constants, active sets, contracts, configuration, or float relabeling are rejected. Exact replay uses no arithmetic tolerance. `frontier --certify` remains a numerical frontier report with exact branch checks; `export-exact` is the complete ledger/replay envelope. Finite fixture verification establishes neither a global theorem nor production G10.

For robustness and frontier grids, capture the process record with committed CLI code:

```sh
.venv/bin/python -m prepared_learning robustness --manifest runs/campaign/robustness-manifest.json --output runs/campaign/robustness --process-record runs/campaign/robustness-process.json
.venv/bin/python -m prepared_learning frontier --config runs/campaign/grid.json --output runs/campaign/grid --process-record runs/campaign/grid-process.json
```

Use the full scientific file as `raw_path`: `robustness.json` under its output directory, or `frontier.json` under the printed `attempt_path`. A robustness descriptor saved in `runs/campaign/` has this shape:

```json
{
  "kind": "diagnostic",
  "raw_path": "robustness/robustness.json",
  "process_path": "robustness-process.json",
  "output_artifact": {"filename": "robustness.json"},
  "command": ["robustness", "--manifest", "runs/campaign/robustness-manifest.json", "--output", "runs/campaign/robustness"],
  "assets": ["robustness-manifest.json"]
}
```

For a grid, use `filename: "frontier.json"`, the actual attempt's `raw_path`, its process record and configuration asset, and `command: ["frontier", "--config", "runs/campaign/grid.json", "--output", "runs/campaign/grid"]`. The recorded command excludes the operational `--process-record` option. The sidecar preserves stdout, stderr, exit status and command, and refuses overwrite before execution. Replay inventories the sidecar and full file, replaces only the output directory with a fresh isolated destination, compares the full records and summary/status, and retains freshly measured output. Missing, stale, escaped, symlinked or changed artifacts fail replay. Failed grid points and adverse/descriptive robustness rows remain included; an equal failure replays as a recorded failure, never successful scientific support.

Frontier replay accepts only separated, ordered arguments. Without an artifact selector, the supported command is `["frontier", "--config", PATH]` with an optional trailing `"--certify"`, and the configuration must be a single point. Grids require the artifact route above. Combined `--config=PATH`, abbreviated flags, duplicate options, reordered options and extra flags are refused before relocation or producer dispatch, even where the interactive CLI parser accepts them.

For a finalized benchmark the frozen configuration in raw `inputs.json` supplies the executed source identity and calibration location. Preserve the calibration directory and its original JSON as well as all campaign raw worker blocks. Timing summaries and Pareto comparisons are rebuilt from workers; calibration schedules are checked against the raw pilot blocks. Supplied resident-input and exploratory benchmarks remain exploratory. Repeated timing blocks measure fixed selected assets; they do not add independent risk tasks or preparations.

After all implementation/dependency commits, execute the reviewed campaign's actual commands. G6/G7/G9 have not been established merely by installing this package or running its tests. The final checkpoint commands are:

```sh
.venv/bin/python -m prepared_learning analyze --manifest protocol/confirmation.json
.venv/bin/python -m prepared_learning report --manifest protocol/report_inputs.json --inputs runs/campaign/confirmation-input.json runs/campaign/benchmark-input.json runs/campaign/natural-not-run.json --output reports
```

The `--inputs` order is preserved and hashed. Add the actual diagnostic/source/family descriptors to that argument list; missing files fail instead of being silently omitted. Duplicate confirmation identities are rejected, including relocated technical replays. Reconstructing a report does not pool separate inputs into a larger sample. Omit `--inputs` on later reconstructions to use the frozen inventory. Input/raw corruption or a changed scientific identity must be investigated; never rehash an altered protected run to make it pass. Descriptor JSON, small manifests and report artifacts may be committed after protected execution finishes. Preserve the earlier **executed** commit in every run; the later evidence commit is not the execution revision.

Report and replay admission checks run before opening a RunStore, parsing raw JSON, loading arrays or serializing the input inventory. The default reserve is half the currently available memory; set `PREPARED_REPORT_MEMORY_BYTES` to a positive byte count to impose a smaller local cap. Confirmation raw/adverse tables stream one verified block at a time, preserving complete candidate records in `raw_records.json` and the `record_json` column of `raw_records.csv`. Analysis retains declared scalar outcomes and scientific keys. It does not retain every learned-preparation snapshot. Benchmark and diagnostic ledgers remain in memory under an aggregate admission bound.

The recorded memory estimate includes a 64 MiB base, 256 times the largest confirmation JSON block plus retained non-confirmation JSON, 16 times the largest array file, and inventory/scalar allowances. Inventory allowance is 4,096 bytes plus 16 times the encoded path length per file; an already supplied manifest is charged 16 times its Python object size. Scalar allowances are 8,192 bytes per declared task and preparation, another 2,048 bytes per task/family contrast, and 512 bytes per source key. Replay doubles the JSON/inventory component. Native JSON parsing has its own 64-times-file-size guard before allocation. Every admission records actual file sizes, the offending largest path, declared working set and live reserve. An over-limit CLI/API report or replay returns `LOCAL_RESOURCE_LIMIT` and writes `resource_limit.json`; lower-level table/input builders raise `ReportingResourceLimit`. Any partial output is marked incomplete. Retry identical inputs with enough local reserve; do not reduce frozen J, omit candidates or truncate evidence.

Prepare the replay manifest from the same ordered inputs, then recover and re-execute on this laptop:

```sh
cp protocol/report_inputs.json protocol/replay.json
.venv/bin/python -m prepared_learning replay --manifest protocol/replay.json --output runs/clean_replay
```

`replay` verifies the recorded revision's code and lock hashes, checks every wheel against that revision's `wheel_manifest.json`, creates a temporary local clone at the recorded commit, restores explicitly listed assets byte for byte at their repository-relative paths, and reconstructs `source_bundle/` from the preserved ZIP and `source_manifest.json`. It creates a fresh venv using the recorded Python executable, installs only verified local wheels, installs the project editable with `--no-build-isolation --no-deps`, and runs `pip check`. It restores recorded thread environment values before numerical imports. The primary branch remains unchanged; no second machine, remote fetch or development worktree is needed. Source, runtime, lock, population and prior checks remain strict. Mixed executed revisions require separate replay manifests and receipts, never an identity rewrite.

Keep `wheelhouse/`, the original Python installation, original raw run directories and all explicit assets locally. To restore missing wheels, use the download command above with the recorded source's lock, then verify hashes before replay. Preserve the supplied PDF, blueprint, ZIP and source-manifest bytes; never edit the read-only snapshot. Replay refuses uncommitted scientific source, missing assets, fixture promotion, unresolved operational schedules and identity changes. A restored source must already contain the replay implementation. All plotting uses Matplotlib's Agg backend with a temporary configuration directory. Plotting wheel versions, PyPI SHA-256 checks and license files are recorded in `licenses/plotting-wheel-verification.json`; core NumPy/SciPy builds remain unchanged.

Replay compares exact keys, counts and frozen asset hashes plus float arrays/metrics with the manifest's default `rtol=1e-10`, `atol=1e-12`. Confirmation comparison uses a temporary SQLite disk index with a 2 MiB cache and reads one record pair at a time; arrays use memory maps and chunks of at most 65,536 elements. Index admission requires eight times both stores' raw JSON bytes plus 64 MiB to fit within half the temporary filesystem's free space. Insufficient disk space also returns `LOCAL_RESOURCE_LIMIT` before building the index. Durations and memory measurements are freshly recorded, not required to match bitwise, and never pooled with original observations. The temporary environment and index are deleted after execution; commands, installation logs, replayed runs and `runs/clean_replay/replay.json` remain. Attach that receipt to the report manifest using its actual hash:

```sh
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
path = Path('protocol/report_inputs.json')
manifest = json.loads(path.read_text())
receipt = Path('runs/clean_replay/replay.json').resolve()
manifest['replay']['receipt'] = {'path': str(receipt), 'sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}
path.write_text(json.dumps(manifest, indent=2) + '\n')
PY
.venv/bin/python -m prepared_learning report --manifest protocol/report_inputs.json --output reports
```

`reports/results.md`, `claim_ledger.csv`, raw/family/adverse-outcome tables, and standalone SVG/PDF figures are rebuilt together. Every figure has its underlying CSV, units, analytic/sample labels and independent sample counts. Empty panels say `not_run`. Exact rational fixture claims require rechecking the raw rational QP and naming its full contract; float64 diagnostics cannot become certificates. Protected and practical claims require their corresponding executed evidence and exact architecture/population/baseline/metric/resource contract. Direct-map equality, ridge/EB wins, ties, negative or zero-crossing intervals, failed candidates and unsupported moments remain visible. The G10 decision records whether matching clean replay exists; it is distinct from the sign of any scientific comparison. The report revisits all twenty blueprint §11.2 challenges and explicitly limits unresolved claims.

Frontier figures consume the grid command's `metadata`/`records` artifact and retain widths, norm caps, both branches, failed endpoints and unresolved intervals in the underlying CSV. Executed records with unavailable objectives are labelled as such. Historical source/diagnostic failures remain in the report and do not veto unrelated scoped claims. The claim ledger identifies unavailable supporting inputs; an asserted scientific claim cannot use failed evidence, and an invalid reconstructed comparison cannot establish G10. Matching nonfixture replay is still required for completion, regardless of whether a valid scientific outcome is positive, negative or inconclusive.
