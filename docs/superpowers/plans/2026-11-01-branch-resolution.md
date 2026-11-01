# Branch-aware Local Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the diagnosed discontinuous-selector stopping failure using a versioned numerical improvement bound while preserving fixed-gamma estimators and scientific evidence standards.

**Architecture:** Add a focused source-only diagnostic module, called only for otherwise unresolved local shrinkage comparisons. It computes noneligible branch probes and an interval lower envelope from fixed-family sensitivity, validation convexity and a declared uniform solver-error model. The existing adjacent-gap rule, tolerance, ranking and unsupported-family behavior remain available.

**Tech Stack:** Existing locked Python, NumPy and SciPy; float64 numerical diagnostics; unittest.

**Spec:** `docs/superpowers/specs/2026-11-01-branch-resolution-design.md`, subordinate to the approved full-program design.

## Global Constraints

- Work on the current `main` branch; make meaningful local commits. Do not push, merge, publish or create a worktree.
- Preserve all fixed-gamma estimators, B04's prior-objective branch selection, candidate eligibility, population, endpoints, inference, precision target and resource reserves.
- Preserve tau=`min(left_marginal_loss_se,right_marginal_loss_se)`, independent task units, paired diagnostics, initial interior probes, all fixed-family groups, resource ties, ridge boundaries, failures and refinement caps.
- Extra branch losses/gradients are noneligible resolution probes using the same observed validation tasks. They cannot enter ranking, alter B04's prior-risk branch selection, or enlarge tau.
- Ordinary float64 diagnostics remain `float64_diagnostic`; they are not rigorous enclosures or exact certificates. Fail closed when margins cannot be distinguished from numerical error.
- Version the source policy as `min_marginal_loss_se_branch_envelope_v2`. Historical v1 artifacts keep their identities and unresolved verdicts.
- No protected keys or confirmation before reviewed source, full G6 evidence and a valid freeze. Reused exploratory observations add zero independent preparations.

## Review Focus

- A lower-loss unselected branch must never become the eligible incumbent or enlarge tau.
- Fixed basis/noise/width/base witnesses and full-gain F tail multiplicity must prevent unsupported sensitivity claims.
- Endpoint and unsampled successful-solver errors both enter the allowance; finite KKT output alone must not become an exact certificate.
- Zero/missing uncertainty, failed probes, underflow and unresolved comparisons must not receive successful resolution by default.
- Existing hidden-interior, all-family, ridge-boundary and failure/cost behavior must survive the new route.

### Task 1: Implement and review the numerical branch-envelope route

**Files:**
- Create: `src/prepared_learning/tuning_bounds.py`
- Modify: `src/prepared_learning/selection.py`
- Modify: `README.md`
- Create: `tests/test_tuning_bounds.py`
- Modify: `tests/test_tuning_resolution.py` only where version metadata changes or integration coverage belongs.

**Interfaces:**
- Consumes: current candidate row dictionaries (including asset, prepared_state, validation_task_losses and solver_diagnostics), observed validation `SourceTask`s, and authorized public method config.
- Produces: `branch_interval_check(left: dict, right: dict, incumbent: dict, validation: tuple, config: dict) -> dict`. Return a JSON-compatible diagnostic with `status` (`resolved`, `unresolved`, `unsupported`), `certificate_level`, local interval, method/base/basis witnesses, branches, noneligible probe records, error model/constants, incumbent ID, unchanged tau and improvement bound. It never mutates ranking or fitted estimator state.
- `select_method` attaches the returned diagnostic to the relevant existing tuning check only when its ordinary shrinkage gap asks to refine; a supported successful bound changes that check's decision to `objective_resolved_by_branch_envelope`. It does not add a winning candidate. Unsupported/nonpassing checks continue existing refinement.

- [ ] **Step 1: Write failing behavior tests.** Use actual small branch QPs and observed source arrays. Assert the validation quadratic equals the mean of both literal query MSE directions; F complement column matches floor-plus-increments; real B04 F/Z switching keeps the same branch at every evaluated gamma while the diagnostic bounds possible improvement. Test B02/Z and B03/F supported paths, a conservatively unresolved unavailable lower-loss branch, unsupported full-width Z, basis/noise/config drift, completed-status but invalid numerical prerequisites, missing/nonpositive tau, extreme/nonfinite arithmetic and probe cost/eligibility. Retain an independent dense-gamma comparison as a falsification check, never as the mathematical proof. Assert selected candidate/tau unchanged by noneligible probes and all prior regressions remain applicable.
- [ ] **Step 2: Run focused unittest discovery and record the expected missing-feature failures before implementation.** Use `.venv/bin/python -m unittest discover -s tests -p 'test_tuning_bounds.py' -v`.
- [ ] **Step 3: Implement the helper and integration.** Use the full-gain K in the spec and the validation supporting hyperplane in matching QP coordinates. B04 uses both existing retained frontier branch solutions; B03 needs F; B02 uses its actual Z gains and separately diagnoses that QP without changing its solve/map. Refuse unsupported cases rather than inventing probes. Validate both endpoint provenance witnesses. Build conservative numerical allowances from a reviewed uniform residual/complementarity/feasibility model, checking finite nonnegative multipliers and all required successful-solve thresholds; retain the model and floating qualification. Anchor and unsampled allowances must both be present. Record elapsed probe cost without increasing labels/independence. Store compact reconstructable probe diagnostics, not source arrays or duplicated full ledgers. Keep `selection.py` focused on orchestration.
- [ ] **Step 4: Run focused tests and the full `.venv/bin/python -m unittest discover -v` suite once green.** Record exact commands, counts and any failures. Run `git diff --check`.
- [ ] **Step 5: Update README/public policy metadata and commit.** Explain local scope, unchanged tau, ordinary float qualification, conservative nontermination and historical identity separation. Commit with a concrete fix subject; include red/green/full-suite evidence in the task report. Obtain independent task review before Task 2.

### Task 2: Verify the repair on the existing exploratory cohort

**Files:**
- Create: `runs/campaign/2026-11-01-main-branch-envelope-v2/` (ignored raw evidence).
- Create: `docs/results/2026-11-01-branch-resolution-followup.md` (selected durable findings).
- Modify: `protocol/decisions.jsonl` and `protocol/evidence_ledger.csv` by append only.

**Interfaces:**
- Consumes: reviewed committed Task 1 source; exact config `runs/campaign/2026-11-01-main-objective-v1/source-cost-probe-cap8-config.json`; original cohort `source-cost-probe-cap8/pilot-04993ecbd02f2de0c6c9` under that same historical root.
- Produces: a new captured public `pilot` command, complete raw attempts and a comparison of observation/provenance hashes and local resolution diagnostics. The unchanged root `202611011007` is deliberately reused development data; independent-new-preparations count is exactly zero.

- [ ] **Step 1: Prepare and validate capture metadata on a clean committed checkout.** Copy the exact config without changing its source counts, spectrum, population, root, methods, tolerance or cap. Capture argv, config SHA, source/runtime/lock identity, start/end/status/stdout/stderr and current memory/disk reserves. Do not write into old sealed scopes.
- [ ] **Step 2: Execute `.venv/bin/python -m prepared_learning pilot --config <copied-config> --output runs/campaign/2026-11-01-main-branch-envelope-v2/reused-cohort` with durable file-backed stdout/stderr.** Respect its existing max_seconds=2400 and local guards. No concurrent source edits or commits during execution.
- [ ] **Step 3: Compare all source keys/observation hashes and the known outer-12 group.** Report actual resolved counts, candidate/probe costs, failed/unresolved cases, unchanged pointwise maps on common gamma points, and the diagnostic's numerical margin. Never pool this rerun with old data as additional independent evidence. Do not use primary effect signs to choose settings.
- [ ] **Step 4: Append the actual continuation decision and commit durable findings.** A successful repair permits a separate bounded-memory/full-G6 continuation, not protected execution. If the envelope fails, retain its constants and counterexample and investigate the specific cause without lowering tau or silently enlarging caps. Obtain artifact-scoped review; continue the original staged program where its gates permit.
