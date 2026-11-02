# Wide-QP Numerical Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove two verified numerical admission failures without weakening quality thresholds or changing the mathematical learning methods.

**Architecture:** Treat polish as optional relative to a successful native iterate and the unchanged final KKT gate. Reconstruct the production Z polish's known active zero face before final diagnosis, retaining the independent rank oracle. All raw warnings, corrections and candidate origins remain visible.

**Tech Stack:** Existing locked Python, NumPy, SciPy and unittest; no dependencies.

**Spec:** `docs/superpowers/specs/2026-11-02-wide-qp-repair-design.md`, subordinate to the original approved full-program design.

## Global Constraints

- Work on current `main`, meaningful local commits, no push/merge/worktree or subagents inside workers.
- Preserve estimator objectives/constraints/budgets/basis/population/candidates/tau/precision, independent units and resource reserves.
- Keep all existing final original-unit quality thresholds and Z's independent relative rank comparison at 1e-6. No absolute gain clipping, denominator floor, arbitrary support cutoff or oracle-selected production face.
- Admit a rejected-polish native point only with native success and unchanged final KKT/feasibility/nonnegative-dual checks; retain native status, polish rejection and candidate origin.
- Ordinary float64 diagnostics remain such; no exact/global/confidence claims. Historical source36ab424 data and G6 verdict remain immutable; repaired execution needs its own committed source and fresh full pilot.

## Review Focus

- A rejected optional polish must not veto a genuinely successful, quality-checked native point.
- Native failure or bad post-repair residuals must not be accepted by the new path.
- Z face reconstruction must use valid production active constraints and positive multipliers, preserving independent comparison and genuine weak-positive gains.
- Correction/provenance diagnostics and exact support/resource counts must remain reconstructable across zeros, ties and extreme scales.
- Scientific reruns must preserve old observation hashes/zero new development units and never relabel old G6 or allocate protected keys.

### Task 1: Repair native/polish admission and Z active-face reconstruction

**Files:** Modify `src/prepared_learning/quadratic.py`; modify `src/prepared_learning/frontier.py` only if final acceptance/nonnegative-dual checks require an explicit shared contract; tests in `tests/test_frontier.py` and a focused new `tests/test_qp_numerical_admission.py`; optional minimal README numerical-diagnostic note. Tiny regression inputs may be stored in `tests/fixtures/wide_qp_regressions.json` with origin hashes and no observations/truth.

**Interfaces:** Keep `solve_qp(qp, variable_scale)` and all public method/solver APIs. Diagnostics add separate native success/status, polish validity/rejection details, candidate origin, and active-face reconstruction indices/size. Existing `solver_success` semantics must be explicit and coherent with `solve_frontier` and tuning-bound consumers. No tolerance/config knob is introduced.

- [ ] Write failing tests using the two retained mathematical inputs, with exact source/row provenance. Assert W1 native-success/rejected-polish case is completed only after unchanged final checks; assert W2 Z has exact inactive zeros and passes the existing independent comparison. Add native-failure/bad-KKT, weak-positive and correction-provenance regressions. Watch the failures before source edits.
- [ ] Implement the minimal source repair. Preserve the unsuccessful polish attempt in diagnostics; do not apply its invalid candidate. Reconstruct valid-polish Z lower-bound zeros only from active production constraints with strictly positive multipliers, recompute final diagnostics, and keep independent waterfill validation. Do not replace the solver with the oracle or globally snap gains.
- [ ] Run focused frontier/numerical-admission/tuning-bound tests, then the full `.venv/bin/python -m unittest discover -v` once green; `git diff --check`. Self-review all existing support/zero/extreme/native-failure behaviors and acceptance metadata.
- [ ] Commit the verified repair and obtain independent scoped source review before development execution. Do not run a scientific pilot before that review.
- [ ] Write the source-task report with red/green/full-suite, source/metadata/support evidence, commit and self-review. The controller reviews this deliverable before Task 2's development execution.

### Task 2: Verify both retained wide preparations through public execution

**Files:** Create ignored `runs/campaign/2026-11-02-wide-qp-repair-verification/` captures and append selected findings to `docs/results/2026-11-02-full-pilot-verdict.md` only after execution. No production source edits.

**Interfaces:** Consume Task 1's reviewed clean commit and the exact original full-pilot config/root. Reuse the original W1/W2 preparation config/outer/phases from `pilot._preparation_replication`; preserve original source-observation identities. Produce complete raw evidence, a bounded comparison and actual local numerical/tuning outcomes, not protected evidence.

- [ ] On the clean reviewed source, reconstruct the same two representative wide preparations using the public execution API, original exact Python integer root/config/phases/outer IDs. Do not import tests or fabricate a source population.
- [ ] Capture every result/failure/cost, observation hashes, source/runtime/lock/argv and direct awaited statuses. Compare common pointwise maps to ordinary numerical precision, retaining any changed support/corrections and every failure. This deliberately reuses development observations and adds zero independent units. Preserve source36 originals; do not change cap/threshold/precision or inspect primary effect signs.
- [ ] Write the verification report and selected durable findings with actual hashes/source identity, then obtain artifact-scoped review. If both cases resolve, return the new identity for a genuinely fresh full G6 schedule; otherwise retain the specific failure without speculatively enlarging the patch. No freeze/confirmation or old-G6 verdict relabelling.
