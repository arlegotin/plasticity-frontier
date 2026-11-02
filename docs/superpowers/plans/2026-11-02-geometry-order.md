# Deterministic Geometry Ordering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make learned family diagnostics independent of process hash randomization while preserving strict replay semantics.

**Architecture:** Order family aliases once where `prepared_candidates` builds the prepared-state dictionary. Keep candidate/selected order, all scientific data, strict comparisons and identity/resource policies intact. Historical failed replay remains immutable; new-source scientific execution belongs to the parent full-campaign plan.

**Tech Stack:** Locked Python/NumPy/SciPy, unittest, subprocess fixture confirmation stores and existing native RunStore/replay comparison.

**Spec:** `docs/superpowers/specs/2026-11-02-geometry-order-design.md`

## Global Constraints

Use current main and meaningful local commits; no push/merge/worktree. Source09fe scientific hold is released only after its failed artifact gate. Never mutate old raw files, manifests, seals or source identities. Preserve all candidate grids/selection/tie rules/risks/arrays/unknown fields/tolerances/guards and scientific counts. No favorable Python-hash seed or TaskKey root selection. Root300 is existing audited software-only fixture data. No production root/pilot/freeze/confirmation/benchmark/report/replay belongs to this repair plan; the parent executes fresh source-bound gates afterward.

## Review Focus

- Candidate/selected ordering and associations must survive sorted family alias insertion.
- Separate interpreter hash seeds must exercise the real learned serialization path, including B11:F and B11:Z.
- Strict comparison must still reject changed identity, arithmetic, sequence order and multiplicity.
- All native frozen arrays and cost-sharing aliases must remain equal across software reruns.
- Historical positive G7 and failed G10 stay source09fe; tests are not fresh scientific evidence.

### Task 1: Stabilize producer ordering and prove cross-process replay

**Files:**
- Modify: `src/prepared_learning/execution.py`, only the family-alias loop in `prepared_candidates`.
- Modify: `tests/test_replay.py`, reuse existing tiny confirmation helpers; isolate any child script in test code.

**Interfaces:**
- Consumes: `prepared_candidates(result, config)`, `tests.test_confirmation.tiny`/`execution_manifest`, `confirmation.run_confirmation`, `reporting.compare_run_stores`, `reporting.compare_scientific` and `_scientific`.
- Produces: deterministic eligible family suffix for all later traversals; existing return types/signatures unchanged.

- [ ] Write `test_learned_replay_across_process_hash_seeds`: use a single fixture manifest/config derived from existing tiny learned helper, methods `(B01,B02,B03,B04,B11:F,B11:Z)`, original baseline/joint selector sets and audited root300. Start fresh locked interpreters with predeclared PYTHONHASHSEED strings0..7 before imports. Use existing test-only identity accommodation and real native preparation/publication. Preserve source inputs across children. Assert complete family IDs/specified sorted suffix, candidate prefix and selected-role boundaries, same candidate/source/selected associations, and full native strict store/array comparison against the first process. Test outputs are temporary software fixtures.
- [ ] Run that focused regression before the fix and retain the observed failing order/comparison evidence; do not alter seeds to hide a failure. If existing tiny helper needs explicit matching config/selector fields, keep changes local to this test and its declared six-method fixture.
- [ ] Change the producer family loop to iterate `sorted({r['method_id'] for r in ledger})`. No downstream sorting/field omission/comparator edit.
- [ ] Run the new regression and focused `tests.test_replay`, `tests.test_confirmation` checks. Retain full passing logs and observed record/array counts.
- [ ] Add/extend one focused strict-comparison negative test over complete geometry records: gamma change beyond tolerance, method identity change, dropped row, duplicated row, selected candidate change and positional permutation all raise. Reuse existing arithmetic/asset tests rather than duplicating them.
- [ ] Run one full `.venv/bin/python -m unittest discover -s tests` on repaired code. Fix only concrete regressions. Do not run production scientific commands or allocate keys.
- [ ] Self-review the full diff and commit source/test changes with message `fix: stabilize learned family record ordering`. Report TDD red/green/full-suite evidence, commit IDs, preserved scopes and any concern to the owned repair workspace.
