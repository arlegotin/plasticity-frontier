# Native Phased Replay Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete native replay under available RAM through correct phase admission and object lifetimes, preserving every scientific check and the full native receipt.

**Architecture:** Separate native inventory validation from actual semantic consumers. Isolate each input's temporary objects, account for all caller/map/receipt/writer retention, and keep existing paired and SQLite comparison algorithms.

**Tech Stack:** Locked local Python/NumPy/SciPy, standard-library unittest/weakref, existing JSON/SQLite/file-backed execution.

**Spec:** `docs/superpowers/specs/2026-11-02-phased-replay-memory-design.md`

## Global Constraints

Currentmain/local meaningful commits; no push/merge/worktree. Preserve original assets/history, strict source identity and every full scientific record. Keep native64/semantic256/array16/object16/base64MiB/half/lower-only-cap, copies2 on native validation/real pairs, and existing copies1 SQLite block comparison. No counts/grids/caps/h/alpha/budgets/inference/producer algorithm change, source rebinding, altered comparator, receipt aggregation or reduced evidence. Use existing deterministic test roots; no new production science. Fresh source-bound program belongs to the parent plan only after this code/review/audit closes.

## Review Focus

- Diagnostic objects surviving into later consumers must be collectible; accumulated results remain fully charged.
- Inventory validation must not relabel actual semantic comparison work as native parsing.
- Caller state and growing protected key/path/hash maps must be reserved before allocation without repeated whole-store scans.
- Actual diagnostic output and final receipt/writer growth must refuse under unchanged native guards before large parsing.
- Full input coverage, mutations, strict comparisons and actual G10 attachment must remain fail-closed.

### Task 1: Correct native phase admission and lifetime boundaries

**Files:**
- Modify `src/prepared_learning/reporting.py`: native read/input verification, protected comparison retention, receipt/report writer admission.
- Modify `src/prepared_learning/_replay.py`: coordinator retention, per-entry lifetime helper and diagnostic file-backed consumers.
- Test `tests/test_replay.py` and `tests/test_reporting.py`; extend existing confirmation/benchmark tests only where their public contracts need direct coverage.

**Interfaces:**
- Preserve `_guard_inputs(manifest, *, replay=False, retained_bytes=0)`; replay=True is dedicated native inventory validation, report=False semantics unchanged.
- Extend `_verify_inputs(manifest, *, replay=False, retained_bytes=0)` and `_read(path, *, retained_bytes=0)` with default-compatible caller reservations.
- Extend `compare_run_stores(left, right, *, rtol, atol, retained_bytes=0)`; strict algorithm/return science unchanged, caller/map/buffer retention added.
- Extract `_replay_entry(entry, destination, root, replay, *, retained_bytes)` where retained_bytes is a zero-argument current-byte callback and return is only the existing result/reference record. `worker` retains the complete manifest/results and appends that return.
- Preserve CLI, receipt schema/full input hash and actual native G10 attachment. Additional resource evidence may describe the real admitted phases.

- [ ] Write failing regression tests for a full ordered manifest where old global semantic two-copy admission fails but native validation and every true consumer fit. Assert all exact constants/copies/half/lower-cap policies, full input coverage, identical keyed science/duplicates/arrays and one complete receipt.
- [ ] Add fail-closed tests for oversized native inventory, actual diagnostic pair/output, protected caller/maps, unchanged G9 pair, accumulated result reserve and final receipt/writer. Assert refusal precedes the offending read/allocation and no matched receipt is published.
- [ ] Add weakref lifetime tests for independently allocated expected/actual/normalized diagnostic graphs across consecutive inputs and G8→G9; retained manifest/results/protected hashes must still affect the next admission. Retain strict existing mutation/order/type/count/candidate/array/source/canonical negatives.
- [ ] Run the new targeted tests and record expected RED failures on the current code. Do not count an unrelated error as the regression.
- [ ] Implement the interfaces and phase contract in the two existing modules. Reuse existing file-backed subprocess/encoding patterns; preserve full process JSON/raw artifacts, avoid large capture_output allocations, and reserve actual file bytes before decoding. Scope raw bodies/projections/protocol/config locals to one helper call.
- [ ] Account for caller and prospective map/hash/key buffers in SQLite comparison while preserving its one-block algorithm and temp8×combinedJSON+64MiB guard. Use a bounded metadata pass/cached scale as needed; do not rescan the entire store per record or weaken comparisons.
- [ ] Guard coordinator final revalidation/worker-result read/receipt write and report receipt read/write with all actual live native structures. Keep complete tables/ledger/sections; do not add a no-longer-live scientific block to writer state or omit a still-live object.
- [ ] Run ` .venv/bin/python -m unittest tests.test_replay tests.test_reporting tests.test_confirmation tests.test_benchmark` and resolve only task findings. Then run `.venv/bin/python -m unittest discover -s tests` once and record actual result/wall. Re-run broadly only for new code/failure evidence.
- [ ] Inspect the diff for unchanged producer/comparator/source contracts and complete raw outputs; commit the meaningful source/test correction. Report RED/GREEN commands, counts, actual coverage, identities, retention arithmetic and remaining empirical limits.

The root supplies a scoped spec/quality review and one final whole-branch review, then archives the owned software audit and commits context. Do not run fresh scientific producers or select a held scientific source from this task; the parent full-program controller performs legitimate new-source admission/G6/G9/protected/report/replay next.
