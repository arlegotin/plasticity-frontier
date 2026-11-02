# Wide-QP numerical repair

This is a numerical refinement under section 11 of the approved full-program design and the user's instruction to continue where a clear improvement path exists. The completed `36ab424` pilot remains immutable and G6 remains `INSENSITIVE_OR_UNSUITABLE`; no protected execution is authorized by this document.

## Evidence and chosen approach

The fresh pilot completed all 180 W0 pairs with resolved tuning, stable variance, adequate centered-reference calibration and feasible resources. It chose T=V=128 and provisionally planned J=37. G6 withheld readiness because two d64 representative resource calibrations retained three failed candidate records.

The bounded diagnosis in `runs/campaign/2026-11-02-full-program/full-pilot-capture/wide-solver-diagnosis.json` distinguishes two causes:

- W1 B03/B04 F: the optional active-set polish was rejected, which vetoed the native iterate. Its retained post-repair slack is zero, stationarity 6.11e-16, complementarity 9.12e-18 and dual gap 5.55e-17. The native success message is retained; its original boolean/status and exact polish-rejection reason were not separately saved. Reproduce that distinction in tests rather than assuming an internal reason.
- W2 B04 Z: native/polish success and original KKT checks pass. Two lower-bound coordinates are about 2.20e-17 while the independent rank oracle implies zero, so the strict relative comparison reports 1.0. The corresponding mathematical lower-bound multipliers are about 0.0900, separating this case from weak-positive support.

Keep the native candidate when an optional polish is rejected **only if native success and all unchanged final feasibility, nonnegative-dual, stationarity and complementarity checks pass**. Retain separate native status, polish validity/rejection details and candidate origin. This explicitly removes the polish veto from the acceptance contract; it does not remove the final quality checks.

For Z only, reconstruct exact zeros belonging to the valid production polish's active lower-bound face with strictly positive multipliers. The justification is complementarity: a positive lower-bound multiplier forces that coordinate to zero on the chosen face. Choose coordinates from the production active-set calculation, never from the independent oracle. Recompute diagnostics after reconstruction/repair and retain its indices/size. No absolute gain cutoff, denominator floor or silent removal of arbitrary weak directions is permitted. The unchanged independent relative rank check must still reject an accidentally erased positive reference gain.

Alternatives are to relax tolerances or replace the full solver with structured prefix enumeration. Tolerance relaxation is excluded. Prefix formulas have been independently derived and remain a fallback if the smaller repair is insufficient; a full solver replacement is not needed by the current trace.

## Binding constraints

Preserve all real-arithmetic estimator objectives, constraints, ordered basis, budgets, populations, candidate families, validation rule, independence units and resource reserves. Keep the existing final original-unit thresholds and the independent Z comparison at relative 1e-6. Preserve exact zeros, representable weak-positive directions and declared support policies. Native failure, invalid/nonfinite iterates, failed KKT or an unresolved independent comparison must not become success merely because a polish was attempted.

Results remain ordinary float64 diagnostics, not rigorous certificates. More valid computed points can change a selected gamma, so source identity and acceptance metadata must expose the change. Historical failures remain failures at their recorded source. Every code/dependency change precedes a newly committed execution identity and genuinely fresh full pilot; do not patch the old G6 verdict or freeze its draft under the repaired source.

## Validation and continuation

TDD must reproduce both retained inputs, native-success/rejected-polish admission, bad native and failed-KKT refusal, valid-face zeros, weak-positive support, ties, zero budget and prior extreme-scale regressions. Use mathematical inputs/eigenvalues as unit regression data, without importing synthetic observation fixtures into scientific commands. Record native/polish provenance and all corrections.

After focused/full tests and independent source review, run a labelled development verification of the same two wide preparations through public execution APIs; verify observation hashes and no new independent units. A successful repair permits a genuinely fresh full G6 invocation on the new source. Source count, accuracy target, precision and all gates stay unchanged. No favorable primary effect signs may choose the repair or schedule.
