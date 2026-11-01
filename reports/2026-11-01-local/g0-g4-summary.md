# Actual G0–G4 and predeclared frontier grid

All eight public commands returned exit 0 / pass on clean source `536a4003ca402d2e67d51b3bf24da256c3829314`. Total captured command wall time: 4.634523 s. Numerical CPU window was released immediately after grid completion; only metadata inspection and hashing followed.

| Command | Seconds | Result |
|---|---:|---|
| g0-doctor | 0.779583 | exit 0 / pass |
| g1-theory | 0.254891 | exit 0 / pass |
| g2-geometry | 0.263620 | exit 0 / pass |
| g3-reproduction | 2.000221 | exit 0 / pass |
| g4-frontier | 0.291302 | exit 0 / pass |
| g4-exact-F | 0.273541 | exit 0 / pass |
| g4-exact-Z | 0.266673 | exit 0 / pass |
| g4-grid | 0.504691 | exit 0 / pass |

Runtime SHA-256: `811ac6ab70956520cb018d53719f84c374cf8e1504961395d9eaf278b8fb7a7a`. Code SHA-256: `04806bbd50355d1c16986541e6986c37e8f1c8f5b28fff5f2ba8e249772ce89d`. Dependency lock SHA-256: `6e648a5a51d603b8a9fcd4ea85c2d7316fd5b022ab6dcd48b80b021685507fe6`. Full before/after identity, inherited thread environment, input hashes, exact public argv, UTC times, perf-counter durations, stdout/stderr hashes and returned attempt paths are retained in `g0-g4-identity*.json`, `g0-g4-summary.json` and `raw/*.process.json`. Source/runtime/input identity and clean Git state were rechecked after all commands.

Doctor: `/Users/artemlegotin/Documents/git_alter/plasticity-frontier/runs/campaign/2026-11-01-main-objective-v1/doctor/attempt-zfd81a82`. All requested operations passed. Theory: six exact checks passed, with the finite-fixture qualification retained. Adversarial geometry: 605 derivatives, maximum relative error 2.23593269695138e-14; the noncanonical coefficient is 4 versus linear prediction 3.

Source reproduction: `/Users/artemlegotin/Documents/git_alter/plasticity-frontier/runs/campaign/2026-11-01-main-objective-v1/reproduction/attempt-42xa5s38`. All three preserved scripts exited 0. Phase Monte Carlo used 200,000 tasks and all six diagonal QPs succeeded. Verify-plasticity checked 5,600 random null networks and 1,400 constructions; the independent source audit checked 12,600 random null networks, 2,100 constructions and 5,341 complex-step parameter derivatives. Maximum spectral risk residual was 6.661338147750939e-16. Direct B01/B02 residuals and ridge/posterior residuals remain in the full summary. Optional full-matrix diagnostics were not requested. Source internal seeds 4813, 20261031 and 123 remain in the exclusion inventory.

Source stdout/stderr/output JSON files are retained in the original attempt and copied byte-for-byte to `raw/g3-source-*`. Child return codes are those recorded by the public dispatcher. Child argv is explicitly labelled as reconstructed from the committed dispatcher and actual attempt path, rather than a process trace; no child-specific timing was invented.

Exact F and Z exports verified rational primal=dual values `186321/276200` and `81247/115000`. Their raw exports retain complete QPs, active sets, contracts and configuration/fixture hashes. The certified point frontier retains both branches. These finite exact checks do not establish a global theorem.

Predeclared grid: `/Users/artemlegotin/Documents/git_alter/plasticity-frontier/runs/campaign/2026-11-01-main-objective-v1/grid/attempt-q8qgzc3t`; full `frontier.json` and `frontier.csv`, plus `grid-process.json`, are retained. Widths 1/2/3, initial norm caps 0/.2/.4/.6/.8/1/1.2, risk tolerance .0002 and budget tolerance .005 were unchanged; only the copied config's operational output path changed. Adaptive evaluation produced 251 completed points and 502 F/Z branch records, with zero failed points. There are 248 selected and 3 unresolved point selections. All three unresolved intervals are [0, .003125], one at each width, with reason `unresolved numerical selection` and midpoint interpolation diagnostic 1.0375976562504441e-05. They are preserved, not treated as certified or resolved. The CLI process sidecar stdout/stderr matches the external raw capture byte-for-byte.

The summary indexes 57 immutable-at-capture artifacts by path, byte count and SHA-256. Existing old-revision authority files were not overwritten. No fixture promotion, protected empirical evidence, fitting, pilot, freeze, confirmation, source edit or dependency change was part of this G0–G4/grid execution.
