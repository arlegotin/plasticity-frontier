# Source audit for the prepared-learning spec and plan

**Date:** 1 November 2026.
**Scope:** local input identity, source-to-blueprint traceability, and exact development fixtures. This is a planning audit, not a fresh reproduction of the complete source experiments.

## Inputs and preservation

The initial repository had an unborn `main` branch and only three untracked research documents. The first commit preserves their bytes. `.DS_Store` is ignored. No application code, dependencies, experimental outputs, or prior project conventions existed.

| Input | SHA-256 |
|---|---|
| `docs/blueprint.md` | `b866527eaae4cf73991ceb9afda1459691aa0d8a68cfd7f3d7d528f53a2deb7e` |
| `docs/prepared_learning_research.pdf` | `2e0bf090bb2c8504e513b08b14c3bb1ed356dde5d7dd39ebb90f30b281f430b1` |
| `docs/prepared_learning_research_bundle.zip` | `cc94445ed1a15b852ee0accd32daca3de028c1c3b6e936d8d32cfd21273d6b9f` |
| ZIP member `prepared_learning_research/research_note.pdf` | `d21569825103249a352bba5e69f951bcc163dbc748b33df9124ff637920de4f9` |

The two PDF hashes agree with blueprint §12.1. Extracting text with local `pdftotext -layout` and comparing whitespace-separated tokens gives equality. The external PDF and the ZIP PDF are distinct binary assets with the same normalized textual content.

The ZIP has 17 entries, including LaTeX geometry/phase/verification sections, the paper, three numerical scripts, recorded JSON outputs, figures, and literature notes. No source was modified or executed in place. The numerical audit script writes beside itself, so future reproduction must use a copy. Future extraction must validate archive paths, retain a source manifest, and send new outputs to `runs/`.

## Evidence boundaries

| Claim or mechanism | Original evidence | Blueprint / new handoff treatment |
|---|---|---|
| Sharp norm, width, parameter-count formula | PDF Theorem 1; `geometry_section.tex` | Supplied theorem; blueprint §4.1 and design §4.1 |
| Canonical finite simultaneous update | PDF corollary; `geometry_section.tex` | Implement literal gradients and independently compare with compact `Gz` |
| Flat-subspace phase transition | `phase_section.tex`; `results_phase.json` | Six-budget development reproduction; not a protected empirical claim |
| Arbitrary-spectrum isotropic onset | `phase_section.tex` | Preserve scalar `d=1` exception described in blueprint |
| Nonlinear finite step for a deeper null network | `verification_section.tex` | Retain `3h+h^2` counterexample; do not generalize canonical cancellation |
| Width-constrained F/Z reduction and primary richer-spectrum fixture | Blueprint §§4.3–4.6 | Blueprint extension, not a theorem supplied by the original note; require proof obligations and independent verification |
| Estimated-prior advantage and M3 performance | Proposed in blueprint §§5–7 | Unmeasured future work, subject to pilot/protected/resource gates |

Recorded results were read, not newly generated: 5,600 random networks and 1,400 constructions in `results_plasticity.json`; 12,600 random networks, 2,100 constructions and 42 complex-step audits in `audit_geometry_results.json`; 200,000 tasks and six budgets in `results_phase.json`. That last file retains four failed optional full-matrix solver diagnostics. The original records and the blueprint's Linux rerun narrative are historical evidence, not measurements by this planning session.

The selected `python3` had neither NumPy nor SciPy installed. No packages were installed for document preparation. Full numerical reruns and target-machine discovery belong to WP00–WP03. Existing software/hardware/literature links were not freshly researched; later execution must verify its actual environment and must not treat historical versions or novelty assessments as current facts.

## Fresh exact arithmetic checks

A local standard-library `fractions.Fraction` calculation independently evaluated the primary QP coordinates, constraints, gradients, dual values, and objectives. These are small exact fixture checks, not a proof of the global F/Z reduction.

For `s=(1/2,1/4,3/20,7/100,1/50,1/100)`, `c=5/4`, `rho=1/2`, `k=3/5`:

| Check | F | Z |
|---|---|---|
| Coordinates | `(2963,1978,798)/6905` | `(1013,518,194)/2875` |
| Budget constraint row A | `(-1,-1,1)` | `(-1,-1,-1)` |
| Nonzero dual multiplier | `953/27620` | `2383/11500` |
| Exact primal = exact dual | `186321/276200` | `81247/115000` |
| Approximate risk, for readability | `0.6745872556118755` | `0.7064956521739131` |

Both budget slacks are zero; all primal positivity/gap constraints have positive slack and zero multipliers. The multipliers are nonnegative, `l+2Dx-A^T v=0`, and complementarity holds exactly. The exact difference is `1266883/39703750`.

The same calculation checked the source thresholds `4/25` and `8/9`, source risk `677/1025` at `k=2/5`, and width-two Z risk `69/100`. All assertions passed.

The following compact check reproduces the primary arithmetic without numerical dependencies:

```python
from fractions import Fraction as F

s = list(map(F, [".50", ".25", ".15", ".07", ".02", ".01"]))
q = [F(5, 4) * si + F(1, 2) for si in s]
k = F(3, 5)
cases = [
    ([F(2963, 6905), F(1978, 6905), F(798, 6905)],
     [-1, -1, 1], [-2*s[0], -2*s[1], -2*sum(s[2:])],
     [q[0], q[1], sum(q[2:])], F(953, 27620), F(186321, 276200)),
    ([F(1013, 2875), F(518, 2875), F(194, 2875)],
     [-1, -1, -1], [-2*si for si in s[:3]],
     q[:3], F(2383, 11500), F(81247, 115000)),
]
for x, A, linear, diagonal, dual_multiplier, expected in cases:
    assert k + sum(ai*xi for ai, xi in zip(A, x)) == 0
    assert min(x) > 0 and dual_multiplier >= 0
    assert all(li + 2*di*xi == dual_multiplier*ai
               for li, di, xi, ai in zip(linear, diagonal, x, A))
    primal = 1 + sum(li*xi + di*xi*xi
                     for li, di, xi in zip(linear, diagonal, x))
    dual = 1 - dual_multiplier*k - sum(
        (li-dual_multiplier*ai)**2/di
        for li, di, ai in zip(linear, diagonal, A)) / 4
    assert primal == dual == expected
assert cases[0][0][0] > cases[0][0][2]
assert cases[0][0][1] > cases[0][0][2]
assert cases[1][-1] - cases[0][-1] == F(1266883, 39703750)
```

## Defects and implementation consequences

`audit_geometry.py::optimal_network` selects positive gaps using `eigenvalues > lam + 1e-12`. Reading the code shows that for target `diag(1e-14,2e-14)` it retains no excess direction and constructs only `1e-14 I`. Its relative Frobenius error is `sqrt(1/5)`, about `0.4472135955`. This session confirmed the source code path by inspection; it did not run that NumPy script. The new implementation must use explicit support metadata and scale-aware policies.

The spec also makes the zero-metric positive-width count, full-rank Z decomposition, independent head-zero constructor, singular empirical-QP diagnostics, preparation-level inference, and scientific/operational resume identities explicit. These preserve or clarify blueprint requirements; they do not amend its scientific population or endpoints.

## Builder check and document validation

A fresh numerical-software builder reviewer read the design with instructions to report consequential questions that would block planning, excluding decisions explicitly deferred to the builder or pilot. Its response was “nothing important left.” This is a model-based document review, not independent mathematical peer review.

Local checks passed for UTF-8 decoding, relative document links, balanced code/math fences, execution of the exact-fraction snippet above, and unchanged source hashes. Further plan self-review and final document validation belong to the companion plan commit.

The user approved the full-program approach and then the written design before plan preparation. A subsequent standard-library fraction check matched all six recorded analytic source risks exactly up to their stored decimal representation: `542/625`, `493/625`, `677/1025`, `573/1025`, `5/9`, `5/9`. Literal fraction matrix multiplication also matched `3h+h^2` at `h=-2,0,1/3,1,2`.

The plan self-review mapped all approved sections and WP00–WP10 to fifteen tasks, compared shared types and interfaces, checked endpoint-specific inference and the five Review Focus failure classes, and separated implementation commits from the real scientific freeze/confirmation campaign. Corrections made during that review include the zero-target parameter count, an explicit exact-fixture provenance path, B05 target identities, and filtering truth-bearing generator configuration before method calls. These are specification-preserving implementation decisions. No package, test suite, pilot, protected experiment or target-machine benchmark was implemented or run during this documentation task.
