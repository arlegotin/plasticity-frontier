# Mathematical review: branch-aware numerical resolution

Two read-only investigations and a separate numerical-contract review support the [follow-up design](../specs/2026-11-01-branch-resolution-design.md), conditionally on its fixed-family and numerical prerequisites. These reviews ran no tests, fits or new scientific observations. They establish neither actual cohort resolution nor a population advantage.

## Reviewed inequalities

For a fixed ordered basis, target n, noise estimate, budget and branch polytope, let `c=(n+1)/n`, `rho=(1+sigma_hat_squared)/n>0`, and `k=(B/(L-1))^(L-1)`. The branch objective has full-gain curvature at least `2*rho`. Variational inequalities imply the safe sensitivity constant

`K = max(1, abs(c*k-1)) * ||1/d-s0||_2 / rho`.

The F floor is repeated d-j times in the full gain vector, where j=min(width-1,d-1). It must retain that multiplicity. Z is the actual nonnegative trace-budget simplex, including at full width. The initial implementation may decline the full-width Z case to avoid a separate canonical-floor mapping qualification.

For observed validation, the existing two-direction query MSE is a convex quadratic in matching branch coordinates. Its supporting hyperplane provides a lower bound at any anchor. A normalized F coordinate vector `(a_1,...,a_j,sqrt(d-j)*b)` has exactly the full-gain Euclidean norm. Alternatively, an unweighted compressed gradient norm is conservative because compressed-coordinate distances are no larger than full-gain distances. The prediction design must use the actual floor-plus-increments map rather than silently assuming an exactly orthogonal floating basis.

For an interval with width delta, an accepted computed branch output is at distance at most

`K*delta + E_anchor + E_uniform`

from an accepted endpoint anchor, provided E uniformly bounds the distance between a successful computed solve and the ideal optimizer. Thus an endpoint loss minus a gradient-norm upper bound times this distance supplies a branch lower bound, after evaluation/model arithmetic allowances. Take the maximum valid endpoint lower bound within each branch, then the minimum across the branches B04 may select. Compare against an eligible actually selected incumbent using the unchanged tau. Extra probes cannot become candidates or affect tau.

The uniform solver-error bound was independently checked even for a slightly infeasible iterate. In original QP units, suppose stationarity infinity norm <=epsS, absolute complementarity infinity norm <=epsC, multipliers are nonnegative, and each slack >=-epsF. Let p be QP dimension, M the number of inequalities, e a distance-to-feasibility bound in full gains, and G bound the objective gradient at the optimum. Then

`E = [sqrt(p)*epsS + sqrt(p*epsS^2 + 8*rho*(M*epsC + G*e))] / (4*rho)`.

Proof: writing delta for the full-gain optimizer distance, the residual identity and feasible optimum give `f(x)-f* <= sqrt(p)*epsS*delta + M*epsC - rho*delta^2`. A feasible repair witness, the variational inequality at the optimum and the gradient bound give `f(x)-f* >= rho*delta^2-G*e`. Combining yields the displayed quadratic root. For F, the normalized residual only decreases relative to the raw floor-coordinate residual; no extra tail multiplicity belongs on sqrt(p).

Valid conservative repair bounds are:

- Z: `p=min(width,d)`, `M=p+1`, `e <= (sqrt(p)+p+1)*epsF`.
- F: `p=j+1`, `M=j+2`, `e <= [sqrt(d+3*j)+sqrt(d)*(2*j+1)]*epsF`. The j=0 case is covered conservatively.

A safe uniform gradient choice for trace-one spectra is `G=2*(1+(c+rho)*sqrt(d)*k)`. Interval endpoint maxima can tighten spectrum-based constants, provided they cover the interval. The production completed thresholds are epsS=epsC=1e-9 and epsF=1e-10*max(min(k,1),1e-300), before arithmetic allowances. Solver repair intent does not justify e=0: the actual completed gate permits small negative slack.

## Conditions and limits

The mathematical inequalities are exact-arithmetic statements. Applying computed residuals, norms, multipliers, losses and SEs requires a declared scale-aware numerical error model. Ordinary float64 padding and consistency checks support a numerical diagnostic; they do not create a rigorous enclosure. Anchor checks alone do not bound unsampled successful outputs. Reject unsupported, failed, inconsistent, nonfinite, underflowed or numerically ambiguous comparisons. Do not enlarge tau.

This route remains local to the checked intervals. It provides no global optimum or confidence guarantee. It does not guarantee solver success throughout a continuum or finite termination. A validation-good branch that prior risk never selects can keep the lower envelope loose indefinitely. Zero/missing uncertainty and failed candidates retain their existing treatment. Reusing source-validation observations creates no new independent evidence.

## Retained full investigation records

The complete reports remain under `runs/campaign/2026-11-01-main-objective-v1/followup/`:

- `branch-bound-independent-review.md`: fixed-family sensitivity, validation normalization, counterexamples and coverage requirements.
- `branch-bound-numerical-review.md`: independent proof of the solver-error root and repair bounds, and minimum numerical contract.
- `branch-resolution-investigation.md`: supporting derivation and source-case arithmetic; SHA-256 `4a66f1606cfd696c977e01d7de9737d1c3d71ed27a76afe17ed1a12f1fde0452`.

The recorded source case requires an actual validation-gradient calculation before predicting whether the diagnostic passes. No such pass was claimed by these mathematical reviews.
