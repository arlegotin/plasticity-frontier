# Branch-aware local tuning resolution

This is a numerical refinement under the approved [full-program design](2026-11-01-plasticity-frontier-design.md), section 11. The user explicitly authorized continuation on the current branch after the recorded local off-ramp. It does not supersede the historical results or authorize weaker scientific claims.

## Problem and decision

The retained source-only cohort contains a B04 group whose estimated-prior optimum switches from Z to F between gamma=0 and gamma=1/256. Their validation MSEs differ by 0.110202 while the existing objective tolerance is 0.070326. Each branch changes smoothly; the selected validation loss jumps. Bisecting the interval cannot make that jump vanish. See [the diagnosis](../../results/2026-11-01-local-verdict.md) and its linked raw evidence.

Three approaches are available: increase the finite refinement cap, change which branch B04 selects, or bound possible improvement separately along the existing branches. The first does not address the discontinuity; the second changes the estimators. Use the third, subject to the numerical checks below. This is a versioned selection-procedure change because the final selected gamma can change, even though every fixed-gamma estimator remains the same.

## Mathematical basis

Fix the ordered prior basis, base spectrum, noise, target n, depth, budget and branch feasible sets. Write s(t)=(1-t)s0+t/d, c=(n+1)/n, rho=(1+sigma_hat_squared)/n, and k=(B/(L-1))^(L-1). The F and Z branch objectives are

`f_t(g) = 1 - 2 s(t)^T g + sum_i (c*s_i(t)+rho)*g_i^2`.

In full physical gain coordinates, both feasible sets are fixed compact convex sets with 0<=g_i<=k. Strong monotonicity and the variational inequalities imply

`||g_b(t)-g_b(u)|| <= K*|t-u|`,

`K = max(1, abs(c*k-1)) * ||1/d-s0|| / rho`.

F's tied tail must be expanded in this norm. Z uses the actual trace-budget simplex, including at full width. No inverse validation Hessian is needed.

The observed two-direction validation query MSE is a convex quadratic in branch QP coordinates x: `ell(x)=x^T H x-2 h^T x+C`. For F, construct its prediction design with the top-direction columns and the literal complement column `X_query*z_context - sum(top_direction_columns)`. This corresponds to the actual floor-plus-increments metric. For Z below full width, use its active direction columns. The compressed-coordinate distance is bounded by full-gain distance. A supporting hyperplane yields

`ell(x) >= ell(x_anchor) - ||grad ell(x_anchor)|| * ||x-x_anchor||`.

Combine that inequality with K and explicit endpoint/uniform successful-solver error allowances. For each branch, take the strongest valid endpoint lower bound over the interval; for B04 take the smaller F/Z lower bound. The incumbent is always an eligible actually selected candidate. If its loss minus this lower bound is at most the existing tau, the interval meets a **local numerical possible-improvement diagnostic** despite a branch switch.

## Numerical and selection contract

- Preserve tau=`min(left_marginal_loss_se,right_marginal_loss_se)`, independent task units, paired diagnostics, initial interior probes, all fixed-family groups, resource ties, ridge boundaries, failures and refinement caps.
- Preserve the existing local neighbor scope. Neither the old adjacent-gap route nor the new route claims a global optimum or a confidence bound.
- Apply the extra route to supported B02/Z, B03/F and B04/F+Z groups. Unsupported cases retain the old rule; they do not receive an exemption. Initially require width strictly below d for the Z route to avoid silently treating a positive canonical floor as zero.
- Extra branch losses/gradients are noneligible resolution probes using the same observed validation tasks. They cannot enter ranking, alter B04's prior-risk branch selection, or enlarge tau. Retain their identities and computation cost, with no new label or independence count.
- Require fixed basis/base/noise/budget/width witnesses, completed finite endpoint solves, validated feasibility/KKT multipliers and consistency between the compact prediction quadratic and literal selected validation loss. Refuse the extra route on failed or inconsistent prerequisites, missing/nonfinite uncertainty, nonpositive tau, unsupported powers or normalization, or unrepresentable progress.
- Sensitivity concerns exact optimizers. Include a declared conservative uniform error model derived from the completed solver's residual/complementarity/feasibility contract, with full-gain multiplicities. Endpoint error alone is insufficient for unevaluated successful solves. Record every constant, padding term and lower-bound component. Ordinary float64 diagnostics remain `float64_diagnostic`; they are not rigorous enclosures or exact certificates. Fail closed when margins cannot be distinguished from numerical error.
- Do not assert guaranteed termination. A validation-good branch which B04 never selects can make the lower envelope conservative indefinitely. Prior-objective exclusion is a possible future refinement, outside this change.
- Version the source policy as `min_marginal_loss_se_branch_envelope_v2`. The v1 historical artifacts keep their identities and original unresolved verdicts. Any future protected evidence requires the new source to be committed and frozen first.

## Validation and next gates

Use TDD for a real branch switch, the observed validation quadratic in both directions, an unselected lower-loss branch, uniform solver-error allowances, failed prerequisites, zero uncertainty and pointwise estimator preservation. Retain the existing hidden-interior, all-group, ridge and failure/cost regressions. Run the full suite before committing implementation, then obtain a scoped independent review.

After source review and commit, rerun the existing exploratory cohort as development verification, checking source observation hashes. It adds zero independent preparations. Only then use a fresh exploratory root for full G6 evidence, after a separate capacity review addresses retention of large preparation records. Do not allocate protected keys or choose settings from primary effect signs. Negative or unresolved results remain reportable outcomes.
