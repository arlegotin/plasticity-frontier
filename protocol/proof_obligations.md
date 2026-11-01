# Mathematical obligations and evidence

This ledger distinguishes the supplied theorem from consequences derived in the blueprint and empirical questions. Exact finite fixtures check arithmetic and consistency; they do not prove a theorem or establish empirical performance. Sources: [design specification](../docs/superpowers/specs/2026-11-01-plasticity-frontier-design.md) §4 and [blueprint](../docs/blueprint.md) §§4.1–4.7.

## Supplied theorem: realization and lower bound

For positive-width homogeneous linear networks with null output and every entry trainable, define `R_0=I`, `R_l=W_l...W_1`, `b_l^T=W_L...W_(l+1)`, `p_0=1` artificially, `p_l=||R_l||²` and `q_l=||b_l||²`. The tangent matrix is `G=sum_l q_l R_(l-1)^T R_(l-1)`. Put `alpha_l=p_l/p_(l-1)` and `beta_l=q_(l-1)/q_l`, using zero for zero denominators. Null output yields `alpha_L=beta_1=0`. Projection onto the backward direction and its orthogonal complement gives `||W_l||² >= alpha_l+beta_l`. With `Gamma=lambda_min(G)+sum_i(mu_i-lambda_min(G))`, the obligation is

`Gamma <= q_1+sum_(l=2)^L q_l p_(l-1) <= product_(j=1)^(L-1)(alpha_j+beta_(j+1)) <= (E/(L-1))^(L-1)`.

The middle product includes the nonnegative sum terms and the last step is AM-GM. See blueprint §4.1 for the denominator and projection argument. The width necessity follows from `G=q_1 I+W_1^T H W_1`, `H` PSD: for `m_1<d`, the minimum is `q_1`; when positive, the backward vector occupies an additional null direction of `W_1^T`. For `m_1>=d`, the stated bound is automatic. The all-zero chain requires positive widths, costing `d+L-1` entries.

Equality is attained by the blueprint §4.2 construction: put one first-layer row along each positive eigen-gap and, when the minimum is positive, add a zero row paired with the nonzero head coordinate; append scalar factors. With `e=Gamma^(1/(L-1))` and `t=e^(L-2)`, this gives `W^T W=(G-lambda I)/t`, `||a||²=lambda/t`, `W^T a=0`, equal contributions to the norm bound, and the minimum width and count simultaneously. Updating all old factors simultaneously gives `w'=-eta Gg`; this is a canonical first-step identity, not a universal finite-step identity.

## Blueprint derivations: optimization and controls

Clipping eigenvalues above `1/c` decreases risk and cannot increase `Gamma` or required width. On the clipped interval `h(g)=c g²-2g` decreases; trace rearrangement pairs the largest task eigenvalues with the largest gains. Rotation averaging is not justified for the nonconvex width constraint. Positive minimum means at most `m-1` positive gaps, hence family F; zero minimum means rank at most `m`, hence family Z. Their union is complete; at `m=d`, F covers all spectra and a full-rank Z point need not have zero minimum or `Gamma=trace(G)` (blueprint §4.3).

For `f=1+l^T x+x^T D x`, `Ax+o>=0`, `D>0`, completing the square in the Lagrangian gives `d(v)=1-v^T o-(l-A^T v)^T D^-1(l-A^T v)/4`, `v>=0`. Exact primal and dual feasibility, stationarity, complementarity and equality certify each rational branch (blueprint §4.4). Float64 values without roundoff enclosures are diagnostics only. The independent fixtures check these conditions; they do not prove completeness or realization.

For `d=1`, directly minimize the scalar risk at `min(k,1/(c+rho))`; the isotropic onset expression for nonscalar spectra may be `0/0` and is inapplicable. A depth-one zero linear layer instead has `G=I` and zero norm and lies outside the fixed-depth theorem. Scaling initial weights by `s` scales `E` by `s²` and `G` by `s^(2(L-1))`; free learning rate can preserve the canonical predictor while initial norm tends to zero. Euler homogeneity gives `Theta·grad L=L w·g=0` at null output, so simultaneous updates satisfy `E_after=E_before+eta² g^T G g`, whose second term grows under that free-rate scaling. These controls limit physical interpretations of initial norm (blueprint §4.7).

The explicit noncanonical depth-three scalar example has `G=3`, `E=5` and `w'=3h+h²`; the exact reference evaluates the simultaneous matrix update. This is a counterexample to extending the canonical first-step linear identity to arbitrary null networks, not to the stated theorem.

## Empirical hypothesis and unresolved priority

Whether preparation yields a practical advantage under finite source data, matched access and resources is an empirical hypothesis, not a consequence of the theorem. Source-recorded Monte Carlo results retain their original provenance and are not newly measured here. Historical novelty and priority of the joint frontier remain unresolved; call it a blueprint derivation, not an established novelty claim. A credible analytic counterexample to a required theorem should be preserved as `MATHEMATICAL_COUNTEREXAMPLE` and stop affected claims; solver failure or a failed fixture check alone is an implementation failure.

### Joint frontier implementation decisions

The exploratory oracle uses the aligned F and Z scalar programs in design §4.3. Variables are scaled by `min(k,1/c)` and the nonconstant objective by its largest coefficient scale; SLSQP's proposed active set is polished through a KKT solve. The rank shrinkage root is separately evaluated as a diagnostic. Original-unit primal, dual and residual values are float64 diagnostics without roundoff enclosures, not certificates. Comparisons whose branch status fails or whose risk gap is at most ten times the larger diagnostic dual gap (with a `1e-14` absolute resolution floor) are unresolved. Both branches are retained; an exact floating-point risk tie may use smaller actual canonical parameter count as an implementation tie-break, without claiming strict superiority. This resolution floor is a computational decision, not a change to the mathematical objective or scientific population. No adaptive frontier mesh is generated by the single-point CLI.

Review correction: an inactive-budget diagonal minimizer is evaluated directly to preserve weak positive directions. Active-set polishing is accepted only when the equality solve is finite, primal feasible, and has nonnegative multipliers; an invalid or singular proposed active set marks the branch failed. Z additionally records the maximum per-coordinate relative discrepancy with the independent rank root and treats discrepancies above `1e-6` as unresolved, including cases where root cancellation limits a tiny-budget comparison. Nonfinite solver iterates produce finite fallback branch records marked failed.
