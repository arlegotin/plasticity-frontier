# Prepared learning joint frontier: design specification

**Date:** 1 November 2026.
**Status:** proposed design for review; this document authorizes no experimental result or implementation claim.
**Scope chosen by the user:** the full blueprint program, staged by decision gates, working on the current branch.
**Scientific source:** [blueprint](../../blueprint.md), particularly §§2, 4–11.
**Original mathematics:** [research PDF](../../prepared_learning_research.pdf) and [research bundle](../../prepared_learning_research_bundle.zip).
**Local evidence:** [source audit](../reviews/2026-11-01-source-audit.md).

## 1. Intent and deliverable

Build a small, local Python research tool that solves the joint initialization-norm, trainable-parameter-count, and first-step-risk frontier for scalar-output deep linear networks. Compile each chosen metric into an initially zero predictor, verify the actual simultaneous gradient update, and determine how much of the constrained advantage survives estimated priors and honest computational comparisons.

Success is a reproducible result with correctly limited claims. A verified frontier with no learned-prior benefit, or a direct estimator that dominates the network operationally, is a valid outcome. The beneficiaries are researchers studying parameterization and learning geometry; no deployed application bottleneck is assumed.

The current assignment produces this spec and its implementation plan. Package creation, dependency installation, pilot studies, protected experiments, benchmarks, and scientific reports are future execution work. The blueprint and original assets stay unchanged.

## 2. Approaches and scope decisions

| Approach | Tradeoff | Decision |
|---|---|---|
| Full program, with mathematical, numerical, statistical, and resource gates | Gives an executable path through all WP00–WP10; later experiments depend on valid early results and measured feasibility | Selected by the user |
| Mathematical core first, with a separate later statistical plan | Shorter initial plan, but leaves prior estimation, protection against leakage, and practical comparisons undefined | Considered; not selected |
| General deep-learning or application benchmark | Introduces a different architecture and scientific population before establishing the proposed mechanism | Excluded by the blueprint's stated purpose |

Implementation choices proposed here: native CPU Python, NumPy/SciPy, standard-library `unittest`, JSON/JSONL/CSV records, and a `src/prepared_learning` package. Keep proof fixtures, independent checks, experiment execution, and analysis separate. Use a few focused modules where the blueprint's suggested files would otherwise combine unrelated responsibilities. These are planning choices, not additional user requirements.

All stages belong to one research program: later inference and performance claims depend on the verified geometry and the same resource/information contract. Do not expand into vector outputs, nonlinear networks, multistep optimization, biological energy, large models, or remote accelerators. Natural covariates and accelerator timing are optional, gated extensions.

## 3. Global constraints

The following exact requirements also apply to every implementation-plan task:

- Core arithmetic uses CPU float64; independent complex-step checks use complex128.
- Main scientific runs use depth L >= 2, eta = 1.0, and loss ||Xw-y||^2/(2n).
- Every network matrix entry is an independent trainable scalar, including initialized zeros; there are no biases, tied weights, residual paths, frozen features, or fixed normalization factors.
- Core tasks have mean zero, PSD second moment S with trace 1, independent Gaussian N(0,I_d) input rows, and independent mean-zero noise of variance sigma2 >= 0.
- Primary risk is expected squared coefficient error, equivalently noiseless isotropic prediction MSE; fresh-label MSE additionally includes sigma2.
- Preserve the original blueprint, PDF, ZIP, extracted source snapshot, and recorded failed diagnostics unchanged.
- Methods receive only their declared prior assets and observed context/source data; evaluator truth never enters learned preparation, adaptation, or validation selection.
- A canonical network and its matching compact direct map must produce the same first-step coefficient within reported numerical error.
- Float64 primal/dual values are diagnostics; only exact rational checks or rigorous roundoff enclosures may be called certificates.
- Confirmation uses frozen scientific inputs and fixed endpoint-specific sample counts; pilot data are excluded, and outcome-based sample extension is forbidden.
- Faithful recovery may change batching or checkpoint cadence; changes to code, dependencies, prior assets, or scientific configuration require a new confirmation identity.
- Record methodological amendments before changing population, coordinates, learning rate, primary metric, information access, baseline class, confirmatory configuration, or inference procedure.
- The core runs locally without external datasets, API keys, trained weights, or a GPU.

## 4. Mathematical contract

### 4.1 Architecture and sharp realization cost

Let `f_Theta(x) = W_L ... W_1 x = w(Theta)^T x`, with positive hidden widths and `w(Theta)=0` for every input. Define

\[
E=\sum_{\ell=1}^{L}\|W_\ell\|_F^2,\qquad
P=\sum_{\ell=1}^{L}m_\ell m_{\ell-1},\quad m_0=d,\ m_L=1.
\]

The input-space tangent matrix satisfies

\[
\langle\nabla_\Theta f(x),\nabla_\Theta f(y)\rangle=x^TGy,
\quad G=\sum_\ell\|b_\ell\|^2 R_{\ell-1}^TR_{\ell-1},
\]

where `R_0=I`, `R_l=W_l...W_1`, and `b_l^T=W_L...W_(l+1)`.

For `G != 0`, set `lambda=lambda_min(G)` and

\[
\Gamma(G)=\lambda+\sum_i(\mu_i-\lambda),\quad
m_*(G)=\operatorname{rank}(G-\lambda I)+\mathbf1_{\lambda>0}.
\]

The supplied theorem gives simultaneous minima

\[
E_{\min,L}=(L-1)\Gamma^{1/(L-1)},\qquad
P_{\min,L}=(d+1)m_*+(L-2).
\]

A compatible architecture realizes `G` iff its first width is at least `m_*`; the remaining hidden widths may be one. The zero metric uses an all-zero positive-width chain, norm zero, and minimum count `d+L-1`. Width zero is never an architecture. A parameter cap below `d+L-1` is infeasible even when `B=0`.

### 4.2 Canonical construction and literal update

For nonzero `G`, use `e=Gamma^(1/(L-1))`, `t=e^(L-2)`. Construct rows `sqrt((mu_i-lambda)/t) v_i^T` for each strictly positive gap. If `lambda>0`, append a zero row to `W` and place `sqrt(lambda/t)` in that coordinate of `a`; otherwise `a=0`. Layers are `[W, a^T]` followed by `L-2` separate scalar arrays `sqrt(e)`.

Then `W^T a=0`. All gradients are evaluated from old arrays before any update. For `z=X^T y/n` and the specified loss, the actual first-step coefficient is exactly `eta Gz` in real arithmetic. Scalar-suffix gradients initially vanish. A literal second update must be computed afresh; the first-step formula does not continue automatically.

A distinct depth-two final-zero-head constructor uses `W=diag(sqrt(g_positive)) V_positive^T`, `a=0`, with preparation norm `tr(G)`. Do not use the minimum-`Gamma` compiler for B02, especially at full rank. For deeper final-zero-head networks the norm formula is `(L-1) sum_i lambda_i(G)^(1/(L-1))`, subject to sufficient intermediate widths; deeper risk comparisons are not primary.

For a compact metric, the stored decomposition can be redundant. Before canonical construction, recover the actual minimum eigenvalue and positive-gap support, including the full-rank Z case. Preserve B02's independent trace-budget decomposition. Exact analytic metadata describes support; a floating-point eigenvalue cutoff cannot prove algebraic rank.

For every homogeneous null network, independently check

\[
E_{\rm after}=E_{\rm before}+\eta^2 g^TGg,
\quad g=\nabla_w\mathcal L(0).
\]

The blueprint's noncanonical scalar counterexample has `G=3`, `E=5`, and finite update `3h+h^2`, `h=eta*u`. It must remain nonlinear in the harness. The first-step linear claim applies to the canonical construction, not arbitrary deeper null networks.

### 4.3 Risk and joint frontier

With `c=(n+1)/n`, `rho=(1+sigma2)/n`,

\[
R_S(G)=1-2\operatorname{tr}(GS)+c\operatorname{tr}(G^2S)+\rho\operatorname{tr}(G^2).
\]

For caps `B>=0`, `P_cap`, define

\[
k=\left(B/(L-1)\right)^{L-1},\qquad
m=\min\left(d,\left\lfloor\frac{P_{cap}-(L-2)}{d+1}\right\rfloor\right).
\]

Reject infeasible architecture caps before handling `B=0`; otherwise `B=0` selects `G=0` and risk 1. Sort `S` eigenvalues descending, `s_1>=...>=s_d>=0`, and set `q_i=c*s_i+rho`. An optimum aligns gains with the task eigenvectors: clipping gains to `1/c` reduces risk/resources, then trace rearrangement orders them. Averaging rotations is not a valid width-constrained proof.

| Family | Variables and constraints | Risk |
|---|---|---|
| F: floor plus excess | `j=min(m-1,d-1)`, `x=(g_1,...,g_j,b)`, `g_i>=b>=0`, `sum(g_i)-(j-1)b<=k` | `1-2(sum(s_i*g_i)+S_tail*b)+sum(q_i*g_i^2)+Q_tail*b^2` |
| Z: trace-budget rank | `p=min(m,d)`, `g_i>=0`, `sum(g_i)<=k`, other gains zero | `1-2 sum(s_i*g_i)+sum(q_i*g_i^2)` |

For `j=0`, F is scalar isotropy, `b=min(k,1/(c+d*rho))`. For Z, `g_i=max(0,(s_i-nu/2)/q_i)`; use `nu=0` if feasible, otherwise a monotone budget root with bracket `[0,2s_1]`. At `m=d`, F covers all spectra and Z may have a positive minimum; do not label every Z matrix zero-floor or assert `Gamma=tr(G)` at full rank.

Solve and retain both branches, their actual resource usage, and verification records. In a numerically unresolved tie, retain both; a smaller actual trainable count is only a recorded implementation tie-break. Never turn a near tie into a claimed strict win. The exact rational fixture below is the normative deterministic oracle; general float computations must have error small relative to any reported risk gap.

### 4.4 Verification levels and numerical policy

For either oracle QP, use `f(x)=1+l^T x+x^T D x`, `A x+o>=0`, `D>0`. Recover nonnegative dual multipliers independently of solver success. The dual value is

\[
d(v)=1-v^To-\tfrac14(l-A^Tv)^TD^{-1}(l-A^Tv),\qquad v\ge0.
\]

Record original-unit primal slacks, dual feasibility, stationarity, complementarity, primal/dual values, active set, solver status/options, repair magnitude, and roundoff treatment. Global exact-arithmetic lower/upper values are the minima of branch lower/feasible-upper values. Exact rational fixtures use `fractions.Fraction` KKT solves and exact equality/inequality checks, independently of production float objectives.

Empirical B11 QPs can have a singular, nondiagonal Hessian. Their residual lower bound is `f(x)-v^T(Ax+o)-k*||grad f(x)-A^T v||_1`; do not apply the diagonal inverse formula. Record regularized and unregularized objectives separately.

For oracle QPs scale variables by `min(k,1/c)` at positive `k`, remove the constant objective term, and scale the remaining coefficients. This clipping bound is not automatically valid for B11. Use logarithms for depth powers, reject unrepresentable weights, and report numerical limitations instead of silently truncating positive gains.

Reject nonfinite values and material asymmetry/negative eigenvalues. Permit only documented rounding-scale symmetrization/PSD repairs. A dense scale policy begins at `64*eps*d*||G||_2`, with an explicit exact-zero branch and no `max(1,scale)` floor. Report absolute and relative kernel errors, repair mass, approximate-rank policy, and retained/dropped mass. Relative error is zero for two exact zeros; a nonzero error over a zero reference is reported explicitly as undefined/infinite with a status, never silently zero. JSON output contains no nonstandard NaN/Infinity numbers.

Use approximately `1e-10` relative residual goals for well-conditioned ordinary-scale cases, adjusted for dimensions/conditioning and recorded. This is not permission to claim effects below numerical resolution. Compare matrices/projectors for degenerate eigenspaces; signs only need a deterministic serialization convention.

## 5. Normative fixtures and workload

| Fixture | Required result |
|---|---|
| Original `d=6,r=2,n=4,sigma2=1` | `k0=4/25`, `k1=8/9`; six budgets `0.08,0.16,0.4,0.8,8/9,1.2`; widths `1 -> 3 -> 2` in positive-budget regimes |
| Original fixture, `k=2/5` | Norm-only risk `677/1025`; width-two Z risk `69/100`; distinguish 21 versus 14 minimum parameters |
| Primary richer spectrum | `s=(.50,.25,.15,.07,.02,.01)`, `d=6,n=4,sigma2=1,L=2,k=B=3/5,P_cap=21` |
| Primary F | `(g1,g2,b)=(2963,1978,798)/6905`, risk `186321/276200`, budget multiplier `953/27620` |
| Primary Z | `(g1,g2,g3)=(1013,518,194)/2875`, risk `81247/115000`, budget multiplier `2383/11500` |
| Primary comparison | Both use norm `3/5`, width 3, and 21 entries; `R_Z-R_F=1266883/39703750`; best isotropic risk `13/17` |
| Richer spectrum at `k=1,m=3` | Z risk about `0.668127428127` beats F about `0.673827673828` |
| Scale defect | Target `diag(1e-14,2e-14)` must retain its positive gap; the original absolute-cutoff constructor loses it |
| Limits | `d=1`, zero/isotropic/singular metrics, repeated minima, rank-one tasks, infeasible caps, boundary/huge/tiny budgets, genuinely rotated matrices, depths 2–8, and noncanonical finite-step counterexample |

W0 is the fixture suite above. W1 uses `d=64,r=8`, flat task spectrum, `n=4,8,32`, `sigma2=1`, widths `1,8,9,64`. W2 uses `d=64`, trace-normalized `s_i proportional to i^(-3/2)`, `n=8`, widths `4,8,16`. W3 adds rank-one/isotropic/nearly repeated/long-tail spectra. W4 varies prior rotations, off-subspace task mass, anisotropic Gaussian inputs, and isotropic non-Gaussian inputs separately. Later numerical grids are pilot outputs, frozen before confirmation; do not invent statistical counts now.

## 6. Methods, observations, and preparation

Keep three result panels: A matches oracle scientific caps; B matches learned source observations and information; C compares natural operational algorithms. Record equal caps and actual usage separately. Raw network norm is not a meaningful imposed cost for ridge or software direct maps; use not-applicable for that resource.

| ID | Required method |
|---|---|
| B00 | Zero estimate, risk 1 |
| B01 | Optimized isotropy; oracle scalar `min(k,1/(c+d*rho))`, learned scalar selected from source validation |
| B02 | Optimized Z with the separate depth-two final-zero-head constructor |
| B03 | F alone |
| B04 | Selected joint frontier, canonical network, all entries trainable |
| B05 | Matching compact direct map for every B01–B04 target; an equality control |
| B06 | Width-unconstrained spectral Gamma-budget QP, independent reference to the source phase formula |
| B07 | Unrestricted fixed map with eigengains `s_i/(c*s_i+rho)`; check whether it actually fits the caps |
| B08 | Zero-initialized depth-one scalar-step GD and validation-tuned ordinary ridge |
| B09 | Projected least squares/ridge, with an independent moment-direction representation comparator |
| B10 | Oracle Gaussian posterior and validation-tuned empirical Bayes |
| B11 | Empirical query-loss F/Z gain fitting in the same allowed basis; include zero regularization as a candidate |

Ridge uses the unnormalized penalty in `(X^T X+lambda_ridge I)w=X^T y`. Projected ridge uses `XU`. The posterior is `S X^T (X S X^T+sigma2 I)^(-1)y`; for `S=P/r` the rank-space penalty is `r*sigma2`. Use solves, Cholesky where positive definite, rank/sample-space alternatives where smaller, and documented pseudoinverses at singular/noiseless limits. Any ridge added for stability is a disclosed change. The Gaussian posterior optimality claim requires Gaussian task coefficients and noise; relabel it for other populations.

Each source task has independent A/B blocks sharing its latent coefficient, initially `n_source=n` rows each. Estimate `S_raw=mean((z_A z_B^T+z_B z_A^T)/2)`. Project negative eigenvalues, report removed mass, normalize retained trace only under the trace-one synthetic contract, and use recorded `I/d` fallback if none remains. The moment-direction comparator uses source averages of `y_i^2 x_i x_i^T`; its eigenvalues are not unbiased task variances. Shrinkage candidates include 0 and 1. The optional estimated-noise arm uses source-only `max(0,mean(y^2)-1)` under the synthetic contract and task-clustered uncertainty.

B11 uses A for context and B for query, then swaps them. For Z its gain-design column is `(X_B v_i)*(v_i^T z_A)`; F's final column is `X_B z_A` minus the sum of selected directional columns. Fit average query squared loss under the same F/Z constraints. All methods share source tasks, partitions, and allowed candidate assets, while validation choices may be method specific. Count unique labels once and every actual computation; record `2*T*n_source` training labels plus validation context/query labels.

The primary learned mechanism selector baseline includes B01/B02 and their allowed estimation/B11 variants. Its joint competitor includes F/Z with the same estimation/fitting improvements. B05, ridge, Bayes, and oracle truth are not eligible primary superiority targets. Validation selects candidates before test evaluation; preserve every tried candidate and failure. Freeze the selection procedure and candidate set, not a single prior shared across all outer replications.

## 7. Architecture and interfaces

Separate these responsibilities; the plan supplies precise signatures and files:

| Boundary | Responsibility |
|---|---|
| Configuration and records | Validate budgets, target context size, input law, units, endpoint-specific sample plans, information flags, artifacts, hashes, and failure classes |
| Geometry | Compact operators, constructors, literal parameter gradients/updates, effective coefficients, norm/count accounting |
| Solvers and verification | QP assembly, F/Z and original spectral reference, exact rational KKT, numerical diagnostics; independent code must not reuse the formula it verifies |
| Source data and methods | Keyed generators, source-training/validation observations, prior fitting, shared B00–B11 lifecycle, evaluator-only truth |
| Execution and protocol | Atomic blocks, deterministic resume, immutable scientific identity, pilot/freeze/confirmation, explicit failure records |
| Analysis and reports | Conditional risk, endpoint-specific paired inference, timing/Pareto comparisons, raw-record reconstruction, claim ledger |

`BudgetSpec` contains depth, squared initialization-norm cap, parameter cap, learning rate. `PreparationSpec` contains target context size, dimension, input law, loss normalization, task second-moment trace. `Context` contains only float64 `X:(n,d)`, `y:(n,)`, task key. Evaluator truth (`theta`, true `S`, true noise, rotations, held-out labels) is separate and is never passed to learned methods. Validation may use its explicitly supplied source query observations.

`CompactMetric` contains floor, orthonormal basis `(d,j)`, nonnegative increments `(j,)`, support metadata. Apply it as `b*z+U@(u*(U.T@z))`; dense matrices are for small audits. `PriorAsset` contains covariance/basis/noise estimate, fit metadata, data hashes, and `information_access`. `PreparedState` contains immutable initial arrays/metric, budget, preparation contract, and resources. `AdaptedState` contains a fresh coefficient/network state and status/timing. Frozen dataclasses alone do not freeze arrays: copy arrays defensively and make reusable initial arrays read-only.

Methods expose `fit_meta`, `prepare`, `adapt`, `predict`, and `diagnostics` as in blueprint §7.2. `prepare` derives `c,rho` from target `n` and the permitted noise estimate. `adapt` rejects mismatched shapes/context counts and creates fresh state; processing one task must never mutate the next task's initializer. Store both branch records in frontier results even when one loses or the comparison is unresolved.

No product command exists yet. The future CLI commands are `doctor`, `check-theory`, `check-geometry`, `reproduce`, `frontier`, `fit-priors`, `pilot`, `freeze`, `confirm`, `robustness`, `benchmark`, `analyze`, `replay`, `report`. Their owning plan task defines each command; editable installation precedes use. CLI failures use nonzero exit codes and machine-readable failure records rather than empty success output.

## 8. Endpoints, protection, and inference

| Analysis mode | Independent unit | Required frozen counts |
|---|---|---|
| `deterministic_frontier` | None; exact model quantity | None; stochastic counts explicitly inapplicable |
| `conditional_risk_over_preparations` | Independent source-training/validation replication | Outer J and all source/validation task/block counts; inner test count inapplicable |
| `fixed_prior_task_simulation` | New task | Fixed test N |
| `nested_task_simulation` | Independent prepared prior, with tasks nested inside | Outer J, inner N, source/validation counts |

Primary learned endpoint: per outer replication `D_j=R_S(G_baseline,j)-R_S(G_joint,j)`, positive favoring joint. Use the full matrix risk when learned and true priors are misaligned. For compact `G=bI+U diag(u)U^T`, let `t_i=U_i^T S U_i`, `h_i=2bu_i+u_i^2`; then `tr(GS)=b+sum(u_i*t_i)`, `tr(G^2 S)=b^2+sum(h_i*t_i)`, and `tr(G^2)=d*b^2+sum(h_i)`. This evaluator-only calculation removes inner simulation noise, not preparation uncertainty.

The starting primary scenario is the richer `d=6,k=0.6,P_cap=21` fixture. Use two exploratory pilot blocks to estimate preparation variance/precision stability and count needs. Start precision planning at `h=oracle_gap/4`; choose approximately `J=(z_(1-alpha/2)*s_D/h)^2`, conservatively accounting for pilot variance uncertainty. Freeze a supported Student or outer-bootstrap interval at final J and state approximate coverage. Source counts increase geometrically to bracket preparation accuracy/cost, not until a favorable test appears. For nested endpoints only, use variance `(v_between+v_within/N)/J` and measured-cost starting allocation `N=sqrt(v_within*a/(v_between*b))` when both components are positive. Do not impose nested testing on the primary analytic endpoint.

One primary contrast uses `alpha=0.05`. Prespecified secondary confirmation families use Holm adjustment or simultaneous intervals. Pilot/development keys are excluded permanently. Freeze scenario, counts, candidate selectors, endpoint, confidence procedure, multiplicity family, code/dependencies, source/preprocessing assets, scientific config, seed mapping, and analysis before protected keys are generated/accessed. Invariants may stop a run; primary effects may not alter sample counts. An underpowered result is inconclusive, followed only by a separately amended fresh confirmation.

Use fixed schedules for network-update Monte Carlo checks and X-dependent methods. For task tails, check the pilot assumptions behind inference. A finite-moment Student input stress arm uses more than eight degrees of freedom; an infinite-fourth-moment arm is a labelled assumption-failure diagnostic with no ordinary Gaussian mean-risk CI.

## 9. Execution records, recovery, and resources

Derive independent NumPy `SeedSequence` streams from `(root_seed,phase_id,outer_replication,task_id,stream_id)` and record the bit generator/version. Separate task, input, noise, query, rotation, and method streams. Per-task identity must survive batch-size and method-order changes. Save prepared arrays with hashes, not just seeds.

Each run contains blueprint §9.2's complete `run.json`, `task_metrics`, `preparation_risks`, `geometry`, `certificates`, `timing_blocks`, and decision records. Include explicit schema version, scientific/operational hashes, full Git revision/dirty diff, protocol/dependency/source/prior hashes, analysis/information mode, environment, statuses and failure reasons. Risk truth hashes are evaluator records. Exact values store rational numerators/denominators. Do not silently drop failures or include only the selected winning branch.

Block protocol: write raw records/metadata to temporary paths; flush and verify counts/hashes; atomically rename; then mark committed. On restart verify every committed block, reconstruct summaries from raw rows, discard uncommitted partial files, and recompute only uncommitted keys. Refuse changed scientific/code/dependency/prior identities. An operational batch/checkpoint change is logged and preserves task identities; a BLAS/thread change needs equivalence verification and a separate timing stratum. Retain every method's paired population even when one fails; report an incomplete/invalid comparison rather than averaging away failures.

WP00 discovers native architecture, macOS, CPU, available memory/storage and BLAS; redact serial/UUID fields. Choose compatible binary wheels, smoke-test float64/complex128 operations and SLSQP, record exact versions/hashes/licenses, then freeze. Historical Linux versions in the blueprint are evidence context, not package pins. Use CPU as the reference; an optional accelerator arm must prove its precision path before use.

Calibrate with W0 then representative W1/W2 blocks. Initially reserve at least half of currently available memory. Stream source moments and losses, avoid a task-by-method-by-dimension prediction cube, use compact maps and smaller rank/sample-space solves, and confine explicit Jacobians to tiny cases. Record measured cold/warm preparation, adaptation, prediction, serialization, peak memory, timing noise, and preparation amortization. Use `perf_counter_ns`, randomized method order, common dtype/thread policy, and explicit boundaries. Normalize platform RSS units and isolate peaks when cumulative process RSS cannot distinguish methods. Allow linear-network coefficient collapse with its measured cost. No time or norm measurement implies joules.

## 10. Stage gates and acceptance

| Gate / blueprint package | Required evidence | Next step |
|---|---|---|
| G0 / WP00 | Preserved source identity, native CPU operation checks, exact environment and validated contract | Mathematical challenge |
| G1 / WP01 | Proof obligations for lower bound, equality, width, alignment and union; rational fixtures; known finite-step limit | Mechanism harness |
| G2 / WP02 | Independent derivatives, literal simultaneous step, scale/zero/degeneracy tests, correct norm/count | Baselines |
| G3 / WP03 | Six source budgets, independent unrestricted QP, correct B00–B02/B05–B10, direct equality and singular posterior checks | Joint solver |
| G4 / WP04 | Both F/Z records, exact primary certificates, independent tiny support checks, ties/reversal, monotonic caps and zero/full-width cases | Learned preparation and pilot |
| G5 / WP05 | Equal source access, validated prior/B11 fitting, no truth/query leakage, accounting/tuning ledger | Learned-prior pilot |
| G6 / WP06 | Stable pilot precision/resource estimates and immutable endpoint-complete confirmation manifest | Protected execution |
| G7 / WP07 | Complete frozen schedule, per-replication conditional differences and required simulations, valid failures/intervals | Attribution and cost |
| G8 / WP08 | Prespecified attribution and shifts, regression/tail records, fresh confirmation for amended claims | Scoped robustness conclusions |
| G9 / WP09 | Equality-valid compact comparisons, randomized timing/memory/bytes/amortization tables | Operational conclusion |
| G10 / WP10 | Rebuilt tables, interrupted/replayed equivalence, audits, evidence-linked claims, negative and unrun outcomes | Delivery |

Core mathematical gates G0–G4 are the first execution milestone. G5–G7 are conditional learned-prior evidence. G8/G9 resolve attribution and operational questions without requiring a positive G7 effect; a valid negative result still supports useful diagnostics. G9 requires baseline/equality/calibration gates, not a statistical win. If an earlier mathematical result fails, do not run descendants claiming that result; retain valid independent controls. Record stop/continue decisions rather than automatically spending the full experimental budget.

Use exactly these gate outcomes: `IMPLEMENTATION_INVALID`, `MATHEMATICAL_COUNTEREXAMPLE`, `INSENSITIVE_OR_UNSUITABLE`, `LOCAL_RESOURCE_LIMIT`, `NO_COMPARATIVE_ADVANTAGE`, `SUPPORTED_IN_SCOPE` (blueprint §7.3). A verified counterexample stops affected theorem claims; solver failure alone does not. Negative practical results narrow practical claims without refuting valid mathematics.

WP08 includes no-floor, isotropy, equal actual count/caps, wrong orientation, finite/shifted priors, input-law violations, coordinates, free learning rate, depth one, approximate width, float32/extreme scales, second step, noncanonical null networks, shared implementation optimizations, and resume/error adversaries. W1/W2 independent transfer confirmations are separately frozen; exploratory changes require fresh confirmation for retained claims.

Optional natural-covariate arm: blueprint §5.6 specifies UCI white Wine Quality, discarding the quality label, with official download/license/hash verification. Group identical rows before deterministic 60/20/20 pool splits; fit transforms on development only. Draw artificial tasks/labels and describe the result as conditional semi-synthetic robustness, not wine prediction. Omit the arm if asset/license/split gates fail. No download is part of this planning assignment.

## 11. Reporting and claims

Deliver frontier tables, resource-matched and operational panels, per-family records, conditional/nested intervals with correct units, preparation-learning and shift boundaries, measured resource Pareto tables, failure logs, and standalone scientific figures plus underlying data. Rebuild them from raw records in a clean same-machine environment with locked dependencies and source revision. A second machine and formal proof assistant are not required.

`reports/results.md` and the claim ledger must distinguish supplied results, fresh exact arithmetic, numerical diagnostics, exploratory measurements, protected evidence, forecasts, and unperformed work. Include all prespecified scenarios and the strongest permitted baselines. Preserve failed nonsmooth source diagnostics. State historical novelty as unresolved; the supplied documents do not establish peer review, formal verification, energy savings, biological claims, or general estimator superiority.

Readiness for implementation requires review of this design and the companion plan. Planning choices left to the builder include compatible package releases, local batch/reserve sizes after calibration, plotting appearance, internal optimizations preserving the specified maps, and numerical refinements with explicit evidence. Choices deliberately deferred to the pilot include source-count levels, statistical sample counts, tuned grids, and feasible timing schedules. They must be resolved and frozen before confirmation, never guessed by an executor at protected-test time.
