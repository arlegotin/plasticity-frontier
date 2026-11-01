# Prepared learning at a zero predictor: research and local proof-of-concept plan

**Research date:** 31 October 2026.  
**Target machine:** one MacBook Pro with an M3 Max; actual memory, storage, macOS, and installed software must be discovered.  
**Recommended direction:** an exact **joint initialization-norm, trainable-parameter-count, and one-step-risk frontier** for scalar-output deep linear networks, followed by tests of numerical stability, estimated task priors, and actual computational cost.  
**Status:** research and executable experimental specification. The supplied results were inspected and rerun; a narrow mathematical extension and small additional checks were developed during this investigation. The full proposed PoC has not been implemented or run on an M3 Max. Historical novelty, practical superiority, and broader biological relevance remain unresolved.

## How to use this handoff

This document contains the definitions, equations, algorithms, baseline specifications, experimental decisions, and acceptance gates needed to work without the originating conversation. Implement the proposed repository and commands in Section 7. Commands labelled **proposed** do not exist yet. Commands labelled **existing** are ordinary operating-system/package commands or scripts in the original bundle, when that bundle is available.

The coding agent may make routine, reversible implementation choices, repair defects, reduce execution batch size, and select compatible package builds. It must record a methodological amendment before changing the scientific population, coordinate system, learning-rate contract, primary metric, information access, baseline class, confirmatory configuration, or inference procedure. Keep decisions in project files; chat memory is not an execution dependency.

The core is a component-level mathematical and numerical investigation. A useful negative outcome is a verified theorem with no operational advantage over a simpler direct estimator. Do not add a large model, nonlinear benchmark, or application merely to make the project look more ambitious.

## 1. Executive assessment and recommended direction

### 1.1 Verdict

The supplied note asks a substantive, sharply defined question: **how cheaply can a network be prepared to learn a chosen collection of directions while initially predicting exactly zero?** Its proposed answer is an exact minimum squared weight norm, accompanied by exact width and parameter-count characterizations. A separate statistical result identifies a transition from isotropic learning to specialization under that norm budget.

The mathematical audit found no contradiction in these results under the stated assumptions. Fresh execution reproduced the supplied ordinary-scale checks. However, the code has a real scale-sensitivity defect, and the broader interpretation needs tightening:

- Preparing or transferring a learning geometry while preserving current predictions is established prior art, particularly Neural Tangent Transfer, antisymmetrical initialization, warped-gradient learning, and plasticity injection [R02], [R04], [R07], [R08].
- The constructed network's first update is **exactly the direct estimator** `G @ (X.T @ y / n)`. It cannot have lower first-step prediction error than a correct implementation of that same estimator.
- Squared initialization norm is a coordinate-dependent mathematical resource. It is not measured energy, memory, development cost, description length, or biological expenditure. An unrestricted learning rate can make its apparent cost arbitrarily small.
- The strongest plausible differentiation is the exact prescribed-tangent extremum and its consequences for **simultaneously constrained norm and width**. No equivalent exact result was located in the primary material inspected, but this is not a priority certificate.

**Select the joint frontier as the primary direction.** Retain the original zero-predictor mechanism, Euclidean gradient coordinates, fixed step size, and linear-network setting. Add a parameter cap to the original norm constraint, solve for arbitrary task spectra, and identify when a positive isotropic floor is worth using one hidden coordinate. This extension addresses a concrete omission: the source's norm-optimal intermediate regime needs more trainable entries than either adjacent regime.

### 1.2 What the PoC should establish

The primary result is a constructive mathematical characterization of the best risk attainable by the specified class of PSD linear first-step maps. The implementation should recover that frontier using two small convex quadratic programs, realize each selected map as a zero-output network, and verify the exact simultaneous first update. Distinguish exact rational certificates from floating-point diagnostics as specified in Section 4.4.

The secondary result is a boundary map: how much of the constrained advantage survives finite prior estimation, approximate eigenspaces, finite precision, input-distribution violations, and honest preprocessing/adaptation/storage accounting. These are empirical questions. They may have negative answers.

A practical claim requires comparison with structured direct updates, tuned ridge, projected ridge, and Gaussian oracle/empirical Bayes. None requires a remote GPU. An application-level claim is outside the core and would need a separately justified real task population.

### 1.3 Findings actually obtained during this investigation

All computations below used **Linux x86_64, Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0**, except that the Bayes diagnostic required only NumPy. They are not M3 measurements. Parallel model-based reviewers examined mathematics, prior art, and feasibility; this is not independent human peer review or formal verification.

| Check performed now | Result and interpretation |
|---|---|
| Both supplied PDFs | Both have 14 pages; extracted text is identical after whitespace normalization. The main phase diagram was visually inspected. The PDF files themselves have different hashes. |
| Supplied `verify_plasticity.py` rerun | 5,600 random null networks and 1,400 attaining constructions passed its assertions. Largest kernel error about `6.28e-15`; largest normalized bound ratio `1.0000000000000002`. |
| Supplied `audit_geometry.py` rerun from a copy | 12,600 random null networks, 2,100 constructions, and 42 complex-step Jacobian checks passed. Complex-step checked 5,341 parameter derivatives; maximum kernel discrepancy about `1.42e-14`. |
| Supplied `phase_transition.py` rerun | 200,000 tasks, six budgets, and all six diagonal QPs reproduced the reported behavior. At `k=0.4`, analytic risk `0.660487804878`, Monte Carlo `0.661088500307`, task-level SE `0.001566367345`. The optional nonsmooth full-matrix diagnostic was not rerun. |
| Fresh joint-budget check | 1,392 tiny support-QP cases; maximum primal-dual gap `8.18e-14`; no alternative support or random rotation improved the proposed aligned solution beyond roundoff. This supports, but does not prove, the reduction in Section 4. |
| Fresh exact rational fixture | At the richer spectrum in Section 4.6, `k=3/5`, both branches use norm `3/5` and 21 actual trainable parameters. Exact rational primal feasibility, nonnegative multipliers, stationarity, and complementarity give risks `186321/276200` for F and `81247/115000` for Z. Their exact difference is `1266883/39703750`, about `0.0319084`. |
| Fresh scale adversary | The supplied audit constructor drops eigen-gaps below an absolute `1e-12`. For `G=diag(1e-14,2e-14)`, relative Frobenius error is `0.4472135955`. This is a code defect, not a counterexample to the theorem. |
| Fresh post-update norm check | 125 random null networks verified the identity in Section 4.7; largest absolute residual about `7.99e-15`. |
| Fresh broader-estimator diagnostic | On 50,000 Gaussian tasks with `d=6,r=2,n=4,sigma2=1,k=0.4`, constrained risk `0.654742` (SE `0.003097`), unrestricted fixed-map risk `0.550381` (SE `0.002809`), Gaussian posterior risk `0.439128` (SE `0.002258`). The paired constrained-minus-posterior difference was `0.215614`, SE `0.002131`. These exploratory results show why constrained optimality is not general estimator superiority. |

The source risk and the fresh Bayes diagnostic use different random samples; their means are not directly paired with each other. The favorable joint-frontier examples in this document are development fixtures, not protected confirmation.

## 2. Problem, scope, assumptions, and hypotheses

### 2.1 Beneficiary and bottleneck

The immediate beneficiaries are researchers studying initialization, trainable parameterizations, and learning geometry. The workflow is:

1. Specify or estimate a distribution of future linear-regression tasks.
2. Choose a desired first-step learning operator before seeing a new task.
3. Compile that operator into network weights while keeping the current predictor zero.
4. Adapt once using ordinary simultaneous gradient descent.
5. Measure predictive risk and the resources needed to prepare and execute the learner.

The demonstrated bottleneck is mathematical: in this fixed architecture, not every tangent matrix can be realized with a given norm and number of independent trainable coordinates. It is **not established** that initialization norm is a bottleneck in a deployed learning system. That distinction determines the permissible claims.

### 2.2 Network and resources

Use separate symbols for network parameters `Theta` and a task coefficient vector `theta`:

\[
f_\Theta(x)=W_LW_{L-1}\cdots W_1x=w(\Theta)^Tx,
\quad x\in\mathbb R^d,\quad L\ge2.
\]

The hidden widths are positive integers. Every matrix entry is an independent trainable scalar, including entries initialized to zero. There are no biases, tied entries, residual paths, fixed normalization factors, or frozen feature matrices in this theorem's architecture.

\[
E(\Theta)=\sum_{\ell=1}^L\|W_\ell\|_F^2,\qquad
P(\Theta)=\sum_{\ell=1}^L m_\ell m_{\ell-1},
\]

where `m_0=d`, `m_L=1`. Initialization must satisfy `w(Theta)=0` for **every** input, not just a training sample. Define the input-space tangent matrix by

\[
\langle\nabla_\Theta f_\Theta(x),\nabla_\Theta f_\Theta(y)\rangle=x^TG_\Theta y.
\]

`G` is a `d by d` PSD matrix. It is not the empirical sample Gram matrix `X G X.T`; input-space isotropy does not make that sample Gram an identity matrix. The tangent-kernel interpretation is established [R01].

The primary contract fixes `eta=1`, raw Euclidean gradient descent, and the loss

\[
\mathcal L(w;X,y)=\frac1{2n}\|Xw-y\|_2^2.
\]

Changing `1/(2n)` to `1/n`, normalizing layers, using Adam, or assigning layer-specific learning rates changes the mechanism. Such changes belong in labelled controls or a new study.

### 2.3 Task model and risk

For each independent task, draw a coefficient `theta` with mean zero and second moment

\[
S=\mathbb E[\theta\theta^T]\succeq0,\qquad \operatorname{tr}S=1.
\]

Draw `n` independent rows `x_i ~ N(0,I_d)`, independent of `theta`, and observations `y_i=x_i.T theta+xi_i`. Noises are mutually independent, independent of inputs/tasks, mean zero, and variance `sigma2 >= 0`. Gaussian tasks and noise are used in the core simulation so the Bayes reference is exact; the source risk formula itself only needs their stated second moments.

Set

\[
z=X^Ty/n,\quad c=(n+1)/n,\quad \rho=(1+\sigma^2)/n,
\quad Q=cS+\rho I.
\]

The first-step estimator class is a deterministic matrix chosen before the new task:

\[
\widehat\theta=Gz,\qquad G\succeq0.
\]

The primary risk is `E ||theta_hat-theta||_2^2`, in squared coefficient units. Under the isotropic test distribution it equals expected prediction MSE against the noiseless target. Prediction MSE against a noisy fresh label adds `sigma2`; never mix these two definitions.

The moments and exact risk are

\[
\mathbb E[z\theta^T]=S,\quad \mathbb E[zz^T]=Q,
\]
\[
R_S(G)=1-2\operatorname{tr}(GS)+c\operatorname{tr}(G^2S)+\rho\operatorname{tr}(G^2).\tag{1}
\]

To check the formula independently, expand the `n` diagonal and `n(n-1)` off-diagonal sample pairs. Gaussian fourth moments give `E[x x.T S x x.T]=2S+tr(S)I`. The observation-noise contribution is `sigma2 I/n`. These identities explain all coefficients in (1).

### 2.4 Joint scientific objective

For fixed depth `L`, norm cap `B >= 0`, and trainable-coordinate cap `P_cap`, minimize (1) over PSD maps admitting a zero-predictor realization with `E <= B` and `P <= P_cap`.

This is an optimum over **linear first-step maps**, not over every finite update of every zero-output deep network. Other null networks can produce nonlinear polynomial functions of `z` after one simultaneous update. Estimators such as ridge and posterior means also use `X` beyond its contribution to `z`. These larger classes remain operational comparisons, not counterexamples to a restricted theorem.

### 2.5 Claims and falsification criteria

| ID | Claim or hypothesis | Support and alternative explanation | Weakening or falsifying evidence |
|---|---|---|---|
| H1 | The supplied exact norm/width theorem holds under the specified coordinates. | Proof structure and reruns support it. A numerical optimizer agreeing with a formula alone would be circular evidence. | A valid analytic counterexample, a lower-norm exact realization, or a flaw in a necessary inequality. A failed approximate solver is not such evidence. |
| H2 | The joint norm/width risk problem reduces to the two QPs in Section 4. | A clipping-and-rearrangement derivation is supplied, with tiny exhaustive checks. Rank constraints invalidate the source's simpler averaging proof. | A feasible metric with lower certified risk; a failure of the clipping, orientation, or union-of-families argument. |
| H3 | A positive isotropic floor can improve risk over the best pure-rank/zero-head preparation under the same resource caps. | The source and fresh examples show local headroom. Extra actual parameters, weak spectral tuning, or unequal priors could explain a careless comparison. | Correctly optimized equal-cap comparisons show equality or reversal in the declared population; the plan expects both favorable and unfavorable regimes. |
| H4 | Some H3 advantage survives estimation of the prior from finite source tasks. | Plausible continuity of risk; no proof or target-laptop evidence supplied. It may require too many tasks or accurate eigenvectors to be useful. | Confirmatory paired intervals show no advantage, adverse tails, or prior-estimation expenditure overwhelms any operational benefit. |
| H5 | The network representation has an operational benefit relative to direct structured `Gz`. | No current evidence. Exact output equality makes any advantage a question of integration, storage, or execution. | Direct `Gz` is equally accurate and dominates time/storage/preparation. This outcome is compatible with H1-H3. |

H1-H3 are the core. H4-H5 are conditional practical investigations. Nonlinear networks, vector outputs, continuous learning over many updates, evolution/development, genome compression, and biological energy are speculative extensions, not necessary PoC milestones.

### 2.6 Strongest case for and against

**For:** the question constrains the initial function and its derivative separately; its sharp lower bound applies to all compatible widths and has a constructive equality case. The resource geometry permits cheap isotropic sensitivity and produces a specific competition between a floor and directional capacity. A joint width cap turns that into an exact, locally testable architecture-level prediction.

**Against:** the preparation cost is not invariant under reparameterization or learning-rate changes; a direct linear model already learns without hidden layers; the useful prior must be acquired and stored; and the entire first-step predictor can be executed without the network. Exact rank transitions can also disappear under approximation. The project can establish a meaningful variational result while offering no better learner in practice.

## 3. Prior art, reuse, and precise novelty assessment

### 3.1 Closest comparison matrix

The rows below distinguish conceptual overlap from an identical extremal problem. Primary-source locations and access limits are recorded in Section 12.

| Prior work | Problem/mechanism and assumptions | Evidence and computation | Exact overlap and implication |
|---|---|---|---|
| Jacot et al., NTK, 2018 [R01] | Parameter-Jacobian Gram controls learning; infinite-width results under their parameterization. | Theory and numerical kernels; finite Jacobians can be costly. | Tangent geometry is established. The full prescribed-zero-function resource extremum is a separate question. |
| Arora et al., 2018 [R03] | Deep linear factorization induces preconditioning in optimization. | Gradient-flow analysis and experiments. | Depth as a learning metric is prior art; do not claim it as a new principle. |
| Zhang et al., ASI, 2020 [R04] | Paired networks cancel initial outputs while preserving a tangent kernel under chosen normalization. | Theorem 4 and initialization experiments. | Zero predictions with nonzero learning geometry are established. Extra copies and normalization differ from the present all-entry coordinate contract. |
| Liu and Zenke, NTT, 2020 [R02] | Match initial outputs and empirical NTKs while finding a sparse initializer. | Eq. (14), optimization with mask updates and weight decay; public legacy JAX repository. | Closest omitted conceptual predecessor. Approximate transfer optimization and frozen masks differ from an exact global norm/count extremum. In a linear model, a rank-`d` probe design already identifies the full `G`; finite probes alone do not distinguish this project. |
| Khodak et al., 2021 [R05] | Initialization and regularization of factorized layers; exact two-factor update expansion. | Appendix B and model experiments. | The second-order expansion behind first-step cancellation already exists. The corollary is not a standalone novelty claim. |
| Dai et al., 2021 [R06] | Minimum factor norm for a prescribed end-to-end linear map. | Theorem 1 gives a Schatten-power representation cost. | Reuse factorization inequalities and the final-zero-head baseline. Fixing a nonzero tangent at output zero adds a different constraint. |
| Flennerhag et al., WarpGrad, 2020 [R07] | Learn a Jacobian-induced preconditioner across tasks. | Geometry derivation and several nonlinear learning settings. | Learning a favorable metric is established; those benchmark scores do not constitute a controlled comparison here. |
| Nikishin et al., plasticity injection, 2023 [R08] | Preserve predictions with a trainable fresh head and frozen cancellation copies. | NeurIPS experiments and explicit function-preserving construction. | Broad prediction-preserving plasticity is known. Count total stored parameters as well as trainable ones. |
| Classical LMMSE and reduced-rank estimation [R09] | Optimize linear prediction using covariance and rank restrictions. | Closed-form covariance/whitening/SVD solutions. | Direct `Gz`, rank selection, and spectral shrinkage are established ingredients. |
| Sun et al., 2021 [R10] | Optimize representation dimension and spectral weighting for few-shot regression. | Weighted interpolation theory, covariance-error analysis, experiments. | Generic spectrum-plus-rank optimization is known; their estimator and constraints differ from the present finite-step `Gamma` geometry. |
| Tripuraneni et al., 2021; Even and Massoulié, 2025 [R11], [R12] | Learn shared task representations from multiple tasks. | Moment estimators, rank/nuclear-norm methods, transfer and sample-complexity analyses. | Prior orientation is not free; reuse strong representation-learning controls and negative-transfer analysis. |
| Jin et al., 2024 [R13] | Generalized ridge using estimated task covariance. | Random-effects theory and covariance-estimation methods. | Covariance-aware ridge/empirical Bayes is a necessary stronger practical reference, not a new extension invented here. |
| Mulayoff and Michaeli, 2020; Marion and Chizat, 2024 [R14], [R15] | Sensitivity/flatness extrema at prescribed linear predictors. | Jacobian/Hessian formulas and sharpness bounds. | Close mathematical neighbors; prescribing the entire tangent while minimizing norm/width differs from optimizing one sharpness statistic. |
| Rosseau et al., 2026 [R16] | Preserve plasticity using dynamical isometry and AdamO. | Verified ICML 2026 proceedings and inspected theory discussion. | NTK isotropy as a plasticity concept is current prior art; continual nonlinear training is outside this one-step theorem. |
| Braun et al., 2025; Theiss et al., 2026 [R26], [R27] | Function-preserving parameterizations, hidden representation geometry, minimum weight norms, and parameter-noise robustness. | Exact linear results and nonlinear symmetry-orbit characterizations. | Close fixed-function neighbors. Their inspected norm results constrain hidden activation geometry; the object here is the full all-parameter tangent. Recheck these connections before any broader robustness claim. |

### 3.2 Answers to the novelty questions

**What exists?** Tangent kernels, zero-output trainable initializers, learned preconditioners, factorization costs, representation-rank selection, reduced-rank estimation, and empirical Bayes.

**What can be reused?** NumPy/SciPy linear algebra and constrained solvers; the source's formulas and small fixtures; complex-step differentiation of matrix products; standard ridge/posterior solvers; published moment estimators. Reuse source code only when its actual license permits it. The attached bundle has no verified redistribution license; use it for this authorized investigation, retain attribution, and do not silently label its contents permissively licensed.

**What is substantively different, if anything?** The exact functional

\[
\Gamma(G)=\operatorname{tr}G-(d-1)\lambda_{\min}(G)
\]

combined with the sharp all-trainable width requirement, and the joint frontier these constraints induce. For width `m`, positive floor permits at most `m-1` excess directions, whereas zero floor permits `m` nonzero directions. This precise structural tradeoff is the candidate contribution.

**Why could it matter?** It reveals what a particular parameterization can prepare under simultaneous resource caps, supplies an exact reference for numerical initialization procedures, and predicts when sacrificing one directional degree of freedom to retain a shared isotropic response is worthwhile. Operational value remains conditional.

**Which experiment isolates it?** Compare the floor-family and rank-family optima at the same `L,B,P_cap,S,n,sigma2`, with exact rational or sufficiently validated numerical bounds; then realize both and confirm the same risks from actual simultaneous network updates. Verify direct `Gz` equality. Repeat after learning `S` from exactly the same source data for every method.

**What would invalidate novelty?** A prior result with an equivalent zero-map derivative constraint and the same norm/width frontier, even under different notation or a matrix-product sensitivity formulation. If found, relabel the relevant parts as replication/application. A new name, implementation, or combination of standard shrinkage with the source formula is not sufficient novelty.

### 3.3 Direction selection and exclusions

| Direction | Decision and technical reason |
|---|---|
| Rerun the original norm-only phase plot | Required foundation, insufficient as the full research contribution. It is already in the bundle. |
| Joint norm/parameter-count frontier for arbitrary spectra | **Primary.** Directly extends the given mechanism, resolves a real resource omission, has a global reduction and cheap discriminating tests. |
| Estimated-prior and input-shift robustness | Required after the oracle frontier for any useful-learning claim; established estimation methods, potentially new empirical boundaries. |
| End-to-end nonlinear/continual-learning competition | Defer. It changes the mechanism and cannot be justified by a successful first-step linear calculation. |
| Vector-output, tied-weight, or multiple-step exact theory | Evidence-triggered later work. Operator-valued kernels and moving geometry require new arguments. |
| Post-update norm or finite-precision resource model | Conditional pivot if the fixed-learning-rate resource interpretation proves unhelpful; Section 4.7 supplies a diagnostic identity. |

No nonlocal benchmark number will be used as a direct head-to-head result. Original NTT, WarpGrad, plasticity-injection, and AdamO experiments provide conceptual context, not scores for the local regression population. The NTT README was inspected, but its historical JAX stack and code-reuse license were not verified for the target Mac [R28]. The exact oracles and closed-form baselines here are stronger for the stated mathematical claim than an arbitrarily ported neural optimizer.

## 4. Mechanism and implementation specification

### 4.1 Exact realization theorem from the supplied note

For a PSD matrix `G`, let `lambda=lambda_min(G)` and

\[
\Gamma(G)=\lambda+\sum_i(\mu_i-\lambda),
\]

where `mu_i` are its eigenvalues. The sum-of-gaps expression is preferable numerically to subtracting nearly equal traces.

Under Section 2's architecture and coordinates, the supplied theorem is

\[
E_{\min,L}(G)=(L-1)\Gamma(G)^{1/(L-1)}.\tag{2}
\]

For `G != 0`,

\[
m_*(G)=\operatorname{rank}(G-\lambda I)+\mathbf 1_{\lambda>0},\qquad
P_{\min,L}(G)=(d+1)m_*(G)+(L-2).\tag{3}
\]

An architecture realizes the target iff its first hidden width is at least `m_*`; all later hidden widths can be one. Norm and count minima are attained simultaneously. For `G=0`, use all-zero weights and width one; minimum positive-width architecture count is `d+L-1`, not zero.

**Proof outline sufficient to audit the reduction.** Set `R_l=W_l...W_1`, `R_0=I`, and let `b_l.T=W_L...W_(l+1)`, with scalar `b_L=1`. Then

\[
G=\sum_{\ell=1}^L\|b_\ell\|^2R_{\ell-1}^TR_{\ell-1}.\tag{4}
\]

Let `p_0=1` artificially, `p_l=||R_l||_F^2` for `l>=1`, and `q_l=||b_l||^2`. Define `alpha_l=p_l/p_(l-1)` and `beta_l=q_(l-1)/q_l`, assigning zero for zero denominators; vanishing products stay zero. The null condition gives `alpha_L=beta_1=0`. Orthogonally project each `W_l` onto the backward direction and its complement. The projected part has squared norm `beta_l` and annihilates `R_(l-1)`; submultiplicativity bounds the complement by `alpha_l`. Hence `||W_l||_F^2 >= alpha_l+beta_l`.

Since the first tangent term is `q_1 I`,

\[
\Gamma(G)\le q_1+\sum_{\ell=2}^Lq_\ell p_{\ell-1}
\le\prod_{j=1}^{L-1}(\alpha_j+\beta_{j+1})
\le\left(\frac{E}{L-1}\right)^{L-1}.
\]

The middle product contains all nonnegative terms of the preceding sum; the last inequality is AM-GM. This proves the lower bound. For width, (4) has form `G=q_1 I+W_1.T H W_1`, `H PSD`. If `m_1<d`, its smallest eigenvalue is `q_1`; if `q_1>0`, the nonzero backward vector is in `null(W_1.T)`, costing one rank. If `m_1>=d`, `m_*<=d` makes the lower bound automatic. The construction below proves attainability and the count in (3).

### 4.2 Canonical constructor and exact update

For nonzero `G`, set

\[
e=\Gamma(G)^{1/(L-1)},\quad t=e^{L-2},\quad p=\sqrt t.
\]

Take one row of `W` for each positive eigen-gap `mu_i-lambda`:

\[
W_i=\sqrt{(\mu_i-\lambda)/t}\;v_i^T.
\]

If `lambda>0`, append one zero row and put `sqrt(lambda/t)` on the corresponding coordinate of `a`; all other coordinates of `a` are zero. If `lambda=0`, use `a=0` with length `rank G`. The first two factors are `W` and `a.T`; append `L-2` scalar factors `sqrt(e)`.

This gives

\[
W^TW=(G-\lambda I)/t,\quad \|a\|^2=\lambda/t,\quad W^Ta=0.
\]

For a differentiable loss with effective-coefficient gradient `g` at zero, compute all gradients from the **old** weights:

\[
a'=a-\eta pWg,\qquad W'=W-\eta p\,ag^T.
\]

The scalar suffix has zero initial gradient. The quadratic cross term in `p W'.T a'` vanishes because `W.T a=0`, so

\[
w'=-\eta Gg=\eta Gz\quad\text{for the stated squared loss}.\tag{5}
\]

Do not implement these assignments sequentially using an already updated factor. Do not freeze the initially zero row: its trainability supplies the isotropic tangent term. After this first update, the geometry generally changes; (5) is not a formula for subsequent steps. The factorized expansion is related to existing work [R05].

**Reference pseudocode; the agent must implement it:**

```python
def construct_from_spectrum(V, mu, depth, exact_support):
    # V has orthonormal eigenvectors as columns; mu is ascending, PSD.
    # exact_support records analytic zeros/multiplicities when available.
    lam = mu[0]
    gaps = mu - lam
    gamma = lam + sum(gaps)
    if gamma == 0:  # explicit exact zero case
        return [zeros((1, len(mu)))] + [zeros((1, 1)) for _ in range(depth - 1)]
    e = exp(log(gamma) / (depth - 1))
    log_t = (depth - 2) * log(e)
    # Reject overflow/underflow; do not silently replace a small nonzero gap.
    indices = positive_gap_indices(mu, exact_support)
    W = stack([exp(0.5 * (log(gaps[i]) - log_t)) * V[:, i]
               for i in indices], allow_empty=True)
    a = zeros(len(indices) + int(lam > 0))
    if lam > 0:
        W = append_zero_row(W)
        a[-1] = exp(0.5 * (log(lam) - log_t))
    layers = [W, a[None, :]]
    layers += [array([[sqrt(e)]]) for _ in range(depth - 2)]
    return layers
```

`zeros`, `stack`, and related functions here denote ordinary array operations, not a promised package API. Every layer must own a separate mutable array. For a supplied compact `G=bI+U diag(u) U.T`, avoid a dense eigendecomposition: use its exact branch/support metadata to construct the rows directly.

### 4.3 Joint norm and parameter-count reduction: new consequence derived here

Define

\[
k=\left(\frac B{L-1}\right)^{L-1},\quad
m=\min\left(d,\left\lfloor\frac{P_{\rm cap}-(L-2)}{d+1}\right\rfloor\right).
\]

If `m<1`, no permitted positive-width architecture exists. For `B=0`, only `G=0` is feasible. Otherwise (2)-(3) reduce the problem to `Gamma(G)<=k` and `m_*(G)<=m`.

Let `S=V diag(s) V.T`, with `s_1>=...>=s_d>=0`, `sum(s)=1`; define `q_i=c s_i+rho`.

**Why it suffices to align gains with the largest task eigenvalues.** In an arbitrary metric eigenbasis, write `t_i=v_i.T S v_i>=0`; risk is

\[
1+\sum_i\{\rho g_i^2+t_i(cg_i^2-2g_i)\}.
\]

Clip any `g_i>1/c` down to `1/c`. This reduces risk, cannot increase `Gamma`, and cannot increase the exact width requirement. If the old minimum was already below the cap it stays fixed while large gaps shrink; if all gains exceeded the cap the result is isotropic and costs only `1/c`. On `[0,1/c]`, `h(g)=c g^2-2g` decreases. The trace rearrangement inequality therefore minimizes `tr(S h(G))` by pairing the largest `s_i` with the largest `g_i`. All constraints depend only on eigenvalues. Thus a globally optimal aligned metric exists.

This argument is needed because the width-feasible set is nonconvex. Averaging over rotations can increase width and is not valid here. Equal task eigenvalues may permit many optimal orientations; claim uniqueness only within each strictly convex scalar program, not for the global matrix.

The aligned feasible set is covered by the union of two families:

**F: isotropic floor plus excess directions.** Put `j=min(m-1,d-1)`. Choose `b>=0` and `u_i>=0` for `i<=j`, with

\[
b+\sum_{i=1}^j u_i\le k,\qquad
g_i=b+u_i\ (i\le j),\quad g_i=b\ (i>j).
\tag{6}
\]

For `b>0`, required width is one plus the number of positive increments. Zero increments include all smaller supports. The closed `b=0` boundary is allowed and overlaps the rank family. In variables `x=(g_1,...,g_j,b)`, its objective is

\[
1-2\left(\sum_{i\le j}s_i g_i+S_{\rm tail}b\right)
+\sum_{i\le j}q_i g_i^2+Q_{\rm tail}b^2,
\tag{7}
\]

where `S_tail=sum_(i>j)s_i`, `Q_tail=sum_(i>j)q_i>0`. Constraints are `g_i-b>=0`, `b>=0`, and `k-sum(g_i)+(j-1)b>=0`. At `j=0`, this is scalar isotropy with `b=min(k,1/(c+d rho))`.

**Z: trace-budget rank family.** Put `p=min(m,d)`. Choose `g_i>=0` for `i<=p`, zero elsewhere, with `sum(g_i)<=k`. Its objective is

\[
1-2\sum_{i\le p}s_i g_i+\sum_{i\le p}q_i g_i^2.\tag{8}
\]

For `m<d` this is exactly the zero-floor family. When `m=d`, a fully positive solution of (8) need not actually have zero minimum; (8) is then a redundant trace-budget subset, while F already covers all feasible spectra. Do not claim `Gamma=trace` for such a full-rank matrix.

The rank-family solution is

\[
g_i=\max\{0,(s_i-\nu/2)/q_i\}.
\]

Use `nu=0` if the sum fits; otherwise solve the monotone budget equation with bracket `[0,2s_1]`. This is standard shrinkage/root-finding, not a novel optimizer. F is another strictly convex QP with linear constraints. Solve both and compare their risks using the verification levels below. At a numerical tie, retain both candidates and their uncertainty; select the smaller actual trainable count only as a declared implementation tie-break.

The union is complete: a positive-minimum feasible matrix has at most `m-1` positive eigen-gaps and belongs to F after alignment; a zero-minimum feasible matrix has rank at most `m` and belongs to Z. This gives the global joint frontier conditional on the supplied realization theorem. It is a new derivation in this investigation, with historical originality unresolved.

### 4.4 Solver, exact certificates, and floating-point diagnostics

The initial implementation can use `scipy.optimize.minimize(method="SLSQP")` with analytic gradients and **linear** constraints [R21]. Do not optimize a nonsmooth minimum-eigenvalue constraint for the production solver. Do not treat `success=True` as an optimality certificate.

Both branch programs can be written

\[
f(x)=1+\ell^Tx+x^TDx,\quad Ax+o\ge0,
\]

with diagonal `D>0` in the variables used in (7)-(8). An arbitrary nonnegative multiplier vector `lambda_dual` gives a valid lower bound

\[
d(\lambda_{\rm dual})=
1-\lambda_{\rm dual}^To
-\frac14(\ell-A^T\lambda_{\rm dual})^TD^{-1}
(\ell-A^T\lambda_{\rm dual}).\tag{9}
\]

Recover multipliers by nonnegative least squares on the active constraints, `A_active.T lambda ~= grad f(x)`. Evaluate primal feasibility, dual feasibility, stationarity, complementarity, and the primal-minus-dual gap independently. In exact arithmetic, the smaller of the two branch lower bounds and the smaller feasible upper bound bracket the global value. Retain both branch records even when one loses.

**Verification levels are explicit.** Ordinary float64 objective/dual calculations and small KKT residuals are numerical diagnostics; they do not automatically enclose roundoff and therefore are not rigorous certified intervals. Record `certificate_level="float64_diagnostic"` unless a rigorous method is used. For the small rational oracle fixtures, encode the stated spectrum, `n`, `sigma2`, and `k` as exact fractions, use the numerical solution only to propose an active set, solve that KKT linear system with `fractions.Fraction`, and check primal feasibility, nonnegative multipliers, stationarity, complementarity, and objective/dual equality exactly. Record `certificate_level="exact_rational"` only after these checks. The `k=3/5` fixture below has already passed such an exact check. An outward-rounded interval calculation with explicit enclosures is another legitimate level, but is optional and has not been performed here. A higher-precision decimal calculation without error enclosures remains a diagnostic.

For oracle Gaussian QPs, rescale variables by `min(k,1/c)` when `k>0`, using the proved clipping bound, and remove the constant `1` from the numerical objective. This handles both tiny budgets and very large inactive caps. Scale the remaining objective by a recorded characteristic linear/quadratic magnitude, so tiny improvements are not lost to an absolute `ftol`. The `1/c` bound is not automatically valid for empirical B11 objectives; use their observed curvature/gradient scale and actual feasible cap instead. Evaluate final risks and diagnostics in original units. If roundoff leaves a tiny primal violation, project to the linear feasible set, recompute the objective, and report the correction. Exact certification additionally requires verified feasibility after rounding.

For B11's potentially singular, nondiagonal convex quadratic, avoid inventing an inverse of its Hessian. With feasible `x`, nonnegative multipliers `lambda_dual`, slack `s=Ax+o`, and residual `e=grad f(x)-A.T@lambda_dual`, convexity and the branch domain `0<=x_i<=k` give the exact-arithmetic lower bound

\[
f_*\ge f(x)-\lambda_{\rm dual}^Ts-k\|e\|_1.
\]

This follows by applying the supporting-hyperplane inequality at `x` to every feasible candidate. It supplies a useful residual bound even for a singular Hessian; ordinary floating-point evaluation still has the qualification above. If regularization is used, identify whether the bound is for the regularized or unregularized objective.

Start ordinary-scale CPU-double checks with relative residual goals around `1e-10`, reflecting matrix size and conditioning. The **scientific** acceptance condition is stronger: optimization and rounding uncertainty must be well below the claimed risk gap or target kernel accuracy. If not, refine the solver, use exact rational KKT arithmetic for rational cases, use explicit roundoff enclosures, or mark the comparison numerically unresolved. There is no universal `1e-10` license to claim a smaller effect. Keep numerical and rigorous certificate language distinct in plots and reports.

An independent verifier must not import the production objective or constructor. For `d<=6`, enumerate all allowed supports, solve their scalar QPs, and check random rotations. At larger `d`, use the proof and dual bounds, not exponential enumeration.

### 4.5 Numerical conventions and invariants

1. **Precision:** CPU float64 for all core calculations; complex128 for complex-step checks. Float32 is a separate sensitivity/performance arm.
2. **PSD/symmetry:** reject materially asymmetric/non-PSD inputs. Symmetrize only rounding-scale skew and record its norm. For arbitrary dense matrices use a scale-aware tolerance such as `64*eps*d*||G||_2`, with an explicit exact-zero branch. Do not use `max(1,||G||)` to erase small legitimate spectra. Record any PSD projection separately from target-realization error.
3. **Exact rank:** analytic fixtures and QP branch metadata define intended zeros and repeated eigenvalues. An eigensolver tolerance cannot certify algebraic rank. For data-derived dense matrices, report approximate rank and retained/dropped eigenvalue mass with tolerance sensitivity.
4. **Kernel comparison:** report absolute and relative Frobenius/operator errors, null-output residual normalized by product norms, budget slack, actual `E`, and actual `P`. Define relative error as zero for two exact-zero objects; handle a zero denominator explicitly otherwise.
5. **Compact representation:** store `floor`, orthonormal `basis`, nonnegative `increments`, branch/support metadata, and units. Apply `Gz = floor*z + basis @ (increments*(basis.T @ z))`. Dense materialization is for small audits.
6. **Multiplicity:** compare projectors or matrices, not raw eigenvectors, when signs or degenerate bases differ. Fix a deterministic sign convention for serialization only.
7. **Depth:** use logs for powers of `Gamma`. Detect overflow/underflow before constructing weights. Near machine extremes, narrow the numerical claim rather than silently changing precision or gain.
8. **Optimizer state:** plain simultaneous SGD, no momentum, clipping, weight decay, or adaptive metric in the exact mechanism. Generic optimizations that preserve the map must be available to equivalent baselines too.

### 4.6 Smallest discriminating experiments and headroom

The smallest complete experiment uses `d=6,L=2,n=4,sigma2=1`, source spectrum `(1/2,1/2,0,0,0,0)`, and `eta=1`. For the source's norm-only optimum, define

\[
\alpha=c+r\rho,\quad\beta=(d-r)\rho,\quad
k_0=\frac{r-1}{\alpha(r-1)+\beta r},\quad k_1=r/\alpha.
\]

The source has `k_0=0.16`, `k_1=8/9`. For completeness, with `D=alpha(r-1)^2+beta r^2`,

\[
(a,b)=\begin{cases}
(k,k),&0\le k\le k_0,\\
(((r-1)^2+\beta r k)/D,\;(r-1)(r-\alpha k)/D),&k_0<k<k_1,\\
(1/\alpha,0),&k\ge k_1.
\end{cases}
\]

`G=aP+b(I-P)` and risk is `1-2a+alpha*a^2+beta*b^2`. Exact required widths are `1`, `r+1`, and `r` in the respective positive-budget regimes. For arbitrary nonscalar `S`, the source isotropic onset is

\[
k_{\rm iso}=\frac{1-s_{\max}}{c(1-s_{\max})+(d-1)\rho},\quad d\ge2.
\]

Do not apply this expression at `d=1`, where it can become `0/0`; solve that scalar case directly. The width restriction can change the later solution and need not preserve uniqueness.

Freshly calculated joint risks for the source example are:

| `k` | Width cap 1 | Width cap 2 | Width cap 3 |
|---:|---:|---:|---:|
| 0.08 | 0.867200000000 | 0.867200000000 | 0.867200000000 |
| 0.16 | 0.788800000000 | 0.788800000000 | 0.788800000000 |
| 0.40 | 0.764705882353 | 0.690000000000 | 0.660487804878 |
| 0.80 | 0.764705882353 | 0.560000000000 | 0.559024390244 |
| 8/9 | 0.764705882353 | 0.555555555556 | 0.555555555556 |

At `k=0.4`, the extra width reduces expected error by `0.029512195122`, about `4.28%` of `0.69`, but raises minimum trainable count from 14 to 21. At `k=0.8`, the gap is only `0.000975609756`; after saturation it is zero. Present the resource expense alongside the gain.

A second fixture avoids making the whole study about a perfectly flat subspace: use `S` eigenvalues `(.50,.25,.15,.07,.02,.01)`, `d=6,n=4,sigma2=1,m=3`. At `k=0.4`, fresh QP outputs were floor risk `0.697161477190`, rank risk `0.759354838710`, and isotropic risk `0.764705882353`. Here Z uses only two positive gains, so its minimum actual count is 14 versus F's 21; this is an equal-cap comparison, not an equal-actual-count result. At `k=1`, the ranking reverses: rank `0.668127428127`, floor `0.673827673828`. The method must select the rank family there.

**Primary mechanism fixture with equal actual resources.** At the same richer spectrum, choose `k=3/5`, `L=2`, and `P_cap=21`. The exact F solution is

\[
(g_1,g_2,b)=(2963,1978,798)/6905,\qquad
R_F=186321/276200\approx0.674587255612.
\]

The exact Z solution is

\[
(g_1,g_2,g_3)=(1013,518,194)/2875,\qquad
R_Z=81247/115000\approx0.706495652174.
\]

All indicated gains/increments are positive. Both realizations have actual width three, 21 trainable entries, and norm `3/5`; the optimum isotropic risk is `13/17`, approximately `0.764705882353`. Only the budget constraint is active in each branch. Its nonnegative multipliers are `953/27620` for F and `2383/11500` for Z under the convention in (9), with all other multipliers zero. Direct exact-fraction checks give feasibility, stationarity, complementarity, and matching primal/dual values. Thus the exact risk gap is

\[
R_Z-R_F=1266883/39703750\approx0.031908396562.
\]

These are development arithmetic fixtures, including one exact certificate, rather than a protected empirical win. They motivate the finite-source investigation without establishing its outcome.

### 4.7 Learning-rate, post-update, and finite-step limits

For every homogeneous depth-`L` null network, Euler's identity gives `Theta.T grad_Theta Loss = L*w.T*g=0`. Therefore one simultaneous step obeys

\[
E_{\rm after}=E_{\rm before}+\eta^2g^TGg.\tag{10}
\]

This identity holds even for null networks whose finite output update is nonlinear. If all initial weights are multiplied by `s>0`, then `E` scales as `s^2` and `G` as `s^(2(L-1))`. In the canonical construction, inversely rescaling `eta` preserves the first-step predictor while driving initial norm to zero; the post-update term in (10) instead grows as `s^(-2(L-1))`. Measure this control. It exposes the limits of initialization norm rather than validating it as physical energy.

A single linear layer initialized at zero has `G=I` and zero initialization norm. It is outside the fixed-depth `L>=2` theorem but must appear in operational discussion. A fixed feature map also changes which quantities count as trainable resources.

Finally, zero output alone does not imply (5). An exact adverse fixture is

\[
W_1=(1,0)^T,\quad W_2=\begin{pmatrix}1&1\\0&1\end{pmatrix},\quad W_3=(0,1).
\]

It has scalar `G=3`, `E=5`, and initial output zero. For loss `0.5*(w-u)^2`, writing `h=eta*u`, its simultaneous update gives `w'=3h+h^2`. Preserve this fixture as a check that the harness does not incorrectly linearize every null network.

## 5. Baselines, data, metrics, and confirmatory protocol

### 5.1 Comparison contracts

Keep three result panels. The distinction prevents incompatible resource and information assumptions from manufacturing a win.

**Panel A: exact constrained geometry.** Fixed `L,B,P_cap,eta=1`, same true `S`, same `n,sigma2`, same loss normalization. Compare the exact mathematical risk, realization residuals, norm, and count. Equal caps do not mean equal actual usage: report both.

**Panel B: finite-data preparation.** Every method receives the same source-training and source-validation observations. True task coefficients, true covariance/eigenvectors, and true test noise are hidden from methods except in explicitly marked oracle arms. The evaluator may use true coefficients to measure error. Count prior estimation, tuning, eigenvectors, and all stored/frozen quantities.

**Panel C: operational alternatives.** Give locally executable direct, ridge, and Bayesian methods their natural algorithms and the same permitted observations. Report risk versus measured time, memory, stored scalars, and preparation amortization. The theorem's raw network norm is not a meaningful matched budget for software ridge or a fixed direct preconditioner; label that column not applicable rather than assigning them artificial costs.

### 5.2 Mandatory local methods

| ID | Method, exact configuration, and purpose |
|---|---|
| B00 | Zero estimate. Analytic risk 1. Harness negative control. |
| B01 | Best isotropic map, `g I`, with `g=min(k,1/(c+d*rho))` in the oracle arm. Width one. In learned arms also fit the scalar on source validation, respecting the same cap. |
| B02 | Best depth-2 final-zero-head preparation. Solve Z exactly: top `m` gains, trace budget, optimized individual eigenvalues. This is much stronger than assigning one arbitrary subspace gain or random features. |
| B03 | F alone, with the verification level of Section 4.4 recorded. Mechanism ablation. |
| B04 | Joint frontier `min(F,Z)`, compiled to the canonical network, all entries trainable. Proposed primary realization. |
| B05 | Direct compact operator for **each** selected B01-B04 matrix. Same low-rank-plus-floor structure and same source information. First-step outputs must agree with the corresponding network. This is an equality control and a serious operational baseline. |
| B06 | Width-unconstrained `Gamma`-budget spectral QP: variables `g_i,t`, constraints `g_i>=t>=0`, `sum(g_i)-(d-1)t<=k`, objective `1+sum(q_i*g_i^2-2*s_i*g_i)`. Independent reference for the original phase formulas; a lower bound when the width cap is restrictive. |
| B07 | Unconstrained fixed linear map `S (cS+rho I)^(-1)`, applied by eigengains `s_i/(c*s_i+rho)`. This is the classical best linear map of `z`, not generally a Gaussian posterior conditioned on `X`. Check whether its actual norm/count happens to fit before labelling it outside the cap. |
| B08 | Zero-initialized single-layer scalar-step GD, `theta_hat=eta*z`, and tuned ordinary ridge. These expose the effects of fixing depth and step size. Their operational resource class differs. |
| B09 | Projected least squares and projected ridge using the same allowed basis. Include an independent method-of-moments representation baseline following Tripuraneni et al. [R11], then strengthen it with validation-tuned ridge. Label that strengthening explicitly. |
| B10 | Oracle Gaussian posterior and plug-in empirical Bayes, with covariance shrinkage tuned on the same source-validation tasks. These are stronger practical references [R13], [R17]. |
| B11 | Empirically fitted gains in the same estimated basis, under F and Z constraints. Fit the source-task query loss directly as a convex QP. This controls for a weak moment plug-in rule, and gives both families the same generic optimization improvement. |

B02 is a faithful **depth-two** comparison. Construct it separately as `W=diag(sqrt(g_positive)) @ V_positive.T`, `a=0`; its norm is `tr G`. Do not route B02 through the minimum-`Gamma` constructor, which could insert an isotropic floor when its target is full rank and thereby change the baseline. B04 uses the canonical constructor. At larger depths, a true final-zero-head network has minimum preparation norm

\[
(L-1)\sum_i\lambda_i(G)^{1/(L-1)}
\]

when intermediate widths support the required rank, using the factorization result [R06]. An early zero second factor followed by scalar suffixes is a different architecture. Do not incorrectly assign a deep final-zero-head baseline `(L-1)*(tr G)^(1/(L-1))`. Keep depth-two risk comparisons primary; use depths 3, 4, and the source's 2-8 range for construction and limiting-case verification before making any depth-efficiency claim.

### 5.3 Exact stronger estimator implementations

Use solves, never explicit inverses in production:

\[
\widehat\theta_{\rm ridge}=(X^TX+\lambda_{\rm ridge}I)^{-1}X^Ty.
\]

Here `lambda_ridge` is the unnormalized normal-equation penalty. If another API minimizes mean squared loss, convert its penalty explicitly; do not lose a factor of `n`. Use the dual `n by n` solve when smaller. For a shared orthonormal basis `U`, projected ridge is

\[
U\{(XU)^T(XU)+\lambda_{\rm ridge}I\}^{-1}(XU)^Ty.
\]

For Gaussian tasks/noise, the exact posterior mean is

\[
\widehat\theta_{\rm Bayes}=SX^T(XSX^T+\sigma^2I)^{-1}y.\tag{11}
\]

If `S=U diag(s) U.T` has rank `r`, use its square root `C=U diag(sqrt(s))` and solve the smaller of the `r`-dimensional primal and `n`-dimensional dual systems. With `S=P/r`, the primal penalty is `r*sigma2`, not `sigma2/r`. For `sigma2=0`, use a documented Moore-Penrose solve and report rank/conditioning; do not quietly add an unreported ridge. Gaussian posterior optimality concerns expected quadratic loss over all estimators with `X,y,S,sigma2`, so beating it reproducibly on that exact population would indicate a defect or changed assumptions. Non-Gaussian task/noise arms must relabel it as a Gaussian-prior reference.

Use Cholesky for positive definite systems, fall back to SVD for rank-deficient limits, and verify residuals. Apply compact/batched implementations to every eligible method. A dense `d by d` implementation of B05 would unfairly weaken the simplest exact competitor.

### 5.4 Finite-source-data preparation

The task prior is a second-moment matrix, not automatically an inherited free asset. Implement two established-style estimators and disclose the data they use.

**Split-task cross moments.** Each source task supplies two independent blocks A and B, each initially of size `n_source=n`. Let `z_A=X_A.T y_A/n_source` and similarly `z_B`. Then

\[
\widehat S_{\rm raw}=\frac1T\sum_{t=1}^T
\frac{z_{A,t}z_{B,t}^T+z_{B,t}z_{A,t}^T}{2},\qquad
\mathbb E\widehat S_{\rm raw}=S.\tag{12}
\]

This follows from conditional independence and isotropic input second moments. It is an elementary estimator, not a novelty claim. Project negative eigenvalues to zero and, in the **trace-one synthetic contract only**, normalize the retained trace to one. If no positive mass remains, use `I/d` and record the fallback. Record the raw negative mass and projection size; do not hide an unstable estimate.

**Moment representation comparator.** Compute the top eigenvectors of the source average of `y_i^2 x_i x_i.T`, following Algorithm 1 of [R11]. Under Gaussian isotropic inputs its expectation is `2S+(1+sigma2)I`. Use this to estimate directions; do not assume its biased eigenvalues are already estimates of `S`.

For each covariance estimate consider shrinkage

\[
\widehat S_\gamma=(1-\gamma)\widehat S+\gamma I/d,\qquad 0\le\gamma\le1.
\]

Choose `gamma` and any rank selection using source validation. All methods receive the same candidate estimates; allow method-specific validation choices because an EB optimum need not use the same shrinkage as a constrained first step. Include both oracle-known noise and a labelled estimated-noise arm. Under the trace-one synthetic model, `max(0,mean(y^2)-1)` is a simple source-only noise estimate; report its uncertainty. Observations within a source task share its coefficient, so source uncertainty estimates must resample or cluster by task, not pretend all labels are independent. Never apply the trace-one normalization or this noise formula to arbitrary real labels without a new justification.

Charge `2*T*n_source` source-training labels for the split-block estimator, plus every context/query label used by validation and tuning. Report task counts, labels per task, total labels, and acquisition/preparation computation separately. All compared learned-prior methods receive the same source tasks, block partitions, and candidate prior assets; source-task reuse does not create new independent information.

**Direct empirical spectral fitting, B11.** For each source task use A as adaptation context and B as query, then symmetrize by swapping the roles. For a rank branch, column `i` of the query design for gain fitting is `(X_B v_i)*(v_i.T z_A)`. For F with variables `(g_1,...,g_j,b)`, the final column is `X_B z_A` minus the sum of those `j` columns. Fit the average query squared loss under exactly the same linear constraints as (7)-(8). This is a convex quadratic program; its Hessian need not be diagonal or strictly positive. A singular solve is valid if objective/KKT residuals are checked. If regularization is needed, tune its magnitude, including zero, on source validation and disclose it. Reusing source data to estimate the basis and fit gains is meta-training, not protected-test leakage; source-validation selection and independent outer replications must account for resulting overfit.

The matched-basis comparison isolates the resource geometry. A separate best-practical comparison lets each baseline select its best permitted covariance/rank/gain-fitting variant on validation. Report both; do not claim practical advantage only against an intentionally weaker shared estimator.

### 5.5 Workload selection

The core needs no external data or pretrained weights. Generated tasks expose the entire proposed mechanism, so a small dimension is scientifically meaningful. Larger dimensions test implementation scaling and prior-estimation burden; they are not prerequisites for the exact theorem.

| Workload | Definition and purpose |
|---|---|
| W0: exact fixtures | The two `d=6` spectra in Section 4.6, plus `d=1`, zero `G`, isotropic `G`, singular `G`, repeated minima, rank-one tasks, and the finite-step counterexample. Development/harness checks. |
| W1: flat low-rank tasks | `d=64,r=8`, uniformly weighted task subspace, `n` around `r/2`, `r`, and `4r`, `sigma2=1`. Widths at 1, `r`, `r+1`, and unrestricted width expose the claimed tradeoff. Budgets include below onset, both sides of onset, an interior mixed regime, saturation, and above saturation. |
| W2: nonflat tasks | `d=64`, `s_i proportional to i^(-3/2)`, normalized to trace one; `n=8`, `sigma2=1`, widths around 4, 8, and 16. This tests directional allocation without an exact finite-rank subspace. Budget range is anchored to the unrestricted metric's `Gamma`, with adaptive deterministic refinement near branch changes. |
| W3: adverse spectral controls | Rank-one task covariance; isotropic task covariance; nearly repeated top eigenvalues; spectrum with a long small tail. These test absence of an isotropic interval, nonunique bases, and approximate-width sensitivity. |
| W4: prior and design mismatch | Estimated priors; source/test eigenspace rotations; signal leakage outside the learned subspace; Gaussian anisotropic inputs; isotropic non-Gaussian inputs. Specify each violation independently before combining them. |

The dimension/ratio choices express distinct mechanisms: low-rank signal versus ambient noise, under/overdetermined task adaptation, and competition for a hidden coordinate. They are not maximum scale or experiment-count limits. For deterministic frontier plots refine the budget mesh until interpolation error in risk is below the numerical reporting target and all detected branch switches are bracketed. Do not keep adding random benchmarks after a desired result appears.

Generate task-space rotations from QR of a seeded Gaussian matrix and fix column signs for serialization. Random rotations test coordinate invariance; record the rotation seed and covariance hash. For isotropic inputs, an implementation may simulate in the task eigenbasis to save work, but must also verify genuinely rotated inputs/networks. Label coordinate-basis shortcuts in timing comparisons.

### 5.6 Optional natural-covariate stress test

A small, legally clear optional input pool is **UCI Wine Quality**, dataset 186, DOI `10.24432/C56S3T`, CC BY 4.0 [R22]. Use the white-wine file's 11 physicochemical variables; discard the quality label. Its documented 4,898 rows are a provenance expectation, not a substitute for validating downloaded bytes. Download through the official dataset link, record exact URL, archive/file SHA-256, citation, and license snapshot. Do not infer an immutable version number that UCI does not supply.

This is explicitly **semi-synthetic**: draw artificial task coefficients and labels on real covariates. It tests sensitivity to non-Gaussian inputs, not wine prediction or a real multi-task application. Group identical feature rows before a deterministic 60/20/20 development/validation/protected-pool split to prevent duplicate-row leakage. Keep within-pool multiplicities; do not silently remove outliers. Fit centering, scaling, and any whitening only on the development pool; save the transform. Retain all positive-variance directions, or record an amendment if a numerical rank deficiency forces a different dimension. Whitening changes the coordinate system and its norm interpretation, so label this as a robustness arm.

Draw independent synthetic task coefficients and observation noise conditional on each fixed input pool. Use disjoint sampled context/query indices within a task when possible. Task-level uncertainty describes the conditional synthetic task distribution, not the population of all wines. A second data-pool split is a sensitivity check, not thousands of new real-world samples. The Gaussian risk optimum does not transfer automatically; compare measured MSE and practical baselines. If the asset is unavailable or licensing changes, omit this optional arm and retain the complete synthetic core.

### 5.7 Tuning fairness and versions

The baseline methods are specified mathematically above, so their operative version is the repository commit plus the locked numerical dependencies. Record the exact resolved package versions, BLAS implementation, source revision, solver options, rank tolerance, preprocessing hash, and all candidate configurations. The versions used for this investigation are historical measurement context, not an assertion that they are the newest or verified Mac builds.

For ridge penalties, start a logarithmic grid around the scale of `trace(X.T X)/d`, include zero/pseudoinverse and the zero-estimator limit, and refine around validation minima. If the best finite penalty is on a boundary, extend that boundary before freezing. For shrinkage, include 0 and 1 and refine the interior until validation uncertainty exceeds differences between adjacent candidates. Do not impose the same arbitrary trial count on an exact analytic method and a tuned estimator; record time and candidate evaluations so the tradeoff is visible.

Use the same independent validation tasks for paired candidate comparisons. Stop tuning when boundary checks and objective resolution are satisfied, not when the proposal first wins. Both floor and rank families receive the same estimation and fitting improvements. The chosen baseline for a confirmatory contrast must be fixed by validation, never selected as the best test result after inspection.

### 5.8 Metrics and primary endpoints

**Primary mathematical endpoint:** at each declared `(S,n,sigma2,L,B,P_cap)` configuration, the optimum of (1), its verification level, and risk gaps to best F, best Z, isotropy, and the unconstrained lower reference. Use exact rational certificates for the stated rational fixtures and separately labelled numerical bounds for general floating-point cases. No p-value is needed for an exact deterministic calculation. Demonstrate each comparison with a gap larger than its combined numerical uncertainty.

**Primary empirical endpoint for learned priors:** the paired difference in conditional expected squared coefficient error, averaged over independent source-training/validation replications:

`D_j = R_S(G_baseline,j) - R_S(G_joint,j)`.

The source data are random, so this remains an empirical investigation of preparation. Once a replication has fitted/selected its metric, that metric is fixed before any new task. Equation (1) therefore gives its exact conditional new-task risk, even when the estimated metric and true `S` do not commute. Use the matrix formula, not an eigenvalue pairing that assumes alignment. The evaluator alone receives true `S` and true noise; they must not enter covariance fitting, gain fitting, method selection, or adaptation. Validation still uses permitted source context/query observations. This exact conditional-risk evaluation removes unnecessary inner Monte Carlo noise from the primary contrast; it does not make prior estimation exact.

For compact `G=bI+U diag(u) U.T`, compute `t_i=U[:,i].T @ S @ U[:,i]` and `h_i=2*b*u_i+u_i**2`. Then `tr(GS)=b+sum(u_i*t_i)`, `tr(G^2 S)=b*b+sum(h_i*t_i)`, and `tr(G^2)=d*b*b+sum(h_i)` under `tr(S)=1`. These give (1) without materializing `G`; use a factorized true `S` in the evaluator when available. Validate this expression against dense matrix traces on rotated small fixtures. Analytic evaluation still has floating-point arithmetic error, which must remain below the statistical precision target.

Positive `D_j` favors the joint method. For this **mechanism** contrast, the baseline selector may choose only B01 or B02, including their source-estimation and B11 empirical-gain-fitting variants. The joint selector may choose F or Z with the same estimation/fitting variants. Both share the same source observations and caps. B05 is an equality control, not an eligible superiority target. Ridge, empirical Bayes, and oracle methods appear in separately labelled operational or information-reference contrasts. Freeze the **selection procedure and candidate sets**, not one fitted prior: each independent replication trains and validates its own methods using that frozen procedure.

Freeze one primary scenario and contrast using the oracle mechanism and source-development results before generating protected source-replication keys. The richer `d=6,n=4,sigma2=1,L=2,m=3,k=0.6,P_cap=21` scenario is the initial mechanism population because the exact oracle fixture has equal actual resources and nonzero headroom. Learned methods may choose smaller active supports; report their actual counts. W1/W2 are independent transfer confirmations, not evidence that the initial scenario represents all tasks. If selecting a different primary scenario after pilot inspection, record why and reserve fresh confirmation.

Report absolute difference, relative difference when the denominator is meaningful, and intervals. Include each family's conditional results, not only their selected minimum. Additional guardrails are tail loss, regressions under task shift, kernel/zero residuals, preparation time, adaptation latency, prediction latency, peak memory, serialized total bytes, actual trainable count, initialization norm, post-update norm, and numerical failure rate. No scalar score should conceal these tradeoffs.

### 5.9 Pilot, sampling, and uncertainty

Use a **fixed-sample confirmatory design**. Pilot task IDs and source-training replications are permanently exploratory and excluded from confirmation. Use the pilot to estimate variability, covariance-estimation error, numerical sensitivity, throughput, and required sample sizes. Freeze the full protocol in a hashed manifest before accessing protected results.

For a fixed oracle `S` and fixed first-step maps, calculate the risk exactly from (1); there is no sampling unit or confidence interval for that deterministic quantity. Use a limited, variance-calibrated Monte Carlo check to validate the actual network update and risk formula. For X-dependent ridge/posterior estimators, or a population without the analytic moment formula, independent new tasks are the sampling unit. Compute per-task paired losses and their variance.

For **estimated-prior fixed maps**, independent source-training/validation replications are the units. Calculate one exact conditional `D_j` per replication as above. Choose a target CI half-width `h` tied to the oracle headroom or an explicit application cost decision; `h=gap/4` is an initial precision specification, not a minimum improvement required for success. With pilot variance `s_D^2`, plan approximately

\[
J\approx (z_{1-\alpha/2}\,s_D/h)^2,
\]

accounting conservatively for uncertainty in the pilot variance and checking calibration in a second pilot block. Use a prespecified Student interval or outer-replication bootstrap at fixed final `J`, with enough independent replications to justify that method from the pilot diagnostics. No inner test-task count is needed for this endpoint. Finite-sample interval coverage remains an approximation and must be stated; exact conditional risks do not remove uncertainty about the distribution of prepared metrics.

When evaluating estimated-prior **X-dependent estimators** or nonanalytic populations, each outer replication fits its own prior/hyperparameters, then evaluates paired methods on `N` independent new tasks. Average test-task differences within each replication first. Thousands of test tasks conditional on one fitted prior do not constitute thousands of independent prior estimates. Estimate pilot variance components `v_between` and `v_within`; then

\[
\operatorname{Var}(\bar D)\approx(v_{\rm between}+v_{\rm within}/N)/J.
\]

For that nested simulation only, measure preparation cost `a` and per-test cost `b`. A useful cost allocation starts near `N=sqrt(v_within*a/(v_between*b))` when both variance components are positive, then chooses `J` for the frozen precision target. For a fixed oracle use the same fixed-sample formula with task variance and `N` instead of `J`. If the variance is too unstable to plan, extend the pilot; do not treat two favorable seeds as adequate uncertainty estimation. Do not apply this nested allocation to the primary analytic conditional-risk endpoint, whose within-replication simulation variance is zero.

Choose source-task counts by a preparation-learning pilot: start where covariance estimation is possible, increase `T` geometrically, and track matrix error/eigenspace error and validation stability. Freeze levels that bracket the transition from poor to accurate priors. This selection is based on preparation accuracy and cost, not protected performance. A required `T` beyond feasible local time is a resource/inconclusive result, not grounds to pretend the prior is free.

Use one declared primary empirical contrast at `alpha=0.05`. For a small set of confirmatory secondary scenario contrasts, use Holm adjustment or simultaneous intervals and declare the family in advance. Exploratory frontier curves and robustness searches are labelled exploratory. Never repeatedly peek and extend a confirmatory run until significant. If the final result is underpowered, report the achieved interval/detectable effect and design a new independent confirmation after an amendment.

For synthetic squared errors with heavy tails, inspect pilot tail behavior and confidence-interval calibration. Avoid Gaussian-theory CIs when required moments do not exist. A Student-t input with more than eight degrees of freedom is a useful finite-moment non-Gaussian control; a deliberately infinite-fourth-moment arm is a labelled failure-of-assumptions diagnostic, not an ordinary mean-risk test.

### 5.10 Harness validity and leakage controls

- Verify B05 equality with the actual network step; a significant risk difference is a defect, not a discovery.
- Test `G=0`, `G=I`, the analytic risk identity, a known positive F-versus-Z fixture, a known family reversal, and two duplicated implementations that should produce exactly zero paired differences.
- Keep true `theta`, `S`, and generator rotations in evaluator-only records. Method interfaces accept only declared prior assets and observed context data.
- Split by source task and outer replication. No new-task query labels may enter adaptation, prior fitting, hyperparameter choice, or normalization.
- Freeze source-validation choices before testing. Preserve all tried candidates and failed runs.
- Record whether uncertainty is conditional on a fixed covariance, a fixed natural input pool, or a distribution of estimated priors. Do not pool these estimands without an explicit population definition.
- Test common-random-number pairing without letting one method's failure or random draws change another method's data stream.

## 6. M3 Max feasibility and hardware-adaptive execution

### 6.1 Stack and verified compatibility

The core stack is a native ARM64 Python interpreter, NumPy, SciPy, and standard-library JSON/CSV/hash utilities. Matplotlib is optional for reporting; a minimal test runner can use `unittest`, with `pytest` an optional convenience. No trained model, API key, remote accelerator, or special equipment is required.

Official NumPy and SciPy release documentation verifies ARM64/macOS Accelerate support and suitable macOS wheels [R18], [R19]. This is support evidence, not a measurement of this workload. Prefer maintained binary wheels compatible with the discovered OS/Python; record actual `show_config()` output. Do not force a source build or Rosetta just to match the Linux validation versions.

The core needs float64 eigendecomposition, matrix products, linear solves, constrained QPs, and complex-step differentiation. These are CPU operations here. Optional PyTorch CPU-double autograd can provide a third derivative check if useful. PyTorch MPS cannot allocate float64/complex128; enabling unsupported-op fallback does not fix that dtype limitation [R20]. A GPU failure therefore does not block the core. MLX is not needed; do not add a second accelerator framework without a measured bottleneck and a verified numerical path.

NumPy and SciPy have permissive BSD-style licensing; inspect the installed distribution's license/third-party notices before redistributing wheels [R23], [R24]. Synthetic data are generated by the project and require no external data license. Save the actual license for every optional copied implementation and downloaded dataset. Publicly downloadable source is not automatically licensed for reuse.

### 6.2 Hardware discovery and installation

Apple's official M3 Max specifications include several memory configurations, beginning at 36 GB and extending to 128 GB depending on chip variant [R25]. Do not assume the user's configuration or free capacity.

**Existing commands to run and save, before dependent work:**

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.logicalcpu
sysctl -n hw.physicalcpu
df -h .
vm_stat
python3 --version
```

Use `platform.machine()` inside the selected interpreter to confirm native ARM64 execution. Record model/chip details from `system_profiler SPHardwareDataType`, but redact serial numbers and hardware UUIDs from published reports. Record power mode, whether plugged in, other substantial workload, and OS memory pressure. Avoid collecting unrelated personal information.

**Existing installation pattern, with local selection rather than invented pins:**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --only-binary=:all: numpy scipy setuptools
.venv/bin/python -m pip freeze --exclude-editable > requirements.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -c 'import numpy as np, scipy; print(np.__version__, scipy.__version__); np.show_config(); scipy.show_config()'
```

If the system Python cannot resolve maintained native wheels, select a supported ARM64 Python from an official distribution and document it. Test the required operations before locking a release. Archive or hash the resolved wheels and capture the installer report; `pip freeze` alone is not a complete supply-chain or hardware record. Subsequent runs use the frozen exact versions. The optional original plotting script additionally requires Matplotlib; a LaTeX installation is not required for the Markdown handoff or the scientific PoC.

**Package bootstrap before any proposed project command.** WP00 must first create `src/prepared_learning/__init__.py`, `__main__.py`, and a minimal `cli.py` implementing `doctor`; add later commands in their owning work packages. Use the following minimal `pyproject.toml` metadata, adding project entry points only if needed:

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "prepared-learning-poc"
version = "0.1.0"
dependencies = ["numpy", "scipy"]

[tool.setuptools.packages.find]
where = ["src"]
```

`__main__.py` imports and calls `cli.main()`. After those files exist, run:

```bash
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pip freeze --exclude-editable > requirements.lock.txt
.venv/bin/python -m prepared_learning doctor --output runs/environment
```

Setuptools is already installed above; record its exact version and wheel hash along with the numerical dependencies. Keep the editable project out of the portable dependency lock and record its Git commit/tree hash separately. All project commands below use this environment's interpreter. On replay, recreate the environment from the locked wheels/versions, restore the matching source revision, and repeat the editable install before running a project command. Do not attempt to download an unrelated package named `prepared_learning`.

### 6.3 Memory and computational model

Let `q_batch` be simultaneously generated tasks, `n` context rows, `d` dimension, `m` width, and `r` prior rank. Let `C_d` and `C_m` count live dense matrix workspaces, measured during calibration. A representative float64 peak is

\[
8q_{\rm batch}(nd+n+3d+r)
+8C_d d^2+8C_m m^2+\text{method/prior buffers and runtime overhead}.
\tag{13}
\]

This includes a materialized context block, labels, coefficient/summary/prediction vectors, and matrix workspaces. It excludes unknown library temporaries, which must be measured. Process methods sequentially or reuse buffers; do not store a `tasks x methods x d` prediction cube.

| Stage | Work and storage model | Practical implication |
|---|---|---|
| Spectrum/frontier | Known spectral risk `O(d)`; two QPs with `O(m)` variables and `O(m^2)` generic workspace; dense prior eigendecomposition `O(d^3)` time, `O(d^2)` storage once per preparation. | Core proof examples are tiny. Do not solve a full matrix SDP when two scalar programs suffice. |
| Constructor | `P=(d+1)m+L-2`; parameters `8P` bytes, old/new weights and gradients roughly a few multiples of this. | Count zero entries. An optimizer with moment buffers is unnecessary. |
| Jacobian audit | Literal basis-input Jacobian `d x P`, `8dP` bytes; complex-step needs about `P` product evaluations. | Confine exhaustive derivative audits to small cases. Use (4) or matrix-free products at scale. |
| Task generation/update | Context `8 q_batch n d` bytes; direct structured operator `O(dm)` per task after `z`; full first network step has comparable arithmetic plus parameter-update work. | Stream independent blocks and count the computation of `z` consistently. |
| Source estimation | Streaming cross moments `O(T d^2)` accumulation if dense; retain only the needed source observations/keys and sufficient statistics. | At large `d`, source estimation may dominate the cheap QPs. Count it. |
| Ridge/Bayes | Choose rank-space `r x r` or sample-space `n x n` solves; work roughly `O(ndr+nr^2+r^3)` for a rank-`r` prior. | Do not penalize the baseline with an avoidable dense `d x d` inverse. |
| Logging | Per-task scalar losses, status, and keys; size proportional to tasks times reported methods. | Keep raw losses and aggregates, regenerate large synthetic inputs from keys. |

Arithmetic examples, **not measured peaks**: `q_batch=1024,d=128,n=32` uses 32 MiB for `X`; `q_batch=256,d=1024,n=128` uses 256 MiB for `X`; a `1024 x 1024` float64 matrix is 8 MiB. These quantities are comfortably smaller than nominal M3 Max memory, but concurrent processes, copies, and OS use must still be measured. The required `d=6` and `d=64` mechanism tests are far smaller.

### 6.4 Calibration, scaling, and honest runtime estimates

Start with small exact fixtures, then a representative W1/W2 task block. Measure cold preparation, warm adaptation, prediction, serialization, and peak resident memory separately. Increase `q_batch` until throughput flattens within timing noise or memory pressure approaches the selected reserve. Initially budget at most half the currently available memory; this is an operational reserve, not a scientific scale limit. Record the reserve and actual headroom.

Use measured tasks/second and measured preparation cost to project the frozen workload. Estimate total time as preparation replications plus test tasks plus reporting/checkpoints; use an interval from pilot throughput, not the fastest observed block. State estimates as estimates, then retain actual elapsed times. No target-Mac time has been measured in this investigation.

Increase dimension, source-task count, or repetitions only to resolve a stated uncertainty: missing asymptotic behavior, prior-estimation transition, numerical sensitivity, or confidence width. A negative small test that faithfully expresses the mechanism is informative; a proxy incapable of expressing the relevant rank/width transition is not.

For performance use `time.perf_counter_ns()`, warm each path, randomize method order in blocks, and maintain the same device, dtype, BLAS/thread policy, and input residency. Choose repetitions from pilot timing variance to resolve the decision-relevant difference; repeated timings are not independent learning tasks. Separate data generation from adaptation timing, then also report an end-to-end measure including it when appropriate. Include prior construction and amortization over the number of future tasks.

If testing MPS float32, compare against CPU first, include transfers, and call `torch.mps.synchronize()` immediately before/after timed sections. Record both tensor and driver allocations using the documented APIs [R20]. Preserve allocator protections. Do not claim joules or energy savings from weight norm, elapsed time, or unvalidated power telemetry.

### 6.5 Faithful fallback and recovery

If memory is lower or busier than expected, reduce batch size, stream source moments, process methods serially, and checkpoint more often. These changes preserve the estimator and sample population. Avoid materializing full Jacobians and dense operators at large scale. Never silently reduce dimension, task count, precision, or rank; those change the scientific claim or its sensitivity and require an amendment.

If a required CPU operation fails, isolate a tiny reproduction and select a verified compatible CPU build. Optional accelerator failure returns to the CPU reference. If the core's necessary dense prior itself will not fit, the larger-scale claim is locally blocked; keep the smaller valid result and quantify the missing scale. No remote GPU fallback is part of this assignment.

Checkpoint at deterministic task-block or preparation-replication boundaries with atomic writes. Store committed keys, method/configuration hashes, prior assets, RNG specification, partial sums, raw losses/conditional risks, and failure records. On interruption, discard only the uncommitted block and regenerate it. Separate a **scientific configuration hash** (population, methods, precision, information access, seed mapping, endpoint, counts, analysis) from an **operational configuration hash** (batch size, checkpoint cadence, output path, measured thread policy). A logged batch-size or checkpoint-cadence change may resume the same scientific run because task identities and arithmetic results must be invariant. A changed dependency lock, code revision, prior asset, or scientific configuration cannot append to the old confirmatory aggregate. A thread/BLAS change requires an equivalence check and a separate timing stratum; do not silently mix performance measurements.

## 7. Executable work packages and decision gates

### 7.1 Proposed repository

Create a small Python package named `prepared_learning`. This is a proposed project name, not an existing package to install from a registry.

| Path | Responsibility |
|---|---|
| `README.md`, `research_poc_plan.md` | Execution instructions and this frozen scientific specification. |
| `pyproject.toml`, `requirements.lock.txt`, `wheel_manifest.json` | Package metadata, exact resolved environment, wheel hashes and licenses. |
| `src/prepared_learning/specs.py` | Validated configuration/dataclasses and schema version. |
| `src/prepared_learning/__init__.py` | Importable package marker for the proposed `src` layout. |
| `src/prepared_learning/geometry.py` | Tangent formula, norm/count accounting, canonical and head-zero constructors, simultaneous update. |
| `src/prepared_learning/frontier.py` | F/Z QPs, original phase formula, independent certificate evaluation. |
| `src/prepared_learning/priors.py` | Cross moments, moment directions, PSD repair, shrinkage, empirical gain fitting. |
| `src/prepared_learning/methods.py` | Shared interface and B00-B11 methods. |
| `src/prepared_learning/tasks.py` | Keyed generators, separate context/evaluator records, optional natural-covariate loader. |
| `src/prepared_learning/runner.py` | Chunk execution, checkpoints, failure handling, and deterministic resume. |
| `src/prepared_learning/analysis.py` | Exact frontiers, paired/hierarchical inference, Pareto tables, multiplicity rules. |
| `src/prepared_learning/cli.py`, `__main__.py` | Explicit proposed commands below. |
| `tests/test_geometry.py`, `test_frontier.py`, `test_protocol.py` | Mechanism, certificate, leakage, and resume tests; no redundant tests of trivial wrappers. |
| `checks/independent_reference.py` | Separate derivative/objective/support implementation; must not import production formulas being checked. |
| `configs/exploration/`, `configs/confirmation/` | Human-readable JSON configurations; confirmation copies immutable. |
| `protocol/` | Hypotheses, preregistration manifest, amendments, evidence ledger, decision log. |
| `data/raw/`, `data/processed/`, `data/manifests/` | Only optional data, hashes, license and split metadata. |
| `runs/<run_id>/` | Resolved config, environment, prior assets, raw losses, certificates, checkpoints, failures, timing/memory. |
| `reports/` | Tables, standalone figure files, results narrative, audits, and claim ledger. |

Use actual Git revisions for code even if the project remains local. Do not invent a remote repository or require external publication. Retain the original bundle unchanged if available. Its scripts are convenient regression checks, but the definitions and fixtures in this document are sufficient to reconstruct the core without them.

### 7.2 Shared interfaces

Use shape checks and explicit units at the boundary. Suggested typed interfaces:

```python
@dataclass(frozen=True)
class BudgetSpec:
    depth: int
    initialization_norm_cap: float  # squared Frobenius sum, not L2 norm
    trainable_parameter_cap: int
    learning_rate: float           # 1.0 in the main contract

@dataclass(frozen=True)
class Context:
    X: np.ndarray                  # float64, shape (n, d)
    y: np.ndarray                  # float64, shape (n,)
    task_key: str

@dataclass(frozen=True)
class PreparationSpec:
    target_context_size: int       # n; distinct from source-task sample count
    input_dimension: int           # d
    input_law: str                 # gaussian_isotropic in the core
    loss_normalization: str        # squared_error_divided_by_2n
    task_second_moment_trace: float  # 1.0 in the core

@dataclass(frozen=True)
class CompactMetric:
    floor: float
    basis: np.ndarray              # shape (d, j), orthonormal columns
    increments: np.ndarray         # shape (j,), nonnegative
    support_metadata: dict

class Method(Protocol):
    def fit_meta(self, train_tasks, validation_tasks, config) -> PriorAsset: ...
    def prepare(self, prior: PriorAsset, budget: BudgetSpec,
                task: PreparationSpec) -> PreparedState: ...
    def adapt(self, prepared: PreparedState, context: Context) -> AdaptedState: ...
    def predict(self, adapted: AdaptedState, X_query: np.ndarray) -> np.ndarray: ...
    def diagnostics(self, prepared, adapted) -> dict: ...
```

`PriorAsset`, `PreparedState`, and `AdaptedState` are project dataclasses to implement, not external APIs. Define the first to contain covariance/basis, noise estimate, fit metadata and hashes; the second to contain the allowed compact operator and/or old network weights plus immutable budget and preparation specifications; the third to contain the resulting coefficient or network state and timing/status. `prepare` derives `c,rho` from the target `n` and permitted noise estimate, not from an implicit source sample size. `adapt` verifies that context dimensions/row count agree with the frozen preparation contract. Oracle assets must carry an `information_access="oracle"` flag. Keep evaluator truth (`theta`, true `S`, query labels) in a different object never passed to `prepare` or `adapt`.

`solve_frontier` returns **both** branch results: variables, metric, objective, computed primal/dual bounds, constraint and KKT residuals, active-set metadata, solver status, verification level/roundoff treatment, and selected/tied branch. `construct` returns layer arrays plus claimed and observed resource counts. `adapt` must create fresh per-task state; sharing a mutated initializer across tasks is a serious defect.

**Example proposed exploration configuration:**

```json
{
  "schema_version": 1,
  "phase": "exploration",
  "analysis_mode": "fixed_prior_task_simulation",
  "workload": {"d": 6, "spectrum": [0.50, 0.25, 0.15, 0.07, 0.02, 0.01],
               "n": 4, "sigma2": 1.0, "input_distribution": "gaussian_isotropic"},
  "budget": {"depth": 2, "initialization_norm_cap": 0.6,
             "trainable_parameter_cap": 21, "learning_rate": 1.0},
  "numerics": {"dtype": "float64", "device": "cpu", "rank_policy": "explicit_support"},
  "information_access": "oracle",
  "methods": ["B00", "B01", "B02", "B03", "B04", "B05", "B07", "B08", "B09", "B10"],
  "root_seed": 20261031,
  "sample_plan": {"outer_replicates": null, "tasks_per_replicate": null,
                  "status": "pilot_must_determine"}
}
```

The schema must reject a confirmation configuration with unresolved **required** counts, missing primary contrast, a changed `eta`, or an unrecorded environment lock. Use endpoint-specific requirements: `analysis_mode="deterministic_frontier"` needs no stochastic counts; `analysis_mode="conditional_risk_over_preparations"` needs fixed outer replications and all source/validation counts, with inner test tasks explicitly not applicable; `analysis_mode="nested_task_simulation"` needs both outer and inner counts; `analysis_mode="fixed_prior_task_simulation"` needs a test-task count. Separate actual-update validation/timing schedules from the primary endpoint. Never encode a deterministic quantity as a misleading sample size of one.

### 7.3 Failure classes used by the gates

| Code | Meaning and required action |
|---|---|
| `IMPLEMENTATION_INVALID` | Broken invariants, inconsistent outputs, leakage, or invalid timing/certificates. Repair; rerun affected checks and dependent results. Draw no scientific conclusion. |
| `MATHEMATICAL_COUNTEREXAMPLE` | A verified feasible construction or analytic argument contradicts a theorem/reduction. Preserve exact inputs and evidence; stop dependent claims and revise the theorem. |
| `INSENSITIVE_OR_UNSUITABLE` | The proxy cannot express the mechanism, or uncertainty exceeds the target effect. Improve the proxy or sampling under an amendment; otherwise declare inconclusive. |
| `LOCAL_RESOURCE_LIMIT` | A faithful required computation cannot fit/run with available resources. Preserve the valid smaller claim and quantify the blocked scale; do not silently substitute an easier problem. |
| `NO_COMPARATIVE_ADVANTAGE` | Correct mechanism works but ties/loses against the strongest admissible baseline or is operationally dominated. Report it and narrow/stop the corresponding superiority claim. |
| `SUPPORTED_IN_SCOPE` | Proof/checks and, where applicable, protected evidence support the prespecified claim without a material unreported tradeoff. State its exact population and resource contract. |

### WP00 — Establish provenance, environment, and the resource contract

**Objective/risk:** prevent hidden coordinate changes, missing assets, unsupported dependencies, or an assumed 128 GB machine. Addresses prerequisites for every hypothesis.

**Inputs and entry:** this document, a fresh M3 Max, and optionally the original bundle. No expensive download is needed.

**Actions:** create the repository and bootstrap metadata/package/`doctor` command as specified in Section 6.2 before invoking it; save hardware discovery and environment output; establish the native CPU-double environment and editable installation; record dependency/licenses. If the source ZIP is available, unpack its preserved `prepared_learning_research/` directory under project `source_bundle/`, retain a read-only snapshot, and hash all files. Implement config validation and explicitly encode `L>=2`, all-entry trainability, no bias, `eta=1`, target context size, and loss normalization. Create `protocol/decisions.jsonl` and `protocol/evidence_ledger.csv`.

**Proposed command:** `.venv/bin/python -m prepared_learning doctor --output runs/environment`. Existing OS/install commands are in Section 6.2.

**Outputs/tests:** `hardware.json`, `environment.json`, exact lock/wheel manifest, source manifest, license manifest, and operation smoke results for float64 `eigh`, QR/SVD, linear solve, SLSQP with analytic gradient, and complex128 matrix multiplication. Reconstruct a tiny symmetric matrix from its eigendecomposition and check a known linear solve residual.

**Resources:** negligible numerical arrays; installation disk usage is measured before optional packages. No GPU prerequisite.

**Acceptance/next:** native CPU operations and source identity recorded -> WP01. Missing optional original files or natural data -> reconstruct fixtures from this document and continue. Unsupported required CPU stack -> repair environment before numerical work. Unknown third-party redistribution rights -> use a clean equation implementation and retain the licensing limitation; do not block the permitted mathematics.

### WP01 — Freeze and challenge the mathematical contract

**Objective/risk:** H1/H2, including nonconvex width constraints, zero cases, and parameterization dependence.

**Dependencies:** WP00; Sections 2 and 4.

**Actions:** transcribe the lower-bound chain, equality construction, width necessity, clipping/orientation proof, two-family completeness, and dual formula into `protocol/proof_obligations.md`. Check scalar and rank-one limits separately. Verify symbolic/rational source thresholds and the finite-step counterexample. Record exactly which claims are supplied, derived here, conjectural, or empirical. Construct the free-learning-rate and depth-one controls on paper and in tiny arrays.

**Proposed command:** `.venv/bin/python -m prepared_learning check-theory --config configs/exploration/exact_cases.json`.

**Outputs/tests:** proof-obligation checklist with argument locations; machine-readable expected fixture values; all units and budget conversions. Show that the richer-spectrum branch can lose and that `Gz` equality is a required outcome.

**Resources:** `d<=6` exact/rational arithmetic and float64 checks. A proof assistant is optional future validation, not a circular entry gate.

**Acceptance/next:** internally consistent derivations -> WP02. A credible counterexample -> `MATHEMATICAL_COUNTEREXAMPLE`, revise dependent objectives before implementation. A novelty uncertainty alone -> retain provisional wording and continue falsification work.

### WP02 — Build the mechanism harness and fix scale handling

**Objective/risk:** distinguish an actual geometry effect from incorrect differentiation, support truncation, or sequential updates.

**Dependencies:** WP01; `geometry.py`, independent reference checks.

**Actions:** implement canonical and **separate head-zero** constructors, effective coefficient, (4), plain simultaneous gradients, norm/count, and compact `G` application. Independently differentiate the matrix product on basis inputs by complex step. Check `G=0`, `d=1`, repeated/singular/isotropic spectra, rotations, depths 2-8, and scales spanning small/ordinary/large representable values. Include the explicit `1e-14` defect fixture; preserve analytic support metadata.

**Proposed command:** `.venv/bin/python -m prepared_learning check-geometry --suite adversarial`.

**Outputs/tests:** `geometry_checks.jsonl`, independent Jacobian residuals, scale-sensitivity table, exact/nonexact rank labels, budget residuals, and failed-case arrays. Test update equality over several `eta` values as a correctness check while leaving the primary statistical contract at one. Test (10) for canonical and noncanonical null networks.

**Resources:** tiny basis/Jacobian cases; projected memory for each derivative audit recorded. Large matrices use no explicit Jacobian.

**Acceptance/next:** scale-aware residuals pass and known nonlinear counterexample remains nonlinear -> WP03. Any mismatch -> `IMPLEMENTATION_INVALID`; localize and fix. Do not erase failures by increasing an absolute tolerance or dropping small eigenvalues.

### WP03 — Reproduce the source and establish exact baselines

**Objective/risk:** H1/H3 baseline correctness and reproduction of the original phase structure.

**Dependencies:** WP02 and the specified baseline equations. This stage implements B00-B02 and B05-B10; B09/B10 use oracle basis/covariance here, while their learned preparation belongs to WP05. The general F solver B03 is implemented in WP04, so it is not a prerequisite for WP03.

**Actions:** implement the source phase formulas and the independent unrestricted spectral QP B06. Implement optimized isotropy, zero-head rank shrinkage, direct structured maps, ridge, projected ridge, and posterior mean. Verify the normal-equation normalization and singular/noiseless behavior. If original scripts are present, execute them into new output locations; copy `audit_geometry.py` before execution because it writes beside itself.

**Existing optional bundle commands:** after creating `runs/source_reproduction/`, run `.venv/bin/python source_bundle/prepared_learning_research/verify_plasticity.py --output runs/source_reproduction/verify.json` and `.venv/bin/python source_bundle/prepared_learning_research/phase_transition.py --output runs/source_reproduction/phase.json`. These are existing scripts only when the bundle is staged as in WP00. The copied audit script has no output flag. Their hard-coded scale policy is a known limitation, not the new implementation specification.

**Proposed command:** `.venv/bin/python -m prepared_learning reproduce --suite source_and_baselines`.

**Outputs/tests:** source-risk table, B06 certificates, baseline residuals, identical-map paired losses, and Gaussian posterior comparisons. Include all six source budgets, not only `k=0.4`.

**Resources:** source Monte Carlo count is 200,000, useful for reproduction rather than a mandated new sample-size rule. Stream it; calibrate elapsed time locally.

**Acceptance/next:** deterministic source formulas agree with independent QPs; simulation discrepancies are consistent with their predeclared sampling check; B05 equals networks -> WP04. Posterior or solve defects -> fix baselines before any comparison. If the original nonlinear full-matrix SLSQP diagnostic fails, retain that status; it is not a mathematical counterexample.

### WP04 — Implement and certify the joint frontier

**Objective/risk:** H2 and the candidate differentiated contribution.

**Dependencies:** WP02-WP03; `frontier.py`, compiler, independent verifier.

**Actions:** implement F and Z from (7)-(8), primal repair, nonnegative multiplier recovery, lower/upper risk diagnostics, exact rational KKT verification for rational fixtures, branch ties, and `P_cap -> m` conversion. Use exact known support metadata when compiling. Enumerate alternative supports at tiny dimension and compare random rotations. Verify that relaxing either budget cannot worsen the optimum, and that `m>=d` recovers B06. Check the `m<1`, `B=0`, and `d=1` branches explicitly. `--certify` must report its verification level and must not relabel a float64 diagnostic as rigorous certification.

**Proposed command:** `.venv/bin/python -m prepared_learning frontier --config configs/exploration/frontier.json --certify`.

**Outputs/tests:** `frontier.csv`, branch-level certificates, compact metrics and networks, exact-risk-versus-budget/width plots, and actual versus capped parameter counts. Reproduce the family reversal in Section 4.6.

**Resources:** two `O(m)`-variable programs per point; exhaustive support search only at tiny dimension. Stop deterministic mesh refinement when risk interpolation and switch-location uncertainty meet the reporting precision.

**Acceptance/next:** proof-backed reduction and verification records agree with independent tiny enumeration -> WP05/WP06. A feasible lower-risk alternative -> `MATHEMATICAL_COUNTEREXAMPLE` after checking numerical validity. A solver gap or rounding uncertainty too large to resolve the declared comparison -> refine or mark unresolved. No F advantage in a declared region -> retain the region as `NO_COMPARATIVE_ADVANTAGE`, not a reason to delete it.

### WP05 — Implement finite-prior and practical preparation baselines

**Objective/risk:** H4, hidden information costs, and weak estimation/tuning confounds.

**Dependencies:** WP04 and all operational baseline checks.

**Actions:** implement (12), moment directions, PSD repair, shrinkage, source-only noise estimation, B11 query-loss fitting, and equal-data validation selectors. Define separate oracle and learned information classes. Stream source sufficient statistics; serialize complete prior assets and account for their bytes/time. Fit F/Z variants with the same improvements; fit ordinary/projected ridge and empirical Bayes on their natural validation objectives.

**Proposed command:** `.venv/bin/python -m prepared_learning fit-priors --config configs/exploration/learned_prior.json`.

**Outputs/tests:** prior accuracy/negative-mass/eigenspace diagnostics, tuning ledger, selected configurations, source sample counts, and preparation resources. A test deliberately hides true `S/theta` from method interfaces and verifies the source/test key separation. Check convergence toward the oracle as source information increases without assuming monotonic finite-sample performance.

**Resources:** source task count and dense moment storage from calibration; no pretrained model. Reuse the same data observations without charging them multiple times, but count each actual computation.

**Acceptance/next:** interfaces and source/validation separation sound -> WP06. Inaccurate priors -> retain as a measured preparation regime and calibrate informative source counts. Access to evaluator truth or query leakage -> `IMPLEMENTATION_INVALID`; all affected learned-prior results invalidated.

### WP06 — Pilot sensitivity, cost, and freeze confirmation

**Objective/risk:** underpowered tests, arbitrary compute limits, and adaptive selection.

**Dependencies:** WP04 for deterministic confirmation; WP05 for learned-prior confirmation.

**Actions:** run only exploratory keys. For the fixed-map primary contrast, calculate conditional risks and estimate variance across preparation replications. Measure nested variance components only for endpoints requiring task simulation. Measure prior-error curves, timing/memory, tuning resolution, and sampled task-tail behavior. Apply Section 5.9's sample/cost rules. Select the primary scenario, eligible baseline IDs, endpoint/analysis mode, confidence procedure, secondary comparison family, required fixed counts, and resource estimate. Freeze code, configs, dependencies, prior-estimation/validation-selection procedures, seed derivation, and analysis before protected evaluation. Protect source-replication/test manifests from accidental use during tuning.

**Proposed commands:** `.venv/bin/python -m prepared_learning pilot --config configs/exploration/pilot.json` followed by `.venv/bin/python -m prepared_learning freeze --protocol protocol/confirmation.json`.

**Outputs/tests:** `pilot_report.md`, `power_precision.json`, timing/memory calibration, immutable confirmation manifest and hash. The freeze command refuses unresolved endpoint-required counts or missing baseline selectors; it accepts an explicitly inapplicable inner count for analytic conditional risk.

**Resources:** selected by measured information requirements and available capacity. The pilot may grow to stabilize estimates; its data never enter confirmation.

**Acceptance/next:** achievable precision and all fields frozen -> WP07. If adequate precision is too costly, report `LOCAL_RESOURCE_LIMIT` or `INSENSITIVE_OR_UNSUITABLE`; narrow a claim only by amendment. Do not choose a larger acceptable error merely to call the result positive.

### WP07 — Execute protected controlled comparisons

**Objective/risk:** confirm H2/H3 and, when prepared, H4 without adaptive inspection.

**Dependencies:** WP06 frozen manifest; baseline correctness gates and recorded numerical verification levels.

**Actions:** run fresh deterministic-spectrum/support checks from the frozen generator, then the fixed outer-replication schedule. Fit/select each learned method from the shared source observations, evaluate fixed-map conditional risk using evaluator-only truth, and commit one paired result per replication. Run the separate fixed task-simulation schedules for actual-update validation, X-dependent estimators, and nonanalytic endpoints. Use common random tasks across methods, immutable per-task initializers, and atomic result commits. Finish the prespecified sample unless a correctness/resource interruption occurs. Do not inspect primary effects to alter sample counts.

**Proposed command:** `.venv/bin/python -m prepared_learning confirm --manifest protocol/confirmation.json`.

**Outputs/tests:** `preparation_risks.jsonl` with each fitted prior/selected metric and its conditional risk/paired difference; raw per-task paired losses for the separate simulation endpoints; all branch verification records; run status/failure logs; fixed-sample intervals and adjusted secondary conclusions. Validate matrix/step invariants on scheduled sampled cases during execution without using outcome-based sampling. Check a small simulated risk subset against its conditional analytic value as a harness diagnostic.

**Resources:** predicted in WP06, actual measured. Preserve the chosen scientific dimension/count/precision across batch-size recovery.

**Acceptance/next:** complete valid run -> WP08/WP09. Broken invariants -> invalidate dependent outcomes and repair; changed code requires a new confirmation identity. Positive interval with guardrails satisfied -> support only the declared contract. Zero-crossing wide interval -> inconclusive; negative interval -> evidence against the corresponding benefit. Exact geometry may remain valid regardless.

### WP08 — Attribution, boundary conditions, and independent confirmation

**Objective/risk:** determine why a gain occurs and whether it survives fairer baselines or nearby conditions.

**Dependencies:** valid WP07; Section 8 prespecified arms and remaining uncertainties.

**Actions:** run the ablation table below. Include no-floor, isotropic, wrong orientation, estimated prior, shared generic optimizations, exact/approximate width, free-learning-rate and depth-one controls, float32 sensitivity, and a small two-step diagnostic. Run a fresh outer-source/test cohort for any claim retained after exploratory robustness changes. Inspect signed worst-case failures, not only averages.

**Proposed command:** `.venv/bin/python -m prepared_learning robustness --manifest protocol/robustness.json`.

**Outputs/tests:** attribution matrix, failure-regime plots, regression table, approximate-width tradeoff, and independent-confirmation record. Every new confirmatory claim gets a frozen amendment and fresh data.

**Resources:** choose arms to resolve a concrete uncertainty; full factorial combinations are not required. Preserve original negative arms.

**Acceptance/next:** mechanism-specific benefit robust in its declared regime -> WP09/WP10. Vanishing benefit after fair tuning -> `NO_COMPARATIVE_ADVANTAGE`. Wrong-prior failures narrow practical applicability; they do not refute H1. Two-step failure narrows the time horizon; it does not invalidate first-step cancellation.

### WP09 — Measure operational cost and optionally test natural inputs

**Objective/risk:** H5 and hidden preparation/storage/latency costs.

**Dependencies:** B05 equality and baseline correctness; calibrated local resource policy. Natural-input arm additionally needs its data/license/split gate.

**Actions:** compare cold preparation, warm adaptation, and prediction for networks, structured direct maps, ridge, and empirical Bayes. Report actual trainable entries, stored/frozen scalars, serialized bytes, workspaces, wall time, and amortized preparation across future-task counts. Allow collapsing a trained linear network to its coefficient for inference and charge the collapse time; use a structured prediction path if it is cheaper for few queries. Apply the same optimization policy to all eligible methods. Exploiting known zero suffix gradients in the canonical first step is allowed only with equivalent algebraic simplifications available to competitors. Keep an additional measurement of raw network inference if integration requires it. Run the optional Section 5.6 stress test only if it answers an unresolved input-distribution question.

**Proposed command:** `.venv/bin/python -m prepared_learning benchmark --manifest protocol/performance.json`.

**Outputs/tests:** randomized timing-block records, uncertainty, end-to-end Pareto table, memory peaks, and optional semi-synthetic results. No reported energy measurements without a validated measurement method.

**Resources:** locally measured operation sizes; no remote execution. Reducing batching is faithful; quantization is a separate changed-method arm.

**Acceptance/next:** report domination, tradeoffs, or a reproducible operational advantage accurately -> WP10. Direct-map domination ends H5's superiority claim while preserving a mathematical contribution. Natural-input failure stops transfer claims and records which assumption failed.

### WP10 — Audit, reproduce, and issue the decision report

**Objective/risk:** prevent a technically correct partial result from becoming an unsupported broad claim.

**Dependencies:** completed relevant stages and preserved negative results; optional arms can be explicitly not run.

**Actions:** independently regenerate tables from raw outputs; verify deterministic interrupted/resumed equivalence; perform the three audits in Section 11; check source/citation/claim alignment. Produce a claim ledger with status, baseline, metric, resource contract, evidence path, and scope. Record continue/revise/stop decisions using Section 10. Prepare a clean environment replay on the same laptop; a second machine is not required.

**Proposed commands:** `.venv/bin/python -m prepared_learning analyze --manifest protocol/confirmation.json`, `.venv/bin/python -m prepared_learning replay --manifest protocol/replay.json`, and `.venv/bin/python -m prepared_learning report --manifest protocol/report_inputs.json`. Create `replay.json` from the selected original run IDs, frozen configurations, code/dependency hashes, and expected deterministic outputs; create `report_inputs.json` as the explicit ordered list of valid/failed run IDs and evidence paths to include. These manifest files are generated project artifacts, not missing external inputs.

**Outputs/tests:** `reports/results.md`, `reports/claim_ledger.csv`, standalone plots/tables, audit log, exact reproduction instructions, and a manifest of all needed files. A reader must be able to identify failed runs and unperformed tests without interpreting missing files as success.

**Resources:** raw scalar outputs and compact assets; report generation should be much cheaper than experiments.

**Acceptance/next:** all reported claims trace to valid evidence and replay -> deliver the scoped result. A new material defect reopens only affected gates and descendants. Unresolved novelty or application value remains a stated limitation, not an excuse to suppress a rigorous negative or replication result.

## 8. Ablations, robustness, and domain-specific adversaries

| Test | What changes, what remains matched | Interpretation and decision |
|---|---|---|
| Remove isotropic floor | Use the best Z program, not arbitrary `b=0` with otherwise unchanged gains. Match caps, prior, and tuning. | Isolates the proposed resource allocation. A tie is expected in some regimes. |
| Force isotropy | Optimize its scalar and width-one realization. | Checks whether apparent specialization value is just a larger learning-rate gain. |
| Equal actual count versus equal cap | Use the unpadded `k=0.6` fixture, then compare minimum-size realizations and optional padded same-width architectures; report actual, minimum, and permitted counts. | The fixture matches minimum usage. Padding only equalizes allocated count and does not establish equal minimum resource requirements. |
| Same map, different realization | Network versus structured `Gz`; optional nonminimal null realizations for the same initial `G`. | Exact first-step equality is mandatory for the canonical and final-zero-head implementations, and holds for every depth-two null factorization. Arbitrary deeper noncanonical null realizations need not agree. Operational difference is separate. |
| Orientation ablation | Replace the learned/true basis with a random orthonormal basis, matching spectrum and norm/count. | Tests whether inherited task information, rather than scale, creates the advantage. |
| Prior uncertainty | Independent finite-source replications, shrinkage and empirical-gain alternatives, same data. | Quantifies the cost of discovering the geometry and prevents oracle-only practical claims. |
| Prior rotation/leakage | Rotate source versus test subspace by known principal angles; mix task mass into its complement and renormalize trace. | A positive floor may help shift tolerance, but that is a new empirical claim requiring a fixed shift population. |
| Input assumptions | Isotropic non-Gaussian inputs, then anisotropic Gaussian inputs, then optional natural covariates. | The realization theorem remains; Gaussian/isotropic statistical optimality may not. Evaluate actual risk and distinguish assumption failure. |
| Coordinate reparameterization | Orthogonal transformations should preserve the theorem; nonorthogonal rescaling changes Euclidean norm/metric. | Do not present coordinate-dependent changes as invariant resource gains. |
| Learning-rate freedom | Scale weights and inverse-scale `eta`; measure initial and post-step norms. | Demonstrates norm-only degeneracy and prevents claiming universal preparation cost. |
| Depth-one/fixed-feature controls | Permit direct zero linear predictor and stored preconditioner. | Tests operational interpretation outside the theorem's architecture. |
| Approximate width | Drop small increments or approximate the selected metric, then reoptimize within each smaller width; plot risk versus actual width and kernel error. | Exact `1 -> r+1 -> r` rank jumps may be practically negligible. No exact-rank claim from a numerical threshold. |
| Precision/extreme scale | Float64 reference, float32 sensitivity, small/large gain rescaling, nearly repeated minima. | A numerical gain erased by rounding is not a stable implementation benefit. Do not use float32 to certify exact multiplicity. |
| One versus two steps | Run a literal second simultaneous GD step and compare with a fixed-metric recurrence. | Quantifies where the proven first-step equivalence ends; no unproved multistep extrapolation. |
| Noncanonical null network | The rational counterexample and random null products. | Ensures the harness can detect nonlinear finite-step terms instead of assuming them away. |
| Information advantage | Oracle versus estimated versus no prior; same covariance asset shared where required. | Removes hidden access to true task parameters or richer design statistics. |
| Generic optimization sharing | Compact representations, Cholesky/dual solves, collapsing linear networks, common precision/thread policy. | A gain erased by a shared optimization is an engineering finding, not evidence for the new mechanism. |
| Resume/error adversary | Interrupt mid-block, corrupt an uncommitted file, change a config hash, introduce a single method failure. | Resume must neither skip nor duplicate tasks, mix configurations, or quietly drop unfavorable observations. |

For approximation, measure both matrix error and decision-relevant risk. Under the core model one can calculate the exact changed risk from (1), so a small spectral perturbation need not be judged only by a generic norm bound. Near an exact branch tie, report the uncertainty and both solutions rather than drawing an artificially sharp phase boundary.

A different task distribution can be favorable to either floor or pure-rank learning; do not optimize the shift distribution after observing test outcomes. Nonzero task means introduce a separate benefit from an inherited answer; that would violate the intended separation. Keep mean zero in the core and explicitly label any mean-shift adversary.

## 9. Reproducibility, experiment records, and reporting

### 9.1 Deterministic data and restartability

Use separate keyed random streams for task coefficients, contexts, noise, query inputs, rotations, and method-internal randomness. A concrete implementation uses NumPy `SeedSequence` from integer fields `(root_seed, phase_id, outer_replication, task_id, stream_id)` and a recorded bit generator. Per-task identity must be invariant to batch size and method order. Do not rely on one mutable RNG whose consumption changes when a method is added or an OOM changes batching.

Save the generator specification/version and all key fields. For expensive prepared assets save actual arrays and hashes, not only seeds. Reproducibility across arbitrary future library versions is not guaranteed by a seed; the environment lock and data-generator revision are part of the result.

Write a block's raw results and metadata to temporary paths, flush them, verify counts/hashes, atomically rename, then mark the block committed. On restart, check committed blocks and recompute only uncommitted ones. Summary statistics are reconstructed from raw records, not trusted as the only state. Preserve invalid/failed records with a reason. Never drop a failing method's hard tasks from the paired population.

### 9.2 Minimum schemas

`run.json` must include:

```json
{
  "schema_version": 1,
  "run_id": "content-derived-or-unique-recorded-id",
  "phase": "exploration_or_confirmation",
  "git_commit": "actual_full_commit",
  "dirty_diff_sha256": "actual_hash_or_null",
  "scientific_config_sha256": "actual_hash",
  "operational_config_sha256": "actual_hash",
  "analysis_mode": "one_of_the_endpoint_modes_in_section_7_2",
  "protocol_sha256": "actual_hash",
  "dependency_lock_sha256": "actual_hash",
  "hardware_record": "relative_path",
  "data_manifest_sha256": "actual_hash_or_null_for_generated_data",
  "prior_asset_sha256": "actual_hash_or_null",
  "information_access": "oracle_or_estimated_or_none",
  "timing_boundaries": "explicit_recorded_definition",
  "status": "running_completed_interrupted_invalid_or_failed"
}
```

The strings above are schema descriptions, not fabricated hashes to copy into results. Actual run creation must populate and validate them.

`task_metrics.csv` or JSONL fields: `run_id`, `outer_replication`, `task_id`, `task_key`, `method_id`, `information_access`, `n`, `d`, `depth`, `norm_cap`, `parameter_cap`, `eta`, `branch`, `loss_parameter_mse`, `loss_noiseless_prediction_mse` when measured, `adaptation_ns`, `prediction_ns`, `status`, `failure_reason`.

`preparation_risks.jsonl` fields: `run_id`, `outer_replication`, `source_key`, `method_id`, `selection_procedure_hash`, `prior_asset_hash`, `metric_hash`, `analysis_mode`, `source_training_tasks`, `source_validation_tasks`, `total_source_labels`, `conditional_risk`, `paired_contrast_id`, `paired_risk_difference`, `risk_evaluator_version`, `true_S_evaluator_hash`, `noise_truth_evaluator_hash`, `numerical_error_assessment`, `status`, `failure_reason`. Truth hashes belong to the evaluator record, not the method's accessible prior. A record's risk is deterministic conditional on its prepared map; its replication is the independent statistical unit.

`geometry.jsonl` fields: target/realized hashes, spectral scale, exact/approximate support policy, minimum eigenvalue, `gamma`, initial/post norm, actual trainable count, total stored scalars/bytes, null residual, kernel absolute/relative error, direct-step error, numerical repairs.

`certificates.jsonl` fields: configuration key, F/Z branch, objective, computed primal upper/dual lower values, primal slack, dual feasibility, stationarity/complementarity residual, solver status/version/options, repair size, `certificate_level`, roundoff treatment/enclosures, claimed risk precision, and exact rational numerators/denominators when applicable. Use the word certified in reporting only for exact rational or rigorous enclosure levels.

`timing_blocks.csv` fields: method order, cold/warm status, block size, generation/preparation/adaptation/prediction/serialization boundaries, repetitions, wall time, peak RSS with platform units, power/thermal condition, and optional GPU allocation/synchronization metadata. macOS `ru_maxrss` and Linux `ru_maxrss` have different units; normalize explicitly. Use process isolation or a suitable sampler when a cumulative process peak cannot isolate a method.

`decisions.jsonl` fields: timestamp, work package, observation, evidence paths/hashes, decision, rationale, affected claim IDs, next action, and whether a methodological amendment/fresh confirmation is required.

### 9.3 Reporting requirements

The final experimental report must contain the complete risk-versus-norm-versus-count frontier and the selected operating points; actual resource use; strongest equal-cap and operational comparisons; uncertainty with the correct independent unit; all important regressions; failed solver/numerical/asset cases; and a list of proposed work not executed.

Report every prespecified scenario even when unfavorable. Present exploratory and confirmatory tables separately. Raw measured numbers, source-provided results, deductions, and forecasts must have explicit evidence-status labels. A result that is exact in the model but unhelpful operationally is still worth reporting accurately.

Use a claim ledger:

| Claim | Estimator/architecture scope | Baseline and metric | Evidence type | Status/limitation |
|---|---|---|---|---|
| Example: exact realization cost | Fixed depth/coordinates/all entries trainable | Universal lower bound plus equality construction | Proof argument and numerical checks | No formal proof-assistant or priority certificate |
| Example: learned-prior benefit | Frozen source/test population and equal caps | Best eligible B01/B02 variants, paired conditional expected MSE | Protected outer-replication comparison | May be inconclusive or negative |
| Example: practical efficiency | Named M3 configuration and timing boundary | Compact direct map/ridge/EB | Local measured Pareto comparison | Cannot infer energy or broader deployment |

Every plot should show units, population/configuration, whether values are analytic or sampled, confidence interpretation, and the count of independent units. Save standalone SVG/PDF or PNG plots and underlying CSVs when the execution agent creates them; do not use generated artwork for precise scientific diagrams.

## 10. Outcome interpretation and evidence-triggered continuation

| Outcome | Correct conclusion | Next action |
|---|---|---|
| Original extremum and joint frontier survive; no exact prior art found | A potentially distinct architecture-specific mathematical result with constructive local evidence. | Seek expert mathematical review and a more specialized priority check before claiming a first result. Publish no stronger application claim. |
| An equivalent prior theorem is found | Replication and possibly a useful engineering/empirical extension. | Cite it, reuse its proof/implementation where licensed, and reassess whether the joint-count consequence is already included. |
| Correct frontier beats optimized Z/isotropy in some equal-cap regimes | A real constrained mechanism advantage, with exact resource expense. | Report full favorable/tie/reversal regions; continue estimated-prior tests if useful. |
| Advantage needs extra actual width and disappears on an operational frontier | A mathematical tradeoff, not an overall efficiency win. | Keep the theory; stop broad efficiency language. |
| Learned-prior benefit disappears or requires excessive source data | The oracle geometry is valid, but acquiring it limits usefulness. | Quantify the source-data/eigenspace-accuracy boundary; do not keep increasing data solely to obtain a positive result. |
| Direct structured `Gz` dominates the network | The architecture is an optimal realization within its model, not the preferred software estimator. | Deliver the compiler/frontier as a research tool; use the direct map for a practical implementation unless network integration supplies a documented benefit. |
| Ridge/EB is better in risk with acceptable cost | One-step constrained optimality does not produce a superior practical learner. | Retain the appropriate component claim; do not weaken those baselines. |
| Tiny but certified deterministic gap | Mathematically real, practical significance unestablished. | Report absolute risk, resource deltas, and required empirical sensitivity; avoid breakthrough language. |
| Wide empirical interval or conflicting outer replications | Inconclusive for the declared population. | Diagnose variance/heterogeneity; design a new fixed confirmation only if expected information justifies it. |
| Source/implementation mismatch | Invalid measurement until resolved. | Repair and replay affected descendants; preserve the failure trail. |
| Need for scale beyond the laptop is justified by a specific missing mechanism | Local result is limited, not a full refutation. | State the missing scale and its prediction quantitatively. Do not invoke unspecified future scale to excuse an already expressive local negative result. |

**Preferred conditional pivot:** if initial norm with fixed `eta` has no useful operational interpretation, investigate a declared bound on post-update norm, update magnitude, or finite-precision robustness. Equation (10) makes that question concrete, but a new resource model needs its own optimization, prior-art check, and preregistration. It is not a retroactive reinterpretation of the present theorem.

**Later architectural expansion:** pursue vector outputs, nonlinear gates, or multiple updates only when an identified use case or a residual limitation of the verified mechanism motivates it. Begin with the smallest example that expresses the new operator/coupling, not a large LLM. A linear component theorem should neither be rejected for lacking an end-to-end deep-model win nor promoted as evidence for one.

## 11. Audit record, unresolved risks, and limitations

### 11.1 Audits performed while preparing this plan

Three separate model-based review tracks examined scientific/prior-art validity, mathematical/experimental validity, and implementation/feasibility. The table records concrete issues that changed the plan. These reviews are not external peer review, independent human replication, or a formal correctness guarantee.

| Audit | Defect or weakness found | Revision and remaining limitation |
|---|---|---|
| Scientific/prior art | Broad novelty for preparing geometry at fixed output was contradicted by NTT/ASI/plasticity injection. | Narrowed novelty to the exact extremum and its induced joint frontier; added closest predecessors. Historical priority remains provisional. |
| Scientific comparison | Generic rank/eigenweight selection is already studied in spectral meta-learning and reduced-rank estimation. | The proposed difference is the architecture-specific `m` versus `m-1` branch constraint, not shrinkage itself. |
| Scientific comparison | Finite empirical kernel matching is not inherently weaker in a linear model: full-rank probes identify `G`. | Removed that as a substantive novelty distinction; compare exact global resource characterization with approximate transfer optimization and different trainability rules. |
| Mathematical | The source's convex averaging argument cannot justify a width-constrained extension. | Added clipping plus trace rearrangement and a complete two-family reduction; allowed nonunique orientations. |
| Implementation | Absolute `1e-12` gap truncation produced 44.72% relative error on a small valid target. | Required exact support metadata, scale-aware policies, repair accounting, and adversarial scale tests. |
| Experimental | Direct `Gz` is function-identical, so apparent superiority over it would be a bug. | Made it an equality gate and an operational baseline with equally compact implementation. |
| Baseline fairness | A true deep final-zero-head baseline has a different factorization cost; a full-rank depth-two head-zero metric must not use the canonical floor constructor. | Primary risk comparison uses depth two and separate constructors; deeper cost formula and rank constraints are explicit. |
| Interpretation | Free learning-rate scaling and a depth-one model undermine a universal cost-of-plasticity claim. | Fixed the main coordinate/step-size contract and added scaling, single-layer, and post-update-norm controls. |
| Experimental | A generic 'best baseline' could accidentally include the identical-map control or an oracle in the learned-prior primary contrast. | Explicit eligible B01/B02/empirical-fitting sets; operational and oracle comparisons are separate. |
| Statistical | Repeated test tasks from one estimated prior would understate preparation variability. | Outer source-training/validation replications, variance-component planning, and fixed-sample confirmation. |
| Experimental efficiency | The fixed-map Gaussian risk is available even for an estimated, misaligned prior; nested simulation would add avoidable variance. | Evaluate conditional risks exactly per preparation replication. Retain task simulations for update checks, X-dependent estimators, and nonanalytic populations; propagate endpoint-specific counts and schemas. |
| Resource fairness | A richer-spectrum `k=0.4` fixture matched caps but used 14 versus 21 minimum trainable entries. | Report that distinction; use the exact rational `k=0.6` fixture with norm `0.6` and 21 entries for both branches as the primary oracle mechanism point. |
| Numerical rigor | A float64 primal/dual gap is not a roundoff-enclosed certificate; empirical QPs may have singular Hessians. | Separate verification levels, add exact rational KKT checks and a convex residual bound that does not invert the empirical Hessian. |
| Implementation handoff | A `src` layout was not installed before module commands; preparation omitted target context size; one work-package dependency was circular. | Add editable bootstrap/interpreter paths, explicit `PreparationSpec`, and move the general F solver to WP04 after source baselines. |
| Recovery | Treating every configuration change identically would forbid faithful batch-size recovery. | Separate scientific and operational hashes, keep task identities invariant, and isolate performance strata. |
| Feasibility | Nominal Apple GPU support does not imply float64/complex128 support. | CPU-double core; MPS is an optional validated float32 arm. Actual Mac compatibility/throughput remains WP00/WP06 work. |
| Evidence | The source contains failed nonsmooth full-matrix diagnostics, and an original-script pass does not validate every numerical scale. | Preserve failures and use linear-constraint QPs with independent certificates. |

After these revisions, targeted rechecks of the mathematics/experimental protocol, source/novelty claims, and fresh-laptop workflow found no further material correction in the reviewed scope. Final file checks verified UTF-8 decoding, all 12 sections and 11 work packages, citation definitions, balanced code/math blocks, valid example JSON/TOML, and the corrected interpreter/source paths. This closes the planning audit, not the proposed execution gates or the remaining scientific questions.

### 11.2 Required challenge pass before the execution report

The execution agent must revisit each scenario: exact prior art exists; strongest local estimator is outside the theorem's information class; memory is scarce; a critical operation fails; optional data/license is unavailable; a synthetic success fails on natural covariates; width requirements are not expressed by the chosen toy; the gain is small or vanishes after tuning; mean improvement conceals shifted-task regression; interrupted output resumes inconsistently; and the result is negative or inconclusive. Section 7's gates and Section 10's outcomes specify the response.

Domain-specific challenges additionally include hidden trainability of zeros, loss-normalization factors, basis degeneracy, numerical rank versus exact rank, covariance leakage, accidental reuse of a mutated initializer, full-rank head-zero miscompilation, sequential instead of simultaneous updates, and using a source-subspace oracle without charging information/storage. A remaining material issue blocks only the affected claim, not the valid parts of the research.

### 11.3 Residual uncertainty

The proofs have not been checked in a proof assistant or externally peer reviewed. A bounded literature search cannot establish historical priority, patentability, or freedom to operate. Official delivery of one recent spectral paper and some legacy repository details remained blocked; the actual inspected alternatives are recorded below. The source does not prove a biological theory, a physical energy law, multistep learning improvement, nonlinear-model gains, or an inexpensive way to discover the task prior.

The fresh numerical checks are finite, exploratory, and from a different platform. No M3 Max performance, protected learned-prior result, natural-covariate transfer, or end-to-end application gain has been measured here. Those unknowns have explicit early gates or contingent experiments; they are not implied successes.

## 12. References, evidence provenance, and search coverage

### 12.1 Supplied evidence and computations

**S00 — Supplied research note and bundle.** *The cost of preparing a network to learn: A sharp norm formula for zero-output deep linear networks and an isotropic-to-specialized adaptation transition*, credited in the note to “An autonomous mathematical investigation,” dated 31 October 2026. Unpublished user-supplied material; no public publication URL or redistribution license was verified. Inspected both 14-page PDFs, LaTeX theorem/proof sections, the verification/geometry/phase scripts, stored numerical outputs, and the main phase figure. The source is evidence for its stated mathematics and recorded experiments, not external peer-reviewed confirmation.

The attached PDF SHA-256 is `2e0bf090bb2c8504e513b08b14c3bb1ed356dde5d7dd39ebb90f30b281f430b1`. The bundle's `research_note.pdf` SHA-256 is `d21569825103249a352bba5e69f951bcc163dbc748b33df9124ff637920de4f9`. Their extracted text agrees after whitespace normalization. The essential statements, proof outline, constructor, risk model, phase formulas, counterexample, and fixture values have been restated in this document, so access to either file is not an execution prerequisite.

Fresh source reruns and additional checks are reported in Section 1.3 with their actual environment. The support-QP exploration used seed `91720261031`; the broader Gaussian/Bayes diagnostic used seed `2026103117` and 50,000 tasks; the post-update-norm check used seed `7091`. Seeds identify those exploratory calculations, not the protected data for the future PoC. Reimplementation need not reproduce bitwise Monte Carlo means across versions; it must reproduce the deterministic formulas and statistically compatible behavior. The exact-fraction fixture is fully reconstructible from Section 4.6 and does not depend on an intermediate JSON file.

### 12.2 Primary research references

**[R01] Arthur Jacot, Franck Gabriel, Clément Hongler (NeurIPS 2018).** *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* Inspected the kernel definition and learning-dynamics setup. Establishes the Jacobian-Gram perspective; its infinite-width claims are not assumed in the finite linear theorem here. [Paper](https://papers.nips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf).

**[R02] Tianlin Liu and Friedemann Zenke (ICML 2020).** *Finding trainable sparse networks through Neural Tangent Transfer.* Inspected §§3.1–3.3, especially Eq. (14), and experimental setup. The output/kernel-matching objective is the closest conceptual predecessor. Its sparse transfer optimization does not supply the inspected theorem's global all-entry norm/count extremum. [Proceedings](https://proceedings.mlr.press/v119/liu20o.html); [paper](https://proceedings.mlr.press/v119/liu20o/liu20o.pdf). Repository access is recorded separately in R28.

**[R03] Sanjeev Arora, Nadav Cohen, Elad Hazan (ICML 2018).** *On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization.* Inspected the scalar warmup and depth-induced preconditioning discussion. Supports treating learning metrics induced by linear factorization as prior art; its optimization setting differs from the prescribed-zero-output extremum. [Paper](https://proceedings.mlr.press/v80/arora18a/arora18a.pdf).

**[R04] Yaoyu Zhang, Zhi-Qin John Xu, Tao Luo, Zheng Ma (MSML 2020).** *A type of generalization error induced by initialization in deep neural networks.* Inspected §6 and Theorem 4 on antisymmetrical initialization. Paired, normalized copies provide prior zero-output/kernel-preserving preparation. Their parameterization and doubling costs must be distinguished from this theorem's coordinates. [Paper](https://proceedings.mlr.press/v107/zhang20a/zhang20a.pdf).

**[R05] Mikhail Khodak, Neil Tenenholtz, Lester Mackey, Nicolò Fusi (ICLR 2021; inspected arXiv v2, 2022).** *Initialization and Regularization of Factorized Neural Layers.* Inspected the initialization/regularization discussion and Appendix B, Eq. (6), containing the two-factor finite-update expansion. The cancellation used here is a specialization of existing algebra, not a separate novelty claim. [Paper](https://arxiv.org/pdf/2105.01029).

**[R06] Zhen Dai, Mina Karzand, Nathan Srebro (NeurIPS 2021).** *Representation Costs of Linear Neural Networks: Analysis and Design.* Inspected Theorem 1 and its factorization-cost interpretation. Used for the true final-zero-head baseline, whose preceding factors realize a fixed Gram. Fixing an end-to-end matrix alone does not fix the tangent at the zero map. [Paper](https://papers.nips.cc/paper_files/paper/2021/file/e22cb9d6bbb4c290a94e4fff4d68a831-Paper.pdf).

**[R07] Sebastian Flennerhag, Andrei A. Rusu, Razvan Pascanu, Francesco Visin, Hujun Yin, Raia Hadsell (ICLR 2020).** *Meta-Learning with Warped Gradient Descent.* Inspected §2.3, Eqs. (7)–(9). Supports the established idea of a learned Jacobian-induced geometry. Its nonlinear task benchmarks and resource conventions are not directly comparable to this PoC. [Paper](https://arxiv.org/pdf/1909.00025).

**[R08] Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, André Barreto (NeurIPS 2023).** *Deep Reinforcement Learning with Plasticity Injection.* Inspected §4, Eq. (1), and construction details. A trainable fresh head with frozen cancellation preserves predictions; this motivates the storage-versus-trainability distinction. The RL results are contextual, not local head-to-head evidence. [Paper](https://papers.nips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf).

**[R09] Edwin K. P. Chong (2022).** *Well-Conditioned Linear Minimum Mean Square Error Estimation.* Inspected §II A–E for unconstrained and rank-constrained linear estimators and conditioning. Supplies a primary derivation and historical references for the LMMSE/SVD baselines. The `Gamma` feasible set here is an additional architectural constraint. [Paper](https://arxiv.org/pdf/2201.02275); [DOI](https://doi.org/10.1109/LCSYS.2022.3162404). Historical tracing also inspected Yingbo Hua, Maziar Nikpour, Petre Stoica, *Optimal Reduced-Rank Estimation and Filtering* (2001), introduction/§II via its institutional copy, [DOI](https://doi.org/10.1109/78.905856); equation extraction was incomplete, so no formula in this handoff depends on that extraction.

**[R10] Yue Sun, Adhyyan Narang, Halil Ibrahim Güllük, Samet Oymak, Maryam Fazel (NeurIPS 2021).** *Towards Sample-efficient Overparameterized Meta-learning.* Inspected §3, Theorem 1/OPT-REP, and §5, Theorem 3. Establishes spectrum/dimension optimization and covariance-estimation concerns in weighted interpolation. Its asymptotic and covariance assumptions differ from the finite-context first-step objective here. [Paper](https://proceedings.neurips.cc/paper/2021/file/ed46558a56a4a26b96a68738a0d28273-Paper.pdf).

**[R11] Nilesh Tripuraneni, Chi Jin, Michael I. Jordan (ICML 2021).** *Provable Meta-Learning of Linear Representations.* Inspected Algorithms 1–2, Gaussian-design theory, and transfer/negative-transfer discussion. Informs the moment-direction comparator and source-data accounting. Projected ridge and the split-block covariance estimator in this plan are specified here; do not misattribute them as the paper's unchanged algorithms. [Proceedings](https://proceedings.mlr.press/v139/tripuraneni21a.html); [paper](https://proceedings.mlr.press/v139/tripuraneni21a/tripuraneni21a.pdf).

**[R12] Mathieu Even and Laurent Massoulié (2025 preprint).** *Meta-learning of shared linear representations beyond well-specified linear regression.* Inspected the introduction and low-rank/nuclear-norm formulation and scope, not every proof appendix. Relevant to learning the prior orientation and task-count requirements; it is not a cost-free-prior justification. [HTML v1](https://arxiv.org/html/2501.18975v1).

**[R13] Yanhao Jin, Krishnakumar Balasubramanian, Debashis Paul (submitted 27 March 2024).** *Meta-Learning with Generalized Ridge Regression: High-dimensional Asymptotics, Optimality and Hyper-covariance Estimation.* Inspected the primary PDF's model and generalized-ridge setup, especially its opening equations; did not verify every asymptotic proof. Supports covariance-aware ridge/empirical Bayes as a strong operational comparator. The finite formulas in Section 5 are explicitly supplied rather than delegated to the paper. [Record](https://arxiv.org/abs/2403.19720); [paper](https://arxiv.org/pdf/2403.19720).

**[R14] Rotem Mulayoff and Tomer Michaeli (ICML 2020).** *Unique Properties of Flat Minima in Deep Networks.* Inspected the linear-network setup, Hessian/Jacobian relation, and sharpness characterization, including Lemma 3 and Theorem 1. Close mathematical sensitivity precedent; minimizing one sharpness statistic is different from prescribing a complete tangent and minimizing resources. [Paper](https://proceedings.mlr.press/v119/mulayoff20a/mulayoff20a.pdf).

**[R15] Pierre Marion and Lénaïc Chizat (NeurIPS 2024).** *Deep linear networks for regression are implicitly regularized towards flat minima.* Inspected Theorem 1, sharpness comparisons, and initialization assumption A1 in v2. Relevant fixed-function sensitivity analysis; it does not establish the specific full-tangent extremum in the inspected results. [HTML v2](https://arxiv.org/html/2405.13456v2).

**[R16] Andries Rosseau, Robert Müller, Ann Nowé (ICML 2026).** *Preserving Plasticity in Continual Learning via Dynamical Isometry.* Verified the primary proceedings record and inspected §2.2 and isotropy-as-prior discussion. This makes dynamical isometry/plasticity current contextual prior art. The continual nonlinear setting differs from the present component theorem, and isotropic input-space `G` does not mean `XGX.T` is an identity matrix. [Proceedings](https://proceedings.mlr.press/v306/rosseau26a.html); [HTML](https://arxiv.org/html/2606.09762v1).

**[R17] Carl Edward Rasmussen and Christopher K. I. Williams (2006).** *Gaussian Processes for Machine Learning*, Chapter 2. Inspected the Bayesian linear model and posterior equations, especially Eq. (2.8). Supports the exact Gaussian posterior comparator; the low-rank implementation and normalization for this task model are derived in Section 5. [Chapter PDF](https://gaussianprocess.org/gpml/chapters/RW2.pdf).

**[R26] Lukas Braun, Erin Grant, Andrew M. Saxe (ICML 2025).** *Not all solutions are created equal: An analytical dissociation of functional and representational similarity in deep linear neural networks.* Verified the 2025 publication; inspected §§2–3, Theorem 3.7, §4.4, and §5.2/Theorems 5.4–5.6. A close predecessor on fixed-function geometry, norms, and parameter-noise robustness. The inspected object is hidden representation geometry, not the prescribed all-parameter tangent. Its arXiv upload is dated September 2026; that is not its publication year. [Proceedings](https://proceedings.mlr.press/v267/braun25a.html); [full HTML](https://arxiv.org/html/2609.38998v1).

**[R27] Marvin Theiss, Lukas Braun, Andrew M. Saxe, Erin Grant (30 September 2026 preprint).** *Parameter symmetries determine representational geometry in overparameterized nonlinear networks.* Inspected §2 definitions, §6 Proposition 6.1/Corollary 6.2, and Appendix F.1's lower-bound/equality argument. Its minimum-weight-norm selection acts within function-preserving symmetry orbits and characterizes hidden activation Grams. This is a nearby but different constrained object; no historical non-overlap claim beyond the inspected results is warranted. Full HTML became accessible on a later retrieval. [HTML v1](https://arxiv.org/html/2609.39078v1).

**[R29] Jasraj Singh, Enea Monzio Compagnoni, Antonio Orvieto (July 2026 preprint).** *How the Hessian-Spectrum of Neural Networks Depends on Data.* Official full-text routes were blocked during this investigation. The paper text was inspected through a public mirror: §§2.1–2.2, Assumptions 1–2, and Theorems 1–3/Corollary 1. Its spectral setting includes strong balance and design restrictions. It is a coverage lead, not a dependency for the present proofs; no displayed rank formula is reused. [arXiv identity](https://arxiv.org/abs/2607.13631); [inspected mirror](https://www.researchgate.net/publication/410159634_How_the_Hessian-Spectrum_of_Neural_Networks_Depends_on_Data).

### 12.3 Software, hardware, dataset, and implementation evidence

**[R18] NumPy developers (NumPy 2.0.0 release documentation, 2024; accessed 31 October 2026).** Inspected the macOS Accelerate/wheel support notes. Supports choosing native CPU linear algebra; does not validate a particular current Mac installation or this workload's speed. [Release notes](https://numpy.org/doc/2.0/release/2.0.0-notes.html).

**[R19] SciPy developers (SciPy 1.14.0 release documentation, 2024; accessed 31 October 2026).** Inspected Accelerate support and platform notes. Informs the native-wheel strategy; local operation gates and exact version locking remain required. [Release notes](https://docs.scipy.org/doc/scipy/release/1.14.0-notes.html).

**[R20] PyTorch contributors (documentation accessed 31 October 2026).** Inspected MPS double-precision limitations and synchronization/memory APIs. Float64/complex128 are unsupported on MPS, so the core uses CPU precision. The stable MPS API page redirected to a versioned 2.15 page during inspection; this is not a recommended package pin. [Backend notes](https://docs.pytorch.org/docs/main/notes/mps.html); [MPS API](https://docs.pytorch.org/docs/stable/mps.html).

**[R21] SciPy developers (SLSQP API documentation, accessed 31 October 2026).** Inspected objective/constraint tolerances and return semantics. Used for an initial solver implementation, with an independent optimality check; solver success is not a theorem or an enclosure of roundoff. [API documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).

**[R22] Paulo Cortez, António Cerdeira, Fernando Almeida, Telmo Matos, José Reis / UCI Machine Learning Repository (Wine Quality, 2009).** Inspected dataset description, variables, and CC BY 4.0 license. Dataset 186; white-wine table has 11 inputs and 4,898 documented instances. Raw bytes were not downloaded in this investigation. Section 5.6 defines an optional semi-synthetic use and a download/hash/split gate. [Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality); [DOI](https://doi.org/10.24432/C56S3T).

**[R23] NumPy contributors (license text accessed 31 October 2026).** Inspected BSD-style redistribution clauses. Installed wheels may include additional bundled notices; preserve those too. [License](https://numpy.org/doc/stable/license.html).

**[R24] SciPy contributors (license text accessed 31 October 2026).** Inspected BSD-style source license. Wheel/build-specific dependency notices must be recorded separately before redistribution. [License](https://github.com/scipy/scipy/blob/main/LICENSE.txt).

**[R25] Apple (MacBook Pro 16-inch, November 2023 technical specifications; accessed 31 October 2026).** Inspected M3 Max chip and memory configurations. Supports the hardware-discovery requirement; no bandwidth-to-runtime extrapolation or assumption of maximum RAM is made. [Official specifications](https://support.apple.com/en-gb/117737).

**[R28] Liu/Zenke group, official Neural Tangent Transfer repository (README accessed 31 October 2026).** Inspected repository landing page and historical dependency instructions, including Python 3.7, JAX 0.1.55, and jaxlib 0.1.37. Direct source/license-file retrieval was blocked; no license was visible in the retrieved root listing. This means code-reuse terms are unverified, not that a license definitely does not exist. No source from this repository was copied into the proposed implementation. [Repository](https://github.com/fmi-basel/neural-tangent-transfer).

**[R30] Patent discovery lead, JP7275350B2 (published 17 May 2023).** Retrieved public record metadata during a limited keyword search involving neural tangent kernels and initialization. Detailed claims, patent families, jurisdictional status, and legal relevance were not analyzed. It supplies no conclusion about the project's novelty, patentability, or freedom to operate. [Public record](https://patents.google.com/patent/JP7275350B2/en).

### 12.4 Search record and stopping rationale

Searches and source inspection were completed on **31 October 2026** using two search systems, primary publisher/arXiv pages, official repositories/documentation, and citation tracing. Secondary pages were discovery aids; consequential technical decisions above use the papers or official documentation themselves. The Singh paper was read as reproduced full text on a public mirror and is explicitly not relied on for an essential result.

| Search family | Representative queries and tracing | Material effect on the plan |
|---|---|---|
| Exact realization and factorization | Prescribed neural tangent kernel minimum norm; zero-function deep linear Jacobian; matrix-product derivative sensitivity; factorization representation cost. | Identified the norm/derivative distinction and nearby factorization/flatness results; retained a provisional exact-extremum contribution. |
| Prediction-preserving preparation | Zero-output trainable initialization; antisymmetrical initialization; neural tangent transfer; function-preserving plasticity injection; learned gradient geometry. | Found NTT and plasticity injection beyond the source's terminology; removed broad novelty language. |
| Joint rank/spectrum refinement | Optimal low-rank preconditioner; task covariance eigenweighting; reduced-rank Wiener/LMMSE; representation dimension meta-learning. | Found direct shrinkage and spectral meta-learning predecessors; narrowed the refinement to the architecture-induced floor-versus-direction capacity tradeoff. |
| Fixed-function geometry and current work | Deep linear flat minima; minimum-norm hidden representations; parameter symmetry plasticity; 2025/2026 dynamical isometry and Hessian spectrum. | Added recent primary neighbors and traced Theiss backward to Braun; retained their different constrained object and access limitations. |
| Practical/statistical comparisons | Generalized ridge meta-learning; empirical task covariance; Gaussian posterior; shared linear representations and negative transfer. | Required efficient direct maps, ridge/projected ridge, Bayes/EB, and finite-prior uncertainty. |
| Local implementation and assets | Official M3 Max memory; NumPy/SciPy Accelerate; MPS float64 and timing; solver semantics; dependency licenses; UCI covariate provenance. | Chose CPU float64, explicit local operation gates, a small optional legal dataset, and honest resource accounting. |
| Limited patent discovery | Public patent keyword search for neural tangent initialization/kernel terminology. | Recorded a lead only; no legal or exhaustive patent conclusion. |

Backward tracing covered NTT's related initialization work, factorization-cost results, classical reduced-rank estimation, and recent symmetry papers. Forward/current checks covered 2025–2026 representation, isometry, and spectral work. Later retrieval successfully opened Theiss and Braun full texts; they must not be described as permanently inaccessible. The NTT repository's raw implementation/license and Singh's official full-text route remain the concrete access limits.

After adding these competing mechanism classes, repeated targeted searches chiefly produced variants of learned preconditioning, kernel matching, factorization cost, and representation selection rather than a materially different local comparator or an equivalent exact `Gamma`/width theorem. That is the stopping rationale for this bounded investigation, not proof that no equivalent result exists. Matrix-analysis literature with different terminology, older non-digitized work, specialized sensitivity theorems, unpublished manuscripts, and comprehensive patent-family research remain blind spots. The next priority review should be triggered by an intended publication claim or a concrete new lead, not used to postpone the fully specified local falsification program.

[R01]: https://papers.nips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf
[R02]: https://proceedings.mlr.press/v119/liu20o.html
[R03]: https://proceedings.mlr.press/v80/arora18a/arora18a.pdf
[R04]: https://proceedings.mlr.press/v107/zhang20a/zhang20a.pdf
[R05]: https://arxiv.org/pdf/2105.01029
[R06]: https://papers.nips.cc/paper_files/paper/2021/file/e22cb9d6bbb4c290a94e4fff4d68a831-Paper.pdf
[R07]: https://arxiv.org/pdf/1909.00025
[R08]: https://papers.nips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf
[R09]: https://arxiv.org/pdf/2201.02275
[R10]: https://proceedings.neurips.cc/paper/2021/file/ed46558a56a4a26b96a68738a0d28273-Paper.pdf
[R11]: https://proceedings.mlr.press/v139/tripuraneni21a.html
[R12]: https://arxiv.org/html/2501.18975v1
[R13]: https://arxiv.org/abs/2403.19720
[R14]: https://proceedings.mlr.press/v119/mulayoff20a/mulayoff20a.pdf
[R15]: https://arxiv.org/html/2405.13456v2
[R16]: https://proceedings.mlr.press/v306/rosseau26a.html
[R17]: https://gaussianprocess.org/gpml/chapters/RW2.pdf
[R18]: https://numpy.org/doc/2.0/release/2.0.0-notes.html
[R19]: https://docs.scipy.org/doc/scipy/release/1.14.0-notes.html
[R20]: https://docs.pytorch.org/docs/main/notes/mps.html
[R21]: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
[R22]: https://archive.ics.uci.edu/dataset/186/wine+quality
[R23]: https://numpy.org/doc/stable/license.html
[R24]: https://github.com/scipy/scipy/blob/main/LICENSE.txt
[R25]: https://support.apple.com/en-gb/117737
[R26]: https://proceedings.mlr.press/v267/braun25a.html
[R27]: https://arxiv.org/html/2609.39078v1
[R28]: https://github.com/fmi-basel/neural-tangent-transfer
[R29]: https://arxiv.org/abs/2607.13631
[R30]: https://patents.google.com/patent/JP7275350B2/en

