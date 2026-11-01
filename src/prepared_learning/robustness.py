"""Prespecified exploratory mechanism diagnostics, separate from protected inference.

Boundary configurations are deliberately invalid under specs.validate_config.
They are consumed only here; Gaussian risk is used only for core fixed maps.
"""
from collections.abc import Sequence
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import numpy as np
from .records import (RunConfig, PriorAsset, BudgetSpec, PreparationSpec, FrontierResult,
                      JsonRecord, GateDecision, GateCode, SupportMetadata, PopulationTruth,
                      TaskKey, content_hash, to_json_record)
from .specs import validate_config
from .protocol import config_from_record
from .tasks import population_for, generate_source_task
from .frontier import solve_frontier
from .metrics import CompactMetric, dense_metric, apply_metric, metric_resources
from .geometry import construct_canonical, construct_head_zero, effective_coefficient, tangent_matrix
from .evaluation import conditional_risk

ARM_NAMES = ('no_floor', 'isotropy', 'actual_counts', 'same_map', 'wrong_basis',
             'prior_uncertainty', 'prior_rotation', 'leakage', 'mean_shift',
             'anisotropic', 'student10', 'student3', 'orthogonal_coordinates',
             'nonorthogonal_coordinates', 'free_eta', 'depth_one', 'fixed_feature',
             'approximate_width', 'float32', 'extreme_scale', 'second_step',
             'noncanonical', 'degeneracy', 'information', 'shared_optimizations', 'resume_error')
# Each tuple states the scientific question and the exact intervention. These
# declarations are frozen into the manifest and checked before execution.
ARM_SEMANTICS = {
    'no_floor': ('Does the floor improve feasible allocation?', 'Solve F and Z independently at identical caps; additionally report infeasible unchanged-gain b=0 control.'),
    'isotropy': ('Does specialization beat scalar gain?', 'Optimize scalar min(k, 1/((n+1)/n+d*(1+sigma2)/n)); width one.'),
    'actual_counts': ('Are caps masking different minimum counts?', 'Richer spectrum at k=0.6 and 0.4, L=2, width cap 3; zero-pad allocated width to 3 and report minimum counts separately.'),
    'same_map': ('Does representation alter the first update?', 'Canonical network, final-zero-head construction, and compact Gz; identical map and context, separate actual storage/counts.'),
    'wrong_basis': ('Is orientation informative beyond spectrum?', 'Independent QR rotation of the metric basis; preserve all eigenvalues, caps and actual norm/count.'),
    'prior_uncertainty': ('What is the finite-source preparation cost?', '3 independent replications of 8 observed split tasks; shrinkage 0, 0.5, 1 and empirical F/Z gains at regularization 0; same observations and known noise; no test selection.'),
    'prior_rotation': ('Does a fixed source/test rotation alter performance?', 'Source flat rank max(1,d//3); test principal angles pi/6; 24 paired test tasks, mean zero and trace one.'),
    'leakage': ('Does off-source task mass alter performance?', 'Source flat rank max(1,d//3); test 0.8 source mass plus 0.2 orthogonal-complement mass, trace one; 24 paired tasks.'),
    'mean_shift': ('Does an inherited answer confound the interpretation?', 'Test mean 0.5 times first source eigenvector; covariance unchanged; 24 paired tasks; second moment trace rises to 1.25.'),
    'anisotropic': ('What fails outside input isotropy?', 'Input covariance diag(linspace(0.5,1.5,d)); 24 paired sampled tasks; actual coefficient and prediction loss, no Gaussian analytic risk.'),
    'student10': ('Does a finite-moment non-Gaussian law change risk?', 'Independent standardized Student df=10 inputs; 24 paired sampled tasks; finite eighth moment; descriptive-only.'),
    'student3': ('What fails when input fourth moments do not exist?', 'Independent standardized Student df=3 inputs; 24 paired sampled tasks; infinite fourth moment, no ordinary mean-risk CI.'),
    'orthogonal_coordinates': ('Are orthogonal coordinates invariant?', 'QR orthogonal input rotation applied to W; compare transformed tangent and norm with T G T^T.'),
    'nonorthogonal_coordinates': ('Is Euclidean resource cost coordinate-dependent?', 'Input rescaling diag(linspace(0.5,2,d)) applied to W; report changed norm and tangent; no invariant cost claim.'),
    'free_eta': ('Does learning-rate freedom destroy norm-only cost?', 'Multiply every layer by 2; eta=2^(-2*(L-1)); initial/post norms and exact null norm identity; initial norm cap deliberately relaxed.'),
    'depth_one': ('What does an outside-class zero linear predictor achieve?', 'One zero d-parameter linear layer, eta=1, identical observed context; norm zero; outside deep architecture.'),
    'fixed_feature': ('How does frozen feature accounting change the cost?', 'Z square-root features stored and frozen, zero head alone trained; same map/context; feature storage separate from trainable head count.'),
    'approximate_width': ('Are exact width jumps practically relevant?', 'For every width 1 through cap-1 drop smallest F increments and independently reoptimize both F/Z; exact conditional formula evaluated in float64.'),
    'float32': ('Does the literal mechanism survive lower precision?', 'Cast initialized layers to float32, execute float32 gradients and simultaneous updates; double-precision evaluation/accounting separately.'),
    'extreme_scale': ('Where do finite precision and support fail?', 'Gain scales 1e-30, 1e-14, 1, 1e14, 1e30 at float64 and float32; overflow/support failures are sensitivity outcomes.'),
    'second_step': ('Where does first-step equivalence end?', 'Depth max(3,L); recompute every second-step derivative including scalar suffixes; compare with fixed initial G recurrence on same context.'),
    'noncanonical': ('Does the harness detect nonlinear null-network terms?', 'Rational depth-three scalar counterexample with h=0.2 and four random null products; compare actual simultaneous update to tangent prediction.'),
    'degeneracy': ('Are multiplicity and support precision-sensitive?', 'Rotate an exactly repeated two-dimensional eigenspace; execute float64/float32 near-minimum spectra [0.2,nextafter(0.2,1)].'),
    'information': ('Does prior access explain performance?', 'Oracle, observed cross-moment estimate, and no-prior isotropic covariance at shared caps and explicitly known noise; all candidates reported.'),
    'shared_optimizations': ('Does shared numerical optimization preserve outputs?', 'Compact/dense F and Z maps, collapsed networks, primal ridge and Cholesky dual ridge with penalty 0.7; same precision/process/BLAS, no timing claim.'),
    'resume_error': ('Can interruption or failed methods bias retained evidence?', 'Create abandoned corrupt staging block; reopen and replay one pair with one method failure; reject duplicate pair and changed config identity.'),
}
LAWS = {'anisotropic':'gaussian_anisotropic', 'student10':'student_t_10', 'student3':'student_t_3'}
DEFERRED = 'Awaiting committed dependencies and fresh scenario-specific pilot, freeze, and confirmation.'


def _deferred(claim):
    return {'claim_id':claim, 'status':'unperformed', 'reason':DEFERRED,
            'amendment_status':'declared_before_execution', 'confirmation_status':'unperformed',
            'required_workflow':['pilot', 'freeze_protocol', 'run_confirmation'],
            'fresh_source_and_test_cohorts_required':True}


def build_robustness_arms(base: RunConfig, arms: Sequence[str]) -> list[RunConfig]:
    validate_config(base, False)
    if not arms or len(set(arms)) != len(arms) or any(a not in ARM_NAMES for a in arms):
        raise ValueError('select distinct known robustness arms')
    if 'exploration' not in base.keys or type(base.keys['exploration']) is not int or base.keys['exploration'] < 0:
        raise ValueError('explicit exploratory root required')
    result = []
    for name in arms:
        metadata = {'arm':name, 'phase':'exploration', 'base_config':to_json_record(base),
                    'uncertainty':'descriptive diagnostic; no confirmatory CI; float64 values are not certificates',
                    'matching':{'caps':'base caps unless explicitly reported outside-class or reduced-width',
                                'information':'oracle covariance for geometry; observed-only estimates for source arms',
                                'data':'paired target contexts; common source observations within a replication'},
                    'amendment':_deferred(name), 'question':ARM_SEMANTICS[name][0],
                    'changes':ARM_SEMANTICS[name][1]}
        budget = replace(base.budget, learning_rate=2.**(-2*(base.budget.depth-1))) if name == 'free_eta' else base.budget
        if name == 'depth_one': budget = replace(base.budget, depth=1)
        p = replace(base.preparation, input_law=LAWS.get(name, base.preparation.input_law))
        result.append(replace(base, workload={**base.workload, 'robustness':metadata}, budget=budget,
                              preparation=p, numerics={**base.numerics,'dtype':'float32' if name == 'float32' else 'float64'},
                              access={'prior':'mixed' if name in ('information','prior_uncertainty') else 'oracle',
                                      'selection':'none', 'noise_access':'known',
                                      'permitted_noise_variance':base.workload['sigma2']},
                              methods=('robustness:'+name,), confirmation={}))
    return result


def oracle_inputs(base):
    truth, artifact = population_for(base, TaskKey(base.keys['exploration'], 0, 0, 0, 0))
    prior = PriorAsset(truth.second_moment, artifact['rotation'], artifact['spectrum'], truth.sigma2,
                       'oracle', {'purpose':'exploratory geometry reference'}, truth.population_hash)
    return prior, base.budget, base.preparation


def reoptimize_approximate_width(prior: PriorAsset, budget: BudgetSpec,
                                task: PreparationSpec, width: int) -> FrontierResult:
    maximum = min(task.input_dimension, (budget.trainable_parameter_cap-budget.depth+2)//(task.input_dimension+1))
    if type(width) is not int or not 1 <= width <= maximum:
        raise ValueError('approximate width must lie within the original cap')
    return solve_frontier(prior, replace(budget, trainable_parameter_cap=(task.input_dimension+1)*width+budget.depth-2), task)


def literal_step(layers, X, y, eta=1.):
    """Recompute ALL derivatives, preserving the chosen parameter dtype."""
    dtype = layers[0].dtype
    if dtype not in (np.dtype('float32'), np.dtype('float64')) or any(a.dtype != dtype for a in layers):
        raise ValueError('one explicit real precision required')
    X, y = np.asarray(X, dtype=dtype), np.asarray(y, dtype=dtype)
    coefficient = effective_coefficient(layers)
    g = X.T @ (X @ coefficient-y) / dtype.type(len(y))
    gradients = []
    for i in range(len(layers)):
        right = np.eye(X.shape[1], dtype=dtype)
        for layer in layers[:i]: right = layer @ right
        left = np.eye(1, dtype=dtype)
        for layer in reversed(layers[i+1:]): left = left @ layer
        gradients.append(left.T @ (g @ right.T)[None, :])
    updated = tuple(a-dtype.type(eta)*g for a,g in zip(layers, gradients))
    if any(a.dtype != dtype for a in (*gradients, *updated)): raise RuntimeError('precision promotion')
    if not all(np.isfinite(a).all() for a in updated): raise FloatingPointError('nonfinite literal update')
    return updated, tuple(gradients)


def _norm(layers):
    return float(sum(np.sum(np.asarray(a, dtype=np.float64)**2) for a in layers))


def _risk(metric, prior, task):
    return conditional_risk(metric, PopulationTruth(prior.covariance, prior.noise_variance,
                            prior.asset_hash, target_context_size=task.target_context_size))


def _branch(branch, prior, budget, task):
    return {'risk':_risk(branch.metric, prior, task), 'solver_status':branch.status,
            'resources':to_json_record(metric_resources(branch.metric, budget.depth)),
            'metric':to_json_record(branch.metric), 'diagnostics':to_json_record(branch.diagnostics)}


def _rotate(rng, d):
    q, r = np.linalg.qr(rng.normal(size=(d,d)))
    return q*np.where(np.diag(r)<0, -1., 1.)


def _population(covariance, mean, **details):
    record = {'covariance':covariance, 'mean':mean, **details}
    return {**record, 'hash':content_hash(record)}


def _sample_boundary(name, prior, budget, task, frontier, rng):
    """Actual losses; no isotropic Gaussian substitution for a changed law."""
    d, n = task.input_dimension, task.target_context_size
    covariance, mean = prior.covariance.copy(), np.zeros(d)
    source = _population(covariance, mean, rotation=prior.basis, leakage_mass=0.)
    details = {'rotation':prior.basis, 'leakage_mass':0.}
    if name in ('prior_rotation', 'leakage'):
        # Prespecified rank-r source with an explicit orthogonal complement.
        r = max(1, d//3); U = prior.basis; source_cov = U[:,:r] @ U[:,:r].T/r
        source = _population(source_cov, mean, rotation=U, leakage_mass=0.)
        if name == 'prior_rotation':
            angle = np.pi/6; V = U.copy()
            for j in range(r):
                V[:,j] = np.cos(angle)*U[:,j]+np.sin(angle)*U[:,r+j]
                V[:,r+j] = -np.sin(angle)*U[:,j]+np.cos(angle)*U[:,r+j]
            covariance = V[:,:r] @ V[:,:r].T/r
            details = {'rotation':V,'principal_angles':[float(angle)]*r,'leakage_mass':0.}
        else:
            covariance = .8*source_cov+.2*(U[:,r:]@U[:,r:].T)/(d-r)
            details = {'rotation':U,'leakage_mass':.2}
        source_prior = replace(prior, covariance=source_cov, eigenvalues=np.r_[np.full(r,1/r),np.zeros(d-r)])
        frontier = solve_frontier(source_prior, budget, task)
    if name == 'mean_shift': mean = .5*prior.basis[:,0]
    test = _population(covariance, mean, **details)
    law = LAWS.get(name, 'gaussian_isotropic'); df = 10 if name == 'student10' else 3 if name == 'student3' else None
    input_cov = np.diag(np.linspace(.5, 1.5, d)) if name == 'anisotropic' else np.eye(d)
    test.update(input_law=law, input_covariance=input_cov)
    test['hash'] = content_hash({k:v for k,v in test.items() if k!='hash'})
    vals, vectors = np.linalg.eigh(covariance); root = vectors*np.sqrt(np.maximum(vals,0))
    losses = []; prediction = []
    for _ in range(24):
        theta = root@rng.normal(size=d)+mean
        X = rng.normal(size=(n,d)) if df is None else rng.standard_t(df, size=(n,d))*np.sqrt((df-2)/df)
        X = X@np.linalg.cholesky(input_cov).T
        y = X@theta+np.sqrt(prior.noise_variance)*rng.normal(size=n); z = X.T@y/n
        errors = [apply_metric(b.metric,z)-theta for b in (frontier.F, frontier.Z)]
        losses.append([float(e@e) for e in errors]); prediction.append([float(e@input_cov@e) for e in errors])
    differences = np.array(losses)[:,1]-np.array(losses)[:,0]
    return {'source_population':source,'test_population':test, 'input_law':law,'student_df':df,
            'moment_status':'infinite_fourth_moment' if df == 3 else 'finite_eighth_moment' if df == 10 else 'gaussian_moments',
            'risk_evaluator':'sampled_coefficient_and_prediction_loss','losses_F_Z':losses,
            'prediction_losses_F_Z':prediction,'signed_differences':differences,
            'tails':{'minimum':float(differences.min()),'maximum':float(differences.max()),
                     'negative_count':int(np.sum(differences<0))},
            'mean_risk_ci':None,'ci_reason':'exploratory only; infinite fourth input moment also invalidates ordinary mean-risk inference' if df == 3 else 'fixed tiny descriptive diagnostic; no confirmation',
            'confirmation':_deferred(name)}


def _source_diagnostics(base, prior, budget, task):
    from .priors import estimate_cross_moment, shrink_prior, build_empirical_qp, fit_empirical_gains
    rows = []
    for replication in range(3):
        sources = [generate_source_task(base, TaskKey(base.keys['exploration'], 41, replication, i, 0)) for i in range(8)]
        estimated = estimate_cross_moment(sources, task.input_dimension)
        # Known noise is explicit and shared; covariance fitting sees observations only.
        estimated = replace(estimated, noise_variance=prior.noise_variance,
                            fit_metadata={**estimated.fit_metadata,'noise_access':'known'})
        estimated = replace(estimated, asset_hash=content_hash({k:v for k,v in to_json_record(estimated).items() if k!='asset_hash'}))
        candidates = {}
        for gamma in (0., .5, 1.):
            shrunk = shrink_prior(estimated, gamma); solved = solve_frontier(shrunk, budget, task)
            candidates[str(gamma)] = {b.branch:_branch(b, prior, budget, task) for b in (solved.F,solved.Z)}
        empirical = {}
        for branch in ('F','Z'):
            qp = build_empirical_qp(sources, estimated.basis, budget, branch, 0.)
            b = fit_empirical_gains(qp, budget, estimated.basis, branch)
            empirical[branch] = _branch(b, prior, budget, task)
        rows.append({'replication':replication,'source_keys':[s.source_key for s in sources],
                     'source_labels':sum(len(s.A.y)+len(s.B.y) for s in sources),
                     'estimated_asset_hash':estimated.asset_hash,'noise_access':'known and shared',
                     'shrinkage':candidates,'empirical':empirical,
                     'signed_risk_difference':candidates['0.0']['Z']['risk']-candidates['0.0']['F']['risk']})
    return {'replications':rows,'uncertainty_unit':'independent source preparation',
            'selection':'all prespecified candidates reported; no test-risk selection',
            'mean_risk_ci':None,'confirmation':_deferred('finite_prior')}


def diagnose_arm(config):
    metadata = config.workload.get('robustness')
    if not metadata or metadata['arm'] not in ARM_NAMES: raise ValueError('explicit arm declaration required')
    base = config_from_record(metadata['base_config']); validate_config(base, False)
    # Reject hand-edited labels/configuration: boundary routing is explicit.
    expected = build_robustness_arms(base, [metadata['arm']])[0]
    if content_hash(config) != content_hash(expected): raise ValueError('arm configuration differs from declaration')
    name = metadata['arm']; prior, budget, task = oracle_inputs(base)
    frontier = solve_frontier(prior, budget, task); F, Z = frontier.F, frontier.Z
    if any(b.status!='completed' for b in (F,Z)): raise ValueError('unresolved base frontier solver')
    rng = np.random.default_rng(np.random.SeedSequence(base.keys['exploration'], spawn_key=(42,)))
    d, n = task.input_dimension, task.target_context_size
    X = rng.normal(size=(n,d)); y = rng.normal(size=n); z = X.T@y/n
    layers = construct_canonical(F.metric, budget.depth).layers
    G = dense_metric(F.metric)
    out = {'arm':name,'executed':True,'status':'completed','uncertainty':metadata['uncertainty'],
           'matching':metadata['matching'],'amendment':metadata['amendment'],
           'interpretation':'exploratory sensitivity; no theorem or comparative confirmation',
           'matched_reference':{'permitted_parameter_cap':budget.trainable_parameter_cap,
                                'permitted_norm_cap':budget.initialization_norm_cap,
                                'information_access':'oracle', 'noise_access':'known',
                                'F_actual_resources':to_json_record(metric_resources(F.metric,budget.depth)),
                                'Z_actual_resources':to_json_record(metric_resources(Z.metric,budget.depth))}}
    if name == 'no_floor':
        zeroed = replace(F.metric, floor=0., increments=F.metric.increments+F.metric.floor)
        out.update(F=_branch(F,prior,budget,task), Z=_branch(Z,prior,budget,task), zeroed_F_risk=_risk(zeroed,prior,task),
                   zeroed_F_norm=metric_resources(zeroed,budget.depth).initialization_norm,
                   zeroed_F_feasible=metric_resources(zeroed,budget.depth).initialization_norm<=budget.initialization_norm_cap)
    elif name == 'isotropy':
        gamma = (budget.initialization_norm_cap/(budget.depth-1))**(budget.depth-1)
        scalar = min(gamma, 1/((n+1)/n+d*(1+prior.noise_variance)/n))
        metric = CompactMetric(scalar,np.empty((d,0)),np.empty(0),SupportMetadata('analytic',0,d,()))
        out.update(scalar=scalar, risk=_risk(metric,prior,task), resources=to_json_record(metric_resources(metric,budget.depth)))
    elif name == 'actual_counts':
        rows=[]
        for k in (.6,.4):
            b = replace(budget,depth=2,initialization_norm_cap=k,trainable_parameter_cap=3*(d+1))
            f = solve_frontier(prior,b,task); nets=[construct_canonical(x.metric,2).layers for x in (f.F,f.Z)]
            padded=[]
            for W,a in nets:
                missing=3-len(W); padded.append((np.pad(W,((0,missing),(0,0))),np.pad(a,((0,0),(0,missing)))))
            rows.append({'k':k,'actual_counts':[sum(a.size for a in net) for net in nets],
                         'minimum_counts':[metric_resources(x.metric,2).trainable_parameters for x in (f.F,f.Z)],
                         'permitted_counts':[b.trainable_parameter_cap]*2,'padded_counts':[sum(a.size for a in net) for net in padded],
                         'padding_map_error':max(float(np.linalg.norm(tangent_matrix(a)-tangent_matrix(b))) for a,b in zip(nets,padded))})
        out['fixtures']=rows
    elif name == 'same_map':
        head=construct_head_zero(F.metric).layers
        errors=[]
        for net in (layers,head):
            updated,_=literal_step(net,X,y); errors.append(float(np.linalg.norm(effective_coefficient(updated)-G@z)))
        out.update(first_step_errors=errors, realization_counts=[sum(a.size for a in net) for net in (layers,head)],
                   direct_stored_scalars=F.metric.basis.size+F.metric.increments.size+1)
    elif name == 'wrong_basis':
        q=_rotate(rng,d); metric=replace(F.metric,basis=q@F.metric.basis)
        out.update(original_spectrum=np.linalg.eigvalsh(G), ablated_spectrum=np.linalg.eigvalsh(dense_metric(metric)),
                   original_resources=to_json_record(metric_resources(F.metric,budget.depth)), ablated_resources=to_json_record(metric_resources(metric,budget.depth)),
                   original_basis=F.metric.basis, ablated_basis=metric.basis,
                   risk_delta=_risk(metric,prior,task)-_risk(F.metric,prior,task))
    elif name in ('prior_rotation','leakage','mean_shift',*LAWS):
        out.update(_sample_boundary(name,prior,budget,task,frontier,rng))
    elif name in ('prior_uncertainty','information'):
        out.update(_source_diagnostics(base,prior,budget,task))
        if name=='information':
            none=replace(prior,covariance=np.eye(d)/d,basis=np.eye(d),eigenvalues=np.full(d,1/d),information_access='none')
            no_prior=solve_frontier(none,budget,task)
            out.update(information_access=['oracle','estimated','none'],oracle={b.branch:_branch(b,prior,budget,task) for b in (F,Z)},
                       no_prior={b.branch:_branch(b,prior,budget,task) for b in (no_prior.F,no_prior.Z)},
                       access_note='same known noise and caps; no-prior covariance is isotropic; oracle truth never enters estimated fitting')
    elif name in ('orthogonal_coordinates','nonorthogonal_coordinates'):
        T=_rotate(rng,d) if name=='orthogonal_coordinates' else np.diag(np.linspace(.5,2.,d))
        transformed=(layers[0]@T.T,*layers[1:]); expected=T@G@T.T
        actual=tangent_matrix(transformed)
        out.update(coordinate_matrix=T,tangent_error=float(np.linalg.norm(actual-expected)),
                   norm_change=abs(_norm(transformed)-_norm(layers)), original_norm=_norm(layers), changed_norm=_norm(transformed),
                   metric_change=float(np.linalg.norm(actual-G)),
                   coordinate_contract='Euclidean coordinates changed; nonorthogonal rescaling is not a resource invariant')
    elif name == 'free_eta':
        scale=2.; exponent=2*(budget.depth-1); eta=scale**(-exponent)
        scaled=tuple(scale*a for a in layers); first,_=literal_step(layers,X,y); updated,_=literal_step(scaled,X,y,eta)
        out.update(scale=scale,depth=budget.depth,eta=eta,initial_norm_ratio=_norm(scaled)/_norm(layers),
                   tangent_ratio=float(np.trace(tangent_matrix(scaled))/np.trace(G)),
                   first_coefficient_error=float(np.linalg.norm(effective_coefficient(first)-effective_coefficient(updated))),
                   post_norm=_norm(updated),norm_identity=_norm(scaled)+eta**2*float(z@tangent_matrix(scaled)@z),
                   post_increment_ratio=(_norm(updated)-_norm(scaled))/(_norm(first)-_norm(layers)))
    elif name == 'depth_one':
        linear=(np.zeros((1,d)),); updated,_=literal_step(linear,X,y)
        out.update(coefficient=effective_coefficient(updated),expected_coefficient=z,
                   initial_norm=0.,trainable_parameters=d,outside_class=True)
    elif name == 'fixed_feature':
        # Rank-Z square-root features; only the zero head is trained.
        head=construct_head_zero(Z.metric).layers; W,a=head
        updated=a+(y@X@W.T/n)[None,:]
        out.update(coefficient=(updated@W).ravel(),direct_error=float(np.linalg.norm((updated@W).ravel()-apply_metric(Z.metric,z))),
                   trainable_parameters=a.size,stored_scalars=W.size+a.size,initial_trainable_norm=0.,outside_class=True)
    elif name == 'approximate_width':
        maximum=min(d,(budget.trainable_parameter_cap-budget.depth+2)//(d+1)); rows=[]
        for width in range(1,maximum):
            order=np.argsort(F.metric.increments)[::-1]; keep=order[:width-1]; drop=order[width-1:]
            metric=replace(F.metric,basis=F.metric.basis[:,keep],increments=F.metric.increments[keep],
                           support_metadata=SupportMetadata('approximate',0,d-len(keep),tuple(map(int,keep)),(),({'dropped_mass':float(F.metric.increments[drop].sum())},)))
            result=reoptimize_approximate_width(prior,budget,task,width)
            selected=min((result.F,result.Z),key=lambda x:x.objective)
            rows.append({'width':width,'dropped_mass':float(F.metric.increments[drop].sum()),'kernel_error':float(np.linalg.norm(G-dense_metric(metric))),
                         'truncated_risk':_risk(metric,prior,task),'reoptimized_risk':selected.objective,'evaluated_risk':_risk(selected.metric,prior,task),
                         'reoptimized_kernel_error':float(np.linalg.norm(G-dense_metric(selected.metric))),
                         'actual_resources':to_json_record(metric_resources(selected.metric,budget.depth)),
                         'branches':to_json_record(result),'selection_status':result.selection_status})
        out['widths']=rows
    elif name == 'float32':
        low=tuple(a.astype(np.float32) for a in layers); updated,gradients=literal_step(low,X,y)
        reference,_=literal_step(layers,X,y)
        out.update(layer_dtypes=[str(a.dtype) for a in low],gradient_dtypes=[str(a.dtype) for a in gradients],update_dtypes=[str(a.dtype) for a in updated],
                   evaluation_dtype='float64',coefficient_error=float(np.linalg.norm(effective_coefficient(updated).astype(float)-effective_coefficient(reference))),
                   post_norm_float64=_norm(updated),interpretation='precision sensitivity; never an exact multiplicity certificate')
    elif name == 'extreme_scale':
        outcomes=[]
        for scale in (1e-30,1e-14,1.,1e14,1e30):
            metric=replace(F.metric,floor=F.metric.floor*scale,increments=F.metric.increments*scale)
            for dtype in (np.float64,np.float32):
                try:
                    net=tuple(a.astype(dtype) for a in construct_canonical(metric,budget.depth).layers)
                    with np.errstate(over='raise',invalid='raise',under='ignore'):
                        updated,_=literal_step(net,X,y)
                    actual=tangent_matrix(tuple(a.astype(float) for a in net)); target=dense_metric(metric)
                    outcomes.append({'gain_scale':scale,'dtype':np.dtype(dtype).name,'status':'completed','relative_kernel_error':float(np.linalg.norm(actual-target)/np.linalg.norm(target)),
                                     'first_coefficient_error':float(np.linalg.norm(effective_coefficient(updated).astype(float)-target@z))})
                except (FloatingPointError,ValueError,OverflowError) as exc:
                    outcomes.append({'gain_scale':scale,'dtype':np.dtype(dtype).name,'status':'sensitivity_failure','reason':str(exc)})
        out.update(outcomes=outcomes,interpretation='numerical support/precision boundary; not a theorem counterexample')
    elif name == 'second_step':
        deep=construct_canonical(F.metric,max(3,budget.depth)).layers
        first,_=literal_step(deep,X,y); second,gradients=literal_step(first,X,y)
        w1=effective_coefficient(first); g1=X.T@(X@w1-y)/n
        recurrence=w1-tangent_matrix(deep)@g1
        out.update(X=X,y=y,first_layers=first,second_gradients=gradients,second_coefficient=effective_coefficient(second),
                   fixed_metric_coefficient=recurrence,fixed_metric_disagreement=float(np.linalg.norm(effective_coefficient(second)-recurrence)))
    elif name == 'noncanonical':
        rational=(np.array([[1.],[0.]]),np.array([[1.,1.],[0.,1.]]),np.array([[0.,1.]]))
        updated,_=literal_step(rational,np.ones((1,1)),np.array([.2]))
        discrepancies=[]
        for _ in range(4):
            W=rng.normal(size=(2,1)); M=rng.normal(size=(2,2)); v=(M@W).ravel(); a=np.array([[-v[1],v[0]]])
            net=(W,M,a); changed,_=literal_step(net,np.ones((1,1)),np.array([.2]))
            discrepancies.append(float(effective_coefficient(changed)[0]-.2*tangent_matrix(net)[0,0]))
        out.update(rational_coefficient=float(effective_coefficient(updated)[0]),rational_expected=.64,
                   rational_disagreement=float(effective_coefficient(updated)[0]-.6),random_signed_disagreements=discrepancies)
    elif name == 'degeneracy':
        U=_rotate(rng,d); rotation=np.eye(d); rotation[:2,:2]=[[0.,-1.],[1.,0.]]
        vals=np.r_[np.full(2,.4),np.full(d-2,.2)]
        first=(U*vals)@U.T; V=U@rotation; second=(V*vals)@V.T
        near=np.array([.2,np.nextafter(.2,1.)]); low=near.astype(np.float32)
        out['near_minimum_execution']={'float64_gap':float(np.diff(np.linalg.eigvalsh(np.diag(near)))[0]),
                                       'float32_gap':float(np.diff(np.linalg.eigvalsh(np.diag(low)))[0]),
                                       'status':'precision_support_loss' if low[0]==low[1] else 'retained',
                                       'interpretation':'not an exact multiplicity certificate'}
        out.update(projector_error=float(np.linalg.norm(U[:,:2]@U[:,:2].T-V[:,:2]@V[:,:2].T)),metric_error=float(np.linalg.norm(first-second)),
                   nearly_repeated_eigenvalues=[.2,np.nextafter(.2,1.)],precision_gap=float(np.nextafter(.2,1.)-.2))
    elif name == 'shared_optimizations':
        penalty=.7; primal=np.linalg.solve(X.T@X+penalty*np.eye(d),X.T@y)
        C=X@X.T+penalty*np.eye(n); chol=np.linalg.cholesky(C)
        dual=X.T@np.linalg.solve(chol.T,np.linalg.solve(chol,y))
        families={}
        for branch in (F,Z):
            net=construct_canonical(branch.metric,budget.depth).layers
            updated,_=literal_step(net,X,y); coefficient=effective_coefficient(updated)
            prediction=X
            for layer in updated: prediction=prediction@layer.T
            families[branch.branch]={'compact_dense_error':float(np.linalg.norm(apply_metric(branch.metric,z)-dense_metric(branch.metric)@z)),
                                     'collapsed_prediction_error':float(np.linalg.norm(X@coefficient-prediction.ravel()))}
        out.update(ridge_primal_dual_error=float(np.linalg.norm(primal-dual)),families=families,
                   precision='float64 shared',thread_policy='same process and BLAS',timing_claim='none; numerical equivalence only')
    elif name == 'resume_error': out.update(_storage_adversary())
    return to_json_record(out)


def _storage_adversary():
    """Exercise abandoned staging, failure pairing and changed scientific identity."""
    from .runner import RunStore, IDENTITY_FIELDS
    from .records import RAW_RECORD_FIELDS
    digest=content_hash({'purpose':'robustness storage diagnostic'})
    run={'schema_version':1,'run_id':'robustness-storage','phase':'exploration','git_commit':'0'*40,
         'dirty_diff_sha256':None,'scientific_config_sha256':digest,'operational_config_sha256':digest,
         'protocol_sha256':digest,'dependency_lock_sha256':digest,'prior_asset_sha256':None,
         'data_manifest_sha256':None,'analysis_mode':'fixed_prior_task_simulation','hardware_record':'diagnostic',
         'information_access':'none','timing_boundaries':{'adaptation':'unmeasured'},'status':'running',
         'pairing_spec':{'task_metrics':{'unit_fields':['outer_replication','task_id'],'method_ids':['F','Z']}}}
    identity={k:run[k] for k in IDENTITY_FIELDS}; rows=[]
    for method in ('F','Z'):
        row={k:None for k in RAW_RECORD_FIELDS['task_metrics']}
        row.update(kind='task_metrics',run_id=run['run_id'],outer_replication=0,task_id=0,task_key='shared',
                   method_id=method,information_access='none',n=4,d=6,depth=2,norm_cap=.6,parameter_cap=21,eta=1.,
                   status='failed' if method=='Z' else 'completed',failure_reason='injected method failure' if method=='Z' else None,
                   loss_parameter_mse=None if method=='Z' else 1.)
        row['null_reasons']={k:'not measured in storage diagnostic' for k,v in row.items() if v is None and k!='failure_reason'}
        rows.append(row)
    with tempfile.TemporaryDirectory() as directory:
        store=RunStore.create(Path(directory),run)
        # An interrupted write has no published receipt. Its corrupt staging
        # payload must never count as a completed task.
        abandoned=store.path/'blocks'/'.interrupted'; abandoned.mkdir(); (abandoned/'records.json').write_text('corrupt partial payload')
        reopened=RunStore.open(store.path,identity); before=len(reopened.committed_keys())
        reopened.commit_block('pair',rows,{})
        recovered=RunStore.open(store.path,identity); records=recovered.read_records('task_metrics')
        rejected=False
        try: RunStore.open(store.path,{**identity,'scientific_config_sha256':content_hash({'changed':True})})
        except ValueError: rejected=True
        duplicate=False
        try: recovered.commit_block('pair',rows,{})
        except ValueError: duplicate=True
        return {'uncommitted_keys_before_replay':before,'recovered_pairs':len(recovered.committed_keys()),
                'failed_rows_retained':sum(r['status']=='failed' for r in records),'config_change_rejected':rejected,
                'duplicate_rejected':duplicate,'replayed_rows':records}


def validate_amendment(amendment, old_protected_keys):
    if not isinstance(amendment,dict) or not isinstance(amendment.get('claim_id'),str) or not amendment['claim_id']:
        raise ValueError('amendment claim required')
    if amendment.get('status')=='unperformed':
        if not isinstance(amendment.get('reason'),str) or not amendment['reason'].strip():
            raise ValueError('unperformed amendment needs a reason')
        if 'protected_keys' in amendment: raise ValueError('unperformed amendment cannot allocate protected keys')
        return
    if amendment.get('status')!='retained': raise ValueError('unknown amendment status')
    keys=amendment.get('protected_keys',{})
    old_roots={k['root_seed'] for k in old_protected_keys}
    if keys.get('root_seed') in old_roots: raise ValueError('retained claim requires fresh source/test keys')
    # This command produces exploratory evidence only. Valid frozen protocols
    # are executed by the existing confirm API; accepting labels here would
    # falsely turn a diagnostic into a confirmation.
    raise ValueError('retained claim requires fresh scenario pilot/freeze/confirmation evidence via existing confirmation APIs; this executor only supports explicit unperformed status')


def make_manifest(base, arms):
    configs=build_robustness_arms(base,arms)
    return {'schema_version':1,'phase':'exploration','purpose':'prespecified tiny fixture diagnostics; no production campaign',
            'base_config':to_json_record(base),'selected_arms':list(arms),
            'arm_declarations':[to_json_record(c.workload['robustness']) for c in configs],
            'old_protected_keys':[], 'amendments':[_deferred(a) for a in arms],
            'transfer_confirmations':[
                {**_deferred('W1'),'scenario_family':{'dimension':64,'rank':8,'spectrum':'flat','n':[4,8,32],'widths':[1,8,9,64],'sigma2':1}},
                {**_deferred('W2'),'scenario_family':{'dimension':64,'spectrum':'normalized i^(-3/2)','n':[8],'widths':[4,8,16],'sigma2':1}}]}


def run_robustness(manifest: JsonRecord, output: Path) -> GateDecision:
    if manifest.get('schema_version')!=1 or manifest.get('phase')!='exploration':
        raise ValueError('robustness accepts exploratory manifests only')
    base=config_from_record(manifest['base_config'])
    configs=build_robustness_arms(base,manifest['selected_arms'])
    expected=make_manifest(base,manifest['selected_arms'])
    if content_hash(manifest['arm_declarations'])!=content_hash(expected['arm_declarations']):
        raise ValueError('arm declarations do not match executed configuration')
    if [a['claim_id'] for a in manifest['amendments']]!=list(manifest['selected_arms']):
        raise ValueError('one amendment record per selected arm required')
    old=manifest['old_protected_keys']
    for amendment in (*manifest['amendments'],*manifest['transfer_confirmations']): validate_amendment(amendment,old)
    if manifest['transfer_confirmations']!=expected['transfer_confirmations']: raise ValueError('prespecified transfer scenario declarations required')
    if base.keys['exploration'] in {k['root_seed'] for k in old}: raise ValueError('exploration cannot reuse protected keys')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'robustness.json').exists(): raise ValueError('refusing to overwrite diagnostic evidence')
    rows=[]
    for config in configs:
        try: row=diagnose_arm(config)
        except (ValueError,RuntimeError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
            metadata=config.workload['robustness']
            row={'arm':metadata['arm'],'executed':True,'status':'failed','failure_reason':str(exc),
                 'uncertainty':metadata['uncertainty'],'matching':metadata['matching'],
                 'amendment':to_json_record(metadata['amendment']),
                 'interpretation':'diagnostic execution failure; not a theorem counterexample'}
        rows.append(row)
    failed=any(row['status']=='failed' for row in rows)
    report={'schema_version':1,'manifest_hash':content_hash(manifest),'phase':'exploration',
            'original_primary_result':'unchanged','arms':rows,'amendments':manifest['amendments'],
            'transfer_confirmations':manifest['transfer_confirmations'],
            'status':'failed' if failed else 'completed','confirmation_performed':False}
    (output/'robustness.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    decision=GateDecision(GateCode.IMPLEMENTATION_INVALID if failed else GateCode.INSENSITIVE_OR_UNSUITABLE,
                          {'report':str(output/'robustness.json'),'status':report['status'],'selected_arms':manifest['selected_arms']},
                          'Diagnostic execution failed; inspect retained failure rows.' if failed else 'Exploratory attribution completed; no protected evidence or retained amended claim.',
                          'Repair failed diagnostics.' if failed else 'After implementation commits, perform scenario-specific pilot/freeze/confirmation for any retained claim.')
    (output/'gate_decision.json').write_text(json.dumps(to_json_record(decision),indent=2,allow_nan=False)+'\n')
    return decision
