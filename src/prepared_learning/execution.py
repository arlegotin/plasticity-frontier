"""Shared scientific source preparation and immutable task execution."""
import numpy as np

def _restore_candidate(row):
    from .records import PriorAsset
    from .methods import make_method
    prior=dict(row['asset'])
    for key in ('covariance','basis','eigenvalues'):
        if prior[key] is not None: prior[key]=np.asarray(prior[key],dtype=float)
    return make_method(row['method_id'],row['options']),PriorAsset(**prior)


def preparation_replication(config, root, outer, phase, methods, *, selectors=None):
    """Trusted driver assigns source authorization before generating observations."""
    import time
    from .pilot import _rss_bytes
    from .records import TaskKey, content_hash, to_json_record
    from .tasks import generate_source_task, population_for
    from .selection import public_method_config, select_method
    from .methods import make_method
    from .priors import estimate_cross_moment
    from .protocol import BASELINE, JOINT
    sample=config.sample_plan
    train_keys=[TaskKey(root,phase,outer,i,0) for i in range(sample.source_training_tasks)]
    val_keys=[TaskKey(root,phase+1,outer,i,0) for i in range(sample.source_validation_tasks)]
    public=public_method_config(config)
    public['source_access_manifest']={'training_keys':[content_hash({'key':key}) for key in train_keys],
                                      'validation_keys':[content_hash({'key':key}) for key in val_keys]}
    generation_start=time.perf_counter_ns()
    train=[generate_source_task(config,key) for key in train_keys]
    validation=[generate_source_task(config,key) for key in val_keys]
    generation_ns=time.perf_counter_ns()-generation_start
    order=list(methods); np.random.default_rng(np.random.SeedSequence(root,spawn_key=(phase,outer,17))).shuffle(order)
    started=time.perf_counter_ns()
    selection=select_method([make_method(mid,{}) for mid in order],train,validation,{**public,'selection_class':'operational'})
    preparation_ns=time.perf_counter_ns()-started
    # Evaluation truth is constructed after fitting, and never enters public config.
    truth,_=population_for(config,TaskKey(root,phase,outer,0,0))
    raw=estimate_cross_moment(train,config.preparation.input_dimension)
    relative_error=float(np.linalg.norm(raw.covariance-truth.second_moment)/np.linalg.norm(truth.second_moment))
    r=min(2,config.preparation.input_dimension)
    _,vectors=np.linalg.eigh(truth.second_moment)
    projection_error=float(np.linalg.norm(raw.basis[:,:r]@raw.basis[:,:r].T-vectors[:,-r:]@vectors[:,-r:].T))
    ledger=to_json_record(selection.candidate_records)
    final=max((row.get('refinement_iteration',0) for row in ledger),default=0)
    selected={}; risks={}; failures={}; assets={}
    selectors = selectors or {'baseline':BASELINE,'joint':JOINT}
    for role,allowed in selectors.items():
        candidates=[row for row in ledger if row.get('status')=='completed' and row['method_id'] in allowed and row.get('refinement_iteration',0)==final]
        if not candidates:
            failures[role]='no completed candidate in final refinement'; continue
        row=min(candidates,key=lambda row:(row['validation_loss'],row['method_id'],row['logical_candidate_id']))
        try:
            method,prior=_restore_candidate(row)
            prepared=method.prepare(prior,config.budget,config.preparation)
            selected[role]={'candidate_id':row['candidate_id'],'method_id':row['method_id'],'asset_hash':prior.asset_hash,
                'metric_hash':content_hash(prepared.metric),'validation_loss':row['validation_loss'],
                'validation_task_losses':row['validation_task_losses'],'prepared_state':to_json_record(prepared),
                'numerical_error_assessment':{'level':'float64_diagnostic'}}
            if prior.covariance is not None: assets[role+'_covariance']=prior.covariance
            if prepared.metric is not None:
                assets[role+'_basis']=prepared.metric.basis
                assets[role+'_increments']=prepared.metric.increments
            for i,layer in enumerate(prepared.initial_layers or ()): assets[f'{role}_layer_{i}']=layer
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            failures[role]=str(exc); risks.pop(role,None); selected.pop(role,None)
    source_keys=public['source_access_manifest']['training_keys']+public['source_access_manifest']['validation_keys']
    return {'selected':selected,'risks':risks,'failures':failures,'candidate_records':ledger,
        'source_keys':source_keys,'source_key':content_hash(public['source_access_manifest']),
        'training_observation_hash':content_hash(train),'validation_observation_hash':content_hash(validation),
        'relative_covariance_error':relative_error,'eigenspace_projection_error':projection_error,
        'tuning_resolution':'resolved' if ledger and all(row.get('tuning_resolution')=='resolved' and row.get('status')=='completed' for row in ledger) else 'unresolved',
        'generation_ns':generation_ns,'preparation_ns':preparation_ns,'method_order':order,
        'truth_second_moment_hash':content_hash(truth.second_moment),'truth_noise_hash':content_hash(truth.sigma2),
        'peak_rss_bytes':_rss_bytes()},assets



def adapt_fresh(method, prepared, context):
    """Defensive fresh state per task, with before/after immutable identity checks."""
    import time
    from dataclasses import replace
    from .records import content_hash
    from .metrics import apply_metric
    snapshot=content_hash(prepared)
    fresh=replace(prepared)
    started=time.perf_counter_ns()
    adapted=method.adapt(fresh, context)
    elapsed=time.perf_counter_ns()-started
    if content_hash(fresh)!=snapshot or content_hash(prepared)!=snapshot:
        raise ValueError('initializer mutation invalidates replication')
    if adapted.status!='completed' or not np.all(np.isfinite(adapted.coefficient)):
        raise ValueError('invalid adapted state')
    error=None
    if prepared.metric is not None:
        direct=apply_metric(prepared.metric,context.X.T@context.y/len(context.y))
        error=float(np.linalg.norm(adapted.coefficient-direct))
        if error>1e-9*max(1.,float(np.linalg.norm(direct))):
            raise ValueError('literal update/direct-map diagnostic failed')
    return adapted, {'adaptation_ns':elapsed,'literal_direct_step_error':error,
                     'initializer_hash':snapshot,'level':'float64_diagnostic'}


def prepared_candidates(result, config):
    """Restore every final candidate, retaining failures; never select by risk."""
    ledger=result['candidate_records']
    final=max((r.get('refinement_iteration',0) for r in ledger),default=0)
    prepared={}; failures={}
    for row in ledger:
        if row.get('refinement_iteration',0)!=final: continue
        key='candidate:'+row['candidate_id']
        try:
            if row['status']!='completed': raise ValueError(row.get('failure_reason','candidate fitting failed'))
            method,prior=_restore_candidate(row)
            prepared[key]=(method,method.prepare(prior,config.budget,config.preparation))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc: failures[key]=str(exc)
    for mid in {r['method_id'] for r in ledger}:
        eligible=[r for r in ledger if r['method_id']==mid and r.get('status')=='completed' and r.get('refinement_iteration',0)==final]
        if eligible:
            chosen=min(eligible,key=lambda r:(r['validation_loss'],r['logical_candidate_id']))
            key='candidate:'+chosen['candidate_id']
            if key in prepared: prepared['family:'+mid]=prepared[key]
    for role,row in result['selected'].items():
        key='candidate:'+row['candidate_id']
        if key in prepared: prepared[role]=prepared[key]
        else: failures[role]=failures.get(key,'missing selected candidate')
    failures.update(result['failures'])
    return prepared, failures


def fixed_preparation(config):
    """Explicit oracle or content-addressed declared prior; no source fitting."""
    from .records import PriorAsset, content_hash
    from .tasks import population_for
    from .records import TaskKey
    from .methods import make_method
    if config.access.get('prior')=='oracle':
        truth,_=population_for(config,TaskKey(0,0,0,0,0))
        values,basis=np.linalg.eigh(truth.second_moment); order=np.argsort(values)[::-1]
        prior=PriorAsset(truth.second_moment,basis[:,order],np.maximum(values[order],0),truth.sigma2,
                         'oracle',{'basis_provenance':'declared oracle population'},content_hash({'S':truth.second_moment,'sigma2':truth.sigma2}))
    else:
        record=dict(config.access['prior_asset']); expected=config.access['prior_asset_sha256']
        if content_hash(record)!=expected: raise ValueError('fixed prior asset hash mismatch')
        for key in ('covariance','basis','eigenvalues'):
            if record[key] is not None: record[key]=np.asarray(record[key],dtype=float)
        prior=PriorAsset(**record)
        if prior.information_access!=config.access['prior']: raise ValueError('fixed information access mismatch')
    prepared={}
    options=config.numerics.get('method_options',{})
    for mid in config.methods:
        if mid.startswith('B05:'): continue
        method=make_method(mid,options.get(mid,{}))
        prepared[mid]=(method,method.prepare(prior,config.budget,config.preparation))
    for mid in config.methods:
        if mid.startswith('B05:'):
            target=mid[4:]
            method=make_method(mid,{'metric':prepared[target][1].metric})
            prepared[mid]=(method,method.prepare(prior,config.budget,config.preparation))
    return prepared


def execute_replication(config, scientific, root, outer, phases):
    """Fit/select once, freeze every candidate, then evaluate prespecified tasks."""
    import time
    from .records import TaskKey, content_hash, to_json_record
    from .tasks import generate_task, population_for
    from .evaluation import conditional_risk, coefficient_loss
    mode=config.sample_plan.analysis_mode
    learned=mode in ('conditional_risk_over_preparations','nested_task_simulation')
    start=time.perf_counter_ns()
    if learned:
        result,assets=preparation_replication(config,root,outer,phases['training_phase'],config.methods,
            selectors=scientific['selectors'])
        prepared,failures=prepared_candidates(result,config)
    else:
        prepared=fixed_preparation(config); failures={}; assets={}
        for role,ids in scientific['selectors'].items():
            if len(ids)!=1: raise ValueError('fixed prior selector must identify one frozen method')
            prepared[role]=prepared[ids[0]]
        result={'candidate_records':[], 'selected':{role:{'method_id':pair[0].method_id,
            'candidate_id':pair[0].method_id,'asset_hash':pair[1].prior.asset_hash,
            'metric_hash':content_hash(pair[1].metric),'prepared_state':to_json_record(pair[1])}
            for role,pair in prepared.items() if role in scientific['selectors']},
            'source_keys':[],'source_key':content_hash({'fixed_prior':scientific['assets']}),
            'tuning_resolution':'inapplicable','generation_ns':0,'preparation_ns':0}
    if mode=='deterministic_frontier':
        from .frontier import solve_frontier
        prior=prepared['joint'][1].prior
        result['frontier_result']=to_json_record(solve_frontier(prior,config.budget,config.preparation))
    # Freeze all state hashes before evaluator truth or test observations exist.
    snapshots={key:content_hash(state) for key,(_,state) in prepared.items()}
    for key,(_,state) in prepared.items():
        if state.initial_layers is not None:
            from .geometry import effective_coefficient
            norm=sum(float(np.sum(layer*layer)) for layer in state.initial_layers)
            count=sum(layer.size for layer in state.initial_layers)
            if norm>config.budget.initialization_norm_cap+1e-10 or count>config.budget.trainable_parameter_cap or np.linalg.norm(effective_coefficient(state.initial_layers))>1e-10:
                failures[key]='initializer violates frozen norm/count/null invariants'
        prefix=content_hash(key)[:20]
        if state.metric is not None:
            assets[prefix+'_basis']=state.metric.basis; assets[prefix+'_increments']=state.metric.increments
        for i,layer in enumerate(state.initial_layers or ()): assets[f'{prefix}_layer_{i}']=layer
    preparation_ns=time.perf_counter_ns()-start-result['generation_ns']
    evaluation_start=time.perf_counter_ns()
    population,_=population_for(config,TaskKey(root,phases['test_phase'],outer,0,0))
    risks={}
    if mode in ('conditional_risk_over_preparations','deterministic_frontier'):
        for key,(_,state) in prepared.items():
            if state.metric is None: continue
            try:
                value=conditional_risk(state.metric,population)
                from .metrics import dense_metric
                G=dense_metric(state.metric); S=population.second_moment; n=population.target_context_size
                dense=float(1-2*np.trace(G@S)+(n+1)/n*np.trace(G@G@S)+(1+population.sigma2)/n*np.trace(G@G))
                error=abs(value-dense)
                precision=scientific.get('precision',{}).get('target_half_width')
                tolerance=min(1e-10*max(1.,abs(value)),precision/100) if precision else 1e-10*max(1.,abs(value))
                if error>tolerance: failures[key]='conditional dense/compact disagreement exceeds declared numerical precision'
                risks[key]={'conditional_risk':value,'numerical_error_assessment':{'dense_compact_absolute_difference':error,'level':'float64_diagnostic'}}
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures[key]=str(exc)
    evaluation_ns=time.perf_counter_ns()-evaluation_start
    tasks=[]
    schedules=[]
    if mode in ('fixed_prior_task_simulation','nested_task_simulation'):
        schedules.append(('primary',range(config.sample_plan.tasks_per_replicate),phases['test_phase']))
    checks=scientific['update_check_schedule']
    if outer in checks['outer_replications']:
        schedules.append(('update_check',range(checks['tasks_per_replication']),phases['test_phase']+100))
    secondary=scientific.get('simulation_schedule',{'outer_replications':[]})
    if outer in secondary['outer_replications']:
        schedules.append(('secondary',range(secondary['tasks_per_replication']),phases['test_phase']+200))
    task_start=time.perf_counter_ns(); task_generation_ns=0
    for endpoint,indices,phase in schedules:
        for task in indices:
            generation_start=time.perf_counter_ns()
            context,truth=generate_task(config,TaskKey(root,phase,outer,task,0))
            task_generation_ns+=time.perf_counter_ns()-generation_start
            values,task_failures=evaluate_task(prepared,snapshots,context,truth,config)
            failures.update(task_failures)
            tasks.append({'endpoint':endpoint,'outer_replication':outer,'task_id':task,'task_key':context.task_key,'values':values})
            if failures: break
        if failures: break
    result.update(outer_replication=outer,risks=risks,tasks=tasks,failures=failures,
        frozen_state_hashes=snapshots,preparation_ns=preparation_ns,evaluation_ns=evaluation_ns,task_generation_ns=task_generation_ns,task_ns=time.perf_counter_ns()-task_start,
        truth_second_moment_hash=content_hash(population.second_moment),truth_noise_hash=content_hash(population.sigma2),
        status='invalid' if failures else 'completed')
    return result,assets


def evaluate_task(prepared, snapshots, context, truth, config):
    """Evaluate one shared observation, retaining every method and failure."""
    import time
    from .records import content_hash
    from .evaluation import coefficient_loss
    values={}; failures={}
    for key,(method,state) in prepared.items():
        try:
            if content_hash(state)!=snapshots[key]: raise ValueError('initializer changed between tasks')
            adapted,diagnostics=adapt_fresh(method,state,context)
            prediction_start=time.perf_counter_ns(); method.predict(adapted,context.X)
            diagnostics['prediction_ns']=time.perf_counter_ns()-prediction_start
            loss=coefficient_loss(adapted.coefficient,truth)
            direct_loss=None
            if state.metric is not None:
                from .methods import make_method
                direct=make_method('B05:'+method.method_id,{'metric':state.metric})
                direct_state=direct.prepare(state.prior,config.budget,config.preparation)
                control,_=adapt_fresh(direct,direct_state,context)
                direct_loss=coefficient_loss(control.coefficient,truth)
                diagnostics['duplicate_coefficient_error']=float(np.linalg.norm(control.coefficient-adapted.coefficient))
            values[key]={'status':'completed','loss':loss,'direct_loss':direct_loss,'diagnostics':diagnostics}
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            failures[key]=str(exc); values[key]={'status':'failed','loss':None,'failure_reason':str(exc)}
    return values,failures
