"""Exploratory precision planning; no protected observations are consumed here."""
from fractions import Fraction
from pathlib import Path
from .records import JsonRecord, RunConfig
import math
from statistics import NormalDist
import numpy as np
from scipy.stats import chi2

INITIAL_HALF_WIDTH = Fraction(1266883, 39703750) / 4


def plan_sample_sizes(pilot: JsonRecord, target_half_width: float, alpha: float) -> JsonRecord:
    """Plan from independent pilot variability, including an explicitly approximate variance-uncertainty allowance.

    Returns an explicit extension decision when variance cannot be estimated. The
    estimated precision is a planning approximation, not a significance threshold.
    """
    h, alpha = float(target_half_width), float(alpha)
    if not math.isfinite(h) or h <= 0 or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('invalid precision or alpha')
    mode = pilot.get('analysis_mode')
    z = NormalDist().inv_cdf(1-alpha/2)
    result = {'analysis_mode': mode, 'target_half_width': h, 'alpha': alpha,
              'coverage': 'approximate', 'decision': 'extend_pilot',
              'outer_replicates': None, 'tasks_per_replicate': None}
    if mode == 'deterministic_frontier':
        result.update(decision='inapplicable', coverage=None, reason='exact model quantity; no sampling allocation')
        return result
    if mode == 'fixed_prior_task_simulation':
        surrogate={**pilot, 'analysis_mode':'conditional_risk_over_preparations', 'source_training_tasks':1, 'source_validation_tasks':1, 'source_block_size':1}
        allocation=plan_sample_sizes(surrogate,h,alpha)
        result.update({k:v for k,v in allocation.items() if k not in ('analysis_mode','outer_replicates','tasks_per_replicate','source_training_tasks','source_validation_tasks','source_block_size')})
        result['tasks_per_replicate']=allocation['outer_replicates']
        return result
    if mode == 'conditional_risk_over_preparations':
        for name in ('source_training_tasks','source_validation_tasks','source_block_size'):
            value = pilot.get(name)
            if type(value) is not int or value <= 0: raise ValueError('missing source counts')
            result[name] = value
        blocks = pilot.get('blocks')
        if not isinstance(blocks, list) or len(blocks) < 2:
            result['reason'] = 'at least two independent pilot blocks required'; return result
        values = []
        variances = []
        for block in blocks:
            data = np.asarray(block.get('differences', []), dtype=float)
            if data.ndim != 1 or len(data) < 3 or not np.all(np.isfinite(data)):
                result['reason'] = 'each block needs at least three finite differences'; return result
            values.extend(data.tolist()); variances.append(float(np.var(data, ddof=1)))
        result['block_variances'] = variances
        if min(variances) <= 0 or max(variances)/min(variances) > 4:
            result['reason'] = 'block variances are zero or unstable'; return result
        n = len(values); pooled = float(np.var(values, ddof=1))
        upper = (n-1)*pooled / float(chi2.ppf(.05, n-1))
        result.update({'variance_estimate': pooled, 'variance_upper_planning': upper, 'variance_allowance_model':'chi-squared normal-variance approximation; not a distribution-free bound',
                       'outer_replicates': max(30, math.ceil(z*z*upper/(h*h))),
                       'decision': 'provisional', 'reason': 'variance ratio heuristic passed; independence, tails and calibration still require review'})
    elif mode == 'nested_task_simulation':
        try: between, within, a, b = (float(pilot[k]) for k in ('variance_between','variance_within','preparation_cost','task_cost'))
        except (KeyError, TypeError, ValueError):
            result['reason'] = 'measured variance components and costs required'; return result
        if not pilot.get('variance_stable') or not all(math.isfinite(v) and v > 0 for v in (between,within,a,b)):
            result['reason'] = 'positive stable variance components and measured costs required'; return result
        n = max(1, round(math.sqrt(within*a/(between*b))))
        result.update({'tasks_per_replicate': n, 'outer_replicates': max(2, math.ceil(z*z*(between+within/n)/(h*h))), 'decision':'provisional', 'reason':'nested allocation requires independent variance and cost verification'})
    else:
        raise ValueError('unsupported pilot analysis mode')
    return result


# Pilot policy is deliberately explicit. These are diagnostic planning criteria,
# not guarantees of coverage and never a minimum favorable effect.
MIN_BLOCK_REPLICATES = 30


def _available_memory():
    import os
    import platform
    import re
    import subprocess
    from pathlib import Path
    if platform.system()=='Darwin':
        raw=subprocess.check_output(['vm_stat'],text=True)
        page=int(re.search(r'page size of (\d+) bytes',raw).group(1))
        counts={k:int(v) for k,v in re.findall(r'([^\n:]+):\s+(\d+)\.',raw)}
        return page*sum(counts.get(k,0) for k in ('Pages free','Pages inactive','Pages speculative'))
    if Path('/proc/meminfo').exists():
        raw=Path('/proc/meminfo').read_text()
        return int(re.search(r'MemAvailable:\s+(\d+)',raw).group(1))*1024
    return int(os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'))


def _rss_bytes():
    import resource
    import sys
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)


def _write_json(path, value):
    import json
    from .records import to_json_record
    path.write_text(json.dumps(to_json_record(value),sort_keys=True,indent=2,allow_nan=False)+'\n')


def _restore_candidate(row):
    from .records import PriorAsset
    from .methods import make_method
    prior=dict(row['asset'])
    for key in ('covariance','basis','eigenvalues'):
        if prior[key] is not None: prior[key]=np.asarray(prior[key],dtype=float)
    return make_method(row['method_id'],row['options']),PriorAsset(**prior)


def _preparation_replication(config, root, outer, phase, methods):
    """Trusted driver assigns source authorization before generating observations."""
    import time
    from .records import TaskKey, content_hash, to_json_record
    from .tasks import generate_source_task, population_for
    from .selection import public_method_config, select_method
    from .methods import make_method
    from .evaluation import conditional_risk
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
    selection=select_method([make_method(mid,{}) for mid in order],train,validation,{**public,'selection_class':'mechanism_all'})
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
    adaptation_ns=0; prediction_ns=0; check_generation_ns=0
    for role,allowed in (('baseline',BASELINE),('joint',JOINT)):
        candidates=[row for row in ledger if row.get('status')=='completed' and row['method_id'] in allowed and row.get('refinement_iteration',0)==final]
        if not candidates:
            failures[role]='no completed candidate in final refinement'; continue
        row=min(candidates,key=lambda row:(row['validation_loss'],row['method_id'],row['logical_candidate_id']))
        try:
            method,prior=_restore_candidate(row)
            prepared=method.prepare(prior,config.budget,config.preparation)
            risks[role]=conditional_risk(prepared.metric,truth)
            # Independent dense formula diagnoses arithmetic sensitivity, not a certificate.
            from .metrics import dense_metric
            G=dense_metric(prepared.metric); S=truth.second_moment; n=truth.target_context_size
            dense=float(1-2*np.trace(G@S)+(n+1)/n*np.trace(G@G@S)+(1+truth.sigma2)/n*np.trace(G@G))
            # Fixed update diagnostics are separate from the analytic endpoint;
            # their losses never enter preparation-risk inference or allocation.
            from .tasks import generate_task
            from .metrics import apply_metric
            check_start=time.perf_counter_ns()
            context,check_truth=generate_task(config,TaskKey(root,phase+500,outer,0,0))
            check_generation_ns+=time.perf_counter_ns()-check_start
            adapt_start=time.perf_counter_ns(); adapted=method.adapt(prepared,context)
            adaptation_ns+=time.perf_counter_ns()-adapt_start
            predict_start=time.perf_counter_ns(); method.predict(adapted,context.X)
            prediction_ns+=time.perf_counter_ns()-predict_start
            direct=apply_metric(prepared.metric,context.X.T@context.y/len(context.y))
            direct_error=float(np.linalg.norm(adapted.coefficient-direct))
            if direct_error>1e-9*max(1.,float(np.linalg.norm(direct))): raise ValueError('literal update/direct-map diagnostic failed')
            selected[role]={'candidate_id':row['candidate_id'],'method_id':row['method_id'],'asset_hash':prior.asset_hash,
                'metric_hash':content_hash(prepared.metric),'validation_loss':row['validation_loss'],
                'validation_task_losses':row['validation_task_losses'],'prepared_state':to_json_record(prepared),
                'numerical_error_assessment':{'dense_compact_absolute_difference':abs(dense-risks[role]),'literal_direct_step_error':direct_error,'level':'float64_diagnostic'}}
            assets[role+'_covariance']=prior.covariance; assets[role+'_basis']=prepared.metric.basis
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
        'diagnostic_adaptation_ns':adaptation_ns,'diagnostic_prediction_ns':prediction_ns,'diagnostic_generation_ns':check_generation_ns,
        'generation_ns':generation_ns,'preparation_ns':preparation_ns,'method_order':order,
        'truth_second_moment_hash':content_hash(truth.second_moment),'truth_noise_hash':content_hash(truth.sigma2),
        'peak_rss_bytes':_rss_bytes()},assets


def calibration_diagnostics(blocks, procedure, resamples, seed, planned_count=None, memory_budget_bytes=None, deadline=None):
    """Centered empirical reference simulation: a diagnostic, not true coverage.

    Each block supplies an independent empirical reference. Resampling is at the
    preparation level. Finite moments follow from constrained maps, not tail plots.
    """
    from scipy import stats
    import time
    result={'status':'insufficient','procedure':procedure,'moment_basis':'bounded_constrained_fixed_maps',
            'replicates':resamples,'sample_size':planned_count,'coverage':None,'reference':'centered_empirical_pilot',
            'qualification':'empirical-reference coverage diagnostic; population coverage remains approximate'}
    tails=[]
    for block in blocks:
        x=np.asarray(block['differences'],dtype=float)
        if len(x):
            x=x-x.mean()
            tails.append({'max_absolute_centered_difference':float(np.max(np.abs(x))),
                'excess_kurtosis':float(stats.kurtosis(x)) if np.var(x)>0 else None,
                'quantiles':np.quantile(x,[.01,.05,.5,.95,.99]).tolist()})
        else: tails.append({'status':'no complete paired observations'})
    result['tails']=tails
    if len(blocks)<2 or any(len(block['differences'])<MIN_BLOCK_REPLICATES for block in blocks) or type(planned_count)!=int or planned_count<30: return result
    if procedure not in ('student','outer_bootstrap'): raise ValueError('unsupported calibration procedure')
    memory_budget_bytes=memory_budget_bytes if memory_budget_bytes is not None else _available_memory()//2
    if _rss_bytes()+8*planned_count*4>memory_budget_bytes:
        result['status']='resource_limit'; return result
    rng=np.random.default_rng(seed); cover=[]
    for block in blocks:
        x=np.asarray(block['differences']); x=x-x.mean(); n=planned_count
        for _ in range(resamples):
            if deadline is not None and time.perf_counter()>deadline:
                result['status']='resource_limit'; return result
            sample=rng.choice(x,size=n,replace=True)
            if procedure=='student':
                half=float(stats.t.ppf(.975,n-1)*sample.std(ddof=1)/np.sqrt(n)); lo,hi=sample.mean()-half,sample.mean()+half
            else:
                means=np.array([np.mean(rng.choice(sample,size=n,replace=True)) for _ in range(500)])
                lo,hi=np.quantile(means,[.025,.975])
            cover.append(bool(lo<=0<=hi))
    coverage=float(np.mean(cover)); se=float(np.sqrt(coverage*(1-coverage)/len(cover)))
    result.update(coverage=coverage,coverage_standard_error=se,tails=tails,
                  status='adequate' if resamples>=500 and coverage>=.90 else 'extend_pilot')
    return result


def _draft_for_pilot(config, result, versions):
    from .records import to_json_record, content_hash
    from .protocol import BASELINE, JOINT
    from .tasks import RNG_SPEC
    record=to_json_record(config)
    # Exploration settings and seed are excluded from the protected procedure.
    record['numerics'].pop('pilot',None)
    record['keys']={}; record['confirmation']={}
    record['sample_plan'].update({k:result['source_count_decision'].get(k) for k in ('source_training_tasks','source_validation_tasks','source_block_size')})
    record['sample_plan']['outer_replicates']=result['precision_diagnostics'].get('planned_outer_replicates')
    record['sample_plan']['tasks_per_replicate']=None
    return {'schema_version':1,'scientific':{'config':record,'scenario_id':'primary_richer_k_3_5','amendment':None,
        'selectors':{'baseline':list(BASELINE),'joint':list(JOINT)},
        'endpoint':{'metric':'squared_coefficient_error','contrast':'baseline_minus_joint'},
        'inference':{'procedure':'student','alpha':.05,'coverage':'approximate','moment_basis':'bounded_constrained_fixed_maps'},
        'secondary_family':{'contrasts':[],'adjustment':'holm'},
        'update_check_schedule':{'outer_replications':[0],'tasks_per_replication':2},
        'precision':{'target_half_width':float(INITIAL_HALF_WIDTH)},'generator':RNG_SPEC,
        'versions':versions,'stop_rule':'fixed_counts_invariants_only',
        'assets':{'prior':'refit_each_replication','preprocessing':'identity'}},
        'operational':{'batch_size':1,'reserve_memory_fraction':.5,'thread_policy':result['runtime_identity']['thread_environment'],
                       'blas_policy':content_hash(result['runtime_identity']['builds'])}}



def choose_source_level(curves, relative_error_target, required_replications):
    """Select accuracy/stability-qualified T at measured cost; never read effects."""
    accurate=[row for row in curves if row['replications']==required_replications
              and row['relative_covariance_error_upper_diagnostic'] is not None
              and row['relative_covariance_error_upper_diagnostic']<=relative_error_target
              and row['validation_stability']=='stable']
    poor=any(row['relative_covariance_error_upper_diagnostic'] is not None
             and row['relative_covariance_error_upper_diagnostic']>relative_error_target for row in curves)
    chosen=min(accurate,key=lambda row:row['preparation_seconds']) if accurate else (curves[-1] if curves else None)
    return chosen, 'resolved' if accurate and poor and len(curves)>=2 else 'extend_pilot'


def _validation_stability(blocks):
    values=[]
    for block in blocks:
        observations=[float(np.mean([selected['validation_loss'] for selected in r['selected'].values()])) for r in block if len(r['selected'])==2]
        if len(observations)<3: return 'unverified',None
        values.append((float(np.mean(observations)),float(np.var(observations,ddof=1)/len(observations))))
    diagnostics=[]
    for i,(mean,var) in enumerate(values):
        for other,other_var in values[i+1:]:
            denominator=math.sqrt(var+other_var)
            if denominator==0:
                if mean!=other: return 'unstable',None
                diagnostics.append(0.)
            else: diagnostics.append(abs(mean-other)/denominator)
    maximum=max(diagnostics)
    return ('stable' if maximum<=3 else 'unstable'),maximum

def run_pilot(config: RunConfig, output: Path) -> JsonRecord:
    """Run a prespecified exploratory source-learning schedule with real replications.

    W0 conditional-risk calibration is primary. Representative W1/W2 blocks only
    assess local scaling; neither their effects nor pilot effects choose source T.
    Resource stops retain all committed pairs and explicitly fail the G6 gate.
    """
    import json
    import platform
    import time
    from datetime import datetime, timezone
    from dataclasses import replace
    from pathlib import Path
    from .records import content_hash, to_json_record
    from .specs import validate_config
    from .protocol import BASELINE, JOINT, checkout_identity, scientific_contract, runtime_identity, project_runtime
    from .runner import RunStore
    validate_config(config,False)
    sample=config.sample_plan
    if sample.analysis_mode!='conditional_risk_over_preparations': raise ValueError('run_pilot currently calibrates the primary fixed-map endpoint; other endpoint allocations require separate evidence')
    if sample.tasks_per_replicate is not None: raise ValueError('analytic primary has no inner allocation')
    if sample.source_block_size!=config.preparation.target_context_size: raise ValueError('source block size must match the declared adaptation context')
    if set(config.methods)!=set(BASELINE)|set(JOINT): raise ValueError('pilot requires complete matched mechanism candidate sets')
    if config.access.get('prior')!='estimated' or config.access.get('selection')!='learned_only': raise ValueError('learned-only source preparation required')
    root=config.keys.get('exploration')
    if type(root)!=int or root<0 or set(config.keys)!={'exploration'}: raise ValueError('only an exploratory root is permitted')
    settings=to_json_record(config.numerics.get('pilot',{}))
    blocks=settings.get('blocks',2); reps=settings.get('replicates_per_block',4)
    levels=settings.get('source_levels',[8,16,32]); validations=settings.get('validation_levels',levels)
    workloads=settings.get('workloads',['W0','W1','W2'])
    resamples=settings.get('calibration_resamples',1000)
    threshold=settings.get('source_relative_error_target',.5); max_seconds=settings.get('max_seconds',1800.)
    if type(blocks)!=int or blocks<2 or type(reps)!=int or reps<3: raise ValueError('two independent blocks with at least three replications required')
    if len(levels)<2 or len(validations)!=len(levels) or any(type(v)!=int or v<2 for v in levels+validations) or any(b<2*a for a,b in zip(levels,levels[1:])) or any(b<a for a,b in zip(validations,validations[1:])): raise ValueError('geometric source levels and nondecreasing validation levels required')
    if not workloads or workloads[0]!='W0' or len(set(workloads))!=len(workloads) or set(workloads)-{'W0','W1','W2'}: raise ValueError('calibrate W0 before representative W1/W2')
    if type(resamples)!=int or resamples<100 or not math.isfinite(threshold) or threshold<=0 or not math.isfinite(max_seconds) or max_seconds<=0: raise ValueError('invalid pilot policy')
    selectors={'baseline':list(BASELINE),'joint':list(JOINT)}
    runtime=runtime_identity()
    versions,dirty=checkout_identity(selectors,to_json_record(config.numerics.get('fitting_candidates',{})),runtime=runtime)
    available=_available_memory(); budget=available//2
    if available<=0: raise ValueError('available memory measurement required')
    started=time.perf_counter_ns()
    run_id='pilot-'+content_hash({'config':config,'versions':versions,'started_ns':started})[:20]
    run={'schema_version':1,'run_id':run_id,'phase':'exploration','git_commit':versions['git_commit'],'dirty_diff_sha256':dirty,
         'scientific_config_sha256':content_hash(config),'operational_config_sha256':content_hash({'memory_budget_bytes':budget,'batch_size':1}),
         'analysis_mode':sample.analysis_mode,'protocol_sha256':content_hash({'exploratory_schedule':settings,'config':config,'runtime_sha256':versions['runtime_sha256']}),
         'dependency_lock_sha256':versions['dependency_lock_sha256'],'hardware_record':'hardware.json',
         'data_manifest_sha256':None,'prior_asset_sha256':None,'information_access':'estimated',
         'timing_boundaries':{'generation':'observations only','preparation':'all candidate fitting and held-out validation','serialization':'candidate JSON and immutable block publication'},
         'status':'running','thread_policy':runtime['thread_environment'],'blas_policy':content_hash(runtime['builds']),
         'runtime_sha256':versions['runtime_sha256'],'pairing_spec':{'preparation_risks':{'unit_fields':['outer_replication'],'method_ids':['baseline','joint']}}}
    store=RunStore.create(Path(output),run)
    _write_json(store.path/'hardware.json',{'platform':platform.platform(),'available_memory_bytes':available,'memory_budget_bytes':budget,
        'peak_rss_unit':'bytes','peak_scope':'cumulative process high water; no per-method peak claim',
        'power_thermal_condition':'unmeasured','runtime_identity':runtime,'runtime_sha256':versions['runtime_sha256'],
        'thread_policy':runtime['thread_environment'],'blas_policy':content_hash(runtime['builds'])})
    all_results=[]; curves=[]; block_rows={}; failures=[]; stop=None; outer=0
    for level,(training,validation) in enumerate(zip(levels,validations)):
        local=replace(config,sample_plan=replace(sample,source_training_tasks=training,source_validation_tasks=validation))
        level_results=[]
        for block in range(blocks):
            block_rows[level,block]=[]
            for replication in range(reps):
                # Conservative work-array estimate plus process high water; ledger growth
                # is checked again after each replication. No partial paired population.
                projected=8*(2*(training+validation)*(local.preparation.input_dimension+1)*sample.source_block_size+32*local.preparation.input_dimension**2)
                if _rss_bytes()+projected>budget or (time.perf_counter_ns()-started)/1e9>max_seconds:
                    stop='local memory reserve or pilot time budget reached'; break
                result,assets=_preparation_replication(local,root,outer,10+2*level,config.methods)
                result.update(level=level,block_id=block,replication=replication,outer_replication=outer)
                complete=len(result['risks'])==2
                difference=result['risks']['baseline']-result['risks']['joint'] if complete else None
                result['difference']=difference
                rows=[]
                for role in ('baseline','joint'):
                    chosen=result['selected'].get(role,{})
                    rows.append({'kind':'preparation_risks','run_id':run_id,'outer_replication':outer,
                        'source_key':result['source_key'],'method_id':role,'selection_procedure_hash':versions['selection_sha256'],
                        'prior_asset_hash':chosen.get('asset_hash'),'metric_hash':chosen.get('metric_hash'),
                        'analysis_mode':sample.analysis_mode,'source_training_tasks':training,'source_validation_tasks':validation,
                        'total_source_labels':2*(training+validation)*sample.source_block_size,
                        'conditional_risk':result['risks'].get(role),'paired_contrast_id':'baseline_minus_joint',
                        'paired_risk_difference':difference,'risk_evaluator_version':versions['evaluation_sha256'],
                        'true_S_evaluator_hash':result['truth_second_moment_hash'],'noise_truth_evaluator_hash':result['truth_noise_hash'],
                        'numerical_error_assessment':chosen.get('numerical_error_assessment'),
                        'status':'completed' if role in result['risks'] else 'failed','failure_reason':result['failures'].get(role),
                        'null_reasons':{'paired_risk_difference':'paired procedure failed'}})
                serialization_start=time.perf_counter_ns()
                _write_json(store.path/f'preparation-{outer}.json',result)
                store.commit_block(f'preparation-{outer}',rows,assets)
                serialization_ns=time.perf_counter_ns()-serialization_start
                timing={'kind':'timing_blocks','method_order':result['method_order'],'cold_warm':'cold' if outer==0 else 'warm','block_size':1,
                    'generation_boundary':'source observations only','preparation_boundary':'all fitting and validation candidates',
                    'adaptation_boundary':'candidate validation plus separately measured fixed update check; no inner primary simulation',
                    'prediction_boundary':'candidate validation plus separately measured fixed diagnostic prediction','serialization_boundary':'candidate JSON and RunStore block',
                    'repetitions':1,'wall_time_ns':result['generation_ns']+result['preparation_ns']+serialization_ns+result['diagnostic_generation_ns']+result['diagnostic_adaptation_ns']+result['diagnostic_prediction_ns'],
                    'peak_rss':result['peak_rss_bytes'],'peak_rss_unit':'bytes','power_thermal_condition':'unmeasured',
                    'timing_stratum':'CPU float64 runtime '+versions['runtime_sha256'],'gpu_metadata':None,'null_reasons':{'gpu_metadata':'CPU only'},
                    'generation_ns':result['generation_ns'],'preparation_ns':result['preparation_ns'],'serialization_ns':serialization_ns,
                    'diagnostic_adaptation_ns':result['diagnostic_adaptation_ns'],'diagnostic_prediction_ns':result['diagnostic_prediction_ns'],'diagnostic_generation_ns':result['diagnostic_generation_ns']}
                store.commit_block(f'timing-{outer}',[timing],{})
                result['serialization_ns']=serialization_ns
                all_results.append(result); level_results.append(result); block_rows[level,block].append(result)
                failures.extend(result['failures'].values()); outer+=1
            if stop: break
        if level_results:
            errors=[r['relative_covariance_error'] for r in level_results]
            validation_variance=[float(np.var(row['validation_task_losses'],ddof=1)/validation) for r in level_results for row in r['selected'].values()]
            validation_status,validation_difference=_validation_stability([block_rows.get((level,b),[]) for b in range(blocks)])
            curves.append({'level':level,'validation_stability':validation_status,'validation_block_standardized_difference':validation_difference,'source_training_tasks':training,'source_validation_tasks':validation,
                'source_block_size':sample.source_block_size,'replications':len(level_results),
                'relative_covariance_error_mean':float(np.mean(errors)),
                'relative_covariance_error_upper_diagnostic':float(np.mean(errors)+1.96*np.std(errors,ddof=1)/np.sqrt(len(errors))) if len(errors)>1 else None,
                'eigenspace_projection_error_mean':float(np.mean([r['eigenspace_projection_error'] for r in level_results])),
                'validation_mean_variance':float(np.mean(validation_variance)) if validation_variance else None,
                'preparation_seconds':float(np.mean([r['preparation_ns']/1e9 for r in level_results])),
                'tuning_resolution':'resolved' if all(r['tuning_resolution']=='resolved' for r in level_results) else 'unresolved'})
        if stop: break
    # Choice deliberately reads only preparation error, validation precision and cost.
    chosen,source_status=choose_source_level(curves,threshold,blocks*reps)
    selected_level=chosen['level'] if chosen else None
    pilot_blocks=[{'block_id':b,'differences':[r['difference'] for r in block_rows.get((selected_level,b),[]) if r['difference'] is not None],
                   'source_keys':[k for r in block_rows.get((selected_level,b),[]) for k in r['source_keys']]} for b in range(blocks)]
    from .analysis import paired_interval
    observed=[value for block in pilot_blocks for value in block['differences']]
    uncertainty=paired_interval(observed,.05,'student',root) if len(observed)>=2 else {'interval':None,'n':len(observed)}
    uncertainty['status']='exploratory_only'
    source_decision={'status':source_status,
        'basis':'preparation_accuracy_and_cost','source_training_tasks':chosen['source_training_tasks'] if chosen else None,
        'source_validation_tasks':chosen['source_validation_tasks'] if chosen else None,'source_block_size':sample.source_block_size,
        'relative_error_target':threshold,'reason':'bracket poor/accurate estimation using geometric levels; no protected performance accessed'}
    planning={'analysis_mode':sample.analysis_mode,'blocks':pilot_blocks,**{k:source_decision[k] for k in ('source_training_tasks','source_validation_tasks','source_block_size')}}
    plan=plan_sample_sizes(planning,INITIAL_HALF_WIDTH,.05) if chosen else {'decision':'extend_pilot','outer_replicates':None}
    calibration=calibration_diagnostics(pilot_blocks,'student',resamples,root,planned_count=plan['outer_replicates'],memory_budget_bytes=budget,deadline=started/1e9+max_seconds)
    if calibration['status']=='resource_limit': stop='local calibration resource budget reached'
    stable=plan['decision']=='provisional' and calibration['status']=='adequate'
    precision={'status':'stable' if stable else 'extend_pilot','planned_outer_replicates':plan['outer_replicates'],
        'target_half_width':float(INITIAL_HALF_WIDTH),'alpha':.05,'coverage':'approximate','planning_details':plan}
    cost_values=[(r['generation_ns']+r['preparation_ns']+r['serialization_ns'])/1e9 for r in all_results if r['level']==selected_level]
    check_costs=[(r['diagnostic_generation_ns']+r['diagnostic_adaptation_ns']+r['diagnostic_prediction_ns'])/1e9 for r in all_results if r['level']==selected_level and len(r['selected'])==2]
    update_schedule={'outer_replications':[0],'tasks_per_replication':2}
    mean_cost=float(np.mean(cost_values)) if cost_values else None
    count=plan['outer_replicates']
    estimate=project_runtime({'replication_seconds':cost_values,'update_check_seconds':check_costs},count,update_schedule) if count is not None and len(cost_values)>=2 and len(check_costs)>=2 else None
    scaling=[]
    if not stop:
        for index,name in enumerate(workloads[1:]):
            if (time.perf_counter_ns()-started)/1e9>max_seconds: stop='pilot time budget reached before scaling calibration'; break
            spectrum=np.r_[np.full(8,1/8),np.zeros(56)] if name=='W1' else np.arange(1,65,dtype=float)**-1.5
            spectrum=spectrum/spectrum.sum()
            local=replace(config,workload={'spectrum':spectrum.tolist(),'sigma2':1.},
                preparation=replace(config.preparation,input_dimension=64,target_context_size=8),
                budget=replace(config.budget,trainable_parameter_cap=65*8),
                sample_plan=replace(sample,source_training_tasks=levels[0],source_validation_tasks=validations[0],source_block_size=8))
            if _rss_bytes()+8*128*64**2>budget: stop='memory reserve reached before scaling calibration'; break
            measured,assets=_preparation_replication(local,root,outer,70+2*index,config.methods)
            _write_json(store.path/f'calibration-{name}.json',measured)
            scaling.append({'workload':name,'preparation_seconds':measured['preparation_ns']/1e9,'generation_seconds':measured['generation_ns']/1e9,
                'peak_rss_bytes':measured['peak_rss_bytes'],'tuning_resolution':measured['tuning_resolution'],'failures':measured['failures']})
            outer+=1
    resource={'status':'feasible' if estimate is not None and estimate<=max_seconds and _rss_bytes()<=budget and not stop else 'unresolved',
        'preparation_seconds':chosen['preparation_seconds'] if chosen else None,'estimated_total_seconds':estimate,
        'replication_seconds':cost_values,'update_check_seconds':check_costs,
        'estimated_total_seconds_interval':[float(np.quantile(cost_values,q))*count+float(np.quantile(check_costs,q))*2 for q in (.1,.9)] if count and cost_values and check_costs else None,
        'estimate_qualification':'empirical 10–90% per-replication throughput range, not a runtime confidence interval',
        'peak_rss_bytes':_rss_bytes(),'available_memory_bytes':available,'memory_budget_bytes':budget,
        'rss_scope':'cumulative process high water, not method-isolated','max_seconds':max_seconds,
        'measured_total_seconds':(time.perf_counter_ns()-started)/1e9,'scaling':scaling}
    result={'schema_version':1,'phase':'exploration','status':'failed' if failures else ('incomplete' if stop else 'completed'),
        'run_path':str(store.path),'runtime_identity':runtime,'versions':versions,'dirty_diff_sha256':dirty,'excluded_roots':[root],
        'blocks':pilot_blocks,'uncertainty':uncertainty,'source_curve':curves,'source_count_decision':source_decision,
        'precision_diagnostics':precision,'calibration':calibration,'resources':resource,
        'tuning_resolution':'resolved' if all_results and all(r['tuning_resolution']=='resolved' for r in all_results) else 'unresolved',
        'failures':failures,'resource_stop':stop,'all_pilot_source_keys':[k for r in all_results for k in r['source_keys']],
        'planning_policy':{'minimum_independent_replications_per_block':MIN_BLOCK_REPLICATES,'variance_ratio_limit':4.,'source_relative_error_target':threshold,'coverage_diagnostic_minimum':.90}}
    draft=_draft_for_pilot(config,result,versions)
    result['scientific_contract_sha256']=scientific_contract(draft['scientific'])
    from .protocol import _validate_draft
    try:
        _validate_draft(draft,result)
        if dirty: raise ValueError('exploratory evidence was generated on an uncommitted checkout')
        if set(workloads)!={'W0','W1','W2'} or len(scaling)!=2: raise ValueError('representative W1/W2 resource calibration required')
        reasons=[]
    except ValueError as exc: reasons=[str(exc)]
    for condition,message in ((result['tuning_resolution']!='resolved','candidate tuning remains unresolved'),
                              (source_decision['status']!='resolved','source counts have not bracketed accurate stable preparation'),
                              (calibration['status']!='adequate','independent-block CI calibration is insufficient'),
                              (precision['status']!='stable','precision planning remains provisional'),
                              (resource['status']!='feasible','local resource feasibility remains unresolved')):
        if condition and message not in reasons: reasons.append(message)
    code='LOCAL_RESOURCE_LIMIT' if stop or (estimate is not None and estimate>max_seconds) else ('IMPLEMENTATION_INVALID' if failures else ('INSENSITIVE_OR_UNSUITABLE' if reasons else 'SUPPORTED_IN_SCOPE'))
    result['gate']={'code':code,'work_package':'WP06','gate':'G6','reasons':reasons,
        'next_action':'freeze the reviewed draft before generating protected observations' if not reasons else 'extend exploration or amend resources; retain precision target and use fresh confirmation only after a valid freeze'}
    _write_json(store.path/'pilot.json',result); _write_json(store.path/'power_precision.json',precision)
    _write_json(store.path/'protocol_draft.json',{'draft':draft,'evidence_path':'pilot.json','status':'ready' if not reasons else 'unresolved'})
    report=(f'# Exploratory pilot\n\nG6: {code}. {"; ".join(reasons)}\n\n'
        f'Completed {len(all_results)} independent preparation replications across {len(curves)} source levels. '
        f'Selected source counts: {source_decision}.\n\n'
        f'Precision target: {float(INITIAL_HALF_WIDTH):.12g}; alpha 0.05; inner primary allocation inapplicable. '
        f'Outer count estimate: {count}; coverage remains approximate.\n\n'
        f'Local resource estimate: {resource}.\n\n'
        'Every candidate and failure is retained in preparation JSON; paired risks and arrays are committed through RunStore. '
        'No protected keys or confirmation observations were generated.\n')
    (store.path/'pilot_report.md').write_text(report)
    store.commit_block('decision',[{'kind':'decisions','timestamp':datetime.now(timezone.utc).isoformat(),'work_package':'WP06',
        'observation':code,'evidence_paths':['pilot.json','power_precision.json'],'evidence_hashes':[content_hash(result),content_hash(precision)],
        'decision':code,'rationale':'; '.join(reasons) or 'pilot diagnostics and resource criteria met; freeze still required',
        'affected_claim_ids':['primary_learned_mechanism'],'next_action':result['gate']['next_action'],
        'methodological_amendment_required':False,'fresh_confirmation_required':True}],{})
    store.finalize(result['status'])
    return result
