"""Immutable fixed-schedule comparisons, atomic pairing and raw-record analysis."""
import json
import time
from pathlib import Path
import numpy as np
from .records import JsonRecord, GateDecision, content_hash, to_json_record
from .runner import RunStore
from .protocol import config_from_record, validate_frozen_protocol, checkout_identity
from .execution import execute_replication
from .analysis import paired_interval, classify_gate


def run_record(manifest, *, phase='confirmation', dirty=None):
    s=manifest['scientific']; c=config_from_record(s['config']); v=s['versions']
    pair={'task_metrics':{'unit_fields':['outer_replication','task_id'],'method_ids':['baseline','joint']}}
    if c.sample_plan.analysis_mode=='conditional_risk_over_preparations':
        pair['preparation_risks']={'unit_fields':['outer_replication'],'method_ids':['baseline','joint']}
    return {'schema_version':1,'run_id':('confirmation-' if phase=='confirmation' else 'pilot-')+manifest['protocol_sha256'][:20],
        'phase':phase,'git_commit':v['git_commit'],'dirty_diff_sha256':dirty,
        'scientific_config_sha256':manifest['scientific_sha256'],'operational_config_sha256':manifest['operational_sha256'],
        'analysis_mode':c.sample_plan.analysis_mode,'protocol_sha256':manifest['protocol_sha256'],
        'dependency_lock_sha256':v['dependency_lock_sha256'],'hardware_record':'hardware.json',
        'data_manifest_sha256':content_hash(s['assets']['population']),'prior_asset_sha256':content_hash(s['assets']),
        'information_access':c.access['prior'],'timing_boundaries':{'preparation':'source generation, fitting and validation before evaluation',
        'tasks':'generation, fresh adaptation, coefficient loss and literal/direct diagnostics'},'status':'running',
        'thread_policy':manifest['operational']['thread_policy'],'blas_policy':manifest['operational']['blas_policy'],
        'pairing_spec':pair,'input_manifest_content_sha256':content_hash(manifest)}


def _live_identity(manifest, manifest_path=None):
    validate_frozen_protocol(manifest)
    s=manifest['scientific']
    actual,dirty=checkout_identity(s['selectors'],s['config']['numerics'].get('fitting_candidates',{}))
    if actual!=s['versions']:
        raise ValueError('protected execution requires the exact frozen code, runtime and dependency identity')
    if dirty:
        import subprocess
        root=Path(__file__).resolve().parents[2]
        diff=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root)
        untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=root).split(b'\0')
        paths=[(root/path.decode()).resolve() for path in untracked if path]
        allowed=Path(manifest_path).resolve() if manifest_path is not None else None
        if diff or not paths or any(path!=allowed for path in paths):
            raise ValueError('uncommitted changes beyond the explicit frozen manifest')
        if json.loads(allowed.read_text())!=manifest:
            raise ValueError('untracked frozen manifest content changed')


def _rows(result, manifest, run_id):
    s=manifest['scientific']; c=config_from_record(s['config']); p=c.sample_plan
    reason='; '.join(result['failures'].values()) or None
    rows=[{'kind':'execution','outer_replication':result['outer_replication'],'result':result,
           'status':result['status'],'failure_reason':reason}]
    if p.analysis_mode=='conditional_risk_over_preparations':
        valid=not reason and all(role in result['risks'] for role in ('baseline','joint'))
        difference=result['risks']['baseline']['conditional_risk']-result['risks']['joint']['conditional_risk'] if valid else None
        for role in ('baseline','joint'):
            selected=result['selected'].get(role,{})
            risk=result['risks'].get(role,{})
            rows.append({'kind':'preparation_risks','run_id':run_id,'outer_replication':result['outer_replication'],
                'source_key':result['source_key'],'method_id':role,'selection_procedure_hash':s['versions']['selection_sha256'],
                'prior_asset_hash':selected.get('asset_hash'),'metric_hash':selected.get('metric_hash'),
                'analysis_mode':p.analysis_mode,'source_training_tasks':p.source_training_tasks,
                'source_validation_tasks':p.source_validation_tasks,'total_source_labels':2*(p.source_training_tasks+p.source_validation_tasks)*p.source_block_size,
                'conditional_risk':risk.get('conditional_risk'),'paired_contrast_id':'baseline_minus_joint',
                'paired_risk_difference':difference,'risk_evaluator_version':s['versions']['evaluation_sha256'],
                'true_S_evaluator_hash':result['truth_second_moment_hash'],'noise_truth_evaluator_hash':result['truth_noise_hash'],
                'numerical_error_assessment':risk.get('numerical_error_assessment'),
                'status':'completed' if valid else 'invalid','failure_reason':reason or (None if valid else 'incomplete risk pair')})
    for task in result['tasks']:
        # Distinct task IDs and RNG phases keep diagnostics outside primary inference.
        offset={'primary':0,'update_check':10**9,'secondary':2*10**9}[task['endpoint']]
        for role in ('baseline','joint'):
            value=task['values'].get(role,{'status':'failed','failure_reason':'missing paired procedure'})
            diagnostics=value.get('diagnostics',{})
            rows.append({'kind':'task_metrics','run_id':run_id,'outer_replication':result['outer_replication'],
                'task_id':task['task_id']+offset,'task_key':task['task_key'],'method_id':role,
                'information_access':c.access['prior'],'n':c.preparation.target_context_size,'d':c.preparation.input_dimension,
                'depth':c.budget.depth,'norm_cap':c.budget.initialization_norm_cap,'parameter_cap':c.budget.trainable_parameter_cap,
                'eta':c.budget.learning_rate,'branch':None,'loss_parameter_mse':value.get('loss'),
                'loss_noiseless_prediction_mse':value.get('loss'),'adaptation_ns':diagnostics.get('adaptation_ns'),
                'prediction_ns':diagnostics.get('prediction_ns'),'status':value['status'],'failure_reason':value.get('failure_reason'),
                'endpoint':task['endpoint'],'direct_loss':value.get('direct_loss'),'numerical_error_assessment':diagnostics,
                'null_reasons':{'branch':'selected procedure can choose either branch'}})
    return rows


def evaluate_replication(replication_id: int, manifest: JsonRecord, store: RunStore) -> JsonRecord:
    """One atomic complete replication, including failed pairs and every candidate."""
    s=manifest['scientific']; c=config_from_record(s['config']); keys=manifest['protected_keys']
    result,assets=execute_replication(c,s,keys['root_seed'],replication_id,keys)
    run=json.loads((store.path/'run.json').read_text())
    store.commit_block(f'replication-{replication_id}',_rows(result,manifest,run['run_id']),assets)
    return result


def _validate_schedule(results, scientific):
    c=config_from_record(scientific['config']); p=c.sample_plan
    count=p.outer_replicates or 1
    if len(results)!=count or {r['outer_replication'] for r in results}!=set(range(count)):
        raise ValueError('incomplete or duplicated outer schedule')
    seen=set()
    for result in results:
        keys=result['source_keys']+[t['task_key'] for t in result['tasks']]
        if len(keys)!=len(set(keys)) or seen.intersection(keys): raise ValueError('reused source or task observations')
        seen.update(keys)
        if result['status']!='completed' or result['failures']: raise ValueError('invalid replication; failures cannot be averaged away')
        outer=result['outer_replication']; expected=set()
        if p.analysis_mode in ('fixed_prior_task_simulation','nested_task_simulation'):
            expected.update(('primary',i) for i in range(p.tasks_per_replicate))
        for endpoint,key in (('update_check','update_check_schedule'),('secondary','simulation_schedule')):
            schedule=scientific.get(key,{'outer_replications':[]})
            if outer in schedule['outer_replications']:
                expected.update((endpoint,i) for i in range(schedule['tasks_per_replication']))
        actual=[(t['endpoint'],t['task_id']) for t in result['tasks']]
        if len(actual)!=len(expected) or set(actual)!=expected: raise ValueError('task schedule differs from freeze')
        if p.source_training_tasks is not None and len(result['source_keys'])!=p.source_training_tasks+p.source_validation_tasks:
            raise ValueError('source schedule differs from freeze')


def analyze_results(results, scientific, seed):
    _validate_schedule(results,scientific)
    results=sorted(results,key=lambda r:r['outer_replication'])
    mode=scientific['config']['sample_plan']['analysis_mode']
    values=[]
    for result in results:
        if mode in ('conditional_risk_over_preparations','deterministic_frontier'):
            values.append(result['risks']['baseline']['conditional_risk']-result['risks']['joint']['conditional_risk'])
        else:
            differences=[t['values']['baseline']['loss']-t['values']['joint']['loss'] for t in result['tasks'] if t['endpoint']=='primary']
            values.extend(differences if mode=='fixed_prior_task_simulation' else [float(np.mean(differences))])
    inference=scientific['inference']
    if mode=='deterministic_frontier':
        analysis={'mean':values[0],'n':None,'interval':None,'coverage':None,'status':'deterministic',
                  'numerical_level':'float64_diagnostic','procedure':'inapplicable'}
    else: analysis=paired_interval(values,inference['alpha'],inference['procedure'],seed)
    analysis.update(analysis_mode=mode,independent_unit={'conditional_risk_over_preparations':'source_preparation',
        'nested_task_simulation':'source_preparation','fixed_prior_task_simulation':'task','deterministic_frontier':None}[mode],
        completed_outer_replications=len(results),primary_task_count=sum(t['endpoint']=='primary' for r in results for t in r['tasks']),
        update_check_count=sum(t['endpoint']=='update_check' for r in results for t in r['tasks']),
        secondary_task_count=sum(t['endpoint']=='secondary' for r in results for t in r['tasks']),
        differences=values)
    baseline=scientific['selectors']['baseline']; joint=scientific['selectors']['joint']
    if len(baseline)==len(joint)==1 and (baseline[0]=='B05:'+joint[0] or joint[0]=='B05:'+baseline[0]):
        analysis.update(interval=None,coverage=None,status='arithmetic_equality_diagnostic',
            comparison_kind='network_direct_equality_control',
            max_absolute_paired_difference=max(abs(v) for v in values),
            qualification='algebraically identical estimators; recorded float64 discrepancies are not a superiority endpoint')
    secondary={}
    for mid in scientific['secondary_family']['contrasts']:
        differences=[]
        for result in results:
            paired=[t['values']['family:'+mid]['loss']-t['values']['joint']['loss'] for t in result['tasks'] if t['endpoint']=='secondary']
            if paired: differences.append(float(np.mean(paired)))
        secondary[mid]=paired_interval(differences,.05,'descriptive_only',seed)
        secondary[mid].update(contrast=mid+'_minus_joint',independent_unit='source_preparation',
            multiplicity='no hypothesis test or CI; Holm is inapplicable to descriptive values')
    analysis['secondary']=secondary
    return analysis


def analyze_run(run_path: Path, manifest: JsonRecord, *, manifest_path=None) -> JsonRecord:
    _live_identity(manifest, manifest_path)
    store=RunStore.open(Path(run_path),run_record(manifest))
    run=json.loads((store.path/'run.json').read_text())
    if run['status'] not in ('running','completed'): raise ValueError('run stopped before a valid complete schedule')
    results=[r['result'] for r in store.read_records('execution')]
    return analyze_results(results,manifest['scientific'],manifest['protected_keys']['root_seed'])



def _finish(store, evidence, status):
    from datetime import datetime, timezone
    decision=classify_gate({**evidence,'run_path':str(store.path)})
    if 'decision' not in store.committed_keys():
        store.commit_block('decision',[{'kind':'decisions','timestamp':datetime.now(timezone.utc).isoformat(),
            'work_package':'WP07','observation':decision.code,'evidence_paths':['blocks'],
            'evidence_hashes':[content_hash(decision.evidence)],'decision':decision.code,
            'rationale':decision.rationale,'affected_claim_ids':['frozen_scoped_comparison'],
            'next_action':decision.next_action,'methodological_amendment_required':False,
            'fresh_confirmation_required':decision.code!='SUPPORTED_IN_SCOPE',
            'gate_evidence':to_json_record(decision.evidence),'terminal_status':status}],{})
    store.finalize(status)
    (store.path/'gate_decision.json').write_text(json.dumps(to_json_record(decision),indent=2,allow_nan=False)+'\n')
    return decision

def run_confirmation(manifest: JsonRecord, output: Path, *, manifest_path=None) -> GateDecision:
    manifest=json.loads(json.dumps(to_json_record(manifest),allow_nan=False))
    _live_identity(manifest, manifest_path)  # Before generating any observations or creating a run.
    run=run_record(manifest); path=Path(output)/run['run_id']
    store=RunStore.open(path,run) if path.exists() else RunStore.create(Path(output),run)
    (path/'hardware.json').write_text(json.dumps(manifest['evidence']['runtime_identity'],indent=2)+'\n')
    existing=json.loads((path/'run.json').read_text())
    decisions=store.read_records('decisions')
    if decisions:
        row=decisions[-1]
        store.finalize(row['terminal_status'])
        return classify_gate({**row['gate_evidence'],'code':row['decision']})
    if existing['status']=='completed': return classify_gate({**analyze_run(path,manifest,manifest_path=manifest_path),'run_path':str(path)})
    if existing['status']!='running':
        return classify_gate({'status':'invalid','run_path':str(path),'reason':'terminal failed or resource-limited run is sealed'})
    from .pilot import _available_memory, _rss_bytes
    count=manifest['scientific']['config']['sample_plan']['outer_replicates'] or 1
    memory=_available_memory()*(1-manifest['operational']['reserve_memory_fraction'])
    max_seconds=manifest['evidence']['resources']['max_seconds']; started=time.perf_counter()
    try:
        committed=store.committed_keys()
        for outer in range(count):
            if f'replication-{outer}' in committed: continue
            if _rss_bytes()>memory or time.perf_counter()-started>max_seconds:
                return _finish(store,{'code':'LOCAL_RESOURCE_LIMIT','reason':'frozen validity/resource stop; no sample extension'},'incomplete')
            result=evaluate_replication(outer,manifest,store)
            if result['failures']:
                return _finish(store,{'status':'invalid','failures':result['failures']},'invalid')
        analysis=analyze_run(path,manifest,manifest_path=manifest_path)
        (path/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
        return _finish(store,analysis,'completed')
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        return _finish(store,{'status':'invalid','reason':str(exc)},'invalid')
    # Process interruption leaves running metadata and only committed pairs;
    # replay verifies every block and recomputes only uncommitted replications.
