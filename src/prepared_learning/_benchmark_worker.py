"""Private fresh-interpreter benchmark worker; replay request is ordinary JSON."""
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from .benchmark import (_conditions, _json_bytes, _timed, measure_operation,
                        normalize_peak_rss, storage_inventory)
from .methods import make_method
from .records import (AdaptedState, BudgetSpec, Context, PreparationSpec, PriorAsset,
                      RunConfig, SamplePlan, TaskKey, content_hash, to_json_record)


def restore_prior(record):
    record=dict(record)
    for key in ('covariance','basis','eigenvalues'):
        if record[key] is not None: record[key]=np.asarray(record[key],dtype=np.float64)
    return PriorAsset(**record)


def restore_method(record):
    options=dict(record['options'])
    if isinstance(options.get('metric'),dict):
        from .metrics import CompactMetric
        from .records import SupportMetadata
        metric=options['metric']
        options['metric']=CompactMetric(metric['floor'],np.asarray(metric['basis'],dtype=float),np.asarray(metric['increments'],dtype=float),SupportMetadata(**metric['support_metadata']))
    return make_method(record['method_id'],options)


def generated_config(manifest):
    d=int(manifest['dimension']); n=int(manifest['context_size']); seed=int(manifest['seed'])
    if d<1 or n<1: raise ValueError('positive dimension and context size required')
    config=RunConfig(1,{'spectrum':manifest.get('spectrum',[1/d]*d),'sigma2':manifest.get('noise_variance',.1),'population_seed':manifest.get('population_seed',seed)},
        BudgetSpec(int(manifest.get('depth',2)),float(manifest.get('norm_cap',1.)),int(manifest.get('parameter_cap',2*d*d+2*d+1)),float(manifest.get('learning_rate',1.))),
        PreparationSpec(n,d,manifest.get('input_law','gaussian_isotropic'),manifest.get('loss_normalization','squared_error_over_2n'),manifest.get('task_second_moment_trace',1.)),
        {'dtype':manifest.get('dtype','float64'),'fitting_candidates':manifest.get('fitting_candidates',{})},
        {'prior':'estimated','selection':'learned_only','noise_access':'estimated'},tuple(manifest['methods']),{'root_seed':seed},
        SamplePlan('nested_task_simulation',1,int(manifest.get('context_count',1)),int(manifest.get('source_training_tasks',4)),int(manifest.get('source_validation_tasks',4)),int(manifest.get('source_block_size',n))),{})
    from .specs import validate_config
    validate_config(config,False)
    if config.sample_plan.tasks_per_replicate < 1: raise ValueError('context_count must be positive')
    return config


def _runtime_payload(state):
    # All actual retained execution assets, including initialized zeros and prior.
    # Fit ledgers/solver diagnostics are incremental evidence, stored separately.
    prior=state.prior
    if prior is not None:
        prior={name:getattr(prior,name) for name in ('covariance','basis','eigenvalues','noise_variance','information_access','asset_hash')}
    return {'metric':state.metric,'initial_layers':state.initial_layers,'budget':state.budget,
            'preparation':state.preparation,'prior':prior,'options':state.options,'method_id':state.method_id}


def _compact_adapt(method,state,contexts,batch_size):
    from .metrics import apply_metric
    adapted=[]; workspace=0
    for start in range(0,len(contexts),batch_size):
        batch=contexts[start:start+batch_size]
        if state.metric is not None:
            # Explicit bounded gradient buffer; each task's arithmetic/order is
            # unchanged when memory pressure changes the number of buffered rows.
            gradients=np.stack([c.X.T@c.y/len(c.y) for c in batch])
            workspace=max(workspace,gradients.nbytes)
            for gradient in gradients:
                coefficient=apply_metric(state.metric,gradient)
                adapted.append(AdaptedState(coefficient,None,'completed',None))
            del gradients
        else:
            for context in batch: adapted.append(method.adapt(state,context))
    return adapted,workspace


def _raw_predict(layers,X):
    value=X.T
    for layer in layers: value=layer@value
    return value.reshape(-1)


def _provenance(selected):
    method,prior,ledger=selected
    return {'method_id':method.method_id,'options':to_json_record(method.options),'asset':to_json_record(prior)}



def _prepare_selected(selected,descriptor,budget,preparation):
    method,prior,ledger=selected; mid=descriptor['method_id']; target_state=None
    # Reuse the exact metric already fitted/validated during source preparation
    # for EVERY eligible method. B05 does not instantiate a canonical network.
    metric_record=ledger.get('prepared_metric')
    if metric_record is not None:
        method=make_method(method.method_id,{**method.options,'fitted_metric':metric_record})
        metric=method.options['fitted_metric']
    else:
        target_state=method.prepare(prior,budget,preparation); metric=target_state.metric
    target_method=method
    if mid.startswith('B05:'):
        supplied=restore_method(descriptor).options.get('metric')
        if supplied is not None and content_hash(supplied)!=content_hash(metric): raise ValueError('B05 metric differs from target')
        if metric is None: raise ValueError('B05 target has no fixed metric')
        method=make_method(mid,{'metric':metric}); state=method.prepare(prior,budget,preparation)
    else:
        state=target_state if target_state is not None else method.prepare(prior,budget,preparation)
        target_state=state
    return {'state':state,'target_state':target_state,'target_method':target_method,'method':method}

def operation(request,clock,upstream=None):
    generated=request['workflow']=='generated'; inputs=request['inputs']; settings=request['settings']
    descriptor=request['method']; mid=descriptor['method_id']; target=mid[4:] if mid.startswith('B05:') else mid
    target_descriptor=next((m for m in request['methods'] if m['method_id']==target),None)
    if target_descriptor is None: raise ValueError('B05 requires present target')
    generation_parts={}; active={}
    if generated:
        manifest=inputs['manifest']; config=generated_config(manifest); budget=config.budget; preparation=config.preparation
        from .tasks import generate_source_task, generate_task, population_for
        from .selection import public_method_config, select_method
        root=int(manifest['seed'])
        train_keys=[TaskKey(root,710,0,i,0) for i in range(config.sample_plan.source_training_tasks)]
        val_keys=[TaskKey(root,711,0,i,0) for i in range(config.sample_plan.source_validation_tasks)]
        test_keys=[TaskKey(root,712,0,i,0) for i in range(config.sample_plan.tasks_per_replicate)]
        public=public_method_config(config)
        public['source_access_manifest']={'training_keys':[content_hash({'key':key}) for key in train_keys],
                                          'validation_keys':[content_hash({'key':key}) for key in val_keys]}
        def generate():
            def sources(): return ([generate_source_task(config,key) for key in train_keys],[generate_source_task(config,key) for key in val_keys])
            (train,validation),generation_parts['source_generation_ns']=_timed(clock,sources)
            tasks,generation_parts['target_generation_ns']=_timed(clock,lambda:[generate_task(config,key) for key in test_keys])
            population,_=population_for(config,test_keys[0])
            return {'train':train,'validation':validation,'contexts':[p[0] for p in tasks],'truth':[p[1] for p in tasks],'population':population}
        def select(data):
            if target in ('B06','B07','B10:posterior') or manifest.get('prior_mode')=='oracle':
                from .priors import _asset
                prior=_asset(data['population'].second_moment,data['population'].sigma2,{'source':'explicit oracle benchmark arm'})
                payload={**to_json_record(prior),'information_access':'oracle'}; payload.pop('asset_hash')
                prior=replace(prior,information_access='oracle',asset_hash=content_hash(payload))
                method=restore_method(target_descriptor)
                return method,prior,{'selected_id':'oracle:'+prior.asset_hash,'candidate_records':[],
                                     'information_access':'oracle','source_access_manifest':public['source_access_manifest']}
            selection=select_method([restore_method(target_descriptor)],data['train'],data['validation'],public)
            if not selection.selected_id:
                error=ValueError('source selection failed')
                error.details=to_json_record(selection)
                raise error
            row=next(r for r in selection.candidate_records if r.get('candidate_id')==selection.selected_id)
            method=restore_method({'method_id':target,'options':to_json_record(row['options'])}); prior=restore_prior(row['asset'])
            return method,prior,{'selected_id':selection.selected_id,'candidate_records':to_json_record(selection.candidate_records),
                                'source_access_manifest':public['source_access_manifest'],'selected_options':to_json_record(row['options']),
                                'selected_asset_hash':prior.asset_hash,'prepared_metric':to_json_record(row['prepared_state']['metric']),'source_hash':row['source_hash'],'validation_hash':row['validation_hash']}
    else:
        budget=BudgetSpec(**inputs['budget']); preparation=PreparationSpec(**inputs['preparation'])
        contexts=[Context(np.asarray(c['X'],dtype=float),np.asarray(c['y'],dtype=float),c['task_key']) for c in inputs['contexts']]
        prior=restore_prior(inputs['prior'])
        def generate(): return {'contexts':contexts,'truth':None,'population':None}
        def select(data):
            if upstream is not None:
                value=upstream()
                if content_hash(value)!=content_hash(prior): raise ValueError('upstream preparation differs from supplied prior')
            return restore_method(target_descriptor),prior,{'selected_id':None,'candidate_records':[],'status':'supplied_asset_upstream_unavailable'}
    def prepare(selected):
        prepared=_prepare_selected(selected,descriptor,budget,preparation)
        active.update({k:v for k,v in prepared.items() if k!='state'})
        return prepared['state']
    def adapt(state,data):
        adapted,workspace=_compact_adapt(active['method'],state,data['contexts'],settings['batch_size'])
        active['batch_workspace_bytes']=workspace
        return adapted
    def predict(adapted,data): return [active['method'].predict(a,c.X) for a,c in zip(adapted,data['contexts'])]
    serialization_parts={}
    def serialize(state,adapted):
        prepared,serialization_parts['prepared_serialization_ns']=_timed(clock,lambda:_json_bytes(_runtime_payload(state)))
        coefficients,serialization_parts['adapted_serialization_ns']=_timed(clock,lambda:_json_bytes([a.coefficient for a in adapted]))
        return prepared+b'\n'+coefficients
    measured=measure_operation({'generation':generate,'selection':select,'preparation':prepare,'adaptation':adapt,
                                'prediction':predict,'serialization':serialize},clock=clock)
    measured['timing'].update(serialization_parts)
    if generated:
        measured['timing'].update(generation_parts)
        measured['timing']['conditional_execution_ns']=None
    else:
        measured['timing']['conditional_execution_ns']=measured['timing'].pop('end_to_end_ns')
        measured['timing'].update(end_to_end_ns=None,generation_ns=None,source_generation_ns=None,target_generation_ns=None)
        if upstream is None: measured['timing']['selection_ns']=None
    measured.update(active)
    return measured


def _audit(measured,clock):
    from .geometry import effective_coefficient
    state=measured['state']; target=measured['target_state']; method=measured['method']; contexts=measured['data']['contexts']
    if target is None:
        target=measured['target_method'].prepare(state.prior,state.budget,state.preparation)
        measured['target_state']=target
    errors=[]; raw_errors=[]; raw_timing={'literal_network_adaptation_ns':None,'collapse_ns':None,'raw_network_prediction_ns':None}
    if target.initial_layers is not None:
        # Audit uses original literal path after cold timing; never prewarms it.
        literal,raw_timing['literal_network_adaptation_ns']=_timed(clock,lambda:[measured['target_method'].adapt(target,c) for c in contexts])
        collapsed,raw_timing['collapse_ns']=_timed(clock,lambda:[effective_coefficient(a.layers) for a in literal])
        raw,raw_timing['raw_network_prediction_ns']=_timed(clock,lambda:[_raw_predict(a.layers,c.X) for a,c in zip(literal,contexts)])
        for a,c,coefficient,prediction,optimized in zip(literal,contexts,collapsed,raw,measured['adapted']):
            errors.append(float(np.max(np.abs(optimized.coefficient-a.coefficient))))
            raw_errors.append(float(np.max(np.abs(prediction-c.X@coefficient))))
            if not np.allclose(optimized.coefficient,a.coefficient,rtol=1e-10,atol=1e-12): raise ValueError('B05 equality failed: literal/compact coefficient mismatch')
            if not np.allclose(prediction,c.X@optimized.coefficient,rtol=1e-10,atol=1e-12): raise ValueError('raw/collapsed prediction mismatch')
    else:
        for adapted,context in zip(measured['adapted'],contexts):
            literal=method.adapt(state,context)
            if not np.allclose(adapted.coefficient,literal.coefficient,rtol=1e-10,atol=1e-12): raise ValueError('optimized execution mismatch')
    equality=[{'target':target.method_id,'method_id':'B05:'+target.method_id,'status':'pass','max_absolute_error':max(errors,default=0.),'raw_prediction_max_error':max(raw_errors,default=0.)}] if target.initial_layers is not None else []
    return equality,raw_timing


def _risk(measured):
    data=measured['data']; state=measured['state']
    if data['truth'] is None: return {'status':'unavailable','reason':'no evaluator truth for resident contexts'}
    losses=[float(np.sum((a.coefficient-t.theta)**2)) for a,t in zip(measured['adapted'],data['truth'])]
    finite_moments=(state.metric is not None or state.options.get('zero_estimator',False) or state.method_id=='B08:gd' or (state.method_id.endswith(':ridge') and state.options.get('penalty',0)>0) or (state.method_id.startswith('B10:') and state.prior.noise_variance>0))
    result={'status':'measured','uncertainty_status':'finite_moment_diagnostic' if finite_moments else 'descriptive_only','moment_reason':'fixed map or positive regularization bounds the estimator' if finite_moments else 'unregularized inverse/pseudoinverse: finite risk moments not established by finite samples','losses':losses,'task_keys':[c.task_key for c in data['contexts']],
            'mean':float(np.mean(losses)),'conditional_exact':False,'scope':'coefficient error on fixed independent Gaussian test tasks; selected preparation fixed'}
    if state.metric is not None:
        from .evaluation import conditional_risk
        result.update(mean=conditional_risk(state.metric,data['population']),conditional_exact=True,
                      scope='analytic conditional expected coefficient error under declared Gaussian population; float64 diagnostic')
    return result


def run_block(request, *, clock=time.perf_counter_ns, upstream=None, in_process=False):
    settings=request['settings']; mid=request['method']['method_id']
    conditions=_conditions({'input_residency':'generated_in_timed_operation' if request['workflow']=='generated' else 'provided_contexts',**settings})
    baseline=normalize_peak_rss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,platform.system())
    cold=operation(request,clock,upstream)
    # Capture peak before equality, raw-network auditing and evidence accounting.
    cold_peak=normalize_peak_rss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,platform.system())
    execution_hash=content_hash(_runtime_payload(cold['state']))
    selected_hash=content_hash(_provenance(cold['asset']))
    samples=[]
    if 'fixed_repetitions' in settings:
        repetitions=settings['fixed_repetitions']
        if type(repetitions) is not int or repetitions<1: raise ValueError('invalid fixed repetition count')
        pilot={'method_id':mid,'block':request['block'],'samples_ns':[], 'repetitions':repetitions,'status':'frozen_no_recalibration'}
    else:
        for _ in range(settings['pilot_blocks']):
            sample=operation(request,clock,upstream)
            samples.append(sample['timing']['end_to_end_ns'] if request['workflow']=='generated' else sample['timing']['conditional_execution_ns'])
            if content_hash(_runtime_payload(sample['state']))!=execution_hash: raise ValueError('pilot changed selected assets')
            del sample
        mean=float(np.mean(samples)); cv=float(np.std(samples,ddof=1)/mean) if len(samples)>1 and mean else 0.
        desired=max(1,math.ceil(settings['target_block_ns']/max(1,mean)),math.ceil((cv/settings['target_relative_error'])**2))
        repetitions=min(settings['max_repetitions'],desired)
        pilot={'method_id':mid,'block':request['block'],'samples_ns':samples,'coefficient_of_variation':cv,
               'desired_repetitions':desired,'repetitions':repetitions,'capped':desired>repetitions,
               'target_block_ns':settings['target_block_ns'],'target_relative_error':settings['target_relative_error']}
    values=[]; raw_values=[]; warm_peaks=[]
    for _ in range(repetitions):
        warm=operation(request,clock,upstream)
        warm_peaks.append(normalize_peak_rss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,platform.system()))
        if content_hash(_runtime_payload(warm['state']))!=execution_hash: raise ValueError('warm operation changed selected assets')
        values.append(warm['timing'])
        del warm
    warm_timing={k:float(np.mean([r[k] for r in values])) if values[0][k] is not None else None for k in values[0]}
    # Optional literal work is strictly after ALL optimized time/RSS samples.
    equality,cold_raw=_audit(cold,clock)
    for _ in range(repetitions):
        _,raw=_audit(cold,clock); raw_values.append(raw)
    warm_raw={k:float(np.mean([r[k] for r in raw_values])) if raw_values[0][k] is not None else None for k in raw_values[0]}
    rows=[]
    for temperature,timing,raw,reps,peak in [('cold',cold['timing'],cold_raw,1,cold_peak),('warm',warm_timing,warm_raw,repetitions,max(warm_peaks))]:
        if in_process and temperature=='cold': continue
        label=temperature if not in_process else 'warm'
        repetition_samples=[cold['timing']] if temperature=='cold' else values
        elapsed_field='end_to_end_ns' if request['workflow']=='generated' else 'conditional_execution_ns'
        stratum=content_hash({'conditions':conditions,'cold_warm':label,'in_process':in_process})
        row={'kind':'timing_blocks','method_id':mid,'block':request['block'],'method_order':request['method_order'],
             'process_id':os.getpid(),'cold_warm':label,'raw_network_timing_scope':'post-optimized diagnostic; no cold raw-inference claim','block_size':len(cold['data']['contexts']),'repetitions':reps,
             'generation_boundary':'Gaussian source/train/held-out/test generation inside operation' if request['workflow']=='generated' else 'unavailable; resident contexts',
             'preparation_boundary':'source-only fit/held-out selection separate selection_ns; prepared execution asset construction',
             'adaptation_boundary':'shared compact fixed-map application or production small-space solve; all target contexts',
             'prediction_boundary':'collapsed coefficient times resident query X; all contexts',
             'serialization_boundary':'prepared runtime assets and adapted coefficients to compact UTF-8 JSON in memory',
             'wall_time_ns':int(round(sum(sample[elapsed_field] for sample in repetition_samples))),'repetition_samples':repetition_samples,'block_timing_statistic':'stage fields are per-repetition means; wall_time_ns sums nested pipeline intervals, excluding scheduling gaps',
             'peak_rss':peak,'peak_rss_unit':'bytes','power_thermal_condition':str(conditions['power'])+'/'+str(conditions['thermal']),
             'gpu_metadata':None,'timing_stratum':stratum,'null_reasons':{'gpu_metadata':'CPU','end_to_end_ns':'unavailable for supplied upstream inputs'},
             'selected_execution_hash':selected_hash,'execution_hash':execution_hash,
             'input_residency':conditions['input_residency'],'batch_size':settings['batch_size'],'batch_workspace_bytes':cold['batch_workspace_bytes'],**timing,**raw}
        from .records import validate_raw_record
        validate_raw_record(row,{'run_id':'benchmark'})
        rows.append(row)
    state=cold['state']; runtime=_runtime_payload(state); inventory=storage_inventory(runtime)
    evidence={'solver_diagnostics':state.solver_diagnostics,'prior_fit_metadata':state.prior.fit_metadata,'selection':cold['asset'][2]}
    full_resident=storage_inventory({'state':state,'method':cold['method'].options,'target':cold['target_state'],
                                     'selection':cold['asset'][2],'inputs':cold['data'],'adapted':cold['adapted']})
    resource_row={'method_id':mid,'initialization_norm':state.resources.initialization_norm,'trainable_parameters':state.resources.trainable_parameters,
        'runtime_stored_scalars':inventory['array_scalars']+inventory['scalar_values'],'runtime_serialized_bytes':inventory['serialized_bytes'],
        'runtime_storage':inventory,'resident_inventory':full_resident,'resident_inventory_scope':'retained inputs/evaluator truth, selected state/evidence and optional raw audit state; distinct from pre-audit optimized peak',
        'frozen_prior_storage':storage_inventory(state.prior),'adapted_storage':storage_inventory(cold['adapted']),
        'evidence_serialized_bytes':len(_json_bytes(evidence)),'evidence_bytes_are_additional':True,'evidence_contains_candidate_runtime_snapshots':True,
        'evidence_storage':storage_inventory(evidence),'serialized_payload_bytes':len(cold['payload']),
        'memory':{'status':'measured' if not in_process else 'unavailable','isolation':'fresh_interpreter' if not in_process else 'in_process_test_or_callback',
                  'peak_rss_bytes':cold_peak if not in_process else None,'baseline_rss_bytes':baseline,
                  'cold_peak_rss_bytes':cold_peak if not in_process else None,
                  'warm_sequence_peak_rss_bytes':max(warm_peaks) if not in_process else None,
                  'workspace_bytes':None,'reason':'comparable peak is first optimized pipeline before optional diagnostics; whole process includes imports/inputs/selection evidence; warm sequence also retains first execution and is not comparable workspace'},
        'selected_asset':to_json_record(state.prior),'selection':cold['asset'][2],'selected_execution_hash':selected_hash,
        'shared_upstream_target':mid[4:] if mid.startswith('B05:') else None,
        'tuning_resolution':next((r.get('tuning_resolution') for r in cold['asset'][2]['candidate_records'] if r.get('candidate_id')==cold['asset'][2]['selected_id']),None),
        'upstream_charge':'full reproducible target source fit/selection charged; exact selected metric reused by every eligible method; B05 never retunes independently',
        'execution_hash':execution_hash,'risk':_risk(cold),'equality':equality,'equality_status':'pass',
        'diagnostics':cold['method'].diagnostics(state,cold['adapted'][0])}
    return {'rows':rows,'resource':to_json_record(resource_row),'pilot':pilot,'conditions':conditions}


def main():
    try:
        if sys.argv[1]=='--identity':
            from .protocol import runtime_identity
            print(json.dumps(runtime_identity(),allow_nan=False)); return 0
        request=json.loads(Path(sys.argv[1]).read_text())
        print(json.dumps(run_block(request),allow_nan=False))
    except Exception as exc:
        print(json.dumps({'status':'failed','error':str(exc),'details':getattr(exc,'details',None),'timing_failure':getattr(exc,'timing_failure',None)},allow_nan=False),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
