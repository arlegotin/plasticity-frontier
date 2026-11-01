"""Exploratory operational measurements with explicit accounting boundaries."""
import json
import csv
import contextlib
import io
import warnings
import math
import multiprocessing as mp
import os
import platform
import random
import resource
import statistics
import time
from pathlib import Path

import numpy as np
from .geometry import effective_coefficient
from .methods import make_method
from .records import to_json_record, content_hash


def normalize_peak_rss(raw_value, platform_name):
    if raw_value < 0: raise ValueError('negative RSS')
    if platform_name == 'Darwin': return int(raw_value)
    if platform_name == 'Linux': return int(raw_value) * 1024
    raise ValueError('unknown RSS units')


def amortized_cost(preparation_ns, adaptation_ns, prediction_ns, future_tasks):
    if not isinstance(future_tasks, int) or isinstance(future_tasks, bool) or future_tasks < 1: raise ValueError('future_tasks must be positive integer')
    if any(not math.isfinite(x) or x < 0 for x in (preparation_ns, adaptation_ns, prediction_ns)): raise ValueError('invalid cost')
    return preparation_ns / future_tasks + adaptation_ns + prediction_ns


def balanced_orders(ids, blocks, seed):
    if len(set(ids)) != len(ids) or not ids or blocks < 1: raise ValueError('invalid schedule')
    rng = random.Random(seed); result = []
    while len(result) < blocks:
        base = list(ids); rng.shuffle(base)
        result.extend(base[i:]+base[:i] for i in range(len(base)))
    return result[:blocks]


def memory_schedule(science, operations, pressure=False):
    result = dict(operations)
    if pressure:
        for key in ('batch_size', 'checkpoint_tasks'):
            if key in result: result[key] = max(1, result[key]//2)
    return {'science': dict(science), 'operations': result}


def _json_bytes(value):
    return json.dumps(to_json_record(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _scalars(value):
    # Serialized occurrences are counted even when arrays alias: serialization duplicates them.
    from dataclasses import fields, is_dataclass
    from collections.abc import Mapping
    if isinstance(value, np.ndarray): return int(value.size)
    if is_dataclass(value): return sum(_scalars(getattr(value, f.name)) for f in fields(value))
    if isinstance(value, Mapping): return sum(_scalars(v) for v in value.values())
    if isinstance(value, (list, tuple)): return sum(_scalars(v) for v in value)
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool): return 1
    return 0


def _runtime_payload(state):
    # Diagnostics are retained evidence, not required to execute adapt/predict.
    return {'metric': state.metric, 'initial_layers': state.initial_layers, 'budget': state.budget,
            'preparation': state.preparation, 'prior': state.prior, 'options': state.options,
            'method_id': state.method_id}


def _memory_child(pipe, method, prior, budget, preparation, context):
    try:
        state = method.prepare(prior, budget, preparation)
        adapted = method.adapt(state, context)
        method.predict(adapted, context.X)
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        pipe.send({'status':'measured', 'peak_rss_bytes':normalize_peak_rss(raw, platform.system()),
                   'scope':'isolated child process including interpreter, imports and inherited inputs'})
    except Exception as exc:
        pipe.send({'status':'unavailable', 'reason':str(exc), 'peak_rss_bytes':None})
    finally: pipe.close()


def _isolated_memory(method, prior, budget, preparation, context):
    # Fork starts a distinct RSS high-water mark; process baseline and inherited input are included.
    try:
        ctx = mp.get_context('fork'); parent, child = ctx.Pipe(False)
        process = ctx.Process(target=_memory_child, args=(child, method, prior, budget, preparation, context))
        process.start(); child.close()
        if not parent.poll(60):
            process.terminate(); process.join(); return {'status':'unavailable','reason':'child timed out','peak_rss_bytes':None}
        value = parent.recv(); process.join(); parent.close()
        if process.exitcode != 0: return {'status':'unavailable','reason':f'child exit {process.exitcode}','peak_rss_bytes':None}
        return value
    except (OSError, EOFError, ValueError) as exc:
        return {'status':'unavailable','reason':str(exc),'peak_rss_bytes':None}


def _timed(clock, fn):
    start = clock(); value = fn(); elapsed = clock()-start
    if elapsed < 0: raise ValueError('clock moved backwards')
    return value, elapsed


def _raw_predict(layers, X):
    value = X.T
    for layer in layers: value = layer @ value
    return value.reshape(-1)


def benchmark_methods(methods, contexts, config, output):
    methods = list(methods); contexts = list(contexts)
    if not methods or not contexts or len({m.method_id for m in methods}) != len(methods): raise ValueError('methods and contexts must be nonempty with unique IDs')
    prior, budget, preparation = (config[k] for k in ('prior','budget','preparation'))
    clock = config.get('clock', time.perf_counter_ns)
    pilot_blocks = int(config.get('pilot_blocks', 3)); blocks = int(config.get('blocks', 6)); max_reps = int(config.get('max_repetitions', 100))
    if min(pilot_blocks, blocks, max_reps) < 1: raise ValueError('invalid timing schedule')
    states = {}; by_id = {m.method_id:m for m in methods}
    # Construct B05 from the actual target metric, never an independently fitted approximation.
    for m in methods:
        if not m.method_id.startswith('B05:'): states[m.method_id] = m.prepare(prior, budget, preparation)
    for m in methods:
        if m.method_id.startswith('B05:'):
            target = m.method_id[4:]
            if target not in states or states[target].metric is None: raise ValueError('B05 requires present target with a metric')
            supplied = m.options.get('metric')
            if supplied is not None and _json_bytes(supplied) != _json_bytes(states[target].metric): raise ValueError('B05 metric differs from target')
            by_id[m.method_id] = make_method(m.method_id, dict(m.options, metric=states[target].metric))
            states[m.method_id] = by_id[m.method_id].prepare(prior, budget, preparation)
    equality = []
    for mid, method in by_id.items():
        if mid.startswith('B05:'):
            target=mid[4:]; errors=[]
            for context in contexts:
                a=by_id[target].adapt(states[target],context).coefficient
                b=method.adapt(states[mid],context).coefficient
                errors.append(float(np.max(np.abs(a-b))))
                if not np.allclose(a,b,rtol=1e-10,atol=1e-12): raise ValueError(f'B05 equality failed: {mid}')
            equality.append({'target':target,'method_id':mid,'status':'pass','max_absolute_error':max(errors)})
    upstream_samples=[]
    upstream = config.get('upstream_prepare')
    if upstream is not None:
        for _ in range(pilot_blocks):
            measured, elapsed = _timed(clock, upstream)
            if content_hash(measured) != content_hash(prior): raise ValueError('upstream preparation differs from supplied prior')
            upstream_samples.append(elapsed)
    rows=[]; resources=[]; pilots={}
    for mid, method in by_id.items():
        state=states[mid]; adapted=method.adapt(state, contexts[0]); method.predict(adapted,contexts[0].X)
        samples=[]
        for _ in range(pilot_blocks):
            _, elapsed = _timed(clock, lambda: method.adapt(state, contexts[0])); samples.append(elapsed)
        mean=statistics.mean(samples); cv=statistics.stdev(samples)/mean if len(samples)>1 and mean else 0.
        # Conservative target 10% relative standard error; cap and expose unmet target.
        desired=max(1, math.ceil((cv/.1)**2)); reps=min(max_reps, desired)
        pilots[mid]={'adaptation_samples_ns':samples,'coefficient_of_variation':cv,'target_relative_standard_error':.1,'desired_repetitions':desired,'repetitions':reps,'capped':desired>max_reps}
        payload=_runtime_payload(state); runtime_bytes=len(_json_bytes(payload)); evidence_bytes=len(_json_bytes(state))
        memory = _isolated_memory(method, prior, budget, preparation, contexts[0]) if config.get('measure_memory', True) else {'status':'not_run','reason':'disabled','peak_rss_bytes':None}
        resources.append({'method_id':mid,'initialization_norm':state.resources.initialization_norm,'trainable_parameters':state.resources.trainable_parameters,'runtime_stored_scalars':_scalars(payload),'runtime_serialized_bytes':runtime_bytes,'evidence_serialized_bytes':evidence_bytes,'evidence_includes_runtime':True,'memory':memory,'diagnostics':method.diagnostics(state, adapted)})
    orders=balanced_orders(list(by_id),blocks,int(config.get('seed',0)))
    for block, order in enumerate(orders):
        for mid in order:
            method=by_id[mid]; reps=pilots[mid]['repetitions']; sums={k:0 for k in ('preparation_ns','adaptation_ns','prediction_ns','serialization_ns','collapse_ns','raw_network_prediction_ns','end_to_end_ns')}
            raw_available=False
            for i in range(reps):
                context=contexts[(block*reps+i)%len(contexts)]
                state,t=_timed(clock,lambda:method.prepare(prior,budget,preparation)); sums['preparation_ns']+=t
                adapted,t=_timed(clock,lambda:method.adapt(state,context)); sums['adaptation_ns']+=t
                if adapted.layers is not None:
                    collapsed,t=_timed(clock,lambda:effective_coefficient(adapted.layers)); sums['collapse_ns']+=t
                    if not np.allclose(collapsed,adapted.coefficient,rtol=1e-10,atol=1e-12): raise ValueError('collapse mismatch')
                    raw,t=_timed(clock,lambda:_raw_predict(adapted.layers,context.X)); sums['raw_network_prediction_ns']+=t; raw_available=True
                    if not np.allclose(raw,method.predict(adapted,context.X),rtol=1e-10,atol=1e-12): raise ValueError('raw prediction mismatch')
                _,t=_timed(clock,lambda:method.predict(adapted,context.X)); sums['prediction_ns']+=t
                _,t=_timed(clock,lambda:_json_bytes(state)); sums['serialization_ns']+=t
                def operational_path():
                    prepared = method.prepare(prior,budget,preparation)
                    updated = method.adapt(prepared,context)
                    method.predict(updated,context.X)
                    _json_bytes(prepared)
                _,t=_timed(clock,operational_path); sums['end_to_end_ns']+=t
            row={'block':block,'method_id':mid,'method_order':order,'cold_warm':'warm','repetitions':reps,'generation_ns':None,'generation_status':'not measured; contexts provided resident','input_residency':'provided_contexts'}
            row.update({k:v/reps for k,v in sums.items()});
            if not raw_available: row['collapse_ns']=None; row['raw_network_prediction_ns']=None
            rows.append(row)
    summaries=[]
    for mid in by_id:
        own=[r for r in rows if r['method_id']==mid]
        timing={}
        for field in ('preparation_ns','adaptation_ns','prediction_ns','serialization_ns','end_to_end_ns'):
            values=[r[field] for r in own]; timing[field]={'mean':statistics.mean(values),'standard_deviation':statistics.stdev(values) if len(values)>1 else None,'blocks':len(values)}
        amortization={str(n):amortized_cost(timing['preparation_ns']['mean'],timing['adaptation_ns']['mean'],timing['prediction_ns']['mean'],n) for n in config.get('future_tasks',[1,10,100])}
        summary={'method_id':mid,'timing':timing,'amortized_ns_per_task':amortization}
        if upstream_samples:
            summary['amortized_with_upstream_ns_per_task']={str(n):amortized_cost(statistics.mean(upstream_samples)+timing['preparation_ns']['mean'],timing['adaptation_ns']['mean'],timing['prediction_ns']['mean'],n) for n in config.get('future_tasks',[1,10,100])}
        summaries.append(summary)
    # Cost-only Pareto table; no scientific risk claim follows from this ordering.
    resource_by_id={r['method_id']:r for r in resources}
    pareto=[]
    for summary in summaries:
        mid=summary['method_id']; time_cost=summary['timing']['end_to_end_ns']['mean']; size=resource_by_id[mid]['runtime_serialized_bytes']
        dominators=[other['method_id'] for other in summaries if other['method_id'] != mid and other['timing']['end_to_end_ns']['mean'] <= time_cost and resource_by_id[other['method_id']]['runtime_serialized_bytes'] <= size and (other['timing']['end_to_end_ns']['mean'] < time_cost or resource_by_id[other['method_id']]['runtime_serialized_bytes'] < size)]
        pareto.append({'method_id':mid,'mean_ns':time_cost,'runtime_serialized_bytes':size,'dominated_by':dominators})
    blas_buffer=io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(blas_buffer):
        warnings.simplefilter('ignore', UserWarning)
        np.show_config()
    result={'pareto_cost_only':pareto,'context_generation':config.get('context_generation', {'status':'not_run','reason':'contexts supplied by caller'}), 'inputs':to_json_record({'prior':prior,'budget':budget,'preparation':preparation,'contexts':contexts}), 'input_hash':content_hash({'prior':prior,'budget':budget,'preparation':preparation,'contexts':contexts}), 'status':'exploratory','equality':equality,'timing_blocks':rows,'pilot':pilots,'resources':resources,'summaries':summaries,'conditions':{'device':'CPU','dtype':'float64','input_residency':'provided_contexts','thread_policy':config.get('thread_policy','unknown'),'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},'blas':blas_buffer.getvalue(),'thermal':config.get('thermal','unknown'),'power':config.get('power','unknown'),'cold':'not_run; process isolation only for memory','warm':'paths warmed before measurement','platform':platform.platform()},'prior_cost':({'status':'measured','samples_ns':upstream_samples,'mean_ns':statistics.mean(upstream_samples),'scope':config.get('upstream_scope','caller supplied upstream preparation; source generation and selection included only if callable performs them')} if upstream_samples else {'status':'unavailable','reason':'provided prior; fitting and selection not measured'}),'accelerator':{'status':'not_run','reason':'CPU core only'},'energy':{'status':'not_run','joules':None,'reason':'no validated energy measurement'},'timing_scope':{'end_to_end':'warm preparation, adaptation, prediction and serialization; provided inputs resident; excludes prior construction and generation','collapse':'diagnostic repeated collapse; adaptation already returns collapsed coefficient','raw_network_prediction':'additional separately timed diagnostic, excluded from end_to_end','comparison':'descriptive only; no timing inference without equality and matched risk'},'decision':{'status':'inconclusive','reason':'exploratory timing only; no matched risk or production schedule'}}
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/'benchmark.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'timing_blocks.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows(dict(row,method_order=json.dumps(row['method_order'])) for row in rows)
    return result
