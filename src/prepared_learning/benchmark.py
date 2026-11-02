"""Replayable CPU operational measurements, with explicitly conditional costs.

Timing repetitions repeat a fixed preparation/task workload. They are never
additional independent scientific observations. Every production timing block
runs in a fresh interpreter; imports and input decoding precede the cold scope.
"""
import contextlib
import csv
import io
import json
import math
import os
import platform
import random
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
from .records import to_json_record, content_hash, Context, JsonRecord
from .methods import Method

THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
STAGES = ('generation', 'selection', 'preparation', 'adaptation', 'prediction', 'serialization')


def normalize_peak_rss(raw_value: int, platform_name: str) -> int:
    if raw_value < 0: raise ValueError('negative RSS')
    if platform_name == 'Darwin': return int(raw_value)
    if platform_name == 'Linux': return int(raw_value) * 1024
    raise ValueError('unknown RSS units')


def amortized_cost(preparation_ns: float, adaptation_ns: float, prediction_ns: float, future_tasks: int) -> float:
    if type(future_tasks) is not int or future_tasks < 1: raise ValueError('future_tasks must be positive integer')
    if any(not math.isfinite(x) or x < 0 for x in (preparation_ns, adaptation_ns, prediction_ns)): raise ValueError('invalid cost')
    return preparation_ns / future_tasks + adaptation_ns + prediction_ns



def amortized_pipeline_cost(timing, *, future_tasks, measured_tasks):
    if type(measured_tasks) is not int or measured_tasks < 1: raise ValueError('positive task count required')
    one_time=sum(timing[k] for k in ('source_generation_ns','selection_ns','preparation_ns','prepared_serialization_ns'))
    per_task=sum(timing[k] for k in ('target_generation_ns','adaptation_ns','prediction_ns','adapted_serialization_ns'))/measured_tasks
    return amortized_cost(one_time,per_task,0.,future_tasks)


def scientific_configuration(config):
    operational={'batch_size','checkpoint_tasks','output','threads','thread_policy','thermal','power',
                 'memory_limit_bytes','measure_memory','worker_timeout_seconds'}
    return {k:to_json_record(v) for k,v in config.items() if k not in operational}

def balanced_orders(ids, blocks, seed):
    if len(set(ids)) != len(ids) or not ids or blocks < 1: raise ValueError('invalid schedule')
    rng = random.Random(seed); result = []
    # Complete cycles are necessary for exact position balance.
    count = math.ceil(blocks / len(ids)) * len(ids)
    while len(result) < count:
        base = list(ids); rng.shuffle(base)
        rotations = [base[i:] + base[:i] for i in range(len(base))]
        rng.shuffle(rotations); result.extend(rotations)
    return result


def memory_schedule(science, operations, pressure=False):
    result = dict(operations)
    if pressure:
        for key in ('batch_size', 'checkpoint_tasks'):
            if key in result: result[key] = max(1, result[key] // 2)
    return {'science': to_json_record(science), 'operations': result}


def _json_bytes(value):
    return json.dumps(to_json_record(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def storage_inventory(value):
    """Count live unique array storage separately from JSON numeric occurrences.

    Array views are unioned by address ranges; shared objects are traversed once
    for resident counts, but JSON duplicates remain charged in serialization.
    Scalar values count numeric fields (not Python interned-object identities).
    Python/container allocator overhead is observed only by process RSS.
    """
    seen = set(); intervals = []; scalars = 0; serialized_arrays = 0
    def items(obj):
        if is_dataclass(obj): return [getattr(obj, f.name) for f in fields(obj)]
        if isinstance(obj, Mapping): return list(obj.values())
        if isinstance(obj, (tuple, list)): return obj
        return None
    def serialized(obj):
        nonlocal serialized_arrays
        if isinstance(obj, np.ndarray): serialized_arrays += obj.size
        else:
            children = items(obj)
            if children is not None:
                for child in children: serialized(child)
    def resident(obj):
        nonlocal scalars
        if isinstance(obj, (int, float, np.number)) and not isinstance(obj, (bool, np.bool_)):
            scalars += 1; return
        if id(obj) in seen: return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            if not obj.flags.c_contiguous: raise ValueError('storage inventory requires contiguous arrays')
            start = obj.__array_interface__['data'][0]
            intervals.append((start, start + obj.nbytes, obj.dtype.itemsize))
        else:
            children = items(obj)
            if children is not None:
                for child in children: resident(child)
    serialized(value); resident(value)
    merged=[]
    for start, end, itemsize in sorted(intervals):
        if merged and start < merged[-1][1]:
            if itemsize != merged[-1][2]: raise ValueError('aliased arrays with mixed dtypes')
            merged[-1]=(merged[-1][0], max(end, merged[-1][1]), itemsize)
        else: merged.append((start,end,itemsize))
    return {'array_scalars':sum((b-a)//s for a,b,s in merged), 'array_bytes':sum(b-a for a,b,_ in merged),
            'scalar_values':scalars, 'serialized_array_scalars':int(serialized_arrays),
            'serialized_bytes':len(_json_bytes(value)), 'format':'canonical compact UTF-8 JSON',
            'python_object_overhead':'included only in process RSS'}


def _timed(clock, fn):
    start=clock(); value=fn(); elapsed=clock()-start
    if elapsed < 0: raise ValueError('clock moved backwards')
    return value, elapsed


def measure_operation(stages, *, clock=time.perf_counter_ns):
    """Single nested boundary, preserving partial timing on a failed stage."""
    start=clock(); timing={}
    def measure(name,fn):
        stage_start=clock()
        try: value=fn()
        except Exception as exc:
            timing[name+'_ns']=clock()-stage_start
            exc.timing_failure={'failed_stage':name,**timing,'end_to_end_ns':clock()-start}
            raise
        timing[name+'_ns']=clock()-stage_start
        if timing[name+'_ns']<0: raise ValueError('clock moved backwards')
        return value
    data=measure('generation',stages['generation'])
    asset=measure('selection',lambda:stages['selection'](data))
    state=measure('preparation',lambda:stages['preparation'](asset))
    adapted=measure('adaptation',lambda:stages['adaptation'](state,data))
    prediction=measure('prediction',lambda:stages['prediction'](adapted,data))
    payload=measure('serialization',lambda:stages['serialization'](state,adapted))
    timing['end_to_end_ns']=clock()-start
    return {'timing':timing,'data':data,'asset':asset,'state':state,'adapted':adapted,'prediction':prediction,'payload':payload}


def _conditions(config):
    buffer=io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buffer):
        warnings.simplefilter('ignore',UserWarning); np.show_config()
    return {'device':'CPU','dtype':'float64','input_residency':config.get('input_residency','provided_contexts'),
            'threads_requested':config.get('threads',1), 'thread_environment':{k:os.environ.get(k) for k in THREAD_ENV},
            'blas':buffer.getvalue(),'thermal':config.get('thermal','unmeasured'),'power':config.get('power','unmeasured'),
            'platform':platform.platform(), 'cold':'first method operation in a fresh interpreter after imports/input decode',
            'warm':'same process after cold execution and pilot; repeated identical assets and tasks'}


def _atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_bytes(_json_bytes(value)+b'\n')
    os.replace(temporary,path)


def _subprocess_block(request, output, env):
    request_path=output/'requests'/f"{request['block']:05d}-{request['position']:03d}.json"
    _atomic_json(request_path,request)
    completed=subprocess.run([sys.executable,'-m','prepared_learning._benchmark_worker',str(request_path)],
                             capture_output=True,text=True,env=env,timeout=request['settings'].get('worker_timeout_seconds',300))
    if completed.returncode:
        try: failure=json.loads(completed.stderr)
        except ValueError: failure={'status':'failed','error':completed.stderr,'stdout':completed.stdout}
        failure_path=request_path.with_suffix('.failure.json'); _atomic_json(failure_path,failure)
        raise ValueError('benchmark worker failed: '+failure.get('error','unknown')+'; full evidence: '+str(failure_path))
    return json.loads(completed.stdout)


def _interval(values, confidence=.95):
    from scipy.stats import t
    values=np.asarray(values,dtype=float); mean=float(values.mean())
    if len(values)<2: return {'mean':mean,'low':None,'high':None,'standard_deviation':None,'blocks':len(values)}
    sd=float(values.std(ddof=1)); radius=float(t.ppf((1+confidence)/2,len(values)-1)*sd/np.sqrt(len(values)))
    return {'mean':mean,'low':mean-radius,'high':mean+radius,'standard_deviation':sd,'blocks':len(values)}


def comparison_table(rows, resources, *, gate, confidence=.95, risk_tolerance=1e-10, phase='exploratory'):
    """Conservative scoped Pareto relation using paired timing and risk intervals.

    Peak RSS/storage comparisons are measured descriptive maxima, not memory
    confidence bounds. Timing intervals are Student-t block diagnostics. Risk
    bounds describe fixed selected assets on independent generated target tasks.
    """
    mids=[r['method_id'] for r in resources]; by_id={r['method_id']:r for r in resources}
    pairs=[]; strata=sorted({r['timing_stratum'] for r in rows})
    # Bonferroni simultaneous pairwise/axis diagnostic intervals.
    adjusted=1-(1-confidence)/max(1,len(mids)*(len(mids)-1)*max(1,len(strata))*2)
    for stratum in strata:
        selected=[r for r in rows if r['timing_stratum']==stratum]
        for a in mids:
            for b in mids:
                if a==b: continue
                ra,rb=by_id[a],by_id[b]
                left={r['block']:r for r in selected if r['method_id']==a}; right={r['block']:r for r in selected if r['method_id']==b}
                common=sorted(set(left)&set(right)); field='end_to_end_ns'
                available=common and all(left[i][field] is not None and right[i][field] is not None for i in common)
                timing=_interval([left[i][field]-right[i][field] for i in common],adjusted) if available else None
                risk=None
                if ra['risk']['status']=='measured' and rb['risk']['status']=='measured':
                    if ra['risk'].get('conditional_exact') and rb['risk'].get('conditional_exact'):
                        diff=ra['risk']['mean']-rb['risk']['mean']; risk={'mean':diff,'low':diff,'high':diff,'scope':'analytic conditional risk; float64 diagnostic'}
                    elif (ra['risk'].get('uncertainty_status')!='descriptive_only' and rb['risk'].get('uncertainty_status')!='descriptive_only' and ra['risk']['task_keys']==rb['risk']['task_keys']):
                        risk=_interval(np.asarray(ra['risk']['losses'])-np.asarray(rb['risk']['losses']),adjusted)
                memory_a=ra['memory']['peak_rss_bytes']; memory_b=rb['memory']['peak_rss_bytes']
                storage_delta=ra['runtime_serialized_bytes']-rb['runtime_serialized_bytes']
                memory_delta=memory_a-memory_b if memory_a is not None and memory_b is not None else None
                status='unavailable'
                if gate['status']=='pass' and timing and risk and timing['high'] is not None and risk['high'] is not None and memory_delta is not None:
                    no_worse=timing['high']<=0 and risk['high']<=risk_tolerance and memory_delta<=0 and storage_delta<=0
                    strict=timing['high']<0 or risk['high'] < -risk_tolerance or memory_delta<0 or storage_delta<0
                    status='scoped_domination' if no_worse and strict else 'tradeoff_or_unresolved'
                pairs.append({'method_id':a,'comparator':b,'timing_stratum':stratum,'status':status,
                              'paired_time_difference_ns':timing,'paired_risk_difference':risk,
                              'peak_rss_difference_bytes':memory_delta,'storage_difference_bytes':storage_delta})
    return {'rows':pairs,'confidence':confidence,'pair_interval_confidence':adjusted,
            'uncertainty':'Bonferroni Student-t paired block/task diagnostics; repeated timings do not add risk samples',
            'memory_scope':'maximum measured process RSS; interpreter/imports/input and evidence included, not incremental workspace',
            'scope':'fixed selected assets and generated population, this CPU/dtype/thread/thermal condition; '+('finalized operational campaign' if phase=='finalized' else 'exploratory calibration')}



def _benchmark_identity(config):
    """Fingerprint the interpreter with the SAME pre-import thread environment."""
    from .protocol import checkout_identity
    env=dict(os.environ); env.update({k:str(config.get('threads',1)) for k in THREAD_ENV})
    completed=subprocess.run([sys.executable,'-m','prepared_learning._benchmark_worker','--identity'],
                             capture_output=True,text=True,env=env,timeout=60,check=True)
    runtime=json.loads(completed.stdout)
    versions,dirty=checkout_identity({'benchmark':config.get('methods',[])},config.get('fitting_candidates',{}),runtime=runtime)
    return {'versions':versions,'dirty_diff_sha256':dirty,'runtime':runtime}


def _campaign_configuration(config):
    return {k:v for k,v in scientific_configuration(config).items() if k not in ('phase','frozen','note')}


def _fixed_schedule(evidence):
    config=evidence['config']; ids=config['methods']; repetitions={}
    for mid in ids:
        pilots=[p for p in evidence['pilot'] if p['method_id']==mid]
        if not pilots: raise ValueError('missing measured calibration')
        for pilot in pilots:
            samples=pilot['samples_ns']
            if len(samples)!=config.get('pilot_blocks',3) or any(v<=0 for v in samples): raise ValueError('invalid calibration samples')
            mean=float(np.mean(samples)); cv=float(np.std(samples,ddof=1)/mean) if len(samples)>1 else 0.
            desired=max(1,math.ceil(config.get('target_block_ns',1000000)/mean),math.ceil((cv/config.get('target_relative_error',.1))**2))
            if pilot['desired_repetitions']!=desired or pilot['capped'] or desired>config.get('max_repetitions',100):
                raise ValueError('calibration schedule unresolved or capped')
        repetitions[mid]=max(p['desired_repetitions'] for p in pilots)
    count=evidence['schedule']['balanced_blocks']
    if count<2 or count%len(ids): raise ValueError('complete replicated balanced calibration blocks required')
    return {'blocks':count,'orders':balanced_orders(ids,count,int(config.get('seed',0))^71313),'repetitions':repetitions}


def _guard_calibration_parse(path: Path, *, phase: str) -> None:
    """The native consumer parses JSON regardless of the control's filename."""
    from .reporting import _empty_scale, _resource_check, REPORT_PARSE_FACTOR
    size=path.stat().st_size; scale=_empty_scale()
    scale.update(files_scanned=1,json_bytes=size,max_json_bytes=size,json_working_bytes=size,
                 semantic_json_bytes=size,max_semantic_json_bytes=size,
                 inventory_bytes=4096+16*len(os.fsencode(path)),
                 largest_file_bytes=size,largest_file=str(path))
    _resource_check(scale,phase=phase,json_factor=REPORT_PARSE_FACTOR)


def _canonical_calibration_path(calibration_path: Path) -> Path:
    return calibration_path.with_suffix('.canonical.json')


def _write_canonical_calibration(evidence: dict, path: Path, expected_sha256: str) -> None:
    """Stream the complete native JSON tree, verify it, then publish without replacement."""
    from .reporting import _sha
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix='.'+path.name+'-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            encoder=json.JSONEncoder(sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
            for chunk in encoder.iterencode(evidence): stream.write(chunk)
            stream.flush(); os.fsync(stream.fileno())
        if _sha(temporary)!=expected_sha256: raise ValueError('canonical calibration hash mismatch')
        # A pre-existence check or replace/rename would lose a concurrent creator.
        os.link(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def _verify_calibration_control(frozen: dict) -> tuple[Path, Path] | None:
    """Only a known complete binding with both byte proofs permits hash-only reads."""
    from .reporting import _sha
    if 'calibration_control' not in frozen: return None
    binding=frozen['calibration_control']
    if (not isinstance(binding,dict) or set(binding)!={'format','path','source_sha256'} or
            binding['format']!='prepared_calibration_canonical_v1'):
        raise ValueError('invalid calibration control binding')
    for digest in (binding['source_sha256'],frozen.get('calibration_sha256')):
        if not isinstance(digest,str) or re.fullmatch('[0-9a-f]{64}',digest) is None:
            raise ValueError('invalid calibration control digest')
    original=frozen.get('calibration_path')
    if not isinstance(original,str) or not Path(original).is_absolute():
        raise ValueError('invalid original calibration path')
    original=Path(original); canonical=_canonical_calibration_path(original)
    if (not isinstance(binding['path'],str) or not Path(binding['path']).is_absolute() or
            binding['path']!=str(canonical) or canonical==original):
        raise ValueError('invalid canonical calibration path')
    if _sha(original)!=binding['source_sha256']: raise ValueError('original calibration bytes changed')
    if _sha(canonical)!=frozen['calibration_sha256']: raise ValueError('canonical calibration bytes changed')
    return original,canonical


def finalize_benchmark_manifest(calibration_path, manifest_path, campaign_output):
    """Freeze a measured operational schedule; does not run the campaign.

    Calibration and campaign use the same fixed source-selected assets. Timing
    samples are disjoint fresh executions. This is not new preparation inference.
    """
    from .reporting import _sha
    manifest_path=Path(manifest_path)
    if manifest_path.exists() or manifest_path.is_symlink(): raise ValueError('finalized manifest already exists')
    calibration_path=Path(calibration_path).resolve()
    _guard_calibration_parse(calibration_path,phase='calibration_finalize_before_parse')
    source_sha256=_sha(calibration_path)
    evidence=json.loads(calibration_path.read_text())
    if evidence.get('status')!='exploratory' or evidence.get('config',{}).get('workflow')!='generated': raise ValueError('generated exploratory calibration required')
    if evidence['comparison_gate']['status']!='pass' or evidence['comparison_gate']['missing_recommended_controls']: raise ValueError('equality/risk and direct/ridge/EB controls required')
    identity=_benchmark_identity(evidence['config'])
    if identity['dirty_diff_sha256'] is not None or identity!=evidence['identity']: raise ValueError('calibration must use the same committed code/runtime identity')
    schedule=_fixed_schedule(evidence)
    manifest={**evidence['config'],'phase':'finalized','output':str(Path(campaign_output).resolve())}
    manifest['frozen']={**schedule,'calibration_path':str(calibration_path),'calibration_sha256':content_hash(evidence),
                        'configuration_sha256':content_hash(_campaign_configuration(manifest)),
                        'identity':identity,'selected_execution_hashes':{r['method_id']:r['selected_execution_hash'] for r in evidence['resources']},
                        'condition_hash':content_hash(evidence['conditions'])}
    canonical=_canonical_calibration_path(calibration_path)
    manifest['frozen']['calibration_control']={'format':'prepared_calibration_canonical_v1',
        'path':str(canonical),'source_sha256':source_sha256}
    # Stage beside the source, but publish neither authority until the source
    # has survived the complete parse, semantic checks and canonical export.
    with tempfile.TemporaryDirectory(prefix='.calibration-',dir=calibration_path.parent) as temporary:
        staged=Path(temporary)/'control.json'
        _write_canonical_calibration(evidence,staged,manifest['frozen']['calibration_sha256'])
        if _sha(calibration_path)!=source_sha256: raise ValueError('calibration changed during finalization')
        try:
            os.link(staged,canonical)
        except FileExistsError:
            if _sha(canonical)!=manifest['frozen']['calibration_sha256']:
                raise ValueError('conflicting canonical calibration already exists')
        # A matching orphan sidecar supports retry. The manifest is always new.
        _write_canonical_calibration(manifest,manifest_path,content_hash(manifest))
    return manifest



def _allow_manifest_only(identity,manifest,manifest_path):
    """Match confirmation's sole explicit untracked-manifest exception."""
    if identity['dirty_diff_sha256'] is None: return identity
    if manifest_path is None: raise ValueError('uncommitted campaign inputs')
    root=Path(__file__).resolve().parents[2]
    diff=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=root).split(b'\0')
    paths=[Path(os.path.abspath(root/os.fsdecode(path))) for path in untracked if path]
    allowed=Path(os.path.abspath(manifest_path))
    if diff or paths!=[allowed]: raise ValueError('uncommitted changes beyond explicit frozen manifest')
    if allowed.is_symlink() or not allowed.is_file() or json.loads(allowed.read_text())!=manifest:
        raise ValueError('explicit frozen manifest content or file type changed')
    return {**identity,'dirty_diff_sha256':None}

def validate_campaign_manifest(manifest, *, manifest_path=None):
    frozen=manifest.get('frozen',{})
    if manifest.get('phase')!='finalized': raise ValueError('finalized phase required')
    if content_hash(_campaign_configuration(manifest))!=frozen.get('configuration_sha256'): raise ValueError('campaign configuration changed')
    from .reporting import _sha
    calibration_path=Path(frozen['calibration_path'])
    _guard_calibration_parse(calibration_path,phase='calibration_validate_before_parse')
    source_sha256=_sha(calibration_path)
    _verify_calibration_control(frozen)
    evidence=json.loads(calibration_path.read_text())
    if _sha(calibration_path)!=source_sha256: raise ValueError('calibration changed during validation')
    if content_hash(evidence)!=frozen['calibration_sha256']: raise ValueError('calibration evidence changed')
    schedule=_fixed_schedule(evidence)
    if any(frozen.get(k)!=v for k,v in schedule.items()): raise ValueError('frozen schedule differs from measured calibration')
    if _campaign_configuration(manifest)!=_campaign_configuration(evidence['config']): raise ValueError('campaign configuration differs from calibration')
    expected={r['method_id']:r['selected_execution_hash'] for r in evidence['resources']}
    if frozen['selected_execution_hashes']!=expected: raise ValueError('selected assets changed')
    if frozen['condition_hash']!=content_hash(evidence['conditions']): raise ValueError('timing conditions changed')
    identity=_allow_manifest_only(_benchmark_identity(manifest),manifest,manifest_path)
    if identity['dirty_diff_sha256'] is not None or identity!=frozen['identity'] or identity!=evidence['identity']:
        raise ValueError('campaign requires exact committed code/runtime identity')
    return {'status':'pass','calibration_sha256':frozen['calibration_sha256']}

def benchmark_methods(methods: Sequence[Method], contexts: Sequence[Context], config: JsonRecord, output: Path) -> JsonRecord:
    from ._benchmark_worker import run_block
    methods=list(methods); contexts=list(contexts)
    if not methods or len({m.method_id for m in methods})!=len(methods): raise ValueError('unique nonempty methods required')
    clock=config.get('clock'); upstream=config.get('upstream_prepare'); manifest_path=config.get('_manifest_path')
    # Detach all reproducible configuration. Nonserializable test clock/callbacks
    # are named explicitly and never advertised as replayable worker evidence.
    clean={k:v for k,v in config.items() if k not in ('clock','upstream_prepare','_manifest_path')}
    clean=json.loads(_json_bytes(clean)); generated=clean.get('workflow')=='generated'
    campaign=clean.get('phase')=='finalized'
    if campaign:
        if clock is not None or upstream is not None: raise ValueError('campaign cannot use runtime callbacks')
        validate_campaign_manifest(clean,manifest_path=manifest_path)
        if Path(output).exists(): raise ValueError('campaign output must be fresh')
    identity=_benchmark_identity(clean)
    if campaign: identity=_allow_manifest_only(identity,clean,manifest_path)
    if not generated and not contexts: raise ValueError('resident contexts required')
    settings={k:clean.get(k,default) for k,default in {'pilot_blocks':3,'blocks':6,'max_repetitions':100,
              'target_block_ns':1000000,'target_relative_error':.1,'seed':0,'threads':1,'batch_size':16,'checkpoint_tasks':16}.items()}
    for key in ('pilot_blocks','blocks','max_repetitions','threads','batch_size','checkpoint_tasks'):
        if type(settings[key]) is not int or settings[key]<1: raise ValueError('invalid '+key)
    if settings['target_block_ns']<=0 or not 0<settings['target_relative_error']<1: raise ValueError('invalid calibration target')
    settings.update({k:clean[k] for k in ('thermal','power','worker_timeout_seconds','memory_limit_bytes','measure_memory') if k in clean})
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    inputs=({k:clean[k] for k in ('prior','budget','preparation')} | {'contexts':to_json_record(contexts)}) if not generated else {'manifest':clean}
    descriptors=[{'method_id':m.method_id,'options':to_json_record(m.options)} for m in methods]
    scientific_inputs={'manifest':scientific_configuration(clean)} if generated else inputs
    science={'inputs':scientific_inputs,'methods':descriptors,'schedule':{k:settings[k] for k in ('pilot_blocks','blocks','max_repetitions','seed','target_block_ns','target_relative_error')}}
    science_hash=content_hash(science); operations={k:settings[k] for k in ('batch_size','checkpoint_tasks')}
    from .pilot import _available_memory
    available_memory=_available_memory()
    reserve=clean.get('memory_limit_bytes',available_memory//2 if available_memory>0 else None); baseline=normalize_peak_rss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,platform.system())
    pressure=reserve is not None and baseline>reserve
    effective=memory_schedule(science,operations,pressure)['operations']; settings.update(effective)
    env=dict(os.environ); env.update({k:str(settings['threads']) for k in THREAD_ENV})
    orders=clean['frozen']['orders'] if campaign else balanced_orders([m.method_id for m in methods],settings['blocks'],settings['seed'])
    rows=[]; records=[]; pilots=[]; conditions=[]; pending=[]
    _atomic_json(output/'inputs.json',{'schema_version':1,'scientific_hash':science_hash,**science})
    try:
        for block,order in enumerate(orders):
            for position,mid in enumerate(order):
                descriptor=next(m for m in descriptors if m['method_id']==mid)
                request={'schema_version':1,'workflow':'generated' if generated else 'conditional','inputs':inputs,
                         'method':descriptor,'methods':descriptors,'settings':dict(settings,**({'fixed_repetitions':clean['frozen']['repetitions'][mid]} if campaign else {})),'block':block,'position':position,
                         'method_order':order,'scientific_hash':science_hash}
                if clock is not None or upstream is not None:
                    record=run_block(request,clock=clock or time.perf_counter_ns,upstream=upstream,in_process=True)
                else: record=_subprocess_block(request,output,env)
                if campaign:
                    if record['resource']['selected_execution_hash']!=clean['frozen']['selected_execution_hashes'][mid]: raise ValueError('campaign selected assets changed')
                    if content_hash(record['conditions'])!=clean['frozen']['condition_hash']: raise ValueError('campaign timing conditions changed')
                rows.extend(record['rows'])
                # Keep one full selection ledger per method in coordinator memory;
                # every complete worker record is still checkpointed below.
                resource_record=record['resource']
                if any(r['method_id']==mid for r in records):
                    resource_record={k:resource_record[k] for k in ('method_id','execution_hash','selected_execution_hash','memory')}
                records.append(resource_record); pilots.append(record['pilot']); conditions.append(record['conditions'])
                pending.append(record)
                if len(pending)>=settings['checkpoint_tasks']:
                    _atomic_json(output/'blocks'/f'{block:05d}-{position:03d}.json',pending); pending=[]
                if reserve is not None and record['resource']['memory']['peak_rss_bytes'] is not None and record['resource']['memory']['peak_rss_bytes']>reserve:
                    settings.update(memory_schedule(science,{k:settings[k] for k in operations},True)['operations'])
            if pending:
                _atomic_json(output/'blocks'/f'{block:05d}-complete.json',pending); pending=[]
    except Exception as exc:
        if pending: _atomic_json(output/'blocks'/'partial.json',pending)
        _atomic_json(output/'failure.json',{'status':'failed','reason':str(exc),'scientific_hash':science_hash,'completed_rows':len(rows)})
        raise
    resources=[]; summaries=[]; equality=[]
    for mid in [m.method_id for m in methods]:
        own=[r for r in records if r['method_id']==mid]; chosen=own[0]
        if len({r['execution_hash'] for r in own})!=1: raise ValueError('nonreproducible selected execution state')
        measured=[r['memory']['peak_rss_bytes'] for r in own if r['memory']['peak_rss_bytes'] is not None]
        chosen['memory']['peak_rss_bytes']=max(measured) if measured else None
        chosen['memory']['samples_bytes']=measured; resources.append(chosen); equality.extend(chosen['equality'])
        for stratum in sorted({r['timing_stratum'] for r in rows if r['method_id']==mid}):
            block_rows=[r for r in rows if r['method_id']==mid and r['timing_stratum']==stratum]
            timing={}
            for field in (*[s+'_ns' for s in STAGES],'end_to_end_ns','conditional_execution_ns','prepared_serialization_ns','adapted_serialization_ns','collapse_ns','raw_network_prediction_ns','literal_network_adaptation_ns'):
                values=[r[field] for r in block_rows if r.get(field) is not None]
                timing[field]=_interval(values) if values else {'mean':None,'low':None,'high':None,'reason':'outside measured boundary'}
            amortization={}; full={}
            for n in clean.get('future_tasks',[1,10,100]):
                prep=timing['preparation_ns']['mean']; adapt=timing['adaptation_ns']['mean']/len(inputs['contexts']) if not generated else timing['adaptation_ns']['mean']/clean.get('context_count',1)
                count=len(inputs['contexts']) if not generated else clean.get('context_count',1)
                predict=timing['prediction_ns']['mean']/count
                amortization[str(n)]=amortized_cost(prep,adapt,predict,n)
                if generated:
                    means={key:statistics.mean(r[key] for r in block_rows) for key in ('source_generation_ns','selection_ns','preparation_ns','prepared_serialization_ns','target_generation_ns','adaptation_ns','prediction_ns','adapted_serialization_ns')}
                    full[str(n)]=amortized_pipeline_cost(means,future_tasks=n,measured_tasks=count)
                elif upstream is not None:
                    full[str(n)]=amortized_cost(timing['selection_ns']['mean']+prep,adapt,predict,n)
            item={'method_id':mid,'timing_stratum':stratum,'timing':timing,'amortized_ns_per_task':amortization,
                  'amortization_scope':'conditional: prepare amortized, per-task adaptation/prediction; full: also source generation/selection/target generation/runtime serialization; excludes raw-network diagnostics and artifact publication'}
            if full: item['amortized_with_upstream_ns_per_task']=full
            summaries.append(item)
    by_id={r['method_id']:r for r in resources}
    for mid,row in by_id.items():
        if mid.startswith('B05:'):
            target=by_id.get(mid[4:])
            if target is None: raise ValueError('B05 requires present target')
            if row['selected_execution_hash']!=target['selected_execution_hash']: raise ValueError('B05 did not reproduce target selected asset/options')
    required={'B08:ridge','B10:empirical_bayes'} | {'B05:'+r['method_id'] for r in resources if r['trainable_parameters'] is not None and r['method_id']!='B08:gd'}
    missing=sorted(required-set(by_id))
    risk_available=all(r['risk']['status']=='measured' for r in resources)
    gate={'status':'pass' if risk_available and all(r['equality_status']=='pass' for r in resources) and clock is None and upstream is None else 'unavailable',
          'missing_recommended_controls':missing,'reason':'measured risk and executable equality checks required; full practical claim also requires ridge and EB controls',
          'literal_network_equality':equality}
    pareto=comparison_table(rows,resources,gate=gate,phase='finalized' if campaign else 'exploratory')
    dominates=[r for r in pareto['rows'] if r['status']=='scoped_domination']
    decision={'status':'scoped_measured_domination' if dominates and not missing else ('measured_tradeoff_or_unresolved' if gate['status']=='pass' else 'inconclusive'),
              'gate':'G9_measured_comparison' if campaign else 'G9_not_established','reason':'fixed operational campaign on recorded selected assets; conditional practical scope' if campaign else 'exploratory calibration; no finalized production schedule or protected practical claim',
              'scope':pareto['scope'],'domination_pairs':dominates}
    result={'schema_version':1,'status':'completed' if campaign else 'exploratory','phase':'campaign' if campaign else 'calibration','identity':identity,'inputs':inputs,'input_hash':content_hash(inputs),'scientific_hash':science_hash,
            'config':clean,'timing_blocks':rows,'resources':resources,'summaries':summaries,'pilot':pilots,'equality':equality,
            'comparison_gate':gate,'pareto':pareto,'decision':decision,'conditions':conditions[0], 'condition_strata':conditions,
            'operations':{'requested':operations,'effective':{k:settings[k] for k in operations},'memory_limit_bytes':reserve,
                          'available_memory_bytes':available_memory,'initial_headroom_bytes':reserve-baseline if reserve is not None else None,'observed_parent_peak_bytes':baseline,'pressure_detected':pressure,'scientific_hash_before':science_hash,'scientific_hash_after':content_hash(science),
                          'limit_policy':'soft pressure threshold reduces batching/checkpoint cadence only; a single indivisible source fit may exceed the reserve'},
            'schedule':{'requested_blocks':clean.get('blocks',6),'balanced_blocks':len(orders),'orders':orders},
            'prior_cost':{'status':'measured' if generated or upstream else 'unavailable','reason':'actual per-method source fitting/held-out selection' if generated else ('caller callable; provenance unverified' if upstream else 'supplied resident assets')},
            'context_generation':{'status':'measured' if generated else 'unavailable','elapsed_ns':sum(sample['generation_ns'] or 0 for row in rows for sample in row['repetition_samples'])},
            'energy':{'status':'not_run','joules':None,'reason':'no validated measurement; weight norm and elapsed time are not energy'},
            'accelerator':{'status':'not_run','reason':'CPU float64 core'},
            'timing_scope':{'end_to_end':'generation + exact selected source fitting + prepare + optimized adapt + predict + runtime serialization in one enclosing clock; unavailable for resident inputs',
                            'cold':'process cold numerical workload; excludes interpreter startup/imports/input decoding; OS file cache not flushed',
                            'conditional_execution':'prepare/adapt/predict/runtime serialization with supplied assets; upstream unavailable',
                            'raw_network':'literal adaptation (already includes collapse), repeated collapse and raw prediction separately; excluded from optimized path',
                            'optimization':'exact source-selected metrics reused for every eligible method; shared compact metric application for every fixed-map method; all posterior/ridge paths use common production small-space solvers; collapsed coefficient prediction for all',
                            'serialization':'UTF-8 execution payload in memory; evidence checkpoint disk I/O outside estimator timings'}}
    if campaign:
        if _allow_manifest_only(_benchmark_identity(clean),clean,manifest_path)!=identity: raise ValueError('identity changed during campaign')
        result['decision']['outcomes']={mid:[r for r in pareto['rows'] if r['method_id']==mid] for mid in by_id}
    _atomic_json(output/'benchmark.json',result)
    with (output/'timing_blocks.csv').open('w',newline='') as stream:
        names=sorted(set().union(*(r.keys() for r in rows))); writer=csv.DictWriter(stream,fieldnames=names); writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    return result
