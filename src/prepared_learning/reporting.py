"""Reconstruct scoped reports from sealed raw inventories, never summary caches.

Input descriptors are explicit: a confirmation needs its original protocol and
RunStore; a benchmark needs its raw worker directory; a diagnostic needs its raw
JSON and optional replay command/assets. An absent experiment is a not_run input,
not a missing file silently omitted from the scientific record.
"""
from collections.abc import Sequence
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .records import JsonRecord, GateDecision, GateCode, content_hash, to_json_record
from .runner import RunStore, IDENTITY_FIELDS


CHALLENGES = {
    'exact prior art': 'No historical priority claim; retain supplied source and literature scope.',
    'strongest estimator outside theorem class': 'Compare direct maps, tuned ridge and empirical Bayes; architectural savings do not imply operational superiority.',
    'scarce memory': 'Retain resource stops and measured RSS; report blocked scale without extending protected samples.',
    'critical operation failure': 'Preserve failure rows; block only affected claims.',
    'optional data or license unavailable': 'Natural arm remains not_run until an official acquisition/license receipt exists.',
    'synthetic success fails on natural covariates': 'No transfer claim without an independently executed natural-covariate arm.',
    'toy misses width requirements': 'Retain F/Z branches, width and trainable counts; exact rank differs from numerical truncation.',
    'gain vanishes after tuning': 'Report ties, reversals and strong selected controls; narrow to the architectural result.',
    'mean hides shifted-task regression': 'Retain every shift/family outcome and tail diagnostic; do not substitute the primary mean.',
    'interrupted resume inconsistency': 'Compare scientific keys, arrays, counts and hashes; reject changed identity.',
    'negative or inconclusive result': 'Keep negative and zero-crossing intervals and unsupported moments; no sample extension.',
    'hidden trainability of zeros': 'All matrix entries count as trainable, including initialized zeros.',
    'loss normalization': 'Use ||Xw-y||^2/(2n), eta=1, and squared coefficient error.',
    'basis degeneracy': 'Retain repeated-minimum/subspace diagnostics and basis-independent residuals.',
    'numerical versus exact rank': 'Separate floating support diagnostics from exact rational fixture deductions.',
    'covariance leakage': 'Source training/validation keys and evaluator-only truth remain separate.',
    'mutated initializer reuse': 'Fresh adaptation from the frozen initializer for each task; inspect raw hashes.',
    'full-rank head-zero miscompilation': 'Retain full-rank Z realization and actual trainable parameter counts.',
    'sequential updates': 'Require simultaneous literal updates and direct-map equality diagnostics.',
    'uncharged source-subspace oracle': 'Label information access and charge prior/metric storage; oracle evidence cannot support a practical learned claim.',
}
CONTRACT = ('architecture', 'population', 'baseline', 'metric', 'resource_contract')

# Admission bounds for this module's retained JSON ledgers, inverse snapshots,
# plot objects and serialization. They are deliberately much larger than the
# on-disk JSON. The guard uses stat/scandir, never a parsed raw ledger or a list
# of all paths. A local cap can only reduce the live half-available reserve.
REPORT_BASE_BYTES = 64*1024*1024
REPORT_JSON_FACTOR = 256
REPORT_PARSE_FACTOR = 64
REPORT_ARRAY_FACTOR = 16


class ReportingResourceLimit(RuntimeError):
    def __init__(self, evidence):
        self.evidence=evidence
        super().__init__('report/replay working set exceeds local reserve: '+str(evidence))


def _report_budget():
    from .pilot import _available_memory
    available=max(0,int(_available_memory()))
    declared=os.environ.get('PREPARED_REPORT_MEMORY_BYTES')
    cap=None if declared is None else int(declared)
    if cap is not None and cap<=0: raise ValueError('PREPARED_REPORT_MEMORY_BYTES must be a positive integer')
    return available,min(available//2,cap) if cap is not None else available//2


def _resource_check(scale, *, phase, copies=1, object_bytes=0, scalar_bytes=0, json_factor=REPORT_JSON_FACTOR):
    available,budget=_report_budget()
    estimate=(REPORT_BASE_BYTES + copies*(json_factor*scale['json_working_bytes']+scale['inventory_bytes'])
              + REPORT_ARRAY_FACTOR*scale['max_array_bytes'] + 16*object_bytes + scalar_bytes)
    evidence={**scale,'phase':phase,'available_memory_bytes':available,'reserve_fraction':.5,
              'budget_bytes':budget,'estimated_peak_bytes':estimate,'object_bytes':object_bytes,
              'scalar_and_key_working_bytes':scalar_bytes,'retained_copies':copies,
              'policy':{'base_bytes':REPORT_BASE_BYTES,'json_expansion_factor':json_factor,
                        'array_workspace_factor':REPORT_ARRAY_FACTOR,'inventory_entry_bytes':4096},
              'scientific_counts':'not inferred from file counts; all raw observations remain unchanged',
              'recovery':'retry the identical inputs with sufficient local reserve; no truncation, sampling or changed scientific identity'}
    if estimate>budget: raise ReportingResourceLimit(evidence)
    return evidence


def _empty_scale():
    return {'files_scanned':0,'json_bytes':0,'max_json_bytes':0,'json_working_bytes':0,'array_bytes':0,'max_array_bytes':0,
            'inventory_bytes':0,'largest_file_bytes':0,'largest_file':None,'inventory_complete':True}


def _guard_paths(paths, *, phase, copies=1, object_bytes=0, streaming=False, json_factor=REPORT_JSON_FACTOR):
    scale=_empty_scale()
    for value in paths:
        path=Path(value); size=path.stat().st_size
        scale['files_scanned']+=1
        scale['inventory_bytes']+=4096+16*len(os.fsencode(path))
        if path.suffix in ('.json','.jsonl'):
            scale['json_bytes']+=size
            scale['max_json_bytes']=max(scale['max_json_bytes'],size)
        elif path.suffix=='.npy':
            scale['array_bytes']+=size; scale['max_array_bytes']=max(scale['max_array_bytes'],size)
        if size>scale['largest_file_bytes']:
            scale['largest_file_bytes']=size; scale['largest_file']=str(path)
    scale['json_working_bytes']=scale['max_json_bytes'] if streaming else scale['json_bytes']
    return _resource_check(scale,phase=phase,copies=copies,object_bytes=object_bytes,json_factor=json_factor)


def _walk_files(root, depth=0):
    """A bounded-depth scandir walk; Path.rglob/os.walk retain path lists."""
    root=Path(root)
    if not root.exists(): return
    if depth>32: raise ValueError('raw inventory exceeds supported directory depth')
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False): yield from _walk_files(entry.path,depth+1)
            elif entry.is_file(): yield Path(entry.path)


def _store_paths(root):
    root=Path(root)
    for name in ('run.json','run.sha256','initial_run.json','terminal.json'):
        if (root/name).exists(): yield root/name
    for folder in ('blocks','journal'): yield from _walk_files(root/folder)


def _input_paths(entry):
    yield Path(entry['descriptor_path'])
    kind=entry['kind']
    if kind=='confirmation':
        yield from _store_paths(entry['run_path'])
        yield Path(entry['protocol_path'])
    elif kind=='benchmark':
        root=Path(entry['run_path']); yield root/'inputs.json'
        for folder in ('blocks','requests'): yield from _walk_files(root/folder)
        if (root/'failure.json').exists(): yield root/'failure.json'
        # This bounded control JSON locates a second raw inventory. Its own
        # read guard runs before parsing; no worker ledger is read here.
        config=_read(root/'inputs.json')['inputs'].get('manifest',{})
        if config.get('phase')=='finalized':
            calibration=Path(config['frozen']['calibration_path'])
            yield calibration; yield calibration.parent/'inputs.json'
            for folder in ('blocks','requests'): yield from _walk_files(calibration.parent/folder)
    elif kind!='not_run': yield Path(entry['raw_path'])
    for asset in entry.get('assets',[]): yield Path(asset)


def _object_bytes(value):
    """Count the supplied JSON tree without making a serialization/copy first."""
    import sys
    size=sys.getsizeof(value)
    if isinstance(value,dict):
        for key,child in value.items(): size+=sys.getsizeof(key)+_object_bytes(child)
    elif isinstance(value,(list,tuple)):
        for child in value: size+=_object_bytes(child)
    return size


def _declared_count(value):
    if value is None: return 0
    if type(value)!=int or value<0: raise ValueError('nonnegative integer scientific count required for report admission')
    return value


def _guard_inputs(manifest, *, replay=False):
    scale=_empty_scale(); largest_streamed=0; retained_json=0; scalars=0
    for entry in manifest['inputs']:
        streamed=entry['kind']=='confirmation'
        measured=_guard_paths(_input_paths(entry),phase='input_before_store_open',streaming=streamed)
        for key in ('files_scanned','json_bytes','array_bytes','inventory_bytes'): scale[key]+=measured[key]
        for key in ('max_json_bytes','max_array_bytes'): scale[key]=max(scale[key],measured[key])
        if measured['largest_file_bytes']>scale['largest_file_bytes']:
            scale['largest_file_bytes']=measured['largest_file_bytes']; scale['largest_file']=measured['largest_file']
        if streamed:
            largest_streamed=max(largest_streamed,measured['max_json_bytes'])
            if not replay:
                scientific=_read(entry['protocol_path'])['scientific']; plan=scientific['config']['sample_plan']
                j=_declared_count(plan['outer_replicates']) or 1
                tasks=j*_declared_count(plan['tasks_per_replicate'])
                for name in ('update_check_schedule','simulation_schedule'):
                    schedule=scientific.get(name,{})
                    tasks+=len(schedule.get('outer_replications',[]))*_declared_count(schedule.get('tasks_per_replication',0))
                keys=j*(_declared_count(plan['source_training_tasks'])+_declared_count(plan['source_validation_tasks']))
                # Scalar task dictionaries, paired roles/family losses, source
                # and task key sets, completion indices and inference buffers.
                scalars+=8192*tasks+2048*tasks*len(scientific['secondary_family']['contrasts'])+512*keys+8192*j
        else: retained_json+=measured['json_bytes']
    scale['json_working_bytes']=largest_streamed+retained_json
    return _resource_check(scale,phase='replay_inputs' if replay else 'report_inputs',
                           copies=2 if replay else 1,object_bytes=_object_bytes(manifest),scalar_bytes=scalars)


def resource_limit_decision(exc, output):
    output=Path(output)
    evidence={**exc.evidence,'report_status':'not_completed',
              'partial_artifacts':'any already streamed output is incomplete; original raw evidence and scientific counts are unchanged'}
    decision=GateDecision(GateCode.LOCAL_RESOURCE_LIMIT,evidence,
        'Report/replay allocation was refused before loading the offending raw working set.',
        exc.evidence['recovery'])
    _write(output/'resource_limit.json',evidence); _write(output/'gate_decision.json',decision)
    return decision


def _read(path):
    # Parsing a native JSON tree does not allocate analysis/plot workspaces.
    # In particular, inventory manifests grow with J but contain mostly paths
    # and hashes. Raw semantic consumers have the separate 256x admission above.
    _guard_paths((Path(path),),phase='json_before_parse',json_factor=REPORT_PARSE_FACTOR)
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Native JSON ledgers stay shared: no whole-tree to_json_record copy and no
    # complete encoded string in addition to the retained scientific records.
    with path.open('w') as stream:
        json.dump(value,stream,default=to_json_record,indent=2,allow_nan=False)
        stream.write('\n')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def _path(value, base):
    path = Path(value)
    return (base/path).resolve() if not path.is_absolute() else path.resolve()


def _raw_files(entry):
    """Canonical inventory deliberately excludes rebuildable summaries."""
    _guard_paths(_input_paths(entry),phase='raw_inventory_before_store_open',streaming=entry['kind']=='confirmation')
    files = {Path(entry['descriptor_path'])}
    kind = entry['kind']
    if kind == 'confirmation':
        root = Path(entry['run_path']); run = _read(root/'run.json')
        store = RunStore.open(root, run)
        store.committed_keys()  # Validate complete lifecycle and every committed asset.
        files.update(root/name for name in ('run.json', 'run.sha256', 'initial_run.json', 'terminal.json') if (root/name).exists())
        files.update(p for folder in ('blocks','journal') for p in (root/folder).rglob('*') if p.is_file())
        files.add(Path(entry['protocol_path']))
    elif kind == 'benchmark':
        root = Path(entry['run_path']); files.add(root/'inputs.json')
        files.update(p for folder in ('blocks','requests') for p in (root/folder).glob('*.json'))
        if (root/'failure.json').exists(): files.add(root/'failure.json')
        if not (root/'blocks').exists(): raise ValueError('benchmark raw blocks missing')
        config=_read(root/'inputs.json')['inputs'].get('manifest',{})
        if config.get('phase')=='finalized':
            calibration=Path(config['frozen']['calibration_path'])
            files.add(calibration); files.add(calibration.parent/'inputs.json')
            files.update(p for folder in ('blocks','requests') for p in (calibration.parent/folder).glob('*.json'))
    elif kind != 'not_run':
        files.add(Path(entry['raw_path']))
    files.update(Path(p) for p in entry.get('assets', []))
    return {str(p): _sha(p) for p in sorted(files)}


def build_report_inputs(run_paths: Sequence[Path]) -> JsonRecord:
    """Freeze the supplied order and exact raw inventory of descriptor files.

    Paths inside descriptors are relative to the descriptor's directory. The
    returned manifest uses absolute recovery paths and can be archived as-is on
    the recorded machine. Additional assets must be explicitly enumerated.
    """
    # Even descriptor/control JSON is admitted before parsing or fingerprinting
    # the live source. The aggregate guard precedes every RunStore.open below.
    _guard_paths(run_paths,phase='descriptors_before_parse')
    entries=[]
    for index, value in enumerate(run_paths):
        path = Path(value).resolve(); descriptor = _read(path)
        kind = descriptor.get('kind')
        if kind not in ('confirmation','benchmark','diagnostic','source','exact','forecast','not_run'):
            raise ValueError('explicit supported input descriptor kind required')
        entry = {**descriptor, 'input_id': index, 'descriptor_path': str(path),
                 'label': descriptor.get('label', path.stem)}
        for name in ('run_path','protocol_path','raw_path'):
            if name in entry: entry[name] = str(_path(entry[name], path.parent))
        entry['assets'] = [str(_path(p, path.parent)) for p in descriptor.get('assets', [])]
        if kind == 'not_run' and not entry.get('reason'): raise ValueError('not_run requires a reason')
        entries.append(entry)
    _guard_inputs({'inputs':entries})
    from .protocol import checkout_identity
    root = Path(__file__).resolve().parents[2]
    versions, dirty = checkout_identity({}, {})
    inputs=[]; scientific_runs=set()
    for entry in entries:
        kind=entry['kind']
        entry['inventory'] = _raw_files(entry)
        if kind == 'confirmation':
            protocol = _read(entry['protocol_path']); run = _read(Path(entry['run_path'])/'run.json')
            entry['source'] = protocol['scientific']['versions']
            entry['status'] = run['status']
            entry['evidence_level'] = 'fixture' if entry.get('fixture') else 'protected'
            identity=tuple(run[k] for k in IDENTITY_FIELDS)
            if identity in scientific_runs: raise ValueError('duplicate scientific run: technical replay is verification, not independent samples')
            scientific_runs.add(identity)
        elif kind == 'benchmark':
            config=_read(Path(entry['run_path'])/'inputs.json')['inputs'].get('manifest',{})
            entry.setdefault('source',config.get('frozen',{}).get('identity',{}).get('versions',versions))
            entry['evidence_level']='exploratory'
            entry.setdefault('status','recorded')
        else:
            entry.setdefault('source', versions)
            entry.setdefault('status', 'not_run' if kind == 'not_run' else 'recorded')
            entry['evidence_level'] = {'source':'source_record','forecast':'forecast','not_run':'not_run'}.get(kind, 'numerical_diagnostic')
        inputs.append(entry)
    if not inputs: raise ValueError('at least one explicit input required')
    source = next((r['source'] for r in inputs if r['kind']=='confirmation'), versions)
    return {'schema_version':1, 'inputs':inputs, 'claims':[], 'challenges':{},
            'replay':{'source_root':str(root), 'source':source, 'builder_dirty_sha256':dirty,
                      'wheelhouse':str(root/'wheelhouse'), 'rtol':1e-10, 'atol':1e-12,
                      'timing_policy':'remeasured_not_equal', 'receipt':None},
            'input_order_sha256':content_hash(inputs)}


def _verify_inputs(manifest, *, replay=False):
    if manifest.get('schema_version') != 1 or not manifest.get('inputs'): raise ValueError('invalid report manifest')
    preflight=_guard_inputs(manifest,replay=replay)
    if content_hash(manifest['inputs']) != manifest.get('input_order_sha256'): raise ValueError('input manifest identity changed')
    scientific_runs=set()
    for index, entry in enumerate(manifest['inputs']):
        if entry['input_id'] != index: raise ValueError('input order changed')
        if _raw_files(entry) != entry['inventory']: raise ValueError('raw inventory hash changed')
        if entry['kind']=='confirmation':
            run=_read(Path(entry['run_path'])/'run.json'); identity=tuple(run[k] for k in IDENTITY_FIELDS)
            if identity in scientific_runs: raise ValueError('duplicate scientific run cannot add independent samples')
            scientific_runs.add(identity)
    return preflight


def _contract(science):
    config = science['config']
    return {'architecture':{'class':'all-entry deep linear, zero-output initialization', **config['budget']},
            'population':{'workload':config['workload'], 'preparation':config['preparation'], 'access':config['access']},
            'baseline':science['selectors'], 'metric':'expected squared coefficient error [coefficient^2]',
            'resource_contract':{'budget':config['budget'], 'sample_plan':config['sample_plan'],
                                 'prior_assets':science['assets'], 'information_access':config['access']}}


def _outcome(analysis):
    if analysis.get('mean') is None: return 'unavailable'
    if analysis.get('comparison_kind')=='network_direct_equality_control': return 'tie_equality_control'
    if analysis['mean'] == 0: return 'tie'
    interval = analysis.get('interval')
    if interval and interval[0] <= 0 <= interval[1]: return 'zero_crossing_inconclusive'
    if analysis['mean'] < 0: return 'reversal_negative'
    if interval is None: return 'positive_descriptive_or_analytic_no_superiority_CI'
    return 'positive_in_scope' if interval[0] > 0 else 'inconclusive'


def _confirmation(entry, emit=lambda row:None):
    from .confirmation import run_record, analyze_results
    from .protocol import validate_frozen_protocol
    from .records import TaskKey
    from .tasks import CONTEXT, NOISE
    _guard_inputs({'inputs':[entry]})
    protocol = _read(entry['protocol_path']); science = protocol['scientific']
    if not entry.get('fixture'): validate_frozen_protocol(protocol)
    run = _read(Path(entry['run_path'])/'run.json')
    expected = run_record(protocol)
    if any(run[k] != expected[k] for k in IDENTITY_FIELDS): raise ValueError('protocol/run identity mismatch')
    if run['input_manifest_content_sha256'] != content_hash(protocol): raise ValueError('scientific input manifest changed')
    if run['phase'] != 'confirmation' or run['dirty_diff_sha256'] is not None:
        raise ValueError('dirty or exploratory store cannot become protected evidence')
    store = RunStore.open(Path(entry['run_path']), expected)
    plan=science['config']['sample_plan']; outer_limit=_declared_count(plan['outer_replicates']) or 1
    source_limit=_declared_count(plan['source_training_tasks'])+_declared_count(plan['source_validation_tasks'])
    primary_limit=_declared_count(plan['tasks_per_replicate'])
    schedules={endpoint:(set(science.get(name,{}).get('outer_replications',[])),
                        _declared_count(science.get(name,{}).get('tasks_per_replication',0)))
               for endpoint,name in (('update_check','update_check_schedule'),('secondary','simulation_schedule'))}
    def endpoint_limit(endpoint,outer):
        if endpoint=='primary': return primary_limit
        selected,count=schedules[endpoint]
        return count if outer in selected else 0
    executions={}; tasks={}; completions=[]; raw_count=0
    for path in sorted((store.path/'blocks').iterdir()):
        store._verify(path)
        rows=_read(path/'records.json'); store._validate_rows(rows)
        for row in rows:
            emit(row); raw_count+=1
            if row['kind']=='execution':
                result=row['result']; outer=result['outer_replication']
                if type(outer)!=int or not 0<=outer<outer_limit or len(result['source_keys'])>source_limit:
                    raise ValueError('preparation exceeds declared reporting working set')
                if outer in executions: raise ValueError('duplicate preparations')
                executions[outer]={key:result[key] for key in ('outer_replication','status','source_keys')}
                executions[outer]['failures']={'raw_failure_recorded':True} if result['failures'] else {}
                executions[outer]['risks']={role:{'conditional_risk':value.get('conditional_risk')} for role,value in result['risks'].items() if role in ('baseline','joint')}
                executions[outer]['tasks']=[]
            elif row['kind']=='replication_completion' and row['status']=='completed':
                if len(completions)>=outer_limit: raise ValueError('completion count exceeds frozen schedule')
                completions.append(row['outer_replication'])
            elif row['kind']=='execution_task':
                task=row['result']; keys=protocol['protected_keys']
                outer=task['outer_replication']; endpoint=task['endpoint']; task_id=task['task_id']
                if type(outer)!=int or not 0<=outer<outer_limit or endpoint not in ('primary','update_check','secondary'):
                    raise ValueError('task exceeds declared reporting working set')
                if type(task_id)!=int or not 0<=task_id<endpoint_limit(endpoint,outer):
                    raise ValueError('task exceeds declared endpoint count')
                maximum=sum(endpoint_limit(e,outer) for e in ('primary','update_check','secondary'))
                if len(tasks.get(outer,()))>=maximum: raise ValueError('duplicate task count exceeds frozen schedule')
                phase=keys['test_phase']+{'primary':0,'update_check':100,'secondary':200}[task['endpoint']]
                expected_key=content_hash({'key':TaskKey(keys['root_seed'],phase,task['outer_replication'],task['task_id'],0),
                                          'context_stream':CONTEXT,'noise_stream':NOISE})
                if task['task_key']!=expected_key: raise ValueError('scientific task key changed')
                reduced={key:task[key] for key in ('endpoint','task_id','task_key')}
                roles=('baseline','joint',*('family:'+mid for mid in science['secondary_family']['contrasts']))
                reduced['values']={role:{key:value[key] for key in ('loss','status') if key in value} for role,value in task['values'].items() if role in roles}
                tasks.setdefault(task['outer_replication'],[]).append(reduced)
    for outer,values in tasks.items():
        if outer not in executions: raise ValueError('task without preparation')
        executions[outer]['tasks']=values
    for result in executions.values(): result['tasks'].sort(key=lambda t:(t['endpoint'],t['task_id']))
    try:
        if run['status'] != 'completed' or sorted(completions) != sorted(executions):
            raise ValueError('uncompleted or failed run; no success-only inference')
        analysis = analyze_results(list(executions.values()), science, protocol['protected_keys']['root_seed'])
    except ValueError as exc:
        analysis = {'status':'invalid_or_incomplete', 'reason':str(exc), 'mean':None, 'interval':None,
                    'n':None, 'secondary':{}, 'independent_unit':None}
    analysis.update(input_id=entry['input_id'], label=entry['label'], outcome=_outcome(analysis),
                    evidence_level=entry['evidence_level'], run_status=run['status'], **_contract(science))
    return analysis, raw_count


def _benchmark(entry):
    """Recompute paired resource comparisons from worker blocks, not benchmark.json."""
    from .benchmark import comparison_table, _campaign_configuration, _fixed_schedule
    _guard_paths(_input_paths(entry),phase='benchmark_before_raw_allocation')
    root = Path(entry['run_path']); inputs = _read(root/'inputs.json')
    rows=[]; workers=[]; seen=set(); resources={}
    for path in sorted((root/'blocks').glob('*.json')):
        for worker in _read(path):
            resource = worker['resource']; mid=resource['method_id']
            for row in worker['rows']:
                key=(mid,row['block'],row['timing_stratum'])
                if key in seen: raise ValueError('duplicate benchmark worker stratum')
                seen.add(key); rows.append(row)
            if mid in resources:
                old=resources[mid]
                if resource['selected_execution_hash'] != old['selected_execution_hash']: raise ValueError('benchmark selected asset changed')
                a,b=old['memory']['peak_rss_bytes'],resource['memory']['peak_rss_bytes']
                old['memory']['peak_rss_bytes']=max(a,b) if a is not None and b is not None else None
            else: resources[mid]=json.loads(json.dumps(resource))
            workers.append(worker)
    generated=inputs['inputs'].get('manifest', {})
    campaign=generated.get('phase')=='finalized'
    declared=[d['method_id'] for d in inputs['methods']]
    orders=generated.get('frozen',{}).get('orders', [])
    complete=bool(workers) and set(resources)==set(declared) and not (root/'failure.json').exists()
    if campaign:
        frozen=generated['frozen']
        if frozen['identity']['dirty_diff_sha256'] is not None or frozen['identity']['versions']!=entry['source']:
            raise ValueError('benchmark requires clean recorded source identity')
        if content_hash(_campaign_configuration(generated))!=frozen['configuration_sha256']:
            raise ValueError('benchmark frozen scientific inputs changed')
        calibration_path=Path(frozen['calibration_path'])
        if content_hash(_read(calibration_path))!=frozen['calibration_sha256']:
            raise ValueError('benchmark calibration identity changed')
        calibration_inputs=_read(calibration_path.parent/'inputs.json')
        calibration_config=calibration_inputs['inputs']['manifest']
        if _campaign_configuration(calibration_config)!=_campaign_configuration(generated):
            raise ValueError('calibration/campaign scientific contract changed')
        pilots=[]; calibration_blocks=set()
        for path in sorted((calibration_path.parent/'blocks').glob('*.json')):
            for worker in _read(path):
                pilots.append(worker['pilot']); calibration_blocks.update(r['block'] for r in worker['rows'])
        rebuilt=_fixed_schedule({'config':calibration_config,'pilot':pilots,
                                 'schedule':{'balanced_blocks':len(calibration_blocks)}})
        if any(rebuilt[k]!=frozen[k] for k in rebuilt): raise ValueError('frozen timing schedule differs from raw calibration')
        for worker in workers:
            resource=worker['resource']
            if resource['selected_execution_hash']!=frozen['selected_execution_hashes'][resource['method_id']]:
                raise ValueError('benchmark selected execution identity changed')
            if content_hash(worker['conditions'])!=frozen['condition_hash']:
                raise ValueError('benchmark timing conditions changed')
        expected={(mid,block,stratum) for block,order in enumerate(orders) for mid in order for stratum in ('cold','warm')}
        complete &= seen==expected
    required={'B08:ridge','B10:empirical_bayes'} | {'B05:'+mid for mid,r in resources.items() if r['trainable_parameters'] is not None and mid!='B08:gd'}
    missing=sorted(required-set(resources))
    equal=all(r['equality_status']=='pass' for r in resources.values())
    gate={'status':'pass' if complete and equal else 'unavailable'}
    pareto=comparison_table(rows,list(resources.values()),gate=gate,phase='finalized' if campaign else 'exploratory')
    # Keep the complete candidate ledgers and failures in raw output, even when
    # they were not selected, and never count warm repeats as fresh risk samples.
    return {'input_id':entry['input_id'], 'label':entry['label'], 'status':'completed' if complete else 'failed_or_incomplete',
            'evidence_level':'fixture' if entry.get('fixture') else ('operational' if campaign and complete and not missing and equal else 'exploratory'),
            'missing_controls':missing, 'rows':rows, 'resources':list(resources.values()), 'pareto':pareto,
            'inputs':inputs, 'raw_workers':workers, 'energy':{'status':'not_run'}, 'accelerator':{'status':'not_run'}}


def _benchmark_contract(benchmark):
    from ._benchmark_worker import generated_config
    config=benchmark['inputs']['inputs'].get('manifest')
    if not config: raise ValueError('operational claim needs generated scientific contract')
    c=to_json_record(generated_config(config))
    return {'architecture':{'class':'declared deep-linear and outside-class operational controls','budget':c['budget']},
            'population':{'workload':c['workload'],'preparation':c['preparation'],'access':c['access']},
            'baseline':c['methods'],'metric':'expected squared coefficient error [coefficient^2]',
            'resource_contract':{'conditions':benchmark['raw_workers'][0]['conditions'],
                'sample_plan':c['sample_plan'],'scope':benchmark['pareto']['scope'],
                'timing':'measured enclosing pipeline nanoseconds; preparation and selection included',
                'memory':'peak process RSS bytes; not incremental workspace',
                'storage':'runtime serialized bytes; evidence storage separate'}}


def _adverse_records(value, path='raw'):
    """Retain nested failed candidates as well as top-level outcome rows."""
    result=[]
    if isinstance(value,dict):
        status=value.get('status')
        if isinstance(status,str) and any(word in status for word in ('fail','invalid','unsupported','unavailable','not_run','incomplete')):
            result.append({'path':path,'record':value})
        for key,child in value.items(): result.extend(_adverse_records(child,path+'.'+key))
    elif isinstance(value,list):
        for index,child in enumerate(value): result.extend(_adverse_records(child,path+f'[{index}]'))
    return result


def _csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with Path(path).open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True,allow_nan=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()} for row in rows)


class _RecordStream:
    """Write complete raw/adverse JSON arrays one verified record at a time."""
    def __init__(self,path):
        self.path=Path(path); self.stream=None; self.count=0
    def __enter__(self):
        self.stream=self.path.open('w'); self.stream.write('[\n'); return self
    def write(self,record):
        if self.count: self.stream.write(',\n')
        json.dump(record,self.stream,default=to_json_record,allow_nan=False)
        self.count+=1
    def __exit__(self,*error):
        self.stream.write('\n]\n'); self.stream.close()


def _figures(output, comparisons, benchmarks, diagnostics):
    # No implicit ~/.matplotlib write, including when a caller set MPLCONFIGDIR.
    import logging
    with tempfile.TemporaryDirectory(prefix='prepared-plots-') as config:
        prior=os.environ.get('MPLCONFIGDIR'); os.environ['MPLCONFIGDIR']=config
        logger=logging.getLogger('matplotlib.font_manager')
        def expected_cache_notice(record):
            return not (record.levelno==logging.WARNING and record.getMessage()==
                        'Matplotlib is building the font cache; this may take a moment.')
        logger.addFilter(expected_cache_notice)
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            from matplotlib import pyplot as plt
            figures=[]
            specifications=[]
            risks=[{'label':r['label'],'value':r['mean'],'low':r.get('interval',[None,None])[0] if r.get('interval') else None,
                    'high':r['interval'][1] if r.get('interval') else None, 'independent_n':r.get('n'),
                    'independent_unit':r.get('independent_unit'), 'outcome':r['outcome'],
                    'evaluation':'analytic' if r.get('analysis_mode') in ('conditional_risk_over_preparations','deterministic_frontier') else 'sampled'} for r in comparisons if r['mean'] is not None]
            specifications.append(('comparisons','Comparison [declared population]','Baseline minus joint risk [coefficient²]','population','coefficient^2',risks))
            frontier=[]; resources=[]; pareto=[]
            for item in diagnostics:
                raw=item['raw']; points=raw if isinstance(raw,list) else [raw]
                metadata=raw.get('metadata',{}) if isinstance(raw,dict) else {}
                if isinstance(raw,dict): points=raw.get('records',raw.get('rows',points))
                for point in points:
                    if not isinstance(point,dict): continue
                    if not any(branch in point for branch in ('F','Z')) and not (metadata and point.get('kind')=='interval'): continue
                    common={'input_id':item['input_id'],'record_kind':point.get('kind','point'),
                        'width':point.get('width'),'budget':point.get('norm_cap',point.get('caps',{}).get('initialization_norm_cap')),
                        'point_status':point.get('status','unresolved' if point.get('kind')=='interval' else 'recorded'),
                        'error':point.get('error',point.get('reason')),'selection_status':point.get('selection_status'),
                        'selected_branches':point.get('selected_branches'),'grid_metadata':metadata,
                        'interval_lower':point.get('lower'),'interval_upper':point.get('upper'),
                        'evaluation':'analytic','independent_n':None,'evidence_level':'float64_diagnostic'}
                    if point.get('kind')=='interval':
                        frontier.append({**common,'label':item['label']+'/unresolved interval','branch':None,'value':None,
                                         'branch_status':None,'actual_resources':None,'diagnostics':point})
                        continue
                    for branch in ('F','Z'):
                        row=point.get(branch)
                        frontier.append({**common,'label':f"{item['label']}/{branch}, width={common['width']}, cap={common['budget']}",
                            'branch':branch,'value':row.get('objective') if isinstance(row,dict) else None,
                            'branch_status':row.get('status') if isinstance(row,dict) else 'unavailable',
                            'actual_resources':row.get('actual_resources') if isinstance(row,dict) else None,
                            'diagnostics':row.get('diagnostics') if isinstance(row,dict) else None})
            specifications.append(('frontier','Branch [declared norm cap]','Expected risk [coefficient²]','branch/norm cap','coefficient^2',frontier))
            for benchmark in benchmarks:
                for r in benchmark['resources']:
                    resources.append({'label':benchmark['label']+'/'+r['method_id'],'value':r['runtime_serialized_bytes'],
                        'evaluation':'sampled','independent_n':len({t['block'] for t in benchmark['rows'] if t['method_id']==r['method_id']}),
                        'independent_unit':'timing block','peak_rss_bytes':r['memory']['peak_rss_bytes'],
                        'risk_independent_n':len(set(r['risk'].get('task_keys',[]))), 'risk':r['risk']})
                for r in benchmark['pareto']['rows']:
                    timing=r['paired_time_difference_ns']; risk=r['paired_risk_difference']
                    if timing and risk:
                        pareto.append({'label':r['method_id']+' minus '+r['comparator']+'/'+r['timing_stratum'],
                            'x':timing['mean'],'value':risk['mean'],'evaluation':'mixed',
                            'independent_n':timing.get('blocks'), 'independent_unit':'paired timing block',
                            'status':r['status'], 'risk_interval':risk,'timing_interval':timing,
                            'risk_independent_n':{resource['method_id']:len(set(resource['risk'].get('task_keys',[])))
                                for resource in benchmark['resources'] if resource['method_id'] in (r['method_id'],r['comparator'])}})
            specifications += [('resources','Method [execution]','Runtime serialized storage [bytes]','method','bytes',resources),
                               ('pareto','Paired elapsed-time difference [ns]','Paired risk difference [coefficient²]','ns','coefficient^2',pareto)]
            for name,xlabel,ylabel,xunit,yunit,rows in specifications:
                _csv(output/(name+'.csv'), rows)
                fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
                plotted=[r for r in rows if r['value'] is not None]
                if plotted:
                    x=[r.get('x',i) for i,r in enumerate(plotted)]; y=[r['value'] for r in plotted]
                    ax.scatter(x,y)
                    if name!='pareto': ax.set_xticks(x,[r['label'] for r in plotted],rotation=25,ha='right')
                    for xx,row in zip(x,plotted):
                        if row.get('low') is not None: ax.plot([xx,xx],[row['low'],row['high']],color='black')
                    ax.set_title('; '.join(f"{r['evaluation']}, independent n={r['independent_n']}" for r in rows[:4]))
                    ax.axhline(0,color='grey',linewidth=.7)
                elif rows: ax.text(.5,.5,'Executed records; objectives unavailable (see CSV)',ha='center',transform=ax.transAxes)
                else: ax.text(.5,.5,'not_run / no applicable verified observations',ha='center',transform=ax.transAxes)
                if rows and len(plotted)<len(rows):
                    ax.text(.01,.99,f'{len(rows)-len(plotted)} unavailable branch/interval records retained in CSV',
                            va='top',fontsize=8,transform=ax.transAxes)
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
                fig.savefig(output/(name+'.svg')); fig.savefig(output/(name+'.pdf')); plt.close(fig)
                labels={r['evaluation'] for r in rows}
                figures.append({'path':name+'.svg','pdf':name+'.pdf','csv':name+'.csv',
                    'x_unit':xunit,'y_unit':yunit,'evaluation':next(iter(labels)) if len(labels)==1 else ('mixed' if rows else 'unavailable'),
                    'independent_n':[r['independent_n'] for r in rows], 'renderer':'matplotlib'})
            return figures
        finally:
            logger.removeFilter(expected_cache_notice)
            if prior is None: os.environ.pop('MPLCONFIGDIR',None)
            else: os.environ['MPLCONFIGDIR']=prior


def regenerate_tables(manifest: JsonRecord, output: Path) -> JsonRecord:
    preflight=_verify_inputs(manifest)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    comparisons=[]; benchmarks=[]; diagnostics=[]; statuses=[]
    with _RecordStream(output/'raw_records.json') as raw, _RecordStream(output/'adverse_records.json') as adverse, (output/'raw_records.csv').open('w',newline='') as raw_csv:
        writer=csv.DictWriter(raw_csv,fieldnames=['input_id','kind','outer_replication','task_id','method_id','status','record_json'])
        writer.writeheader()
        for entry in manifest['inputs']:
            kind=entry['kind']
            if kind=='confirmation':
                def emit(row):
                    record={'input_id':entry['input_id'],**row}; raw.write(record)
                    writer.writerow({**{key:record.get(key) for key in writer.fieldnames if key!='record_json'},
                                     'record_json':json.dumps(record,allow_nan=False)})
                    for failed in _adverse_records(record): adverse.write(failed)
                comparison,_=_confirmation(entry,emit); comparisons.append(comparison)
                status=comparison['run_status']
            elif kind=='benchmark':
                benchmark=_benchmark(entry); benchmarks.append(benchmark); status=benchmark['status']
                for failed in _adverse_records(benchmark): adverse.write(failed)
            elif kind=='not_run': status='not_run'
            else:
                record=_read(entry['raw_path']); diagnostics.append({'input_id':entry['input_id'],'label':entry['label'],
                    'evidence_level':entry['evidence_level'],'raw':record})
                for failed in _adverse_records(diagnostics[-1]): adverse.write(failed)
                status=record.get('status','recorded') if isinstance(record,dict) else 'recorded'
            statuses.append({'input_id':entry['input_id'],'label':entry['label'],'kind':kind,'status':status,
                             'reason':entry.get('reason'), 'evidence_paths':list(entry['inventory'])})
        raw_count=raw.count; adverse_count=adverse.count
    _csv(output/'comparisons_table.csv', comparisons)
    _csv(output/'input_status.csv', statuses)
    _csv(output/'family_comparisons.csv', [{'input_id':r['input_id'],'method':mid,**value} for r in comparisons for mid,value in r.get('secondary',{}).items()])
    _csv(output/'pareto_table.csv', [{'input_id':b['input_id'],**row} for b in benchmarks for row in b['pareto']['rows']])
    _write(output/'diagnostics.json',diagnostics)
    _write(output/'benchmark_workers.json',benchmarks)
    figures=_figures(output,comparisons,benchmarks,diagnostics)
    result={'comparisons':comparisons,'benchmarks':benchmarks,'diagnostics':diagnostics,'statuses':statuses,'figures':figures,
            'adverse_records':{'path':'adverse_records.json','count':adverse_count,'storage':'complete streamed records'},
            'raw_record_count':raw_count,'resource_preflight':preflight}
    _write(output/'tables.json', result)
    return result


def _claims(manifest, tables):
    comparisons={r['input_id']:r for r in tables['comparisons']}
    benchmarks={r['input_id']:r for r in tables['benchmarks']}
    unavailable={r['input_id'] for r in tables.get('statuses',[]) if r['status'] in
        ('failed','invalid','incomplete','running','interrupted','failed_or_incomplete',
         'invalid_or_incomplete','unavailable','unsupported','not_run','unsuccessful_source_diagnostic')}
    unavailable.update(i for i,r in comparisons.items() if r['run_status']!='completed' or r['mean'] is None
                       or r['status']=='invalid_or_incomplete')
    unavailable.update(i for i,r in benchmarks.items() if r.get('status')!='completed')
    ledger=[]
    for row in comparisons.values():
        ledger.append({'claim_id':f"comparison-{row['input_id']}", 'statement':row['outcome'],
            'evidence_level':row['evidence_level'], 'status':row['status'],
            'input_ids':[row['input_id']], **{k:row[k] for k in CONTRACT},
            'evidence_paths':list(manifest['inputs'][row['input_id']]['inventory']),
            'mean':row['mean'],'interval':row.get('interval'),'independent_n':row.get('n'),
            'qualification':'fixed declared contract only; float64 analytic values are not rational certificates'})
    for benchmark in benchmarks.values():
        if 'manifest' not in benchmark['inputs']['inputs']: continue
        contract=_benchmark_contract(benchmark)
        for index,row in enumerate(benchmark['pareto']['rows']):
            target=next(r for r in benchmark['resources'] if r['method_id']==row['method_id'])
            level=benchmark['evidence_level']
            if target['selected_asset']['information_access']=='oracle' and level!='fixture': level='oracle_reference'
            ledger.append({'claim_id':f"operational-{benchmark['input_id']}-{index}",
                'statement':row['status'],'evidence_level':level,
                'input_ids':[benchmark['input_id']], **contract, 'comparison':row,
                'evidence_paths':list(manifest['inputs'][benchmark['input_id']]['inventory']),
                'qualification':'conditional on recorded selected assets and measured machine stratum; no energy claim'})
    for claim in manifest.get('claims',[]):
        if any(not claim.get(k) for k in CONTRACT): raise ValueError('claim requires full architecture/population/baseline/metric/resource contract')
        ids=claim.get('input_ids',[])
        if not ids or any(type(i)!=int or not 0<=i<len(manifest['inputs']) for i in ids): raise ValueError('claim requires actual evidence input ids')
        level=claim['evidence_level']; entries=[manifest['inputs'][i] for i in ids]
        if level not in ('source_record','forecast') and unavailable.intersection(ids):
            raise ValueError('claim lacks valid supporting evidence: '+str(sorted(unavailable.intersection(ids))))
        if level=='rigorous_certificate':
            # A source text or float report cannot certify itself. Exact fixture
            # claims have an independently evaluated rational input and verifier.
            from checks.exact_reference import verify_rational_kkt
            if any(e['kind']!='exact' or not _read(e['raw_path']).get('rational_qp') for e in entries):
                raise ValueError('rigorous claim requires exact rational raw evidence')
            for e in entries:
                raw=_read(e['raw_path'])
                if not verify_rational_kkt(raw['rational_qp'],tuple(raw['active_indices']))['verified']:
                    raise ValueError('exact evidence verification failed')
                if any(claim[k]!=raw.get('contract',{}).get(k) for k in CONTRACT): raise ValueError('exact claim contract differs from raw rational fixture')
                if claim.get('statement')!='exact_rational_fixture_kkt_verified': raise ValueError('exact fixture evidence does not establish a global theorem')
        elif level=='protected':
            if any(i not in comparisons or comparisons[i]['evidence_level']!='protected' or comparisons[i]['run_status']!='completed' or comparisons[i]['mean'] is None for i in ids):
                raise ValueError('protected claim lacks valid protected evidence')
            for i in ids:
                if any(claim[k]!=comparisons[i][k] for k in CONTRACT): raise ValueError('claim contract differs from executed evidence')
                if claim.get('statement')!=comparisons[i]['outcome']: raise ValueError('claim conclusion differs from evidence')
        elif level in ('operational','practical'):
            if any(i not in benchmarks or benchmarks[i]['evidence_level']!='operational' for i in ids):
                raise ValueError('operational claim requires finalized baseline/calibration evidence')
            for i in ids:
                config=benchmarks[i]['inputs']['inputs']['manifest']
                comparison=claim.get('comparison',{})
                target=next((r for r in benchmarks[i]['resources'] if r['method_id']==comparison.get('method_id')),None)
                if target is None or target['selected_asset']['information_access']=='oracle' or config.get('workflow')!='generated':
                    raise ValueError('oracle evidence cannot support operational learned claim')
                contract=_benchmark_contract(benchmarks[i])
                if any(claim[k]!=contract[k] for k in CONTRACT): raise ValueError('operational claim contract differs from actual evidence')
                allowed=[p for p in benchmarks[i]['pareto']['rows'] if p['status']=='scoped_domination']
                if claim.get('comparison') not in allowed: raise ValueError('practical claim must name an actual scoped dominance row')
        elif level not in ('exploratory','numerical_diagnostic','source_record','forecast'):
            raise ValueError('unknown claim evidence level')
        ledger.append({**claim,'evidence_paths':[p for e in entries for p in e['inventory']]})
    for claim in ledger:
        # Historical records and forecasts disclose provenance, not successful
        # execution. Their failures cannot invalidate another scoped claim.
        descriptive=claim['evidence_level'] in ('source_record','forecast')
        blocked=[] if descriptive else sorted(unavailable.intersection(claim['input_ids']))
        claim['unavailable_input_ids']=blocked
        claim['support_status']='historical_or_forecast_only' if descriptive else ('unavailable' if blocked else 'available')
    return ledger


def _write_report(manifest: JsonRecord, output: Path) -> GateDecision:
    output=Path(output)
    tables=regenerate_tables(manifest,output)
    ledger=_claims(manifest,tables)
    _csv(output/'claim_ledger.csv',ledger)
    introduction=['# Scoped execution report','',
           'Raw records and assets were verified against the ordered input manifest. Cached summaries were not used.',
           'Exact rational fixture deductions, float64 analytic evaluations, sampled diagnostics, exploration, protected comparisons and forecasts have separate evidence labels.',
           'No physical energy, biological, multistep, nonlinear-model, historical-priority or natural-transfer conclusion follows from this experiment.','']
    sections=[('Input outcomes',tables['statuses']),('Reconstructed comparisons and full contracts',tables['comparisons']),
              ('All source, diagnostic, exact and forecast records',tables['diagnostics']),
              ('Operational comparisons',[{k:v for k,v in b.items() if k not in ('raw_workers','rows')} for b in tables['benchmarks']])]
    lines=[
           'All failed candidates, family task rows, branch diagnostics and timing records are retained in raw_records.json / raw_records.csv and benchmark_workers.json. No success-only subgroup replaces the declared population.',
           '', '## Blueprint §11.2 challenge pass','']
    for challenge,response in CHALLENGES.items():
        supplied=manifest.get('challenges',{}).get(challenge)
        lines.extend([f'### {challenge}', '', response, ''])
        if supplied:
            ids=supplied.get('input_ids',[])
            if any(type(i)!=int or not 0<=i<len(manifest['inputs']) for i in ids): raise ValueError('challenge evidence input missing')
            lines.extend([json.dumps(supplied,sort_keys=True),''])
        else: lines.extend(['Status: not independently resolved by this report; affected claims remain limited.',''])
    lines.extend(['## Scientific artifacts',''])
    for figure in tables['figures']:
        lines.append(f"- [{figure['path']}]({figure['path']}); [{figure['csv']}]({figure['csv']}); units {figure['x_unit']} / {figure['y_unit']}; {figure['evaluation']}; independent n={figure['independent_n']}.")
    receipt=manifest.get('replay',{}).get('receipt'); replay=None
    if receipt:
        path=Path(receipt['path'])
        if _sha(path)!=receipt['sha256']: raise ValueError('clean replay receipt hash changed')
        replay=_read(path)
        if replay.get('input_order_sha256')!=manifest['input_order_sha256'] or replay.get('status')!='matched' or not replay.get('clean_environment'):
            raise ValueError('replay evidence does not match these report inputs')
    blocked_claims=[{'claim_id':c['claim_id'],'input_ids':c['unavailable_input_ids']} for c in ledger if c['support_status']=='unavailable']
    completed=(bool(replay) and not replay.get('fixture') and not any(e.get('fixture') for e in manifest['inputs'])
               and not blocked_claims)
    decision=GateDecision(GateCode.SUPPORTED_IN_SCOPE if completed else GateCode.INSENSITIVE_OR_UNSUITABLE,
        {'gate':'G10','report':str(output/'results.md'),'claim_ledger':str(output/'claim_ledger.csv'),
         'clean_replay':'matched' if replay else 'not_run','input_order_sha256':manifest['input_order_sha256'],
         'blocked_claims':blocked_claims,
         'scientific_conclusions':[r['outcome'] for r in tables['comparisons']]},
        'Report reconstruction and clean replay completed in the recorded scope.' if completed else 'Reconstruction retained all outcomes; G10 requires matching nonfixture replay and valid support for its scoped claims.',
        'Retain scoped negative and positive conclusions and unsuccessful history alike.' if completed else 'Complete missing replay or resolve only affected claims; fixtures cannot establish production completion.')
    lines.extend(['','## G10 decision','',json.dumps(to_json_record(decision),indent=2),''])
    import shutil
    with (output/'results.md').open('w') as stream:
        stream.write('\n'.join(introduction))
        for title,value in sections:
            stream.write('\n## '+title+'\n\n```json\n')
            json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n```\n')
        stream.write('\n## Failed, unsupported and unavailable raw outcomes\n\n```json\n')
        with (output/'adverse_records.json').open() as raw: shutil.copyfileobj(raw,stream,length=64*1024)
        stream.write('\n```\n'+ '\n'.join(lines))
    _write(output/'gate_decision.json',decision)
    _write(output/'report_inputs.json',manifest)
    return decision


def write_report(manifest: JsonRecord, output: Path) -> GateDecision:
    try: return _write_report(manifest,output)
    except ReportingResourceLimit as exc: return resource_limit_decision(exc,output)


def compare_scientific(left, right, *, rtol, atol, path='root'):
    """Float arithmetic is tolerant; integer counts, keys and identities are exact."""
    if isinstance(left,dict):
        if not isinstance(right,dict) or set(left)!=set(right): raise ValueError('scientific fields differ: '+path)
        for key in left: compare_scientific(left[key],right[key],rtol=rtol,atol=atol,path=path+'.'+key)
    elif isinstance(left,(list,tuple)):
        if not isinstance(right,(list,tuple)) or len(left)!=len(right): raise ValueError('scientific count differs: '+path)
        for i,(a,b) in enumerate(zip(left,right)): compare_scientific(a,b,rtol=rtol,atol=atol,path=path+f'[{i}]')
    elif isinstance(left,float):
        if not isinstance(right,(int,float)) or not np.isfinite(left) or not np.isfinite(right) or not np.isclose(left,right,rtol=rtol,atol=atol):
            raise ValueError('scientific arithmetic differs: '+path)
    elif type(left)!=type(right) or left!=right: raise ValueError('scientific identity/count differs: '+path)


def _scientific(value, _measurement_ids=None):
    """Exclude measured duration; preserve aliases of clock-derived cost IDs.

    selection.py uses perf_counter_ns IDs to charge a shared source fit once.
    Their sharing topology must replay, but the clock suffix cannot. Scientific
    asset hashes, candidate identifiers, data keys and counts remain untouched.
    """
    if _measurement_ids is None: _measurement_ids={}
    if isinstance(value,dict):
        result={}
        for key in sorted(value):
            if key.endswith('_ns') or key in ('timing','timing_records','timestamp','run_path','attempt_path','elapsed_seconds','wall_time_seconds','peak_rss_bytes'): continue
            item=value[key]
            if key in ('asset_fit_id','base_fit_id','shrink_fit_id','noise_fit_id') and isinstance(item,str):
                if item not in _measurement_ids:
                    _measurement_ids[item]=item.rsplit(':',1)[0]+':measurement-'+str(len(_measurement_ids))
                result[key]=_measurement_ids[item]
            else: result[key]=_scientific(item,_measurement_ids)
        return result
    if isinstance(value,list): return [_scientific(v,_measurement_ids) for v in value]
    return value


def compare_run_stores(left: Path, right: Path, *, rtol, atol) -> JsonRecord:
    import sqlite3,shutil,itertools
    preflight=_guard_paths((path for root in (Path(left),Path(right)) for path in _store_paths(root)),
                            phase='run_comparison_before_open',streaming=True)
    arrays=[]; count=0
    for key in IDENTITY_FIELDS:
        if _read(left/'run.json')[key]!=_read(right/'run.json')[key]: raise ValueError('replay run identity mismatch: '+key)
    # Payloads and their order index live on disk. RAM holds one verified raw
    # block and one record pair, independent of J and checkpoint layout.
    with tempfile.TemporaryDirectory(prefix='prepared-replay-index-') as temporary:
        disk_budget=shutil.disk_usage(temporary).free//2
        disk_estimate=8*preflight['json_bytes']+REPORT_BASE_BYTES
        if disk_estimate>disk_budget:
            raise ReportingResourceLimit({**preflight,'phase':'comparison_disk_index','resource':'disk',
                'estimated_disk_bytes':disk_estimate,'disk_budget_bytes':disk_budget})
        index=sqlite3.connect(str(Path(temporary)/'records.sqlite'))
        try:
            index.execute('PRAGMA cache_size=-2048'); index.execute('PRAGMA temp_store=FILE'); index.execute('PRAGMA threads=1')
            index.execute('CREATE TABLE records (side INTEGER, unit TEXT, payload TEXT)')
            for side,root in enumerate((Path(left),Path(right))):
                store=RunStore.open(root,_read(root/'run.json')); assets={}
                for block in sorted((root/'blocks').iterdir()):
                    store._verify(block)
                    for row in _read(block/'records.json'):
                        if row['kind'] in ('timing_blocks','decisions'): continue
                        scientific=_scientific(row); nested=scientific.get('result',{})
                        key=json.dumps({key:scientific.get(key,nested.get(key)) for key in
                            ('kind','outer_replication','task_id','task_key','method_id','configuration_key','branch','endpoint')},sort_keys=True)
                        index.execute('INSERT INTO records VALUES (?,?,?)',
                                      (side,key,json.dumps(scientific,sort_keys=True,allow_nan=False)))
                    marker=_read(block/'commit.json')
                    for asset in (block/'assets').glob('*.npy'): assets[marker['key']+'/'+asset.name]=asset
                arrays.append(assets)
            index.commit(); index.execute('CREATE INDEX record_order ON records(side,unit,payload)')
            cursors=[index.execute('SELECT unit,payload FROM records WHERE side=? ORDER BY unit,payload',(side,)) for side in (0,1)]
            for a,b in itertools.zip_longest(*cursors):
                if a is None or b is None or a[0]!=b[0]: raise ValueError('scientific record keys/count differ')
                compare_scientific(json.loads(a[1]),json.loads(b[1]),rtol=rtol,atol=atol)
                count+=1
        finally: index.close()
    if set(arrays[0])!=set(arrays[1]): raise ValueError('scientific array inventory differs')
    hashes={}
    for key,a in arrays[0].items():
        b=arrays[1][key]; aa=np.load(a,allow_pickle=False,mmap_mode='r'); bb=np.load(b,allow_pickle=False,mmap_mode='r')
        if aa.shape!=bb.shape or aa.dtype!=bb.dtype: raise ValueError('scientific array differs: '+key)
        for a_chunk,b_chunk in np.nditer([aa,bb],flags=['external_loop','buffered','zerosize_ok'],
                                         op_flags=[['readonly'],['readonly']],buffersize=65536):
            if not np.allclose(a_chunk,b_chunk,rtol=rtol,atol=atol): raise ValueError('scientific array differs: '+key)
        # The fixed environment reconstructs the exact frozen assets, independently
        # of the tolerance for derived arithmetic metrics.
        if _sha(a)!=_sha(b): raise ValueError('frozen asset hash differs: '+key)
        hashes[key]=_sha(a)
    return {'status':'matched','scientific_records_compared':count,'arrays_compared':len(hashes),
            'asset_hashes':hashes,'timing_policy':'remeasured_not_equal','rtol':rtol,'atol':atol,
            'resource_preflight':preflight,'record_comparison':'disk index; one raw block and record pair in memory'}


def replay_run(manifest: JsonRecord, output: Path) -> GateDecision:
    from ._replay import clean_replay
    try:
        _verify_inputs(manifest,replay=True)
        return clean_replay(manifest,Path(output))
    except ReportingResourceLimit as exc: return resource_limit_decision(exc,output)
