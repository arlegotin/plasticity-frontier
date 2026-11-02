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
REPORT_SCHEMA_VERSION = "benchmark_raw_refs_v2"
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
              'qualification':'ordinary resource estimate, not a formal memory bound',
              'scientific_counts':'not inferred from file counts; all raw observations remain unchanged',
              'recovery':'retry the identical inputs with sufficient local reserve; no truncation, sampling or changed scientific identity'}
    if estimate>budget: raise ReportingResourceLimit(evidence)
    return evidence


def _empty_scale():
    return {'files_scanned':0,'json_bytes':0,'max_json_bytes':0,'json_working_bytes':0,'array_bytes':0,'max_array_bytes':0,
            'hash_only_files':0,'hash_only_bytes':0,'semantic_json_bytes':0,'max_semantic_json_bytes':0,
            'inventory_bytes':0,'largest_file_bytes':0,'largest_file':None,'inventory_complete':True}


def _guard_paths(paths, *, phase, copies=1, object_bytes=0, streaming=False, json_factor=REPORT_JSON_FACTOR):
    return _guard_assets(((path,'semantic') for path in paths),phase=phase,copies=copies,
                         object_bytes=object_bytes,streaming=streaming,json_factor=json_factor)


def _guard_assets(assets, *, phase, copies=1, object_bytes=0, streaming=False, json_factor=REPORT_JSON_FACTOR) -> dict:
    scale=_empty_scale()
    for value,role in assets:
        if role not in ('semantic','hash_only'): raise ValueError('invalid report asset role')
        path=Path(value); size=path.stat().st_size
        if role=='hash_only':
            scale['hash_only_files']+=1; scale['hash_only_bytes']+=size
        scale['files_scanned']+=1
        scale['inventory_bytes']+=4096+16*len(os.fsencode(path))
        # Verified hash-only controls are JSON regardless of their original suffix.
        if role=='hash_only' or path.suffix in ('.json','.jsonl'):
            scale['json_bytes']+=size
            scale['max_json_bytes']=max(scale['max_json_bytes'],size)
            if role=='semantic':
                scale['semantic_json_bytes']+=size
                scale['max_semantic_json_bytes']=max(scale['max_semantic_json_bytes'],size)
        elif path.suffix=='.npy':
            scale['array_bytes']+=size; scale['max_array_bytes']=max(scale['max_array_bytes'],size)
        if size>scale['largest_file_bytes']:
            scale['largest_file_bytes']=size; scale['largest_file']=str(path)
    scale['json_working_bytes']=scale['max_semantic_json_bytes'] if streaming else scale['semantic_json_bytes']
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


def _input_paths(entry, *, retained_bytes=0):
    for path,_ in _input_assets(entry,retained_bytes=retained_bytes): yield path


def _input_assets(entry, *, retained_bytes=0):
    yield Path(entry['descriptor_path']),'semantic'
    kind=entry['kind']
    if kind=='confirmation':
        for path in _store_paths(entry['run_path']): yield path,'semantic'
        yield Path(entry['protocol_path']),'semantic'
    elif kind=='benchmark':
        root=Path(entry['run_path']); yield root/'inputs.json','semantic'
        for folder in ('blocks','requests'):
            for path in _walk_files(root/folder): yield path,'semantic'
        if (root/'failure.json').exists(): yield root/'failure.json','semantic'
        # This bounded control JSON locates a second raw inventory. Its own
        # read guard runs before parsing; no worker ledger is read here.
        _guard_paths((root/'inputs.json',),phase='benchmark_control_before_parse',object_bytes=retained_bytes)
        config=_read(root/'inputs.json')['inputs'].get('manifest',{})
        if config.get('phase')=='finalized':
            from .benchmark import _verify_calibration_control
            verified=_verify_calibration_control(config['frozen'])
            calibration=Path(config['frozen']['calibration_path'])
            if verified:
                for path in verified: yield path,'hash_only'
            else: yield calibration,'semantic'
            yield calibration.parent/'inputs.json','semantic'
            for folder in ('blocks','requests'):
                for path in _walk_files(calibration.parent/folder): yield path,'semantic'
    elif kind!='not_run': yield Path(entry['raw_path']),'semantic'
    if 'process_path' in entry: yield Path(entry['process_path']),'semantic'
    for asset in entry.get('assets',[]): yield Path(asset),'semantic'


def _object_bytes(value):
    """Count the supplied JSON tree without making a serialization/copy first."""
    import sys
    size=sys.getsizeof(value)
    if isinstance(value,dict):
        for key,child in value.items(): size+=sys.getsizeof(key)+_object_bytes(child)
    elif isinstance(value,(list,tuple,set,frozenset)):
        for child in value: size+=_object_bytes(child)
    return size


def _declared_count(value):
    if value is None: return 0
    if type(value)!=int or value<0: raise ValueError('nonnegative integer scientific count required for report admission')
    return value


def _guard_inputs(manifest, *, replay=False, retained_bytes=0):
    scale=_empty_scale(); largest_streamed=0; retained_json=0; scalars=0
    retained=retained_bytes+_object_bytes(manifest)
    for entry in manifest['inputs']:
        streamed=entry['kind'] in ('confirmation','benchmark')
        measured=_guard_assets(_input_assets(entry,retained_bytes=retained),phase='input_before_store_open',
            streaming=streamed,object_bytes=retained,copies=2 if replay else 1,
            json_factor=REPORT_PARSE_FACTOR if replay else REPORT_JSON_FACTOR)
        for key in ('files_scanned','json_bytes','array_bytes','inventory_bytes','hash_only_files','hash_only_bytes','semantic_json_bytes'): scale[key]+=measured[key]
        for key in ('max_json_bytes','max_array_bytes','max_semantic_json_bytes'): scale[key]=max(scale[key],measured[key])
        if measured['largest_file_bytes']>scale['largest_file_bytes']:
            scale['largest_file_bytes']=measured['largest_file_bytes']; scale['largest_file']=measured['largest_file']
        if streamed:
            largest_streamed=max(largest_streamed,measured['json_working_bytes'])
            if not replay and entry['kind']=='confirmation':
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
        else: retained_json+=measured['json_working_bytes']
    scale['json_working_bytes']=largest_streamed+retained_json
    return _resource_check(scale,phase='replay_inputs' if replay else 'report_inputs',
                           copies=2 if replay else 1,object_bytes=retained,scalar_bytes=scalars,
                           json_factor=REPORT_PARSE_FACTOR if replay else REPORT_JSON_FACTOR)


def resource_limit_decision(exc, output):
    output=Path(output)
    evidence={**exc.evidence,'report_status':'not_completed',
              'partial_artifacts':'any already streamed output is incomplete; original raw evidence and scientific counts are unchanged'}
    decision=GateDecision(GateCode.LOCAL_RESOURCE_LIMIT,evidence,
        'Report/replay allocation was refused before loading the offending raw working set.',
        exc.evidence['recovery'])
    _write(output/'resource_limit.json',evidence); _write(output/'gate_decision.json',decision)
    return decision


def _read(path, *, retained_bytes=0):
    # Parsing a native JSON tree does not allocate analysis/plot workspaces.
    # In particular, inventory manifests grow with J but contain mostly paths
    # and hashes. Raw semantic consumers have the separate 256x admission above.
    _guard_paths((Path(path),),phase='json_before_parse',json_factor=REPORT_PARSE_FACTOR,object_bytes=retained_bytes)
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


def _raw_files(entry, *, retained_bytes=0):
    """Canonical inventory deliberately excludes rebuildable summaries."""
    _guard_assets(_input_assets(entry,retained_bytes=retained_bytes),phase='raw_inventory_before_store_open',
                  streaming=entry['kind'] in ('confirmation','benchmark'),object_bytes=retained_bytes)
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
        _guard_paths((root/'inputs.json',),phase='benchmark_control_before_parse')
        config=_read(root/'inputs.json')['inputs'].get('manifest',{})
        if config.get('phase')=='finalized':
            calibration=Path(config['frozen']['calibration_path'])
            from .benchmark import _verify_calibration_control
            verified=_verify_calibration_control(config['frozen'])
            if verified: files.update(verified)
            files.add(calibration); files.add(calibration.parent/'inputs.json')
            files.update(p for folder in ('blocks','requests') for p in (calibration.parent/folder).glob('*.json'))
    elif kind != 'not_run':
        files.add(Path(entry['raw_path']))
    files.update(Path(p) for p in entry.get('assets', []))
    if 'process_path' in entry: files.add(Path(entry['process_path']))
    return {str(p): _sha(p) for p in sorted(files)}


def _artifact_summary(command, process):
    """Normalize only the producer's declared output location, never raw values."""
    summary=json.loads(process['stdout'])
    if type(process['returncode']) is not int:
        raise ValueError('diagnostic process returncode must be an integer')
    if command=='robustness':
        artifact=Path(summary['evidence']['report'])
        summary['evidence']['report']='<output-artifact>'
    else:
        artifact=Path(summary['attempt_path'])/'frontier.json'
        summary['attempt_path']='<output-directory>'
    return artifact, summary


def _validate_output_artifact(entry, source_root, *, retained_bytes=0):
    command=entry.get('command')
    if (entry['kind']!='diagnostic' or not isinstance(command,list)
            or not all(isinstance(arg,str) for arg in command) or len(command) not in (3,5)):
        raise ValueError('output artifact requires a declared diagnostic CLI command')
    routes={'robustness':('--manifest','robustness.json'), 'frontier':('--config','frontier.json')}
    if command[0] not in routes:
        raise ValueError('unsupported output artifact producer')
    flag,filename=routes[command[0]]
    if (command[1]!=flag or (len(command)==5 and command[3]!='--output')
            or entry['output_artifact']!={'filename':filename}):
        raise ValueError('invalid output artifact selector or producer arguments')
    source_root=Path(source_root).resolve()
    config=_path(command[2],source_root)
    if not config.is_relative_to(source_root) or str(config) not in entry.get('assets',[]):
        raise ValueError('output artifact configuration must be an inventoried repository asset')
    if not entry.get('process_path') or entry['process_path'] not in entry['inventory']:
        raise ValueError('output artifact requires an inventoried process record')
    _guard_paths((entry['process_path'],),phase='diagnostic_process_before_parse',object_bytes=retained_bytes)
    process=_read(entry['process_path'],retained_bytes=retained_bytes)
    if set(process)!={'command','returncode','stdout','stderr'} or not isinstance(process['stderr'],str):
        raise ValueError('complete diagnostic process record required')
    if process.get('command')!=command:
        raise ValueError('output artifact command differs from recorded producer process')
    artifact,summary=_artifact_summary(command[0],process)
    if _path(artifact,source_root)!=Path(entry['raw_path']):
        raise ValueError('output artifact differs from recorded producer output')
    return process,summary


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
        for name in ('run_path','protocol_path','raw_path','process_path'):
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
        if 'output_artifact' in entry:
            _validate_output_artifact(entry,root)
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


def _verify_inputs(manifest, *, replay=False, retained_bytes=0):
    if manifest.get('schema_version') != 1 or not manifest.get('inputs'): raise ValueError('invalid report manifest')
    preflight=_guard_inputs(manifest,replay=replay,retained_bytes=retained_bytes)
    if content_hash(manifest['inputs']) != manifest.get('input_order_sha256'): raise ValueError('input manifest identity changed')
    scientific_runs=set()
    for index, entry in enumerate(manifest['inputs']):
        retained=retained_bytes+_object_bytes(manifest)+_object_bytes(scientific_runs)
        if entry['input_id'] != index: raise ValueError('input order changed')
        if _raw_files(entry,retained_bytes=retained) != entry['inventory']: raise ValueError('raw inventory hash changed')
        if 'output_artifact' in entry:
            _validate_output_artifact(entry,manifest['replay']['source_root'],retained_bytes=retained)
        if entry['kind']=='exact':
            from .exact_evidence import verify_export
            verify_export(_read(entry['raw_path'],retained_bytes=retained),entry,manifest['replay']['source_root'])
        if entry['kind']=='confirmation':
            run=_read(Path(entry['run_path'])/'run.json'); identity=tuple(run[k] for k in IDENTITY_FIELDS)
            if identity in scientific_runs: raise ValueError('duplicate scientific run cannot add independent samples')
            scientific_runs.add(identity)
            del run,identity
    return preflight


def _contract(science):
    config = science['config']
    return {'architecture':{'class':'declared deep-linear and outside-class operational controls',
                            'reference_budget':config['budget']},
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


def _confirmation(entry, emit=lambda row:None, *, retained_bytes=0):
    from .confirmation import run_record, analyze_results
    from .protocol import validate_frozen_protocol
    from .records import TaskKey
    from .tasks import CONTEXT, NOISE
    _guard_inputs({'inputs':[entry]},retained_bytes=retained_bytes)
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
    executions={}; tasks={}; completions=[]; raw_count=0; role_metadata={}
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
                for role, metadata in result.get('role_execution_metadata',{}).items():
                    if metadata not in role_metadata.setdefault(role,[]):
                        role_metadata[role].append(metadata)
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
                    evidence_level=entry['evidence_level'], run_status=run['status'],
                    role_execution_metadata=role_metadata, **_contract(science))
    analysis['architecture']['executed_roles']=role_metadata
    return analysis, raw_count


def _worker_key(worker):
    rows=worker['rows']; mid=worker['resource']['method_id']
    if not rows: raise ValueError('benchmark worker has no rows')
    block=rows[0]['block']
    if type(block)!=int or block<0 or not isinstance(mid,str): raise ValueError('invalid benchmark worker key')
    if any(row['block']!=block or row['method_id']!=mid for row in rows):
        raise ValueError('benchmark worker/row key changed')
    return {'block':block,'method_id':mid}


def _package_path(package, relative):
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts: raise ValueError('raw reference escapes package')
    result=(Path(package)/path).resolve()
    if not result.is_relative_to(Path(package).resolve()): raise ValueError('raw reference escapes package')
    return result


def _verify_package_file(reference, package):
    """Verify generated package evidence without loading its contents."""
    path=_package_path(package,reference['path'])
    if 'size_bytes' in reference and path.stat().st_size!=reference['size_bytes']:
        raise ValueError('package evidence size changed')
    if _sha(path)!=reference['sha256']: raise ValueError('package evidence hash changed')
    return path


def _verify_worker_source(reference, package, *, retained_bytes=0):
    # Historical external source paths are preserved provenance, not portable
    # package dependencies. Generated replay evidence has an explicit scope.
    scope=reference.get('source_scope','external_provenance')
    if scope=='external_provenance':
        if 'process_ref' in reference: raise ValueError('generated process evidence requires package source scope')
        return
    if scope!='package' or reference['source_json_pointer']!='':
        raise ValueError('invalid generated worker source binding')
    process_ref=reference.get('process_ref')
    if not isinstance(process_ref,dict): raise ValueError('generated worker process reference missing')
    process_path=_package_path(package,process_ref['path'])
    _guard_paths((process_path,),phase='benchmark_process_evidence_before_parse',object_bytes=retained_bytes)
    _verify_package_file(process_ref,package)
    process=_read(process_path)
    if (process['returncode']!=0 or process['stdout']['path']!=reference['source_block_path'] or
            process['stdout']['sha256']!=reference['source_block_sha256']):
        raise ValueError('generated worker/process source binding changed')
    for name in ('stdout','stderr'): _verify_package_file(process[name],package)


def _load_worker_index(reference, package, *, retained_bytes=0):
    path=_package_path(package,reference['path'])
    _guard_paths((path,),phase='benchmark_index_before_parse',object_bytes=retained_bytes)
    if _sha(path)!=reference['sha256']: raise ValueError('benchmark index hash changed')
    index=_read(path)
    if index.get('schema_version')!=1 or index.get('record_count')!=len(index['records']) or (reference['record_count'] is not None and index['record_count']!=reference['record_count']):
        raise ValueError('benchmark index count changed')
    seen=set(); sources={}; pointers=set(); paths=set()
    for ordinal,record in enumerate(index['records']):
        key=record['worker_key']; pair=(key['block'],key['method_id'])
        if pair in seen or type(key['block'])!=int or key['block']<0 or not isinstance(key['method_id'],str):
            raise ValueError('duplicate or invalid benchmark index worker key')
        if record['ordinal']!=ordinal or record['path'] in paths: raise ValueError('benchmark index order/path changed')
        _package_path(package,record['path'])
        source=record['source_block_path']; digest=record['source_block_sha256']
        scope=record.get('source_scope','external_provenance')
        if scope=='package':
            _package_path(package,source)
            if not isinstance(record.get('process_ref'),dict): raise ValueError('generated worker process reference missing')
            _package_path(package,record['process_ref']['path'])
        elif scope!='external_provenance' or 'process_ref' in record:
            raise ValueError('invalid benchmark source scope')
        if source in sources and sources[source]!=digest: raise ValueError('inconsistent benchmark source binding')
        pointer=(source,record['source_json_pointer'])
        if pointer in pointers: raise ValueError('duplicate benchmark source pointer')
        seen.add(pair); sources[source]=digest; pointers.add(pointer); paths.add(record['path'])
    return index


def _load_worker_ref(reference, package, *, retained_bytes=0):
    """Resolve a complete semantic worker, with admitted parse and key/count checks."""
    path=_package_path(package,reference['path'])
    _guard_paths((path,),phase='benchmark_worker_before_parse',object_bytes=retained_bytes)
    if _sha(path)!=reference['sha256']: raise ValueError('benchmark worker hash changed')
    index_ref=reference.get('index_ref')
    if index_ref is None:
        # Records read directly from an index carry no circular self-hash.
        # Still enforce its key/count/source binding. Package consumers should
        # admit the index with _load_worker_index using the summary's hash first.
        index_path=path.parent/'workers.index.json'
        _guard_paths((index_path,),phase='benchmark_reference_index_before_parse',object_bytes=retained_bytes)
        index_ref={'path':index_path.relative_to(Path(package).resolve()).as_posix(),
                   'sha256':_sha(index_path),'record_count':None}
    index=_load_worker_index(index_ref,package,retained_bytes=retained_bytes)
    ordinal=reference['ordinal']
    if type(ordinal)!=int or not 0<=ordinal<len(index['records']): raise ValueError('benchmark reference ordinal changed')
    recorded=index['records'][ordinal]
    if any(reference.get(k)!=v for k,v in recorded.items()): raise ValueError('benchmark reference/source binding changed')
    if 'record_count' in reference and reference['record_count']!=reference.get('candidate_count'):
        raise ValueError('benchmark reference candidate count changed')
    del recorded,index
    _verify_worker_source(reference,package,retained_bytes=retained_bytes)
    worker=_read(path)
    if _worker_key(worker)!=reference['worker_key']: raise ValueError('benchmark reference worker key changed')
    candidates=worker['resource'].get('selection',{}).get('candidate_records')
    if (candidates is not None)!=('candidate_count' in reference) or (candidates is not None and len(candidates)!=reference['candidate_count']):
        raise ValueError('benchmark reference candidate count changed')
    return worker


def _publish_worker(worker, output, directory, ordinal, *, source_path, source_sha256, source_pointer,
                    source_scope="external_provenance", process_ref=None):
    if source_scope=='package':
        source_path=Path(source_path).resolve().relative_to(Path(output).resolve()).as_posix()
        if not isinstance(process_ref,dict): raise ValueError('generated worker process reference missing')
        _package_path(output,process_ref['path'])
    elif source_scope!='external_provenance' or process_ref is not None:
        raise ValueError('invalid benchmark source scope')
    relative=directory/f'{ordinal:08d}.json'
    _write(output/relative,worker)
    reference={'ordinal':ordinal,'worker_key':_worker_key(worker),'path':relative.as_posix(),
               'sha256':_sha(output/relative),'source_block_path':str(source_path),
               'source_block_sha256':source_sha256,'source_json_pointer':source_pointer}
    if source_scope=='package': reference.update(source_scope='package',process_ref=dict(process_ref))
    candidates=worker['resource'].get('selection',{}).get('candidate_records')
    if candidates is not None: reference['candidate_count']=len(candidates)
    return reference


def _publish_worker_index(records, output, directory):
    relative=directory/'workers.index.json'
    _write(output/relative,{'schema_version':1,'record_count':len(records),'records':records})
    return {'path':relative.as_posix(),'sha256':_sha(output/relative),'record_count':len(records)}


def _retain_benchmark_workers(root, output, *, phase, consume, retained_bytes=lambda:0):
    """Publish complete workers, retaining only references between indivisible blocks.

    output is the per-input package directory. Consumers detach every retained
    field; neither this loop nor its callback may retain a completed raw block.
    """
    if phase not in ('campaign','calibration'): raise ValueError('invalid worker retention phase')
    output=Path(output); package=output.parent; directory=Path(output.name)/phase
    index_path=package/directory/'workers.index.json'
    # A failed rerun must not leave an earlier completed index authoritative.
    if index_path.exists(): index_path.unlink()
    records=[]; seen=set()
    for path in sorted((Path(root)/'blocks').glob('*.json')):
        reserve=retained_bytes()+_object_bytes(records)+_object_bytes(seen)
        _guard_paths((path,),phase='benchmark_block_before_parse',object_bytes=reserve)
        source_hash=_sha(path); block=_read(path)
        if not isinstance(block,list): raise ValueError('benchmark block must contain a worker list')
        for position,worker in enumerate(block):
            key=_worker_key(worker); pair=(key['block'],key['method_id'])
            if pair in seen: raise ValueError('duplicate benchmark worker key')
            seen.add(pair)
            reference=_publish_worker(worker,package,directory,len(records),source_path=path,
                                      source_sha256=source_hash,source_pointer='/'+str(position))
            records.append(reference)
            consume(worker,reference)
            _guard_paths((path,),phase='benchmark_retained_growth',
                         object_bytes=retained_bytes()+_object_bytes(records)+_object_bytes(seen))
            del worker,reference,key
        del block
    _resource_check(_empty_scale(),phase='benchmark_index_publication',
                    object_bytes=retained_bytes()+_object_bytes(records)+_object_bytes(seen))
    return _publish_worker_index(records,package,directory)


def _compact_benchmark_resource(resource, reference):
    fields=('method_id','initialization_norm','trainable_parameters','runtime_stored_scalars',
            'runtime_serialized_bytes','execution_hash','selected_execution_hash','equality_status','risk')
    compact={key:resource[key] for key in fields if key in resource}
    compact['memory']={key:resource['memory'][key] for key in ('isolation','peak_rss_bytes') if key in resource['memory']}
    if 'selected_asset' in resource:
        compact['selected_asset']={key:resource['selected_asset'][key] for key in ('information_access',) if key in resource['selected_asset']}
    compact['resource_ref']={**reference,'json_pointer':'/resource'}
    if 'selection' in resource:
        compact['selection']={key:resource['selection'][key] for key in
                              ('selected_id','selected_options','source_hash','validation_hash') if key in resource['selection']}
        compact['selection_ref']={**reference,'json_pointer':'/resource/selection'}
        if 'candidate_records' in resource['selection']:
            compact['candidate_records_ref']={**reference,'json_pointer':'/resource/selection/candidate_records',
                                               'record_count':len(resource['selection']['candidate_records'])}
    # Native roundtrip detaches subclasses and nested lists from input blocks.
    return json.loads(json.dumps(compact,allow_nan=False))


def _validate_benchmark_worker(worker, frozen=None):
    resource=worker['resource']; mid=resource['method_id']; _worker_key(worker)
    for row in worker['rows']:
        temperature=row['cold_warm']
        if temperature not in ('cold','warm'): raise ValueError('invalid benchmark temperature')
        in_process=resource['memory']['isolation']=='in_process_test_or_callback'
        stratum=content_hash({'conditions':worker['conditions'],'cold_warm':temperature,'in_process':in_process})
        if row['timing_stratum']!=stratum: raise ValueError('benchmark hashed timing stratum changed')
        if any(row[field]!=resource[field] for field in ('method_id','execution_hash','selected_execution_hash')):
            raise ValueError('benchmark worker/row identity changed')
        if frozen:
            block=row['block']; orders=frozen['orders']
            if not 0<=block<len(orders) or row['method_order']!=orders[block] or mid not in orders[block]:
                raise ValueError('benchmark worker order changed')
            repetitions=1 if temperature=='cold' else frozen['repetitions'][mid]
            if row['repetitions']!=repetitions: raise ValueError('benchmark worker repetition schedule changed')
    if frozen:
        if resource['selected_execution_hash']!=frozen['selected_execution_hashes'].get(mid):
            raise ValueError('benchmark selected execution identity changed')
        if content_hash(worker['conditions'])!=frozen['condition_hash']: raise ValueError('benchmark timing conditions changed')
        if resource['memory']['isolation']!='fresh_interpreter': raise ValueError('finalized benchmark requires fresh worker isolation')


def _benchmark(entry, *, output, emit=lambda row:None, retained_bytes=lambda:0):
    """Reconstruct unchanged science with detached summaries and verified full workers."""
    from .benchmark import comparison_table, _campaign_configuration, _fixed_schedule, _verify_calibration_control
    _guard_assets(_input_assets(entry,retained_bytes=retained_bytes()),phase='benchmark_before_raw_allocation',streaming=True,object_bytes=retained_bytes())
    root=Path(entry['run_path']); inputs=_read(root/'inputs.json')
    generated=inputs['inputs'].get('manifest',{}); campaign=generated.get('phase')=='finalized'
    rows=[]; seen=set(); resources={}; conditions=None; pilots=[]; calibration_blocks=set()
    def reserve():
        return retained_bytes()+sum(_object_bytes(v) for v in (inputs,rows,seen,resources,conditions,pilots,calibration_blocks))
    directory=Path(output)/f"benchmark-input-{entry['input_id']}"
    calibration_ref=None; frozen=generated['frozen'] if campaign else None
    if campaign:
        if frozen['identity']['dirty_diff_sha256'] is not None or frozen['identity']['versions']!=entry['source']:
            raise ValueError('benchmark requires clean recorded source identity')
        if content_hash(_campaign_configuration(generated))!=frozen['configuration_sha256']:
            raise ValueError('benchmark frozen scientific inputs changed')
        calibration_path=Path(frozen['calibration_path'])
        if _verify_calibration_control(frozen) is None:
            _guard_paths((calibration_path,),phase='benchmark_calibration_before_parse',object_bytes=reserve())
            calibration=_read(calibration_path)
            if content_hash(calibration)!=frozen['calibration_sha256']: raise ValueError('benchmark calibration identity changed')
            del calibration
        calibration_inputs=_read(calibration_path.parent/'inputs.json')
        calibration_config=calibration_inputs['inputs']['manifest']
        if _campaign_configuration(calibration_config)!=_campaign_configuration(generated):
            raise ValueError('calibration/campaign scientific contract changed')
        def consume_pilot(worker,reference):
            # Only the fields consumed by _fixed_schedule survive publication.
            pilot=worker['pilot']; fields=('method_id','samples_ns','desired_repetitions','capped')
            _resource_check(_empty_scale(),phase='benchmark_pilot_retention',
                            object_bytes=reserve()+_object_bytes(pilot))
            pilots.append(json.loads(json.dumps({k:pilot[k] for k in fields})))
            calibration_blocks.update(row['block'] for row in worker['rows'])
        calibration_ref=_retain_benchmark_workers(calibration_path.parent,directory,phase='calibration',
            consume=consume_pilot,retained_bytes=lambda:reserve()+_object_bytes(calibration_inputs))
        rebuilt=_fixed_schedule({'config':calibration_config,'pilot':pilots,'schedule':{'balanced_blocks':len(calibration_blocks)}})
        if any(rebuilt[k]!=frozen[k] for k in rebuilt): raise ValueError('frozen timing schedule differs from raw calibration')
        pilots.clear(); calibration_blocks.clear(); del rebuilt,calibration_inputs,calibration_config
    def consume(worker,reference):
        nonlocal conditions
        _validate_benchmark_worker(worker,frozen)
        resource=worker['resource']; mid=resource['method_id']
        # Charge prospective retained fields before detaching, while this raw
        # block is still live. The block guard covers transient raw traversals.
        growth=_object_bytes(worker['rows'])
        if conditions is None: growth+=_object_bytes(worker['conditions'])
        if mid not in resources:
            growth+=sum(_object_bytes(resource[k]) for k in ('risk','memory') if k in resource)
            growth+=sum(_object_bytes(resource.get('selection',{}).get(k)) for k in
                        ('selected_id','selected_options','source_hash','validation_hash'))
        _guard_paths((Path(reference['source_block_path']),),phase='benchmark_summary_before_copy',object_bytes=reserve()+growth)
        if conditions is None: conditions=json.loads(json.dumps(worker['conditions']))
        for row in worker['rows']:
            key=(mid,row['block'],row['cold_warm'])
            if key in seen: raise ValueError('duplicate benchmark worker stratum')
            seen.add(key)
        rows.extend(json.loads(json.dumps(worker['rows'])))
        first=mid not in resources
        if not first:
            old=resources[mid]
            if any(resource[field]!=old[field] for field in ('selected_execution_hash','execution_hash')):
                raise ValueError('benchmark selected asset or execution changed')
            a,b=old['memory']['peak_rss_bytes'],resource['memory']['peak_rss_bytes']
            old['memory']['peak_rss_bytes']=max(a,b) if a is not None and b is not None else None
        else: resources[mid]=_compact_benchmark_resource(resource,reference)
        # Stream every nested adverse record at its full raw provenance. The
        # first resource occurrence used to appear separately in the summary.
        for failed in _adverse_records(worker): emit({'input_id':entry['input_id'],**failed,'resource_ref':reference})
        if first:
            for failed in _adverse_records(resource,'resources.'+mid):
                emit({'input_id':entry['input_id'],**failed,'resource_ref':reference})
    worker_ref=_retain_benchmark_workers(root,directory,phase='campaign',consume=consume,retained_bytes=reserve)
    for resource in resources.values():
        for name in ('resource_ref','selection_ref','candidate_records_ref'):
            if name in resource: resource[name]['index_ref']=worker_ref
    declared=[d['method_id'] for d in inputs['methods']]
    complete=worker_ref['record_count']>0 and set(resources)==set(declared) and not (root/'failure.json').exists()
    if campaign:
        expected={(mid,block,stratum) for block,order in enumerate(frozen['orders']) for mid in order for stratum in ('cold','warm')}
        complete &= seen==expected
    required={'B08:ridge','B10:empirical_bayes'} | {'B05:'+mid for mid,r in resources.items() if r['trainable_parameters'] is not None and mid!='B08:gd'}
    missing=sorted(required-set(resources)); equal=all(r['equality_status']=='pass' for r in resources.values())
    _resource_check(_empty_scale(),phase='benchmark_comparison_before_allocation',object_bytes=reserve(),
                    scalar_bytes=8192*len(resources)**2*len({row['timing_stratum'] for row in rows}))
    pareto=comparison_table(rows,list(resources.values()),gate={'status':'pass' if complete and equal else 'unavailable'},phase='finalized' if campaign else 'exploratory')
    result={'report_schema_version':REPORT_SCHEMA_VERSION,'input_id':entry['input_id'],'label':entry['label'],
            'status':'completed' if complete else 'failed_or_incomplete',
            'evidence_level':'fixture' if entry.get('fixture') else ('operational' if campaign and complete and not missing and equal else 'exploratory'),
            'missing_controls':missing,'rows':rows,'resources':list(resources.values()),'pareto':pareto,'conditions':conditions,
            'inputs':inputs,'worker_records_ref':worker_ref,'energy':{'status':'not_run'},'accelerator':{'status':'not_run'}}
    if calibration_ref: result['calibration_worker_records_ref']=calibration_ref
    _resource_check(_empty_scale(),phase='benchmark_summary_complete',object_bytes=retained_bytes()+_object_bytes(result))
    for failed in _adverse_records({k:v for k,v in result.items() if k not in ('rows','resources','inputs')}):
        emit({'input_id':entry['input_id'],**failed,'worker_records_ref':worker_ref})
    return result


def _benchmark_contract(benchmark):
    from ._benchmark_worker import generated_config
    config=benchmark['inputs']['inputs'].get('manifest')
    if not config: raise ValueError('operational claim needs generated scientific contract')
    c=to_json_record(generated_config(config))
    return {'architecture':{'class':'declared deep-linear and outside-class operational controls','budget':c['budget']},
            'population':{'workload':c['workload'],'preparation':c['preparation'],'access':c['access']},
            'baseline':c['methods'],'metric':'expected squared coefficient error [coefficient^2]',
            'resource_contract':{'conditions':benchmark['conditions'],
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
    def reserve():
        return sum(_object_bytes(v) for v in (manifest,comparisons,benchmarks,diagnostics,statuses))
    with _RecordStream(output/'raw_records.json') as raw, _RecordStream(output/'adverse_records.json') as adverse, (output/'raw_records.csv').open('w',newline='') as raw_csv:
        writer=csv.DictWriter(raw_csv,fieldnames=['input_id','kind','outer_replication','task_id','method_id','status','record_json'])
        writer.writeheader()
        for entry in manifest['inputs']:
            kind=entry['kind']
            _guard_assets(_input_assets(entry,retained_bytes=reserve()),phase='report_next_input',streaming=kind in ('benchmark','confirmation'),object_bytes=reserve())
            if kind=='confirmation':
                def emit(row):
                    record={'input_id':entry['input_id'],**row}; raw.write(record)
                    writer.writerow({**{key:record.get(key) for key in writer.fieldnames if key!='record_json'},
                                     'record_json':json.dumps(record,allow_nan=False)})
                    for failed in _adverse_records(record): adverse.write(failed)
                comparison,_=_confirmation(entry,emit,retained_bytes=reserve()); comparisons.append(comparison)
                status=comparison['run_status']
            elif kind=='benchmark':
                benchmark=_benchmark(entry,output=output,emit=adverse.write,retained_bytes=reserve)
                benchmarks.append(benchmark); status=benchmark['status']
                del benchmark
            elif kind=='not_run': status='not_run'
            else:
                record=_read(entry['raw_path']); diagnostics.append({'input_id':entry['input_id'],'label':entry['label'],
                    'evidence_level':entry['evidence_level'],'raw':record})
                for failed in _adverse_records(diagnostics[-1]): adverse.write(failed)
                status=record.get('status','recorded') if isinstance(record,dict) else 'recorded'
            statuses.append({'input_id':entry['input_id'],'label':entry['label'],'kind':kind,'status':status,
                             'reason':entry.get('reason'), 'evidence_paths':list(entry['inventory'])})
        raw_count=raw.count; adverse_count=adverse.count
    _resource_check(_empty_scale(),phase='report_artifacts_before_allocation',object_bytes=reserve())
    _csv(output/'comparisons_table.csv', comparisons)
    _csv(output/'input_status.csv', statuses)
    _csv(output/'family_comparisons.csv', [{'input_id':r['input_id'],'method':mid,**value} for r in comparisons for mid,value in r.get('secondary',{}).items()])
    _csv(output/'pareto_table.csv', [{'input_id':b['input_id'],**row} for b in benchmarks for row in b['pareto']['rows']])
    _write(output/'diagnostics.json',diagnostics)
    _write(output/'benchmark_workers.json',{'report_schema_version':REPORT_SCHEMA_VERSION,'benchmarks':[
        {k:b[k] for k in ('input_id','worker_records_ref','calibration_worker_records_ref') if k in b} for b in benchmarks]})
    figures=_figures(output,comparisons,benchmarks,diagnostics)
    result={'report_schema_version':REPORT_SCHEMA_VERSION,'comparisons':comparisons,'benchmarks':benchmarks,'diagnostics':diagnostics,'statuses':statuses,'figures':figures,
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
            from .exact_evidence import verify_export
            if any(e['kind']!='exact' or not _read(e['raw_path']).get('rational_qp') for e in entries):
                raise ValueError('rigorous claim requires exact rational raw evidence')
            for e in entries:
                raw=_read(e['raw_path'])
                verify_export(raw,e)
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
    _resource_check(_empty_scale(),phase='report_ledger_before_write',
                    object_bytes=sum(_object_bytes(v) for v in (manifest,tables,ledger)),scalar_bytes=64*1024)
    _csv(output/'claim_ledger.csv',ledger)
    introduction=['# Scoped execution report','',
           'Raw records and assets were verified against the ordered input manifest. Cached summaries were not used.',
           'Exact rational fixture deductions, float64 analytic evaluations, sampled diagnostics, exploration, protected comparisons and forecasts have separate evidence labels.',
           'No physical energy, biological, multistep, nonlinear-model, historical-priority or natural-transfer conclusion follows from this experiment.','']
    sections=[('Input outcomes',tables['statuses']),('Reconstructed comparisons and full contracts',tables['comparisons']),
              ('All source, diagnostic, exact and forecast records',tables['diagnostics']),
              ('Operational comparisons',[{k:v for k,v in b.items() if k not in ('raw_workers','rows')} for b in tables['benchmarks']])]
    lines=[
           'Family task rows are retained in raw_records.json / raw_records.csv. The benchmark_raw_refs_v2 directory benchmark_workers.json links hash-verified indexes of complete workers, including every candidate, probe, failure and unknown field. Compact tables reference these full workers; consumers of historical embedded raw_workers must migrate to the indexes. No success-only subgroup replaces the declared population.',
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
    def reserve():
        return sum(_object_bytes(v) for v in (manifest,tables,ledger,introduction,sections,lines,replay))
    if receipt:
        path=Path(receipt['path'])
        if _sha(path)!=receipt['sha256']: raise ValueError('clean replay receipt hash changed')
        replay=_read(path,retained_bytes=reserve())
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
    _resource_check(_empty_scale(),phase='report_before_write',
                    object_bytes=reserve()+_object_bytes(decision)+_object_bytes(blocked_claims),scalar_bytes=64*1024)
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


def _comparison_container_sizes(count, *, scale, retained_bytes):
    """Measure this runtime's actual growing str-dict/set/list table capacities.

    The probe retains numeric string keys and None values only. Its admission
    bounds every entry by a whole one-entry container plus its largest key;
    this temporary bound is never multiplied into the real per-asset graph.
    """
    import sys
    key_size=sys.getsizeof(str(max(0,count-1)))
    temporary=(count*(sys.getsizeof({'0':None})+sys.getsizeof({'0'})+
                      sys.getsizeof([None])+key_size)+sys.getsizeof(None))
    _resource_check(scale,phase='run_comparison_container_probe_before_allocation',
                    object_bytes=retained_bytes+temporary)
    dictionary={}; sequence=[]
    for ordinal in range(count):
        dictionary[str(ordinal)]=None; sequence.append(None)
    keys=set(dictionary)
    return {'dict':sys.getsizeof(dictionary),'set':sys.getsizeof(keys),'list':sys.getsizeof(sequence)}


def _comparison_map_plan(left, right, *, scale, retained_bytes):
    """Project the complete native graph; retain only bounded marker metadata.

    Map values are str paths, so their graph is directly measurable. Logical
    keys and path strings are measured individually, including Unicode width.
    Set/hash containers share map keys; hash values are separate ASCII digests.
    Marker padding is live once during this pass, never once per array.
    """
    import struct,sys
    counts=[]; key_sizes=[]; path_sizes=[]; block_counts=[]; block_name_sizes=[]
    digest_size=sys.getsizeof('0'*64)
    for root in (Path(left),Path(right)):
        count=key_size=path_size=blocks=names=0
        for block in (root/'blocks').iterdir():
            marker=_read(block/'commit.json',retained_bytes=retained_bytes)
            key=marker.get('key') if isinstance(marker,dict) else None
            if not isinstance(key,str): raise ValueError('scientific array logical key must be a string')
            blocks+=1; names+=sys.getsizeof(block.name)
            for asset in (block/'assets').glob('*.npy'):
                logical=key+'/'+asset.name; path=str(asset)
                count+=1; key_size+=sys.getsizeof(logical); path_size+=sys.getsizeof(path)
                del logical,path
            # These strings are prospective; no full map has been built yet.
            # Account for the one real parsed marker (including unknown fields)
            # separately, and stop actual excessive key/path growth early.
            projected=sum(key_sizes)+sum(path_sizes)+key_size+path_size
            _resource_check(scale,phase='run_comparison_map_metadata_growth',
                            object_bytes=retained_bytes+projected+_object_bytes(marker))
            del marker,key
        counts.append(count); key_sizes.append(key_size); path_sizes.append(path_size)
        block_counts.append(blocks); block_name_sizes.append(names)
    strings=sum(key_sizes)+sum(path_sizes)+counts[0]*digest_size
    containers={}
    for count in set(counts+block_counts+[2]):
        containers[count]=_comparison_container_sizes(count,scale=scale,retained_bytes=retained_bytes+strings)
    # All three dictionaries and both simultaneous key sets are reserved even
    # though the key-set expression ends before the returned hash map grows.
    dictionaries=sum(containers[n]['dict'] for n in counts)+containers[counts[0]]['dict']
    sets=sum(containers[n]['set'] for n in counts)
    # A growing table may coexist briefly with the old table it replaces.
    # Only one map/set grows at a time; reserve one complete largest table.
    growth=max((containers[n][kind] for n in counts for kind in ('dict','set')),default=0)
    arrays_list=containers[2]['list']
    # One store's sorted name list is live while indexing. Reserve its strings,
    # growing list table and a full pointer per name for timsort scratch space.
    block_names=max((containers[n]['list']+size+n*struct.calcsize('P')
                     for n,size in zip(block_counts,block_name_sizes)),default=0)
    native=strings+dictionaries+sets+growth+arrays_list+block_names
    return {'native_bytes':native,'array_counts':counts,'logical_key_bytes':sum(key_sizes),
            'path_string_bytes':sum(path_sizes),'hash_value_bytes':counts[0]*digest_size,
            'dictionary_capacity_bytes':dictionaries,'key_set_capacity_bytes':sets,
            'container_growth_bytes':growth,
            'array_list_bytes':arrays_list,'block_names_and_sort_bytes':block_names}


def _comparison_pair_scale(lengths):
    """One semantic256 workspace covers BOTH native64 decoded record trees.

    Only parsed pairs remain here, with no scientific projection/plot copy.
    For L/R encoded lengths: 64*(L+R) <= 128*max(L,R) <= 256*max(L,R).
    Complete encoded strings/keys and all retained maps are charged separately.
    """
    working=max((record[1] for record in lengths),default=0)
    total=sum(record[1] for record in lengths)
    return {**_empty_scale(),'json_bytes':total,'semantic_json_bytes':total,
            'max_json_bytes':working,'max_semantic_json_bytes':working,'json_working_bytes':working}


def compare_run_stores(left: Path, right: Path, *, rtol, atol, retained_bytes=0) -> JsonRecord:
    import sqlite3,shutil,itertools
    left=Path(left); right=Path(right)
    preflight=_guard_paths((path for root in (left,right) for path in _store_paths(root)),
                            phase='run_comparison_before_open',streaming=True,object_bytes=retained_bytes)
    scale={key:preflight[key] for key in _empty_scale()}
    plan=_comparison_map_plan(left,right,scale=scale,retained_bytes=retained_bytes)
    map_bytes=plan['native_bytes']
    buffers=2*1024*1024+4*65536*8
    def admit(phase, *, objects=0, working_scale=scale):
        return _resource_check(working_scale,phase=phase,object_bytes=retained_bytes+map_bytes+objects,
                               scalar_bytes=buffers)
    preflight={**admit('run_comparison_maps_before_open'),'native_map_projection':plan}
    arrays=[]; count=0
    left_run=_read(left/'run.json',retained_bytes=retained_bytes+map_bytes)
    right_run=_read(right/'run.json',retained_bytes=retained_bytes+map_bytes+_object_bytes(left_run))
    for key in IDENTITY_FIELDS:
        if left_run[key]!=right_run[key]: raise ValueError('replay run identity mismatch: '+key)
    del left_run,right_run
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
            for side,root in enumerate((left,right)):
                admit('run_comparison_store_before_open')
                store=RunStore.open(root,_read(root/'run.json',retained_bytes=retained_bytes+map_bytes)); assets={}
                block_names=os.listdir(root/'blocks'); block_names.sort()
                for name in block_names:
                    block=root/'blocks'/name
                    admit('run_comparison_block_before_parse')
                    store._verify(block)
                    rows=_read(block/'records.json',retained_bytes=retained_bytes+map_bytes)
                    for row in rows:
                        if row['kind'] in ('timing_blocks','decisions'): continue
                        scientific=_scientific(row); nested=scientific.get('result',{})
                        key=json.dumps({key:scientific.get(key,nested.get(key)) for key in
                            ('kind','outer_replication','task_id','task_key','method_id','configuration_key','branch','endpoint')},sort_keys=True)
                        payload=json.dumps(scientific,sort_keys=True,allow_nan=False)
                        index.execute('INSERT INTO records VALUES (?,?,?)',(side,key,payload))
                        del scientific,nested,key,payload
                    del rows
                    # The loop variable otherwise retains the last full record.
                    if 'row' in locals(): del row
                    marker=_read(block/'commit.json',retained_bytes=retained_bytes+map_bytes)
                    for asset in (block/'assets').glob('*.npy'): assets[marker['key']+'/'+asset.name]=str(asset)
                    del marker
                arrays.append(assets)
                del block_names
            del store,assets
            index.commit(); index.execute('CREATE INDEX record_order ON records(side,unit,payload)')
            # SQL performs one bounded metadata scan over encoded ASCII JSON;
            # refuse a prospective full pair before either cursor fetches it.
            lengths=index.execute('SELECT side, max(length(payload)), max(length(unit)) FROM records GROUP BY side').fetchall()
            pair_scale=_comparison_pair_scale(lengths)
            import sys
            encoded=sum(2*sys.getsizeof('')+payload+unit for _,payload,unit in lengths)
            encoded+=2*sys.getsizeof((None,None))+sys.getsizeof((None,None))
            pair_preflight=admit('run_comparison_record_pair_before_fetch',objects=encoded,working_scale=pair_scale)
            preflight['record_pair_preflight']=pair_preflight
            cursors=[index.execute('SELECT unit,payload FROM records WHERE side=? ORDER BY unit,payload',(side,)) for side in (0,1)]
            for a,b in itertools.zip_longest(*cursors):
                if a is None or b is None or a[0]!=b[0]: raise ValueError('scientific record keys/count differ')
                actual_scale=_comparison_pair_scale(((0,len(a[1]),len(a[0])),(1,len(b[1]),len(b[0]))))
                admit('run_comparison_record_pair_before_parse',objects=_object_bytes((a,b)),working_scale=actual_scale)
                compare_scientific(json.loads(a[1]),json.loads(b[1]),rtol=rtol,atol=atol)
                count+=1
            if 'a' in locals(): del a,b
        finally: index.close()
    array_scale={**_empty_scale(),'max_array_bytes':scale['max_array_bytes']}
    preflight['array_preflight']=admit('run_comparison_array_maps',working_scale=array_scale)
    if set(arrays[0])!=set(arrays[1]): raise ValueError('scientific array inventory differs')
    hashes={}
    for key,a in arrays[0].items():
        admit('run_comparison_array_before_load',working_scale=array_scale)
        b=arrays[1][key]; aa=np.load(a,allow_pickle=False,mmap_mode='r'); bb=np.load(b,allow_pickle=False,mmap_mode='r')
        if aa.shape!=bb.shape or aa.dtype!=bb.dtype: raise ValueError('scientific array differs: '+key)
        for a_chunk,b_chunk in np.nditer([aa,bb],flags=['external_loop','buffered','zerosize_ok'],
                                         op_flags=[['readonly'],['readonly']],buffersize=65536):
            if not np.allclose(a_chunk,b_chunk,rtol=rtol,atol=atol): raise ValueError('scientific array differs: '+key)
        # Derived arithmetic uses declared tolerances; frozen assets still
        # require exact hashes in the fixed environment.
        if _sha(a)!=_sha(b): raise ValueError('frozen asset hash differs: '+key)
        hashes[key]=_sha(a)
        del aa,bb
        if 'a_chunk' in locals(): del a_chunk,b_chunk
    return {'status':'matched','scientific_records_compared':count,'arrays_compared':len(hashes),
            'asset_hashes':hashes,'timing_policy':'remeasured_not_equal','rtol':rtol,'atol':atol,
            'resource_preflight':preflight,'record_comparison':'disk index; one raw block and record pair in memory'}


def replay_run(manifest: JsonRecord, output: Path) -> GateDecision:
    from ._replay import clean_replay
    try:
        _verify_inputs(manifest,replay=True)
        return clean_replay(manifest,Path(output))
    except ReportingResourceLimit as exc: return resource_limit_decision(exc,output)
