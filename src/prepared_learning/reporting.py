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


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_record(value), indent=2, allow_nan=False)+'\n')


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
    from .protocol import checkout_identity
    root = Path(__file__).resolve().parents[2]
    versions, dirty = checkout_identity({}, {})
    inputs = []
    scientific_runs=set()
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


def _verify_inputs(manifest):
    if manifest.get('schema_version') != 1 or not manifest.get('inputs'): raise ValueError('invalid report manifest')
    if content_hash(manifest['inputs']) != manifest.get('input_order_sha256'): raise ValueError('input manifest identity changed')
    scientific_runs=set()
    for index, entry in enumerate(manifest['inputs']):
        if entry['input_id'] != index: raise ValueError('input order changed')
        if _raw_files(entry) != entry['inventory']: raise ValueError('raw inventory hash changed')
        if entry['kind']=='confirmation':
            run=_read(Path(entry['run_path'])/'run.json'); identity=tuple(run[k] for k in IDENTITY_FIELDS)
            if identity in scientific_runs: raise ValueError('duplicate scientific run cannot add independent samples')
            scientific_runs.add(identity)


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


def _confirmation(entry):
    from .confirmation import run_record, analyze_results
    from .protocol import validate_frozen_protocol
    from .records import TaskKey
    from .tasks import CONTEXT, NOISE
    protocol = _read(entry['protocol_path']); science = protocol['scientific']
    if not entry.get('fixture'): validate_frozen_protocol(protocol)
    run = _read(Path(entry['run_path'])/'run.json')
    expected = run_record(protocol)
    if any(run[k] != expected[k] for k in IDENTITY_FIELDS): raise ValueError('protocol/run identity mismatch')
    if run['input_manifest_content_sha256'] != content_hash(protocol): raise ValueError('scientific input manifest changed')
    if run['phase'] != 'confirmation' or run['dirty_diff_sha256'] is not None:
        raise ValueError('dirty or exploratory store cannot become protected evidence')
    store = RunStore.open(Path(entry['run_path']), expected)
    raw = []
    for path in sorted((store.path/'blocks').iterdir()):
        store._verify(path)
        rows = _read(path/'records.json'); store._validate_rows(rows); raw.extend(rows)
    executions = {r['outer_replication']:r['result'] for r in raw if r['kind']=='execution'}
    if len(executions) != sum(r['kind']=='execution' for r in raw): raise ValueError('duplicate preparations')
    for result in executions.values(): result['tasks'] = []
    for row in raw:
        if row['kind'] != 'execution_task': continue
        task = row['result']; keys = protocol['protected_keys']
        phase = keys['test_phase']+{'primary':0,'update_check':100,'secondary':200}[task['endpoint']]
        expected_key = content_hash({'key':TaskKey(keys['root_seed'],phase,task['outer_replication'],task['task_id'],0),
                                     'context_stream':CONTEXT,'noise_stream':NOISE})
        if task['task_key'] != expected_key: raise ValueError('scientific task key changed')
        if task['outer_replication'] not in executions: raise ValueError('task without preparation')
        executions[task['outer_replication']]['tasks'].append(task)
    for result in executions.values(): result['tasks'].sort(key=lambda t:(t['endpoint'],t['task_id']))
    completions = [r['outer_replication'] for r in raw if r['kind']=='replication_completion' and r['status']=='completed']
    try:
        if run['status'] != 'completed' or sorted(completions) != sorted(executions):
            raise ValueError('uncompleted or failed run; no success-only inference')
        analysis = analyze_results(list(executions.values()), science, protocol['protected_keys']['root_seed'])
    except ValueError as exc:
        analysis = {'status':'invalid_or_incomplete', 'reason':str(exc), 'mean':None, 'interval':None,
                    'n':None, 'secondary':{}, 'independent_unit':None}
    analysis.update(input_id=entry['input_id'], label=entry['label'], outcome=_outcome(analysis),
                    evidence_level=entry['evidence_level'], run_status=run['status'], **_contract(science))
    return analysis, raw


def _benchmark(entry):
    """Recompute paired resource comparisons from worker blocks, not benchmark.json."""
    from .benchmark import comparison_table, _campaign_configuration, _fixed_schedule
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


def _figures(output, comparisons, benchmarks, diagnostics):
    # No implicit ~/.matplotlib write, including when a caller set MPLCONFIGDIR.
    with tempfile.TemporaryDirectory(prefix='prepared-plots-') as config:
        prior=os.environ.get('MPLCONFIGDIR'); os.environ['MPLCONFIGDIR']=config
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
                if isinstance(raw,dict) and 'rows' in raw: points=raw['rows']
                for point in points:
                    if not isinstance(point,dict): continue
                    for branch in ('F','Z'):
                        row=point.get(branch)
                        if isinstance(row,dict) and row.get('objective') is not None:
                            frontier.append({'label':item['label']+'/'+branch,'value':row['objective'],
                                'budget':point.get('norm_cap',point.get('caps',{}).get('initialization_norm_cap')),
                                'evaluation':'analytic','independent_n':None,'evidence_level':'float64_diagnostic'})
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
                if rows:
                    x=[r.get('x',i) for i,r in enumerate(rows)]; y=[r['value'] for r in rows]
                    ax.scatter(x,y)
                    if name!='pareto': ax.set_xticks(x,[r['label'] for r in rows],rotation=25,ha='right')
                    for xx,row in zip(x,rows):
                        if row.get('low') is not None: ax.plot([xx,xx],[row['low'],row['high']],color='black')
                    ax.set_title('; '.join(f"{r['evaluation']}, independent n={r['independent_n']}" for r in rows[:4]))
                    ax.axhline(0,color='grey',linewidth=.7)
                else: ax.text(.5,.5,'not_run / no applicable verified observations',ha='center',transform=ax.transAxes)
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
                fig.savefig(output/(name+'.svg')); fig.savefig(output/(name+'.pdf')); plt.close(fig)
                labels={r['evaluation'] for r in rows}
                figures.append({'path':name+'.svg','pdf':name+'.pdf','csv':name+'.csv',
                    'x_unit':xunit,'y_unit':yunit,'evaluation':next(iter(labels)) if len(labels)==1 else ('mixed' if rows else 'unavailable'),
                    'independent_n':[r['independent_n'] for r in rows], 'renderer':'matplotlib'})
            return figures
        finally:
            if prior is None: os.environ.pop('MPLCONFIGDIR',None)
            else: os.environ['MPLCONFIGDIR']=prior


def regenerate_tables(manifest: JsonRecord, output: Path) -> JsonRecord:
    _verify_inputs(manifest)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    comparisons=[]; benchmarks=[]; diagnostics=[]; raw=[]; statuses=[]
    for entry in manifest['inputs']:
        kind=entry['kind']
        if kind=='confirmation':
            comparison,records=_confirmation(entry); comparisons.append(comparison)
            raw.extend({'input_id':entry['input_id'], **row} for row in records)
            status=comparison['run_status']
        elif kind=='benchmark':
            benchmark=_benchmark(entry); benchmarks.append(benchmark); status=benchmark['status']
        elif kind=='not_run': status='not_run'
        else:
            record=_read(entry['raw_path']); diagnostics.append({'input_id':entry['input_id'],'label':entry['label'],
                'evidence_level':entry['evidence_level'],'raw':record})
            status=record.get('status','recorded') if isinstance(record,dict) else 'recorded'
        statuses.append({'input_id':entry['input_id'],'label':entry['label'],'kind':kind,'status':status,
                         'reason':entry.get('reason'), 'evidence_paths':list(entry['inventory'])})
    _csv(output/'comparisons_table.csv', comparisons)
    _csv(output/'input_status.csv', statuses)
    _csv(output/'raw_records.csv', raw)
    _csv(output/'family_comparisons.csv', [{'input_id':r['input_id'],'method':mid,**value} for r in comparisons for mid,value in r.get('secondary',{}).items()])
    _csv(output/'pareto_table.csv', [{'input_id':b['input_id'],**row} for b in benchmarks for row in b['pareto']['rows']])
    _write(output/'raw_records.json', raw); _write(output/'diagnostics.json',diagnostics)
    _write(output/'benchmark_workers.json',benchmarks)
    adverse=_adverse_records(raw)+_adverse_records(diagnostics)+_adverse_records(benchmarks)
    _write(output/'adverse_records.json',adverse)
    figures=_figures(output,comparisons,benchmarks,diagnostics)
    result={'comparisons':comparisons,'benchmarks':benchmarks,'diagnostics':diagnostics,'statuses':statuses,'figures':figures,'adverse_records':adverse}
    _write(output/'tables.json', result)
    return result


def _claims(manifest, tables):
    comparisons={r['input_id']:r for r in tables['comparisons']}
    benchmarks={r['input_id']:r for r in tables['benchmarks']}
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
    return ledger


def write_report(manifest: JsonRecord, output: Path) -> GateDecision:
    output=Path(output); tables=regenerate_tables(manifest,output); ledger=_claims(manifest,tables)
    _csv(output/'claim_ledger.csv',ledger)
    lines=['# Scoped execution report','',
           'Raw records and assets were verified against the ordered input manifest. Cached summaries were not used.',
           'Exact rational fixture deductions, float64 analytic evaluations, sampled diagnostics, exploration, protected comparisons and forecasts have separate evidence labels.',
           'No physical energy, biological, multistep, nonlinear-model, historical-priority or natural-transfer conclusion follows from this experiment.','',
           '## Input outcomes','', '```json',json.dumps(tables['statuses'],indent=2),'```','',
           '## Reconstructed comparisons and full contracts','', '```json',json.dumps(tables['comparisons'],indent=2),'```','',
           '## All source, diagnostic, exact and forecast records','', '```json',json.dumps(tables['diagnostics'],indent=2),'```','',
           '## Failed, unsupported and unavailable raw outcomes','', '```json',json.dumps(tables['adverse_records'],indent=2),'```','',
           '## Operational comparisons','', '```json',json.dumps([{k:v for k,v in b.items() if k not in ('raw_workers','rows')} for b in tables['benchmarks']],indent=2),'```','',
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
    completed=(bool(replay) and not replay.get('fixture') and not any(e.get('fixture') for e in manifest['inputs'])
               and all(s['status'] not in ('failed','invalid','incomplete','running','interrupted','failed_or_incomplete') for s in tables['statuses']))
    decision=GateDecision(GateCode.SUPPORTED_IN_SCOPE if completed else GateCode.INSENSITIVE_OR_UNSUITABLE,
        {'gate':'G10','report':str(output/'results.md'),'claim_ledger':str(output/'claim_ledger.csv'),
         'clean_replay':'matched' if replay else 'not_run','input_order_sha256':manifest['input_order_sha256'],
         'scientific_conclusions':[r['outcome'] for r in tables['comparisons']]},
        'Report reconstruction and clean replay completed in the recorded scope.' if completed else 'Reconstruction retained all outcomes; G10 remains incomplete without matching clean replay or with failed inputs.',
        'Retain scoped negative and positive conclusions alike.' if completed else 'Complete missing replay or inspect failed evidence; do not infer a positive scientific result.')
    lines.extend(['','## G10 decision','',json.dumps(to_json_record(decision),indent=2),''])
    (output/'results.md').write_text('\n'.join(lines)); _write(output/'gate_decision.json',decision)
    _write(output/'report_inputs.json',manifest)
    return decision


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


def _scientific(value):
    """Exclude measured duration and reporting locations, never scientific IDs."""
    if isinstance(value,dict):
        return {k:_scientific(v) for k,v in value.items() if not k.endswith('_ns') and k not in ('timing','timing_records','timestamp','run_path','attempt_path','elapsed_seconds','wall_time_seconds')}
    if isinstance(value,list): return [_scientific(v) for v in value]
    return value


def compare_run_stores(left: Path, right: Path, *, rtol, atol) -> JsonRecord:
    stores=[]; rows=[]; arrays=[]
    for root in (Path(left),Path(right)):
        store=RunStore.open(root,_read(root/'run.json')); stores.append(store)
        records=[]; assets={}
        for block in sorted((root/'blocks').iterdir()):
            store._verify(block)
            for row in _read(block/'records.json'):
                if row['kind'] not in ('timing_blocks','decisions'): records.append(_scientific(row))
            marker=_read(block/'commit.json')
            for asset in (block/'assets').glob('*.npy'):
                assets[marker['key']+'/'+asset.name]=asset
        def scientific_key(row):
            nested=row.get('result',{})
            return json.dumps({key:row.get(key,nested.get(key)) for key in
                ('kind','outer_replication','task_id','task_key','method_id','configuration_key','branch','endpoint')},sort_keys=True)
        rows.append(sorted(records,key=scientific_key))
        arrays.append(assets)
    for key in IDENTITY_FIELDS:
        if _read(left/'run.json')[key]!=_read(right/'run.json')[key]: raise ValueError('replay run identity mismatch: '+key)
    compare_scientific(rows[0],rows[1],rtol=rtol,atol=atol)
    if set(arrays[0])!=set(arrays[1]): raise ValueError('scientific array inventory differs')
    hashes={}
    for key,a in arrays[0].items():
        b=arrays[1][key]; aa=np.load(a,allow_pickle=False); bb=np.load(b,allow_pickle=False)
        if aa.shape!=bb.shape or aa.dtype!=bb.dtype or not np.allclose(aa,bb,rtol=rtol,atol=atol): raise ValueError('scientific array differs: '+key)
        # The fixed environment reconstructs the exact frozen assets, independently
        # of the tolerance for derived arithmetic metrics.
        if _sha(a)!=_sha(b): raise ValueError('frozen asset hash differs: '+key)
        hashes[key]=_sha(a)
    return {'status':'matched','scientific_records_compared':len(rows[0]),'arrays_compared':len(hashes),
            'asset_hashes':hashes,'timing_policy':'remeasured_not_equal','rtol':rtol,'atol':atol}


def replay_run(manifest: JsonRecord, output: Path) -> GateDecision:
    _verify_inputs(manifest)
    from ._replay import clean_replay
    return clean_replay(manifest,Path(output))
