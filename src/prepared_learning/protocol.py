"""Endpoint-complete, content-addressed protocols and immutable publication.

The library validates evidence and identity; the CLI additionally verifies the
executed checkout. A frozen protocol fixes procedures, never a pilot-fitted prior.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile

from .records import (BudgetSpec, PreparationSpec, SamplePlan, RunConfig,
                      JsonRecord, content_hash)
from .specs import validate_config

BASELINE = ('B01', 'B02', 'B11:Z')
JOINT = ('B03', 'B04', 'B11:F', 'B11:Z')
HASH_FIELDS = ('code_sha256', 'dependency_lock_sha256', 'analysis_sha256',
               'generator_sha256', 'candidate_sha256', 'selection_sha256', 'evaluation_sha256',
               'source_manifest_sha256', 'preprocessing_sha256', 'runtime_sha256')


def config_from_record(data):
    return RunConfig(schema_version=data['schema_version'], workload=data['workload'],
        budget=BudgetSpec(**data['budget']), preparation=PreparationSpec(**data['preparation']),
        numerics=data['numerics'], access=data['access'], methods=tuple(data['methods']),
        keys=data['keys'], sample_plan=SamplePlan(**data['sample_plan']), confirmation=data.get('confirmation',{}))


def scientific_contract(scientific):
    return content_hash({k:v for k,v in scientific.items() if k != 'precision'})


def _positive(value, name):
    if type(value) is not int or value <= 0: raise ValueError(f'{name} must be a positive integer')


def _finite(value, name, minimum=0):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'invalid {name}')


def _validate_draft(draft, evidence):
    if draft.get('scientific',{}).get('config',{}).get('sample_plan',{}).get('analysis_mode')!='conditional_risk_over_preparations':
        return _validate_mode_draft(draft,evidence)
    try:
        if draft['schema_version'] != 1 or 'protected_keys' in draft: raise ValueError('invalid draft or premature protected keys')
        s=draft['scientific']; config=config_from_record(s['config']); validate_config(config,False)
        if config.keys: raise ValueError('exploratory keys must be excluded from frozen config')
        mode=config.sample_plan.analysis_mode; counts=s['config']['sample_plan']
        required={'conditional_risk_over_preparations':('outer_replicates','source_training_tasks','source_validation_tasks','source_block_size'),
                  'nested_task_simulation':('outer_replicates','tasks_per_replicate','source_training_tasks','source_validation_tasks','source_block_size'),
                  'fixed_prior_task_simulation':('tasks_per_replicate',), 'deterministic_frontier':()}[mode]
        for name in ('outer_replicates','tasks_per_replicate','source_training_tasks','source_validation_tasks','source_block_size'):
            if name in required: _positive(counts[name],name)
            elif counts[name] is not None: raise ValueError(f'{name} is inapplicable for {mode}')
        if mode=='conditional_risk_over_preparations' and counts['source_block_size']!=config.preparation.target_context_size:
            raise ValueError('source block size must match primary adaptation context')
        if config.access.get('prior') != 'estimated' or config.access.get('selection') != 'learned_only':
            raise ValueError('this learned protocol requires observed-source information access')
        selectors=s['selectors']
        if set(selectors) != {'baseline','joint'}: raise ValueError('both selectors required')
        for name,allowed in (('baseline',BASELINE),('joint',JOINT)):
            ids=selectors[name]
            if not isinstance(ids,list) or not ids or len(set(ids))!=len(ids) or not set(ids)<=set(allowed): raise ValueError('ineligible primary comparator')
        secondary_methods=set(s.get('simulation_schedule',{}).get('method_ids',[]))
        if set(config.methods)!=set(selectors['baseline'])|set(selectors['joint'])|secondary_methods: raise ValueError('method and selector candidates differ')
        if not secondary_methods<=set(('B00','B08:gd','B08:ridge','B09:least_squares','B09:ridge','B10:empirical_bayes')): raise ValueError('unsupported operational method')
        if set(selectors['baseline'])!=set(BASELINE) or set(selectors['joint'])!=set(JOINT):
            if not s.get('amendment'): raise ValueError('candidate parity change requires an amendment')
        if not s['scenario_id']: raise ValueError('scenario required')
        if s['scenario_id']=='primary_richer_k_3_5':
            if (config.preparation.input_dimension,config.preparation.target_context_size,config.workload['sigma2'],config.budget.depth,config.budget.initialization_norm_cap,config.budget.trainable_parameter_cap)!=(6,4,1.,2,.6,21) or list(config.workload['spectrum'])!=[.5,.25,.15,.07,.02,.01]:
                raise ValueError('primary scenario mismatch')
        elif not s.get('amendment'): raise ValueError('changed primary scenario requires amendment')
        if s['endpoint']!={'metric':'squared_coefficient_error','contrast':'baseline_minus_joint'}: raise ValueError('unsupported primary endpoint')
        inference=s['inference']
        if inference['alpha']!=.05 or inference['procedure'] not in ('student','outer_bootstrap') or inference['coverage']!='approximate': raise ValueError('unsupported inference')
        if inference['moment_basis']!='bounded_constrained_fixed_maps' or mode!='conditional_risk_over_preparations':
            raise ValueError('only bounded fixed-map primary inference is currently calibrated')
        family=s['secondary_family']
        if family['adjustment'] not in ('holm','simultaneous') or not isinstance(family['contrasts'],list) or len(set(family['contrasts']))!=len(family['contrasts']): raise ValueError('invalid secondary family')
        if secondary_methods:
            if s['simulation_schedule'].get('inference')!='descriptive_only' or set(family['contrasts'])!=secondary_methods: raise ValueError('operational schedule requires explicit descriptive contrasts')
            costs=evidence['resources'].get('simulation_task_seconds',[])
            if len(costs)!=sum(len(b['differences']) for b in evidence['blocks']) or any(not math.isfinite(v) or v<=0 for v in costs): raise ValueError('measured operational simulation costs required')
        elif family['contrasts']: raise ValueError('secondary contrasts need a simulation schedule')
        schedule=s['update_check_schedule']; _positive(schedule['tasks_per_replication'],'update checks')
        if not isinstance(schedule['outer_replications'],list) or not schedule['outer_replications'] or any(type(j)!=int or not 0<=j<counts['outer_replicates'] for j in schedule['outer_replications']) or len(set(schedule['outer_replications']))!=len(schedule['outer_replications']): raise ValueError('invalid fixed update schedule')
        if s['stop_rule']!='fixed_counts_invariants_only': raise ValueError('outcome-dependent stopping forbidden')
        from .tasks import population_identity
        if s['assets']!={'prior':'refit_each_replication','preprocessing':'identity','population':population_identity(config)}: raise ValueError('invalid learned/population asset contract')
        validate_schedules(s,counts['outer_replicates'])
        if s['generator'].get('stream_mapping_version')!=1 or s['generator'].get('bit_generator')!='PCG64': raise ValueError('unknown generator')
        versions=s['versions']
        for key in HASH_FIELDS:
            if not re.fullmatch('[0-9a-f]{64}',str(versions[key])): raise ValueError(f'invalid {key}')
        if not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}',str(versions['git_commit'])): raise ValueError('invalid code revision')
        candidate=content_hash({'selectors':selectors,'fitting_candidates':s['config']['numerics']['fitting_candidates']})
        if versions['candidate_sha256']!=candidate: raise ValueError('candidate hash mismatch')
        runtime=evidence['runtime_identity']
        if versions['runtime_sha256']!=content_hash(runtime): raise ValueError('executed runtime identity mismatch')
        validate_runtime_identity(runtime)
        op=draft['operational']; _positive(op['batch_size'],'batch size')
        _finite(op['reserve_memory_fraction'],'memory reserve',.5)
        if op['reserve_memory_fraction']>=1: raise ValueError('invalid memory reserve')
        if op['thread_policy']!=runtime['thread_environment'] or op['blas_policy']!=content_hash(runtime['builds']): raise ValueError('runtime timing stratum mismatch')
        h=s['precision']['target_half_width']; _finite(h,'precision')
        if h<=0: raise ValueError('positive precision required')
        if evidence.get('dirty_diff_sha256') is not None: raise ValueError('pilot code must be committed before freezing')
        if evidence['phase']!='exploration' or evidence['status']!='completed' or evidence['tuning_resolution']!='resolved': raise ValueError('unresolved pilot')
        if evidence['scientific_contract_sha256']!=scientific_contract(s) or evidence['versions']!=versions: raise ValueError('evidence scientific identity mismatch')
        blocks=evidence['blocks']; seen=set()
        if not isinstance(blocks,list) or len(blocks)<2: raise ValueError('independent pilot blocks required')
        for block in blocks:
            values=block['differences']; keys=block['source_keys']
            if len(values)<30 or len(keys)!=len(values)*(counts['source_training_tasks']+counts['source_validation_tasks']) or len(set(keys))!=len(keys) or seen.intersection(keys): raise ValueError('insufficient or reused pilot units')
            if any(not isinstance(k,str) or not k for k in keys): raise ValueError('invalid pilot keys')
            for value in values: _finite(value,'difference',-math.inf)
            seen.update(keys)
        roots=evidence['excluded_roots']
        if not roots or any(type(r)!=int or r<0 for r in roots) or len(set(roots))!=len(roots): raise ValueError('pilot roots must be excluded')
        source=evidence['source_count_decision']
        if source['status']!='resolved' or source['basis']!='preparation_accuracy_and_cost': raise ValueError('unresolved source accuracy')
        for name in ('source_training_tasks','source_validation_tasks','source_block_size'):
            if source[name]!=counts[name]: raise ValueError('source count evidence mismatch')
        curves=evidence['source_curve']
        if not isinstance(curves,list) or len(curves)<2: raise ValueError('measured source curve required')
        for row in curves:
            for name in ('source_training_tasks','source_validation_tasks','source_block_size','replications'): _positive(row[name],name)
            _finite(row['relative_covariance_error_upper_diagnostic'],'source estimation error')
            _finite(row['preparation_seconds'],'source preparation cost')
            if row['preparation_seconds']<=0 or row['validation_stability'] not in ('stable','unstable','unverified'): raise ValueError('invalid source stability/cost evidence')
        if any(b['source_training_tasks']<2*a['source_training_tasks'] for a,b in zip(curves,curves[1:])): raise ValueError('geometric source-count evidence required')
        from .pilot import choose_source_level
        _finite(source['relative_error_target'],'source error target')
        chosen,resolution=choose_source_level(curves,source['relative_error_target'],sum(len(b['differences']) for b in blocks))
        if resolution!='resolved' or any(chosen[name]!=source[name] for name in ('source_training_tasks','source_validation_tasks','source_block_size')): raise ValueError('source curve does not support chosen counts')
        precision=evidence['precision_diagnostics']
        if precision['status']!='stable' or precision['target_half_width']!=h or precision['alpha']!=.05 or precision['coverage']!='approximate': raise ValueError('unstable precision')
        _positive(precision['planned_outer_replicates'],'planned outer count')
        from .pilot import plan_sample_sizes
        recomputed=plan_sample_sizes({'analysis_mode':mode,'blocks':blocks,**{k:source[k] for k in ('source_training_tasks','source_validation_tasks','source_block_size')}},h,.05)
        if recomputed['decision']!='provisional' or precision['planned_outer_replicates']<recomputed['outer_replicates']: raise ValueError('variance evidence does not support planned count')
        if counts['outer_replicates']<precision['planned_outer_replicates']: raise ValueError('inadequate frozen count')
        cal=evidence['calibration']
        if cal['status']!='adequate' or cal['procedure']!=inference['procedure'] or cal['moment_basis']!=inference['moment_basis'] or cal['reference']!='centered_empirical_pilot': raise ValueError('unsupported moment inference or calibration')
        _positive(cal['replicates'],'calibration replicates')
        if cal['sample_size']!=counts['outer_replicates'] or cal['replicates']<500 or not .90<=cal['coverage']<=1.: raise ValueError('insufficient calibration')
        resource=evidence['resources']
        scaling=resource['scaling']
        if not isinstance(scaling,list) or {r.get('workload') for r in scaling}!={'W1','W2'} or len(scaling)!=2: raise ValueError('W1/W2 calibration required')
        for row in scaling:
            for key in ('preparation_seconds','generation_seconds','peak_rss_bytes'): _finite(row[key],key)
            if row['preparation_seconds']<=0 or row['failures']: raise ValueError('incomplete scaling calibration')
        if resource['status']!='feasible': raise ValueError('unresolved local feasibility')
        for key in ('preparation_seconds','estimated_total_seconds','peak_rss_bytes','available_memory_bytes','memory_budget_bytes'): _finite(resource[key],key)
        _finite(resource['max_seconds'],'runtime budget')
        projection=project_runtime(resource,counts['outer_replicates'],schedule,s.get('simulation_schedule'))+validate_execution_envelope(s,resource)['seconds']
        if resource['max_seconds']<=0 or projection>resource['max_seconds'] or not math.isclose(resource['estimated_total_seconds'],projection,rel_tol=1e-12,abs_tol=1e-9): raise ValueError('runtime projection or declared budget infeasible')
        if len(resource['replication_seconds'])!=sum(len(b['differences']) for b in blocks): raise ValueError('runtime observations do not match pilot schedule')
        if not math.isclose(resource['preparation_seconds'],chosen['preparation_seconds'],rel_tol=1e-12): raise ValueError('source preparation timing mismatch')
        if math.fsum(resource['replication_seconds'])/len(resource['replication_seconds'])<resource['preparation_seconds']: raise ValueError('total replication cost is below its preparation component')
        if resource['preparation_seconds']<=0 or resource['estimated_total_seconds']<=0 or not 0<resource['peak_rss_bytes']<=resource['memory_budget_bytes']<=resource['available_memory_bytes']*(1-op['reserve_memory_fraction']): raise ValueError('memory/time feasibility unverified')
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError(f'incomplete protocol/evidence: {exc}') from exc


def freeze_protocol(draft: JsonRecord, evidence: JsonRecord, output: Path) -> JsonRecord:
    """Validate before allocating a fresh protected namespace; publish without overwrite."""
    draft=copy.deepcopy(draft); evidence=copy.deepcopy(evidence); output=Path(output)
    _validate_draft(draft,evidence)
    if output.exists(): raise FileExistsError(output)
    root=secrets.randbits(128)
    while root in evidence['excluded_roots']: root=secrets.randbits(128)
    manifest={**draft,'status':'frozen','evidence':evidence,
              'scientific_sha256':content_hash(draft['scientific']),
              'operational_sha256':content_hash(draft['operational']),
              'evidence_sha256':content_hash(evidence),
              'protected_keys':{'root_seed':root,'training_phase':101,'validation_phase':102,'test_phase':103,'stream_mapping_version':1}}
    manifest['protocol_sha256']=content_hash(manifest)
    validate_frozen_protocol(manifest)
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.freeze-',dir=output.parent)
    try:
        with os.fdopen(fd,'w') as file:
            json.dump(manifest,file,sort_keys=True,indent=2,allow_nan=False); file.write('\n'); file.flush(); os.fsync(file.fileno())
        os.chmod(name,0o444)
        os.link(name,output)  # Atomic create-if-absent; concurrent freezes cannot replace it.
        directory=os.open(output.parent,os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(name)
    return manifest


def validate_frozen_protocol(manifest: JsonRecord) -> None:
    try:
        if manifest['status']!='frozen': raise ValueError('not frozen')
        _validate_draft({k:manifest[k] for k in ('schema_version','scientific','operational')},manifest['evidence'])
        for key,field in (('scientific_sha256','scientific'),('operational_sha256','operational'),('evidence_sha256','evidence')):
            if manifest[key]!=content_hash(manifest[field]): raise ValueError('content identity changed')
        if manifest['protocol_sha256']!=content_hash({k:v for k,v in manifest.items() if k!='protocol_sha256'}): raise ValueError('protocol identity changed')
        keys=manifest['protected_keys']
        if set(keys)!={'root_seed','training_phase','validation_phase','test_phase','stream_mapping_version'} or type(keys['root_seed'])!=int or keys['root_seed']<0 or keys['root_seed'] in manifest['evidence']['excluded_roots'] or [keys['training_phase'],keys['validation_phase'],keys['test_phase'],keys['stream_mapping_version']] != [101,102,103,1]: raise ValueError('protected/pilot key reuse or unknown mapping')
    except (KeyError,TypeError) as exc: raise ValueError('incomplete frozen protocol') from exc


def checkout_identity(selectors, fitting_candidates, *, runtime=None):
    """Hash actual checked-out scientific source and dependency bytes."""
    root=Path(__file__).resolve().parents[2]
    def digest(path): return hashlib.sha256((root/path).read_bytes()).hexdigest()
    source={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/prepared_learning').glob('*.py'))}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    diff=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines()
    dirty=bool(diff or untracked)
    dirty_hash=content_hash({'diff_sha256':hashlib.sha256(diff).hexdigest(),'untracked':{p:digest(Path(p)) for p in untracked if (root/p).is_file()}}) if dirty else None
    runtime=runtime_identity() if runtime is None else runtime
    validate_runtime_identity(runtime)
    return {'runtime_sha256':content_hash(runtime),'git_commit':commit,'code_sha256':content_hash(source),'dependency_lock_sha256':digest(Path('requirements.lock.txt')),
            'analysis_sha256':digest(Path('src/prepared_learning/analysis.py')),'generator_sha256':digest(Path('src/prepared_learning/tasks.py')),
            'evaluation_sha256':digest(Path('src/prepared_learning/evaluation.py')),'selection_sha256':digest(Path('src/prepared_learning/selection.py')),'candidate_sha256':content_hash({'selectors':selectors,'fitting_candidates':fitting_candidates}),
            'source_manifest_sha256':digest(Path('source_manifest.json')),'preprocessing_sha256':content_hash({'preprocessing':'identity'})},dirty_hash


THREAD_ENVIRONMENT = ('OMP_NUM_THREADS','OMP_DYNAMIC','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS',
                      'MKL_NUM_THREADS','MKL_DYNAMIC','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS',
                      'GOTO_NUM_THREADS','NUMEXPR_NUM_THREADS')


def runtime_identity():
    """Fingerprint executed numerical packages/builds and actual thread controls.

    All installed numerical Python/native-library bytes are hashed, not merely
    wheel RECORD declarations. System BLAS is identified by vendor build metadata
    plus the operating-system build. Unset thread variables are explicit nulls;
    no claim about a vendor's dynamically chosen worker count is inferred.
    """
    import importlib.metadata
    import platform
    import sys
    import numpy
    import scipy
    root=Path(__file__).resolve().parents[2]
    locked={line.split('==')[0]:line.split('==')[1] for line in (root/'requirements.lock.txt').read_text().splitlines() if line and not line.startswith('#')}
    distributions={}
    for name,module in (('numpy',numpy),('scipy',scipy)):
        dist=importlib.metadata.distribution(name)
        if module.__version__!=dist.version or dist.version!=locked.get(name): raise ValueError(f'executed {name} does not match dependency lock')
        installed={}; declarations={}
        for file in dist.files or ():
            if str(file).endswith(('.py','.so','.dylib','.dll','.pyd')):
                path=Path(dist.locate_file(file))
                installed[str(file)]=hashlib.sha256(path.read_bytes()).hexdigest()
                declarations[str(file)]={'hash':{'mode':file.hash.mode,'value':file.hash.value} if file.hash else None,'size':file.size}
        if not installed: raise ValueError(f'cannot fingerprint installed {name}')
        distributions[name]={'version':dist.version,'installed_files_sha256':content_hash(installed),
            'metadata_sha256':hashlib.sha256((dist.read_text('METADATA') or '').encode()).hexdigest(),
            # Ignore installer-specific console shebangs, pyc, and provenance paths.
            # The numerical RECORD subset and its actual bytes define the build.
            'record_sha256':content_hash(declarations)}
    system_build=subprocess.check_output(['sw_vers','-buildVersion'],text=True).strip() if sys.platform=='darwin' else platform.release()
    return {'schema_version':1,'python':{'version':sys.version,'implementation':platform.python_implementation(),
                'cache_tag':sys.implementation.cache_tag,'compiler':platform.python_compiler(),'build':list(platform.python_build()),
                'executable_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()},
            'platform':{'system':platform.system(),'release':platform.release(),'machine':platform.machine(),'build':system_build},
            'distributions':distributions,'builds':{'numpy':numpy.show_config(mode='dicts'),'scipy':scipy.show_config(mode='dicts')},
            'thread_environment':{name:os.environ.get(name) for name in THREAD_ENVIRONMENT},
            'thread_count_interpretation':'actual environment controls; unset means vendor defaults, effective dynamic worker count is not measured'}


def validate_runtime_identity(runtime):
    try:
        if runtime['schema_version']!=1: raise ValueError('invalid runtime identity')
        for name in ('version','implementation','cache_tag','compiler','build'):
            if not runtime['python'][name]: raise ValueError('missing Python build identity')
        if not re.fullmatch('[0-9a-f]{64}',runtime['python']['executable_sha256']): raise ValueError('invalid interpreter hash')
        if not all(runtime['platform'][name] for name in ('system','release','machine','build')): raise ValueError('missing OS build identity')
        for name in ('numpy','scipy'):
            distribution=runtime['distributions'][name]
            if not distribution['version']: raise ValueError('missing executed dependency version')
            for key in ('installed_files_sha256','metadata_sha256','record_sha256'):
                if not re.fullmatch('[0-9a-f]{64}',distribution[key]): raise ValueError('invalid installed dependency hash')
            dependencies=runtime['builds'][name]['Build Dependencies']
            if not dependencies['blas']['found'] or not dependencies['lapack']['found']: raise ValueError('unidentified numerical build')
        if set(runtime['thread_environment'])!=set(THREAD_ENVIRONMENT) or any(v is not None and not isinstance(v,str) for v in runtime['thread_environment'].values()): raise ValueError('missing actual thread controls')
    except (KeyError,TypeError,AttributeError) as exc: raise ValueError('incomplete runtime identity') from exc


def project_runtime(resources, outer_count, update_schedule, simulation_schedule=None):
    """Recompute a mean cost projection for the exact frozen outer/update schedule."""
    _positive(outer_count,'outer count')
    for key in ('replication_seconds','update_check_seconds'):
        values=resources[key]
        if not isinstance(values,list) or len(values)<2: raise ValueError('measured runtime repetitions required')
        for value in values:
            _finite(value,key)
            if value<=0: raise ValueError('positive measured runtime required')
    base=math.fsum(resources['replication_seconds'])/len(resources['replication_seconds'])
    check=math.fsum(resources['update_check_seconds'])/len(resources['update_check_seconds'])
    total=base*outer_count+check*len(update_schedule['outer_replications'])*update_schedule['tasks_per_replication']
    if simulation_schedule:
        values=resources['simulation_task_seconds']
        if len(values)<2 or any(not math.isfinite(v) or v<=0 for v in values): raise ValueError('measured simulation costs required')
        total+=math.fsum(values)/len(values)*len(simulation_schedule['outer_replications'])*simulation_schedule['tasks_per_replication']
    return total


def moment_policy(config):
    """Structural loss-second-moment policy; observed finite losses prove nothing.

    Gaussian pseudoinverse squared loss requires inverse-Wishart second moments
    (dimension gap > 3). This sufficient policy intentionally declines noiseless
    special cases unless their exact cancellation has a separate proof.
    """
    learned=config.sample_plan.analysis_mode in ('nested_task_simulation','conditional_risk_over_preparations')
    n=config.preparation.target_context_size; d=config.preparation.input_dimension
    records={}
    for mid in config.methods:
        family=mid.split(':')[0]; options=config.numerics.get('method_options',{}).get(mid,{})
        eligible=False; basis='unverified random inverse or population moments'
        if family in ('B00','B01','B02','B03','B04','B11') or (not learned and family in ('B05','B06','B07')):
            eligible=True; basis='bounded constrained fixed map and Gaussian task/input/noise moments' if learned else 'fixed finite linear map and Gaussian task/input/noise moments'
        elif mid=='B08:gd':
            eligible=True; basis='finite frozen step grid and Gaussian polynomial moments'
        elif not learned and family in ('B08','B09'):
            penalty=0 if mid=='B09:least_squares' else float(options.get('penalty',0))
            rank=int(options.get('rank',d)) if family=='B09' else d
            if not 1<=rank<=d: raise ValueError('rank outside declared dimension')
            if penalty>0:
                eligible=True; basis='fixed strictly positive ridge penalty bounds inverse'
            elif abs(n-rank)>3:
                eligible=True; basis='Gaussian rectangular pseudoinverse; dimension gap > 3 gives loss second moment'
            else: basis='near-interpolation Gaussian pseudoinverse; loss second moment unsupported'
        elif not learned and family=='B10':
            if config.access.get('prior')=='oracle' and config.workload['sigma2']>0:
                eligible=True; basis='fixed covariance posterior with strictly positive noise'
        records[mid]={'status':'finite_loss_second_moment' if eligible else 'unsupported_or_unverified','basis':basis}
    return {'mean_ci_eligible':bool(records) and all(v['status']=='finite_loss_second_moment' for v in records.values()),
            'estimators':records,'population':'Gaussian theta, isotropic Gaussian X, independent Gaussian noise',
            'finite_observed_losses_are_not_evidence':True}


def validate_schedules(scientific, count):
    for name in ('update_check_schedule','simulation_schedule'):
        schedule=scientific.get(name)
        if schedule is None:
            if name=='update_check_schedule': raise ValueError('fixed update schedule required')
            continue
        indices=schedule['outer_replications']; n=schedule['tasks_per_replication']
        if not isinstance(indices,list) or len(set(indices))!=len(indices) or any(type(j)!=int or not 0<=j<count for j in indices):
            raise ValueError('invalid fixed schedule')
        if type(n)!=int or n<0 or (bool(indices)!=(n>0)) or n>=10**9: raise ValueError('invalid diagnostic task count')


def _validate_mode_draft(draft,evidence):
    """Mode-specific real evidence validation outside the primary learned endpoint."""
    from .tasks import population_identity
    from .confirmation import analyze_results
    import numpy as np
    try:
        s=draft['scientific']; c=config_from_record(s['config']); p=c.sample_plan
        validate_config(c,False)
        if draft['schema_version']!=1 or 'protected_keys' in draft or c.keys: raise ValueError('invalid draft namespace')
        mode=p.analysis_mode
        if mode not in ('deterministic_frontier','fixed_prior_task_simulation','nested_task_simulation'): raise ValueError('unsupported mode')
        required={'deterministic_frontier':(), 'fixed_prior_task_simulation':('tasks_per_replicate',),
                  'nested_task_simulation':('outer_replicates','tasks_per_replicate','source_training_tasks','source_validation_tasks','source_block_size')}[mode]
        for name in ('outer_replicates','tasks_per_replicate','source_training_tasks','source_validation_tasks','source_block_size'):
            value=getattr(p,name)
            if name in required: _positive(value,name)
            elif value is not None: raise ValueError('inapplicable stochastic count')
        if s['assets']['population']!=population_identity(c): raise ValueError('population identity mismatch')
        if s['assets']['preprocessing']!='identity': raise ValueError('unknown preprocessing')
        if not s['amendment'] or not s['scenario_id']: raise ValueError('explicit nonprimary scope required')
        if s['endpoint']!={'metric':'squared_coefficient_error','contrast':'baseline_minus_joint'}: raise ValueError('unsupported endpoint')
        if s['stop_rule']!='fixed_counts_invariants_only': raise ValueError('outcome stopping forbidden')
        if set(s['selectors'])!={'baseline','joint'} or any(not ids or not set(ids)<=set(c.methods) for ids in s['selectors'].values()): raise ValueError('invalid selectors')
        if mode!='nested_task_simulation' and any(len(ids)!=1 for ids in s['selectors'].values()): raise ValueError('fixed selectors required')
        if mode=='nested_task_simulation':
            if c.access.get('prior')!='estimated' or s['assets']['prior']!='refit_each_replication': raise ValueError('source-only nested preparation required')
        else:
            if s['assets']['prior']!='fixed_declared_prior': raise ValueError('fixed prior required')
            from .execution import fixed_preparation
            states=fixed_preparation(c)
            if s['assets']['fixed_prior_hashes']!={key:state.prior.asset_hash for key,(_,state) in states.items()}: raise ValueError('fixed prior identity mismatch')
            if mode=='deterministic_frontier' and any(state.metric is None for _,state in states.values()): raise ValueError('deterministic risk requires fixed maps')
        validate_schedules(s,p.outer_replicates or 1)
        if s['secondary_family']!={'contrasts':[],'adjustment':'holm'}: raise ValueError('nonprimary secondary inference not declared')
        if s['generator']['stream_mapping_version']!=1 or s['generator']['bit_generator']!='PCG64': raise ValueError('unsupported generator')
        versions=s['versions']; runtime=evidence['runtime_identity']; validate_runtime_identity(runtime)
        if versions['runtime_sha256']!=content_hash(runtime): raise ValueError('runtime identity mismatch')
        for key in HASH_FIELDS:
            if not re.fullmatch('[0-9a-f]{64}',str(versions[key])): raise ValueError('invalid version hash')
        if versions['candidate_sha256']!=content_hash({'selectors':s['selectors'],'fitting_candidates':s['config']['numerics'].get('fitting_candidates',{})}): raise ValueError('candidate hash mismatch')
        op=draft['operational']; _positive(op['batch_size'],'batch size')
        if not .5<=op['reserve_memory_fraction']<1 or op['thread_policy']!=runtime['thread_environment'] or op['blas_policy']!=content_hash(runtime['builds']): raise ValueError('runtime stratum mismatch')
        if evidence['dirty_diff_sha256'] is not None or evidence['phase']!='exploration' or evidence['status']!='completed': raise ValueError('incomplete or uncommitted pilot')
        if evidence['versions']!=versions or evidence['scientific_contract_sha256']!=scientific_contract(s): raise ValueError('evidence identity mismatch')
        roots=evidence['excluded_roots']
        if not roots or any(type(r)!=int or r<0 for r in roots) or len(set(roots))!=len(roots): raise ValueError('exploratory roots required')
        policy=moment_policy(c)
        if s['moment_policy']!=policy: raise ValueError('estimator moment policy mismatch')
        inference=s['inference']; procedure=inference['procedure']
        if inference['alpha']!=.05: raise ValueError('unsupported alpha')
        if mode=='deterministic_frontier':
            if procedure!='inapplicable' or inference['coverage'] is not None or s['precision']['target_half_width'] is not None: raise ValueError('deterministic CI/count obligation forbidden')
        elif procedure=='descriptive_only':
            if inference['coverage'] is not None: raise ValueError('descriptive results have no coverage claim')
        elif procedure not in ('student','outer_bootstrap') or not policy['mean_ci_eligible'] or inference['coverage']!='approximate': raise ValueError('unsupported mean inference')
        blocks=evidence['execution_blocks']; expected_blocks=1 if mode=='deterministic_frontier' else 2
        if len(blocks)<expected_blocks: raise ValueError('independent measured pilot blocks required')
        seen=set(); independent=[]; base_costs=[]; task_costs=[]
        for block in blocks:
            local=copy.deepcopy(s); local['config']['sample_plan']=block['sample_plan']
            local['update_check_schedule']=block['update_check_schedule']
            analysis=analyze_results(block['results'],local,roots[0])
            independent.append({'differences':analysis['differences']})
            for result in block['results']:
                keys=result['source_keys']+[t['task_key'] for t in result['tasks']]
                if seen.intersection(keys) or len(keys)!=len(set(keys)): raise ValueError('reused exploratory units')
                seen.update(keys)
                base_costs.append((result['generation_ns']+result['preparation_ns']+result['evaluation_ns'])/1e9)
                task_costs.append(result['task_ns']/1e9/max(1,len(result['tasks'])))
        components=[]
        if mode=='nested_task_simulation':
            from .analysis import nested_variance_components
            for block in blocks:
                rows=[{'outer_replication':r['outer_replication'],'difference':t['values']['baseline']['loss']-t['values']['joint']['loss']} for r in block['results'] for t in r['tasks'] if t['endpoint']=='primary']
                components.append(nested_variance_components(rows))
            if evidence['variance_components']!=components: raise ValueError('variance components differ from measured tasks')
            actual_tuning='resolved' if all(r['tuning_resolution']=='resolved' for b in blocks for r in b['results']) else 'unresolved'
            if evidence['tuning_resolution']!=actual_tuning: raise ValueError('tuning evidence differs from measured candidates')
            if evidence['source_accuracy']['status']=='resolved':
                from .pilot import choose_source_level
                chosen,status=choose_source_level(evidence['source_curve'],evidence['source_accuracy']['relative_error_target'],sum(len(b['results']) for b in blocks))
                if status!='resolved' or any(chosen[name]!=getattr(p,name) for name in ('source_training_tasks','source_validation_tasks','source_block_size')):
                    raise ValueError('source curves do not support frozen counts')
        if procedure in ('student','outer_bootstrap'):
            if mode=='nested_task_simulation' and any(b['sample_plan']['tasks_per_replicate']!=p.tasks_per_replicate for b in blocks): raise ValueError('nested calibration needs the final inner N')
            if any(len(b['differences'])<30 for b in independent): raise ValueError('insufficient independent calibration units')
            cal=evidence['calibration']; n=p.outer_replicates if mode=='nested_task_simulation' else p.tasks_per_replicate
            if cal['status']!='adequate' or cal.get('moment_basis')!='structural_per_estimator' or cal['sample_size']!=n or cal['procedure']!=procedure or cal['replicates']<500 or not .9<=cal['coverage']<=1: raise ValueError('insufficient final-count calibration')
            h=s['precision']['target_half_width']; _finite(h,'precision')
            if h<=0: raise ValueError('positive precision required')
            from .pilot import plan_sample_sizes
            if mode=='fixed_prior_task_simulation':
                plan=plan_sample_sizes({'analysis_mode':mode,'blocks':independent},h,.05)
            else:
                between=[r['between_variance'] for r in components]; within=[r['within_variance'] for r in components]
                stable=min(between)>0 and min(within)>0 and max(between)/min(between)<=4 and max(within)/min(within)<=4
                plan=plan_sample_sizes({'analysis_mode':mode,'variance_between':max(between),'variance_within':max(within),
                    'variance_stable':stable,'preparation_cost':float(np.mean(base_costs)),'task_cost':float(np.mean(task_costs))},h,.05)
                if evidence['source_accuracy']['status']!='resolved' or evidence['tuning_resolution']!='resolved': raise ValueError('nested source accuracy or tuning unresolved')
            if plan['decision']!='provisional' or p.tasks_per_replicate<plan['tasks_per_replicate'] or (mode=='nested_task_simulation' and p.outer_replicates<plan['outer_replicates']): raise ValueError('insufficient frozen allocation')
        elif mode=='nested_task_simulation':
            if evidence['source_accuracy']['status'] not in ('resolved','descriptive_unverified'): raise ValueError('source accuracy status required')
        resource=evidence['resources']; _finite(resource['max_seconds'],'runtime budget')
        tasks=(p.outer_replicates or 1)*(p.tasks_per_replicate or 0)+sum(len(s.get(name,{}).get('outer_replications',[]))*s.get(name,{}).get('tasks_per_replication',0) for name in ('update_check_schedule','simulation_schedule'))
        projection=float(np.mean(base_costs))*(p.outer_replicates or 1)+float(np.mean(task_costs))*tasks+validate_execution_envelope(s,resource)['seconds']
        if resource['status']!='feasible' or resource['max_seconds']<=0 or projection>resource['max_seconds'] or not math.isclose(projection,resource['estimated_total_seconds'],rel_tol=1e-12): raise ValueError('unverified schedule resource projection')
        if not 0<resource['peak_rss_bytes']<=resource['memory_budget_bytes']<=resource['available_memory_bytes']*(1-op['reserve_memory_fraction']): raise ValueError('memory reserve unverified')
    except (KeyError,TypeError,AttributeError) as exc: raise ValueError(f'incomplete mode evidence: {exc}') from exc


def execution_overhead(scientific, resources):
    """Final-count memory and I/O/analysis allowance for the streaming driver.

    Bounds account for bounded JSON copies, scalar inference, bootstrap draws,
    and the small per-block journal inventory, rather than extrapolating pilot
    peak RSS as if the final N were unchanged.
    """
    p=scientific['config']['sample_plan']; j=p['outer_replicates'] or 1
    tasks=j*(p['tasks_per_replicate'] or 0)+sum(len(scientific.get(key,{}).get('outer_replications',[]))*scientific.get(key,{}).get('tasks_per_replication',0) for key in ('update_check_schedule','simulation_schedule'))
    units=p['tasks_per_replicate'] if p['analysis_mode']=='fixed_prior_task_simulation' else j
    measured=resources['execution_envelope']
    if measured['policy']!='paired_chunks_v1': raise ValueError('unknown bounded execution policy')
    for key in ('max_preparation_record_bytes','max_task_record_bytes','serialization_seconds_per_byte','checkpoint_seconds','analysis_seconds_per_unit'):
        _finite(measured[key],key)
        if measured[key]<=0: raise ValueError('measured serialization/analysis envelope required')
    blocks=16*j+2*math.ceil(tasks/16)+4
    analysis=16*units+32*j*len(scientific['secondary_family']['contrasts'])+250000
    memory=8*measured['max_preparation_record_bytes']+8*16*measured['max_task_record_bytes']+analysis+4096*blocks
    # Two passes cover publication and subsequent verified raw-record analysis.
    seconds=2*(j*measured['max_preparation_record_bytes']+tasks*measured['max_task_record_bytes'])*measured['serialization_seconds_per_byte']+blocks*measured['checkpoint_seconds']+units*measured['analysis_seconds_per_unit']
    return {'additional_peak_bytes':memory,'analysis_working_bytes':analysis,'projected_block_inventory':blocks,'seconds':seconds}


def validate_execution_envelope(scientific, resources):
    bounds=execution_overhead(scientific,resources)
    if resources['peak_rss_bytes']+bounds['additional_peak_bytes']>resources['memory_budget_bytes']:
        raise ValueError('final checkpoint, serialization and inference working sets exceed memory reserve')
    return bounds
