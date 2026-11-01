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
               'source_manifest_sha256', 'preprocessing_sha256')


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
        if config.access.get('prior') != 'estimated' or config.access.get('selection') != 'learned_only':
            raise ValueError('this learned protocol requires observed-source information access')
        selectors=s['selectors']
        if set(selectors) != {'baseline','joint'}: raise ValueError('both selectors required')
        for name,allowed in (('baseline',BASELINE),('joint',JOINT)):
            ids=selectors[name]
            if not isinstance(ids,list) or not ids or len(set(ids))!=len(ids) or not set(ids)<=set(allowed): raise ValueError('ineligible primary comparator')
        if set(config.methods)!=set(selectors['baseline'])|set(selectors['joint']): raise ValueError('method and selector candidates differ')
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
        schedule=s['update_check_schedule']; _positive(schedule['tasks_per_replication'],'update checks')
        if not isinstance(schedule['outer_replications'],list) or not schedule['outer_replications'] or any(type(j)!=int or not 0<=j<counts['outer_replicates'] for j in schedule['outer_replications']) or len(set(schedule['outer_replications']))!=len(schedule['outer_replications']): raise ValueError('invalid fixed update schedule')
        if s['stop_rule']!='fixed_counts_invariants_only': raise ValueError('outcome-dependent stopping forbidden')
        if s['assets']!={'prior':'refit_each_replication','preprocessing':'identity'}: raise ValueError('invalid learned asset contract')
        if s['generator'].get('stream_mapping_version')!=1 or s['generator'].get('bit_generator')!='PCG64': raise ValueError('unknown generator')
        versions=s['versions']
        for key in HASH_FIELDS:
            if not re.fullmatch('[0-9a-f]{64}',str(versions[key])): raise ValueError(f'invalid {key}')
        if not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}',str(versions['git_commit'])): raise ValueError('invalid code revision')
        candidate=content_hash({'selectors':selectors,'fitting_candidates':s['config']['numerics']['fitting_candidates']})
        if versions['candidate_sha256']!=candidate: raise ValueError('candidate hash mismatch')
        op=draft['operational']; _positive(op['batch_size'],'batch size')
        _finite(op['reserve_memory_fraction'],'memory reserve',.5)
        if op['reserve_memory_fraction']>=1: raise ValueError('invalid memory reserve')
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


def checkout_identity(selectors, fitting_candidates):
    """Hash actual checked-out scientific source and dependency bytes."""
    root=Path(__file__).resolve().parents[2]
    def digest(path): return hashlib.sha256((root/path).read_bytes()).hexdigest()
    source={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/prepared_learning').glob('*.py'))}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    diff=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines()
    dirty=bool(diff or untracked)
    dirty_hash=content_hash({'diff_sha256':hashlib.sha256(diff).hexdigest(),'untracked':{p:digest(Path(p)) for p in untracked if (root/p).is_file()}}) if dirty else None
    return {'git_commit':commit,'code_sha256':content_hash(source),'dependency_lock_sha256':digest(Path('requirements.lock.txt')),
            'analysis_sha256':digest(Path('src/prepared_learning/analysis.py')),'generator_sha256':digest(Path('src/prepared_learning/tasks.py')),
            'evaluation_sha256':digest(Path('src/prepared_learning/evaluation.py')),'selection_sha256':digest(Path('src/prepared_learning/selection.py')),'candidate_sha256':content_hash({'selectors':selectors,'fitting_candidates':fitting_candidates}),
            'source_manifest_sha256':digest(Path('source_manifest.json')),'preprocessing_sha256':content_hash({'preprocessing':'identity'})},dirty_hash
