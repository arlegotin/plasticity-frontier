"""Freeze tests use synthetic exploratory evidence, never protected observations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from functools import lru_cache

@lru_cache(maxsize=1)
def runtime_fixture():
    from prepared_learning.protocol import runtime_identity
    return runtime_identity()
from prepared_learning.records import content_hash


def draft_fixture():
    config = json.loads(Path('configs/exploration/learned_prior.json').read_text())
    config['keys']={}
    config['workload']['population_seed']=42
    from prepared_learning.tasks import population_identity
    from prepared_learning.protocol import config_from_record
    population=population_identity(config_from_record(config))
    config['sample_plan'].update(outer_replicates=400, source_training_tasks=16, source_validation_tasks=8)
    return {'schema_version': 1, 'scientific': {
        'config': config, 'scenario_id': 'primary_richer_k_3_5', 'amendment': None,
        'selectors': {'baseline': ['B01','B02','B11:Z'], 'joint':['B03','B04','B11:F','B11:Z']},
        'endpoint': {'metric':'squared_coefficient_error','contrast':'baseline_minus_joint'},
        'inference': {'procedure':'student','alpha':.05,'coverage':'approximate','moment_basis':'bounded_constrained_fixed_maps'},
        'secondary_family': {'contrasts':[], 'adjustment':'holm'},
        'update_check_schedule': {'outer_replications':[0], 'tasks_per_replication':2},
        'precision': {'target_half_width': .0079770991405094},
        'generator': {'stream_mapping_version':1, 'bit_generator':'PCG64'},
        'versions': {k:'a'*64 for k in ['code_sha256','dependency_lock_sha256','analysis_sha256','generator_sha256','candidate_sha256','selection_sha256','evaluation_sha256','source_manifest_sha256','preprocessing_sha256']},
        'stop_rule':'fixed_counts_invariants_only', 'assets': {'prior':'refit_each_replication','preprocessing':'identity','population':population},
    }, 'operational': {'batch_size':1,'reserve_memory_fraction':.5}}


def evidence_fixture(draft):
    scientific = draft['scientific']
    runtime=copy.deepcopy(runtime_fixture())
    scientific['versions']['runtime_sha256']=content_hash(runtime)
    draft['operational'].update(thread_policy=runtime['thread_environment'],blas_policy=content_hash(runtime['builds']))
    scientific['versions']['git_commit'] = 'b'*40
    scientific['versions']['candidate_sha256'] = content_hash({'selectors':scientific['selectors'], 'fitting_candidates':scientific['config']['numerics']['fitting_candidates']})
    values = [-.02, -.01, .01, .02]*10
    evidence={'schema_version':1, 'phase':'exploration','status':'completed','runtime_identity':runtime,
        'scientific_contract_sha256':content_hash({k:v for k,v in scientific.items() if k not in ('precision',)}),
        'blocks':[{'block_id':j,'differences':values,'source_keys':[f'{j}:{i}' for i in range(40*24)]} for j in range(2)],
        'tuning_resolution':'resolved', 'source_count_decision':{'status':'resolved','source_training_tasks':16,'source_validation_tasks':8,'source_block_size':4,'basis':'preparation_accuracy_and_cost','relative_error_target':.5},
        'source_curve':[{'level':0,'source_training_tasks':8,'source_validation_tasks':4,'source_block_size':4,'replications':80,'relative_covariance_error_upper_diagnostic':.9,'validation_stability':'stable','preparation_seconds':.1},
                        {'level':1,'source_training_tasks':16,'source_validation_tasks':8,'source_block_size':4,'replications':80,'relative_covariance_error_upper_diagnostic':.3,'validation_stability':'stable','preparation_seconds':.2}],
        'precision_diagnostics':{'status':'stable','planned_outer_replicates':400,'target_half_width':scientific['precision']['target_half_width'],'alpha':.05,'coverage':'approximate'},
        'calibration':{'status':'adequate','procedure':'student','moment_basis':'bounded_constrained_fixed_maps','replicates':1000,'sample_size':400,'coverage':.952,'reference':'centered_empirical_pilot'},
        'resources':{'status':'feasible','preparation_seconds':.2,'max_seconds':200.,'replication_seconds':[.25]*80,'update_check_seconds':[.01]*80,'estimated_total_seconds':100.+.01*len(scientific['update_check_schedule']['outer_replications'])*scientific['update_check_schedule']['tasks_per_replication'],'peak_rss_bytes':1000,'available_memory_bytes':10000,'memory_budget_bytes':5000, 'scaling':[{'workload':'W1','preparation_seconds':.2,'generation_seconds':.01,'peak_rss_bytes':1000,'failures':{}}, {'workload':'W2','preparation_seconds':.3,'generation_seconds':.01,'peak_rss_bytes':1000,'failures':{}}]},
        'dirty_diff_sha256':None, 'excluded_roots':[42], 'versions': copy.deepcopy(scientific['versions'])}
    resources=evidence['resources']
    resources.update(available_memory_bytes=10**9,memory_budget_bytes=5*10**8,
        execution_envelope={'policy':'paired_chunks_v1','max_preparation_record_bytes':1000,'max_task_record_bytes':500,
            'serialization_seconds_per_byte':1e-9,'checkpoint_seconds':1e-6,'analysis_seconds_per_unit':1e-6})
    from prepared_learning.protocol import execution_overhead
    resources['estimated_total_seconds']+=execution_overhead(scientific,resources)['seconds']
    return evidence



class ProtocolTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('prepared_learning.protocol'), 'protocol API is missing')
        from prepared_learning import protocol
        return protocol

    def test_freeze_rejects_unresolved_or_leaky_protocol(self):
        api = self.api(); draft = draft_fixture(); evidence = evidence_fixture(draft)
        mutations = [lambda d,e: d['scientific']['config']['sample_plan'].update(outer_replicates=None),
            lambda d,e: d['scientific']['config']['budget'].update(learning_rate=.5),
            lambda d,e: d['scientific']['versions'].pop('analysis_sha256'),
            lambda d,e: d['scientific']['versions'].pop('code_sha256'),
            lambda d,e: d['scientific']['versions'].pop('dependency_lock_sha256'),
            lambda d,e: d['scientific']['versions'].pop('candidate_sha256'),
            lambda d,e: d['scientific']['selectors'].update(baseline=['B05:B02']),
            lambda d,e: d['scientific']['selectors'].update(baseline=['B10:posterior']),
            lambda d,e: d['scientific']['config']['access'].update(prior='oracle'),
            lambda d,e: e.update(tuning_resolution='unresolved'),
            lambda d,e: e['calibration'].update(moment_basis='unverified'),
            lambda d,e: e['blocks'][1].update(source_keys=e['blocks'][0]['source_keys']),
            lambda d,e: d.update(protected_keys={'root_seed':42})]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as root:
                d,e=copy.deepcopy(draft),copy.deepcopy(evidence); mutate(d,e)
                with self.assertRaises(ValueError): api.freeze_protocol(d,e,Path(root)/'frozen.json')
                self.assertFalse((Path(root)/'frozen.json').exists())

    def test_freeze_is_immutable(self):
        api=self.api(); draft=draft_fixture(); evidence=evidence_fixture(draft)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'frozen.json'; frozen=api.freeze_protocol(draft,evidence,path)
            api.validate_frozen_protocol(frozen)
            self.assertNotIn(frozen['protected_keys']['root_seed'], evidence['excluded_roots'])
            self.assertEqual(json.loads(path.read_text()),frozen)
            with self.assertRaises(FileExistsError): api.freeze_protocol(draft,evidence,path)
            altered=copy.deepcopy(frozen); altered['scientific']['config']['workload']['sigma2']=2.
            with self.assertRaises(ValueError): api.validate_frozen_protocol(altered)
            changed=copy.deepcopy(draft); changed['scientific']['update_check_schedule']['tasks_per_replication']=3
            ev=evidence_fixture(changed); other=api.freeze_protocol(changed,ev,Path(root)/'new.json')
            self.assertNotEqual(other['scientific_sha256'],frozen['scientific_sha256'])

    def test_endpoint_required_and_inapplicable_counts(self):
        api=self.api(); draft=draft_fixture(); evidence=evidence_fixture(draft)
        with tempfile.TemporaryDirectory() as root:
            draft['scientific']['config']['sample_plan']['tasks_per_replicate']=2
            with self.assertRaises(ValueError): api.freeze_protocol(draft,evidence,Path(root)/'bad.json')

    def test_rejects_uncommitted_and_incomplete_calibration_evidence(self):
        api=self.api(); draft=draft_fixture(); evidence=evidence_fixture(draft)
        mutations=[lambda d,e:e.update(dirty_diff_sha256='c'*64),
                   lambda d,e:e['resources'].update(scaling=[]),
                   lambda d,e:e['precision_diagnostics'].update(planned_outer_replicates=2),
                   lambda d,e:e.update(excluded_roots=[]),
                   lambda d,e:e.pop('source_curve'),
                   lambda d,e:e['source_curve'][1].update(validation_stability='unstable')]
        for mutate in mutations:
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as root:
                d,e=copy.deepcopy(draft),copy.deepcopy(evidence); mutate(d,e)
                with self.assertRaises(ValueError): api.freeze_protocol(d,e,Path(root)/'bad.json')

    def test_tampered_protected_key_reuse_rejected_even_after_rehash(self):
        api=self.api(); draft=draft_fixture(); evidence=evidence_fixture(draft)
        with tempfile.TemporaryDirectory() as root:
            frozen=api.freeze_protocol(draft,evidence,Path(root)/'frozen.json')
            frozen['protected_keys']['root_seed']=42
            frozen['protocol_sha256']=content_hash({k:v for k,v in frozen.items() if k!='protocol_sha256'})
            with self.assertRaises(ValueError): api.validate_frozen_protocol(frozen)

    def test_rejects_infeasible_or_inconsistent_runtime_projection(self):
        api=self.api()
        for changes in ({'max_seconds':1.,'estimated_total_seconds':1e12},
                        {'max_seconds':None},
                        {'replication_seconds':[10.]*80,'estimated_total_seconds':40.,'max_seconds':100.},
                        {'replication_seconds':[.001]*80,'estimated_total_seconds':.42,'max_seconds':1.}):
            with self.subTest(changes=changes):
                draft=draft_fixture(); evidence=evidence_fixture(draft)
                evidence['resources'].update(changes)
                with self.assertRaises(ValueError): api._validate_draft(draft,evidence)

    def test_source_block_must_match_primary_adaptation_context(self):
        api=self.api(); draft=draft_fixture(); evidence=evidence_fixture(draft)
        draft['scientific']['config']['sample_plan']['source_block_size']=999
        evidence['source_count_decision']['source_block_size']=999
        for row in evidence['source_curve']: row['source_block_size']=999
        evidence['scientific_contract_sha256']=api.scientific_contract(draft['scientific'])
        with self.assertRaisesRegex(ValueError,'source block'): api._validate_draft(draft,evidence)

    def test_runtime_identity_binds_executed_build_and_thread_policy(self):
        api=self.api(); self.assertTrue(callable(getattr(api,'runtime_identity',None)))
        from unittest.mock import patch
        import scipy
        identity=api.runtime_identity()
        self.assertEqual(identity['distributions']['scipy']['version'],scipy.__version__)
        self.assertEqual(len(identity['distributions']['scipy']['installed_files_sha256']),64)
        self.assertTrue(identity['builds']['numpy']['Build Dependencies']['blas']['found'])
        with patch.dict('os.environ',{'OMP_NUM_THREADS':'7'}): changed=api.runtime_identity()
        self.assertNotEqual(content_hash(identity),content_hash(changed))
        base,_=api.checkout_identity({'baseline':['B01'],'joint':['B03']},{})
        with patch.dict('os.environ',{'OMP_NUM_THREADS':'7'}):
            live,_=api.checkout_identity({'baseline':['B01'],'joint':['B03']},{})
        self.assertEqual(base['dependency_lock_sha256'],live['dependency_lock_sha256'])
        self.assertNotEqual(base['runtime_sha256'],live['runtime_sha256'])
        with patch.object(scipy,'__version__','0.0.review-mismatch'):
            with self.assertRaises(ValueError): api.checkout_identity({'baseline':['B01'],'joint':['B03']},{})

    def test_runtime_hash_is_required_and_verified(self):
        api=self.api(); draft=draft_fixture(); evidence=evidence_fixture(draft)
        evidence['runtime_identity']={'tampered':'runtime'}
        with self.assertRaises(ValueError): api._validate_draft(draft,evidence)

    def test_runtime_fingerprint_survives_relocated_environment_and_cwd(self):
        import os
        import subprocess
        import sysconfig
        import venv
        import shutil
        import csv
        import hashlib
        import base64
        api=self.api(); expected=api.runtime_identity()
        source=str(Path('src').resolve()); installed=Path(sysconfig.get_path('purelib'))
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); relocated=root/'relocated-environment'
            venv.EnvBuilder(with_pip=False,symlinks=True).create(relocated)
            python=relocated/'bin/python'
            site=Path(subprocess.check_output([str(python),'-c','import sysconfig; print(sysconfig.get_path("purelib"))'],text=True).strip())
            for entry in installed.iterdir():
                if entry.name.startswith(('numpy','scipy')):
                    if entry.name.endswith('.dist-info'): shutil.copytree(entry,site/entry.name)
                    else: (site/entry.name).symlink_to(entry,target_is_directory=entry.is_dir())
            # pip regenerates console-entry shebangs and their RECORD hashes at
            # the new installation path; these scripts are not numerical runtime.
            for record in site.glob('*.dist-info/RECORD'):
                with record.open(newline='') as file: rows=list(csv.reader(file))
                for row in rows:
                    if '/bin/' in row[0]:
                        payload=('#!'+str(python)+'\n# relocated entry point\n').encode()
                        (site/row[0]).write_bytes(payload)
                        row[1]='sha256='+base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip('=')
                        row[2]=str(len(payload))
                with record.open('w',newline='') as file: csv.writer(file).writerows(rows)
            other_cwd=root/'other-cwd'; other_cwd.mkdir()
            env={**os.environ,'PYTHONPATH':source}
            actual=json.loads(subprocess.check_output([str(python),'-c','import json; from prepared_learning.protocol import runtime_identity; print(json.dumps(runtime_identity()))'],cwd=other_cwd,env=env,text=True))
            self.assertEqual(actual,expected)

class OperationalCheckpointBoundTests(unittest.TestCase):
    def test_requested_chunks_and_endpoint_rounding_bound_inventory(self):
        from prepared_learning.protocol import execution_overhead
        from tests.test_confirmation import tiny
        _,s=tiny('fixed_prior_task_simulation')
        s['config']['sample_plan']['tasks_per_replicate']=1000000
        resources=evidence_fixture(draft_fixture())['resources']
        small=execution_overhead(s,resources,checkpoint_tasks=1,existing_blocks=37)
        self.assertGreaterEqual(small['projected_block_inventory'],2000002+37)
        self.assertGreater(small['seconds'],execution_overhead(s,resources)['seconds'])
        s['config']['sample_plan'].update(analysis_mode='nested_task_simulation',outer_replicates=3,tasks_per_replicate=1)
        s['update_check_schedule']={'outer_replications':[0,1,2],'tasks_per_replication':1}
        bounds=execution_overhead(s,resources,checkpoint_tasks=16)
        # Each replication and endpoint closes its own chunk: six chunks.
        self.assertGreaterEqual(bounds['projected_block_inventory'],16*3+2*6+4)

    def test_recovery_bounds_remaining_chunks_plus_actual_old_inventory(self):
        from prepared_learning.protocol import execution_overhead
        from tests.test_confirmation import tiny
        _,s=tiny('fixed_prior_task_simulation')
        s['config']['sample_plan']['tasks_per_replicate']=1000000
        resources=evidence_fixture(draft_fixture())['resources']
        bounds=execution_overhead(s,resources,checkpoint_tasks=1024,existing_blocks=1999998,
            completed_tasks={(0,'primary'):999999})
        self.assertEqual(bounds['projected_block_inventory'],1999998+16+4+4)
