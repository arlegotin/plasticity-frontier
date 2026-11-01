"""Freeze tests use synthetic exploratory evidence, never protected observations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from prepared_learning.records import content_hash


def draft_fixture():
    config = json.loads(Path('configs/exploration/learned_prior.json').read_text())
    config['keys']={}
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
        'stop_rule':'fixed_counts_invariants_only', 'assets': {'prior':'refit_each_replication','preprocessing':'identity'},
    }, 'operational': {'batch_size':1,'reserve_memory_fraction':.5}}


def evidence_fixture(draft):
    scientific = draft['scientific']
    scientific['versions']['git_commit'] = 'b'*40
    scientific['versions']['candidate_sha256'] = content_hash({'selectors':scientific['selectors'], 'fitting_candidates':scientific['config']['numerics']['fitting_candidates']})
    values = [-.02, -.01, .01, .02]*10
    return {'schema_version':1, 'phase':'exploration','status':'completed',
        'scientific_contract_sha256':content_hash({k:v for k,v in scientific.items() if k not in ('precision',)}),
        'blocks':[{'block_id':j,'differences':values,'source_keys':[f'{j}:{i}' for i in range(40*24)]} for j in range(2)],
        'tuning_resolution':'resolved', 'source_count_decision':{'status':'resolved','source_training_tasks':16,'source_validation_tasks':8,'source_block_size':4,'basis':'preparation_accuracy_and_cost','relative_error_target':.5},
        'source_curve':[{'level':0,'source_training_tasks':8,'source_validation_tasks':4,'source_block_size':4,'replications':80,'relative_covariance_error_upper_diagnostic':.9,'validation_stability':'stable','preparation_seconds':.1},
                        {'level':1,'source_training_tasks':16,'source_validation_tasks':8,'source_block_size':4,'replications':80,'relative_covariance_error_upper_diagnostic':.3,'validation_stability':'stable','preparation_seconds':.2}],
        'precision_diagnostics':{'status':'stable','planned_outer_replicates':400,'target_half_width':scientific['precision']['target_half_width'],'alpha':.05,'coverage':'approximate'},
        'calibration':{'status':'adequate','procedure':'student','moment_basis':'bounded_constrained_fixed_maps','replicates':1000,'sample_size':400,'coverage':.952,'reference':'centered_empirical_pilot'},
        'resources':{'status':'feasible','preparation_seconds':.1,'estimated_total_seconds':40.,'peak_rss_bytes':1000,'available_memory_bytes':10000,'memory_budget_bytes':5000, 'scaling':[{'workload':'W1','preparation_seconds':.2,'generation_seconds':.01,'peak_rss_bytes':1000,'failures':{}}, {'workload':'W2','preparation_seconds':.3,'generation_seconds':.01,'peak_rss_bytes':1000,'failures':{}}]},
        'dirty_diff_sha256':None, 'excluded_roots':[42], 'versions': copy.deepcopy(scientific['versions'])}


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
