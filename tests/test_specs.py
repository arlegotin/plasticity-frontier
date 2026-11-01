import copy
import json
import tempfile
import unittest
from pathlib import Path

from prepared_learning.specs import load_config, validate_config
from prepared_learning.records import content_hash

BASE = {
    'schema_version': 1,
    'budget': {'depth': 2, 'initialization_norm_cap': 0.6, 'trainable_parameter_cap': 21, 'learning_rate': 1.0},
    'preparation': {'target_context_size': 4, 'input_dimension': 6, 'input_law': 'isotropic_gaussian', 'loss_normalization': 'squared_error_over_2n', 'task_second_moment_trace': 1.0},
    'workload': {'spectrum': [0.5, 0.25, 0.15, 0.07, 0.02, 0.01], 'sigma2': 1.0},
    'numerics': {'dtype': 'float64'}, 'access': {'selection': 'learned_only', 'prior': 'estimated'},
    'methods': ['estimated'], 'keys': {'exploration': 'fixture'}, 'confirmation': {'primary_contrast': 'fixture', 'environment_lock_sha256': 'a'*64},
    'sample_plan': {'analysis_mode': 'conditional_risk_over_preparations', 'outer_replicates': 2, 'tasks_per_replicate': None, 'source_training_tasks': 4, 'source_validation_tasks': 2, 'source_block_size': 1},
}

def load(data):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'config.json'; p.write_text(json.dumps(data))
        return load_config(p)

class ConfigTests(unittest.TestCase):
    def test_confirmation_counts_follow_endpoint(self):
        validate_config(load(BASE), True)
        for mode, counts in [('deterministic_frontier', (None, None)), ('conditional_risk_over_preparations', (2, None))]:
            data = copy.deepcopy(BASE); data['sample_plan'].update(analysis_mode=mode, outer_replicates=counts[0], tasks_per_replicate=counts[1])
            validate_config(load(data), True)
        for mode, outer, inner in [('nested_task_simulation', None, 2), ('nested_task_simulation', 2, None), ('conditional_risk_over_preparations', None, None)]:
            data = copy.deepcopy(BASE); data['sample_plan'].update(analysis_mode=mode, outer_replicates=outer, tasks_per_replicate=inner)
            with self.subTest(mode=mode, outer=outer, inner=inner), self.assertRaises(ValueError): validate_config(load(data), True)

    def test_invalid_scientific_contracts(self):
        cases = [('budget','depth',1), ('budget','learning_rate',0.5), ('budget','initialization_norm_cap',-1), ('budget','initialization_norm_cap',float('inf')), ('budget','trainable_parameter_cap',-1), ('preparation','target_context_size',0), ('workload','sigma2',-1), ('workload','spectrum',[0.5]*6), ('access','prior','oracle')]
        for section, key, value in cases:
            data = copy.deepcopy(BASE); data[section][key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError): validate_config(load(data), True)
        data = copy.deepcopy(BASE); data['budget'].update(initialization_norm_cap=0, trainable_parameter_cap=6)
        with self.assertRaises(ValueError): validate_config(load(data), True)

    def test_confirmation_requires_identity(self):
        for key in ('primary_contrast', 'environment_lock_sha256'):
            data = copy.deepcopy(BASE); del data['confirmation'][key]
            with self.assertRaises(ValueError): validate_config(load(data), True)

    def test_nonstandard_json_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.json'; path.write_text('{"x": NaN}')
            with self.assertRaises(ValueError): load_config(path)
            path.write_text('{"x": Infinity}')
            with self.assertRaises(ValueError): load_config(path)
        self.assertEqual(content_hash({'b': 2, 'a': 1}), content_hash({'a': 1, 'b': 2}))
        with self.assertRaises(ValueError): content_hash({'x': float('nan')})

class ImmutableRecordTests(unittest.TestCase):
    def test_nested_assets_are_copied_frozen_and_serializable(self):
        import numpy as np
        from prepared_learning.records import PriorAsset, content_hash, to_json_record
        array = np.array([1., 2.]); metadata = {'nested': [{'values': array}]}
        asset = PriorAsset(None, None, None, None, 'estimated', metadata, 'fixture')
        identity = content_hash(to_json_record(asset))
        array[0] = 9; metadata['nested'].append('changed')
        self.assertEqual(asset.fit_metadata['nested'][0]['values'][0], 1.)
        self.assertEqual(content_hash(to_json_record(asset)), identity)
        with self.assertRaises(TypeError): asset.fit_metadata['nested'][0]['other'] = 1
        with self.assertRaises((TypeError, AttributeError)): asset.fit_metadata['nested'].append('changed')
        with self.assertRaises(ValueError): asset.fit_metadata['nested'][0]['values'][0] = 8
        with self.assertRaises(ValueError): asset.fit_metadata['nested'][0]['values'].setflags(write=True)

    def test_loaded_config_nested_values_cannot_be_mutated(self):
        config = load(BASE); identity = content_hash(to_json_record_for_test(config))
        with self.assertRaises(TypeError): config.workload['spectrum'][0] = 0.7
        self.assertEqual(identity, content_hash(to_json_record_for_test(config)))

def to_json_record_for_test(record):
    from prepared_learning.records import to_json_record
    return to_json_record(record)
