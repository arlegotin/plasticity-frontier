import tempfile
import unittest
from pathlib import Path
import numpy as np
from prepared_learning.runner import RunStore
from prepared_learning.records import RAW_RECORD_FIELDS

def task_row(method, status='completed'):
    row = {field: None for field in RAW_RECORD_FIELDS['task_metrics']}
    row.update(kind='task_metrics', run_id='r', outer_replication=0, task_id=1, task_key='x', method_id=method, status=status, failure_reason='boom' if status == 'failed' else None, information_access='estimated', n=2, d=3, depth=2, norm_cap=1., parameter_cap=10, eta=1., loss_parameter_mse=1. if status == 'completed' else None)
    row['null_reasons'] = {key: 'not measured' for key in RAW_RECORD_FIELDS['task_metrics'] if row[key] is None and key != 'failure_reason'}
    return row

IDENTITY = {'git_commit':'a'*40, 'dirty_diff_sha256':None, 'scientific_config_sha256':'1'*64, 'protocol_sha256':'2'*64, 'dependency_lock_sha256':'3'*64, 'prior_asset_sha256':None, 'data_manifest_sha256':None}
RUN = dict(IDENTITY, schema_version=1, run_id='r', phase='exploration', operational_config_sha256='4'*64, analysis_mode='conditional_risk_over_preparations', hardware_record='hardware.json', information_access='estimated', timing_boundaries={'adaptation':'elapsed nanoseconds'}, status='running', pairing_spec={'task_metrics': {'unit_fields':['outer_replication','task_id'], 'method_ids':['a','b']}, 'preparation_risks': {'unit_fields':['outer_replication'], 'method_ids':['a','b']}})

class RunnerTests(unittest.TestCase):
    def test_atomic_resume_and_failure_pair(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            rows = [task_row('a'), task_row('b', 'failed')]
            store.commit_block('x', rows, {'map':np.array([1., 2.])})
            store = RunStore.open(store.path, IDENTITY)
            self.assertEqual(store.committed_keys(), {'x'})
            self.assertEqual(store.read_records('task_metrics'), rows)
            with self.assertRaises(ValueError): store.commit_block('x', rows, {})
    def test_corruption_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            store.commit_block('x', [], {'map':np.array([1.])})
            for key in IDENTITY:
                changed = dict(IDENTITY); changed[key] = 'changed'
                with self.assertRaises(ValueError): RunStore.open(store.path, changed)
            asset = next(store.path.glob('blocks/*/assets/*.npy')); asset.write_bytes(b'bad')
            with self.assertRaises(ValueError): RunStore.open(store.path, IDENTITY)

    def test_interrupted_before_marker_reconciles_and_replays(self):
        from unittest.mock import patch
        import prepared_learning.runner as runner
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            real_write = runner._write
            def interrupted(path, data):
                if path.name == 'commit.pending': raise OSError('simulated interruption')
                return real_write(path, data)
            with patch.object(runner, '_write', side_effect=interrupted):
                with self.assertRaises(OSError): store.commit_block('x', [task_row('a', 'failed'), task_row('b')], {})
            store = RunStore.open(store.path, IDENTITY)
            self.assertEqual(store.committed_keys(), set())
            store.commit_block('x', [task_row('a', 'failed'), task_row('b')], {})
            self.assertEqual(len(store.read_records('task_metrics')), 2)

    def test_thread_change_requires_equivalence_and_stratum(self):
        with tempfile.TemporaryDirectory() as d:
            run = dict(RUN, thread_policy='one')
            store = RunStore.create(Path(d), run)
            with self.assertRaises(ValueError): RunStore.open(store.path, dict(IDENTITY, thread_policy='two'))
            with self.assertRaises(ValueError): RunStore.open(store.path, dict(IDENTITY, thread_policy='two', equivalence_verified=True, timing_stratum='two'))

    def test_schema_definitions_include_blueprint_fields(self):
        from prepared_learning.records import RAW_RECORD_FIELDS
        self.assertIn('true_S_evaluator_hash', RAW_RECORD_FIELDS['preparation_risks'])
        self.assertIn('failure_reason', RAW_RECORD_FIELDS['task_metrics'])
        self.assertIn('certificate_level', RAW_RECORD_FIELDS['certificates'])
        self.assertIn('peak_rss_unit', RAW_RECORD_FIELDS['timing_blocks'])
        self.assertIn('methodological_amendment_required', RAW_RECORD_FIELDS['decisions'])

    def test_pairing_and_complete_schemas_required(self):
        from prepared_learning.records import RAW_RECORD_FIELDS
        run = dict(RUN, pairing_spec={'task_metrics': {'unit_fields':['outer_replication','task_id'], 'method_ids':['a','b']}})
        row = task_row('a')
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), run)
            with self.assertRaises(ValueError): store.commit_block('missing', [row], {})
            with self.assertRaises(ValueError): store.commit_block('incomplete', [{'kind':'task_metrics'}], {})
            other = dict(row, method_id='b', status='failed', failure_reason='error')
            store.commit_block('valid', [row, other], {})
            with self.assertRaises(ValueError): store.commit_block('duplicate-unit', [row, other], {})

    def test_missing_committed_marker_is_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            store.commit_block('x', [], {})
            next(store.path.glob('blocks/*/commit.json')).unlink()
            with self.assertRaises(ValueError): RunStore.open(store.path, IDENTITY)

    def test_crash_boundaries_reconcile(self):
        from unittest.mock import patch
        import prepared_learning.runner as runner
        rows = [task_row('a'), task_row('b', 'failed')]
        rows += [dict(task_row(role), task_id=2, task_key='y', loss_parameter_mse=value) for role, value in [('a', 2.), ('b', 1.)]]
        for boundary in ('records.json', 'manifest.json', 'commit.pending', 'journal'):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as d:
                store = RunStore.create(Path(d), RUN)
                real = runner._write
                def crash(path, data):
                    if path.name == boundary or (boundary == 'journal' and path.parent.name == 'journal'):
                        raise OSError('crash')
                    return real(path, data)
                with patch.object(runner, '_write', side_effect=crash):
                    with self.assertRaises(OSError): store.commit_block('x', rows, {'a':np.array([1.])})
                if boundary == 'journal':
                    with self.assertRaises(ValueError): RunStore.open(store.path, IDENTITY)
                    import json
                    persisted = next(store.path.glob('blocks/*/records.json'))
                    self.assertEqual(json.loads(persisted.read_text()), rows)
                else:
                    store = RunStore.open(store.path, IDENTITY)
                    self.assertEqual(store.committed_keys(), set())
                    store.commit_block('x', rows, {'a':np.array([1.])})
                    recovered = store.read_records('task_metrics')
                    self.assertEqual(recovered, rows)
                    from prepared_learning.analysis import paired_differences
                    np.testing.assert_equal(paired_differences(recovered[2:], ['outer_replication','task_id'], 'a', 'b', 'loss_parameter_mse'), [1.])
                    with self.assertRaises(ValueError): paired_differences(recovered, ['outer_replication','task_id'], 'a', 'b', 'loss_parameter_mse')

    def test_pairing_spec_required_and_fixed_units(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): RunStore.create(Path(d), {k:v for k,v in RUN.items() if k != 'pairing_spec'})
            invalid = dict(RUN, pairing_spec={'task_metrics': {'unit_fields':['task_id'], 'method_ids':['a','b']}})
            with self.assertRaises(ValueError): RunStore.create(Path(d), invalid)

    def test_scientific_record_rejects_null_required_fields(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            bad = task_row('a'); bad['task_key'] = None
            with self.assertRaises(ValueError): store.commit_block('bad', [bad, task_row('b')], {})

    def test_operational_change_without_thread_identity_refused(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), dict(RUN, thread_policy='one'))
            with self.assertRaises(ValueError): RunStore.open(store.path, dict(IDENTITY, operational_config_sha256='5'*64))

    def test_run_schema_rejects_fake_hashes_and_invalid_status(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): RunStore.create(Path(d), dict(RUN, scientific_config_sha256='fake'))
            with self.assertRaises(ValueError): RunStore.create(Path(d), dict(RUN, status='made_up'))

    def test_success_record_requires_scientific_values(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            bad = task_row('a'); bad['n'] = None
            with self.assertRaises(ValueError): store.commit_block('bad', [bad, task_row('b')], {})

    def test_tampered_run_record_is_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            path = store.path / 'run.json'; run = json.loads(path.read_text()); run['pairing_spec'] = {}
            path.write_text(json.dumps(run))
            with self.assertRaises(ValueError): RunStore.open(store.path, IDENTITY)

    def test_other_raw_schemas_reject_null_core_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            geometry = {field: None for field in RAW_RECORD_FIELDS['geometry']}; geometry['kind'] = 'geometry'
            with self.assertRaises(ValueError): store.commit_block('bad', [geometry], {})

    def test_pairs_must_share_task_key(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            other = task_row('b'); other['task_key'] = 'different'
            with self.assertRaises(ValueError): store.commit_block('x', [task_row('a'), other], {})

    def test_crash_after_journal_commit_retains_block(self):
        from unittest.mock import patch
        import prepared_learning.runner as runner
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            real = runner._syncdir
            def crash(path):
                if path.name == 'journal' and any(not p.name.startswith('.') for p in path.iterdir()): raise OSError('crash')
                return real(path)
            with patch.object(runner, '_syncdir', side_effect=crash):
                with self.assertRaises(OSError): store.commit_block('x', [task_row('a'), task_row('b', 'failed')], {})
            reopened = RunStore.open(store.path, IDENTITY)
            self.assertEqual(reopened.committed_keys(), {'x'})
            self.assertEqual(reopened.read_records('task_metrics'), [task_row('a'), task_row('b', 'failed')])

    def test_batch_change_logged_when_thread_and_blas_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            run = dict(RUN, thread_policy='one', blas_policy='reference')
            store = RunStore.create(Path(d), run)
            RunStore.open(store.path, dict(IDENTITY, operational_config_sha256='5'*64, thread_policy='one', blas_policy='reference'))
            self.assertIn('5'*64, (store.path / 'operational_changes.jsonl').read_text())

def prep_row(method, status='completed'):
    row = {field: None for field in RAW_RECORD_FIELDS['preparation_risks']}
    row.update(kind='preparation_risks', run_id='r', outer_replication=0, source_key='source', method_id=method,
               selection_procedure_hash='a'*64, prior_asset_hash='b'*64, metric_hash='c'*64 if status == 'completed' else None,
               analysis_mode='conditional_risk_over_preparations', source_training_tasks=3, source_validation_tasks=2,
               total_source_labels=10, conditional_risk=2. if status == 'completed' else None,
               paired_contrast_id='contrast', paired_risk_difference=1. if status == 'completed' else None,
               risk_evaluator_version='v1', true_S_evaluator_hash='d'*64, noise_truth_evaluator_hash='e'*64,
               numerical_error_assessment={'absolute_error': 0.}, status=status, failure_reason='no metric' if status == 'failed' else None)
    row['null_reasons'] = {key: 'unavailable' for key in RAW_RECORD_FIELDS['preparation_risks'] if row[key] is None and key != 'failure_reason'}
    return row

class ReviewRegressionTests(unittest.TestCase):
    def test_missing_journal_preserves_committed_pair(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            rows = [task_row('a'), task_row('b', 'failed')]
            store.commit_block('x', rows, {})
            journal = next(store.path.joinpath('journal').iterdir()); journal.unlink()
            with self.assertRaises(ValueError): RunStore.open(store.path, IDENTITY)
            self.assertEqual(len(list(store.path.glob('blocks/*/records.json'))), 1)

    def test_failed_preparation_without_metric_is_retained(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            rows = [prep_row('a'), prep_row('b', 'failed')]
            store.commit_block('x', rows, {})
            self.assertEqual(RunStore.open(store.path, IDENTITY).read_records('preparation_risks'), rows)

    def test_bad_success_preparation_fields_rejected(self):
        for field, value in [('conditional_risk', None), ('paired_risk_difference', None), ('metric_hash', 'fake'), ('analysis_mode', 'bad'), ('source_training_tasks', -1), ('source_validation_tasks', '2'), ('total_source_labels', True)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                store = RunStore.create(Path(d), RUN); bad = prep_row('a'); bad[field] = value
                with self.assertRaises(ValueError): store.commit_block('x', [bad, prep_row('b')], {})

    def test_invalid_unit_identifiers_rejected(self):
        for value in [0.0, True, '0', -1]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as d:
                store = RunStore.create(Path(d), RUN)
                rows = [task_row('a'), task_row('b')]
                for row in rows: row['outer_replication'] = value
                with self.assertRaises(ValueError): store.commit_block('x', rows, {})

    def test_nonprimary_schema_types_and_units(self):
        from prepared_learning.records import validate_raw_record
        row = {key: None for key in RAW_RECORD_FIELDS['timing_blocks']}
        row.update(kind='timing_blocks', method_order=['a','b'], cold_warm='warm', block_size=2,
                   generation_boundary='ns', preparation_boundary='ns', adaptation_boundary='ns',
                   prediction_boundary='ns', serialization_boundary='ns', repetitions=3, wall_time_ns=10,
                   peak_rss=100, peak_rss_unit='bytes', power_thermal_condition='normal', timing_stratum='cpu',
                   null_reasons={'gpu_metadata':'not applicable'})
        validate_raw_record(row, RUN)
        for field, value in [('repetitions', True), ('wall_time_ns', -1), ('peak_rss_unit','watts'), ('method_order', 'a')]:
            with self.subTest(field=field), self.assertRaises(ValueError): validate_raw_record(dict(row, **{field:value}), RUN)

    def test_run_timing_and_hardware_metadata_are_validated(self):
        with tempfile.TemporaryDirectory() as d:
            for changes in ({'hardware_record': None}, {'timing_boundaries': {}}, {'timing_boundaries': {'adaptation': 5}}):
                with self.subTest(changes=changes), self.assertRaises(ValueError): RunStore.create(Path(d), dict(RUN, **changes))
