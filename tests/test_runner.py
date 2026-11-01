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
            self.assertEqual(store.committed_keys(), {'x'})
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

    def test_missing_secondary_marker_is_reconstructed(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore.create(Path(d), RUN)
            store.commit_block('x', [], {})
            next(store.path.glob('blocks/*/commit.json')).unlink()
            self.assertEqual(RunStore.open(store.path, IDENTITY).committed_keys(), {'x'})

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
                store = RunStore.open(store.path, IDENTITY)
                if boundary == 'commit.pending':
                    self.assertEqual(store.committed_keys(), {'x'})
                else:
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
            successful = prep_row('a'); successful['paired_risk_difference'] = None
            successful['null_reasons']['paired_risk_difference'] = 'paired method failed'
            rows = [successful, prep_row('b', 'failed')]
            store.commit_block('x', rows, {})
            recovered = RunStore.open(store.path, IDENTITY).read_records('preparation_risks')
            self.assertEqual(recovered, rows)
            from prepared_learning.analysis import paired_differences
            with self.assertRaises(ValueError): paired_differences(recovered, ['outer_replication'], 'a', 'b', 'conditional_risk')

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

    def test_certificate_payload_contents(self):
        from prepared_learning.records import validate_raw_record
        row = {key: None for key in RAW_RECORD_FIELDS['certificates']}
        row.update(kind='certificates', configuration_key='c', branch='F', solver_status='ok', solver_version='v1', solver_options={}, certificate_level='exact_rational', roundoff_treatment='exact', rational_numerators={'a':[1, -2]}, rational_denominators={'a':[2, 3]})
        row['null_reasons'] = {key:'not applicable' for key in RAW_RECORD_FIELDS['certificates'] if row[key] is None}
        validate_raw_record(row, RUN)
        for numerator, denominator in [(['bad'], [1]), ([1], [0]), ([True], [1]), ([1,2], [1]), ([1], [-2]), ({'a':[1]}, {'b':[1]})]:
            with self.subTest(numerator=numerator, denominator=denominator), self.assertRaises(ValueError):
                validate_raw_record(dict(row, rational_numerators=numerator, rational_denominators=denominator), RUN)
        enclosure = dict(row, certificate_level='rigorous_enclosure', roundoff_enclosures={'risk':[0., 1.]})
        validate_raw_record(enclosure, RUN)
        for bad in ({'risk':['x',1.]}, {'risk':[2.,1.]}, {'risk':[0., float('inf')]}, {'risk':[0., 1., 2.]}):
            with self.subTest(enclosure=bad), self.assertRaises(ValueError): validate_raw_record(dict(enclosure, roundoff_enclosures=bad), RUN)

    def test_rename_interruptions_preserve_pair(self):
        from unittest.mock import patch
        import prepared_learning.runner as runner
        for stage in ('block', 'receipt', 'marker'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as d:
                store = RunStore.create(Path(d), RUN)
                rows = [task_row('a'), task_row('b', 'failed')]
                real = runner.os.rename
                def crash(source, destination):
                    source, destination = Path(source), Path(destination)
                    is_block = source.parent.name == 'blocks'
                    is_receipt = source.parent.name == 'journal'
                    is_marker = destination.name == 'commit.json'
                    if {'block':is_block, 'receipt':is_receipt, 'marker':is_marker}[stage]: raise OSError('crash')
                    return real(source, destination)
                with patch.object(runner.os, 'rename', side_effect=crash):
                    with self.assertRaises(OSError): store.commit_block('x', rows, {})
                reopened = RunStore.open(store.path, IDENTITY)
                if stage != 'marker':
                    self.assertEqual(reopened.committed_keys(), set())
                    reopened.commit_block('x', rows, {})
                else: self.assertEqual(reopened.committed_keys(), {'x'})
                self.assertEqual(reopened.read_records('task_metrics'), rows)

class LifecycleTests(unittest.TestCase):
    def test_finalize_is_durable_idempotent_and_seals_blocks(self):
        import json
        self.assertTrue(callable(getattr(RunStore,'finalize',None)))
        with tempfile.TemporaryDirectory() as root:
            store=RunStore.create(Path(root),RUN); store.commit_block('x',[task_row('a'),task_row('b')],{})
            store.finalize('completed'); store.finalize('completed')
            self.assertEqual(json.loads((store.path/'run.json').read_text())['status'],'completed')
            reopened=RunStore.open(store.path,IDENTITY)
            self.assertEqual(len(reopened.read_records('task_metrics')),2)
            with self.assertRaises(ValueError): reopened.commit_block('late',[],{})
            with self.assertRaises(ValueError): reopened.finalize('failed')

    def test_finalize_recovers_interrupted_metadata_publication(self):
        import json
        from unittest.mock import patch
        import prepared_learning.runner as runner
        self.assertTrue(callable(getattr(RunStore,'finalize',None)))
        for destination in ('run.json','run.sha256'):
            with self.subTest(destination=destination),tempfile.TemporaryDirectory() as root:
                store=RunStore.create(Path(root),RUN)
                real=runner.os.replace
                def interrupt(source,target):
                    if Path(target).name==destination: raise OSError('injected lifecycle interruption')
                    return real(source,target)
                with patch.object(runner.os,'replace',side_effect=interrupt):
                    with self.assertRaises(OSError): store.finalize('completed')
                reopened=RunStore.open(store.path,IDENTITY)
                self.assertEqual(json.loads((reopened.path/'run.json').read_text())['status'],'completed')
                self.assertEqual(reopened.committed_keys(),set())

    def test_terminal_metadata_corruption_and_missing_receipt_are_rejected(self):
        import json
        self.assertTrue(callable(getattr(RunStore,'finalize',None)))
        for corrupt in ('metadata','receipt','initial_and_receipt','late_block'):
            with self.subTest(corrupt=corrupt),tempfile.TemporaryDirectory() as root:
                store=RunStore.create(Path(root),RUN); store.finalize('completed')
                if corrupt=='metadata':
                    run=json.loads((store.path/'run.json').read_text()); run['pairing_spec']={}
                    (store.path/'run.json').write_text(json.dumps(run))
                elif corrupt=='receipt': (store.path/'terminal.json').unlink()
                elif corrupt=='initial_and_receipt':
                    (store.path/'terminal.json').unlink(); (store.path/'initial_run.json').unlink()
                else: (store.path/'journal'/'unexpected').write_text('{}')
                with self.assertRaises(ValueError): RunStore.open(store.path,IDENTITY)

class AllocationGuardTests(unittest.TestCase):
    def test_open_and_iteration_guard_before_reading_raw_blocks(self):
        from unittest.mock import patch
        from prepared_learning.execution import ExecutionResourceLimit
        with tempfile.TemporaryDirectory() as root:
            store=RunStore.create(Path(root),RUN)
            store.commit_block('large',[],{'map':np.ones(5)})
            raw=next(store.path.glob('blocks/*/records.json'))
            raw.write_text(' '*1000000)  # Guard must reject before parsing/hashing corrupt payload.
            def check(size=0):
                if size>4*1024*1024: raise ExecutionResourceLimit('bounded before allocation')
            with self.assertRaisesRegex(ExecutionResourceLimit,'before allocation'):
                RunStore.open(store.path,IDENTITY,resource_check=check)
            store.resource_check=check
            with self.assertRaises(ExecutionResourceLimit): list(store.iter_records('execution_task'))

    def test_actual_inventory_is_counted_without_loading_receipts(self):
        from prepared_learning.execution import ExecutionResourceLimit
        with tempfile.TemporaryDirectory() as root:
            store=RunStore.create(Path(root),RUN)
            for index in range(30): (store.path/'journal'/str(index)).write_text('invalid receipt')
            def check(size=0):
                if size>=1024*1024+30*4096: raise ExecutionResourceLimit('inventory reserve')
            with self.assertRaisesRegex(ExecutionResourceLimit,'inventory reserve'):
                RunStore.guard_existing(store.path,check)
