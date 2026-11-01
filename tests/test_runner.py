import tempfile
import unittest
from pathlib import Path
import numpy as np
from prepared_learning.runner import RunStore
from prepared_learning.records import RAW_RECORD_FIELDS

def task_row(method, status='completed'):
    row = {field: None for field in RAW_RECORD_FIELDS['task_metrics']}
    row.update(kind='task_metrics', run_id='r', outer_replication=0, task_id=1, task_key='x', method_id=method, status=status, failure_reason='boom' if status == 'failed' else None, information_access='estimated', n=2, d=3, depth=2, norm_cap=1., parameter_cap=10, eta=1., loss_parameter_mse=1. if status == 'completed' else None)
    return row

IDENTITY = {'git_commit':'a'*40, 'dirty_diff_sha256':None, 'scientific_config_sha256':'1'*64, 'protocol_sha256':'2'*64, 'dependency_lock_sha256':'3'*64, 'prior_asset_sha256':None, 'data_manifest_sha256':None}
RUN = dict(IDENTITY, schema_version=1, run_id='r', phase='exploration', operational_config_sha256='4'*64, analysis_mode='conditional_risk_over_preparations', hardware_record='hardware.json', information_access='estimated', timing_boundaries={}, status='running', pairing_spec={'task_metrics': {'unit_fields':['outer_replication','task_id'], 'method_ids':['a','b']}, 'preparation_risks': {'unit_fields':['outer_replication'], 'method_ids':['a','b']}})

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
        for boundary in ('records.json', 'manifest.json', 'commit.pending', 'journal'):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as d:
                store = RunStore.create(Path(d), RUN)
                real = runner._write
                def crash(path, data):
                    if path.name == boundary or (boundary == 'journal' and path.parent.name == 'journal'):
                        raise OSError('crash')
                    return real(path, data)
                with patch.object(runner, '_write', side_effect=crash):
                    with self.assertRaises(OSError): store.commit_block('x', [], {'a':np.array([1.])})
                store = RunStore.open(store.path, IDENTITY)
                self.assertEqual(store.committed_keys(), set())
                store.commit_block('x', [], {'a':np.array([1.])})
                self.assertEqual(store.committed_keys(), {'x'})

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
                with self.assertRaises(OSError): store.commit_block('x', [], {})
            self.assertEqual(RunStore.open(store.path, IDENTITY).committed_keys(), {'x'})

    def test_batch_change_logged_when_thread_and_blas_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            run = dict(RUN, thread_policy='one', blas_policy='reference')
            store = RunStore.create(Path(d), run)
            RunStore.open(store.path, dict(IDENTITY, operational_config_sha256='5'*64, thread_policy='one', blas_policy='reference'))
            self.assertIn('5'*64, (store.path / 'operational_changes.jsonl').read_text())
