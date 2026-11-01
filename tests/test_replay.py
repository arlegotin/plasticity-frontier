"""Replay identity and arithmetic comparisons on explicitly tiny fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tests.test_reporting import reporting, confirmation_input


class ReplayTests(unittest.TestCase):
    def test_replay_orders_tasks_by_scientific_key_across_many_chunks(self):
        api=reporting(self)
        from prepared_learning import confirmation
        from tests.test_confirmation import execution_manifest
        with tempfile.TemporaryDirectory() as tmp, patch.object(confirmation,'_live_identity'):
            root=Path(tmp); manifest=execution_manifest('fixed_prior_task_simulation')
            manifest['scientific']['config']['sample_plan']['tasks_per_replicate']=19
            a=confirmation.run_confirmation(manifest,root/'a',checkpoint_tasks=1)
            b=confirmation.run_confirmation(manifest,root/'b',checkpoint_tasks=16)
            result=api.compare_run_stores(Path(a.evidence['run_path']),Path(b.evidence['run_path']),rtol=1e-10,atol=1e-12)
            self.assertEqual(result['status'],'matched')

    def test_replay_interrupt_equivalence(self):
        api = reporting(self)
        from prepared_learning import confirmation
        from prepared_learning.runner import RunStore
        from tests.test_confirmation import execution_manifest
        with tempfile.TemporaryDirectory() as tmp, patch.object(confirmation, '_live_identity'):
            root = Path(tmp); manifest = execution_manifest('fixed_prior_task_simulation')
            original = RunStore.commit_block; stopped = []
            def interrupted(store, key, rows, assets):
                original(store, key, rows, assets)
                if key.startswith('tasks-') and not stopped:
                    stopped.append(key); raise KeyboardInterrupt()
            with patch.object(RunStore, 'commit_block', interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    confirmation.run_confirmation(manifest, root/'interrupted', checkpoint_tasks=1)
            resumed = confirmation.run_confirmation(manifest, root/'interrupted', checkpoint_tasks=2)
            fresh = confirmation.run_confirmation(manifest, root/'fresh', checkpoint_tasks=3)
            result = api.compare_run_stores(Path(resumed.evidence['run_path']),
                                            Path(fresh.evidence['run_path']), rtol=1e-10, atol=1e-12)
            self.assertEqual(result['status'], 'matched')
            self.assertGreater(result['arrays_compared'], 0)
            self.assertGreater(result['scientific_records_compared'], 0)
            self.assertEqual(result['timing_policy'], 'remeasured_not_equal')

    def test_replay_uses_recorded_revision_and_lock(self):
        api = reporting(self)
        from prepared_learning._replay import _validate_source
        import subprocess,hashlib
        from prepared_learning.records import content_hash
        repository=Path(__file__).resolve().parents[1]
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
        def git_file(name): return subprocess.check_output(['git','show',revision+':'+name],cwd=repository)
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'src/prepared_learning'],cwd=repository,text=True).splitlines()
        source={'git_commit':revision,'dependency_lock_sha256':hashlib.sha256(git_file('requirements.lock.txt')).hexdigest(),
                'code_sha256':content_hash({n:hashlib.sha256(git_file(n)).hexdigest() for n in names if n.endswith('.py') and Path(n).parent.as_posix()=='src/prepared_learning'})}
        for field,pattern in (('git_commit','revision'),('dependency_lock_sha256','lock'),('code_sha256','source identity')):
            changed=dict(source); changed[field]='0'*64
            with self.assertRaisesRegex(ValueError,pattern): _validate_source(repository,changed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); descriptor, _ = confirmation_input(root)
            manifest = api.build_report_inputs([descriptor])
            for field in ('git_commit', 'dependency_lock_sha256', 'code_sha256'):
                changed = json.loads(json.dumps(manifest))
                changed['replay']['source'][field] = '0'*64
                with self.assertRaisesRegex(ValueError, 'identity|revision|lock|fixture'):
                    api.replay_run(changed, root/field)
            protocol = root/'frozen.json'; value = json.loads(protocol.read_text())
            value['scientific']['assets']['population']['unexpected_prior_change'] = 1
            protocol.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'hash|inventory'):
                api.replay_run(manifest, root/'changed_prior')

    def test_arithmetic_tolerances_do_not_cover_keys_or_counts(self):
        api = reporting(self)
        self.assertTrue(hasattr(api, 'compare_scientific'), 'strict scientific comparator missing')
        api.compare_scientific({'loss': 1.}, {'loss': 1.+1e-13}, rtol=1e-10, atol=1e-12)
        for left, right in (({'task_key':'a'}, {'task_key':'b'}), ({'n':3}, {'n':4}),
                            ({'loss':1.}, {'loss':1.1}), ({'loss':1.}, {})):
            with self.assertRaises(ValueError):
                api.compare_scientific(left, right, rtol=1e-10, atol=1e-12)
