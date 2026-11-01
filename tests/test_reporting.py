"""Raw reconstruction fixtures; never production protected evidence."""
import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def reporting(test):
    test.assertIsNotNone(importlib.util.find_spec('prepared_learning.reporting'),
                         'raw reporting API is missing')
    return importlib.import_module('prepared_learning.reporting')


def confirmation_input(root, mode='fixed_prior_task_simulation'):
    from tests.test_confirmation import execution_manifest
    from prepared_learning import confirmation
    manifest = execution_manifest(mode)
    with patch.object(confirmation, '_live_identity'):
        gate = confirmation.run_confirmation(manifest, root/'runs', checkpoint_tasks=1)
    protocol = root/'frozen.json'; protocol.write_text(json.dumps(manifest))
    descriptor = root/'input.json'
    descriptor.write_text(json.dumps({'schema_version': 1, 'kind': 'confirmation',
        'run_path': gate.evidence['run_path'], 'protocol_path': str(protocol),
        'fixture': True, 'label': 'tiny fixture'}))
    return descriptor, Path(gate.evidence['run_path'])


class ReportingTests(unittest.TestCase):
    def test_tables_rebuilt_from_raw(self):
        api = reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); descriptor, run = confirmation_input(root)
            manifest = api.build_report_inputs([descriptor])
            first = api.regenerate_tables(manifest, root/'first')
            (run/'gate_decision.json').write_text('{"evidence":{"mean":999999}}')
            (run/'analysis.json').write_text('{"mean":999999}')
            second = api.regenerate_tables(manifest, root/'second')
            self.assertEqual(first['comparisons'], second['comparisons'])
            self.assertEqual(first['comparisons'][0]['primary_task_count'], 3)
            self.assertNotEqual(first['comparisons'][0]['mean'], 999999)
            block = next((run/'blocks').glob('*/records.json'))
            block.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'hash|inventory'):
                api.regenerate_tables(manifest, root/'corrupt')

    def test_claim_requires_evidence_level(self):
        api = reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); descriptor, _ = confirmation_input(root)
            manifest = api.build_report_inputs([descriptor])
            contract = dict(architecture='all-entry depth 2', population='Gaussian fixture',
                            baseline='B01', metric='coefficient MSE', resource_contract='K=.6 P=6')
            for level in ('rigorous_certificate', 'operational', 'protected'):
                manifest['claims'] = [{'claim_id': 'overclaim', 'evidence_level': level,
                                      'input_ids': [0], 'statement': 'strong claim', **contract}]
                with self.assertRaisesRegex(ValueError, 'evidence|claim'):
                    api.write_report(manifest, root/level)
            manifest['claims'] = [{'claim_id': 'incomplete', 'evidence_level': 'exploratory',
                                  'input_ids': [0], 'statement': 'missing contract'}]
            with self.assertRaisesRegex(ValueError, 'contract'):
                api.write_report(manifest, root/'contract')

    def test_negative_and_unrun_results_visible(self):
        api = reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/'raw.json'; raw.write_text(json.dumps({'status': 'failed', 'rows': [
                {'status': 'tie', 'mean': 0}, {'status': 'reversal', 'mean': -.4},
                {'status': 'negative', 'interval': [-.5, -.1]},
                {'status': 'zero_crossing', 'interval': [-.1, .2]},
                {'status': 'failed', 'method_id': 'B10:empirical_bayes'},
                {'status': 'unsuccessful_source_diagnostic'}]}))
            source = root/'source.json'; source.write_text(json.dumps({'kind': 'diagnostic',
                'raw_path': str(raw), 'label': 'source failures', 'evidence_level': 'numerical_diagnostic'}))
            unrun = root/'natural.json'; unrun.write_text(json.dumps({'kind': 'not_run',
                'label': 'natural arm', 'reason': 'official data receipt unavailable'}))
            manifest = api.build_report_inputs([source, unrun])
            api.write_report(manifest, root/'report')
            result = (root/'report/results.md').read_text()
            for text in ('tie', 'reversal', '-0.5', 'zero_crossing', 'B10:empirical_bayes',
                         'unsuccessful_source_diagnostic', 'not_run', 'natural arm'):
                self.assertIn(text, result)
            self.assertEqual(len(api.CHALLENGES), 20)
            for challenge in api.CHALLENGES: self.assertIn(challenge, result)

    def test_plots_are_scientific_artifacts(self):
        api = reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); descriptor, _ = confirmation_input(root)
            manifest = api.build_report_inputs([descriptor])
            result = api.regenerate_tables(manifest, root/'report')
            self.assertTrue(result['figures'])
            for figure in result['figures']:
                self.assertTrue((root/'report'/figure['path']).is_file())
                self.assertTrue((root/'report'/figure['csv']).is_file())
                self.assertTrue(figure['x_unit']); self.assertTrue(figure['y_unit'])
                self.assertIn(figure['evaluation'], ('analytic', 'sampled', 'mixed', 'unavailable'))
                self.assertIn('independent_n', figure)
                self.assertEqual(figure['renderer'], 'matplotlib')

    def test_input_order_and_missing_evidence_are_not_silently_dropped(self):
        api = reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); paths = []
            for label in ('W2', 'W0', 'W1'):
                path = root/(label+'.json'); paths.append(path)
                path.write_text(json.dumps({'kind': 'not_run', 'label': label, 'reason': 'unperformed'}))
            manifest = api.build_report_inputs(paths)
            self.assertEqual([r['label'] for r in manifest['inputs']], ['W2','W0','W1'])
            with self.assertRaises((FileNotFoundError, ValueError)):
                api.build_report_inputs([root/'missing.json'])

    def test_replay_is_not_a_new_independent_sample(self):
        api=reporting(self)
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=confirmation_input(root)
            copy=root/'technical-replay'; shutil.copytree(run,copy)
            second=json.loads(descriptor.read_text()); second['run_path']=str(copy)
            path=root/'second.json'; path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError,'duplicate|independent|replay'):
                api.build_report_inputs([descriptor,path])

    def test_benchmark_tables_ignore_coordinator_summary(self):
        api=reporting(self)
        from tests.test_benchmark import BenchmarkTests
        from prepared_learning.benchmark import benchmark_methods
        from prepared_learning.methods import make_method
        fixture=BenchmarkTests(); fixture.setUp()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/'benchmark'
            benchmark_methods([make_method('B01',{}),make_method('B05:B01',{})],
                fixture.contexts,fixture.config,run)
            path=root/'input.json'; path.write_text(json.dumps({'kind':'benchmark','run_path':str(run),'fixture':True}))
            manifest=api.build_report_inputs([path])
            first=api.regenerate_tables(manifest,root/'first')
            (run/'benchmark.json').write_text('{"resources": [], "summaries": []}')
            second=api.regenerate_tables(manifest,root/'second')
            self.assertEqual(first['benchmarks'],second['benchmarks'])
            self.assertEqual(len(first['benchmarks'][0]['rows']),8)
            self.assertTrue(first['benchmarks'][0]['pareto']['rows'])

    def test_fixture_replay_cannot_complete_production_g10(self):
        api=reporting(self)
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,_=confirmation_input(root)
            manifest=api.build_report_inputs([descriptor])
            receipt=root/'receipt.json'
            receipt.write_text(json.dumps({'status':'matched','clean_environment':True,'fixture':True,
                'input_order_sha256':manifest['input_order_sha256']}))
            manifest['replay']['receipt']={'path':str(receipt),'sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()}
            result=api.write_report(manifest,root/'report')
            self.assertNotEqual(result.code,'SUPPORTED_IN_SCOPE')
