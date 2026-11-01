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
    def test_native_json_admission_does_not_charge_raw_analysis_workspaces(self):
        api=reporting(self)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'inventory.json'
            path.write_text(json.dumps({'inventory_note':'x'*(6*1024*1024)}))
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(512*1024*1024)}):
                value=api._read(path)
                self.assertEqual(len(value['inventory_note']),6*1024*1024)
                # Raw analysis still needs the larger transformation/plot bound.
                with self.assertRaises(api.ReportingResourceLimit):
                    api._guard_paths([path],phase='raw_analysis')

    def test_report_limit_precedes_store_open_and_raw_parse(self):
        api=reporting(self)
        from prepared_learning.runner import RunStore
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=confirmation_input(root)
            manifest=api.build_report_inputs([descriptor])
            raw=next((run/'blocks').glob('*/records.json'))
            original=raw.read_bytes()
            # A sparse over-budget ledger need not be valid JSON: the guard
            # must reject its size before parsing or opening the RunStore.
            with raw.open('wb') as stream: stream.truncate(4*1024*1024)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(256*1024*1024)}), \
                    patch.object(RunStore,'open',side_effect=AssertionError('store opened before resource guard')):
                result=api.write_report(manifest,root/'report')
            self.assertEqual(result.code,'LOCAL_RESOURCE_LIMIT')
            self.assertGreaterEqual(result.evidence['json_bytes'],4*1024*1024)
            self.assertGreater(result.evidence['estimated_peak_bytes'],result.evidence['budget_bytes'])
            self.assertEqual(raw.stat().st_size,4*1024*1024)
            self.assertFalse((root/'report/tables.json').exists())
            self.assertTrue((root/'report/resource_limit.json').exists())
            raw.write_bytes(original)
            # A declared modest local cap permits the same valid tiny evidence.
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(1024**3)}):
                tables=api.regenerate_tables(manifest,root/'small')
            self.assertEqual(tables['comparisons'][0]['primary_task_count'],3)
            self.assertIn('resource_preflight',tables)

    def test_builder_and_cli_guard_json_before_loading(self):
        api=reporting(self)
        import os,contextlib,io
        from prepared_learning.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'huge.json'
            with path.open('wb') as stream: stream.truncate(4*1024*1024)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(256*1024*1024)}):
                self.assertTrue(hasattr(api,'ReportingResourceLimit'),'explicit reporting resource limit missing')
                with patch.object(api.json,'loads',side_effect=AssertionError('JSON parsed before size guard')):
                    with self.assertRaises(api.ReportingResourceLimit): api.build_report_inputs([path])
                    for command in ('report','replay'):
                        stdout=io.StringIO()
                        with contextlib.redirect_stdout(stdout):
                            code=main([command,'--manifest',str(path),'--output',str(root/command)])
                        self.assertEqual(code,1)
                        self.assertIn('LOCAL_RESOURCE_LIMIT',stdout.getvalue())

    def test_aggregate_guard_counts_all_inputs_before_loading_any_raw(self):
        api=reporting(self)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptors=[]
            for index in range(2):
                raw=root/f'raw-{index}.json'
                raw.write_text(json.dumps({'status':'recorded','payload':'x'*(2*1024*1024)}))
                descriptor=root/f'input-{index}.json'; descriptors.append(descriptor)
                descriptor.write_text(json.dumps({'kind':'diagnostic','raw_path':str(raw)}))
            manifest=api.build_report_inputs(descriptors)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(768*1024*1024)}):
                for entry in manifest['inputs']:
                    admission=api._guard_paths(api._input_paths(entry),phase='single_input_test')
                    self.assertLess(admission['estimated_peak_bytes'],admission['budget_bytes'])
                with patch.object(api,'_read',side_effect=AssertionError('raw JSON loaded before aggregate guard')):
                    result=api.write_report(manifest,root/'report')
            self.assertEqual(result.code,'LOCAL_RESOURCE_LIMIT')
            self.assertGreaterEqual(result.evidence['json_bytes'],4*1024*1024)

    def test_streamed_confirmation_retains_large_ledger_under_bounded_reserve(self):
        api=reporting(self)
        from prepared_learning.runner import RunStore
        import os,numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,original=confirmation_input(root)
            run=json.loads((original/'run.json').read_text()); run['status']='running'
            store=RunStore.create(root/'streamed',run)
            expected=0; decision=None
            for block in (original/'blocks').iterdir():
                rows=json.loads((block/'records.json').read_text()); expected+=len(rows)
                key=json.loads((block/'commit.json').read_text())['key']
                assets={path.stem:np.load(path,allow_pickle=False) for path in (block/'assets').glob('*.npy')}
                store.commit_block(key,rows,assets)
                for row in rows:
                    if row['kind']=='decisions': decision=row
            for index in range(8):
                store.commit_block(f'retained-history-{index}',[{**decision,'rationale':'x'*(3*1024*1024)}],{})
                expected+=1
            store.finalize('completed')
            value=json.loads(descriptor.read_text()); value['run_path']=str(store.path)
            descriptor.write_text(json.dumps(value))
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(1024**3)}):
                manifest=api.build_report_inputs([descriptor])
                tables=api.regenerate_tables(manifest,root/'report')
                comparison=api.compare_run_stores(store.path,store.path,rtol=1e-10,atol=1e-12)
                self.assertEqual(comparison['status'],'matched')
                self.assertIn('disk index',comparison['record_comparison'])
            self.assertEqual(tables['raw_record_count'],expected)
            self.assertEqual(tables['comparisons'][0]['primary_task_count'],3)
            bounds=tables['resource_preflight']
            self.assertGreater(bounds['json_bytes'],8*bounds['json_working_bytes']*.9)
            self.assertLess(bounds['estimated_peak_bytes'],bounds['budget_bytes'])
            retained=0; histories=0
            with (root/'report/raw_records.json').open() as stream:
                for line in stream:
                    text=line.strip().rstrip(',')
                    if text in ('[',']',''): continue
                    row=json.loads(text); retained+=1
                    if row.get('rationale','').startswith('xxx'):
                        self.assertEqual(len(row['rationale']),3*1024*1024); histories+=1
            self.assertEqual(retained,expected); self.assertEqual(histories,8)


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
            self.assertEqual(first['benchmarks'][0]['evidence_level'],'fixture')

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

    def test_oracle_pareto_row_remains_reference(self):
        api=reporting(self)
        benchmark={'input_id':0,'evidence_level':'operational','inputs':{'inputs':{'manifest':{
            'dimension':2,'context_size':2,'seed':5,'methods':['B10:posterior']}}},
            'raw_workers':[{'conditions':{}}], 'resources':[{'method_id':'B10:posterior',
                'selected_asset':{'information_access':'oracle'}}],
            'pareto':{'scope':'fixed fixture','rows':[{'method_id':'B10:posterior','comparator':'B01','status':'scoped_domination'}]}}
        ledger=api._claims({'inputs':[{'inventory':{'raw.json':'hash'}}]},
            {'comparisons':[],'benchmarks':[benchmark]})
        self.assertEqual(ledger[0]['evidence_level'],'oracle_reference')
