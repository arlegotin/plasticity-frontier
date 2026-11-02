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


def scoped_report_case(root, api):
    """Gate unit test only: the receipt is synthetic, never campaign evidence."""
    import hashlib
    descriptors=[]
    for name,kind,raw in (
        ('historical','source',{'status':'failed','error':'retained unsuccessful source attempt'}),
        ('scoped','diagnostic',{'status':'pass','difference':-.4,'interval':[-.5,-.3]}),
        ('historical-diagnostic','diagnostic',{'status':'failed','error':'retained unsuccessful diagnostic attempt'})):
        path=root/(name+'.json');path.write_text(json.dumps(raw))
        descriptor=root/(name+'-input.json');descriptors.append(descriptor)
        descriptor.write_text(json.dumps({'kind':kind,'label':name,'raw_path':str(path)}))
    manifest=api.build_report_inputs(descriptors)
    contract=dict(architecture='declared diagnostic',population='declared toy',baseline='declared control',
                  metric='coefficient squared error',resource_contract='declared tiny shape')
    manifest['claims']=[{'claim_id':'scoped-negative','statement':'negative diagnostic in declared scope',
                        'input_ids':[1],'evidence_level':'numerical_diagnostic',**contract}]
    receipt=root/'unit-test-receipt.json'
    receipt.write_text(json.dumps({'status':'matched','clean_environment':True,'fixture':False,
                                  'input_order_sha256':manifest['input_order_sha256']}))
    manifest['replay']['receipt']={'path':str(receipt),'sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()}
    return manifest


class ReportingTests(unittest.TestCase):
    def test_plot_cache_notice_filter_preserves_real_warnings_and_errors(self):
        api=reporting(self)
        import logging
        logger=logging.getLogger('matplotlib.font_manager');filters=list(logger.filters)
        original=api._csv
        def write_with_notices(path,rows):
            logger.warning('Matplotlib is building the font cache; this may take a moment.')
            logger.warning('distinct font warning remains visible')
            logger.error('font rendering error remains visible')
            return original(path,rows)
        with tempfile.TemporaryDirectory() as tmp, self.assertLogs(logger,level='WARNING') as captured:
            with patch.object(api,'_csv',side_effect=write_with_notices):
                api._figures(Path(tmp),[],[],[])
        self.assertFalse(any('building the font cache' in message for message in captured.output))
        self.assertTrue(any('distinct font warning' in message for message in captured.output))
        self.assertTrue(any('font rendering error' in message for message in captured.output))
        self.assertEqual(logger.filters,filters)

    def test_retained_failed_source_does_not_veto_unrelated_scoped_completion(self):
        api=reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=scoped_report_case(root,api)
            result=api.write_report(manifest,root/'report')
            self.assertEqual(result.code,'SUPPORTED_IN_SCOPE')
            self.assertFalse(result.evidence['blocked_claims'])
            self.assertIn('retained unsuccessful source attempt',(root/'report/results.md').read_text())
            self.assertIn('retained unsuccessful diagnostic attempt',(root/'report/results.md').read_text())
            self.assertIn('-0.4',(root/'report/diagnostics.json').read_text())
            tables=json.loads((root/'report/tables.json').read_text())
            self.assertEqual(tables['statuses'][0]['status'],'failed')
            self.assertEqual(tables['statuses'][2]['status'],'failed')

    def test_failed_support_cannot_establish_a_numerical_claim(self):
        api=reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=scoped_report_case(root,api)
            manifest['claims'][0]['input_ids']=[0]
            with self.assertRaisesRegex(ValueError,'supporting evidence'):
                api.write_report(manifest,root/'report')

    def test_invalid_reconstructed_comparison_blocks_only_its_claim(self):
        api=reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);descriptor,_=confirmation_input(root)
            manifest=api.build_report_inputs([descriptor]);tables=api.regenerate_tables(manifest,root/'tables')
            comparison=tables['comparisons'][0]
            # A completed store is insufficient if its raw inference is invalid.
            comparison.update(status='invalid_or_incomplete',mean=None,interval=None,outcome='unavailable')
            ledger=api._claims(manifest,tables)
            self.assertEqual(ledger[0]['support_status'],'unavailable')
            self.assertEqual(ledger[0]['unavailable_input_ids'],[0])
            manifest['claims']=[{**ledger[0],'claim_id':'overclaim','evidence_level':'protected'}]
            with self.assertRaisesRegex(ValueError,'supporting evidence|protected evidence'):
                api._claims(manifest,tables)

    def test_frontier_cli_records_envelope_preserves_branches_and_failures(self):
        api=reporting(self)
        import contextlib,csv,io
        from prepared_learning.cli import main
        from prepared_learning import frontier
        original=frontier.solve_frontier
        def fail_one_endpoint(prior,budget,task):
            if budget.initialization_norm_cap==.2 and budget.trainable_parameter_cap==3:
                raise RuntimeError('retained endpoint failure')
            return original(prior,budget,task)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=root/'grid.json'
            config.write_text(json.dumps({'schema_version':1,'spectrum':[.6,.4],'n':3,'sigma2':1.,
                'depth':2,'initialization_norm_cap':.2,'trainable_parameter_cap':6,
                'grid':{'widths':[1,2],'norm_caps':[0.,.2],'risk_tolerance':1.,'budget_tolerance':1.,'output':str(root/'grid')}}))
            stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout), patch.object(frontier,'solve_frontier',side_effect=fail_one_endpoint):
                self.assertEqual(main(['frontier','--config',str(config)]),1)
            raw_path=Path(json.loads(stdout.getvalue())['attempt_path'])/'frontier.json'
            raw=json.loads(raw_path.read_text())
            descriptor=root/'input.json';descriptor.write_text(json.dumps({'kind':'diagnostic','raw_path':str(raw_path),'label':'actual grid','fixture':True}))
            result=api.regenerate_tables(api.build_report_inputs([descriptor]),root/'report')
            with (root/'report/frontier.csv').open() as stream: rows=list(csv.DictReader(stream))
            points=[r for r in raw['records'] if r['kind']=='point']
            self.assertTrue(rows,'executed frontier envelope was rendered as an empty panel')
            for point in points:
                for branch in ('F','Z'):
                    row=next(r for r in rows if r['record_kind']=='point' and r['branch']==branch and int(r['width'])==point['width'] and float(r['budget'])==point['norm_cap'])
                    self.assertEqual(json.loads(row['grid_metadata']),raw['metadata'])
                    self.assertEqual(row['point_status'],point['status'])
                    if point[branch] is None:
                        self.assertEqual(row['value'],'');self.assertIn('retained endpoint failure',row['error'])
                    else:
                        self.assertEqual(float(row['value']),point[branch]['objective'])
                        self.assertEqual(row['branch_status'],point[branch]['status'])
                        self.assertEqual(json.loads(row['actual_resources']),point[branch]['actual_resources'])
            self.assertEqual(sum(r['record_kind']=='interval' for r in rows),sum(r['kind']=='interval' for r in raw['records']))
            self.assertNotIn('not_run',(root/'report/frontier.svg').read_text())
            self.assertEqual(next(f for f in result['figures'] if f['csv']=='frontier.csv')['evaluation'],'analytic')

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
            'conditions':{}, 'resources':[{'method_id':'B10:posterior',
                'selected_asset':{'information_access':'oracle'}}],
            'pareto':{'scope':'fixed fixture','rows':[{'method_id':'B10:posterior','comparator':'B01','status':'scoped_domination'}]}}
        ledger=api._claims({'inputs':[{'inventory':{'raw.json':'hash'}}]},
            {'comparisons':[],'benchmarks':[benchmark]})
        self.assertEqual(ledger[0]['evidence_level'],'oracle_reference')


def benchmark_input(root, *, blocks=4, padding=0):
    """Hand-built raw producer-v1 fixture; no fit or scientific execution."""
    from prepared_learning.records import content_hash
    run=root/'benchmark'; (run/'blocks').mkdir(parents=True)
    (run/'requests').mkdir()
    methods=['B01','B05:B01','B08:ridge','B10:empirical_bayes']
    config={'workflow':'generated','phase':'exploratory','dimension':2,'context_size':2,
            'seed':5,'methods':methods,'pilot_blocks':1,'blocks':blocks,
            'max_repetitions':1,'target_block_ns':1,'target_relative_error':.99}
    inputs={'schema_version':1,'inputs':{'manifest':config},
            'methods':[{'method_id':mid,'options':{}} for mid in methods]}
    (run/'inputs.json').write_text(json.dumps(inputs))
    conditions={'cpu':'test-only','threads':1}
    for block in range(blocks):
        workers=[]
        for index,mid in enumerate(methods):
            resource={'method_id':mid,'initialization_norm':1.,
                'trainable_parameters':4 if mid=='B01' else None,'runtime_stored_scalars':4,
                'runtime_serialized_bytes':32+index,'execution_hash':mid+'-exec',
                'selected_execution_hash':mid+'-selected','equality_status':'pass',
                'selected_asset':{'information_access':'source','unknown_asset':{'a':[1,2]}},
                'risk':{'status':'measured','mean':1.+index,'losses':[1.+index,2.+index],
                        'task_keys':['a','b'],'uncertainty_status':'measured'},
                'memory':{'isolation':'fresh_interpreter','peak_rss_bytes':100+block},
                'equality':[{'status':'pass','residual':0.}],
                'diagnostics':{'branch':{'status':'unsupported','detail':'keep diagnostic'}},
                'selection':{'selected_id':'selected','selected_options':{'gamma':0.},
                    'source_hash':'source','validation_hash':'validation','candidate_records':[
                        {'candidate_id':'selected','status':'pass','asset_fit_id':'B01:100',
                         'base_fit_id':'B01:200','loss':1.},
                        {'candidate_id':'failed','status':'failed','shrink_fit_id':'B01:100',
                         'noise_fit_id':'B01:200','unknown':{'payload':'x'*padding},
                         'probes':[{'status':'invalid','reason':'raw nested failure'}]}]}}
            rows=[{'method_id':mid,'execution_hash':mid+'-exec','selected_execution_hash':mid+'-selected',
                   'block':block,'cold_warm':temperature,'method_order':methods,
                   'repetitions':1,'end_to_end_ns':1000+index*10+block,
                   'timing_stratum':content_hash({'conditions':conditions,'cold_warm':temperature,'in_process':False})}
                  for temperature in ('cold','warm')]
            worker={'resource':resource,'rows':rows,'conditions':conditions,
                    'pilot':{'method_id':mid,'samples_ns':[100.],'desired_repetitions':1,'capped':False},
                    'unknown_worker':{'block':block,'status':'failed' if block==blocks-1 else 'pass'}}
            workers.append(worker)
            request={'method':{'method_id':mid},'block':block,'position':index,
                     'method_order':methods,'settings':{'threads':1,'fixed_repetitions':1},'inputs':inputs['inputs']}
            (run/'requests'/f'{block:05d}-{index:03d}.json').write_text(json.dumps(request))
        (run/'blocks'/f'{block:05d}.json').write_text(json.dumps(workers))
    descriptor=root/'input.json'
    descriptor.write_text(json.dumps({'schema_version':1,'kind':'benchmark','run_path':str(run),
                                      'fixture':True,'label':'synthetic raw benchmark'}))
    return descriptor,run


def eager_benchmark(entry):
    """Small test-only reference for the original retained scientific outputs."""
    import copy
    from prepared_learning.benchmark import comparison_table
    workers=[w for p in sorted((Path(entry['run_path'])/'blocks').glob('*.json')) for w in json.loads(p.read_text())]
    resources={}; rows=[]
    for worker in workers:
        rows.extend(worker['rows']); resource=worker['resource']; mid=resource['method_id']
        if mid not in resources: resources[mid]=copy.deepcopy(resource)
        else:
            a,b=resources[mid]['memory']['peak_rss_bytes'],resource['memory']['peak_rss_bytes']
            resources[mid]['memory']['peak_rss_bytes']=max(a,b) if a is not None and b is not None else None
    return {'input_id':entry['input_id'],'label':entry['label'],'status':'completed','evidence_level':'fixture',
            'missing_controls':[],'rows':rows,'resources':list(resources.values()),
            'pareto':comparison_table(rows,list(resources.values()),gate={'status':'pass'},phase='exploratory'),
            'inputs':json.loads((Path(entry['run_path'])/'inputs.json').read_text()),
            'conditions':copy.deepcopy(workers[0]['conditions']), 'raw_workers':workers,
            'energy':{'status':'not_run'},'accelerator':{'status':'not_run'}}


class BoundedBenchmarkReporting(unittest.TestCase):
    def test_benchmark_streaming_matches_eager_tables_and_claims(self):
        api=reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,_=benchmark_input(root)
            manifest=api.build_report_inputs([descriptor]); entry=manifest['inputs'][0]
            expected=eager_benchmark(entry)
            tables=api.regenerate_tables(manifest,root/'report'); actual=tables['benchmarks'][0]
            self.assertEqual(tables.get('report_schema_version'),'benchmark_raw_refs_v2')
            for field in ('rows','pareto','status','missing_controls','evidence_level','conditions','inputs'):
                self.assertEqual(actual[field],expected[field],field)
            for a,b in zip(actual['resources'],expected['resources']):
                for field in ('risk','memory','execution_hash','selected_execution_hash','runtime_serialized_bytes'):
                    self.assertEqual(a[field],b[field])
            self.assertEqual(api._claims(manifest,tables),api._claims(manifest,{'benchmarks':[expected],'comparisons':[]}))
            (root/'eager').mkdir(); api._figures(root/'eager',[],[expected],[])
            for path in (root/'eager').glob('*.csv'):
                self.assertEqual(path.read_text(),(root/'report'/path.name).read_text())

    def test_benchmark_releases_each_full_block_before_next_parse(self):
        api=reporting(self)
        self.assertTrue(hasattr(api,'_retain_benchmark_workers'),'bounded worker retention is missing')
        import gc,weakref
        class TrackedDict(dict): pass
        class TrackedList(list): pass
        refs=[]; reads=[]; original=api._read
        def track(value):
            if isinstance(value,dict): result=TrackedDict({k:track(v) for k,v in value.items()})
            elif isinstance(value,list): result=TrackedList(track(v) for v in value)
            else: return value
            refs.append(weakref.ref(result)); return result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=benchmark_input(root,blocks=12,padding=2048)
            entry=api.build_report_inputs([descriptor])['inputs'][0]
            def read(path):
                if Path(path).parent.resolve()==(run/'blocks').resolve():
                    gc.collect(); self.assertTrue(all(r() is None for r in refs),'prior full block graph still retained')
                    reads.append(str(path)); return track(original(path))
                return original(path)
            with patch.object(api,'_read',side_effect=read):
                # Keyword contract is deliberate: no eager compatibility branch.
                result=api._benchmark(entry,output=root/'report')
            gc.collect(); self.assertTrue(all(r() is None for r in refs)); self.assertEqual(len(reads),12)
            self.assertEqual(len(result['rows']),96)

    def test_benchmark_v1_inputs_v2_reference_package(self):
        api=reporting(self)
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=benchmark_input(root)
            manifest=api.build_report_inputs([descriptor]); tables=api.regenerate_tables(manifest,root/'report')
            summary=tables['benchmarks'][0]
            self.assertNotIn('raw_workers',summary)
            self.assertEqual(summary['report_schema_version'],'benchmark_raw_refs_v2')
            directory=json.loads((root/'report/benchmark_workers.json').read_text())
            self.assertEqual(directory['benchmarks'][0]['worker_records_ref'],summary['worker_records_ref'])
            ref=summary['worker_records_ref']; index=json.loads((root/'report'/ref['path']).read_text())
            self.assertEqual(api._sha(root/'report'/ref['path']),ref['sha256'])
            expected=[w for p in sorted((run/'blocks').glob('*.json')) for w in json.loads(p.read_text())]
            self.assertEqual(index['record_count'],len(expected))
            shutil.move(root/'report',root/'moved')
            for ordinal,(record,worker) in enumerate(zip(index['records'],expected)):
                self.assertEqual(record['ordinal'],ordinal)
                self.assertEqual(record['candidate_count'],2)
                self.assertEqual(record['source_block_sha256'],api._sha(record['source_block_path']))
                self.assertEqual(api._load_worker_ref(record,root/'moved'),worker)
            record=index['records'][0]
            with self.assertRaises(ValueError): api._load_worker_ref(dict(record,path='../input.json'),root/'moved')
            with self.assertRaises(ValueError): api._load_worker_ref(dict(record,sha256='0'*64),root/'moved')
            with self.assertRaises(ValueError): api._load_worker_ref(dict(record,candidate_count=3),root/'moved')
            with self.assertRaises(ValueError): api._load_worker_ref(dict(record,worker_key={'block':99,'method_id':'B01'}),root/'moved')
            for resource in summary['resources']:
                self.assertNotIn('candidate_records',resource['selection'])
                loaded=api._load_worker_ref(resource['resource_ref'],root/'moved')
                self.assertEqual(loaded['resource']['selection']['selected_id'],resource['selection']['selected_id'])
            (root/'moved'/record['path']).unlink()
            with self.assertRaises((ValueError,FileNotFoundError)): api._load_worker_ref(record,root/'moved')

    def test_benchmark_raw_refs_preserve_all_failures_and_unknown_fields(self):
        api=reporting(self)
        self.assertTrue(hasattr(api,'_retain_benchmark_workers'),'bounded worker retention is missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=benchmark_input(root)
            entry=api.build_report_inputs([descriptor])['inputs'][0]; emitted=[]
            result=api._benchmark(entry,output=root/'report',emit=emitted.append)
            records=json.loads((root/'report'/result['worker_records_ref']['path']).read_text())['records']
            for reference in records:
                original=json.loads(Path(reference['source_block_path']).read_text())[int(reference['source_json_pointer'][1:])]
                self.assertEqual(api._load_worker_ref(reference,root/'report'),original)
            texts=json.dumps(emitted)
            self.assertIn('raw nested failure',texts); self.assertIn('keep diagnostic',texts)
            self.assertIn('unknown_worker',texts); self.assertIn('resource_ref',texts)
            # Full candidates cannot survive in compact tables by an alias.
            self.assertNotIn('raw nested failure',json.dumps(result))
            # A failed publish cannot leave a completed index.
            with patch.object(api,'_write',side_effect=OSError('disk full')):
                with self.assertRaises(OSError): api._benchmark(entry,output=root/'failed')
            self.assertFalse((root/'failed/benchmark-input-0/campaign/workers.index.json').exists())

    def test_streamed_benchmark_admission_counts_live_units_and_summaries(self):
        api=reporting(self)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,_=benchmark_input(root,blocks=12,padding=12000)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(160*1024*1024)}):
                manifest=api.build_report_inputs([descriptor])
                api._verify_inputs(manifest,replay=True)
                tables=api.regenerate_tables(manifest,root/'report')
                self.assertEqual(tables['statuses'][0]['status'],'completed')
            entry=manifest['inputs'][0]
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(68*1024*1024)}):
                with self.assertRaises(api.ReportingResourceLimit): api._benchmark(entry,output=root/'small')

    def test_oversized_calibration_rejected_before_parse(self):
        api=reporting(self)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=benchmark_input(root)
            calibration=root/'calibration'/'benchmark.json'; calibration.parent.mkdir()
            (calibration.parent/'inputs.json').write_text('{}')
            with calibration.open('wb') as stream: stream.truncate(4*1024*1024)
            inputs=json.loads((run/'inputs.json').read_text())
            inputs['inputs']['manifest'].update(phase='finalized',frozen={'calibration_path':str(calibration)})
            (run/'inputs.json').write_text(json.dumps(inputs))
            original=api._read
            def read(path):
                self.assertNotEqual(Path(path),calibration,'oversized calibration parsed')
                return original(path)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024*1024)}),patch.object(api,'_read',side_effect=read):
                with self.assertRaises(api.ReportingResourceLimit): api.build_report_inputs([descriptor])

    def test_reference_index_rejects_corruption_and_source_rebinding(self):
        api=reporting(self)
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,_=benchmark_input(root)
            entry=api.build_report_inputs([descriptor])['inputs'][0]
            summary=api._benchmark(entry,output=root/'report'); ref=summary['worker_records_ref']
            path=root/'report'/ref['path']; original=json.loads(path.read_text())
            index=api._load_worker_index(ref,root/'report')
            for mutate in (
                lambda x:x['records'].append(copy.deepcopy(x['records'][0])),
                lambda x:x['records'][1].update(worker_key=x['records'][0]['worker_key']),
                lambda x:x['records'][1].update(source_block_sha256='0'*64),
                lambda x:x['records'][1].update(source_json_pointer=x['records'][0]['source_json_pointer']),
                lambda x:x['records'][0].update(path='../escape.json'),
            ):
                changed=copy.deepcopy(original); mutate(changed); path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): api._load_worker_index(ref,root/'report')
                # Rehashing corrupt metadata cannot bypass structural checks.
                with self.assertRaises(ValueError): api._load_worker_index(dict(ref,sha256=api._sha(path)),root/'report')
            api._write(path,original)
            # Restore original exact JSON formatting/hash, then rebind just the
            # caller's source path: this must fail even for an unchanged worker.
            ref=dict(ref,sha256=api._sha(path))
            record=dict(index['records'][0],source_block_path='unrelated.json')
            with self.assertRaises(ValueError): api._load_worker_ref(record,root/'report')

    def test_retained_summaries_and_risk_are_charged_before_next_parse(self):
        api=reporting(self)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=benchmark_input(root)
            entry=api.build_report_inputs([descriptor])['inputs'][0]
            original=api._read
            def read(path):
                self.assertNotEqual(Path(path).parent.resolve(),(run/'blocks').resolve(),'raw block read despite retained summaries')
                return original(path)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(100*1024*1024)}),patch.object(api,'_read',side_effect=read):
                with self.assertRaises(api.ReportingResourceLimit):
                    api._benchmark(entry,output=root/'report',retained_bytes=lambda:3*1024*1024)
            # A sparse indivisible block is refused without reading it.
            block=next((run/'blocks').glob('*.json'))
            with block.open('wb') as stream: stream.truncate(4*1024*1024)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024*1024)}),patch.object(api,'_read',side_effect=read):
                with self.assertRaises(api.ReportingResourceLimit): api._benchmark(entry,output=root/'large-block')

    def test_multiple_benchmark_inputs_share_retained_summary_reserve(self):
        api=reporting(self)
        from prepared_learning.records import content_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptors=[]
            for i in range(3):
                folder=root/str(i); folder.mkdir(); descriptor,_=benchmark_input(folder,blocks=2)
                descriptors.append(descriptor)
            manifest=api.build_report_inputs(descriptors)
            reserves=[]; original=api._benchmark
            def benchmark(entry,**kwargs):
                reserves.append(kwargs['retained_bytes']())
                return original(entry,**kwargs)
            with patch.object(api,'_benchmark',side_effect=benchmark): api.regenerate_tables(manifest,root/'report')
            self.assertGreater(reserves[1],reserves[0]); self.assertGreater(reserves[2],reserves[1])
            self.assertEqual(content_hash(manifest['inputs']),manifest['input_order_sha256'])

    def test_resource_reserve_counts_retained_key_sets(self):
        api=reporting(self)
        keys={(i,'method-'+str(i)+'x'*10000) for i in range(100)}
        # Index/key membership state remains live between raw reads. Ignoring
        # set members would undercharge over a megabyte here.
        self.assertGreater(api._object_bytes(keys),1000000)

    def test_large_retained_risk_stops_before_compact_allocation(self):
        api=reporting(self)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=benchmark_input(root)
            block=sorted((run/'blocks').glob('*.json'))[0]
            workers=json.loads(block.read_text()); workers[0]['resource']['risk']['losses']=[1.]*10000
            block.write_text(json.dumps(workers))
            entry=api.build_report_inputs([descriptor])['inputs'][0]
            initial=api._guard_paths(api._input_paths(entry),phase='test',streaming=True)
            inputs=api._read(run/'inputs.json')
            cap=initial['estimated_peak_bytes']+16*api._object_bytes(inputs)+1024*1024
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(cap)}), \
                    patch.object(api,'_compact_benchmark_resource',side_effect=AssertionError('oversized risk copied')):
                with self.assertRaises(api.ReportingResourceLimit) as caught: api._benchmark(entry,output=root/'report')
            self.assertEqual(caught.exception.evidence['phase'],'benchmark_summary_before_copy')


    def test_historical_source_provenance_does_not_require_external_files(self):
        api=reporting(self)
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=benchmark_input(root)
            entry=api.build_report_inputs([descriptor])['inputs'][0]
            summary=api._benchmark(entry,output=root/'report')
            index=api._load_worker_index(summary['worker_records_ref'],root/'report')
            reference=index['records'][0]
            expected=api._load_worker_ref(reference,root/'report')
            self.assertEqual(reference.get('source_scope','external_provenance'),'external_provenance')
            shutil.rmtree(run)
            self.assertEqual(api._load_worker_ref(reference,root/'report'),expected)

    def test_generated_source_corruption_is_rejected_before_worker_parse(self):
        api=reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,run=benchmark_input(root)
            worker=json.loads(sorted((run/'blocks').glob('*.json'))[0].read_text())[0]
            package=root/'package'; stdout=package/'processes/worker.stdout.json'; stderr=package/'processes/worker.stderr.txt'
            api._write(stdout,worker); stderr.write_text('complete stderr')
            process=package/'processes/worker.process.json'
            def reference(path):
                return {'path':path.relative_to(package).as_posix(),'sha256':api._sha(path),'size_bytes':path.stat().st_size}
            api._write(process,{'returncode':0,'stdout':reference(stdout),'stderr':reference(stderr)})
            record=api._publish_worker(worker,package,Path('remeasured'),0,source_path=stdout,
                source_sha256=api._sha(stdout),source_pointer='')
            record.update(source_scope='package',source_block_path=stdout.relative_to(package).as_posix(),process_ref=reference(process))
            api._publish_worker_index([record],package,Path('remeasured'))
            original=api._read
            def read(path):
                self.assertNotEqual(Path(path).resolve(),(package/record['path']).resolve(),'copied worker parsed before source verification')
                return original(path)
            for path in (stdout,stderr,process):
                saved=path.read_bytes()
                for missing in (False,True):
                    with self.subTest(path=path.name,missing=missing):
                        if missing: path.unlink()
                        else: path.write_bytes(saved+b'corrupt')
                        with patch.object(api,'_read',side_effect=read):
                            with self.assertRaises((ValueError,FileNotFoundError)): api._load_worker_ref(record,package)
                        path.write_bytes(saved)
