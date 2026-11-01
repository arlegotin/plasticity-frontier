import tempfile
import unittest
from pathlib import Path
import numpy as np
from prepared_learning.benchmark import (normalize_peak_rss, amortized_cost, benchmark_methods, balanced_orders, memory_schedule)
from prepared_learning.methods import make_method
from prepared_learning.records import BudgetSpec, PreparationSpec, PriorAsset, Context


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.prior = PriorAsset(np.eye(2)/2, np.eye(2), np.array([.5,.5]), .1, 'oracle', {}, 'fixture')
        self.budget = BudgetSpec(2, 1., 20, 1.)
        self.preparation = PreparationSpec(2, 2, 'gaussian_isotropic', '1/(2n)', 1.)
        self.contexts = [Context(np.eye(2), np.array([1., 2.]), 'fixture')]
        self.config = {'prior': self.prior, 'budget': self.budget, 'preparation': self.preparation,
                       'pilot_blocks': 2, 'blocks': 2, 'max_repetitions': 2, 'future_tasks': [1,10],
                       'thread_policy': 'environment', 'thermal': 'unknown', 'power': 'unknown',
                       'measure_memory': False}

    def test_rss_units(self):
        self.assertEqual(normalize_peak_rss(1024, 'Darwin'), 1024)
        self.assertEqual(normalize_peak_rss(1024, 'Linux'), 1048576)

    def test_accounting_and_amortization(self):
        self.assertEqual(amortized_cost(100, 10, 5, 10), 25)
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_methods([make_method('B01', {})], self.contexts, self.config, Path(tmp))
        row = result['resources'][0]
        self.assertGreater(row['trainable_parameters'], 0)
        self.assertGreater(row['runtime_serialized_bytes'], 0)
        self.assertGreater(row['evidence_serialized_bytes'], 0)
        self.assertGreaterEqual(row['runtime_stored_scalars'], self.prior.covariance.size + self.prior.basis.size + self.prior.eigenvalues.size)

    def test_timing_boundaries_and_pair_equality(self):
        class Clock:
            def __init__(self): self.t = 0
            def __call__(self): self.t += 10; return self.t
        cfg = dict(self.config, clock=Clock())
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_methods([make_method('B01', {}), make_method('B05:B01', {})], self.contexts, cfg, Path(tmp))
        self.assertEqual(result['equality'][0]['status'], 'pass')
        row = result['timing_blocks'][0]
        for field in ('generation_ns','preparation_ns','adaptation_ns','prediction_ns','serialization_ns','end_to_end_ns'):
            self.assertIn(field, row)
        self.assertIsNone(row['generation_ns'])
        self.assertEqual(result['conditions']['input_residency'], 'provided_contexts')

    def test_order_and_thermal_strata(self):
        orders = balanced_orders(['a','b','c'], 6, 41)
        for position in range(3):
            self.assertEqual({x: sum(order[position] == x for order in orders) for x in 'abc'}, dict.fromkeys('abc', 2))
        self.assertNotEqual(orders[0], orders[1])

    def test_memory_limit_keeps_science(self):
        science = {'d': 50, 'tasks': 100, 'seed': 9}
        result = memory_schedule(science, {'batch_size': 100, 'checkpoint_tasks': 40}, pressure=True)
        self.assertEqual(result['science'], science)
        self.assertLess(result['operations']['batch_size'], 100)

    def test_no_energy_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_methods([make_method('B01', {})], self.contexts, self.config, Path(tmp))
        self.assertEqual(result['energy']['status'], 'not_run')
        self.assertIsNone(result['energy']['joules'])

    def test_shared_metric_mismatch_is_rejected(self):
        from prepared_learning.baselines import metric_from_gains
        other = metric_from_gains(np.eye(2), np.ones(2))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'metric differs'):
                benchmark_methods([make_method('B01', {}), make_method('B05:B01', {'metric': other})], self.contexts, self.config, Path(tmp))

    def test_upstream_measured_and_charged(self):
        cfg = dict(self.config, upstream_prepare=lambda: self.prior)
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_methods([make_method('B01', {})], self.contexts, cfg, Path(tmp))
        self.assertEqual(result['prior_cost']['status'], 'measured')
        self.assertIn('amortized_with_upstream_ns_per_task', result['summaries'][0])
        self.assertEqual(result['inputs']['prior']['asset_hash'], 'fixture')

    def test_cli_exploratory_fixture(self):
        import json
        from prepared_learning.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {'schema_version': 1, 'phase': 'exploratory', 'dimension': 2, 'context_size': 2,
                        'seed': 5, 'methods': ['B01', 'B05:B01', 'B08:ridge'],
                        'pilot_blocks': 2, 'blocks': 2, 'max_repetitions': 2,
                        'output': tmp, 'measure_memory': False}
            path = Path(tmp)/'manifest.json'; path.write_text(json.dumps(manifest))
            self.assertEqual(main(['benchmark', '--manifest', str(path)]), 0)
            result = json.loads((Path(tmp)/'benchmark.json').read_text())
            self.assertEqual(result['equality'][0]['status'], 'pass')
            self.assertGreaterEqual(result['context_generation']['elapsed_ns'], 0)

class IntegrationBenchmarkTests(unittest.TestCase):
    setUp = BenchmarkTests.setUp
    def test_storage_inventory_distinguishes_arrays_aliases_and_evidence(self):
        import prepared_learning.benchmark as b
        self.assertTrue(hasattr(b, 'storage_inventory'), 'actual resident storage inventory missing')
        a = np.zeros((2, 3)); payload = {'weights': a, 'alias': a, 'scalar': 2., 'label': 'evidence'}
        inv = b.storage_inventory(payload)
        self.assertEqual(inv['array_scalars'], 6)
        self.assertEqual(inv['array_bytes'], 48)
        self.assertEqual(inv['scalar_values'], 1)
        self.assertEqual(inv['serialized_array_scalars'], 12)

    def test_conditional_json_config_is_detached_and_unmeasured_full_cost_unavailable(self):
        from prepared_learning.records import to_json_record
        config = to_json_record(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_methods([make_method('B01', {})], self.contexts, config, Path(tmp))
        self.assertIsNone(result['timing_blocks'][0]['end_to_end_ns'])
        self.assertGreater(result['timing_blocks'][0]['conditional_execution_ns'], 0)
        self.assertEqual(result['comparison_gate']['status'], 'unavailable')
        config['prior']['covariance'][0][0] = 55
        self.assertEqual(result['inputs']['prior']['covariance'][0][0], .5)

    def test_fake_clock_charges_exact_nested_boundaries(self):
        from unittest.mock import patch
        import prepared_learning.benchmark as b
        self.assertTrue(hasattr(b, 'measure_operation'), 'explicit operation boundaries missing')
        class Clock:
            t = 0
            def __call__(self): return self.t
            def step(self, amount, value): self.t += amount; return value
        clock = Clock()
        stages = {'generation': lambda: clock.step(3, 'data'),
                  'selection': lambda data: clock.step(5, 'asset'),
                  'preparation': lambda asset: clock.step(7, 'state'),
                  'adaptation': lambda state, data: clock.step(11, 'adapted'),
                  'prediction': lambda adapted, data: clock.step(13, 'prediction'),
                  'serialization': lambda state, adapted: clock.step(17, b'payload')}
        measured = b.measure_operation(stages, clock=clock)
        self.assertEqual(measured['timing'], {'generation_ns':3, 'selection_ns':5, 'preparation_ns':7,
                         'adaptation_ns':11, 'prediction_ns':13, 'serialization_ns':17, 'end_to_end_ns':56})

    def test_full_learned_worker_cold_warm_provenance_pressure_and_schema(self):
        import json
        from prepared_learning.cli import main
        from prepared_learning.records import RAW_RECORD_FIELDS
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {'schema_version':1, 'phase':'exploratory', 'dimension':2, 'context_size':2,
                        'context_count':2, 'seed':19, 'methods':['B01','B05:B01','B10:empirical_bayes'],
                        'source_training_tasks':2, 'source_validation_tasks':2,
                        'fitting_candidates':{'gammas':[0.,1.], 'scalar_fractions':[0.,1.], 'max_refinement_rounds':0},
                        'pilot_blocks':2, 'blocks':3, 'max_repetitions':1, 'target_block_ns':1,
                        'memory_limit_bytes':1, 'batch_size':4, 'checkpoint_tasks':4,
                        'output':tmp, 'threads':1}
            path=Path(tmp)/'manifest.json'; path.write_text(json.dumps(manifest))
            self.assertEqual(main(['benchmark','--manifest',str(path)]),0)
            result=json.loads((Path(tmp)/'benchmark.json').read_text())
            self.assertEqual({r['cold_warm'] for r in result['timing_blocks']},{'cold','warm'})
            self.assertTrue(all(r['end_to_end_ns'] is not None for r in result['timing_blocks']))
            self.assertTrue(all(RAW_RECORD_FIELDS['timing_blocks'] <= set(r) for r in result['timing_blocks']))
            pids={r['process_id'] for r in result['timing_blocks'] if r['cold_warm']=='cold'}
            self.assertEqual(len(pids),9)
            for r in result['resources']:
                self.assertTrue(r['selected_asset']['asset_hash'])
                self.assertTrue(r['selection']['selected_id'])
                self.assertEqual(r['memory']['isolation'],'fresh_interpreter')
                self.assertGreater(r['memory']['peak_rss_bytes'],0)
            self.assertEqual(result['comparison_gate']['status'],'pass')
            self.assertLess(result['operations']['effective']['batch_size'],4)
            self.assertEqual(result['operations']['scientific_hash_before'],result['operations']['scientific_hash_after'])
            self.assertTrue(list((Path(tmp)/'blocks').glob('*.json')))
            self.assertTrue(all(r['generation_ns'] > 0 and r['selection_ns'] > 0 for r in result['timing_blocks']))
            for row in result['timing_blocks']:
                self.assertEqual(len(row['repetition_samples']),row['repetitions'])
                self.assertEqual(row['wall_time_ns'],round(sum(v['end_to_end_ns'] for v in row['repetition_samples'])))

    def test_oracle_arm_preserves_declared_access(self):
        import prepared_learning._benchmark_worker as w
        request={'workflow':'generated','inputs':{'manifest':{'dimension':2,'context_size':2,'seed':5,'methods':['B10:posterior']}},
                 'method':{'method_id':'B10:posterior','options':{}},'methods':[{'method_id':'B10:posterior','options':{}}],
                 'settings':{'batch_size':1}}
        measured=w.operation(request,__import__('time').perf_counter_ns)
        self.assertEqual(measured['state'].prior.information_access,'oracle')
        self.assertIsNotNone(measured['timing']['end_to_end_ns'])

    def test_comparable_peak_excludes_raw_network_diagnostics(self):
        from unittest.mock import patch
        import prepared_learning._benchmark_worker as w
        from prepared_learning.records import to_json_record
        usage=type('Usage',(),{'ru_maxrss':10})()
        request={'workflow':'conditional','inputs':to_json_record({'prior':self.prior,'budget':self.budget,'preparation':self.preparation,'contexts':self.contexts}),
                 'method':{'method_id':'B01','options':{}},'methods':[{'method_id':'B01','options':{}}],
                 'settings':{'batch_size':1,'pilot_blocks':1,'target_block_ns':1,'target_relative_error':.1,'max_repetitions':1},
                 'block':0,'method_order':['B01']}
        real_audit=w._audit
        def expensive_audit(*args):
            result=real_audit(*args); usage.ru_maxrss=10000; return result
        with patch.object(w.resource,'getrusage',return_value=usage), patch.object(w,'_audit',side_effect=expensive_audit), patch.object(w.platform,'system',return_value='Darwin'):
            result=w.run_block(request)
        self.assertEqual(result['resource']['memory']['peak_rss_bytes'],10)
        self.assertTrue(all(r['peak_rss']==10 for r in result['rows']))

    def test_pareto_uses_paired_risk_memory_storage_and_uncertainty(self):
        import prepared_learning.benchmark as b
        resources=[{'method_id':mid,'runtime_serialized_bytes':size,'memory':{'peak_rss_bytes':mem},
                    'risk':{'status':'measured','conditional_exact':True,'mean':risk}}
                   for mid,size,mem,risk in [('a',10,20,.1),('b',20,30,.2)]]
        rows=[{'method_id':mid,'block':block,'timing_stratum':'warm-1','end_to_end_ns':cost}
              for block in range(3) for mid,cost in [('a',10+block),('b',20+block)]]
        result=b.comparison_table(rows,resources,gate={'status':'pass'})
        a=next(r for r in result['rows'] if r['method_id']=='a')
        self.assertEqual(a['status'],'scoped_domination')
        resources[0]['memory']['peak_rss_bytes']=50
        result=b.comparison_table(rows,resources,gate={'status':'pass'})
        self.assertEqual(next(r for r in result['rows'] if r['method_id']=='a')['status'],'tradeoff_or_unresolved')
        result=b.comparison_table(rows,resources,gate={'status':'unavailable'})
        self.assertTrue(all(r['status']=='unavailable' for r in result['rows']))

    def test_amortization_charges_per_task_serialization_each_task(self):
        import prepared_learning.benchmark as b
        self.assertTrue(hasattr(b,'amortized_pipeline_cost'), 'separate source/prepared/task amortization missing')
        cost=b.amortized_pipeline_cost({'source_generation_ns':20,'selection_ns':50,'preparation_ns':100,
                'prepared_serialization_ns':30,'target_generation_ns':4,'adaptation_ns':20,
                'prediction_ns':10,'adapted_serialization_ns':6},future_tasks=10,measured_tasks=2)
        self.assertEqual(cost,40)

    def test_zero_penalty_risk_is_descriptive_not_invented_ci(self):
        import prepared_learning._benchmark_worker as w
        request={'workflow':'generated','inputs':{'manifest':{'dimension':2,'context_size':2,'context_count':3,
                  'seed':5,'prior_mode':'oracle','methods':['B08:ridge']}},
                 'method':{'method_id':'B08:ridge','options':{'penalty':0.}},
                 'methods':[{'method_id':'B08:ridge','options':{'penalty':0.}}], 'settings':{'batch_size':1}}
        measured=w.operation(request,__import__('time').perf_counter_ns)
        risk=w._risk(measured)
        self.assertEqual(risk.get('uncertainty_status'),'descriptive_only')

    def test_pressure_batching_preserves_results_and_reduces_workspace(self):
        import prepared_learning._benchmark_worker as w
        from prepared_learning.records import to_json_record
        contexts=self.contexts*5
        request={'workflow':'conditional','inputs':to_json_record({'prior':self.prior,'budget':self.budget,'preparation':self.preparation,'contexts':contexts}),
                 'method':{'method_id':'B01','options':{}},'methods':[{'method_id':'B01','options':{}}], 'settings':{'batch_size':4}}
        large=w.operation(request,__import__('time').perf_counter_ns)
        request['settings']['batch_size']=1
        small=w.operation(request,__import__('time').perf_counter_ns)
        np.testing.assert_array_equal(large['prediction'],small['prediction'])
        self.assertIn('batch_workspace_bytes',large)
        self.assertGreater(large['batch_workspace_bytes'],small['batch_workspace_bytes'])

    def test_operational_changes_keep_generated_scientific_identity(self):
        import prepared_learning.benchmark as b
        self.assertTrue(hasattr(b,'scientific_configuration'),'scientific/operational projection missing')
        config={'workflow':'generated','dimension':2,'seed':4,'batch_size':4,'checkpoint_tasks':4,'output':'a','threads':1}
        other=dict(config,batch_size=1,checkpoint_tasks=1,output='b',threads=2)
        self.assertEqual(b.scientific_configuration(config),b.scientific_configuration(other))
        other['seed']=5
        self.assertNotEqual(b.scientific_configuration(config),b.scientific_configuration(other))

    def test_finalized_schedule_is_identity_bound_and_never_repilots(self):
        import json
        from unittest.mock import patch
        import prepared_learning.benchmark as b
        import prepared_learning._benchmark_worker as w
        self.assertTrue(hasattr(b,'finalize_benchmark_manifest'),'finalized operational phase missing')
        manifest={'schema_version':1,'phase':'exploratory','workflow':'generated','dimension':2,'context_size':2,
                  'context_count':2,'seed':5,'methods':['B01','B05:B01','B08:ridge','B10:empirical_bayes'],
                  'source_training_tasks':2,'source_validation_tasks':2,'pilot_blocks':2,'blocks':4,
                  'max_repetitions':4,'target_block_ns':1,'target_relative_error':.99,
                  'fitting_candidates':{'gammas':[0.,1.],'scalar_fractions':[0.], 'ridge_multipliers':[1.], 'max_refinement_rounds':0}}
        # Only the checkout identity is substituted: the generated observations,
        # selected assets, process timings, calibration records and hashes are real.
        fixture_identity={'versions':{'git_commit':'0'*40},'dirty_diff_sha256':None,'runtime':{'fixture':'test-only'}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(b,'_benchmark_identity',return_value=fixture_identity):
            root=Path(tmp); calibration=root/'calibration'
            result=b.benchmark_methods([make_method(mid,{}) for mid in manifest['methods']],[],manifest,calibration)
            frozen=b.finalize_benchmark_manifest(calibration/'benchmark.json',root/'frozen.json',root/'campaign')
            self.assertEqual(frozen['phase'],'finalized')
            self.assertEqual(b.validate_campaign_manifest(frozen)['status'],'pass')
            changed=json.loads(json.dumps(frozen)); changed['source_training_tasks']=3
            with self.assertRaisesRegex(ValueError,'configuration'):
                b.validate_campaign_manifest(changed)
            changed=json.loads(json.dumps(frozen)); changed['frozen']['repetitions']['B01']+=1
            with self.assertRaisesRegex(ValueError,'schedule'):
                b.validate_campaign_manifest(changed)
            descriptor={'method_id':'B01','options':{}}
            request={'workflow':'generated','inputs':{'manifest':frozen},'method':descriptor,
                     'methods':[{'method_id':mid,'options':{}} for mid in manifest['methods']],
                     'settings':{'batch_size':1,'pilot_blocks':999,'fixed_repetitions':1,'max_repetitions':4,
                                 'target_block_ns':1,'target_relative_error':.99},'block':0,'method_order':manifest['methods']}
            record=w.run_block(request)
            self.assertEqual(record['pilot']['status'],'frozen_no_recalibration')
            self.assertEqual(record['pilot']['samples_ns'],[])
            self.assertEqual(record['rows'][1]['repetitions'],1)

    def test_direct_reuses_selected_metric_without_canonical_reconstruction(self):
        from unittest.mock import patch
        import prepared_learning._benchmark_worker as w
        from prepared_learning.records import to_json_record
        self.assertTrue(hasattr(w,'_prepare_selected'),'shared selected metric compilation missing')
        target=make_method('B01',{})
        selected_state=target.prepare(self.prior,self.budget,self.preparation)
        selected=(target,self.prior,{'prepared_metric':to_json_record(selected_state.metric)})
        with patch('prepared_learning.methods.construct_canonical',side_effect=AssertionError('direct map constructed a network')):
            prepared=w._prepare_selected(selected,{'method_id':'B05:B01','options':{}},self.budget,self.preparation)
        self.assertIsNone(prepared['state'].initial_layers)
        np.testing.assert_array_equal(prepared['state'].metric.increments,selected_state.metric.increments)

    def test_failed_source_selection_preserves_complete_ledger(self):
        import json
        from prepared_learning.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            manifest={'schema_version':1,'phase':'exploratory','dimension':2,'context_size':2,'seed':7,
                      'source_training_tasks':2,'source_validation_tasks':2,'methods':['B01'],
                      'fitting_candidates':{'scalar_fractions':[2.],'max_refinement_rounds':0},
                      'pilot_blocks':1,'blocks':1,'max_repetitions':1,'output':str(Path(tmp)/'result')}
            path=Path(tmp)/'manifest.json'; path.write_text(json.dumps(manifest))
            self.assertEqual(main(['benchmark','--manifest',str(path)]),1)
            failures=list((Path(tmp)/'result'/'requests').glob('*.failure.json'))
            self.assertEqual(len(failures),1)
            details=json.loads(failures[0].read_text())['details']
            self.assertTrue(details['candidate_records'])
            self.assertTrue(all(r['status']=='failed' for r in details['candidate_records']))

    def test_equality_failure_prevents_publishing_timing_comparison(self):
        from unittest.mock import patch
        import prepared_learning._benchmark_worker as w
        from prepared_learning.records import AdaptedState
        real=w._compact_adapt
        def broken(*args):
            values,workspace=real(*args)
            return [AdaptedState(a.coefficient+1.,None,'completed',None) for a in values],workspace
        with tempfile.TemporaryDirectory() as tmp, patch.object(w,'_compact_adapt',side_effect=broken):
            with self.assertRaisesRegex(ValueError,'equality failed'):
                benchmark_methods([make_method('B01',{})],self.contexts,dict(self.config,clock=__import__('time').perf_counter_ns),Path(tmp))
            self.assertFalse((Path(tmp)/'benchmark.json').exists())

    def test_failed_stage_retains_measured_elapsed_boundaries(self):
        from prepared_learning.benchmark import measure_operation
        class Clock:
            t=0
            def __call__(self): return self.t
        clock=Clock()
        def generate(): clock.t+=3; return 'data'
        def select(data): clock.t+=5; raise ValueError('fit failed')
        try:
            measure_operation({'generation':generate,'selection':select},clock=clock)
        except ValueError as exc:
            self.assertEqual(getattr(exc,'timing_failure',None),{'failed_stage':'selection','generation_ns':3,'selection_ns':5,'end_to_end_ns':8})
        else: self.fail('source failure not propagated')

    def test_generated_manifest_cannot_override_main_scientific_guards(self):
        from prepared_learning._benchmark_worker import generated_config
        base={'dimension':2,'context_size':2,'seed':5,'methods':['B01']}
        for changed in ({'dimension':0},{'context_size':0},{'learning_rate':2.},{'input_law':'student_t'},{'dtype':'float32'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError): generated_config(dict(base,**changed))

    def test_only_explicit_untracked_frozen_manifest_is_allowlisted(self):
        import json
        from unittest.mock import patch
        import prepared_learning.benchmark as b
        self.assertTrue(hasattr(b,'_allow_manifest_only'),'explicit frozen-manifest allowance missing')
        identity={'dirty_diff_sha256':'dirty','versions':{'git_commit':'0'*40}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frozen.json'; manifest={'schema_version':1,'phase':'finalized'}; path.write_text(json.dumps(manifest))
            def checkout(args,**kwargs):
                return b'' if args[1]=='diff' else str(path).encode()+b'\0'
            with patch.object(b.subprocess,'check_output',side_effect=checkout):
                self.assertIsNone(b._allow_manifest_only(identity,manifest,path)['dirty_diff_sha256'])
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError,'manifest'):
                    b._allow_manifest_only(identity,manifest,path)
            with patch.object(b.subprocess,'check_output',return_value=b'tracked change'):
                with self.assertRaisesRegex(ValueError,'uncommitted'):
                    b._allow_manifest_only(identity,manifest,path)
