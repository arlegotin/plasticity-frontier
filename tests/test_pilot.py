import unittest
from fractions import Fraction

from prepared_learning.pilot import plan_sample_sizes, INITIAL_HALF_WIDTH


class _PilotPayload(dict):
    """A JSON-compatible graph whose lifetime can be observed without owning it."""


def _pilot_preparation(config, root, outer, phase, methods, *, missing_joint=False):
    import numpy as np
    level = (phase-10)//2 if phase < 70 else 2
    replication = outer % 3
    payload = _PilotPayload(candidate_detail='x'*(128*1024), probe={'eligible': False})
    parent = np.full(100000, 1.+replication/10)
    selected = {}
    for role, loss in (('baseline', 1.+replication/10), ('joint', 1.2+replication/10)):
        if missing_joint and role == 'joint':
            continue
        selected[role] = _PilotPayload(
            candidate_id=role, method_id='B02' if role == 'baseline' else 'B04',
            asset_hash='a'*64, metric_hash='b'*64, validation_loss=loss,
            validation_task_losses=parent[:config.sample_plan.source_validation_tasks],
            prepared_state=payload, numerical_error_assessment={'level': 'float64_diagnostic'})
    risks = {'baseline': 1.+(replication+1)/100, 'joint': 1.}
    if missing_joint:
        risks.pop('joint')
    result = _PilotPayload(
        selected=selected, risks=risks, failures={'joint': 'missing candidate'} if missing_joint else {},
        candidate_records=[payload, {'status': 'failed', 'failure_reason': 'retained failure'}],
        selection_records=[payload], restoration_failures=[], family_risks={'baseline': payload},
        source_keys=[f'{root}/{phase}/{outer}/train', f'{root}/{phase}/{outer}/validation'],
        source_key='c'*64, training_observation_hash='d'*64, validation_observation_hash='e'*64,
        relative_covariance_error=.9 if level == 0 else .2, eigenspace_projection_error=.1,
        tuning_resolution='resolved', generation_ns=1000000, preparation_ns=2000000*(level+1),
        diagnostic_generation_ns=100, diagnostic_adaptation_ns=200, diagnostic_prediction_ns=300,
        simulation_task_seconds=.00001, method_order=list(methods),
        truth_second_moment_hash='f'*64, truth_noise_hash='0'*64, peak_rss_bytes=1000,
        tasks=[{'values': {'baseline': payload}}], diagnostic_values={'baseline': payload})
    return result, {'baseline_metric': np.arange(16, dtype=float)}, (payload, parent, *selected.values())


class CompactPilotTests(unittest.TestCase):
    def test_compact_record_detaches_validation_views_and_missing_roles(self):
        import gc
        import weakref
        import prepared_learning.pilot as pilot
        result, assets, graphs = _pilot_preparation(PilotIntegrationTests.config(), 123, 0, 10, ['B02'], missing_joint=True)
        result.update(level=0, block_id=0, replication=0, outer_replication=0, difference=None, serialization_ns=42)
        refs = [weakref.ref(graph) for graph in (*graphs, result)]
        compact = pilot._compact_pilot_replication(result)
        self.assertEqual(compact['selected'], {'baseline': {'validation_loss': 1., 'validation_task_losses': [1., 1.]}})
        self.assertEqual(compact['risks'], {'baseline': 1.01})
        self.assertIsNone(compact['difference'])
        self.assertEqual(compact['failures'], {'joint': 'missing candidate'})
        self.assertEqual(compact['source_keys'], ['123/10/0/train', '123/10/0/validation'])
        self.assertEqual(set(compact), set(('level', 'block_id', 'replication', 'outer_replication', 'source_keys', 'risks', 'difference', 'tuning_resolution', 'failures', 'relative_covariance_error', 'eigenspace_projection_error', 'selected', 'generation_ns', 'preparation_ns', 'serialization_ns', 'diagnostic_generation_ns', 'diagnostic_adaptation_ns', 'diagnostic_prediction_ns', 'simulation_task_seconds')))
        result['source_keys'].append('mutation')
        result['failures']['baseline'] = 'mutation'
        del result, assets, graphs
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(len(compact['source_keys']), 2)
        self.assertEqual(compact['failures'], {'joint': 'missing candidate'})

    def test_envelope_folds_full_record_maxima_and_preserves_empty_contract(self):
        import json
        from prepared_learning.records import to_json_record
        import prepared_learning.pilot as pilot
        envelopes = []
        for outer, elapsed in ((0, 1000), (1, 900000)):
            full, _, _ = _pilot_preparation(PilotIntegrationTests.config(), 123, outer, 10, ['B02'])
            full['serialization_ns'] = elapsed
            envelope = pilot.measure_execution_envelope([full], [])
            prep = len(json.dumps(to_json_record({**full, 'tasks': []}), allow_nan=False).encode())
            task = 4*len(json.dumps(to_json_record(full['tasks'][0]), allow_nan=False).encode())
            self.assertEqual(envelope['max_preparation_record_bytes'], prep)
            self.assertEqual(envelope['max_task_record_bytes'], task)
            self.assertEqual(envelope['serialization_seconds_per_byte'], elapsed/1e9/(prep+task))
            envelopes.append(envelope)
        merged = pilot._merge_pilot_envelopes(envelopes)
        for key in ('max_preparation_record_bytes', 'max_task_record_bytes', 'serialization_seconds_per_byte', 'checkpoint_seconds', 'analysis_seconds_per_unit'):
            self.assertEqual(merged[key], max(e[key] for e in envelopes))
        self.assertEqual(merged['policy'], 'paired_chunks_v1')
        empty = pilot._merge_pilot_envelopes([])
        self.assertEqual(empty['max_preparation_record_bytes'], 1)
        self.assertEqual(empty['max_task_record_bytes'], 1)

    def test_primary_releases_every_graph_before_next_preparation_including_scaling(self):
        import gc
        import tempfile
        import weakref
        from dataclasses import replace
        from pathlib import Path
        from unittest.mock import patch
        import prepared_learning.pilot as pilot
        config = PilotIntegrationTests.config()
        config = replace(config, numerics={**config.numerics, 'pilot': {**config.numerics['pilot'], 'workloads': ['W0', 'W1', 'W2']}})
        refs = []
        phases = []
        def prepare(*args):
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs), f'previous graph retained before phase {args[3]}')
            full, assets, graphs = _pilot_preparation(*args)
            refs[:] = [weakref.ref(graph) for graph in (*graphs, full, *assets.values())]
            phases.append(args[3])
            return full, assets
        with tempfile.TemporaryDirectory() as root, patch.object(pilot, '_preparation_replication', side_effect=prepare):
            pilot.run_pilot(config, Path(root))
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(phases, [10]*6+[12]*6+[70, 72])

    def test_primary_preserves_full_publication_science_and_selected_level_envelope(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        import prepared_learning.pilot as pilot
        from prepared_learning.runner import RunStore
        from prepared_learning.records import to_json_record
        expected_raw = {}
        envelopes = {0: [], 1: []}
        measure = pilot.measure_execution_envelope
        compact = pilot._compact_pilot_replication
        def prepare(*args):
            full, assets, _ = _pilot_preparation(*args)
            expected_raw[args[2]] = to_json_record(full)
            return full, assets
        def measured(results, blocks):
            observed = measure(results, blocks)
            for full in results:
                self.assertIn('candidate_records', full)
                envelopes[full['level']].append(observed)
            return observed
        def compact_after_publication(full):
            path = next(Path(root).glob('pilot-*'))
            outer = full['outer_replication']
            self.assertTrue((path/f'preparation-{outer}.json').is_file())
            self.assertEqual(len(list((path/'blocks').glob('*/assets/baseline_metric.npy'))), outer+1)
            self.assertEqual(len(RunStore(path).read_records('preparation_risks')), 2*(outer+1))
            self.assertEqual(len(RunStore(path).read_records('timing_blocks')), outer+1)
            return compact(full)
        with tempfile.TemporaryDirectory() as root, patch.object(pilot, '_preparation_replication', side_effect=prepare), patch.object(pilot, 'measure_execution_envelope', side_effect=measured), patch.object(pilot, '_compact_pilot_replication', side_effect=compact_after_publication):
            result = pilot.run_pilot(PilotIntegrationTests.config(), Path(root))
            path = Path(result['run_path'])
            for outer, raw in expected_raw.items():
                raw.update(level=outer//6, block_id=(outer%6)//3, replication=outer%3, outer_replication=outer,
                           difference=raw['risks']['baseline']-raw['risks']['joint'])
                self.assertEqual(json.loads((path/f'preparation-{outer}.json').read_text()), raw)
            rows = RunStore(path).read_records('preparation_risks')
            self.assertEqual(len(rows), 24)
            self.assertTrue(all(row['numerical_error_assessment'] == {'level': 'float64_diagnostic'} for row in rows))
            assets = list((path/'blocks').glob('*/assets/baseline_metric.npy'))
            self.assertEqual(len(assets), 12)
            for asset in assets:
                np.testing.assert_array_equal(np.load(asset), np.arange(16))
            self.assertEqual(len(RunStore(path).read_records('timing_blocks')), 12)
            self.assertEqual(result['source_count_decision']['source_training_tasks'], 6)
            self.assertEqual(result['source_count_decision']['source_validation_tasks'], 4)
            self.assertEqual([curve['relative_covariance_error_mean'] for curve in result['source_curve']], [.9, .19999999999999998])
            self.assertEqual([curve['validation_stability'] for curve in result['source_curve']], ['stable', 'stable'])
            self.assertEqual([curve['validation_mean_variance'] for curve in result['source_curve']], [0., 0.])
            self.assertEqual([block['differences'] for block in result['blocks']], [[.010000000000000009, .020000000000000018, .030000000000000027]]*2)
            self.assertEqual(result['blocks'][0]['source_keys'], [f'{PilotIntegrationTests.config().keys["exploration"]}/12/{outer}/{kind}' for outer in (6, 7, 8) for kind in ('train', 'validation')])
            self.assertEqual(result['precision_diagnostics']['status'], 'extend_pilot')
            self.assertEqual(result['calibration']['status'], 'insufficient')
            draft = json.loads((path/'protocol_draft.json').read_text())['draft']['scientific']
            self.assertEqual(draft['config']['sample_plan']['source_training_tasks'], 6)
            self.assertEqual(draft['config']['sample_plan']['source_validation_tasks'], 4)
            self.assertEqual(draft['config']['keys'], {})
            # Fixed eager-run reference captured before the retention change;
            # only measured resources and checkout/runtime identity are excluded.
            from prepared_learning.records import content_hash
            draft.pop('versions')
            science = ('blocks', 'uncertainty', 'source_curve', 'source_count_decision', 'precision_diagnostics', 'calibration', 'tuning_resolution', 'failures', 'all_pilot_source_keys', 'planning_policy')
            self.assertEqual(content_hash({'result': {key: result[key] for key in science}, 'draft': draft}),
                             'd88cf1f5964525f21c789ffce4ddac519235fdb630a5859324856333c413b20d')
            for key in ('max_preparation_record_bytes', 'max_task_record_bytes', 'serialization_seconds_per_byte', 'checkpoint_seconds', 'analysis_seconds_per_unit'):
                self.assertEqual(result['resources']['execution_envelope'][key], max(e[key] for e in envelopes[1]))
            self.assertEqual([len(envelopes[level]) for level in (0, 1)], [6, 6])

    def test_missing_role_retains_null_pairs_failures_and_full_raw_evidence(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import prepared_learning.pilot as pilot
        from prepared_learning.runner import RunStore
        def prepare(*args):
            full, assets, _ = _pilot_preparation(*args, missing_joint=True)
            return full, assets
        with tempfile.TemporaryDirectory() as root, patch.object(pilot, '_preparation_replication', side_effect=prepare):
            result = pilot.run_pilot(PilotIntegrationTests.config(), Path(root))
            path = Path(result['run_path'])
            rows = RunStore(path).read_records('preparation_risks')
            self.assertEqual(len(rows), 24)
            self.assertTrue(all(row['paired_risk_difference'] is None for row in rows))
            joint = [row for row in rows if row['method_id'] == 'joint']
            self.assertTrue(all(row['status'] == 'failed' and row['conditional_risk'] is None and row['failure_reason'] == 'missing candidate' for row in joint))
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['failures'], ['missing candidate']*12)
            self.assertEqual(result['resources']['update_check_seconds'], [])
            self.assertEqual([block['differences'] for block in result['blocks']], [[], []])
            for raw_path in path.glob('preparation-*.json'):
                raw = json.loads(raw_path.read_text())
                self.assertIsNone(raw['difference'])
                self.assertEqual(set(raw['selected']), {'baseline'})
                self.assertEqual(raw['candidate_records'][1]['failure_reason'], 'retained failure')

    def test_full_payload_envelope_preserves_resource_rejection_and_simulation_costs(self):
        import json
        import tempfile
        from dataclasses import replace
        from pathlib import Path
        from unittest.mock import patch
        import prepared_learning.pilot as pilot
        from prepared_learning.protocol import validate_execution_envelope
        config = PilotIntegrationTests.config()
        config = replace(config, methods=(*config.methods, 'B00'))
        def prepare(*args):
            full, assets, _ = _pilot_preparation(*args)
            return full, assets
        with tempfile.TemporaryDirectory() as root, patch.object(pilot, '_preparation_replication', side_effect=prepare), patch.object(pilot, '_available_memory', return_value=16*1024**2), patch.object(pilot, '_rss_bytes', return_value=1000):
            result = pilot.run_pilot(config, Path(root))
            science = json.loads((Path(result['run_path'])/'protocol_draft.json').read_text())['draft']['scientific']
            self.assertEqual(science['simulation_schedule'], {'outer_replications': [0], 'tasks_per_replication': 2, 'method_ids': ['B00'], 'inference': 'descriptive_only'})
            self.assertEqual(result['resources']['simulation_task_seconds'], [.00001]*6)
            self.assertEqual(result['resources']['update_check_seconds'], [.0000006]*6)
            with self.assertRaisesRegex(ValueError, 'working sets'):
                validate_execution_envelope(science, result['resources'])
            # A compact-record byte substitute would wrongly admit this same run.
            full, _, _ = _pilot_preparation(config, 42, 6, 12, config.methods)
            full.update(level=1, block_id=0, replication=0, outer_replication=6, difference=.01, serialization_ns=1)
            underestimated = {**result['resources'], 'execution_envelope': pilot.measure_execution_envelope([pilot._compact_pilot_replication(full)], [])}
            validate_execution_envelope(science, underestimated)

class PilotPlanningTests(unittest.TestCase):
    def test_precision_uses_headroom_not_success_threshold(self):
        self.assertEqual(INITIAL_HALF_WIDTH, Fraction(1266883, 39703750) / 4)
        pilot = {'analysis_mode': 'conditional_risk_over_preparations', 'blocks': [
            {'differences': [0.01, 0.02, 0.00, 0.03]},
            {'differences': [0.02, 0.00, 0.03, 0.01]}],
            'source_training_tasks': 8, 'source_validation_tasks': 4, 'source_block_size': 4}
        plan = plan_sample_sizes(pilot, float(INITIAL_HALF_WIDTH), .05)
        self.assertEqual(plan['tasks_per_replicate'], None)
        self.assertEqual(plan['alpha'], .05)
        self.assertNotIn('minimum_win', plan)
        self.assertGreaterEqual(plan['outer_replicates'], 2)

    def test_nested_allocation_and_zero_components(self):
        base = {'analysis_mode': 'nested_task_simulation', 'variance_between': 2., 'variance_within': 8., 'preparation_cost': 10., 'task_cost': 5., 'variance_stable': True}
        plan = plan_sample_sizes(base, .1, .05)
        self.assertEqual(plan['tasks_per_replicate'], 3)
        base['variance_between'] = 0.
        self.assertEqual(plan_sample_sizes(base, .1, .05)['decision'], 'extend_pilot')

    def test_unstable_variance_extends_pilot(self):
        pilot = {'analysis_mode': 'conditional_risk_over_preparations', 'blocks': [{'differences':[0., 0., 0.]}, {'differences':[1., 1., 1.]}], 'source_training_tasks': 8, 'source_validation_tasks':4, 'source_block_size':4}
        self.assertEqual(plan_sample_sizes(pilot, .1, .05)['decision'], 'extend_pilot')

class PilotIntegrationTests(unittest.TestCase):
    @staticmethod
    def config():
        import json
        from pathlib import Path
        from prepared_learning.specs import load_config
        from dataclasses import replace
        config=load_config(Path('configs/exploration/learned_prior.json'))
        numerics=dict(config.numerics)
        numerics['fitting_candidates']={'gammas':[0.,1.], 'regularization':[0.], 'ranks':[1], 'scalar_fractions':[0.,.5,1.], 'max_refinement_rounds':0}
        numerics['pilot']={'blocks':2, 'replicates_per_block':3,'source_levels':[3,6], 'validation_levels':[2,4], 'workloads':['W0'], 'calibration_resamples':100, 'source_relative_error_target':.5, 'max_seconds':1000000000}
        return replace(config, numerics=numerics)

    def test_two_pilot_blocks_and_source_levels(self):
        import json
        import tempfile
        from pathlib import Path
        import prepared_learning.pilot as pilot
        from prepared_learning.runner import RunStore
        self.assertTrue(callable(getattr(pilot,'run_pilot',None)), 'run_pilot API is missing')
        with tempfile.TemporaryDirectory() as root:
            result=pilot.run_pilot(self.config(),Path(root))
            self.assertEqual(result['phase'],'exploration')
            self.assertEqual(len(result['blocks']),2)
            a,b=[set(x['source_keys']) for x in result['blocks']]
            self.assertTrue(a and b and a.isdisjoint(b))
            self.assertEqual([r['source_training_tasks'] for r in result['source_curve']],[3,6])
            self.assertNotIn('protected_keys',result)
            self.assertNotIn('protected_effect',result)
            self.assertEqual(result['uncertainty']['status'],'exploratory_only')
            self.assertIsNotNone(result['uncertainty']['interval'])
            self.assertEqual(len(result['calibration']['tails']),2)
            self.assertEqual(result['source_count_decision']['basis'],'preparation_accuracy_and_cost')
            self.assertEqual(result['gate']['code'],'INSENSITIVE_OR_UNSUITABLE')
            self.assertGreater(result['resources']['preparation_seconds'],0.)
            self.assertLessEqual(result['resources']['memory_budget_bytes'],result['resources']['available_memory_bytes']/2)
            path=Path(result['run_path'])
            rows=RunStore(path).read_records('preparation_risks')
            self.assertEqual(len(rows),24)
            self.assertEqual({r['method_id'] for r in rows},{'baseline','joint'})
            self.assertTrue(all(r['status']=='completed' for r in rows))
            for row in rows: self.assertIsInstance(row['conditional_risk'],float)
            import hashlib
            evaluator_hash=hashlib.sha256(Path('src/prepared_learning/evaluation.py').read_bytes()).hexdigest()
            self.assertTrue(all(row['risk_evaluator_version']==evaluator_hash for row in rows))
            timings=RunStore(path).read_records('timing_blocks')
            self.assertTrue(all(r['diagnostic_adaptation_ns']>0 and r['diagnostic_prediction_ns']>0 and r['serialization_ns']>0 for r in timings))
            self.assertEqual(json.loads((path/'run.json').read_text())['status'],result['status'])
            hardware=json.loads((path/'hardware.json').read_text())
            self.assertEqual(hardware['runtime_identity'],result['runtime_identity'])
            self.assertEqual(Path(hardware['runtime_locations']['working_directory']),Path.cwd())
            self.assertIn('python_executable',hardware['runtime_locations'])
            self.assertNotIn('runtime_locations',result['runtime_identity'])
            self.assertTrue((path/'pilot_report.md').is_file())
            self.assertTrue((path/'power_precision.json').is_file())
            self.assertEqual(json.loads((path/'pilot.json').read_text())['gate'],result['gate'])

    def test_analytic_mode_has_no_inner_allocation(self):
        from dataclasses import replace
        import tempfile
        from pathlib import Path
        import prepared_learning.pilot as pilot
        self.assertTrue(callable(getattr(pilot,'run_pilot',None)), 'run_pilot API is missing')
        config=self.config()
        with tempfile.TemporaryDirectory() as root:
            bad=replace(config,sample_plan=replace(config.sample_plan,tasks_per_replicate=2))
            with self.assertRaises(ValueError): pilot.run_pilot(bad,Path(root))

    def test_planning_other_endpoint_counts(self):
        import prepared_learning.pilot as pilot
        deterministic=pilot.plan_sample_sizes({'analysis_mode':'deterministic_frontier'},.01,.05)
        self.assertIsNone(deterministic['outer_replicates'])
        self.assertIsNone(deterministic['tasks_per_replicate'])
        fixed=pilot.plan_sample_sizes({'analysis_mode':'fixed_prior_task_simulation','blocks':[{'differences':[.01,.02,.03]*12},{'differences':[.03,.01,.02]*12}]},.01,.05)
        self.assertIsNone(fixed['outer_replicates'])
        self.assertGreater(fixed['tasks_per_replicate'],1)

    def test_cli_pilot_writes_report_and_freeze_rejects_unresolved(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from prepared_learning.cli import main
        from prepared_learning.records import to_json_record
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); config=root/'config.json'; config.write_text(json.dumps(to_json_record(self.config())))
            output=io.StringIO()
            try:
                with contextlib.redirect_stdout(output): status=main(['pilot','--config',str(config),'--output',str(root/'runs')])
            except SystemExit:
                self.fail('pilot CLI is not registered')
            self.assertEqual(status,0)
            summary=json.loads(output.getvalue()); path=Path(summary['run_path'])
            with contextlib.redirect_stdout(io.StringIO()): status=main(['freeze','--protocol',str(path/'protocol_draft.json'),'--output',str(root/'confirmation.json')])
            self.assertEqual(status,1)
            self.assertFalse((root/'confirmation.json').exists())


    def test_failed_evaluator_retains_paired_rows_and_invalid_gate(self):
        from unittest.mock import patch
        import tempfile
        from pathlib import Path
        from prepared_learning.pilot import run_pilot
        from prepared_learning.runner import RunStore
        with tempfile.TemporaryDirectory() as root:
            with patch('prepared_learning.evaluation.conditional_risk',side_effect=ValueError('injected evaluator defect')):
                result=run_pilot(self.config(),Path(root))
            self.assertEqual(result['gate']['code'],'IMPLEMENTATION_INVALID')
            rows=RunStore(result['run_path']).read_records('preparation_risks')
            self.assertEqual(len(rows),24)
            self.assertTrue(all(r['status']=='failed' and r['conditional_risk'] is None for r in rows))
            self.assertTrue(all(r['failure_reason']=='injected evaluator defect' for r in rows))

    def test_calibration_uses_fixed_final_outer_count(self):
        from prepared_learning.pilot import calibration_diagnostics
        blocks=[{'differences':[-.02,-.01,.01,.02]*10} for _ in range(2)]
        result=calibration_diagnostics(blocks,'student',500,123,planned_count=80)
        self.assertEqual(result['sample_size'],80)
        self.assertEqual(result['status'],'adequate')
        self.assertGreater(result['coverage'],.90)


    def test_representative_scaling_and_replay_are_measured(self):
        from dataclasses import replace
        import tempfile
        from pathlib import Path
        from prepared_learning.pilot import run_pilot
        from prepared_learning.runner import RunStore
        config=self.config(); numerics=dict(config.numerics); settings=dict(numerics['pilot'])
        settings['workloads']=['W0','W1','W2']; numerics['pilot']=settings
        config=replace(config,numerics=numerics)
        with tempfile.TemporaryDirectory() as root:
            first=run_pilot(config,Path(root)); second=run_pilot(config,Path(root))
            self.assertEqual([r['workload'] for r in first['resources']['scaling']],['W1','W2'])
            self.assertTrue(all(r['preparation_seconds']>0 and r['generation_seconds']>0 for r in first['resources']['scaling']))
            def scientific_rows(result):
                return sorted((r['outer_replication'],r['method_id'],r['source_key'],r['metric_hash'],r['conditional_risk']) for r in RunStore(result['run_path']).read_records('preparation_risks'))
            self.assertEqual(scientific_rows(first),scientific_rows(second))
            self.assertEqual(first['all_pilot_source_keys'],second['all_pilot_source_keys'])

    def test_calibration_honors_memory_reserve(self):
        from prepared_learning.pilot import calibration_diagnostics
        blocks=[{'differences':[-.02,-.01,.01,.02]*10} for _ in range(2)]
        result=calibration_diagnostics(blocks,'student',500,123,planned_count=10**12,memory_budget_bytes=1000)
        self.assertEqual(result['status'],'resource_limit')
        self.assertIsNone(result['coverage'])

    def test_source_choice_uses_validation_stability_and_cost(self):
        import prepared_learning.pilot as pilot
        self.assertTrue(callable(getattr(pilot,'choose_source_level',None)))
        curves=[{'level':0,'replications':60,'relative_covariance_error_upper_diagnostic':.9,'validation_stability':'stable','preparation_seconds':1.},
                {'level':1,'replications':60,'relative_covariance_error_upper_diagnostic':.3,'validation_stability':'unstable','preparation_seconds':2.},
                {'level':2,'replications':60,'relative_covariance_error_upper_diagnostic':.2,'validation_stability':'stable','preparation_seconds':3.}]
        chosen,status=pilot.choose_source_level(curves,.5,60)
        self.assertEqual(chosen['level'],2)
        self.assertEqual(status,'resolved')
        curves[2]['relative_covariance_error_upper_diagnostic']=.8
        _,status=pilot.choose_source_level(curves,.5,60)
        self.assertEqual(status,'extend_pilot')


    def test_resource_stop_publishes_incomplete_terminal_run(self):
        import json
        import tempfile
        from dataclasses import replace
        from pathlib import Path
        from prepared_learning.pilot import run_pilot
        config=self.config(); numerics=dict(config.numerics)
        numerics['pilot']={**dict(numerics['pilot']),'max_seconds':1e-12}
        with tempfile.TemporaryDirectory() as root:
            result=run_pilot(replace(config,numerics=numerics),Path(root))
            self.assertEqual(result['status'],'incomplete')
            self.assertEqual(result['gate']['code'],'LOCAL_RESOURCE_LIMIT')
            self.assertEqual(json.loads((Path(result['run_path'])/'run.json').read_text())['status'],'incomplete')
