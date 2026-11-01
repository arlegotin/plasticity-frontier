import unittest
from fractions import Fraction

from prepared_learning.pilot import plan_sample_sizes, INITIAL_HALF_WIDTH

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
