"""Behavioral regressions for exploratory attribution and assumption boundaries."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from prepared_learning.specs import load_config, validate_config
from prepared_learning.frontier import solve_frontier
from prepared_learning.geometry import construct_canonical, effective_coefficient, tangent_matrix
from prepared_learning.metrics import dense_metric
from prepared_learning.records import to_json_record


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('prepared_learning.robustness'),
                             'missing executable robustness arm/amendment behavior')
        from prepared_learning import robustness
        self.r = robustness
        self.base = load_config(Path('configs/exploration/learned_prior.json'))
        self.base = replace(self.base, workload={**self.base.workload, 'population_seed': 73})

    def arm(self, name):
        return self.r.diagnose_arm(self.r.build_robustness_arms(self.base, [name])[0])

    def test_no_floor_is_optimized_z(self):
        row = self.arm('no_floor')
        self.assertAlmostEqual(row['Z']['risk'], 81247/115000, delta=1e-10)
        self.assertGreater(row['zeroed_F_norm'], .6)
        self.assertGreater(abs(row['zeroed_F_risk']-row['Z']['risk']), .001)
        self.assertEqual(row['Z']['solver_status'], 'completed')

    def test_equal_actual_and_capped_counts(self):
        rows = self.arm('actual_counts')['fixtures']
        self.assertEqual(rows[0]['actual_counts'], [21, 21])
        self.assertEqual(rows[0]['permitted_counts'], [21, 21])
        self.assertEqual(rows[1]['actual_counts'], [21, 14])
        self.assertEqual(rows[1]['padded_counts'], [21, 21])
        self.assertLess(rows[1]['padding_map_error'], 1e-12)

    def test_wrong_basis_and_shift_are_declared(self):
        row = self.arm('wrong_basis')
        np.testing.assert_allclose(row['original_spectrum'], row['ablated_spectrum'], atol=1e-12)
        self.assertEqual(row['original_resources'], row['ablated_resources'])
        self.assertGreater(row['risk_delta'], 0)
        for name in ('prior_rotation', 'leakage', 'mean_shift'):
            row = self.arm(name)
            self.assertNotEqual(row['source_population']['hash'], row['test_population']['hash'])
            self.assertEqual(row['confirmation']['status'], 'unperformed')
            self.assertGreater(len(row['signed_differences']), 0)
        self.assertEqual(self.arm('leakage')['test_population']['leakage_mass'], .2)
        self.assertGreater(np.linalg.norm(self.arm('mean_shift')['test_population']['mean']), 0)

    def test_free_eta_control(self):
        row = self.arm('free_eta')
        self.assertEqual(row['scale'], 2.)
        self.assertAlmostEqual(row['initial_norm_ratio'], 4.)
        self.assertAlmostEqual(row['tangent_ratio'], 2**(2*(row['depth']-1)))
        self.assertLess(row['first_coefficient_error'], 1e-12)
        self.assertAlmostEqual(row['post_norm'], row['norm_identity'], delta=1e-11)
        self.assertAlmostEqual(row['post_increment_ratio'], 1/row['tangent_ratio'], delta=1e-12)

    def test_second_step_is_literal(self):
        row = self.arm('second_step')
        layers = tuple(np.array(x) for x in row['first_layers'])
        X, y = np.array(row['X']), np.array(row['y'])
        gradients = tuple(np.array(x) for x in row['second_gradients'])
        # Independent central differences of the actual loss at the first state.
        def loss(ls):
            residual = X @ effective_coefficient(ls) - y
            return residual @ residual / (2*len(y))
        for i, layer in enumerate(layers):
            for index in np.ndindex(layer.shape):
                plus, minus = [a.copy() for a in layers], [a.copy() for a in layers]
                plus[i][index] += 1e-6; minus[i][index] -= 1e-6
                self.assertAlmostEqual(gradients[i][index], (loss(plus)-loss(minus))/2e-6, delta=1e-7)
        self.assertGreater(abs(gradients[-1][0, 0]), 1e-6)
        self.assertGreater(row['fixed_metric_disagreement'], 1e-6)

    def test_approximate_width_reoptimizes(self):
        row = self.arm('approximate_width')
        for point in row['widths']:
            self.assertGreater(point['dropped_mass'], 0)
            self.assertGreater(point['kernel_error'], 0)
            self.assertLessEqual(point['reoptimized_risk'], point['truncated_risk'] + 1e-10)
            self.assertAlmostEqual(point['reoptimized_risk'], point['evaluated_risk'], delta=1e-10)
        prior, budget, task = self.r.oracle_inputs(self.base)
        with self.assertRaises(ValueError): self.r.reoptimize_approximate_width(prior, budget, task, 4)

    def test_precision_and_input_law_labels(self):
        row = self.arm('float32')
        for field in ('layer_dtypes', 'gradient_dtypes', 'update_dtypes'):
            self.assertEqual(set(row[field]), {'float32'})
        self.assertEqual(row['evaluation_dtype'], 'float64')
        self.assertNotEqual(row['interpretation'], 'theorem_counterexample')
        for name, law in [('anisotropic', 'gaussian_anisotropic'), ('student10', 'student_t_10'), ('student3', 'student_t_3')]:
            config = self.r.build_robustness_arms(self.base, [name])[0]
            self.assertEqual(config.preparation.input_law, law)
            with self.assertRaises(ValueError): validate_config(config, False)
            row = self.r.diagnose_arm(config)
            self.assertEqual(row['risk_evaluator'], 'sampled_coefficient_and_prediction_loss')
            self.assertIsNone(row['mean_risk_ci'])
            self.assertGreater(len(row['signed_differences']), 0)
        self.assertEqual(self.arm('student3')['moment_status'], 'infinite_fourth_moment')
        self.assertEqual(self.arm('student10')['student_df'], 10)

    def test_amendment_requires_fresh_keys(self):
        old = {'root_seed': 77, 'training_phase':101, 'validation_phase':102, 'test_phase':103}
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.r.validate_amendment({'claim_id':'changed','status':'retained','protected_keys':old}, [old])
        with self.assertRaises(ValueError):
            self.r.validate_amendment({'claim_id':'changed','status':'retained','protected_keys':{**old,'root_seed':78}}, [old])
        self.r.validate_amendment({'claim_id':'changed','status':'unperformed','reason':'awaiting scenario pilot and freeze'}, [old])

    def test_all_selected_families_execute_and_preserve_negative_diagnostics(self):
        configs = self.r.build_robustness_arms(self.base, self.r.ARM_NAMES)
        rows = {c.workload['robustness']['arm']:self.r.diagnose_arm(c) for c in configs}
        for row in rows.values():
            self.assertTrue(row['executed']); self.assertTrue(row['uncertainty']); self.assertTrue(row['matching'])
        self.assertAlmostEqual(rows['isotropy']['risk'], 13/17, delta=1e-12)
        self.assertGreater(rows['noncanonical']['rational_disagreement'], .01)
        self.assertLess(rows['orthogonal_coordinates']['tangent_error'], 1e-12)
        self.assertGreater(rows['nonorthogonal_coordinates']['norm_change'], .01)
        self.assertLess(rows['degeneracy']['projector_error'], 1e-12)
        self.assertEqual(rows['depth_one']['trainable_parameters'], 6)
        self.assertEqual(rows['fixed_feature']['trainable_parameters'], 3)
        self.assertGreater(rows['fixed_feature']['stored_scalars'], 3)
        self.assertEqual(len(rows['prior_uncertainty']['replications']), 3)
        self.assertEqual(set(rows['information']['information_access']), {'oracle','estimated','none'})
        self.assertLess(rows['shared_optimizations']['ridge_primal_dual_error'], 1e-12)
        self.assertTrue(rows['resume_error']['config_change_rejected'])
        self.assertEqual(rows['resume_error']['failed_rows_retained'], 1)
        self.assertTrue(rows['extreme_scale']['outcomes'])

    def test_deep_scaling_declarations_and_near_degeneracy(self):
        deep = replace(self.base, budget=replace(self.base.budget, depth=3,
                       initialization_norm_cap=2*np.sqrt(.6), trainable_parameter_cap=22))
        config = self.r.build_robustness_arms(deep, ['free_eta'])[0]
        row = self.r.diagnose_arm(config)
        self.assertAlmostEqual(config.budget.learning_rate, row['eta'])
        self.assertAlmostEqual(row['tangent_ratio'], 16.)
        self.assertLess(row['first_coefficient_error'], 1e-12)
        row = self.arm('degeneracy')
        self.assertIn('near_minimum_execution', row)
        near = row['near_minimum_execution']
        self.assertGreater(near['float64_gap'], 0.)
        self.assertEqual(near['float32_gap'], 0.)
        self.assertEqual(near['status'], 'precision_support_loss')

    def test_arm_changes_and_invalid_manifests_cannot_be_hidden(self):
        config = self.r.build_robustness_arms(self.base, ['student3'])[0]
        with self.assertRaises(ValueError):
            self.r.diagnose_arm(replace(config, preparation=self.base.preparation))
        for arm in self.r.ARM_NAMES:
            c = self.r.build_robustness_arms(self.base, [arm])[0]
            self.assertIn('changes', c.workload['robustness'])
            self.assertTrue(c.workload['robustness']['changes'])
            self.assertTrue(c.workload['robustness']['question'])
            row = self.r.diagnose_arm(c)
            self.assertEqual(row['matched_reference']['permitted_parameter_cap'], 21)
            self.assertEqual(row['matched_reference']['F_actual_resources']['trainable_parameters'], 21)
        manifest = self.r.make_manifest(self.base, ['no_floor'])
        manifest['arm_declarations'][0]['matching'] = {}
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            self.r.run_robustness(manifest, Path(root))

    def test_shared_optimizations_apply_to_both_families_at_depth_three(self):
        deep = replace(self.base, budget=replace(self.base.budget, depth=3,
                       initialization_norm_cap=2*np.sqrt(.6), trainable_parameter_cap=22))
        config = self.r.build_robustness_arms(deep, ['shared_optimizations'])[0]
        self.assertEqual(config.access['prior'], 'oracle')
        row = self.r.diagnose_arm(config)
        self.assertIn('families', row)
        self.assertEqual(set(row['families']), {'F','Z'})
        for family in row['families'].values():
            self.assertLess(family['compact_dense_error'], 1e-12)
            self.assertLess(family['collapsed_prediction_error'], 1e-12)
        self.assertEqual(config.methods, ('robustness:shared_optimizations',))

    def test_manifest_cli_and_invalid_arms(self):
        from prepared_learning.cli import main
        with self.assertRaises(ValueError): self.r.build_robustness_arms(self.base, ['unknown'])
        with self.assertRaises(ValueError): self.r.build_robustness_arms(self.base, ['no_floor','no_floor'])
        manifest = self.r.make_manifest(self.base, ['no_floor', 'student3'])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'manifest.json'; path.write_text(json.dumps(to_json_record(manifest)))
            self.assertEqual(main(['robustness','--manifest',str(path),'--output',str(Path(root)/'out')]), 0)
            report = json.loads((Path(root)/'out'/'robustness.json').read_text())
            self.assertEqual(len(report['arms']), 2)
            self.assertEqual(report['original_primary_result'], 'unchanged')
            self.assertEqual([x['status'] for x in report['transfer_confirmations']], ['unperformed','unperformed'])
            self.assertIsNone(report['arms'][1]['mean_risk_ci'])
            manifest['phase'] = 'confirmation'
            with self.assertRaises(ValueError): self.r.run_robustness(manifest, Path(root)/'bad')
