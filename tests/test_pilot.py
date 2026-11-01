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
