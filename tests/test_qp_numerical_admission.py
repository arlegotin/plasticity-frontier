"""Mathematical regression inputs; these fixtures are never scientific observations."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from prepared_learning.frontier import solve_frontier
from prepared_learning.metrics import apply_metric, metric_resources
from prepared_learning.quadratic import build_oracle_qp, solve_qp
from prepared_learning.records import BudgetSpec, PreparationSpec, PriorAsset, to_json_record


CASES = json.loads((Path(__file__).parent / 'fixtures/wide_qp_regressions.json').read_text())['cases']


def case_args(name):
    c = CASES[name]
    s = np.array(c['spectrum'])
    prior = PriorAsset(np.diag(s), np.eye(len(s)), s, c['noise_variance'],
                       'oracle', {}, 'mathematical-regression')
    return prior, BudgetSpec(**c['budget']), PreparationSpec(**c['preparation'])


def case_qp(name, branch):
    c = CASES[name]
    return build_oracle_qp(c['spectrum'], c['preparation']['target_context_size'],
                           c['noise_variance'], c['k'], c['width'], branch)


class NumericalAdmissionTests(unittest.TestCase):
    def test_w1_rejected_polish_retains_quality_checked_native_point(self):
        result = solve_frontier(*case_args('W1'))
        self.assertEqual(result.F.status, 'completed')
        diag = result.F.diagnostics
        self.assertTrue(diag.kkt['native_solver_success'])
        self.assertEqual(diag.kkt['native_solver_status'], 0)
        self.assertFalse(diag.kkt['active_set_polish_valid'])
        self.assertIn('negative_multiplier', diag.kkt['active_set_polish_rejection_reasons'])
        self.assertEqual(diag.kkt['candidate_origin'], 'native')
        self.assertTrue(diag.kkt['final_quality_passed'])
        self.assertGreaterEqual(diag.dual['minimum_multiplier'], 0.)
        # This changes admission, not the retained native metric. Mathematical
        # optimum is (.6, 0, ..., 0), with a degenerate zero floor multiplier.
        np.testing.assert_array_equal(result.F.variables, CASES['W1']['retained_variables']['F'])
        np.testing.assert_allclose(result.F.variables, [.6] + [0.]*7, atol=1e-15, rtol=0)
        self.assertEqual(diag.kkt['active_face_reconstruction_indices'], ())
        self.assertEqual(result.Z.status, 'completed')

    def test_w2_positive_dual_face_has_exact_zeros_and_original_rank_guard(self):
        result = solve_frontier(*case_args('W2'))
        self.assertEqual(result.Z.status, 'completed')
        x, diag = result.Z.variables, result.Z.diagnostics
        np.testing.assert_allclose(x[:6], .1, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(x[6:], [0., 0.])
        self.assertLessEqual(diag.kkt['rank_oracle_max_relative_error'], 1e-6)
        self.assertEqual(diag.kkt['candidate_origin'], 'active_set_polish')
        self.assertEqual(diag.kkt['active_face_reconstruction_indices'], (6, 7))
        self.assertEqual(diag.kkt['active_face_reconstruction_size'], 2)
        self.assertGreater(diag.kkt['active_face_reconstruction_inf'], 0)
        before = np.array(diag.kkt['active_set_polish_candidate_scaled']) * .6
        self.assertEqual(diag.kkt['repair_inf'], np.max(np.abs(x-before)))
        self.assertGreaterEqual(diag.kkt['repair_inf'], diag.kkt['active_face_reconstruction_inf'])
        qp = case_qp('W2', 'Z')
        mu = diag.dual['multipliers']
        slack = qp.A@x + qp.offset
        np.testing.assert_array_equal(diag.feasibility['slacks'], slack)
        self.assertEqual(diag.kkt['stationarity_inf'],
                         np.max(np.abs(qp.linear + 2*qp.hessian@x - qp.A.T@mu)))
        self.assertEqual(diag.kkt['complementarity_inf'], np.max(np.abs(mu*slack)))
        self.assertEqual(result.Z.metric.support_metadata.exact_zero_multiplicity, 58)
        self.assertEqual(metric_resources(result.Z.metric, 2).trainable_parameters, 390)
        vector = np.arange(64, dtype=float)
        np.testing.assert_allclose(apply_metric(result.Z.metric, vector),
                                   np.r_[.1*vector[:6], np.zeros(58)], atol=1e-15, rtol=0)
        np.testing.assert_allclose(x, CASES['W2']['retained_variables']['Z'], atol=1e-15, rtol=0)

    def test_failed_native_is_not_rescued_by_valid_polish(self):
        # A failure indication must remain failed even if the equality solve
        # independently recovers the optimum from a good native iterate.
        native = OptimizeResult(success=False, status=9, message='Iteration limit reached',
                                x=np.r_[np.full(6, 1/6), 0., 0.])
        with patch('prepared_learning.quadratic.minimize', return_value=native):
            x, diag = solve_qp(case_qp('W2', 'Z'), .6)
        self.assertFalse(diag.kkt['solver_success'])
        self.assertFalse(diag.kkt['native_solver_success'])
        self.assertEqual(diag.kkt['native_solver_status'], 9)
        self.assertTrue(diag.kkt['active_set_polish_valid'])
        self.assertTrue(diag.kkt['final_quality_passed'])
        np.testing.assert_allclose(x[:6], .1, atol=1e-15, rtol=0)

    def test_successful_native_with_rejected_polish_and_bad_kkt_fails(self):
        native = OptimizeResult(success=True, status=0, message='synthetic success', x=np.zeros(8))
        with patch('prepared_learning.quadratic.minimize', return_value=native), \
             patch('prepared_learning.quadratic.np.linalg.solve', side_effect=np.linalg.LinAlgError('singular')):
            x, diag = solve_qp(case_qp('W1', 'F'), .6)
        self.assertFalse(diag.kkt['solver_success'])
        self.assertFalse(diag.kkt['final_quality_passed'])
        self.assertGreaterEqual(diag.kkt['stationarity_inf'], 1e-9)
        self.assertEqual(diag.kkt['candidate_origin'], 'native')
        self.assertEqual(diag.kkt['active_set_polish_rejection_reasons'], ('linear_solve_failure',))
        np.testing.assert_array_equal(x, np.zeros(8))

    def test_nonfinite_polish_rejection_preserves_serializable_native_diagnostics(self):
        # A corrupt optional linear solve must neither replace a good native
        # candidate nor poison the retained failure diagnostic's JSON encoding.
        for bad_index in (0, 8):
            with self.subTest(bad_index=bad_index):
                solution = np.zeros(16)
                solution[bad_index] = np.nan
                with patch('prepared_learning.quadratic.np.linalg.solve', return_value=solution):
                    x, diag = solve_qp(case_qp('W1', 'F'), .6)
                self.assertTrue(diag.kkt['solver_success'])
                self.assertFalse(diag.kkt['active_set_polish_valid'])
                self.assertIn('nonfinite_solution', diag.kkt['active_set_polish_rejection_reasons'])
                self.assertEqual(diag.kkt['candidate_origin'], 'native')
                np.testing.assert_array_equal(x, CASES['W1']['retained_variables']['F'])
                json.dumps(to_json_record(diag), allow_nan=False)

    def test_face_reconstruction_requires_strictly_positive_lower_multiplier(self):
        # Exercise the zero-multiplier boundary of a production-selected face.
        # A tiny positive candidate alone never authorizes erasing that gain.
        solution = np.r_[np.full(6, 1/6), 1e-18, 1e-18, 0., 1e-30, .4]
        with patch('prepared_learning.quadratic.np.linalg.solve', return_value=solution):
            x, diag = solve_qp(case_qp('W2', 'Z'), .6)
        self.assertTrue(diag.kkt['active_set_polish_valid'])
        self.assertEqual(diag.kkt['active_face_reconstruction_indices'], (7,))
        self.assertGreater(x[6], 0.)
        self.assertEqual(x[7], 0.)

    def test_erased_weak_positive_gain_still_fails_relative_rank_check(self):
        s = np.array([1., 1e-20])
        prior = PriorAsset(np.diag(s), np.eye(2), s, 0., 'oracle', {}, 'weak-positive')
        args = prior, BudgetSpec(2, 10., 6, 1.), PreparationSpec(10**16, 2, 'gaussian', '1/(2n)', 1.)
        real_solve = solve_qp
        def erase(qp, scale):
            x, diag = real_solve(qp, scale)
            # The small erased gain still passes absolute KKT tolerance; only
            # the independent relative guard protects this pointwise direction.
            x[-1] = 0.
            return x, diag
        with patch('prepared_learning.frontier.solve_qp', side_effect=erase):
            result = solve_frontier(*args)
        self.assertEqual(result.Z.status, 'failed')
        self.assertEqual(result.Z.diagnostics.kkt['rank_oracle_max_relative_error'], 1.)

    def test_unconstrained_weak_positive_and_zero_budget_keep_support(self):
        s = np.array([1., 1e-20])
        qp = build_oracle_qp(s, 10**16, 0., 10., 2, 'Z')
        x, diag = solve_qp(qp, 1.)
        self.assertGreater(x[1], 0.)
        self.assertAlmostEqual(x[1]/(1e-20/(1e-20+1e-16)), 1.)
        self.assertEqual(diag.kkt['candidate_origin'], 'unconstrained')
        self.assertEqual(diag.kkt['active_face_reconstruction_indices'], ())
        qp = build_oracle_qp([.5, .5], 4, 1., 0., 2, 'Z')
        x, diag = solve_qp(qp, 0.)
        np.testing.assert_array_equal(x, [0., 0.])
        self.assertTrue(diag.kkt['solver_success'])
        self.assertIsNone(diag.kkt['native_solver_success'])
        self.assertIsNone(diag.kkt['native_solver_status'])
        self.assertEqual(diag.kkt['candidate_origin'], 'zero_budget')
        self.assertEqual(diag.kkt['repair_inf'], 0.)


if __name__ == '__main__':
    unittest.main()
