import unittest
import numpy as np
from prepared_learning.analysis import paired_interval, nested_replication_means, holm_adjust, classify_gate

class AnalysisTests(unittest.TestCase):
    def test_student_outer_units_and_zero(self):
        x = paired_interval(np.array([1., 2., 3.]), .05, 'student', 0)
        self.assertEqual(x['n'], 3); self.assertEqual(x['mean'], 2.)
        self.assertAlmostEqual(x['standard_error'], 1 / np.sqrt(3))
        z = paired_interval(np.zeros(3), .05, 'student', 0)
        self.assertEqual((z['mean'], z['standard_error'], z['interval']), (0., 0., [0., 0.]))
    def test_descriptive_moments(self):
        x = paired_interval(np.array([1., 2., 3.]), .05, 'descriptive_only', 0)
        self.assertIsNone(x['interval']); self.assertEqual(x['status'], 'unsupported_or_unverified_moments')
    def test_nested_equal_outer(self):
        rows = [{'outer_replication': 1, 'difference': 0.}, {'outer_replication': 2, 'difference': 2.}, {'outer_replication': 2, 'difference': 4.}]
        np.testing.assert_allclose(nested_replication_means(rows), [0., 3.])
    def test_holm_and_gate(self):
        np.testing.assert_allclose(holm_adjust(np.array([.01, .04, .03])), [.03, .06, .06])
        self.assertEqual(classify_gate({'interval': [1., 2.]}).code, 'SUPPORTED_IN_SCOPE')
        self.assertEqual(classify_gate({'interval': [-2., -1.]}).code, 'NO_COMPARATIVE_ADVANTAGE')
        self.assertEqual(classify_gate({'interval': [-1., 1.]}).code, 'INSENSITIVE_OR_UNSUITABLE')

    def test_nested_components_and_declared_family(self):
        from prepared_learning.analysis import nested_variance_components, holm_family
        rows = [{'outer_replication': j, 'difference': value} for j, values in enumerate([[0., 2.], [2., 4.], [4., 6.]]) for value in values]
        result = nested_variance_components(rows)
        self.assertEqual(result['outer_replicates'], 3)
        self.assertAlmostEqual(result['within_variance'], 2.)
        self.assertAlmostEqual(result['between_variance'], 3.)
        adjusted = holm_family({'a':.01, 'b':.04, 'c':.03}, ['a', 'b', 'c'])
        self.assertAlmostEqual(adjusted['a'], .03)
        with self.assertRaises(ValueError): holm_family({'a':.01}, ['a', 'b'])

    def test_failed_pair_is_not_silently_reduced(self):
        from prepared_learning.analysis import paired_differences
        rows = [{'outer_replication':0, 'method_id':'a', 'status':'completed', 'conditional_risk':2.}, {'outer_replication':0, 'method_id':'b', 'status':'failed', 'conditional_risk':None}]
        with self.assertRaises(ValueError): paired_differences(rows, ['outer_replication'], 'a', 'b', 'conditional_risk')
        rows[1].update(status='completed', conditional_risk=1.)
        np.testing.assert_allclose(paired_differences(rows, ['outer_replication'], 'a', 'b', 'conditional_risk'), [1.])

    def test_unit_identity_is_nonnegative_integer(self):
        from prepared_learning.analysis import paired_differences, nested_variance_components
        for bad in [0.0, True, '0', -1]:
            rows = [{'outer_replication':bad, 'method_id':role, 'status':'completed', 'conditional_risk':1., 'difference':1.} for role in ('a', 'b')]
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError): paired_differences(rows, ['outer_replication'], 'a', 'b', 'conditional_risk')
                with self.assertRaises(ValueError): nested_replication_means(rows)
                with self.assertRaises(ValueError): nested_variance_components(rows)
