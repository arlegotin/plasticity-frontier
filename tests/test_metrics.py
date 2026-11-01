import unittest
import numpy as np
from prepared_learning.metrics import CompactMetric, apply_metric, dense_metric, metric_resources, metric_from_dense
from prepared_learning.records import SupportMetadata


def metric(floor, values, d=None):
    values = np.asarray(values, dtype=float)
    d = len(values) if d is None else d
    return CompactMetric(floor, np.eye(d)[:, :len(values)], values,
        SupportMetadata('analytic', int(floor == 0) * (d - np.count_nonzero(values)), d - np.count_nonzero(values), tuple(np.flatnonzero(values))))

class MetricTests(unittest.TestCase):
    def test_compact_application_and_canonical_resources(self):
        m = metric(0, [1, 2])
        np.testing.assert_allclose(apply_metric(m, np.array([2., 3.])), [2, 6])
        np.testing.assert_allclose(dense_metric(m), np.diag([1, 2]))
        self.assertEqual(metric_resources(m, 2).trainable_parameters, 6)
        self.assertAlmostEqual(metric_resources(m, 2).initialization_norm, 2)

    def test_invalid_dense_inputs_and_repairs(self):
        for a in (np.array([[np.nan]]), np.array([[1., 1.], [0., 1.]]), np.diag([1., -0.1])):
            with self.assertRaises(ValueError): metric_from_dense(a)
        m = metric_from_dense(np.diag([1., -1e-16]))
        self.assertTrue(m.support_metadata.projections)
        self.assertEqual(m.support_metadata.policy, 'approximate')
