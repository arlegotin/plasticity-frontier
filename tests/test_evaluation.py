import unittest
import numpy as np
from prepared_learning.records import PopulationTruth, SupportMetadata
from prepared_learning.evaluation import EvaluationTruth, conditional_risk, coefficient_loss
from prepared_learning.metrics import CompactMetric, dense_metric

class EvaluationTests(unittest.TestCase):
    def test_rotated_conditional_risk(self):
        rng = np.random.default_rng(44)
        U, _ = np.linalg.qr(rng.normal(size=(4,4)))
        V, _ = np.linalg.qr(rng.normal(size=(4,4)))
        S = (V * [0.6, 0.3, 0.1, 0.]) @ V.T
        p = PopulationTruth(S, 0.7, 'test', target_context_size=3)
        meta = SupportMetadata('analytic', 0, 1, (0,1))
        m = CompactMetric(.2, U[:,:2], np.array([.3,.8]), meta)
        G = dense_metric(m); c = 4/3; rho = 1.7/3
        expected = 1 - 2*np.trace(G@S) + c*np.trace(G@G@S) + rho*np.trace(G@G)
        self.assertAlmostEqual(conditional_risk(m,p), expected, delta=abs(expected)*1e-10)
        zero_metric = CompactMetric(0., np.empty((4,0)), np.empty(0), meta)
        identity_metric = CompactMetric(1., np.empty((4,0)), np.empty(0), meta)
        self.assertEqual(conditional_risk(zero_metric, p), 1.0)
        d = 4
        self.assertAlmostEqual(conditional_risk(identity_metric,p), 1.0 - 2.0 + c + rho*d, places=12)

    def test_coefficient_loss(self):
        theta_is_1_0 = EvaluationTruth(np.array([1.,0.]), np.diag([1.,0.]), 0., 'key')
        self.assertEqual(coefficient_loss(np.array([1., 0.]), theta_is_1_0), 0.0)

    def test_conditional_risk_needs_no_test_tasks(self):
        p = PopulationTruth(np.eye(2)/2, 0., 'test', target_context_size=2)
        m = CompactMetric(0., np.empty((2,0)), np.empty(0), SupportMetadata('analytic',2,2,()))
        self.assertEqual(conditional_risk(m,p), 1.)

    def test_unsupported_input_law_rejected(self):
        p = PopulationTruth(np.eye(2)/2, 0., 'test', target_context_size=2, input_law='student')
        m = CompactMetric(0., np.empty((2,0)), np.empty(0), SupportMetadata('analytic',2,2,()))
        with self.assertRaises(ValueError): conditional_risk(m, p)
