import unittest
import numpy as np
from numpy.linalg import norm
from prepared_learning.geometry import (construct_canonical, construct_head_zero, effective_coefficient, tangent_matrix, simultaneous_step, diagnose_geometry)
from prepared_learning.metrics import CompactMetric, dense_metric, metric_from_dense
from prepared_learning.records import SupportMetadata
from checks.independent_reference import complex_step_jacobian
from tests.test_metrics import metric

class GeometryTests(unittest.TestCase):
    def test_tiny_positive_gap(self):
        m = metric(1e-14, [0, 1e-14])
        actual = tangent_matrix(construct_canonical(m, 4).layers)
        self.assertLess(norm(actual-dense_metric(m))/norm(dense_metric(m)), 1e-10)
        self.assertTrue(np.all(np.linalg.eigvalsh(actual) > 0))

    def test_canonical_step_is_direct(self):
        rng = np.random.default_rng(4)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        for floor, increments in [(0, [0,0,0,0]), (2, [0,0,0,0]), (0, [0,0,1,2]), (1, [0,0,1,2]), (1, [0,1e-14,1,2])]:
            m = CompactMetric(floor, q, np.array(increments, float), SupportMetadata('analytic', 0, increments.count(0), tuple(i for i,x in enumerate(increments) if x)))
            target = dense_metric(m)
            for depth in range(2, 9):
                r = construct_canonical(m, depth)
                self.assertEqual(r.resources.trainable_parameters, (4 + depth - 1) if not np.any(target) else 5*(sum(x > 0 for x in increments)+(floor > 0))+depth-2)
                self.assertLess(norm(effective_coefficient(r.layers)), 1e-12)
                J = complex_step_jacobian(r.layers)
                np.testing.assert_allclose(J @ J.T, target, atol=1e-11, rtol=1e-10)
                z = rng.normal(size=4)
                for eta in (0.01, 1., 2.):
                    np.testing.assert_allclose(effective_coefficient(simultaneous_step(r.layers, -z, eta)), eta*target@z, atol=1e-11, rtol=1e-10)
                gamma = floor + sum(increments)
                self.assertAlmostEqual(r.resources.initialization_norm, 0 if gamma == 0 else (depth-1)*gamma**(1/(depth-1)), places=10)

    def test_simultaneous_is_not_sequential(self):
        r = construct_canonical(metric(1, [0,2]), 2)
        z = np.array([1., 2.]); eta = .5
        self.assertTrue(np.allclose(effective_coefficient(simultaneous_step(r.layers, -z, eta)), eta*dense_metric(metric(1,[0,2]))@z))
        W, a = r.layers
        updated_W = W + eta*a.T@z[None,:]
        sequential_a = a + eta*(updated_W@z)[None,:]
        self.assertFalse(np.allclose(effective_coefficient((updated_W, sequential_a)), eta*dense_metric(metric(1,[0,2]))@z))

    def test_initializer_is_immutable(self):
        r = construct_canonical(metric(1,[0,2]), 4)
        self.assertTrue(all(not a.flags.writeable for a in r.layers))
        a = simultaneous_step(r.layers, np.ones(2), 1)
        b = simultaneous_step(r.layers, np.ones(2), 1)
        self.assertFalse(np.shares_memory(a[0], b[0]))
        a[0][0,0] += 1
        self.assertFalse(np.array_equal(a[0], b[0]))

    def test_noncanonical_is_nonlinear(self):
        layers = (np.array([[1.],[0.]]), np.array([[1.,1.],[0.,1.]]), np.array([[0.,1.]]))
        self.assertAlmostEqual(tangent_matrix(layers)[0,0], 3)
        self.assertAlmostEqual(effective_coefficient(simultaneous_step(layers, np.array([-1.]), 1))[0], 4)

    def test_post_norm_identity(self):
        cases = [construct_canonical(metric(1,[0,2]), 4).layers, (np.array([[1.],[0.]]), np.array([[1.,1.],[0.,1.]]), np.array([[0.,1.]]))]
        for layers in cases:
            g = np.ones(layers[0].shape[1]); eta = .7
            updated = simultaneous_step(layers, g, eta)
            self.assertAlmostEqual(sum(np.sum(a*a) for a in updated), sum(np.sum(a*a) for a in layers) + eta**2 * (g@tangent_matrix(layers)@g))

    def test_head_zero_full_rank(self):
        r = construct_head_zero(metric(0,[1,2]))
        self.assertAlmostEqual(r.resources.initialization_norm, 3)
        self.assertEqual(r.resources.trainable_parameters, 6)
        np.testing.assert_allclose(tangent_matrix(r.layers), np.diag([1,2]))

    def test_zero_reference_status_and_representability(self):
        z = metric(0,[0,0]); r = construct_canonical(z,2)
        self.assertEqual(diagnose_geometry(z,r).residuals['relative_frobenius_status'], 'exact_zero')
        with self.assertRaisesRegex(ValueError, 'represent'):
            construct_canonical(metric(0,[np.nextafter(0.,1.), 1e308]), 8)

class GeometryCliTests(unittest.TestCase):
    def test_adversarial_cli_reports_derivative_and_counterexample(self):
        import json, subprocess, sys
        result = subprocess.run([sys.executable, '-m', 'prepared_learning', 'check-geometry', '--suite', 'adversarial'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(report['counterexample']['coefficient'], 4.0)
        self.assertGreater(report['derivative_count'], 0)
        self.assertLess(report['max_relative_error'], 1e-10)

class GeometryScaleTests(unittest.TestCase):
    def test_extreme_finite_scales_have_defined_serializable_relative_errors(self):
        import json
        from prepared_learning.records import to_json_record
        for scale in (1e-200, 1e200):
            with self.subTest(scale=scale):
                m = metric(scale, [0, scale])
                d = diagnose_geometry(m, construct_canonical(m, 4))
                residuals = d.residuals
                self.assertEqual(residuals['relative_frobenius_status'], 'defined')
                self.assertEqual(residuals['relative_operator_status'], 'defined')
                self.assertLess(residuals['relative_frobenius'], 1e-10)
                self.assertLess(residuals['relative_operator'], 1e-10)
                json.dumps(to_json_record(d), allow_nan=False)

    def test_adversarial_cli_checks_extreme_scales(self):
        import json, subprocess, sys
        result = subprocess.run([sys.executable, '-m', 'prepared_learning', 'check-geometry', '--suite', 'adversarial'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        for scale in (1e-200, 1e200):
            records = [r for r in report['records'] if r.get('scale') == scale]
            self.assertTrue(records)
            self.assertTrue(all(r['derivative_relative_error'] < 1e-10 and r['update_scaled_error'] < 1e-10 for r in records))
            self.assertTrue(all(r['diagnostics']['residuals']['relative_frobenius_status'] == 'defined' for r in records))
