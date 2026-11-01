import dataclasses
import unittest
import numpy as np
from prepared_learning.records import BudgetSpec, PreparationSpec, SamplePlan, RunConfig, TaskKey
from prepared_learning.tasks import generate_task, generate_source_task, rng_for, population_for


def config():
    return RunConfig(1, {'spectrum': [0.5, 0.5, 0., 0.], 'sigma2': 0.7}, BudgetSpec(2, 2., 20, 1.), PreparationSpec(3, 4, 'isotropic_gaussian', 'squared_error_over_2n', 1.), {'dtype':'float64'}, {}, (), {}, SamplePlan('nested_task_simulation', 1, 1, 1, 1, 5), {})

class TaskTests(unittest.TestCase):
    def test_keys_ignore_batch_and_method_order(self):
        cfg = config()
        keys = [TaskKey(123, 1, 0, i, 0) for i in range(32)]
        expected = [generate_task(cfg, k) for k in keys]
        for batch in (1, 7, 32):
            for start in range(0, 32, batch):
                for k, (obs, truth) in zip(keys[start:start+batch], expected[start:start+batch]):
                    for _ in reversed(('one', 'two')):
                        other, t = generate_task(cfg, k)
                        np.testing.assert_array_equal(obs.X, other.X)
                        np.testing.assert_array_equal(obs.y, other.y)
                        np.testing.assert_array_equal(truth.theta, t.theta)
        self.assertFalse(np.array_equal(rng_for(keys[0]).normal(size=8), rng_for(dataclasses.replace(keys[0], stream_id=1)).normal(size=8)))
        self.assertFalse(np.array_equal(expected[0][0].X, generate_task(cfg, dataclasses.replace(keys[0], phase_id=2))[0].X))

    def test_source_blocks_share_theta_not_noise(self):
        cfg, key = config(), TaskKey(3, 4, 0, 2, 0)
        source = generate_source_task(cfg, key)
        self.assertEqual(source.A.X.shape, (5, 4))
        self.assertEqual(source.B.X.shape, (5, 4))
        _, truth = generate_task(cfg, key)
        a = source.A.y - source.A.X @ truth.theta
        b = source.B.y - source.B.X @ truth.theta
        self.assertFalse(np.array_equal(source.A.X, source.B.X))
        self.assertFalse(np.array_equal(a, b))
        self.assertGreater(np.linalg.norm(a), 0)
        self.assertGreater(np.linalg.norm(b), 0)
        noiseless = dataclasses.replace(cfg, workload={'spectrum': [0.5, 0.5, 0., 0.], 'sigma2': 0.})
        src = generate_source_task(noiseless, key)
        _, latent = generate_task(noiseless, key)
        np.testing.assert_allclose(src.A.y, src.A.X @ latent.theta, atol=1e-14)
        np.testing.assert_allclose(src.B.y, src.B.X @ latent.theta, atol=1e-14)

    def test_truth_is_not_in_context(self):
        cfg, key = config(), TaskKey(1, 2, 3, 4, 0)
        obs, _ = generate_task(cfg, key)
        source = generate_source_task(cfg, key)
        for record in (obs, source, source.A, source.B):
            self.assertTrue(set(vars(record)).isdisjoint({'theta', 'second_moment', 'sigma2', 'rotation'}))

    def test_repeated_spectrum_rotation(self):
        cfg, key = config(), TaskKey(1, 2, 3, 4, 0)
        p, _ = population_for(cfg, key)
        vals, vecs = np.linalg.eigh(p.second_moment)
        np.testing.assert_allclose(vecs[:, 2:] @ vecs[:, 2:].T, 2*p.second_moment, atol=1e-14)
