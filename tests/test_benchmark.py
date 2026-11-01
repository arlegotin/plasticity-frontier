import tempfile
import unittest
from pathlib import Path
import numpy as np
from prepared_learning.benchmark import (normalize_peak_rss, amortized_cost, benchmark_methods, balanced_orders, memory_schedule)
from prepared_learning.methods import make_method
from prepared_learning.records import BudgetSpec, PreparationSpec, PriorAsset, Context


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.prior = PriorAsset(np.eye(2)/2, np.eye(2), np.array([.5,.5]), .1, 'oracle', {}, 'fixture')
        self.budget = BudgetSpec(2, 1., 20, 1.)
        self.preparation = PreparationSpec(2, 2, 'gaussian_isotropic', '1/(2n)', 1.)
        self.contexts = [Context(np.eye(2), np.array([1., 2.]), 'fixture')]
        self.config = {'prior': self.prior, 'budget': self.budget, 'preparation': self.preparation,
                       'pilot_blocks': 2, 'blocks': 2, 'max_repetitions': 2, 'future_tasks': [1,10],
                       'thread_policy': 'environment', 'thermal': 'unknown', 'power': 'unknown',
                       'measure_memory': False}

    def test_rss_units(self):
        self.assertEqual(normalize_peak_rss(1024, 'Darwin'), 1024)
        self.assertEqual(normalize_peak_rss(1024, 'Linux'), 1048576)

    def test_accounting_and_amortization(self):
        self.assertEqual(amortized_cost(100, 10, 5, 10), 25)
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_methods([make_method('B01', {})], self.contexts, self.config, Path(tmp))
        row = result['resources'][0]
        self.assertGreater(row['trainable_parameters'], 0)
        self.assertGreater(row['runtime_serialized_bytes'], 0)
        self.assertGreater(row['evidence_serialized_bytes'], 0)
        self.assertGreaterEqual(row['runtime_stored_scalars'], self.prior.covariance.size + self.prior.basis.size + self.prior.eigenvalues.size)

    def test_timing_boundaries_and_pair_equality(self):
        class Clock:
            def __init__(self): self.t = 0
            def __call__(self): self.t += 10; return self.t
        cfg = dict(self.config, clock=Clock())
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_methods([make_method('B01', {}), make_method('B05:B01', {})], self.contexts, cfg, Path(tmp))
        self.assertEqual(result['equality'][0]['status'], 'pass')
        row = result['timing_blocks'][0]
        for field in ('generation_ns','preparation_ns','adaptation_ns','prediction_ns','serialization_ns','end_to_end_ns'):
            self.assertIn(field, row)
        self.assertIsNone(row['generation_ns'])
        self.assertEqual(result['conditions']['input_residency'], 'provided_contexts')

    def test_order_and_thermal_strata(self):
        orders = balanced_orders(['a','b','c'], 6, 41)
        for position in range(3):
            self.assertEqual({x: sum(order[position] == x for order in orders) for x in 'abc'}, dict.fromkeys('abc', 2))
        self.assertNotEqual(orders[0], orders[1])

    def test_memory_limit_keeps_science(self):
        science = {'d': 50, 'tasks': 100, 'seed': 9}
        result = memory_schedule(science, {'batch_size': 100, 'checkpoint_tasks': 40}, pressure=True)
        self.assertEqual(result['science'], science)
        self.assertLess(result['operations']['batch_size'], 100)

    def test_no_energy_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_methods([make_method('B01', {})], self.contexts, self.config, Path(tmp))
        self.assertEqual(result['energy']['status'], 'not_run')
        self.assertIsNone(result['energy']['joules'])

    def test_shared_metric_mismatch_is_rejected(self):
        from prepared_learning.baselines import metric_from_gains
        other = metric_from_gains(np.eye(2), np.ones(2))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'metric differs'):
                benchmark_methods([make_method('B01', {}), make_method('B05:B01', {'metric': other})], self.contexts, self.config, Path(tmp))

    def test_upstream_measured_and_charged(self):
        cfg = dict(self.config, upstream_prepare=lambda: self.prior)
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_methods([make_method('B01', {})], self.contexts, cfg, Path(tmp))
        self.assertEqual(result['prior_cost']['status'], 'measured')
        self.assertIn('amortized_with_upstream_ns_per_task', result['summaries'][0])
        self.assertEqual(result['inputs']['prior']['asset_hash'], 'fixture')

    def test_cli_exploratory_fixture(self):
        import json
        from prepared_learning.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {'schema_version': 1, 'phase': 'exploratory', 'dimension': 2, 'context_size': 2,
                        'seed': 5, 'methods': ['B01', 'B05:B01', 'B08:ridge'],
                        'pilot_blocks': 2, 'blocks': 2, 'max_repetitions': 2,
                        'output': tmp, 'measure_memory': False}
            path = Path(tmp)/'manifest.json'; path.write_text(json.dumps(manifest))
            self.assertEqual(main(['benchmark', '--manifest', str(path)]), 0)
            result = json.loads((Path(tmp)/'benchmark.json').read_text())
            self.assertEqual(result['equality'][0]['status'], 'pass')
            self.assertGreaterEqual(result['context_generation']['elapsed_ns'], 0)
