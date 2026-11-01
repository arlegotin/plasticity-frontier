import json
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from checks.exact_reference import (load_exact_cases, check_exact_cases,
    counterexample_coefficient, phase_results, primary_results, scalar_optimum,
    zero_parameter_count)

FIXTURE = Path('tests/fixtures/exact_cases.json')

class TheoryTests(unittest.TestCase):
    def test_phase_thresholds_and_six_risks(self):
        thresholds, risks = phase_results(load_exact_cases(FIXTURE)['phase'])
        self.assertEqual(thresholds, (Fraction(4, 25), Fraction(8, 9)))
        self.assertEqual(risks, [Fraction(542, 625), Fraction(493, 625),
                                Fraction(677, 1025), Fraction(573, 1025),
                                Fraction(5, 9), Fraction(5, 9)])

    def test_primary_fixture_arithmetic(self):
        cases = load_exact_cases(FIXTURE)
        result = primary_results(cases['primary_richer_k_3_5'])
        primary_gap = result['Z'] - result['F']
        self.assertEqual(primary_gap, Fraction(1266883, 39703750))
        checks = check_exact_cases(cases)
        self.assertTrue(checks)
        self.assertTrue(all(item['status'] == 'pass' for item in checks), checks)
        self.assertTrue(all(item.get('evidence') for item in checks))

    def test_noncanonical_update(self):
        self.assertEqual(counterexample_coefficient(h=Fraction(1)), Fraction(4))
        self.assertEqual(counterexample_coefficient(h=Fraction(2)), Fraction(10))

    def test_zero_and_scalar_limits(self):
        self.assertEqual(scalar_optimum(Fraction(3, 5), Fraction(5, 4), Fraction(1, 2)), Fraction(4, 7))
        self.assertEqual(zero_parameter_count(6, 3), 8)
        fixture = json.loads(FIXTURE.read_text())
        self.assertNotIn('monte_carlo', json.dumps(fixture).lower())

    def test_cli_and_bad_fixture(self):
        result = subprocess.run(['.venv/bin/python', '-m', 'prepared_learning', 'check-theory', '--config', 'configs/exploration/exact_cases.json'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        output = json.loads(result.stdout)
        self.assertEqual(output['status'], 'pass')
        self.assertIn('proof_obligations', output)
        config = json.loads(Path('configs/exploration/exact_cases.json').read_text())
        config['budget']['initialization_norm_cap'] = 0.7
        with tempfile.TemporaryDirectory() as tmp:
            altered = Path(tmp) / 'config.json'
            altered.write_text(json.dumps(config))
            bad = subprocess.run(['.venv/bin/python', '-m', 'prepared_learning', 'check-theory', '--config', str(altered)], text=True, capture_output=True)
        self.assertNotEqual(bad.returncode, 0, bad.stdout)
        cases = load_exact_cases(FIXTURE)
        cases['primary_richer_k_3_5']['F']['risk'] = [1, 1]
        self.assertTrue(any(x['status'] == 'failed' for x in check_exact_cases(cases)))
