"""Exact mathematical inputs only; these tests add no scientific observations."""
from collections.abc import Mapping
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from prepared_learning import frontier
from prepared_learning.frontier import solve_frontier
from prepared_learning.methods import make_method, PreparationError
from prepared_learning.metrics import apply_metric, metric_resources
from prepared_learning.quadratic import build_oracle_qp, diagnose_qp, solve_qp
from prepared_learning.records import (BudgetSpec, Context, PreparationSpec, PriorAsset,
                                       SourceTask, to_json_record)

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/rank_one_canonical_regression.json').read_text())


def arguments(d=64, width=8, k=.6, n=8, noise=FIXTURE['noise_variance'], depth=2,
              spectrum=None, rotated=False, metadata=None):
    s = np.array([1.] + [0.]*(d-1) if spectrum is None else spectrum)
    # A deterministic nonidentity orthogonal basis, without random source tasks.
    u = np.arange(1., d+1); u /= np.linalg.norm(u)
    basis = np.eye(d)-2*np.outer(u, u) if rotated else np.eye(d)
    prior = PriorAsset((basis*s)@basis.T, basis, s, noise, 'oracle', metadata or {}, 'mathematical-only')
    norm = (depth-1)*k**(1/(depth-1))
    return prior, BudgetSpec(depth, norm, (d+1)*width+depth-2, 1.), PreparationSpec(n, d, 'gaussian', '1/(2n)', 1.)


def captured_native(*args, **kwargs):
    # Both branch QPs have eight variables; F has negative floor coefficients.
    A = kwargs['constraints']['jac'](np.zeros(8))
    b = 'F' if np.any(A[:-1, -1] < 0) else 'Z'
    return OptimizeResult(success=True, status=0, message='captured mathematical native iterate',
                          x=np.array(FIXTURE['branches'][b]['native_scaled_iterate']))


class RankOneCanonicalTests(unittest.TestCase):
    def target(self, qp, s, k, width, branch):
        self.assertTrue(hasattr(frontier, '_canonical_rank_one_target'), 'missing structural target')
        return frontier._canonical_rank_one_target(qp, np.array(s), k, width, branch)

    def canonical(self, result):
        self.assertIn('rank_one_canonicalization', result.diagnostics.kkt)
        return result.diagnostics.kkt['rank_one_canonicalization']

    def test_captured_rank_one_has_equal_production_points_and_preserved_history(self):
        for captured in (False, True):
            with self.subTest(captured=captured):
                if captured:
                    with patch('prepared_learning.quadratic.minimize', side_effect=captured_native):
                        result = solve_frontier(*arguments())
                else:
                    result = solve_frontier(*arguments())
                for b in (result.F, result.Z):
                    np.testing.assert_array_equal(b.variables, [.6]+[0.]*7)
                    self.assertEqual(b.objective, 0.29084625713422985)
                    self.assertEqual(b.status, 'completed')
                    record = self.canonical(b)
                    self.assertEqual(record['status'], 'applied')
                    self.assertEqual(set(record), {'status','reason','pre_variables','pre_diagnostics',
                        'target_variables','corrected_indices','attempted_correction_inf','correction_inf','post_diagnostics'})
                    qp=build_oracle_qp(FIXTURE['spectrum'],8,FIXTURE['noise_variance'],.6,8,b.branch)
                    fresh=diagnose_qp(qp,b.variables)
                    self.assertEqual(to_json_record(b.diagnostics.feasibility),to_json_record(fresh.feasibility))
                    self.assertEqual(to_json_record(b.diagnostics.dual),to_json_record(fresh.dual))
                    self.assertEqual(b.diagnostics.kkt['stationarity_inf'],fresh.kkt['stationarity_inf'])
                    self.assertEqual(b.diagnostics.kkt['final_point_origin'], 'rank_one_canonical')
                    self.assertTrue(b.diagnostics.kkt['native_solver_success'])
                    self.assertEqual(b.diagnostics.kkt['native_solver_status'], 0)
                self.assertEqual(result.selected_branches, ('F', 'Z'))
                self.assertEqual(result.selection_status, 'unresolved')
                f = result.F; pre = self.canonical(f)
                np.testing.assert_array_equal(pre['pre_variables'], FIXTURE['branches']['F']['retained_variables'])
                self.assertEqual(pre['pre_diagnostics']['kkt']['repair_inf'], 3.363713410704106e-17)
                self.assertEqual(pre['correction_inf'], 1.1102230246251565e-16)
                self.assertEqual(pre['corrected_indices'], tuple(range(8)))
                self.assertEqual(f.diagnostics.kkt['candidate_origin'], 'native')
                self.assertIn('negative_multiplier', f.diagnostics.kkt['active_set_polish_rejection_reasons'])
                np.testing.assert_array_equal(f.diagnostics.kkt['active_set_polish_candidate_scaled'], FIXTURE['branches']['F']['polish_candidate_scaled'])
                np.testing.assert_array_equal(f.diagnostics.kkt['active_set_polish_multipliers_scaled'], FIXTURE['branches']['F']['polish_multipliers_scaled'])
                self.assertEqual(f.metric.floor, 0.)
                self.assertEqual(np.count_nonzero(f.metric.increments), 1)
                self.assertEqual(f.metric.support_metadata.exact_zero_multiplicity, 63)
                self.assertEqual(metric_resources(f.metric, 2).trainable_parameters, 65)
        try:
            state = make_method('B04', {}).prepare(*arguments())
        except PreparationError as exc:
            self.fail(f'captured mathematical B04 preparation still fails: {exc}')
        self.assertEqual(state.resources.trainable_parameters, 65)
        self.assertEqual(state.metric.floor, 0.)
        self.assertEqual(state.solver_diagnostics['F']['objective'], state.solver_diagnostics['Z']['objective'])

    def test_scalar_active_inactive_boundary_and_rotations(self):
        for d, width, n, noise, depth in ((2,2,1,0.,2), (64,2,8,0.,3),
                                         (64,64,8,.5,8), (2,2,13,.5,3)):
            q0 = (n+1)/n+(1+noise)/n
            for k in (.25, 1/q0, 2.):
                with self.subTest(d=d, width=width, depth=depth, k=k):
                    args = arguments(d, width, k, n, noise, depth, rotated=True)
                    # Frontier computes its actual budget using logs; use exactly that
                    # budget, then derive the scalar independently from n and noise.
                    actual_k = math.exp((depth-1)*(math.log(args[1].initialization_norm_cap)-math.log(depth-1)))
                    a = min(actual_k, 1/q0)
                    result = solve_frontier(*args)
                    for b in (result.F, result.Z):
                        self.assertEqual(b.status, 'completed')
                        np.testing.assert_array_equal(b.variables, [a]+[0.]*(len(b.variables)-1))
                        self.assertEqual(b.metric.floor, 0.)
                        self.assertEqual(metric_resources(b.metric, depth).trainable_parameters, d+depth-1)
                        vector = np.arange(d, dtype=float)
                        expected = a*args[0].basis[:,0]*(args[0].basis[:,0]@vector)
                        np.testing.assert_allclose(apply_metric(b.metric, vector), expected, atol=2e-14, rtol=2e-14)

    def test_weak_positive_and_rank_metadata_do_not_select_structure(self):
        for s in ([1.,1e-20], [1.,np.nextafter(0.,1.)], [.5,.5]):
            for branch in ('F','Z'):
                qp = build_oracle_qp(s, 10**16, 0., 10., 2, branch)
                self.assertIsNone(self.target(qp, s, 10., 2, branch))
            result = solve_frontier(*arguments(2, 2, 10., 10**16, 0., spectrum=s, metadata={'rank':1}))
            for b in (result.F, result.Z):
                self.assertEqual(self.canonical(b)['status'], 'ineligible')
                if s[1] == 1e-20: self.assertEqual(b.status,'completed')
                if b.status == 'completed':
                    self.assertGreater(apply_metric(b.metric, np.array([0.,1.]))[1], 0.)
        # Metadata rank eight cannot veto the actual supplied exact zeros.
        result = solve_frontier(*arguments(metadata={'rank':8}))
        self.assertEqual(self.canonical(result.F)['status'], 'applied')

    def test_width_one_scalar_and_zero_budget_are_ineligible(self):
        for d, width, k, depth in ((2,1,2.,2),(64,1,2.,3),(1,1,.6,2),(64,8,0.,8)):
            args = arguments(d, width, k, depth=depth)
            result = solve_frontier(*args)
            for b in (result.F, result.Z):
                record = self.canonical(b)
                self.assertEqual(record['status'], 'ineligible')
                self.assertIsNone(record['pre_diagnostics'])
            if k == 0:
                self.assertIsNone(result.F.diagnostics.kkt['native_solver_success'])
                self.assertEqual(metric_resources(result.F.metric, depth).trainable_parameters, d+depth-1)
                self.assertEqual(result.F.diagnostics.kkt['admitted_candidate_original_units'], (0.,)*8)
            else:
                qsum = 9/8+d*(1+FIXTURE['noise_variance'])/8
                actual_k = math.exp((depth-1)*(math.log(args[1].initialization_norm_cap)-math.log(depth-1)))
                self.assertAlmostEqual(result.F.metric.floor, min(actual_k, 1/qsum))
                self.assertGreater(result.F.metric.floor, 0.)
                if d>1: self.assertNotEqual(result.F.metric.floor, result.Z.variables[0])

    def test_native_quality_and_independent_failures_cannot_be_rescued(self):
        for native in (OptimizeResult(success=False,status=9,message='failed',x=np.r_[1.,np.zeros(7)]),
                       OptimizeResult(success=True,status=0,message='malformed',x=np.zeros(2)),
                       OptimizeResult(success=True,status=0,message='nonfinite',x=np.full(8,np.nan))):
            with self.subTest(native=native.message), patch('prepared_learning.quadratic.minimize', return_value=native):
                result = solve_frontier(*arguments())
                for b in (result.F,result.Z):
                    self.assertEqual(b.status, 'failed')
                    self.assertEqual(self.canonical(b)['status'], 'skipped')
                    self.assertFalse(b.diagnostics.kkt['solver_success'])
                    json.dumps(to_json_record(b), allow_nan=False)
                    if native.message != 'failed': self.assertIsNone(b.diagnostics.kkt['admitted_candidate_original_units'])
        # Frontier's pre gate must independently admit the returned solver point.
        with patch('prepared_learning.frontier.qp_quality_passes', return_value=False):
            result = solve_frontier(*arguments())
            self.assertEqual(result.F.status, 'failed')
            self.assertEqual(self.canonical(result.F)['status'], 'skipped')
        # F pre passes, F post fails; Z pre and post pass.
        with patch('prepared_learning.frontier.qp_quality_passes', side_effect=lambda diag,k: diag.dual['primal'] != 0.29084625713422985):
            result = solve_frontier(*arguments())
            self.assertEqual(result.F.status, 'failed')
            record = self.canonical(result.F)
            self.assertEqual(record['status'], 'rejected')
            self.assertEqual(record['correction_inf'], 0.)
            self.assertGreater(record['attempted_correction_inf'], 0.)
            np.testing.assert_array_equal(result.F.variables, FIXTURE['branches']['F']['retained_variables'])
            self.assertIsNotNone(record['post_diagnostics'])
            json.dumps(to_json_record(result), allow_nan=False)
        with patch('prepared_learning.frontier.solve_rank_gains', return_value=np.zeros(64)):
            result = solve_frontier(*arguments())
            self.assertEqual(result.Z.status, 'failed')
            self.assertEqual(self.canonical(result.Z)['status'], 'skipped')
        # An independent reference within its existing tolerance is checked,
        # never copied into the production target.
        with patch('prepared_learning.frontier.solve_rank_gains', return_value=np.r_[.60000001,np.zeros(63)]):
            result = solve_frontier(*arguments())
            self.assertEqual(result.Z.status,'completed')
            self.assertEqual(result.Z.variables[0],.6)
        # Isolate the final independent guard: a quality-admitted pre point
        # agrees with the reference, but the exact target differs by >1e-6.
        real_solve = solve_qp
        def shifted_pre(qp, scale):
            x, diag = real_solve(qp, scale)
            if np.all(qp.A[:-1] == np.eye(8)): x[0] = .600001
            return x, diag
        with patch('prepared_learning.frontier.solve_qp', side_effect=shifted_pre), \
             patch('prepared_learning.frontier.solve_rank_gains', return_value=np.r_[.600001,np.zeros(63)]):
            result = solve_frontier(*arguments())
            self.assertEqual(result.Z.status, 'failed')
            record = self.canonical(result.Z)
            self.assertEqual(record['status'], 'rejected')
            self.assertEqual(result.Z.variables[0], .600001)
            self.assertEqual(record['pre_diagnostics']['kkt']['rank_oracle_max_relative_error'], 0.)
            self.assertGreater(record['post_diagnostics']['kkt']['rank_oracle_max_relative_error'], 1e-6)

    def test_total_repair_is_exact_distance_not_sum_of_stages(self):
        # Captured floor feasibility correction increases small entries, then
        # canonicalization removes them. The stage norms therefore do not add.
        result = solve_frontier(*arguments())
        for b in (result.F,result.Z):
            kkt=b.diagnostics.kkt; record=self.canonical(b)
            admitted=kkt['admitted_candidate_original_units']
            self.assertIsInstance(admitted, tuple)
            self.assertTrue(all(type(v) is float for v in admitted))
            exact=float(np.max(np.abs(b.variables-np.array(admitted))))
            self.assertEqual(kkt['repair_inf'], exact)
            if b.branch=='F':
                self.assertNotEqual(exact, record['correction_inf']+record['pre_diagnostics']['kkt']['repair_inf'])
                self.assertGreater(record['pre_variables'][3]-admitted[3],0.)
                self.assertLess(b.variables[3]-record['pre_variables'][3],0.)
                for key in ('feasibility_repair_inf','active_face_reconstruction_inf'):
                    self.assertEqual(kkt[key],record['pre_diagnostics']['kkt'][key])
            encoded=json.dumps(to_json_record(b.diagnostics), allow_nan=False)
            decoded=json.loads(encoded)
            self.assertEqual(tuple(decoded['kkt']['admitted_candidate_original_units']), admitted)
            np.testing.assert_array_equal(np.array(decoded['kkt']['admitted_candidate_original_units']).view(np.uint64),
                                          np.array(admitted).view(np.uint64))
            self.assertEqual(decoded['kkt']['repair_inf'], exact)
            def no_arrays(v):
                self.assertNotIsInstance(v,np.ndarray)
                if isinstance(v,Mapping):
                    for value in v.values(): no_arrays(value)
                elif isinstance(v,(tuple,list)):
                    for value in v: no_arrays(value)
            no_arrays(record)
            from prepared_learning.benchmark import storage_inventory
            inventory=storage_inventory(record)
            self.assertEqual(inventory['array_bytes'],0)
            self.assertEqual(inventory['serialized_array_scalars'],0)
            self.assertEqual(inventory['serialized_bytes'],len(json.dumps(to_json_record(record),
                sort_keys=True,separators=(',',':'),allow_nan=False).encode()))
            with self.assertRaises(TypeError): record['status']='changed'
        # Actual existing envelope accounting must include the enlarged full payload.
        from prepared_learning.pilot import measure_execution_envelope
        row={'frontier_result':to_json_record(result),'tasks':[],'serialization_ns':1234567}
        size=len(json.dumps(row,allow_nan=False).encode())
        envelope=measure_execution_envelope([row],[])
        self.assertEqual(envelope['max_preparation_record_bytes'],size)
        self.assertEqual(envelope['serialization_seconds_per_byte'], .001234567/size)

    def test_extreme_representability_remains_explicit(self):
        qp=build_oracle_qp([1.,0.],8,0.,.6,2,'F')
        huge=replace(qp,hessian=np.diag([1e308,1.]))
        with np.errstate(over='raise',under='ignore'):
            target=self.target(huge,[1.,0.],.6,2,'F')
        self.assertEqual(target[0],1e-308)
        tiny=np.nextafter(0.,1.)
        self.assertEqual(self.target(qp,[1.,0.],tiny,2,'F')[0],tiny)
        for invalid in (replace(qp,hessian=np.diag([np.inf,1.])),
                        replace(qp,hessian=np.diag([1.,0.])),
                        replace(qp,hessian=np.diag([1e308,1.]),linear=np.array([-2*tiny,0.])),
                        replace(qp,linear=np.zeros(3))):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.target(invalid,[tiny,0.] if invalid.linear[0] == -2*tiny else [1.,0.],.6,2,'F')
        # Existing objective/budget scaling errors remain failures, without a
        # canonical zero being substituted for an unrepresentable positive a.
        with self.assertRaisesRegex(ValueError,'objective scaling'):
            solve_qp(replace(qp,linear=qp.linear*1e-300,hessian=qp.hessian*1e-300),tiny)
        args=arguments(depth=8)
        with self.assertRaisesRegex(ValueError,'budget power'):
            solve_frontier(args[0],replace(args[1],initialization_norm_cap=tiny),args[2])
        self.assertTrue(hasattr(frontier,'_canonical_rank_one_target'))
        with patch('prepared_learning.frontier._canonical_rank_one_target',side_effect=ValueError('unrepresentable target')):
            result=solve_frontier(*arguments())
        for b in (result.F,result.Z):
            self.assertEqual(b.status,'failed')
            self.assertEqual(self.canonical(b)['status'],'rejected')
            self.assertIsNone(self.canonical(b)['target_variables'])
            json.dumps(to_json_record(b),allow_nan=False)
        with patch('prepared_learning.frontier.diagnose_qp',side_effect=ValueError('attempt diagnosis failed')):
            result=solve_frontier(*arguments())
        self.assertEqual(result.F.status,'failed')
        self.assertEqual(self.canonical(result.F)['status'],'rejected')
        self.assertIsNotNone(self.canonical(result.F)['target_variables'])
        def nonfinite_diagnosis(qp,x):
            diag=diagnose_qp(qp,x)
            return replace(diag,dual={**diag.dual,'primal':float('nan')})
        with patch('prepared_learning.frontier.diagnose_qp',side_effect=nonfinite_diagnosis):
            result=solve_frontier(*arguments())
        for b in (result.F,result.Z):
            record=self.canonical(b)
            self.assertEqual(b.status,'failed')
            self.assertEqual(record['status'],'rejected')
            self.assertIsNone(record['post_diagnostics']['dual']['primal'])
            self.assertEqual(b.diagnostics.kkt['repair_inf'],record['pre_diagnostics']['kkt']['repair_inf'])
            json.dumps(to_json_record(b),allow_nan=False)

    def test_empirical_and_generic_qps_do_not_use_oracle_canonicalization(self):
        self.assertTrue(hasattr(frontier,'_canonical_rank_one_target'))
        from prepared_learning.priors import build_empirical_qp, fit_empirical_gains
        a=Context(np.array([[1.,1.],[2.,0.]]),np.array([1.,2.]),'synthetic-a')
        b=Context(np.array([[2.,1.],[1.,1.]]),np.array([2.,1.]),'synthetic-b')
        args=arguments(2,2,1.)
        qp=build_empirical_qp([SourceTask(a,b,'synthetic')],np.eye(2),args[1],'Z',0.)
        self.assertNotEqual(qp.hessian[0,1],0.)
        with patch('prepared_learning.frontier._canonical_rank_one_target',side_effect=AssertionError('oracle-only helper reached')):
            _,diag=solve_qp(qp,1.)
            self.assertNotIn('rank_one_canonicalization',diag.kkt)
            self.assertNotIn('final_point_origin',diag.kkt)
            fit=fit_empirical_gains(qp,args[1],np.eye(2),'Z')
            self.assertEqual(fit.status,'completed')
            state=make_method('B11:Z',{'fitted_metric':fit.metric}).prepare(*args)
            self.assertGreater(state.resources.trainable_parameters,0)
            self.assertNotIn('rank_one_canonicalization',fit.diagnostics.kkt)


if __name__ == '__main__':
    unittest.main()
