"""Development-only observed-array fixtures for local branch diagnostics."""
import copy
import importlib
import json
import unittest
import numpy as np
from scipy.optimize import brentq
from prepared_learning.methods import make_method, PreparationError
from prepared_learning.priors import _asset, shrink_prior
from prepared_learning.records import BudgetSpec, PreparationSpec, Context, SourceTask, SelectionResult, content_hash, to_json_record
from prepared_learning.frontier import solve_frontier
from prepared_learning.selection import select_method


class BranchFixture:
    def __init__(self, method='B04', width=2, validation=None, seed=412):
        self.method = make_method(method, {})
        self.budget = BudgetSpec(2, 1., 5*width, 1.)
        self.prep = PreparationSpec(4, 4, 'gaussian_isotropic', 'squared_error_over_2n', 1.)
        self.base = _asset(np.diag([.5, .5, 0., 0.]), 0., {'source_hash': 'development-fixture'})
        rng = np.random.default_rng(seed)
        X, W = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
        y, z = rng.normal(size=4), rng.normal(size=4)
        self.validation = validation or tuple(SourceTask(Context(X, scale*y, f'{i}a'),
            Context(W, scale*z, f'{i}b'), f'val{i}') for i, scale in enumerate((1., 1.01)))
        self.config = {'budget': to_json_record(self.budget), 'preparation': to_json_record(self.prep),
            'information_policy': {'noise_access': 'known', 'permitted_noise_variance': 0.},
            'source_access_manifest': {'training_keys': ['train'], 'validation_keys': ['val0', 'val1']}}

    def row(self, gamma):
        prior = shrink_prior(self.base, float(gamma))
        state = self.method.prepare(prior, self.budget, self.prep)
        losses = [np.mean([np.mean((self.method.predict(self.method.adapt(state, c), q.X)-q.y)**2)
            for c, q in ((t.A,t.B),(t.B,t.A))]) for t in self.validation]
        return {'candidate_id': f'{self.method.method_id}:{gamma}', 'method_id': self.method.method_id,
            'status': 'completed', 'gamma': float(gamma), 'asset': to_json_record(prior),
            'asset_hash': prior.asset_hash, 'prepared_state': to_json_record(state),
            'prepared_hash': content_hash(state), 'source_hash': 'development-fixture',
            'validation_hash': content_hash(self.validation), 'validation_task_losses': list(losses),
            'validation_loss': float(np.mean(losses)), 'validation_loss_se': float(np.std(losses, ddof=1)/np.sqrt(2)),
            'unique_labels': 24, 'validation_labels': 16, 'validation_tasks': 2,
            'options': {}, 'tuning_kind': 'default', 'tuning_value': None, 'rank': None, 'elapsed_ns': 1}

    def switch(self):
        def gap(g):
            result = solve_frontier(shrink_prior(self.base, g), self.budget, self.prep)
            return result.F.objective-result.Z.objective
        root = brentq(gap, .2, .5)
        return root-1e-5, root+1e-5


class BranchBoundsTests(unittest.TestCase):
    def setUp(self):
        try:
            self.bounds = importlib.import_module('prepared_learning.tuning_bounds')
        except ModuleNotFoundError:
            self.fail('branch-envelope diagnostic is not implemented')

    def check(self, fixture, a=.3, b=.30001):
        left, right = fixture.row(a), fixture.row(b)
        incumbent = min((left,right), key=lambda row: row['validation_loss'])
        return self.bounds.branch_interval_check(left, right, incumbent, fixture.validation, fixture.config), left, right

    def test_validation_design_matches_both_literal_directions_and_f_tail(self):
        f = BranchFixture()
        # A nontrivial rotation prevents an identity-basis shortcut.
        U, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(4,4)))
        for branch, gains in [('F', np.array([.7,.2])), ('Z', np.array([.7,.2]))]:
            D, y = self.bounds._validation_design(f.validation, U, 2, branch)
            full = np.r_[gains[0], np.repeat(gains[1],3)] if branch=='F' else np.r_[gains,0.,0.]
            metric = (U*full)@U.T
            literal = np.mean([np.mean([np.mean((q.X@metric@(c.X.T@c.y/len(c.y))-q.y)**2)
                for c,q in ((t.A,t.B),(t.B,t.A))]) for t in f.validation])
            self.assertAlmostEqual(float(np.sum((D@gains-y)**2)), literal, places=12)
            if branch=='F':
                t=f.validation[0]; z=t.A.X.T@t.A.y/4
                expected=(t.B.X@z-(t.B.X@U[:,:1])[:,0]*(U[:,:1].T@z)[0])/4
                np.testing.assert_allclose(D[:4,-1], expected, atol=1e-14)

    def test_real_switch_resolves_without_changing_pointwise_maps_or_tau(self):
        f=BranchFixture(); a,b=f.switch(); left,right=f.row(a),f.row(b)
        before=content_hash([left,right]); incumbent=min((left,right),key=lambda r:r['validation_loss'])
        self.assertNotEqual(left['prepared_state']['solver_diagnostics']['selected_branches'], right['prepared_state']['solver_diagnostics']['selected_branches'])
        tau=min(left['validation_loss_se'],right['validation_loss_se'])
        self.assertGreater(abs(left['validation_loss']-right['validation_loss']),tau)
        out=self.bounds.branch_interval_check(left,right,incumbent,f.validation,f.config)
        self.assertEqual(out['status'],'resolved',out)
        self.assertEqual(out['certificate_level'],'float64_diagnostic')
        self.assertEqual(out['objective_tolerance'],tau)
        self.assertEqual(content_hash([left,right]),before)
        self.assertEqual(out['incumbent_candidate_id'],incumbent['candidate_id'])
        self.assertEqual(set(out['branches']),{'F','Z'})
        self.assertEqual(len(out['probes']),4)
        self.assertGreater(out['elapsed_ns'],0)
        for probe in out['probes']:
            self.assertFalse(probe['eligible']); self.assertGreater(probe['elapsed_ns'],0)
            self.assertGreater(probe['anchor_solver_error'],0)
            self.assertGreater(probe['uniform_solver_error'],0)
            required=out['error_model']['sensitivity_full_gain']*(b-a)+probe['anchor_solver_error']+probe['uniform_solver_error']
            self.assertGreaterEqual(probe['optimizer_distance_upper'],required)
        self.assertEqual(out['additional_labels'],0)
        self.assertEqual(out['additional_independent_tasks'],0)
        # Falsification only: dense samples are not the proof behind the bound.
        ambiguous=0
        for g in np.linspace(a,b,41):
            try: row=f.row(g)
            except PreparationError:
                ambiguous+=1
                continue  # Envelope is conditional on accepted future outputs.
            self.assertGreaterEqual(row['validation_loss'],out['lower_bound'])
        self.assertLessEqual(ambiguous,1)
        json.dumps(out,allow_nan=False)

    def test_single_branch_methods_and_actual_b02_output(self):
        for method,branch in [('B02','Z'),('B03','F')]:
            f=BranchFixture(method)
            out,left,right=self.check(f)
            self.assertEqual(out['status'],'resolved',out)
            self.assertEqual(set(out['branches']),{branch})
            self.assertEqual(len(out['probes']),2)
            for p in out['probes']:
                self.assertEqual(p['solution_source'],'actual_b02_metric' if method=='B02' else 'retained_frontier')
            self.assertIsNone(left['prepared_state']['solver_diagnostics'] if method=='B02' else None)

    def test_unselected_lower_loss_branch_remains_conservatively_unresolved(self):
        tasks=tuple(SourceTask(Context(np.eye(4),np.array([v,0,0,0]),f'{i}a'),
            Context(np.eye(4),np.array([v,0,0,0]),f'{i}b'),f'val{i}') for i,v in enumerate((1.,1.00001)))
        f=BranchFixture(validation=tasks); out,left,right=self.check(f,.8,.800001)
        self.assertEqual(left['prepared_state']['solver_diagnostics']['selected_branches'],['F'])
        self.assertEqual(out['status'],'unresolved',out)
        self.assertGreater(out['improvement_bound'],out['objective_tolerance'])
        self.assertLess(out['branches']['Z']['lower_bound'],out['branches']['F']['lower_bound'])

    def test_fixed_family_and_successful_status_are_not_sufficient_without_witnesses(self):
        f=BranchFixture(); left,right=f.row(.3),f.row(.30001)
        mutations=[lambda r:r['asset'].pop('fit_metadata'),
            lambda r:r['asset']['basis'][0].__setitem__(0,-1.),
            lambda r:r['asset'].__setitem__('noise_variance',.1),
            lambda r:r['prepared_state']['budget'].__setitem__('depth',3),
            lambda r:r['prepared_state']['solver_diagnostics']['F']['diagnostics']['dual']['multipliers'].__setitem__(0,-1.),
            lambda r:r['prepared_state']['solver_diagnostics']['F'].__setitem__('status','failed'),
            lambda r:r['prepared_state']['solver_diagnostics']['F']['diagnostics']['kkt'].__setitem__('solver_success',False),
            lambda r:r['prepared_state']['solver_diagnostics']['F']['variables'].__setitem__(0,100.),
            lambda r:r['prepared_state']['solver_diagnostics']['F']['metric'].__setitem__('basis',[1.,0.,0.,0.]),
            lambda r:r['prepared_state']['solver_diagnostics']['Z'].__setitem__('objective',-10.),
            lambda r:r.__setitem__('validation_loss',r['validation_loss']+1.),
            lambda r:r['options'].__setitem__('step_size',2.)]
        for mutate in mutations:
            row=copy.deepcopy(right); mutate(row)
            row['prepared_hash']=content_hash(row['prepared_state'])
            with self.subTest(row=row['candidate_id'],mutation=mutate):
                out=self.bounds.branch_interval_check(left,row,left,f.validation,f.config)
                self.assertEqual(out['status'],'unsupported',out)
        config=copy.deepcopy(f.config); config['preparation']['loss_normalization']='wrong'
        self.assertEqual(self.bounds.branch_interval_check(left,right,left,f.validation,config)['status'],'unsupported')

    def test_task_loss_consistency_cannot_be_replaced_by_matching_mean(self):
        f=BranchFixture(); left,right=f.row(.3),f.row(.30001)
        right['validation_task_losses'].reverse()
        out=self.bounds.branch_interval_check(left,right,left,f.validation,f.config)
        self.assertEqual(out['status'],'unsupported',out)

    def test_b02_actual_metric_must_satisfy_original_qp(self):
        f=BranchFixture('B02'); left,right=f.row(.3),f.row(.30001)
        right['prepared_state']['metric']['increments'][0]+=.1
        right['prepared_hash']=content_hash(right['prepared_state'])
        out=self.bounds.branch_interval_check(left,right,left,f.validation,f.config)
        self.assertEqual(out['status'],'unsupported',out)
        self.assertIn('residual',out['reason'])

    def test_solver_allowance_accounts_for_f_tail_and_uniform_floor(self):
        f=BranchFixture('B03'); out,left,right=self.check(f)
        model=out['branches']['F']['solver_model']
        self.assertAlmostEqual(model['projection_factor'],np.sqrt(7)+6)
        self.assertEqual((model['p'],model['M']),(2,3))
        for probe in out['probes']:
            self.assertGreaterEqual(probe['uniform_solver_error'],probe['anchor_solver_error'])
            self.assertGreater(probe['uniform_map_loss_padding'],0)
            self.assertGreater(probe['loss_padding'],0)
            self.assertGreater(probe['gradient_padding'],0)

    def test_nonfinite_observed_arithmetic_and_underflow_are_unsupported(self):
        for scale in (1e-200,1e200):
            tasks=tuple(SourceTask(Context(scale*np.eye(4),np.ones(4),f'{i}a'),
                Context(scale*np.eye(4),np.ones(4),f'{i}b'),f'val{i}') for i in range(2))
            f=BranchFixture(); left,right=f.row(.3),f.row(.30001)
            for row in (left,right): row['validation_hash']=content_hash(tasks)
            out=self.bounds.branch_interval_check(left,right,left,tasks,f.config)
            self.assertEqual(out['status'],'unsupported',out)
            json.dumps(out,allow_nan=False)

    def test_nonpositive_missing_uncertainty_extremes_and_full_width_fail_closed(self):
        f=BranchFixture(); left,right=f.row(.3),f.row(.30001)
        for value in (None,0.,-1.,float('nan'),float('inf')):
            row=copy.deepcopy(right); row['validation_loss_se']=value
            out=self.bounds.branch_interval_check(left,row,left,f.validation,f.config)
            self.assertNotEqual(out['status'],'resolved'); json.dumps(out,allow_nan=False)
        for field,value in [('gamma',np.nextafter(.3,1.)),('gamma',float('inf'))]:
            row=copy.deepcopy(right); row[field]=value
            self.assertNotEqual(self.bounds.branch_interval_check(left,row,left,f.validation,f.config)['status'],'resolved')
        for amount in (1e-310,1e300,float('inf')):
            config=copy.deepcopy(f.config); config['budget']['initialization_norm_cap']=amount
            self.assertEqual(self.bounds.branch_interval_check(left,right,left,f.validation,config)['status'],'unsupported')
        for method in ('B02','B04'):
            wide=BranchFixture(method,width=4)
            out,_,_=self.check(wide)
            self.assertEqual(out['status'],'unsupported',out)

    def test_selector_attaches_probes_without_ranking_them(self):
        f=BranchFixture(seed=4); a,b=f.switch()
        class FixtureMethod:
            method_id='B04'
            def fit_meta(self,train,validation,config):
                rows=tuple(f.row(g) for g in config['fitting_candidates']['gammas'])
                best=min(rows,key=lambda r:r['validation_loss'])
                return SelectionResult(best['candidate_id'],rows,None)
        config={**f.config,'fitting_candidates':{'gammas':[a,b],'max_refinement_rounds':0}}
        train=(SourceTask(Context(np.eye(4),np.ones(4),'ta'),Context(np.eye(4),np.ones(4),'tb'),'train'),)
        result=select_method([FixtureMethod()],train,f.validation,config)
        best=min(result.candidate_records,key=lambda r:r['validation_loss'])
        self.assertEqual(result.selected_id,best['candidate_id'])
        self.assertEqual(len(result.candidate_records),4)
        self.assertEqual({r['gamma'] for r in result.candidate_records},{0.,a,b,1.})
        self.assertTrue(all(r['unique_labels']==24 and r['validation_tasks']==2 for r in result.candidate_records))
        checks=[c for r in result.candidate_records for c in r.get('tuning_checks',[])]
        self.assertTrue(any(c['decision']=='objective_resolved_by_branch_envelope' for c in checks),checks)
        for c in checks:
            if 'branch_envelope' in c:
                self.assertEqual(c['objective_tolerance'],c['branch_envelope']['objective_tolerance'])
        elapsed=sum(c['branch_envelope']['elapsed_ns'] for c in checks if 'branch_envelope' in c)
        self.assertGreaterEqual(best['total_tuning_elapsed_ns'],elapsed)
        self.assertEqual(best['total_candidate_evaluations'],4)

if __name__=='__main__': unittest.main()
