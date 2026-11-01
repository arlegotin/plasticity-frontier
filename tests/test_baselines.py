import unittest
import numpy as np
from numpy.testing import assert_allclose
from prepared_learning.baselines import solve_rank_gains, ridge_coefficient, posterior_coefficient, unrestricted_fixed_metric
from prepared_learning.methods import make_method
from prepared_learning.records import Context, PriorAsset, BudgetSpec, PreparationSpec

class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.context = Context(np.eye(2), np.array([1., 2.]), 't')
        self.prior = PriorAsset(np.eye(2)/2, np.eye(2), np.array([.5,.5]), 1., 'oracle', {}, 'a')
        self.task = PreparationSpec(2, 2, 'gaussian', '1/(2n)', 1.)
        self.budget = BudgetSpec(2, .6, 6, 1.)
    def test_ridge_penalty_units(self):
        assert_allclose(ridge_coefficient(self.context, 2), [1/3,2/3])
        assert_allclose(posterior_coefficient(self.context, self.prior), [1/3,2/3])
    def test_singular_noiseless_posterior(self):
        p = PriorAsset(np.diag([1.,0.]), np.eye(2), np.array([1.,0.]), 0., 'oracle', {}, 'b')
        assert_allclose(posterior_coefficient(Context(np.eye(2), np.array([2.,0.]), 't'), p), [2.,0.])
    def test_rank_budget_shrinkage(self):
        s=np.array([.5,.3,.2]); g=solve_rank_gains(s, 4, 0, .3, 2)
        self.assertGreaterEqual(g.min(),0); self.assertLessEqual(sum(g), .3+1e-12)
        q=1.25*s+.25
        self.assertAlmostEqual(2*(s[0]-q[0]*g[0]),2*(s[1]-q[1]*g[1]),places=12)
        assert_allclose(solve_rank_gains(s,4,0,10,3),s/q)
    def test_network_and_direct_match(self):
        m=make_method('B02', {}); prepared=m.prepare(self.prior,self.budget,self.task)
        a=m.adapt(prepared,self.context)
        direct=make_method('B05:B02', {'metric':prepared.metric}).prepare(self.prior,self.budget,self.task)
        assert_allclose(a.coefficient,make_method('B05:B02', {'metric':prepared.metric}).adapt(direct,self.context).coefficient,atol=1e-14)
    def test_adapt_rejects_wrong_n_or_d(self):
        m=make_method('B00', {}); p=m.prepare(self.prior,self.budget,self.task)
        with self.assertRaises(ValueError): m.adapt(p,Context(np.ones((1,2)),np.ones(1),'x'))
        with self.assertRaises(ValueError): m.adapt(p,Context(np.ones((2,3)),np.ones(2),'x'))
    def test_projected_pseudoinverse_and_smaller_solves(self):
        U=np.array([[1.],[0.]])
        assert_allclose(ridge_coefficient(self.context,0,U),[1.,0.])
        X=np.array([[1.,2.,0.],[0.,1.,1.]])
        c=Context(X,np.array([1.,-1.]),'t')
        assert_allclose(ridge_coefficient(c,2),np.linalg.solve(X.T@X+2*np.eye(3),X.T@c.y))
    def test_zero_isotropic_and_fixed_gains(self):
        from prepared_learning.metrics import dense_metric
        zero=make_method('B00',{}); p=zero.prepare(self.prior,self.budget,self.task)
        assert_allclose(zero.adapt(p,self.context).coefficient,[0,0])
        iso=make_method('B01',{}); p=iso.prepare(self.prior,self.budget,self.task)
        assert_allclose(dense_metric(p.metric),np.eye(2)*min(.6,1/(1.5+2)))
        fixed=unrestricted_fixed_metric(self.prior,self.task)
        assert_allclose(dense_metric(fixed),np.eye(2)*(.5/(1.5*.5+1)))
    def test_variants_and_diagnostics(self):
        for mid in ('B08:gd','B08:ridge','B09:least_squares','B09:ridge','B10:posterior','B10:empirical_bayes'):
            m=make_method(mid,{}); p=m.prepare(self.prior,self.budget,self.task); a=m.adapt(p,self.context)
            self.assertEqual(m.diagnostics(p,a)['method_id'],mid)
            if mid.startswith('B10'): self.assertEqual(m.diagnostics(p,a)['covariance_rank'],2)
            if mid != 'B08:gd': self.assertIsNone(p.resources.initialization_norm)
