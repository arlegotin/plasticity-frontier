import unittest
import numpy as np
from prepared_learning.records import Context, SourceTask, BudgetSpec
from prepared_learning.priors import build_empirical_qp, fit_empirical_gains

class EmpiricalTests(unittest.TestCase):
    def test_singular_empirical_fit(self):
        a=Context(np.eye(2),np.array([1.,0.]),'a'); b=Context(np.eye(2),np.array([2.,0.]),'b')
        t=SourceTask(a,b,'s'); budget=BudgetSpec(2,1.,6,1.)
        for branch in ('F','Z'):
            qp=build_empirical_qp([t],np.eye(2),budget,branch,0.)
            result=fit_empirical_gains(qp,budget,np.eye(2),branch)
            self.assertEqual(result.status,'completed')
            self.assertGreaterEqual(result.diagnostics.feasibility['minimum_slack'],-1e-8)

    def test_residual_bound_uses_shared_bounded_domain_diagnostic(self):
        from prepared_learning.certificates import empirical_residual_bound
        a=Context(np.array([[1.,1.],[2.,0.]]),np.array([1.,2.]),'a')
        b=Context(np.array([[2.,1.],[1.,1.]]),np.array([2.,1.]),'b')
        budget=BudgetSpec(2,1.,6,1.); qp=build_empirical_qp([SourceTask(a,b,'s')],np.eye(2),budget,'Z',0.)
        self.assertNotEqual(qp.hessian[0,1],0.)
        fit=fit_empirical_gains(qp,budget,np.eye(2),'Z')
        self.assertAlmostEqual(fit.diagnostics.dual['residual_lower_bound'],empirical_residual_bound(qp,fit.variables,fit.diagnostics.dual['multipliers'],1.))
        self.assertLessEqual(fit.diagnostics.dual['gap_bound'],fit.diagnostics.kkt['acceptance_tolerance'])

    def test_design_matches_direct_query_metric_both_directions(self):
        from prepared_learning.metrics import dense_metric
        rng=np.random.default_rng(42); a=Context(rng.normal(size=(3,3)),rng.normal(size=3),'a'); b=Context(rng.normal(size=(3,3)),rng.normal(size=3),'b'); task=SourceTask(a,b,'s')
        basis,_=np.linalg.qr(rng.normal(size=(3,3))); budget=BudgetSpec(2,1.5,8,1.)
        for branch in ('F','Z'):
            qp=build_empirical_qp([task],basis,budget,branch,0.)
            x=np.array([.5,.2])
            losses=[]
            for context,query in ((a,b),(b,a)):
                z=context.X.T@context.y/3
                if branch=='F': G=x[1]*np.eye(3)+(x[0]-x[1])*np.outer(basis[:,0],basis[:,0])
                else: G=(basis[:,:2]*x)@basis[:,:2].T
                losses.extend((query.X@G@z-query.y)**2)
            self.assertAlmostEqual(qp.constant+qp.linear@x+x@qp.hessian@x,np.mean(losses))

    def test_full_psd_qp_reaches_residual_tolerance(self):
        from pathlib import Path
        from prepared_learning.specs import load_config
        from prepared_learning.records import TaskKey
        from prepared_learning.tasks import generate_source_task
        from prepared_learning.priors import estimate_cross_moment
        c=load_config(Path('configs/exploration/learned_prior.json'))
        tasks=[generate_source_task(c,TaskKey(42,1,0,i,0)) for i in range(3)]
        prior=estimate_cross_moment(tasks,6); qp=build_empirical_qp(tasks,prior.basis,c.budget,'Z',0.)
        result=fit_empirical_gains(qp,c.budget,prior.basis,'Z')
        self.assertEqual(result.status,'completed')
        self.assertLessEqual(result.diagnostics.dual['gap_bound'],result.diagnostics.kkt['acceptance_tolerance'])
