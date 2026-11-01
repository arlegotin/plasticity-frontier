import unittest
from unittest.mock import patch
import numpy as np
from prepared_learning.records import BudgetSpec, PreparationSpec, PriorAsset
from prepared_learning.metrics import metric_resources
from prepared_learning.baselines import solve_unrestricted_gamma

S=np.array([.5,.25,.15,.07,.02,.01])
def fixture(s=S,k=.6,m=3):
 d=len(s); return (PriorAsset(np.diag(s),np.eye(d),s,1.,'oracle',{},'fixture'), BudgetSpec(2,k,(d+1)*m,1.), PreparationSpec(4,d,'gaussian','1/(2n)',1.))

class FrontierTests(unittest.TestCase):
 def solve(self,*args):
  from prepared_learning.frontier import solve_frontier
  return solve_frontier(*fixture(*args))
 def test_primary_float_frontier(self):
  r=self.solve()
  self.assertAlmostEqual(r.F.objective,186321/276200,delta=1e-10); self.assertAlmostEqual(r.Z.objective,81247/115000,delta=1e-10)
  for branch in (r.F,r.Z):
   resources=metric_resources(branch.metric,2); self.assertEqual(resources.trainable_parameters,21); self.assertAlmostEqual(resources.initialization_norm,.6)
   self.assertEqual(branch.diagnostics.roundoff['certificate_level'],'float64_diagnostic')
 def test_rank_family_reversal(self):
  r=self.solve(S,1.,3); self.assertEqual(r.selected_branches,('Z',)); self.assertAlmostEqual(r.F.objective,.673827673828,delta=1e-10); self.assertAlmostEqual(r.Z.objective,.668127428127,delta=1e-10)
 def test_zero_infeasible_scalar_full_width(self):
  self.assertEqual(self.solve(S,0.,3).F.objective,1.)
  with self.assertRaises(ValueError): self.solve(S,0.,0)
  self.assertAlmostEqual(self.solve(np.array([1.]),.6,1).F.objective,self.solve(np.array([1.]),.6,1).Z.objective)
  self.assertAlmostEqual(self.solve(S,.6,1).F.metric.floor,min(.6,1/4.25))
  r=self.solve(S,1.,6); g=solve_unrestricted_gamma(S,4,1.,1.); q=1.25*S+.5
  self.assertAlmostEqual(r.F.objective,1-2*S@g+q@(g*g),delta=1e-10)
  self.assertGreater(r.Z.metric.floor,0)
 def test_caps_are_monotone(self):
  values=[min(self.solve(S,k,m).F.objective,self.solve(S,k,m).Z.objective) for k,m in ((.4,2),(.6,2),(.6,3))]
  self.assertGreaterEqual(values[0],values[1]); self.assertGreaterEqual(values[1],values[2])
 def test_extreme_budget_scaling(self):
  for k in (1e-14,.6,1e6):
   r=self.solve(S,k,3)
   self.assertGreater(np.max(r.F.variables),0); self.assertGreater(np.max(r.Z.variables),0)
   self.assertLessEqual(metric_resources(r.F.metric,2).initialization_norm,k*(1+1e-12))
 def test_tie_and_failed_solver_retained(self):
  from prepared_learning.frontier import solve_frontier
  from prepared_learning.quadratic import solve_qp
  def fail(qp,scale):
   x,diag=solve_qp(qp,scale)
   from dataclasses import replace
   return x,replace(diag,kkt={**diag.kkt,'solver_success':False})
  with patch('prepared_learning.frontier.solve_qp',side_effect=fail): r=solve_frontier(*fixture())
  self.assertEqual(r.selection_status,'unresolved'); self.assertIsNotNone(r.F); self.assertIsNotNone(r.Z)
 def test_methods_compile_canonical_frontier_and_direct_map(self):
  from prepared_learning.methods import make_method
  from prepared_learning.records import Context
  prior,budget,task=fixture(); context=Context(np.arange(24,dtype=float).reshape(4,6)/20,np.arange(4,dtype=float),'case')
  for name in ('B03','B04'):
   method=make_method(name,{}); prepared=method.prepare(prior,budget,task)
   self.assertEqual(prepared.resources.trainable_parameters,21)
   direct=make_method('B05:'+name,{'metric':prepared.metric}); dp=direct.prepare(prior,budget,task)
   np.testing.assert_allclose(method.adapt(prepared,context).coefficient,direct.adapt(dp,context).coefficient,atol=1e-12)
