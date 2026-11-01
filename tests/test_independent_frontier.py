import unittest
import numpy as np
from tests.test_frontier import fixture, S

class IndependentFrontier(unittest.TestCase):
 def test_tiny_supports_and_rotations(self):
  from checks.independent_reference import enumerate_support_frontier
  from prepared_learning.frontier import solve_frontier
  result=enumerate_support_frontier(S,4,1.,.6,3)
  f=solve_frontier(*fixture())
  for branch in ('F','Z'):
   candidate=result[branch]; production=getattr(f,branch)
   self.assertIn('parameters',candidate); self.assertIn('basis',candidate)
   self.assertGreaterEqual(candidate['error_bound'],0.)
   self.assertLessEqual(abs(candidate['risk']-production.objective),candidate['error_bound']+abs(production.diagnostics.dual['gap'])+candidate['roundoff_estimate'])
   for alternative in (c for c in result['candidates'] if c['branch']==branch and c['feasible']):
    self.assertGreaterEqual(alternative['risk']+alternative['error_bound']+alternative['roundoff_estimate'],production.objective-abs(production.diagnostics.dual['gap']))
  with self.assertRaises(ValueError): enumerate_support_frontier(np.ones(7)/7,4,1.,.6,3)
 def test_grid_reversal(self):
  from prepared_learning.frontier import evaluate_frontier_grid
  prior,budget,task=fixture()
  rows=evaluate_frontier_grid(prior,budget,task,[3],[.6,1.],1e-3,.02)
  bycap={r['norm_cap']:r for r in rows if r['kind']=='point'}
  self.assertEqual(bycap[.6]['selection_status'],'selected')
  self.assertEqual(bycap[.6]['selected_branches'],['F'])
  self.assertEqual(bycap[1.]['selected_branches'],['Z'])

 def test_cli_grid_writes_raw_files(self):
  import tempfile,json,subprocess
  from pathlib import Path
  with tempfile.TemporaryDirectory() as tmp:
   cfg=json.loads(Path('configs/exploration/frontier.json').read_text())
   cfg['grid']={'widths':[3],'norm_caps':[.6,1.], 'risk_tolerance':.001,'budget_tolerance':.02,'output':tmp}
   path=Path(tmp)/'config.json'; path.write_text(json.dumps(cfg))
   result=json.loads(subprocess.check_output(['.venv/bin/python','-m','prepared_learning','frontier','--config',str(path)],text=True))
   attempt=Path(result['attempt_path']); self.assertTrue((attempt/'frontier.csv').exists()); self.assertTrue((attempt/'frontier.json').exists())

 def test_full_rank_z_exact_count(self):
  from prepared_learning.metrics import CompactMetric
  from prepared_learning.records import SupportMetadata
  from prepared_learning.geometry import construct_canonical,construct_head_zero
  m=CompactMetric(0.,np.eye(2),np.array([1.,2.]),SupportMetadata('analytic',0,0,(0,1)))
  canonical=construct_canonical(m,2); head=construct_head_zero(m)
  self.assertEqual(canonical.resources.trainable_parameters,6)
  self.assertEqual(head.resources.trainable_parameters,6)
  self.assertAlmostEqual(canonical.resources.initialization_norm,2.)
  self.assertAlmostEqual(head.resources.initialization_norm,3.)

 def test_infeasible_candidate_is_not_a_counterexample(self):
  from checks.independent_reference import best_feasible_candidate
  result=best_feasible_candidate([{'risk':-10.,'feasible':False},{'risk':.5,'feasible':True}])
  self.assertEqual(result['risk'],.5)
  from unittest.mock import patch
  from scipy.optimize import minimize, OptimizeResult
  from checks.independent_reference import enumerate_support_frontier
  calls=[0]
  def inject(*args,**kwargs):
   calls[0]+=1
   if calls[0]==1: return OptimizeResult(x=np.full(len(args[1]),10.),success=True,message='infeasible alternative')
   return minimize(*args,**kwargs)
  with patch('scipy.optimize.minimize',side_effect=inject):
   search=enumerate_support_frontier(np.array([.5,.3,.2]),4,1.,.6,2)
  bad=search['candidates'][0]
  self.assertFalse(bad['feasible']); self.assertEqual(bad['raw_parameters'],[10.,10.])
  self.assertNotEqual(search['F'],bad)

 def test_source_grid(self):
  from prepared_learning.frontier import evaluate_frontier_grid
  from prepared_learning.baselines import source_phase
  s=np.array([.5,.5,0.,0.,0.,0.]); prior,budget,task=fixture(s,.4,3)
  rows=evaluate_frontier_grid(prior,budget,task,[3],[.4,.8],.001,.02)
  point=next(r for r in rows if r['kind']=='point' and r['norm_cap']==.4)
  self.assertAlmostEqual(point['F']['objective'],source_phase(6,2,4,1.,.4)[2],delta=1e-10)

 def test_grid_failure_is_preserved_with_artifacts(self):
  import tempfile,json,subprocess
  from pathlib import Path
  with tempfile.TemporaryDirectory() as tmp:
   cfg=json.loads(Path('configs/exploration/frontier.json').read_text()); cfg['depth']=3
   cfg['grid']={'widths':[3],'norm_caps':[.6,1e308], 'risk_tolerance':.001,'budget_tolerance':.02,'output':tmp}
   path=Path(tmp)/'config.json'; path.write_text(json.dumps(cfg))
   run=subprocess.run(['.venv/bin/python','-m','prepared_learning','frontier','--config',str(path)],capture_output=True,text=True)
   self.assertNotEqual(run.returncode,0); result=json.loads(run.stdout); self.assertEqual(result['status'],'incomplete')
   attempt=Path(result['attempt_path']); data=json.loads((attempt/'frontier.json').read_text())
   self.assertEqual(data['metadata']['risk_tolerance'],.001)
   self.assertTrue(any(r['kind']=='point' and r['status']=='completed' for r in data['records']))
   self.assertTrue(any(r['kind']=='point' and r['status']=='failed' for r in data['records']))
   self.assertTrue(any(r['kind']=='interval' for r in data['records']))
   self.assertIn('failed',(attempt/'frontier.csv').read_text())

 def test_grid_tolerances_finite(self):
  from prepared_learning.frontier import evaluate_frontier_grid
  for bad in (float('nan'),float('inf'),0.,-1.):
   for a,b in ((bad,.1),(.1,bad)):
    with self.assertRaises(ValueError): evaluate_frontier_grid(*fixture(),[3],[.6,1.],a,b)
