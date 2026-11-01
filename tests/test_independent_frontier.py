import unittest
import numpy as np
from tests.test_frontier import fixture, S

class IndependentFrontier(unittest.TestCase):
 def test_tiny_supports_and_rotations(self):
  from checks.independent_reference import enumerate_support_frontier
  from prepared_learning.frontier import solve_frontier
  result=enumerate_support_frontier(S,4,1.,.6,3)
  f=solve_frontier(*fixture())
  self.assertAlmostEqual(result['F']['risk'],f.F.objective,delta=1e-8)
  self.assertAlmostEqual(result['Z']['risk'],f.Z.objective,delta=1e-8)
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

 def test_source_grid(self):
  from prepared_learning.frontier import evaluate_frontier_grid
  from prepared_learning.baselines import source_phase
  s=np.array([.5,.5,0.,0.,0.,0.]); prior,budget,task=fixture(s,.4,3)
  rows=evaluate_frontier_grid(prior,budget,task,[3],[.4,.8],.001,.02)
  point=next(r for r in rows if r['kind']=='point' and r['norm_cap']==.4)
  self.assertAlmostEqual(point['F']['objective'],source_phase(6,2,4,1.,.4)[2],delta=1e-10)
