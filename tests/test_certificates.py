import unittest
import numpy as np
from fractions import Fraction as F
from checks import exact_reference as ex
from prepared_learning.records import QPData

class Certificates(unittest.TestCase):
 def test_primary_exact_certificates(self):
  case=ex.load_exact_cases(__import__('pathlib').Path('tests/fixtures/exact_cases.json'))['primary_richer_k_3_5']
  for branch in ('F','Z'):
   result=ex.verify_rational_kkt(ex.primary_rational_qp(case,branch), (3,))
   self.assertTrue(result['verified']); self.assertEqual(F(*result['primal']),ex.frac(case[branch]['risk']))
   self.assertEqual(F(*result['dual']),ex.frac(case[branch]['risk']))
   self.assertEqual(F(*result['multipliers'][3]),ex.frac(case[branch]['multiplier']))
 def test_false_active_set_and_multiplier_rejected(self):
  case=ex.load_exact_cases(__import__('pathlib').Path('tests/fixtures/exact_cases.json'))['primary_richer_k_3_5']
  qp=ex.primary_rational_qp(case,'F')
  self.assertFalse(ex.verify_rational_kkt(qp,(0,))['verified'])
 def test_singular_empirical_bound(self):
  from prepared_learning.certificates import empirical_residual_bound
  qp=QPData(1.,np.array([-2.,-2.]),np.ones((2,2)),np.array([[1.,0.],[0.,1.],[-1.,-1.]]),np.array([0.,0.,1.]))
  self.assertAlmostEqual(empirical_residual_bound(qp,np.array([.3,.4]),np.zeros(3),1.), 0.09-1.2)
  box=QPData(0.,np.array([-1.,-1.]),np.zeros((2,2)),np.eye(2),np.zeros(2))
  self.assertLessEqual(empirical_residual_bound(box,np.zeros(2),np.zeros(2),1.),-2.)

 def test_cli_exact_provenance_and_float_default(self):
  import subprocess,json
  cmd=['.venv/bin/python','-m','prepared_learning','frontier','--config','configs/exploration/frontier.json']
  plain=json.loads(subprocess.check_output(cmd,text=True))
  self.assertEqual(plain['F']['diagnostics']['roundoff']['certificate_level'],'float64_diagnostic')
  exact=json.loads(subprocess.check_output(cmd+['--certify'],text=True))
  self.assertTrue(exact['exact_certificates']['F']['verified'])
  self.assertTrue(exact['exact_certificates']['Z']['verified'])
