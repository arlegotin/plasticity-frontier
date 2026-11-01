import unittest
import numpy as np
from prepared_learning.baselines import source_phase, solve_unrestricted_gamma

class SourceTests(unittest.TestCase):
    def test_phase_matches_independent_spectral_qp(self):
        for k,expected,width in [(.08,542/625,1),(.16,493/625,1),(.4,677/1025,3),(.8,573/1025,3),(8/9,5/9,2),(1.2,5/9,2)]:
            a,b,risk=source_phase(6,2,4,1.,k)
            self.assertAlmostEqual(risk,expected,places=10)
            g=solve_unrestricted_gamma(np.array([.5,.5,0,0,0,0]),4,1.,k)
            q=1.25*np.array([.5,.5,0,0,0,0])+.5
            self.assertAlmostEqual(1-2*np.dot(g,[.5,.5,0,0,0,0])+np.dot(q,g*g),expected,places=10)
            self.assertEqual(1 if a==b else (2 if b==0 else 3), width)
