import unittest
import numpy as np
from prepared_learning.records import Context, SourceTask
from prepared_learning.priors import estimate_cross_moment, estimate_noise, shrink_prior

class PriorTests(unittest.TestCase):
    def test_cross_moment_projection_and_fallback(self):
        def task(a,b):
            return SourceTask(Context(np.eye(2),2*np.array(a), 'a'), Context(np.eye(2),2*np.array(b),'b'),'s')
        p=estimate_cross_moment([task([1,0],[-1,0])],2)
        np.testing.assert_allclose(p.covariance,np.eye(2)/2)
        self.assertEqual(p.fit_metadata['fallback'],'isotropic')
        self.assertAlmostEqual(p.fit_metadata['removed_negative_mass'],1.)
        q=estimate_cross_moment([task([1,0],[1,0])],2)
        np.testing.assert_allclose(shrink_prior(q,1).covariance,np.eye(2)/2)
        self.assertIn('task_standard_error',estimate_noise([task([1,0],[1,0])]))

    def test_cross_moment_exact_repair_and_direction_comparator(self):
        from prepared_learning.priors import estimate_moment_directions
        def task(a,b,k):
            return SourceTask(Context(np.eye(2),2*np.array(a),k+'a'),Context(np.eye(2),2*np.array(b),k+'b'),k)
        tasks=[task([2,0],[1,0],'1'),task([0,1],[0,-1],'2')]
        p=estimate_cross_moment(tasks,2)
        np.testing.assert_allclose(p.fit_metadata['raw'],np.diag([1.,-.5]))
        np.testing.assert_allclose(p.covariance,np.diag([1.,0.]))
        self.assertAlmostEqual(p.fit_metadata['removed_negative_mass'],.5)
        m=estimate_moment_directions(tasks,1)
        np.testing.assert_allclose(m.covariance,np.diag([1.,0.]))
        self.assertFalse(np.allclose(m.eigenvalues,m.fit_metadata['moment_eigenvalues']))

    def test_reused_observation_block_is_rejected(self):
        a=Context(np.eye(2),np.ones(2),'same')
        with self.assertRaises(ValueError): estimate_cross_moment([SourceTask(a,a,'one')],2)
