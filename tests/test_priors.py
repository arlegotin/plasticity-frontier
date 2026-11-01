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

    def test_rotated_moment_order_survives_shrinkage(self):
        from prepared_learning.priors import estimate_moment_directions
        X=np.array([[2.,1.],[1.,2.]])
        a=Context(X,np.array([2.,1.]),'a'); b=Context(X,np.array([2.,1.]),'b')
        task=SourceTask(a,b,'s'); moment=(X.T*np.array([4.,1.]))@X
        p=estimate_moment_directions([task],2)
        self.assertTrue(np.array_equal(p.eigenvalues,np.array([.5,.5])))
        for gamma in (0.,.4,1.):
            q=shrink_prior(p,gamma)
            projected=q.basis.T@moment@q.basis
            self.assertAlmostEqual(projected[0,1],0.,places=10)
            self.assertGreater(projected[0,0],projected[1,1])
            self.assertEqual(q.fit_metadata['variance_interpretation'],'equal-variance surrogate on selected directions')
            if gamma==1.: self.assertTrue(np.array_equal(q.eigenvalues,np.array([.5,.5])))

    def test_isotropic_shrink_preserves_sorted_exact_surrogate_values(self):
        from prepared_learning.priors import _asset
        rng=np.random.default_rng(123); q,_=np.linalg.qr(rng.normal(size=(6,6)))
        p=_asset(q@np.diag([.4,.25,.15,.1,.06,.04])@q.T,None,{})
        shrunk=shrink_prior(p,1.)
        self.assertTrue(np.array_equal(shrunk.eigenvalues,np.full(6,1/6)))
        np.testing.assert_array_equal(shrunk.basis,p.basis)
