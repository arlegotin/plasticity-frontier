import unittest
from prepared_learning.selection import public_method_config, select_method
from prepared_learning.records import BudgetSpec, PreparationSpec, RunConfig, SamplePlan

class SelectionTests(unittest.TestCase):
    def test_config_does_not_expose_truth(self):
        c=RunConfig(1,{'spectrum':[1.], 'sigma2':4.},BudgetSpec(2,1.,2,1.),PreparationSpec(1,1,'isotropic_gaussian','squared_error_over_2n',1.),{}, {},(),{'protected':'secret'},SamplePlan('deterministic_frontier',None,None,None,None,None),{})
        p=public_method_config(c)
        self.assertNotIn('workload',p)
        self.assertNotIn('keys',p)
        from dataclasses import replace
        changed=replace(c,workload={'spectrum':[.1,.9], 'sigma2':999.,'rotation':'secret'})
        self.assertEqual(public_method_config(changed),p)
        with self.assertRaises(ValueError): select_method([],[],[],dict(p,theta=[1]))

    def test_selection_fits_observed_tasks_and_records_labels(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(i):
            return SourceTask(Context(np.eye(2),np.array([1.,0.]),f'{i}a'),Context(np.eye(2),np.array([1.,0.]),f'{i}b'),str(i))
        config={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':6,'learning_rate':1.},'preparation':{'target_context_size':2,'input_dimension':2,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'gammas':[0.,1.],'regularization':[0.]},'information_policy':{'noise_access':'estimated'},'source_access_manifest':{'training_keys':['1','2'],'validation_keys':['3']}}
        result=select_method([make_method('B11:F',{}),make_method('B11:Z',{})],[t(1),t(2)],[t(3)],config)
        self.assertTrue(result.selected_id.startswith('B11:'))
        self.assertGreaterEqual(len(result.candidate_records),4)
        self.assertTrue(all(r['unique_labels']==12 for r in result.candidate_records))

    def test_fit_priors_cli_is_registered(self):
        import subprocess
        import sys
        result=subprocess.run([sys.executable,'-m','prepared_learning','fit-priors','--config','configs/exploration/learned_prior.json'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr+result.stdout)
        import json
        self.assertEqual(json.loads(result.stdout)['unique_labels'],40)

    def test_manifest_rejects_protected_and_role_mismatch(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(key):
            return SourceTask(Context(np.eye(2),np.ones(2),key+'a'),Context(np.eye(2),np.ones(2),key+'b'),key)
        config={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':6,'learning_rate':1.},'preparation':{'target_context_size':2,'input_dimension':2,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'gammas':[0.,1.]},'information_policy':{},'source_access_manifest':{'training_keys':['train'],'validation_keys':['validation']}}
        self.assertTrue(select_method([make_method('B01',{})],[t('train')],[t('validation')],config).selected_id)
        for train,val in [([t('protected')],[t('validation')]),([t('validation')],[t('train')]),([t('train'),t('train')],[t('validation')])]:
            with self.assertRaises(ValueError): select_method([make_method('B01',{})],train,val,config)
            with self.assertRaises(ValueError): make_method('B01',{}).fit_meta(train,val,config)

    def test_fz_receive_equal_assets_and_reconstructable_candidates(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(i):
            x=np.array([[1.,.2],[.3,1.]])
            return SourceTask(Context(x,np.array([1.,.5]),f'{i}a'),Context(x,np.array([.5,1.]),f'{i}b'),str(i))
        c={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':6,'learning_rate':1.},'preparation':{'target_context_size':2,'input_dimension':2,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'gammas':[0.,1.],'regularization':[0.], 'ranks':[1]},'information_policy':{},'source_access_manifest':{'training_keys':['1','2'],'validation_keys':['3','4']}}
        result=select_method([make_method('B11:F',{}),make_method('B11:Z',{})],[t(1),t(2)],[t(3),t(4)],c)
        f=[r for r in result.candidate_records if r['method_id']=='B11:F']; z=[r for r in result.candidate_records if r['method_id']=='B11:Z']
        self.assertEqual({(r['asset_hash'],r['regularization']) for r in f},{(r['asset_hash'],r['regularization']) for r in z})
        self.assertTrue(all('options' in r and 'asset' in r for r in f+z))
        self.assertTrue(any(r['asset']['fit_metadata']['estimator']=='moment_directions' for r in f))

    def test_mechanism_selector_rejects_operational_candidates(self):
        from prepared_learning.methods import make_method
        from prepared_learning.records import Context, SourceTask
        import numpy as np
        def t(key): return SourceTask(Context(np.eye(1),np.ones(1),key+'a'),Context(np.eye(1),np.ones(1),key+'b'),key)
        config={'source_access_manifest':{'training_keys':['a'],'validation_keys':['b']},'selection_class':'mechanism_baseline'}
        with self.assertRaisesRegex(ValueError,'ineligible'): select_method([make_method('B08:ridge',{})],[t('a')],[t('b')],config)

    def test_selected_options_are_recorded_without_mutating_method(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(k): return SourceTask(Context(np.eye(2),np.ones(2),k+'a'),Context(np.eye(2),np.ones(2),k+'b'),k)
        config={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':6,'learning_rate':1.},'preparation':{'target_context_size':2,'input_dimension':2,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'gammas':[0.,1.],'ranks':[1]},'information_policy':{},'source_access_manifest':{'training_keys':['a'],'validation_keys':['b']}}
        m=make_method('B01',{}); result=m.fit_meta([t('a')],[t('b')],config)
        self.assertEqual(m.options,{})
        row=next(r for r in result.candidate_records if r['candidate_id']==result.selected_id)
        self.assertIn('fitted_metric',row['options'])
        self.assertEqual(row['asset_hash'],result.asset.asset_hash)
        from prepared_learning.records import BudgetSpec, PreparationSpec
        restored=make_method(row['method_id'],row['options'])
        restored.prepare(result.asset,BudgetSpec(**config['budget']),PreparationSpec(**config['preparation']))

    def test_shrinkage_refines_unresolved_paired_validation(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(k,y): return SourceTask(Context(np.eye(2),np.array(y),k+'a'),Context(np.eye(2),np.array(y),k+'b'),k)
        c={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':6,'learning_rate':1.},'preparation':{'target_context_size':2,'input_dimension':2,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'gammas':[0.,1.],'ranks':[1],'max_refinement_rounds':2},'information_policy':{},'source_access_manifest':{'training_keys':['a'],'validation_keys':['b','c']}}
        r=select_method([make_method('B02',{})],[t('a',[1.,0.])],[t('b',[1.,0.]),t('c',[1.,0.])],c)
        self.assertIn(.5,{row['gamma'] for row in r.candidate_records if row['status']=='completed'})
        self.assertIn('tuning_resolution',r.candidate_records[0])

    def test_projected_ridge_tunes_representation_rank(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(k): return SourceTask(Context(np.eye(2),np.ones(2),k+'a'),Context(np.eye(2),np.ones(2),k+'b'),k)
        c={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':6,'learning_rate':1.},'preparation':{'target_context_size':2,'input_dimension':2,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'gammas':[0.,1.],'ranks':[1,2]},'information_policy':{},'source_access_manifest':{'training_keys':['a'],'validation_keys':['b']}}
        result=make_method('B09:ridge',{}).fit_meta([t('a')],[t('b')],c)
        self.assertEqual({r['options']['rank'] for r in result.candidate_records if r['status']=='completed'},{1,2})

    def test_nested_generator_truth_is_rejected(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(k): return SourceTask(Context(np.eye(1),np.ones(1),k+'a'),Context(np.eye(1),np.ones(1),k+'b'),k)
        c={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':2,'learning_rate':1.},'preparation':{'target_context_size':1,'input_dimension':1,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'theta':[9.]},'information_policy':{},'source_access_manifest':{'training_keys':['a'],'validation_keys':['b']}}
        with self.assertRaises(ValueError): make_method('B01',{}).fit_meta([t('a')],[t('b')],c)

    def test_fit_meta_rejects_injected_metric(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(k): return SourceTask(Context(np.eye(1),np.ones(1),k+'a'),Context(np.eye(1),np.ones(1),k+'b'),k)
        c={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':2,'learning_rate':1.},'preparation':{'target_context_size':1,'input_dimension':1,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{},'information_policy':{},'source_access_manifest':{'training_keys':['a'],'validation_keys':['b']}}
        with self.assertRaises(ValueError): make_method('B01',{'fitted_metric':'oracle'}).fit_meta([t('a')],[t('b')],c)

    def test_known_noise_is_explicit_and_shared(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(k): return SourceTask(Context(np.eye(2),np.ones(2),k+'a'),Context(np.eye(2),np.ones(2),k+'b'),k)
        c={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':6,'learning_rate':1.},'preparation':{'target_context_size':2,'input_dimension':2,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'gammas':[0.,1.],'ranks':[1],'regularization':[0.]},'information_policy':{'noise_access':'known','permitted_noise_variance':3.},'source_access_manifest':{'training_keys':['a'],'validation_keys':['b']}}
        result=select_method([make_method('B11:F',{}),make_method('B11:Z',{})],[t('a')],[t('b')],c)
        for r in result.candidate_records:
            self.assertEqual(r['asset']['noise_variance'],3.)
            self.assertEqual(r['asset']['fit_metadata']['noise_access'],'known')

    def test_ridge_finite_boundary_is_extended(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(k,a,b): return SourceTask(Context(np.ones((1,1)),np.array([a]),k+'a'),Context(np.ones((1,1)),np.array([b]),k+'b'),k)
        c={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':2,'learning_rate':1.},'preparation':{'target_context_size':1,'input_dimension':1,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'gammas':[0.,1.],'ranks':[1],'ridge_multipliers':[0.,5.],'max_refinement_rounds':4},'information_policy':{},'source_access_manifest':{'training_keys':['a'],'validation_keys':['b','c']}}
        result=select_method([make_method('B08:ridge',{})],[t('a',1.,1.)],[t('b',1.,.05),t('c',1.,.05)],c)
        self.assertGreater(max(r['tuning_value'] for r in result.candidate_records if r.get('tuning_kind')=='penalty'),5.)

    def test_ridge_refines_around_finite_minimum(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(k,a,b): return SourceTask(Context(np.ones((1,1)),np.array([a]),k+'a'),Context(np.ones((1,1)),np.array([b]),k+'b'),k)
        c={'budget':{'depth':2,'initialization_norm_cap':1.,'trainable_parameter_cap':2,'learning_rate':1.},'preparation':{'target_context_size':1,'input_dimension':1,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'gammas':[0.,1.],'ranks':[1],'ridge_multipliers':[0.,1.,10.,100.],'max_refinement_rounds':1},'information_policy':{},'source_access_manifest':{'training_keys':['a'],'validation_keys':['b','c']}}
        result=select_method([make_method('B08:ridge',{})],[t('a',1.,1.)],[t('b',1.,.05),t('c',1.,.05)],c)
        values={r['tuning_value'] for r in result.candidate_records if r.get('tuning_kind')=='penalty'}
        self.assertGreater(len(values),4)

    def test_failed_candidates_remain_in_selection_ledger(self):
        import numpy as np
        from prepared_learning.records import Context, SourceTask
        from prepared_learning.methods import make_method
        def t(k): return SourceTask(Context(np.eye(1),np.ones(1),k+'a'),Context(np.eye(1),np.ones(1),k+'b'),k)
        c={'budget':{'depth':3,'initialization_norm_cap':1.,'trainable_parameter_cap':3,'learning_rate':1.},'preparation':{'target_context_size':1,'input_dimension':1,'input_law':'isotropic_gaussian','loss_normalization':'squared_error_over_2n','task_second_moment_trace':1.},'fitting_candidates':{'gammas':[0.,1.],'ranks':[1]},'information_policy':{},'source_access_manifest':{'training_keys':['a'],'validation_keys':['b']}}
        result=select_method([make_method('B02',{})],[t('a')],[t('b')],c)
        self.assertEqual(result.selected_id,'')
        self.assertTrue(result.candidate_records)
        self.assertTrue(all(r['status']=='failed' for r in result.candidate_records))
