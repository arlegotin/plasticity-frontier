"""Tiny execution fixtures; none are production G6 or protected evidence."""
import importlib.util
import unittest
from dataclasses import replace
import numpy as np
from prepared_learning.records import TaskKey
from tests import test_pilot

class PopulationContractTests(unittest.TestCase):
    def test_population_fixed_across_protected_roots(self):
        from prepared_learning.tasks import population_for, generate_task
        c=test_pilot.PilotIntegrationTests.config()
        c=replace(c,workload={**c.workload,'population_seed':42})
        a,aa=population_for(c,TaskKey(42,10,0,0,0))
        b,bb=population_for(c,TaskKey(900,101,0,0,0))
        np.testing.assert_array_equal(a.second_moment,b.second_moment)
        self.assertEqual(aa['artifact_hash'],bb['artifact_hash'])
        x,_=generate_task(c,TaskKey(42,10,0,0,0)); y,_=generate_task(c,TaskKey(900,101,0,0,0))
        self.assertFalse(np.array_equal(x.X,y.X))

    def test_canonical_law_and_legacy_alias(self):
        from prepared_learning.specs import validate_config
        c=test_pilot.PilotIntegrationTests.config()
        self.assertEqual(c.preparation.input_law,'gaussian_isotropic')
        validate_config(replace(c,preparation=replace(c.preparation,input_law='gaussian_isotropic')),False)
        with self.assertRaises(ValueError): validate_config(replace(c,preparation=replace(c.preparation,input_law='unsupported')),False)

class ConfirmationTests(unittest.TestCase):
    def test_driver_api(self):
        self.assertIsNotNone(importlib.util.find_spec('prepared_learning.confirmation'),'frozen execution driver missing')


def tiny(mode='conditional_risk_over_preparations'):
    from prepared_learning.records import to_json_record
    from prepared_learning.tasks import resolve_population, population_identity
    c=test_pilot.PilotIntegrationTests.config()
    learned=mode in ('conditional_risk_over_preparations','nested_task_simulation')
    c=replace(c,workload={'spectrum':[.7,.3],'sigma2':.2,'population_seed':42},
        preparation=replace(c.preparation,input_dimension=2,target_context_size=3),
        budget=replace(c.budget,initialization_norm_cap=.6,trainable_parameter_cap=6),
        methods=('B01','B02') if learned else ('B01','B02','B04','B05:B04','B08:ridge'),
        access={'prior':'estimated','selection':'learned_only'} if learned else {'prior':'oracle','selection':'fixed'},
        sample_plan=replace(c.sample_plan,analysis_mode=mode,outer_replicates=2 if learned else None,
            tasks_per_replicate=3 if 'task_simulation' in mode else None,
            source_training_tasks=3 if learned else None,source_validation_tasks=2 if learned else None,source_block_size=3 if learned else None))
    return c,{'config':to_json_record(c),'selectors':{'baseline':['B01'],'joint':['B02']},
        'update_check_schedule':{'outer_replications':[0],'tasks_per_replication':1},
        'inference':{'procedure':'descriptive_only','alpha':.05,'coverage':None,'moment_basis':'unverified'},
        'assets':{'population':population_identity(c)},'secondary_family':{'contrasts':[],'adjustment':'holm'}}


class ExecutionTests(unittest.TestCase):
    def execute(self,mode):
        from prepared_learning.execution import execute_replication
        c,s=tiny(mode)
        results=[execute_replication(c,s,300,j,{'training_phase':101,'test_phase':103})[0] for j in range(c.sample_plan.outer_replicates or 1)]
        return c,s,results

    def test_all_four_modes_have_endpoint_correct_counts(self):
        from prepared_learning.confirmation import analyze_results
        for mode in ('deterministic_frontier','fixed_prior_task_simulation','nested_task_simulation','conditional_risk_over_preparations'):
            with self.subTest(mode=mode):
                c,s,results=self.execute(mode)
                self.assertTrue(all(not r['failures'] for r in results))
                analysis=analyze_results(results,s,300)
                self.assertEqual(analysis['update_check_count'],1)
                self.assertEqual(analysis['n'],None if mode=='deterministic_frontier' else (3 if mode=='fixed_prior_task_simulation' else 2))
                self.assertEqual(analysis['primary_task_count'],6 if mode=='nested_task_simulation' else (3 if mode=='fixed_prior_task_simulation' else 0))
                self.assertIsNone(analysis['interval'])

    def test_validation_choice_frozen_before_evaluation(self):
        from prepared_learning.execution import execute_replication
        from unittest.mock import patch
        c,s=tiny(); phases={'training_phase':101,'test_phase':103}
        a,_=execute_replication(c,s,300,0,phases)
        with patch('prepared_learning.evaluation.coefficient_loss',return_value=1e8):
            b,_=execute_replication(c,s,300,0,phases)
        self.assertEqual({k:v['candidate_id'] for k,v in a['selected'].items()}, {k:v['candidate_id'] for k,v in b['selected'].items()})
        self.assertEqual(a['source_keys'],b['source_keys'])
        self.assertEqual(len(a['source_keys']),5)
        self.assertTrue(all(k.startswith(('candidate:','family:')) for k in a['risks'] if k not in ('baseline','joint')))

    def test_reused_initializer_caught_in_driver(self):
        from prepared_learning.execution import execute_replication
        from prepared_learning.methods import Method
        from unittest.mock import patch
        original=Method.adapt
        def corrupt(self,state,context):
            value=original(self,state,context)
            object.__setattr__(state,'method_id','mutated')
            return value
        c,s=tiny('fixed_prior_task_simulation')
        with patch.object(Method,'adapt',corrupt):
            result,_=execute_replication(c,s,300,0,{'training_phase':101,'test_phase':103})
        self.assertEqual(result['status'],'invalid')
        self.assertTrue(result['failures'])

    def test_duplicate_network_direct_pair(self):
        _,_,results=self.execute('fixed_prior_task_simulation')
        for task in results[0]['tasks']:
            a=task['values']['B04']; b=task['values']['B05:B04']
            self.assertAlmostEqual(a['loss'],b['loss'],places=12)
            self.assertLess(a['diagnostics']['duplicate_coefficient_error'],1e-12)

    def test_duplicate_network_direct_pair_never_becomes_superiority_test(self):
        from prepared_learning.execution import execute_replication
        from prepared_learning.confirmation import analyze_results
        c,s=tiny('fixed_prior_task_simulation')
        s['selectors']={'baseline':['B04'],'joint':['B05:B04']}
        s['inference'].update(procedure='student',coverage='approximate')
        result,_=execute_replication(c,s,300,0,{'training_phase':101,'test_phase':103})
        analysis=analyze_results([result],s,300)
        self.assertIsNone(analysis['interval'])
        self.assertEqual(analysis['status'],'arithmetic_equality_diagnostic')

    def test_incomplete_schedule_cannot_be_analyzed(self):
        from prepared_learning.confirmation import analyze_results
        _,s,results=self.execute('nested_task_simulation')
        results[0]['tasks'].pop()
        with self.assertRaisesRegex(ValueError,'schedule'): analyze_results(results,s,300)

    def test_moment_policy_keeps_zero_penalty_controls(self):
        import prepared_learning.protocol as protocol
        self.assertTrue(callable(getattr(protocol,'moment_policy',None)),'per-estimator moment policy missing')
        c,_=tiny('fixed_prior_task_simulation')
        c=replace(c,preparation=replace(c.preparation,target_context_size=2),methods=('B08:ridge',))
        policy=protocol.moment_policy(c)
        self.assertFalse(policy['mean_ci_eligible'])
        self.assertEqual(policy['estimators']['B08:ridge']['status'],'unsupported_or_unverified')
        c=replace(c,numerics={**c.numerics,'method_options':{'B08:ridge':{'penalty':.1}}})
        self.assertTrue(protocol.moment_policy(c)['mean_ci_eligible'])

class ModeFreezeTests(unittest.TestCase):
    def test_real_descriptive_mode_evidence_freezes_and_executes(self):
        import tempfile, json
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import pilot, protocol, confirmation
        for mode in ('deterministic_frontier','fixed_prior_task_simulation','nested_task_simulation'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                c,_=tiny(mode)
                if mode=='deterministic_frontier': c=replace(c,methods=('B01','B02','B04','B05:B04'))
                settings={'blocks':2,'replicates_per_block':3,'procedure':'descriptive_only','max_seconds':1e6}
                c=replace(c,numerics={**c.numerics,'pilot':settings})
                actual=protocol.checkout_identity
                # Only emulate a committed checkout in a temporary integration fixture;
                # all scientific observations, runtime and file hashes are real.
                def clean(*args,**kwargs): return actual(*args,**kwargs)[0],None
                with patch.object(protocol,'checkout_identity',side_effect=clean):
                    evidence=pilot.run_pilot(c,Path(root)/'pilot')
                self.assertEqual(evidence['gate']['reasons'],[])
                record=json.loads((Path(evidence['run_path'])/'protocol_draft.json').read_text())
                frozen=protocol.freeze_protocol(record['draft'],evidence,Path(root)/'fixture.json')
                with patch.object(confirmation,'_live_identity',side_effect=lambda manifest,*args:protocol.validate_frozen_protocol(manifest)):
                    gate=confirmation.run_confirmation(frozen,Path(root)/'confirm')
                    self.assertEqual(gate.code,'INSENSITIVE_OR_UNSUITABLE')
                    analysis=confirmation.analyze_run(Path(gate.evidence['run_path']),frozen)
                    self.assertIsNone(analysis['interval'])
                    replay=confirmation.run_confirmation(frozen,Path(root)/'confirm')
                    self.assertEqual(replay.code,gate.code)
                    from contextlib import redirect_stdout
                    import io
                    from prepared_learning.cli import main
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(main(['confirm','--manifest',str(Path(root)/'fixture.json'),'--output',str(Path(root)/'confirm')]),0)
                        self.assertEqual(main(['analyze','--manifest',str(Path(root)/'fixture.json'),'--output',str(Path(root)/'confirm')]),0)

    def test_live_driver_rejects_uncommitted_or_changed_identity(self):
        import tempfile
        from pathlib import Path
        from prepared_learning.confirmation import run_confirmation
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError): run_confirmation({'status':'frozen'},Path(root))
            self.assertFalse(list(Path(root).iterdir()))

class SecondaryTests(unittest.TestCase):
    def test_update_checks_and_operational_simulations_have_separate_keys(self):
        from prepared_learning.execution import execute_replication
        from prepared_learning.confirmation import analyze_results
        c,s=tiny()
        c=replace(c,methods=(*c.methods,'B08:ridge'),numerics={**c.numerics,'fitting_candidates':{**c.numerics['fitting_candidates'],'ridge_multipliers':[0.,1.,10.]}})
        s['config']=__import__('prepared_learning.records',fromlist=['to_json_record']).to_json_record(c)
        s['simulation_schedule']={'outer_replications':[0,1],'tasks_per_replication':2,'method_ids':['B08:ridge'],'inference':'descriptive_only'}
        s['secondary_family']={'contrasts':['B08:ridge'],'adjustment':'holm'}
        results=[execute_replication(c,s,300,j,{'training_phase':101,'test_phase':103})[0] for j in range(2)]
        analysis=analyze_results(results,s,300)
        self.assertEqual(analysis['n'],2)
        self.assertEqual(analysis['primary_task_count'],0)
        self.assertEqual(analysis['secondary_task_count'],4)
        keys=[t['task_key'] for r in results for t in r['tasks']]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertIn('secondary',analysis)
        self.assertIsNone(analysis['secondary']['B08:ridge']['interval'])


def execution_manifest(mode='conditional_risk_over_preparations'):
    """Detached declared schedule for driver tests, deliberately no G6 evidence."""
    from prepared_learning.protocol import checkout_identity
    from prepared_learning.records import content_hash
    from tests.test_protocol import runtime_fixture
    c,s=tiny(mode); versions,_=checkout_identity(s['selectors'],s['config']['numerics'].get('fitting_candidates',{}),runtime=runtime_fixture())
    s['versions']=versions
    op={'reserve_memory_fraction':.5,'thread_policy':runtime_fixture()['thread_environment'],'blas_policy':content_hash(runtime_fixture()['builds'])}
    return {'scientific':s,'operational':op,'scientific_sha256':content_hash(s),'operational_sha256':content_hash(op),
        'protocol_sha256':content_hash({'schedule':s}),'protected_keys':{'root_seed':300,'training_phase':101,'validation_phase':102,'test_phase':103},
        'evidence':{'runtime_identity':runtime_fixture(),'resources':{'max_seconds':1e6}}}


class DurableConfirmationTests(unittest.TestCase):
    def test_conditional_schedule_and_truth_boundary(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from prepared_learning.runner import RunStore
        from prepared_learning.methods import Method
        manifest=execution_manifest(); original=Method.fit_meta; seen=[]
        def inspect(self,train,val,config):
            self_test=set(config)
            if self_test & {'theta','population_seed','spectrum','second_moment','protected_keys'}: raise AssertionError('truth leaked')
            seen.append((len(train),len(val)))
            return original(self,train,val,config)
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'),patch.object(Method,'fit_meta',inspect):
            gate=confirmation.run_confirmation(manifest,Path(root))
            self.assertEqual(gate.code,'INSENSITIVE_OR_UNSUITABLE')
            store=RunStore(Path(gate.evidence['run_path']))
            rows=store.read_records('preparation_risks')
            self.assertEqual(len(rows),4)
            self.assertEqual({r['method_id'] for r in rows},{'baseline','joint'})
            self.assertTrue(all(counts==(3,2) for counts in seen))
            self.assertEqual(gate.evidence['primary_task_count'],0)
            for row in store.read_records('execution'):
                self.assertIn(row['result']['selected']['baseline']['method_id'],('B01','B02','B11:Z'))

    def test_resume_and_failure_matches_run_store(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from prepared_learning.runner import RunStore
        from prepared_learning.methods import Method
        manifest=execution_manifest('nested_task_simulation'); original=RunStore.commit_block; interrupted=[]
        def interrupt(store,key,rows,assets):
            original(store,key,rows,assets)
            if not interrupted: interrupted.append(key); raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'):
            with patch.object(RunStore,'commit_block',interrupt):
                with self.assertRaises(KeyboardInterrupt): confirmation.run_confirmation(manifest,Path(root))
            resumed=confirmation.run_confirmation(manifest,Path(root))
            path=Path(resumed.evidence['run_path']); store=RunStore(path)
            self.assertEqual(len(store.read_records('execution')),2)
            self.assertEqual(len(store.read_records('task_metrics')),14)
            self.assertEqual(json.loads((path/'run.json').read_text())['status'],'completed')
            with tempfile.TemporaryDirectory() as failed_root,patch.object(Method,'adapt',side_effect=ValueError('injected invariant failure')):
                failed=confirmation.run_confirmation(execution_manifest('fixed_prior_task_simulation'),Path(failed_root))
                self.assertEqual(failed.code,'IMPLEMENTATION_INVALID')
                rows=RunStore(Path(failed.evidence['run_path'])).read_records('task_metrics')
                self.assertEqual({r['method_id'] for r in rows},{'baseline','joint'})
                self.assertTrue(all(r['status']=='failed' for r in rows))

    def test_stop_rules_only_validity_or_resources(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from prepared_learning import execution
        actual=execution.evaluate_task
        for effect in (-1.,0.,1.):
            def controlled(*args,**kwargs):
                values,failures=actual(*args,**kwargs)
                values['baseline']['loss']=10.+effect
                values['joint']['loss']=10.
                return values,failures
            with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'),patch.object(execution,'evaluate_task',side_effect=controlled):
                gate=confirmation.run_confirmation(execution_manifest('nested_task_simulation'),Path(root))
                self.assertEqual(gate.evidence['completed_outer_replications'],2)
                self.assertEqual(gate.evidence['primary_task_count'],6)
                self.assertEqual(gate.evidence['mean'],effect)

    def test_manifest_only_untracked_exception(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning.confirmation import _live_identity
        manifest=execution_manifest()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'fixture.json'; path.write_text(json.dumps(manifest))
            def git(args,**kwargs):
                return b'' if args[1]=='diff' else str(path).encode()+b'\0'
            with patch('prepared_learning.confirmation.validate_frozen_protocol'),patch('prepared_learning.confirmation.checkout_identity',return_value=(manifest['scientific']['versions'],'dirty')),patch('subprocess.check_output',side_effect=git):
                _live_identity(manifest,path)
                with self.assertRaises(ValueError): _live_identity(manifest)
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError,'content'): _live_identity(manifest,path)
                path.write_text(json.dumps(manifest))
                def extra(args,**kwargs):
                    return b'' if args[1]=='diff' else str(path).encode()+b'\0'+str(Path(root)/'unrelated.py').encode()+b'\0'
                with patch('subprocess.check_output',side_effect=extra):
                    with self.assertRaisesRegex(ValueError,'beyond'): _live_identity(manifest,path)

    def test_source_fit_failure_retains_atomic_failed_pair(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from prepared_learning.runner import RunStore
        from prepared_learning.methods import Method
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'),patch.object(Method,'fit_meta',side_effect=ValueError('source fit failure')):
            gate=confirmation.run_confirmation(execution_manifest(),Path(root))
            self.assertEqual(gate.code,'IMPLEMENTATION_INVALID')
            rows=RunStore(Path(gate.evidence['run_path'])).read_records('preparation_risks')
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['status']=='invalid' for r in rows))

class MeanInferenceTests(unittest.TestCase):
    def test_fixed_task_mean_uses_real_calibration_at_frozen_n(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import pilot, protocol, confirmation
        c,_=tiny('fixed_prior_task_simulation')
        c=replace(c,methods=('B01','B02'),sample_plan=replace(c.sample_plan,tasks_per_replicate=40),
            numerics={**c.numerics,'pilot':{'blocks':2,'procedure':'student','target_half_width':.2,'calibration_resamples':500,'max_seconds':1e6}})
        actual=protocol.checkout_identity
        with tempfile.TemporaryDirectory() as root,patch.object(protocol,'checkout_identity',side_effect=lambda *a,**k:(actual(*a,**k)[0],None)):
            evidence=pilot.run_pilot(c,Path(root)/'pilot')
            self.assertEqual(evidence['calibration']['status'],'adequate')
            self.assertEqual(evidence['gate']['reasons'],[])
            draft=json.loads((Path(evidence['run_path'])/'protocol_draft.json').read_text())['draft']
            manifest=protocol.freeze_protocol(draft,evidence,Path(root)/'fixture.json')
            n=draft['scientific']['config']['sample_plan']['tasks_per_replicate']
            self.assertEqual(evidence['calibration']['sample_size'],n)
            with patch.object(confirmation,'_live_identity',side_effect=lambda m,*a:protocol.validate_frozen_protocol(m)):
                result=confirmation.run_confirmation(manifest,Path(root)/'confirm')
            self.assertEqual(result.evidence['n'],n)
            self.assertIsNotNone(result.evidence['interval'])

class ReviewIdentityTests(unittest.TestCase):
    def test_extra_untracked_symlink_to_manifest_is_rejected(self):
        import tempfile, json
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning.confirmation import _live_identity
        manifest=execution_manifest()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'manifest.json'; path.write_text(json.dumps(manifest))
            alias=Path(root)/'extra.json'; alias.symlink_to(path)
            def git(args,**kwargs):
                return b'' if args[1]=='diff' else str(path).encode()+b'\0'+str(alias).encode()+b'\0'
            with patch('prepared_learning.confirmation.validate_frozen_protocol'),patch('prepared_learning.confirmation.checkout_identity',return_value=(manifest['scientific']['versions'],'dirty')),patch('subprocess.check_output',side_effect=git):
                with self.assertRaisesRegex(ValueError,'beyond'):
                    _live_identity(manifest,path)

class ReviewResourceTests(unittest.TestCase):
    def test_fixed_run_checks_resources_inside_its_only_replication(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation, pilot
        actual=pilot._rss_bytes
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'),patch.object(pilot,'_rss_bytes',wraps=actual) as rss:
            gate=confirmation.run_confirmation(execution_manifest('fixed_prior_task_simulation'),Path(root))
            self.assertNotEqual(gate.code,'IMPLEMENTATION_INVALID')
            self.assertGreater(rss.call_count,3)

    def test_checkpoint_size_independent_interrupted_replay(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from prepared_learning.runner import RunStore
        self.assertTrue(hasattr(RunStore,'iter_records'),'bounded verified raw iteration missing')
        manifest=execution_manifest('fixed_prior_task_simulation')
        original=RunStore.commit_block; interrupted=[]
        def interrupt(store,key,rows,assets):
            original(store,key,rows,assets)
            if key.startswith('tasks-') and not interrupted:
                interrupted.append(key); raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'):
            with patch.object(RunStore,'commit_block',interrupt):
                with self.assertRaises(KeyboardInterrupt): confirmation.run_confirmation(manifest,Path(root),checkpoint_tasks=1)
            gate=confirmation.run_confirmation(manifest,Path(root),checkpoint_tasks=2)
            store=RunStore(Path(gate.evidence['run_path']))
            tasks=list(store.iter_records('execution_task'))
            self.assertEqual(len(tasks),4)
            self.assertEqual(len({r['result']['task_key'] for r in tasks}),4)
            self.assertEqual(gate.evidence['primary_task_count'],3)
            self.assertTrue(all(not r['result']['tasks'] for r in store.iter_records('execution')))

    def test_bootstrap_draws_are_bounded_and_chunk_reproducible(self):
        from prepared_learning.analysis import paired_interval
        import inspect
        self.assertIn('working_chunk',inspect.signature(paired_interval).parameters)
        calls=[]
        a=paired_interval(np.arange(19.),.05,'outer_bootstrap',15,working_chunk=5,resource_check=lambda size=0:calls.append(size))
        b=paired_interval(np.arange(19.),.05,'outer_bootstrap',15,working_chunk=11)
        np.testing.assert_allclose(a['interval'],b['interval'],atol=1e-12)
        self.assertGreater(len(calls),10000)

class ReviewEvidenceTests(unittest.TestCase):
    def test_learned_driver_retains_both_frontier_branches_geometry_and_timing(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from prepared_learning.runner import RunStore
        manifest=execution_manifest()
        manifest['scientific']['config']['methods']=['B01','B03','B04']
        manifest['scientific']['selectors']={'baseline':['B01'],'joint':['B04']}
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'):
            gate=confirmation.run_confirmation(manifest,Path(root))
            store=RunStore(Path(gate.evidence['run_path']))
            certificates=store.read_records('certificates')
            self.assertTrue(certificates,'actual frontier solver evidence was dropped')
            self.assertEqual({row['branch'] for row in certificates},{'F','Z'})
            self.assertTrue(all(row['certificate_level']=='float64_diagnostic' for row in certificates))
            geometry=store.read_records('geometry')
            self.assertTrue(geometry)
            self.assertTrue(any(row['post_norm'] is not None for row in geometry))
            self.assertTrue(all(row['kernel_absolute_error'] is not None for row in geometry))
            timings=store.read_records('timing_blocks')
            self.assertTrue(timings)
            self.assertTrue(all(row['wall_time_ns']>0 for row in timings))

class ReviewResourceRecoveryTests(unittest.TestCase):
    def test_resource_stop_keeps_committed_task_prefix_resumable(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from prepared_learning.execution import ExecutionResourceLimit
        from prepared_learning.runner import RunStore
        actual=RunStore.commit_block; stopped=[]
        def limit(store,key,rows,assets):
            actual(store,key,rows,assets)
            if key.startswith('tasks-') and not stopped:
                stopped.append(key); raise ExecutionResourceLimit('injected resource boundary')
        manifest=execution_manifest('fixed_prior_task_simulation')
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'):
            with patch.object(RunStore,'commit_block',limit):
                first=confirmation.run_confirmation(manifest,Path(root),checkpoint_tasks=1)
            self.assertEqual(first.code,'LOCAL_RESOURCE_LIMIT')
            path=Path(first.evidence['run_path'])
            self.assertFalse((path/'terminal.json').exists())
            resumed=confirmation.run_confirmation(manifest,Path(root),checkpoint_tasks=2)
            self.assertEqual(resumed.evidence['primary_task_count'],3)
            observed=[(r['result']['task_key'],{k:v['loss'] for k,v in r['result']['values'].items()}) for r in RunStore(path).iter_records('execution_task')]
            with tempfile.TemporaryDirectory() as fresh:
                reference=confirmation.run_confirmation(manifest,Path(fresh),checkpoint_tasks=3)
                expected=[(r['result']['task_key'],{k:v['loss'] for k,v in r['result']['values'].items()}) for r in RunStore(reference.evidence['run_path']).iter_records('execution_task')]
            self.assertEqual(observed,expected)

    def test_final_schedule_memory_bound_is_recomputed(self):
        from tests.test_protocol import draft_fixture,evidence_fixture
        from prepared_learning.protocol import validate_execution_envelope
        draft=draft_fixture(); evidence=evidence_fixture(draft)
        validate_execution_envelope(draft['scientific'],evidence['resources'])
        draft['scientific']['config']['sample_plan']['outer_replicates']=10**9
        with self.assertRaisesRegex(ValueError,'working sets'):
            validate_execution_envelope(draft['scientific'],evidence['resources'])

class RoundTwoEvidenceTests(unittest.TestCase):
    def test_restoration_failures_retain_actual_payload_in_raw_certificates(self):
        from unittest.mock import patch
        from prepared_learning import execution,confirmation
        from prepared_learning.methods import Method,PreparationError
        from prepared_learning.records import to_json_record
        c,s=tiny(); c=replace(c,methods=('B01','B03','B04')); s['config']=to_json_record(c)
        s['selectors']={'baseline':['B01'],'joint':['B04']}
        phases={'training_phase':101,'test_phase':103}; original=Method.prepare
        payloads=[]
        restore=execution._restore_candidate
        def restored(row):
            method,prior=restore(row); method.review_restoration=True; return method,prior
        def prepare(method,*args):
            state=original(method,*args)
            if getattr(method,'review_restoration',False) and method.method_id=='B04':
                payload=to_json_record(state.solver_diagnostics)
                payload['review_actual_attempt']=len(payloads); payloads.append(payload)
                raise PreparationError('actual restoration failure',payload)
            return state
        with patch.object(execution,'_restore_candidate',restored),patch.object(Method,'prepare',prepare):
            result,_=execution.execute_replication(c,s,300,0,phases)
        records=result.get('restoration_failures',[])
        self.assertEqual(len(records),len(payloads)); self.assertGreater(len(records),1)
        self.assertEqual([r['preparation_diagnostics']['review_actual_attempt'] for r in records],list(range(len(payloads))))
        self.assertTrue(all(r['prepared_state'] is None and 'geometry' in r['unavailable_reasons'] for r in records))
        certificates=[r for r in result['verification_records'] if r['configuration_key'].startswith('restoration:')]
        self.assertEqual(len(certificates),2*len(payloads))
        manifest=execution_manifest(); manifest['scientific']=dict(manifest['scientific'],**s)
        raw=confirmation._rows(result,manifest,'fixture')
        self.assertEqual(sum(r['kind']=='certificates' and r['configuration_key'].startswith('restoration:') for r in raw),len(certificates))

class RoundTwoAnalysisGuardTests(unittest.TestCase):
    def test_analysis_bounds_existing_inventory_before_open(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from prepared_learning.runner import RunStore
        from prepared_learning.execution import ExecutionResourceLimit
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'):
            path=Path(root); (path/'journal').mkdir(); (path/'blocks').mkdir()
            for i in range(40): (path/'journal'/str(i)).write_text('not parsed before guard')
            def check(size=0):
                if size>=1024*1024+40*4096: raise ExecutionResourceLimit('actual old inventory')
            with patch.object(RunStore,'open',side_effect=AssertionError('open before inventory guard')):
                with self.assertRaisesRegex(ExecutionResourceLimit,'actual old inventory'):
                    confirmation.analyze_run(path,execution_manifest(),resource_check=check)

class RoundTwoEntryAndFailureTests(unittest.TestCase):
    def test_failed_validation_has_raw_branch_and_measured_geometry_records(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from prepared_learning.methods import Method
        from prepared_learning.runner import RunStore
        manifest=execution_manifest(); manifest['scientific']['config']['methods']=['B01','B03','B04']
        manifest['scientific']['selectors']={'baseline':['B01'],'joint':['B04']}
        adapt=Method.adapt
        def failed(method,*args):
            if method.method_id=='B03': raise RuntimeError('validation after real preparation')
            return adapt(method,*args)
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'),patch.object(Method,'adapt',failed):
            decision=confirmation.run_confirmation(manifest,Path(root))
            store=RunStore(Path(decision.evidence['run_path']))
            ledger=store.read_records('execution')[0]['result']
            candidates=[r for r in ledger['candidate_records'] if r['method_id']=='B03']
            self.assertTrue(candidates); self.assertTrue(all(r['status']=='failed' for r in candidates))
            certificates=store.read_records('certificates'); geometry=store.read_records('geometry')
            for row in candidates:
                self.assertEqual({r['branch'] for r in certificates if r['configuration_key']==row['candidate_id']},{'F','Z'})
                actual=next(r for r in geometry if r['method_id']==row['candidate_id'])
                self.assertEqual(actual['initial_norm'],row['geometry_record']['initial_norm'])
                self.assertIsNotNone(actual['kernel_absolute_error'])

    def test_checkpoint_time_adjustment_includes_frozen_execution_cost(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from tests.test_protocol import evidence_fixture,draft_fixture
        manifest=execution_manifest('fixed_prior_task_simulation')
        manifest['scientific']['config']['sample_plan']['tasks_per_replicate']=33
        resources=evidence_fixture(draft_fixture())['resources']
        resources['execution_envelope']['checkpoint_seconds']=1.
        resources.update(estimated_total_seconds=200.,max_seconds=210.)
        manifest['evidence']['resources']=resources
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'),patch.object(confirmation,'evaluate_replication',side_effect=AssertionError('unbudgeted execution')):
            result=confirmation.run_confirmation(manifest,Path(root),checkpoint_tasks=1)
            self.assertEqual(result.code,'LOCAL_RESOURCE_LIMIT')
            self.assertIn('runtime',result.evidence['reason'])

    def test_confirmation_guards_recovered_inventory_before_open(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from prepared_learning import confirmation
        from prepared_learning.runner import RunStore
        from prepared_learning.execution import ExecutionResourceLimit
        manifest=execution_manifest(); run=confirmation.run_record(manifest)
        with tempfile.TemporaryDirectory() as root,patch.object(confirmation,'_live_identity'):
            path=Path(root)/run['run_id']; (path/'journal').mkdir(parents=True); (path/'blocks').mkdir()
            for i in range(40): (path/'journal'/str(i)).write_text('do not parse')
            def check(size=0):
                if size>=1024*1024+40*4096: raise ExecutionResourceLimit('existing inventory')
            with patch('prepared_learning.execution.ResourceBudget',return_value=check),patch.object(RunStore,'open',side_effect=AssertionError('unguarded open')):
                result=confirmation.run_confirmation(manifest,Path(root),checkpoint_tasks=1024)
            self.assertEqual(result.code,'LOCAL_RESOURCE_LIMIT')
            self.assertFalse((path/'terminal.json').exists())
