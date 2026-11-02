"""Replay identity and arithmetic comparisons on explicitly tiny fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tests.test_reporting import reporting, confirmation_input


def learned_family_manifest():
    """One audited root300 software fixture with the original learned selectors."""
    from dataclasses import replace
    from prepared_learning.protocol import checkout_identity
    from prepared_learning.records import content_hash, to_json_record
    from tests.test_confirmation import tiny, execution_manifest
    from tests.test_protocol import runtime_fixture
    config, _ = tiny()
    config = replace(config, methods=('B01','B02','B03','B04','B11:F','B11:Z'))
    manifest = execution_manifest()
    scientific = manifest['scientific']
    scientific['config'] = to_json_record(config)
    scientific['selectors'] = {'baseline':['B01','B02','B11:Z'],
                               'joint':['B03','B04','B11:F','B11:Z']}
    scientific['versions'], _ = checkout_identity(scientific['selectors'],
        config.numerics['fitting_candidates'], runtime=runtime_fixture())
    manifest['scientific_sha256'] = content_hash(scientific)
    manifest['protocol_sha256'] = content_hash({'schedule':scientific})
    return manifest


class ReplayTests(unittest.TestCase):
    def test_learned_replay_across_process_hash_seeds(self):
        import os
        import subprocess
        import sys
        from prepared_learning.runner import RunStore
        api = reporting(self)
        # Hash randomization must be declared before interpreter startup/imports.
        seeds = ('0','1','2','3','4','5','6','7')
        child = '''
import json, sys
from pathlib import Path
from unittest.mock import patch
from prepared_learning import confirmation
manifest = json.loads(Path(sys.argv[1]).read_text())
# Existing detached-fixture accommodation only; preparation and publication are real.
with patch.object(confirmation, '_live_identity'):
    decision = confirmation.run_confirmation(manifest, Path(sys.argv[2]))
print(json.dumps({'gate':decision.code, 'run_path':decision.evidence['run_path']}))
'''
        family_ids = ['family:B01','family:B02','family:B03','family:B04',
                      'family:B11:F','family:B11:Z']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = learned_family_manifest()
            source = json.dumps(manifest, sort_keys=True)
            manifest_path = root/'fixture.json'
            manifest_path.write_text(source)
            paths = []
            reference_associations = None
            for seed in seeds:
                process = subprocess.run([sys.executable, '-c', child, str(manifest_path),
                                          str(root/('seed-'+seed))],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, 'PYTHONHASHSEED':seed}, text=True,
                    capture_output=True, check=True)
                decision = json.loads(process.stdout)
                self.assertEqual(decision['gate'], 'INSENSITIVE_OR_UNSUITABLE', process.stderr)
                self.assertEqual(manifest_path.read_text(), source)
                paths.append(Path(decision['run_path']))
            for seed, path in zip(seeds, paths):
                rows = RunStore(path).read_records('execution')
                self.assertEqual(len(rows), 2)
                associations = []
                for row in rows:
                    result = row['result']
                    self.assertEqual(result['status'], 'completed')
                    self.assertEqual(result['failures'], {})
                    self.assertEqual(len(result['source_keys']), 5)
                    ledger = result['candidate_records']
                    self.assertEqual({r['method_id'] for r in ledger}, set(manifest['scientific']['config']['methods']))
                    self.assertTrue(all(r['status']=='completed' for r in ledger))
                    candidates = ['candidate:'+r['candidate_id'] for r in ledger]
                    geometry = result['geometry_records']
                    keys = [r['method_id'] for r in geometry]
                    print('hash seed', seed, 'outer', result['outer_replication'],
                          'candidates', len(candidates), 'family order', keys[len(candidates):-2])
                    with self.subTest(seed=seed, outer=result['outer_replication'], check='geometry order'):
                        self.assertEqual(keys, candidates+family_ids+['baseline','joint'])
                    by_key = {r['method_id']:r for r in geometry}
                    for role in ('baseline','joint'):
                        selected = result['selected'][role]
                        candidate = 'candidate:'+selected['candidate_id']
                        self.assertIn(selected['method_id'], manifest['scientific']['selectors'][role])
                        self.assertEqual(result['prepared_resources'][role], result['prepared_resources'][candidate])
                        self.assertEqual({k:v for k,v in by_key[role].items() if k!='method_id'},
                                         {k:v for k,v in by_key[candidate].items() if k!='method_id'})
                    associations.append({k:result[k] for k in ('candidate_records','source_keys','source_key',
                        'training_observation_hash','validation_observation_hash','selected','prepared_resources')})
                if reference_associations is None:
                    reference_associations = api._scientific(associations)
                api.compare_scientific(reference_associations, api._scientific(associations), rtol=1e-10, atol=1e-12)
                with self.subTest(seed=seed, check='native strict store and assets'):
                    try:
                        comparison = api.compare_run_stores(paths[0], path, rtol=1e-10, atol=1e-12)
                    except ValueError as error:
                        self.fail(str(error))
                    self.assertEqual(comparison['status'], 'matched')
                    self.assertGreater(comparison['scientific_records_compared'], 0)
                    self.assertGreater(comparison['arrays_compared'], 0)
                    print('hash seed', seed, 'strict comparison:', comparison['scientific_records_compared'],
                          'records,', comparison['arrays_compared'], 'exact frozen arrays')

    def test_complete_learned_geometry_comparison_rejects_scientific_changes(self):
        import copy
        from prepared_learning import confirmation
        from prepared_learning.runner import RunStore
        api = reporting(self)
        with tempfile.TemporaryDirectory() as tmp, patch.object(confirmation, '_live_identity'):
            decision = confirmation.run_confirmation(learned_family_manifest(), Path(tmp))
            self.assertEqual(decision.code, 'INSENSITIVE_OR_UNSUITABLE')
            original = RunStore(Path(decision.evidence['run_path'])).read_records('execution')[0]
        geometry = original['result']['geometry_records']
        family = next(i for i,r in enumerate(geometry) if r['method_id']=='family:B11:F')
        other = next(i for i,r in enumerate(geometry) if r['method_id']=='family:B11:Z')
        for change in ('gamma','identity','missing','duplicate','selected','permutation'):
            changed = copy.deepcopy(original)
            records = changed['result']['geometry_records']
            if change=='gamma': records[family]['gamma'] += .1
            elif change=='identity': records[family]['method_id'] = 'family:changed'
            elif change=='missing': records.pop(family)
            elif change=='duplicate': records.append(copy.deepcopy(records[family]))
            elif change=='selected': changed['result']['selected']['joint']['candidate_id'] = 'changed-candidate'
            else: records[family], records[other] = records[other], records[family]
            with self.subTest(change=change), self.assertRaises(ValueError):
                api.compare_scientific(api._scientific(original), api._scientific(changed), rtol=1e-10, atol=1e-12)

    def test_worker_resource_stop_remains_local_resource_limit(self):
        api=reporting(self)
        import subprocess
        from prepared_learning._replay import _run
        evidence={'phase':'worker_comparison','estimated_peak_bytes':200,'budget_bytes':100,
                  'recovery':'retry identical scientific inputs'}
        result=subprocess.CompletedProcess(['worker'],75,json.dumps({'gate':'LOCAL_RESOURCE_LIMIT','evidence':evidence}),'')
        with tempfile.TemporaryDirectory() as tmp,patch('prepared_learning._replay.subprocess.run',return_value=result):
            with self.assertRaises(api.ReportingResourceLimit) as caught:
                _run(['worker'],cwd=Path(tmp),log=Path(tmp)/'worker.json')
            self.assertEqual(caught.exception.evidence,evidence)
            self.assertEqual(json.loads((Path(tmp)/'worker.json').read_text())['returncode'],75)

    def test_replay_limits_arrays_before_store_open_or_loading(self):
        api=reporting(self)
        from prepared_learning.runner import RunStore
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=confirmation_input(root)
            manifest=api.build_report_inputs([descriptor])
            asset=next((run/'blocks').glob('*/assets/*.npy'))
            with asset.open('wb') as stream: stream.truncate(128*1024*1024)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(256*1024*1024)}), \
                    patch.object(RunStore,'open',side_effect=AssertionError('store opened before array-size guard')), \
                    patch.object(api.np,'load',side_effect=AssertionError('array allocated before guard')):
                decision=api.replay_run(manifest,root/'replay')
                self.assertEqual(decision.code,'LOCAL_RESOURCE_LIMIT')
                self.assertGreaterEqual(decision.evidence['max_array_bytes'],128*1024*1024)
                self.assertTrue(hasattr(api,'ReportingResourceLimit'))
                with self.assertRaises(api.ReportingResourceLimit):
                    api.compare_run_stores(run,run,rtol=1e-10,atol=1e-12)
            self.assertEqual(asset.stat().st_size,128*1024*1024)
            self.assertFalse((root/'replay/clone.json').exists())

    def test_learned_replay_remeasures_fit_cost_ids(self):
        api=reporting(self)
        from prepared_learning import confirmation
        from tests.test_confirmation import execution_manifest
        with tempfile.TemporaryDirectory() as tmp, patch.object(confirmation,'_live_identity'):
            root=Path(tmp); manifest=execution_manifest()
            a=confirmation.run_confirmation(manifest,root/'a')
            b=confirmation.run_confirmation(manifest,root/'b')
            result=api.compare_run_stores(Path(a.evidence['run_path']),Path(b.evidence['run_path']),rtol=1e-10,atol=1e-12)
            self.assertEqual(result['status'],'matched')
        left={'asset_fit_id':'B01:100','shrink_fit_id':'B01:100','asset_hash':'scientific-a'}
        right={'asset_hash':'scientific-a','shrink_fit_id':'B01:900','asset_fit_id':'B01:900'}
        self.assertEqual(api._scientific(left),api._scientific(right))
        right['shrink_fit_id']='B01:901'
        self.assertNotEqual(api._scientific(left),api._scientific(right))
        right=dict(left,asset_hash='scientific-b')
        self.assertNotEqual(api._scientific(left),api._scientific(right))
        # One shared map spans every cost field, not a map per field. Both a
        # split of a shared fit and a merge of distinct fits must be detected.
        left={'asset_fit_id':'B01:100','shrink_fit_id':'B01:100',
              'base_fit_id':'B01:200','noise_fit_id':'B01:200'}
        renamed={'noise_fit_id':'B01:900','base_fit_id':'B01:900',
                 'shrink_fit_id':'B01:800','asset_fit_id':'B01:800'}
        self.assertEqual(api._scientific(left),api._scientific(renamed))
        renamed['noise_fit_id']='B01:800'
        self.assertNotEqual(api._scientific(left),api._scientific(renamed))
        renamed['base_fit_id']='B01:800'
        self.assertNotEqual(api._scientific(left),api._scientific(renamed))

    def test_replay_orders_tasks_by_scientific_key_across_many_chunks(self):
        api=reporting(self)
        from prepared_learning import confirmation
        from tests.test_confirmation import execution_manifest
        with tempfile.TemporaryDirectory() as tmp, patch.object(confirmation,'_live_identity'):
            root=Path(tmp); manifest=execution_manifest('fixed_prior_task_simulation')
            manifest['scientific']['config']['sample_plan']['tasks_per_replicate']=19
            a=confirmation.run_confirmation(manifest,root/'a',checkpoint_tasks=1)
            b=confirmation.run_confirmation(manifest,root/'b',checkpoint_tasks=16)
            result=api.compare_run_stores(Path(a.evidence['run_path']),Path(b.evidence['run_path']),rtol=1e-10,atol=1e-12)
            self.assertEqual(result['status'],'matched')

    def test_replay_interrupt_equivalence(self):
        api = reporting(self)
        from prepared_learning import confirmation
        from prepared_learning.runner import RunStore
        from tests.test_confirmation import execution_manifest
        with tempfile.TemporaryDirectory() as tmp, patch.object(confirmation, '_live_identity'):
            root = Path(tmp); manifest = execution_manifest('fixed_prior_task_simulation')
            original = RunStore.commit_block; stopped = []
            def interrupted(store, key, rows, assets):
                original(store, key, rows, assets)
                if key.startswith('tasks-') and not stopped:
                    stopped.append(key); raise KeyboardInterrupt()
            with patch.object(RunStore, 'commit_block', interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    confirmation.run_confirmation(manifest, root/'interrupted', checkpoint_tasks=1)
            resumed = confirmation.run_confirmation(manifest, root/'interrupted', checkpoint_tasks=2)
            fresh = confirmation.run_confirmation(manifest, root/'fresh', checkpoint_tasks=3)
            result = api.compare_run_stores(Path(resumed.evidence['run_path']),
                                            Path(fresh.evidence['run_path']), rtol=1e-10, atol=1e-12)
            self.assertEqual(result['status'], 'matched')
            self.assertGreater(result['arrays_compared'], 0)
            self.assertGreater(result['scientific_records_compared'], 0)
            self.assertEqual(result['timing_policy'], 'remeasured_not_equal')

    def test_replay_uses_recorded_revision_and_lock(self):
        api = reporting(self)
        from prepared_learning._replay import _validate_source
        import subprocess,hashlib
        from prepared_learning.records import content_hash
        repository=Path(__file__).resolve().parents[1]
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
        def git_file(name): return subprocess.check_output(['git','show',revision+':'+name],cwd=repository)
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'src/prepared_learning'],cwd=repository,text=True).splitlines()
        source={'git_commit':revision,'dependency_lock_sha256':hashlib.sha256(git_file('requirements.lock.txt')).hexdigest(),
                'code_sha256':content_hash({n:hashlib.sha256(git_file(n)).hexdigest() for n in names if n.endswith('.py') and Path(n).parent.as_posix()=='src/prepared_learning'})}
        for field,pattern in (('git_commit','revision'),('dependency_lock_sha256','lock'),('code_sha256','source identity')):
            changed=dict(source); changed[field]='0'*64
            with self.assertRaisesRegex(ValueError,pattern): _validate_source(repository,changed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); descriptor, _ = confirmation_input(root)
            manifest = api.build_report_inputs([descriptor])
            for field in ('git_commit', 'dependency_lock_sha256', 'code_sha256'):
                changed = json.loads(json.dumps(manifest))
                changed['replay']['source'][field] = '0'*64
                with self.assertRaisesRegex(ValueError, 'identity|revision|lock|fixture'):
                    api.replay_run(changed, root/field)
            protocol = root/'frozen.json'; value = json.loads(protocol.read_text())
            value['scientific']['assets']['population']['unexpected_prior_change'] = 1
            protocol.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'hash|inventory'):
                api.replay_run(manifest, root/'changed_prior')

    def test_arithmetic_tolerances_do_not_cover_keys_or_counts(self):
        api = reporting(self)
        self.assertTrue(hasattr(api, 'compare_scientific'), 'strict scientific comparator missing')
        api.compare_scientific({'loss': 1.}, {'loss': 1.+1e-13}, rtol=1e-10, atol=1e-12)
        for left, right in (({'task_key':'a'}, {'task_key':'b'}), ({'n':3}, {'n':4}),
                            ({'loss':1.}, {'loss':1.1}), ({'loss':1.}, {})):
            with self.assertRaises(ValueError):
                api.compare_scientific(left, right, rtol=1e-10, atol=1e-12)


class BoundedBenchmarkReplay(unittest.TestCase):
    def test_benchmark_projection_preserves_cross_candidate_aliases(self):
        from prepared_learning import _replay as api
        from tests.test_reporting import benchmark_input
        import copy
        self.assertTrue(hasattr(api,'_benchmark_scientific'),'complete worker projection is missing')
        with tempfile.TemporaryDirectory() as tmp:
            _,run=benchmark_input(Path(tmp))
            worker=json.loads(next((run/'blocks').glob('*.json')).read_text())[0]
            renamed=copy.deepcopy(worker)
            candidates=renamed['resource']['selection']['candidate_records']
            candidates[0].update(asset_fit_id='B01:900',base_fit_id='B01:800')
            candidates[1].update(shrink_fit_id='B01:900',noise_fit_id='B01:800')
            left=api._benchmark_scientific(worker)
            self.assertEqual(left,api._benchmark_scientific(renamed))
            for field in ('shrink_fit_id','noise_fit_id'):
                changed=copy.deepcopy(renamed); changed['resource']['selection']['candidate_records'][1][field]='B01:777'
                with self.assertRaises(ValueError):
                    api.compare_scientific(left,api._benchmark_scientific(changed),rtol=1e-10,atol=1e-12)
            changed=copy.deepcopy(renamed); changed['resource']['selection']['candidate_records'][0]['base_fit_id']='B01:900'
            changed['resource']['selection']['candidate_records'][1]['noise_fit_id']='B01:900'
            with self.assertRaises(ValueError): api.compare_scientific(left,api._benchmark_scientific(changed),rtol=1e-10,atol=1e-12)
            for key in left['resource']:
                changed=copy.deepcopy(worker); changed['resource'][key]={'changed':True}
                with self.subTest(field=key),self.assertRaises((ValueError,TypeError)):
                    api.compare_scientific(left,api._benchmark_scientific(changed),rtol=1e-10,atol=1e-12)
            for delta,passes in ((1e-13,True),(1e-4,False)):
                changed=copy.deepcopy(worker); changed['resource']['selection']['candidate_records'][0]['loss']+=delta
                if passes: api.compare_scientific(left,api._benchmark_scientific(changed),rtol=1e-10,atol=1e-12)
                else:
                    with self.assertRaises(ValueError): api.compare_scientific(left,api._benchmark_scientific(changed),rtol=1e-10,atol=1e-12)

    def test_benchmark_file_output_retains_failure_evidence(self):
        from prepared_learning import _replay as api
        import sys,os
        self.assertTrue(hasattr(api,'_run_file'),'file-backed worker execution is missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for code in (0,3,75):
                stdout=root/'processes'/f'{code}.stdout.json'; stderr=root/'processes'/f'{code}.stderr.txt'; log=root/'processes'/f'{code}.log.json'
                payload={'gate':'LOCAL_RESOURCE_LIMIT','evidence':{'phase':'test','recovery':'retry'}} if code==75 else {'full':'x'*100000}
                expression=repr(payload) if code==75 else "{'full':'x'*100000}"
                script='import sys,json; print(json.dumps('+expression+')); sys.stderr.write("e"*100000); sys.exit('+str(code)+')'
                command=[sys.executable,'-c',script]
                if code==0: self.assertEqual(api._run_file(command,cwd=root,env=dict(os.environ),stdout_path=stdout,stderr_path=stderr,log=log,package=root),stdout)
                else:
                    with self.assertRaises(api.ReportingResourceLimit if code==75 else ValueError):
                        api._run_file(command,cwd=root,env=dict(os.environ),stdout_path=stdout,stderr_path=stderr,log=log,package=root)
                self.assertEqual(json.loads(stdout.read_text()),payload); self.assertEqual(stderr.stat().st_size,100000)
                record=json.loads(log.read_text()); self.assertEqual(record['returncode'],code)
                self.assertEqual(root/record['stdout']['path'],stdout)
                self.assertEqual(record['stdout']['sha256'],api._sha(stdout)); self.assertNotIn('x'*1000,json.dumps(record))

    def test_benchmark_replay_pairs_without_cohort_retention(self):
        self._benchmark_replay_pairs(canonical=False)

    def test_canonical_control_replay_preserves_pairs_and_requests(self):
        self._benchmark_replay_pairs(canonical=True)

    def _benchmark_replay_pairs(self, *, canonical):
        from prepared_learning import _replay as api
        from prepared_learning import reporting as report
        from prepared_learning.benchmark import _campaign_configuration, _fixed_schedule
        from prepared_learning.records import content_hash
        from tests.test_reporting import benchmark_input
        import copy,gc,weakref
        self.assertTrue(hasattr(api,'_replay_benchmark'),'paired benchmark replay is missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=benchmark_input(root,blocks=4)
            inputs=json.loads((run/'inputs.json').read_text()); config=inputs['inputs']['manifest']
            import shutil
            calibration=root/'calibration'; shutil.copytree(run,calibration)
            (calibration/'benchmark.json').write_text(json.dumps({'status':'exploratory','fixture':True}))
            workers=[w for path in sorted((run/'blocks').glob('*.json')) for w in json.loads(path.read_text())]
            schedule=_fixed_schedule({'config':config,'pilot':[w['pilot'] for w in workers],'schedule':{'balanced_blocks':4}})
            # Fixture schedule is recorded, and storage order deliberately differs from requests.
            for w in workers:
                for row in w['rows']: row['method_order']=schedule['orders'][row['block']]
            for i,path in enumerate(sorted((run/'blocks').glob('*.json'))):
                path.write_text(json.dumps(list(reversed(workers[i*4:(i+1)*4]))))
            for path in (run/'requests').glob('*.json'):
                req=json.loads(path.read_text()); req['method_order']=schedule['orders'][req['block']]
                req['position']=req['method_order'].index(req['method']['method_id']); path.write_text(json.dumps(req))
            frozen={**schedule,'identity':{'versions':{'fixture':True},'dirty_diff_sha256':None},
                    'selected_execution_hashes':{w['resource']['method_id']:w['resource']['selected_execution_hash'] for w in workers},
                    'condition_hash':content_hash(workers[0]['conditions'])}
            frozen['configuration_sha256']=content_hash(_campaign_configuration(config))
            frozen['calibration_path']=str(calibration/'benchmark.json')
            frozen['calibration_sha256']=content_hash(json.loads((calibration/'benchmark.json').read_text()))
            if canonical:
                from prepared_learning.benchmark import _write_canonical_calibration
                source=calibration/'benchmark.json'; sidecar=source.with_suffix('.canonical.json')
                _write_canonical_calibration(json.loads(source.read_text()),sidecar,frozen['calibration_sha256'])
                frozen['calibration_control']={'format':'prepared_calibration_canonical_v1','path':str(sidecar),'source_sha256':report._sha(source)}
            config.update(phase='finalized',frozen=frozen)
            (run/'inputs.json').write_text(json.dumps(inputs))
            entry={'run_path':str(run),'input_id':0,'source':{'fixture':True},'label':'fixture',
                   'kind':'benchmark','fixture':True,'descriptor_path':str(descriptor)}
            refs=[]; commands=[]; original=report._read; mutation=None
            project=api._benchmark_scientific
            class Tracked(dict): pass
            def read(path):
                if canonical: self.assertNotIn(Path(path),(source,sidecar),'control decoded during replay')
                value=original(path)
                if isinstance(value,dict) and 'resource' in value:
                    value=Tracked(value); refs.append(weakref.ref(value))
                return value
            def tracked_projection(worker):
                value=Tracked(project(worker)); refs.append(weakref.ref(value)); return value
            def run_file(command,**kw):
                gc.collect(); self.assertTrue(all(ref() is None for ref in refs),'completed pair still retained')
                req=json.loads(Path(command[-1]).read_text()); commands.append((req['block'],req['method']['method_id']))
                match=next(w for w in workers if w['rows'][0]['block']==req['block'] and w['resource']['method_id']==req['method']['method_id'])
                new=copy.deepcopy(match)
                for row in new['rows']: row['end_to_end_ns']+=700
                if mutation is not None: mutation(new)
                report._write(kw['stdout_path'],new); kw['stderr_path'].write_text('')
                def output_ref(path):
                    return {'path':path.relative_to(kw['package']).as_posix(),
                            'sha256':report._sha(path),'size_bytes':path.stat().st_size}
                report._write(kw['log'],{'command':command,'cwd':str(kw['cwd']),'returncode':0,
                    'stdout':output_ref(kw['stdout_path']),'stderr':output_ref(kw['stderr_path'])})
                return kw['stdout_path']
            destination=root/'replay'; destination.mkdir()
            with patch.object(api,'_run_file',side_effect=run_file),patch.object(report,'_read',side_effect=read),patch.object(api,'_read',side_effect=read),patch.object(api,'_benchmark_scientific',new=tracked_projection):
                result=api._replay_benchmark(entry,destination,root,{'rtol':1e-10,'atol':1e-12},config)
            self.assertEqual(result['workers'],16); self.assertEqual(len(set(commands)),16)
            measured=json.loads((destination/'remeasured-workers.json').read_text())
            self.assertEqual(len(measured),16); self.assertEqual(measured[0]['rows'][0]['end_to_end_ns'],1700)
            self.assertEqual(result['timing_stratum'],'technical_replay')
            self.assertEqual(result['report_schema_version'],'benchmark_raw_refs_v2')
            moved=root/'moved-package'; shutil.move(destination,moved)
            self.assertFalse(destination.exists())
            index=report._load_worker_index(result['remeasured_worker_records_ref'],moved)
            for reference,expected_worker in zip(index['records'],measured):
                self.assertEqual(report._load_worker_ref(reference,moved),expected_worker)
                self.assertEqual(reference.get('source_scope'),'package')
                self.assertFalse(Path(reference['source_block_path']).is_absolute())
                self.assertEqual(report._sha(moved/reference['source_block_path']),reference['source_block_sha256'])
                process_ref=reference['process_ref']
                self.assertEqual(report._sha(moved/process_ref['path']),process_ref['sha256'])
                process=report._read(moved/process_ref['path'])
                for name in ('stdout','stderr'):
                    self.assertFalse(Path(process[name]['path']).is_absolute())
                    self.assertEqual(report._sha(moved/process[name]['path']),process[name]['sha256'])
                self.assertEqual(process['stdout']['path'],reference['source_block_path'])
            # A copied worker must not conceal missing or changed generated
            # stdout/stderr/receipt evidence, even after the old base is gone.
            reference=index['records'][0]; process=report._read(moved/reference['process_ref']['path'])
            for relative in (reference['source_block_path'],process['stderr']['path'],reference['process_ref']['path']):
                path=moved/relative; original_bytes=path.read_bytes()
                with self.subTest(evidence=relative):
                    path.write_bytes(original_bytes+b'changed')
                    with self.assertRaises(ValueError): report._load_worker_ref(reference,moved)
                    path.unlink()
                    with self.assertRaises((ValueError,FileNotFoundError)): report._load_worker_ref(reference,moved)
                    path.write_bytes(original_bytes)
            gc.collect(); self.assertTrue(all(ref() is None for ref in refs))

            if canonical:
                for control in (source,sidecar):
                    before=control.read_bytes(); control.write_bytes(before+b' ')
                    target=root/('corrupt-'+control.name); target.mkdir()
                    with patch.object(api,'_run_file',side_effect=AssertionError('corrupt control executed')):
                        with self.assertRaises(ValueError): api._replay_benchmark(entry,target,root,{'rtol':1e-10,'atol':1e-12},config)
                    control.write_bytes(before)

            # An unexpectedly large real worker output keeps the existing
            # semantic256/copies2 pair gate, before decoding either worker.
            import os
            mutation=lambda w:w['resource'].update(unexpected_full_output='x'*(512*1024))
            target=root/'pair-resource-stop'; target.mkdir()
            def no_large_worker(path,**kwargs):
                self.assertFalse(str(path).endswith('.stdout.json'),'oversized replay worker decoded')
                return original(path,**kwargs)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(160*1024**2)}), \
                    patch.object(api,'_run_file',side_effect=run_file),patch.object(api,'_read',new=no_large_worker):
                with self.assertRaises(report.ReportingResourceLimit) as caught:
                    api._replay_benchmark(entry,target,root,{'rtol':1e-10,'atol':1e-12},config,retained_bytes=lambda:12345)
            self.assertEqual(caught.exception.evidence['phase'],'benchmark_replay_pair_before_parse')
            self.assertEqual(caught.exception.evidence['retained_copies'],2)
            self.assertEqual(caught.exception.evidence['policy']['json_expansion_factor'],256)
            self.assertGreater(caught.exception.evidence['object_bytes'],12345)
            self.assertFalse((target/'benchmark-input-0/remeasured/workers.index.json').exists())
            mutation=None

            # Missing, duplicate, wrong-key and changed requests cannot run a
            # different worker set and still receive a matched result.
            request_path=sorted((run/'requests').glob('*.json'))[0]
            original_request=json.loads(request_path.read_text())
            for i,change in enumerate((
                lambda r:r.update(block=99),
                lambda r:r.update(method_order=list(reversed(r['method_order']))),
                lambda r:r['settings'].update(fixed_repetitions=99),
                lambda r:r['inputs']['manifest'].update(dimension=3),
            )):
                changed=copy.deepcopy(original_request); change(changed); request_path.write_text(json.dumps(changed))
                target=root/f'request-failure-{i}'; target.mkdir()
                with patch.object(api,'_run_file',side_effect=AssertionError('invalid request executed')):
                    with self.assertRaises(ValueError): api._replay_benchmark(entry,target,root,{'rtol':1e-10,'atol':1e-12},config)
                self.assertFalse((target/'benchmark-input-0/remeasured/workers.index.json').exists())
            request_path.unlink(); target=root/'missing-request'; target.mkdir()
            with self.assertRaises(ValueError): api._replay_benchmark(entry,target,root,{'rtol':1e-10,'atol':1e-12},config)
            request_path.write_text(json.dumps(original_request))
            duplicate=run/'requests/duplicate.json'; duplicate.write_text(request_path.read_text())
            target=root/'duplicate-request'; target.mkdir()
            with self.assertRaises(ValueError): api._replay_benchmark(entry,target,root,{'rtol':1e-10,'atol':1e-12},config)
            duplicate.unlink()
            for i,mutation in enumerate((
                lambda w:w['rows'][0].update(block=99),
                lambda w:w['resource'].update(method_id='B04'),
                lambda w:w['rows'][0].update(repetitions=99),
                lambda w:w['conditions'].update(thermal='changed'),
                lambda w:w['resource']['selection']['candidate_records'][1]['probes'][0].update(status='pass'),
                lambda w:w['resource']['selected_asset'].update(unknown_asset={'changed':True}),
            )):
                target=root/f'result-failure-{i}'; target.mkdir()
                with patch.object(api,'_run_file',side_effect=run_file):
                    with self.assertRaises(ValueError): api._replay_benchmark(entry,target,root,{'rtol':1e-10,'atol':1e-12},config)
                self.assertTrue((target/'processes/00000000.stdout.json').exists())
                self.assertFalse((target/'benchmark-input-0/remeasured/workers.index.json').exists())


class PhasedReplayMemory(unittest.TestCase):
    """Native inventory is sequential; real pairs and retained results still cost RAM."""
    def setUp(self):
        available=patch("prepared_learning.pilot._available_memory",return_value=4*1024**3)
        available.start(); self.addCleanup(available.stop)

    def diagnostic_manifest(self, root, *, count=2, padding=0):
        api=reporting(self); descriptors=[]
        for index in range(count):
            raw=root/f'raw-{index}.json'
            raw.write_text(json.dumps({'status':'failed','index':index,'unknown':{'full':'x'*padding}}))
            path=root/f'input-{index}.json'; descriptors.append(path)
            path.write_text(json.dumps({'kind':'diagnostic','fixture':True,'raw_path':str(raw),
                                        'command':['fixture',str(index)]}))
        return api.build_report_inputs(descriptors)

    def producer(self, manifest):
        import subprocess
        def run(command, **kw):
            index=int(command[-1]); body=Path(manifest['inputs'][index]['raw_path']).read_text()
            if 'stdout' in kw:
                kw['stdout'].write(body.encode()); kw['stderr'].write(b'complete deliberate failure')
                return subprocess.CompletedProcess(command,1)
            return subprocess.CompletedProcess(command,1,body,'complete deliberate failure')
        return run

    def test_full_ordered_native_receipt_fits_sequential_phases(self):
        from prepared_learning import _replay as replay
        import os
        api=reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.diagnostic_manifest(root,count=16,padding=256*1024)
            before=json.dumps(manifest,sort_keys=True)
            old=api._guard_inputs(manifest)
            # The old global semantic two-copy reservation needs >2GiB.
            self.assertGreater(64*1024**2+2*(256*old['json_working_bytes']+old['inventory_bytes']),768*1024**2)
            def operation(command, **kw):
                if command[:2]==['git','clone']: Path(command[-1]).mkdir()
                elif command[1:3]==['-m','prepared_learning._replay']:
                    replay.worker(api._read(command[-1]))
                return ''
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(768*1024**2)}), \
                    patch('prepared_learning.pilot._available_memory',return_value=4*1024**3), \
                    patch.object(replay,'_validate_source',return_value='a'*40), \
                    patch.object(replay,'_git_bytes',return_value=b'{}'), \
                    patch.object(replay,'_run',side_effect=operation), \
                    patch.object(replay.subprocess,'run',side_effect=self.producer(manifest)):
                decision=api.replay_run(manifest,root/'replay')
            self.assertEqual(decision.code,'SUPPORTED_IN_SCOPE',decision.evidence)
            receipt=api._read(root/'replay/replay.json')
            self.assertEqual(receipt['input_order_sha256'],manifest['input_order_sha256'])
            self.assertEqual([r['input_id'] for r in receipt['executions']],list(range(16)))
            self.assertTrue(all(r['status']=='matched' and r['returncode']==1 for r in receipt['executions']))
            self.assertEqual(json.dumps(manifest,sort_keys=True),before)
            for index in range(16):
                process=api._read(root/f'replay/executions/{index}/process.json')
                self.assertEqual(json.loads(process['stdout']),api._read(manifest['inputs'][index]['raw_path']))
                self.assertEqual(process['stderr'],'complete deliberate failure')
            self.assertEqual(len(list((root/'replay').glob('replay.json'))),1)
            # Report reconstruction deliberately keeps its complete semantic
            # cohort. Its budget also fits below the old two-copy replay bound.
            self.assertGreater(64*1024**2+2*(256*old['json_working_bytes']+old['inventory_bytes']),1536*1024**2)
            manifest['replay']['receipt']={'path':str(root/'replay/replay.json'),'sha256':api._sha(root/'replay/replay.json')}
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(1536*1024**2)}):
                report=api.write_report(manifest,root/'delivery')
            self.assertEqual(report.evidence['clean_replay'],'matched')
            self.assertNotEqual(report.code,'SUPPORTED_IN_SCOPE','fixture receipt established production G10')
            tables=api._read(root/'delivery/tables.json')
            self.assertEqual([r['input_id'] for r in tables['statuses']],list(range(16)))
            self.assertEqual(len(tables['diagnostics']),16)
            for index,record in enumerate(tables['diagnostics']):
                self.assertEqual(record['raw'],api._read(manifest['inputs'][index]['raw_path']))

    def test_native_validation_and_read_charge_caller_before_parse(self):
        api=reporting(self)
        import inspect
        for function in (api._read,api._verify_inputs): self.assertIn("retained_bytes",inspect.signature(function).parameters)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.diagnostic_manifest(root,padding=1024)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024**2)}), \
                    patch.object(api.json,'loads',side_effect=AssertionError('parsed before caller admission')):
                for operation in (lambda:api._read(root/'raw-0.json',retained_bytes=8*1024**2),
                                  lambda:api._verify_inputs(manifest,replay=True,retained_bytes=8*1024**2)):
                    with self.assertRaises(api.ReportingResourceLimit): operation()

    def test_diagnostic_pair_refuses_before_raw_decode(self):
        api=reporting(self)
        from prepared_learning import _replay as replay
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.diagnostic_manifest(root,count=1,padding=128*1024)
            # Native validation fits; actual two-artifact semantic pair does not.
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(160*1024**2)}), \
                    patch.object(replay,'_validate_source'), \
                    patch.object(replay.subprocess,'run',side_effect=self.producer(manifest)):
                with self.assertRaises(api.ReportingResourceLimit) as caught:
                    replay.worker({'manifest':manifest,'output':str(root/'replay')})
            self.assertEqual(caught.exception.evidence['phase'],'diagnostic_replay_pair_before_parse')
            self.assertEqual(caught.exception.evidence['retained_copies'],2)
            self.assertEqual(caught.exception.evidence['policy']['json_expansion_factor'],256)
            self.assertFalse((root/'replay/replay-worker.json').exists())
            self.assertTrue((root/'replay/0/stdout.json').exists())
            self.assertTrue((root/'replay/0/process.json').exists(),'resource stop lost complete process.json')
            process=api._read(root/'replay/0/process.json')
            self.assertEqual(json.loads(process['stdout'])['index'],0)
            self.assertEqual(process['stderr'],'complete deliberate failure')

    def test_diagnostic_graphs_die_before_next_diagnostic_and_g9(self):
        from prepared_learning import _replay as replay
        import gc,weakref
        api=reporting(self)
        class Tracked(dict): pass
        refs=[]; decode=json.loads; project=replay._scientific
        def tracked_decode(*args,**kwargs):
            value=decode(*args,**kwargs)
            if isinstance(value,dict) and 'unknown' in value:
                value=Tracked(value); refs.append(weakref.ref(value))
            return value
        def projection(value):
            value=Tracked(project(value)); refs.append(weakref.ref(value)); return value
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.diagnostic_manifest(root,count=2,padding=100)
            manifest['inputs'].append({'input_id':2,'kind':'benchmark','run_path':str(root/'g9')})
            (root/'g9').mkdir(); (root/'g9/inputs.json').write_text(json.dumps({'inputs':{'manifest':{'phase':'finalized','frozen':{'identity':{}}}}}))
            producer=self.producer(manifest)
            def run(command,**kw):
                gc.collect(); self.assertTrue(all(ref() is None for ref in refs),'previous diagnostic graph retained')
                return producer(command,**kw)
            def benchmark(*args,**kw):
                gc.collect(); self.assertTrue(all(ref() is None for ref in refs),'G8 graph retained into G9')
                self.assertGreater(kw['retained_bytes'](),api._object_bytes(manifest))
                return {'input_id':2,'kind':'benchmark','status':'matched'}
            with patch.object(replay,'_verify_inputs'),patch.object(replay,'_validate_source'), \
                    patch.object(replay.subprocess,'run',side_effect=run),patch.object(json,'loads',side_effect=tracked_decode), \
                    patch.object(replay,'_scientific',new=projection), \
                    patch('prepared_learning.benchmark._benchmark_identity',return_value={}), \
                    patch.object(replay,'_replay_benchmark',side_effect=benchmark):
                replay.worker({'manifest':manifest,'output':str(root/'replay')})
            gc.collect(); self.assertGreaterEqual(len(refs),8)
            self.assertTrue(all(ref() is None for ref in refs))

    def test_protected_caller_and_prospective_maps_refuse_before_comparison_allocation(self):
        api=reporting(self)
        import inspect
        self.assertIn("retained_bytes",inspect.signature(api.compare_run_stores).parameters)
        from prepared_learning.runner import RunStore
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,run=confirmation_input(root)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024**2)}), \
                    patch.object(RunStore,'open',side_effect=AssertionError('opened before caller guard')):
                with self.assertRaises(api.ReportingResourceLimit):
                    api.compare_run_stores(run,run,rtol=1e-10,atol=1e-12,retained_bytes=8*1024**2)
            # A large path/key inventory must be reserved before building maps.
            folder=next((run/'blocks').iterdir())/'assets'
            for index in range(3000): (folder/(str(index)+'-'+'x'*180+'.npy')).touch()
            # Authoritative recovery is separately admitted before map sizing;
            # index/full-map allocation must still wait for prospective sizing.
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(160*1024**2)}), \
                    patch('sqlite3.connect',side_effect=AssertionError('comparison index allocated before map admission')):
                with self.assertRaises(api.ReportingResourceLimit):
                    api.compare_run_stores(run,run,rtol=1e-10,atol=1e-12)

    def test_native_policy_keeps_two_copies_constants_and_lower_only_half_cap(self):
        api=reporting(self)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.diagnostic_manifest(root,count=1,padding=1024)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(8*1024**3)}):
                evidence=api._guard_inputs(manifest,replay=True,retained_bytes=12345)
            self.assertEqual(evidence['policy']['json_expansion_factor'],64)
            self.assertEqual(evidence['retained_copies'],2)
            self.assertEqual(evidence['budget_bytes'],2*1024**3)
            self.assertEqual(evidence['reserve_fraction'],.5)
            self.assertEqual(evidence['policy']['base_bytes'],64*1024**2)
            self.assertEqual(evidence['policy']['array_workspace_factor'],16)
            expected=(64*1024**2+2*(64*evidence['json_working_bytes']+evidence['inventory_bytes'])
                      +16*evidence['max_array_bytes']+16*(12345+api._object_bytes(manifest)))
            self.assertEqual(evidence['estimated_peak_bytes'],expected)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(expected-1)}):
                with self.assertRaises(api.ReportingResourceLimit): api._guard_inputs(manifest,replay=True,retained_bytes=12345)
            raw=root/'raw-0.json'
            with raw.open('wb') as stream: stream.truncate(8*1024**2)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024**2)}), \
                    patch.object(api,'_read',side_effect=AssertionError('oversized native inventory parsed')):
                decision=api.replay_run(manifest,root/'replay')
            self.assertEqual(decision.code,'LOCAL_RESOURCE_LIMIT')
            self.assertFalse((root/'replay/replay.json').exists())

    def test_retained_protected_hashes_stop_the_following_g9_before_read(self):
        api=reporting(self)
        from prepared_learning import _replay as replay
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.diagnostic_manifest(root,count=1)
            manifest['inputs'].append({'input_id':1,'kind':'benchmark','run_path':str(root/'g9')})
            (root/'g9').mkdir(); (root/'g9/inputs.json').write_text('{}')
            execute=replay._replay_entry
            def entry(record,*args,**kwargs):
                if record['input_id']==0:
                    return {'input_id':0,'kind':'confirmation','status':'matched','asset_hashes':{'retained':'x'*(8*1024**2)}}
                return execute(record,*args,**kwargs)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024**2)}), \
                    patch.object(replay,'_verify_inputs'),patch.object(replay,'_validate_source'), \
                    patch.object(replay,'_replay_entry',new=entry), \
                    patch('prepared_learning.benchmark._benchmark_identity',side_effect=AssertionError('G9 allocated despite retained hashes')):
                with self.assertRaises(api.ReportingResourceLimit): replay.worker({'manifest':manifest,'output':str(root/'replay')})
            self.assertFalse((root/'replay/replay-worker.json').exists())

    def test_final_worker_writer_reserves_every_result(self):
        api=reporting(self)
        from prepared_learning import _replay as replay
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.diagnostic_manifest(root,count=1)
            def entry(*args,**kwargs):
                return {'input_id':0,'status':'matched','asset_hashes':{'full':'x'*(8*1024**2)}}
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024**2)}), \
                    patch.object(replay,'_verify_inputs'),patch.object(replay,'_validate_source'),patch.object(replay,'_replay_entry',new=entry):
                with self.assertRaises(api.ReportingResourceLimit) as caught:
                    replay.worker({'manifest':manifest,'output':str(root/'replay')})
            self.assertEqual(caught.exception.evidence['phase'],'replay_worker_before_write')
            self.assertFalse((root/'replay/replay-worker.json').exists())

    def coordinate(self, root, manifest, *, worker_result, after_worker=lambda:None):
        """Only source/environment installation is replaced; final guards are real."""
        from prepared_learning import _replay as replay
        api=reporting(self)
        def operation(command,**kwargs):
            if command[:2]==['git','clone']: Path(command[-1]).mkdir()
            elif command[1:3]==['-m','prepared_learning._replay']:
                api._write(root/'replay/executions/replay-worker.json',worker_result)
                after_worker()
            return ''
        with patch.object(replay,'_validate_source',return_value='a'*40), \
                patch.object(replay,'_git_bytes',return_value=b'{}'),patch.object(replay,'_run',side_effect=operation):
            return api.replay_run(manifest,root/'replay')

    def test_final_original_mutation_blocks_receipt_publication(self):
        api=reporting(self)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.diagnostic_manifest(root,count=1)
            def mutate():
                path=root/'raw-0.json'; path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'inventory'):
                self.coordinate(root,manifest,worker_result=[{'input_id':0,'status':'matched'}],after_worker=mutate)
            self.assertFalse((root/'replay/replay.json').exists())

    def test_final_worker_result_file_is_admitted_before_decode(self):
        api=reporting(self)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.diagnostic_manifest(root,count=1)
            read=api._read
            def guarded(path,**kwargs):
                if Path(path).name=='replay-worker.json':
                    self.assertGreater(kwargs.get('retained_bytes',0),api._object_bytes(manifest))
                return read(path,**kwargs)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024**2)}), \
                    patch('prepared_learning._replay._read',new=guarded):
                decision=self.coordinate(root,manifest,worker_result=[{'full':'x'*(2*1024**2)}])
            self.assertEqual(decision.code,'LOCAL_RESOURCE_LIMIT')
            self.assertEqual(decision.evidence['phase'],'json_before_parse')
            self.assertFalse((root/'replay/replay.json').exists())

    def test_final_receipt_writer_charges_native_worker_and_restored_state(self):
        from prepared_learning import _replay as replay
        api=reporting(self)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=self.diagnostic_manifest(root,count=1)
            guard=api._resource_check
            def limit_writer(scale,**kwargs):
                if kwargs['phase']=='replay_receipt_before_write':
                    with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024**2)}): return guard(scale,**kwargs)
                return guard(scale,**kwargs)
            with patch.object(replay,'_resource_check',new=limit_writer):
                decision=self.coordinate(root,manifest,worker_result=[{'asset_hashes':{'full':'x'*(8*1024**2)}}])
            self.assertEqual(decision.code,'LOCAL_RESOURCE_LIMIT')
            self.assertEqual(decision.evidence['phase'],'replay_receipt_before_write')
            self.assertFalse((root/'replay/replay.json').exists())

    def test_actual_fresh_artifact_size_is_guarded_before_decode(self):
        from prepared_learning import _replay as replay
        from tests.test_final_handoffs import FullArtifactReplay
        import os
        api=reporting(self)
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root=Path(tmp).resolve(); manifest,_=FullArtifactReplay().producer(root,'frontier')
            execute=replay._run_file; read=replay._read; fresh=[]
            def grow_output(command,**kwargs):
                result=execute(command,**kwargs)
                artifact=Path(json.loads(Path(result).read_text())['attempt_path'])/'frontier.json'
                with artifact.open('wb') as stream: stream.truncate(8*1024**2)
                fresh.append(artifact)
                return result
            def no_artifact_decode(path,**kwargs):
                self.assertNotIn(Path(path),fresh,'oversized actual artifact decoded')
                return read(path,**kwargs)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(256*1024**2)}), \
                    patch.object(replay,'_validate_source'),patch.object(replay,'_run_file',new=grow_output), \
                    patch.object(replay,'_read',new=no_artifact_decode):
                with self.assertRaises(api.ReportingResourceLimit) as caught:
                    replay.worker({'manifest':manifest,'output':str(root/'replay')})
            self.assertEqual(caught.exception.evidence['phase'],'diagnostic_replay_pair_before_parse')
            self.assertEqual(caught.exception.evidence['retained_copies'],2)
            self.assertEqual(fresh[0].stat().st_size,8*1024**2)
            self.assertTrue((root/'replay/0/process.json').exists())
            self.assertFalse((root/'replay/replay-worker.json').exists())

    def test_g9_initial_original_validation_retains_supplied_configuration(self):
        from prepared_learning import _replay as replay
        from tests.test_reporting import benchmark_input
        api=reporting(self)
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor,run=benchmark_input(root)
            entry={'input_id':0,'kind':'benchmark','run_path':str(run),'descriptor_path':str(descriptor),
                   'source':{'fixture':True},'fixture':True,'label':'bounded fixture'}
            config={'retained_unknown_configuration':'x'*(8*1024**2)}
            read=api._read
            def no_worker(path,**kwargs):
                self.assertNotEqual(Path(path).parent,run/'blocks','G9 original parsed while supplied configuration was uncharged')
                return read(path,**kwargs)
            destination=root/'replay'; destination.mkdir()
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024**2)}), \
                    patch.object(api,'_read',new=no_worker):
                with self.assertRaises(api.ReportingResourceLimit):
                    replay._replay_benchmark(entry,destination,root,{'rtol':1e-10,'atol':1e-12},config)
            self.assertFalse((destination/'remeasured-workers.json').exists())


class CompleteComparatorCapacity(unittest.TestCase):
    """Production-shaped metadata and complete pair lifetime, without science."""
    def setUp(self):
        available=patch('prepared_learning.pilot._available_memory',return_value=20_000_000_000)
        available.start(); self.addCleanup(available.stop)

    def inventory(self, root, *, blocks=36, assets=1082, key_prefix='replication-', padding=0):
        from prepared_learning.runner import IDENTITY_FIELDS
        import hashlib
        stores=[]
        for side in ('original-','clean-replay-'):
            store=root/(side+'long-source-and-environment-path-'*4); stores.append(store)
            store.mkdir()
            (store/'run.json').write_text(json.dumps({key:'same-fixture-identity' for key in IDENTITY_FIELDS}))
            for index in range(blocks):
                block=store/'blocks'/hashlib.sha256(str(index).encode()).hexdigest(); block.mkdir(parents=True)
                (block/'commit.json').write_text(json.dumps({'key':key_prefix+str(index),'sha256':'0'*64,
                                                            'padding':'p'*padding}))
                (block/'records.json').write_text('[]')
        original_glob=Path.glob
        def virtual_assets(path, pattern, **kwargs):
            if path.name=='assets' and pattern=='*.npy':
                return (path/(f'asset-{index:04d}-'+'a'*64+'.npy') for index in range(assets))
            return original_glob(path,pattern,**kwargs)
        return stores,virtual_assets

    def scale(self):
        from prepared_learning import reporting as api
        # M10 complete raw block and longer-path complete J36 inventories.
        return {**api._empty_scale(),'files_scanned':80_000,'json_bytes':650_000_000,
                'max_json_bytes':14_417_382,'json_working_bytes':14_417_382,
                'semantic_json_bytes':650_000_000,'max_semantic_json_bytes':14_417_382,
                'inventory_bytes':800_000_000,'max_array_bytes':416,'array_bytes':18_000_000}

    def test_complete_j36_maps_and_full_pair_fit_without_retaining_raw_block(self):
        self.complete_pair_case(oversized=False)

    def test_oversized_actual_pair_stops_before_its_decode(self):
        self.complete_pair_case(oversized=True)

    def complete_pair_case(self, *, oversized):
        import gc,os,weakref,sqlite3
        from types import SimpleNamespace
        from prepared_learning import reporting as api
        class PairReached(Exception): pass
        class TrackedList(list): pass
        class TrackedDict(dict): pass
        rows_refs=[]; phase_evidence=[]; pair_decodes=[]; plans=[]
        scalar='🔬'*3_000_000 if oversized else 'x'*13_106_710
        scientific_row={'kind':'execution','outer_replication':0,'result':{'unknown_complete_payload':scalar}}
        class Index:
            def __init__(self): self.rows=[]
            def execute(self, sql, args=()):
                if sql.startswith('INSERT INTO records'): self.rows.append(args)
                if sql.startswith('SELECT side, max(length(payload))'):
                    return SimpleNamespace(fetchall=lambda:[(side,max(len(r[2]) for r in self.rows if r[0]==side),
                                                                  max(len(r[1]) for r in self.rows if r[0]==side)) for side in (0,1)])
                if sql.startswith('SELECT unit,payload'):
                    gc.collect()
                    self_test.assertTrue(all(ref() is None for ref in rows_refs),'raw block/projection retained into pair fetch')
                    return iter((r[1],r[2]) for r in self.rows if r[0]==args[0])
                return self
            def commit(self): pass
            def close(self): self.rows.clear()
        self_test=self; guard=api._resource_check; paths=api._guard_paths; read=api._read; project=api._scientific
        decode=api.json.loads; compare=api.compare_scientific; map_plan=api._comparison_map_plan
        def tracked_read(path,**kwargs):
            value=read(path,**kwargs)
            if Path(path).name=='records.json':
                value=TrackedList(TrackedDict(row) for row in value)
                rows_refs.extend(weakref.ref(row) for row in value); rows_refs.append(weakref.ref(value))
            return value
        def tracked_project(value,*args,**kwargs):
            result=project(value,*args,**kwargs)
            if isinstance(result,dict):
                result=TrackedDict(result); rows_refs.append(weakref.ref(result))
            return result
        def guarded(scale,**kwargs):
            result=guard(scale,**kwargs); phase_evidence.append(result); return result
        def fixture_paths(values,**kwargs):
            if kwargs['phase'] in ('run_comparison_before_open','run_comparison_after_recovery'):
                return guarded(self.scale(),phase=kwargs['phase'],object_bytes=kwargs['object_bytes'])
            return paths(values,**kwargs)
        def planned(*args,**kwargs):
            result=map_plan(*args,**kwargs); plans.append(result); return result
        def tracked_decode(value,*args,**kwargs):
            if isinstance(value,str) and 'unknown_complete_payload' in value and not value.startswith('['):
                pair_decodes.append(len(value))
            return decode(value,*args,**kwargs)
        def compared(left,right,**kwargs):
            compare(left,right,**kwargs)
            if isinstance(left,dict) and 'unknown_complete_payload' in left.get('result',{}): raise PairReached()
        with tempfile.TemporaryDirectory() as tmp:
            stores,assets=self.inventory(Path(tmp))
            for store in stores:
                first=sorted((store/'blocks').iterdir())[0]
                (first/'records.json').write_text(json.dumps([scientific_row],ensure_ascii=False))
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':'7500000000'}), \
                    patch.object(Path,'glob',new=assets),patch.object(api,'_guard_paths',new=fixture_paths), \
                    patch.object(api,'_comparison_map_plan',new=planned), \
                    patch('shutil.disk_usage',return_value=SimpleNamespace(free=100_000_000_000)), \
                    patch.object(api,'_resource_check',new=guarded),patch.object(api,'_read',new=tracked_read), \
                    patch.object(api,'_scientific',new=tracked_project),patch.object(api.json,'loads',new=tracked_decode), \
                    patch.object(api,'compare_scientific',new=compared),patch.object(sqlite3,'connect',return_value=Index()), \
                    patch.object(api.RunStore,'open',return_value=SimpleNamespace(_verify=lambda block:None)):
                if oversized:
                    with self.assertRaises(api.ReportingResourceLimit) as caught:
                        api.compare_run_stores(*stores,rtol=1e-10,atol=1e-12,retained_bytes=21_721_574)
                    self.assertIn('record_pair',caught.exception.evidence['phase'])
                    self.assertEqual(pair_decodes,[],'oversized actual pair decoded')
                    gc.collect(); self.assertTrue(all(ref() is None for ref in rows_refs))
                else:
                    try:
                        with self.assertRaises(PairReached):
                            api.compare_run_stores(*stores,rtol=1e-10,atol=1e-12,retained_bytes=21_721_574)
                    except api.ReportingResourceLimit as exc:
                        self.fail('complete planned J36 maps/pair refused: '+str(exc.evidence))
                    pair=next(e for e in reversed(phase_evidence) if 'record_pair' in e['phase'])
                    self.assertEqual(pair['retained_copies'],1)
                    self.assertEqual(pair['policy']['json_expansion_factor'],256)
                    self.assertEqual(pair['inventory_bytes'],0,'released raw inventory charged to pair')
                    self.assertEqual(pair['json_working_bytes'],max(pair_decodes))
                    self.assertLess(pair['estimated_peak_bytes'],7_500_000_000)
                    self.assertEqual(len(pair_decodes),2)
                    self.assertEqual(plans[0]['array_counts'],[38_952,38_952])
                    self.capacity_evidence={'pair':pair,'phases':phase_evidence,'native_map_projection':plans[0]}

    def test_native_projection_covers_unicode_maps_sets_hashes_and_ignores_padding(self):
        import os,sys,hashlib
        from prepared_learning import reporting as api
        self.assertTrue(hasattr(api,'_comparison_map_plan'),'complete native container projection is missing')
        def actual_size(value,seen=None):
            seen=set() if seen is None else seen
            if id(value) in seen: return 0
            seen.add(id(value)); total=sys.getsizeof(value)
            if isinstance(value,dict):
                return total+sum(actual_size(k,seen)+actual_size(v,seen) for k,v in value.items())
            if isinstance(value,(list,tuple,set)): return total+sum(actual_size(v,seen) for v in value)
            return total
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/('unicode-κ🔬'*8); root.mkdir()
            stores,assets=self.inventory(root,blocks=2,assets=19,key_prefix='κ🔬'*128)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':'7500000000'}),patch.object(Path,'glob',new=assets):
                plan=api._comparison_map_plan(*stores,scale=self.scale(),retained_bytes=21_721_574)
                maps=[]
                for store in stores:
                    mapping={}
                    for block in (store/'blocks').iterdir():
                        key=json.loads((block/'commit.json').read_text())['key']
                        for asset in assets(block/'assets','*.npy'): mapping[key+'/'+asset.name]=str(asset)
                    maps.append(mapping)
                hashes={key:hashlib.sha256(key.encode()).hexdigest() for key in maps[0]}
                complete=(maps,set(maps[0]),set(maps[1]),hashes)
                self.assertGreaterEqual(plan['native_bytes'],actual_size(complete))
                self.assertEqual(plan['array_counts'],[38,38])
                for store in stores:
                    for block in (store/'blocks').iterdir():
                        marker=json.loads((block/'commit.json').read_text()); marker['irrelevant_padding']='x'*(256*1024)
                        (block/'commit.json').write_text(json.dumps(marker))
                padded=api._comparison_map_plan(*stores,scale=self.scale(),retained_bytes=21_721_574)
                self.assertEqual(plan['native_bytes'],padded['native_bytes'],'unretained marker padding multiplied per asset')

    def test_actual_long_unicode_key_growth_refuses_before_full_maps(self):
        import os
        from prepared_learning import reporting as api
        with tempfile.TemporaryDirectory() as tmp:
            stores,assets=self.inventory(Path(tmp),blocks=2,assets=1082,key_prefix='🔬'*32768)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':'7500000000'}), \
                    patch.object(Path,'glob',new=assets), \
                    patch.object(api,'_comparison_container_sizes',side_effect=AssertionError('large maps/probes allocated before growth guard')):
                with self.assertRaises(api.ReportingResourceLimit) as caught:
                    api._comparison_map_plan(*stores,scale=self.scale(),retained_bytes=21_721_574)
            self.assertEqual(caught.exception.evidence['phase'],'run_comparison_map_metadata_growth')
            self.assertGreater(caught.exception.evidence['estimated_peak_bytes'],7_500_000_000)

    def test_marker_native_guard_precedes_metadata_decode(self):
        import os
        from prepared_learning import reporting as api
        with tempfile.TemporaryDirectory() as tmp:
            stores,assets=self.inventory(Path(tmp),blocks=1,assets=1)
            marker=next((stores[0]/'blocks').iterdir())/'commit.json'
            with marker.open('wb') as stream: stream.truncate(4*1024**2)
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024**2)}), \
                    patch.object(Path,'glob',new=assets), \
                    patch.object(api.json,'loads',side_effect=AssertionError('oversized metadata decoded')):
                with self.assertRaises(api.ReportingResourceLimit) as caught:
                    api._comparison_map_plan(*stores,scale=self.scale(),retained_bytes=0)
            self.assertEqual(caught.exception.evidence['phase'],'json_before_parse')
            self.assertEqual(caught.exception.evidence['policy']['json_expansion_factor'],64)


class ComparatorRecoveryAdmission(unittest.TestCase):
    def setUp(self):
        available=patch('prepared_learning.pilot._available_memory',return_value=4*1024**3)
        available.start(); self.addCleanup(available.stop)

    def store(self, root):
        from prepared_learning.runner import RunStore
        from tests.test_runner import RUN,task_row
        store=RunStore.create(root,RUN)
        store.commit_block('x',[task_row('a'),task_row('b')],{})
        return store

    def recovery_case(self, state):
        from prepared_learning import reporting as api
        with tempfile.TemporaryDirectory() as tmp:
            store=self.store(Path(tmp)); block=next((store.path/'blocks').iterdir())
            rows=(block/'records.json').read_bytes(); journal=next((store.path/'journal').iterdir())
            journal_bytes=journal.read_bytes(); marker=block/'commit.json'
            orphan=store.path/'blocks'/'.tmp-interrupted-fixture'
            if state=='missing_secondary_marker': marker.unlink()
            else: orphan.mkdir()
            try: result=api.compare_run_stores(store.path,store.path,rtol=1e-10,atol=1e-12)
            except FileNotFoundError as exc: self.fail('comparator bypassed authoritative recovery: '+str(exc))
            self.assertEqual(result['status'],'matched')
            self.assertEqual(result['scientific_records_compared'],2)
            self.assertEqual(result['arrays_compared'],0)
            self.assertEqual((block/'records.json').read_bytes(),rows)
            self.assertEqual(journal.read_bytes(),journal_bytes)
            self.assertEqual(marker.read_bytes(),journal_bytes)
            self.assertFalse(orphan.exists())

    def test_journal_backed_missing_secondary_marker_recovers_before_projection(self):
        self.recovery_case('missing_secondary_marker')

    def test_unpublished_orphan_recovers_before_projection(self):
        self.recovery_case('unpublished_orphan')

    def test_corrupt_authority_and_committed_evidence_are_not_recovered(self):
        from prepared_learning import reporting as api
        from prepared_learning.runner import _bytes
        import hashlib
        for state in ('journal','marker','records','missing_journal','identity'):
            with self.subTest(state=state),tempfile.TemporaryDirectory() as tmp:
                store=self.store(Path(tmp)); block=next((store.path/'blocks').iterdir())
                journal=next((store.path/'journal').iterdir())
                if state=='journal': journal.write_text('{invalid')
                elif state=='marker': (block/'commit.json').write_text('{}')
                elif state=='records': (block/'records.json').write_text('[]')
                elif state=='missing_journal': journal.unlink()
                else:
                    run=json.loads((store.path/'run.json').read_text()); run['git_commit']='b'*40
                    encoded=_bytes(run); (store.path/'run.json').write_bytes(encoded)
                    (store.path/'run.sha256').write_text(hashlib.sha256(encoded).hexdigest())
                with self.assertRaises(ValueError):
                    api.compare_run_stores(store.path,store.path,rtol=1e-10,atol=1e-12)
                self.assertTrue(block.exists(),'corrupt committed evidence was silently discarded')
                if state=='missing_journal': self.assertFalse(journal.exists())

    def test_initial_caller_guard_precedes_any_recovery_or_mutation(self):
        from prepared_learning import reporting as api
        from prepared_learning.runner import RunStore
        import os
        with tempfile.TemporaryDirectory() as tmp:
            store=self.store(Path(tmp)); marker=next(store.path.glob('blocks/*/commit.json')); marker.unlink()
            orphan=store.path/'blocks'/'.tmp-interrupted-fixture'; orphan.mkdir()
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(128*1024**2)}), \
                    patch.object(RunStore,'open',side_effect=AssertionError('recovery opened before caller admission')):
                with self.assertRaises(api.ReportingResourceLimit):
                    api.compare_run_stores(store.path,store.path,rtol=1e-10,atol=1e-12,retained_bytes=8*1024**2)
            self.assertFalse(marker.exists()); self.assertTrue(orphan.exists())

    def test_recovery_internal_guard_precedes_its_json_reads(self):
        from prepared_learning import reporting as api
        from prepared_learning.runner import RunStore
        import os
        with tempfile.TemporaryDirectory() as tmp:
            store=self.store(Path(tmp)); marker=next(store.path.glob('blocks/*/commit.json')); marker.unlink()
            initial=api._guard_paths((path for _ in (0,1) for path in api._store_paths(store.path)),
                                     phase='recovery_test_initial',streaming=True)
            # The initial complete input estimate fits; the separately charged
            # RunStore recovery workspace does not fit this extra512KiB margin.
            cap=initial['estimated_peak_bytes']+512*1024
            with patch.dict(os.environ,{'PREPARED_REPORT_MEMORY_BYTES':str(cap)}), \
                    patch.object(RunStore,'_read_json',side_effect=AssertionError('recovery decoded before internal admission')):
                try:
                    with self.assertRaises(api.ReportingResourceLimit) as caught:
                        api.compare_run_stores(store.path,store.path,rtol=1e-10,atol=1e-12)
                except FileNotFoundError as exc: self.fail('missing admitted recovery phase: '+str(exc))
            self.assertEqual(caught.exception.evidence['phase'],'run_comparison_recovery_before_allocation')
            self.assertGreater(caught.exception.evidence['scalar_and_key_working_bytes'],1024**2)
            self.assertFalse(marker.exists())

    def test_cross_store_identity_mismatch_precedes_recovery(self):
        from prepared_learning import reporting as api
        from prepared_learning.runner import RunStore
        from tests.test_runner import RUN
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); left=self.store(root/'left')
            right=RunStore.create(root/'right',{**RUN,'git_commit':'b'*40})
            with patch.object(RunStore,'open',side_effect=AssertionError('mismatched source entered recovery')):
                with self.assertRaisesRegex(ValueError,'identity'):
                    api.compare_run_stores(left.path,right.path,rtol=1e-10,atol=1e-12)
