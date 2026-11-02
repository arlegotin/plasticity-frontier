"""Actual producer regressions for final report/replay handoffs; fixture only."""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from prepared_learning import reporting
from prepared_learning.cli import main


class FinalizedBenchmarkHandoff(unittest.TestCase):
    def test_real_finalized_workers_complete_and_reject_schedule_identity_changes(self):
        from prepared_learning import benchmark
        from prepared_learning.methods import make_method
        config = {'schema_version': 1, 'workflow': 'generated', 'phase': 'exploratory',
                  'dimension': 2, 'context_size': 2, 'seed': 5,
                  'methods': ['B01', 'B05:B01', 'B08:ridge', 'B10:empirical_bayes'],
                  'source_training_tasks': 2, 'source_validation_tasks': 2,
                  'pilot_blocks': 1, 'blocks': 4, 'max_repetitions': 1,
                  'target_block_ns': 1, 'target_relative_error': .99,
                  'fitting_candidates': {'gammas': [0.], 'scalar_fractions': [.5],
                                         'ridge_multipliers': [1.], 'max_refinement_rounds': 0}}
        identity = {'versions': {'git_commit': '0'*40}, 'dirty_diff_sha256': None,
                    'runtime': {'fixture': 'test only'}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(benchmark, '_benchmark_identity', return_value=identity):
            root = Path(tmp)
            methods = [make_method(mid, {}) for mid in config['methods']]
            benchmark.benchmark_methods(methods, [], config, root/'calibration')
            frozen = benchmark.finalize_benchmark_manifest(root/'calibration/benchmark.json', root/'frozen.json', root/'campaign')
            benchmark.benchmark_methods(methods, [], frozen, root/'campaign')
            entry = {'input_id': 0, 'label': 'real workers, fixture identity', 'kind': 'benchmark',
                     'run_path': str(root/'campaign'), 'source': identity['versions'], 'fixture': True,
                     'descriptor_path': str(root/'input.json')}
            (root/'input.json').write_text(json.dumps(entry))
            result = reporting._benchmark(entry,output=root/'report')
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['evidence_level'], 'fixture')
            block = next((root/'campaign/blocks').glob('*.json'))
            original = json.loads(block.read_text())
            missing = copy.deepcopy(original)
            missing[0]['rows'].pop()
            block.write_text(json.dumps(missing))
            self.assertEqual(reporting._benchmark(entry,output=root/'report')['status'], 'failed_or_incomplete')
            mutations = (
                lambda w: w['rows'].append(copy.deepcopy(w['rows'][0])),
                lambda w: w['rows'][0].update(timing_stratum='changed'),
                lambda w: w['rows'][0].update(method_id='B04'),
                lambda w: w['rows'][0].update(execution_hash='changed'),
                lambda w: w['rows'][0].update(selected_execution_hash='changed'),
                lambda w: w['rows'][0].update(method_order=list(reversed(w['rows'][0]['method_order']))),
                lambda w: w['conditions'].update(thermal='changed'),
                lambda w: w['rows'][0].update(repetitions=999),
                lambda w: w['rows'][0].update(block=999),
                lambda w: w['resource']['memory'].update(isolation='in_process_test_or_callback'),
                lambda w: w['resource'].update(selected_execution_hash='changed'),
            )
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    changed = copy.deepcopy(original)
                    mutate(changed[0])
                    block.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        reporting._benchmark(entry,output=root/'report')
            block.write_text(json.dumps(original))
            controls=root/'campaign/inputs.json'; control_original=json.loads(controls.read_text())
            for mutate in (
                lambda v:v['inputs']['manifest']['frozen']['identity'].update(dirty_diff_sha256='dirty'),
                lambda v:v['inputs']['manifest']['frozen']['identity'].update(versions={'git_commit':'1'*40}),
                lambda v:v['inputs']['manifest']['frozen'].update(configuration_sha256='changed'),
                lambda v:v['inputs']['manifest']['frozen'].update(calibration_sha256='changed'),
                lambda v:v['inputs']['manifest']['frozen'].update(repetitions={mid:999 for mid in config['methods']}),
            ):
                changed=copy.deepcopy(control_original); mutate(changed); controls.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): reporting._benchmark(entry,output=root/'report')
            controls.write_text(json.dumps(control_original))
            pilot_path=next((root/'calibration/blocks').glob('*.json')); pilot_original=json.loads(pilot_path.read_text())
            changed=copy.deepcopy(pilot_original); changed[0]['pilot']['samples_ns']=[-1]
            pilot_path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): reporting._benchmark(entry,output=root/'report')
            pilot_path.write_text(json.dumps(pilot_original))
            # Missing direct/ridge/EB controls remain incomplete/exploratory.
            all_blocks={p:json.loads(p.read_text()) for p in (root/'campaign/blocks').glob('*.json')}
            for missing_mid in ('B05:B01','B08:ridge','B10:empirical_bayes'):
                for path,workers in all_blocks.items(): path.write_text(json.dumps([w for w in workers if w['resource']['method_id']!=missing_mid]))
                result=reporting._benchmark(entry,output=root/'report')
                self.assertEqual(result['status'],'failed_or_incomplete'); self.assertIn(missing_mid,result['missing_controls'])
            for path,workers in all_blocks.items(): path.write_text(json.dumps(workers))


class OptionalFreezeGrid(unittest.TestCase):
    def test_real_fixed_mode_pilot_freezes_without_optional_grid(self):
        from dataclasses import replace
        from prepared_learning import pilot, protocol
        from tests.test_confirmation import tiny
        for mode in ('deterministic_frontier', 'fixed_prior_task_simulation'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                config, _ = tiny(mode)
                config = replace(config, methods=('B01', 'B02', 'B04', 'B05:B04'),
                                 numerics={'dtype': 'float64', 'method_options': {},
                                           'pilot': {'blocks': 2, 'replicates_per_block': 3,
                                                     'procedure': 'descriptive_only', 'max_seconds': 1e6}})
                actual = protocol.checkout_identity
                def clean(*args, **kwargs):
                    return actual(*args, **kwargs)[0], None
                with patch.object(protocol, 'checkout_identity', side_effect=clean):
                    evidence = pilot.run_pilot(config, Path(tmp)/'pilot')
                    path = Path(evidence['run_path'])/'protocol_draft.json'
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        code = main(['freeze', '--protocol', str(path), '--output', str(Path(tmp)/'frozen.json')])
                self.assertEqual(code, 0, stdout.getvalue())
                frozen = json.loads((Path(tmp)/'frozen.json').read_text())
                protocol.validate_frozen_protocol(frozen)
                self.assertNotIn('fitting_candidates', frozen['scientific']['config']['numerics'])


class OutsideClassMetadata(unittest.TestCase):
    def test_b08_actual_depth_and_reference_caps_survive_confirmation_report(self):
        from prepared_learning import confirmation
        from prepared_learning.records import content_hash
        from tests.test_confirmation import execution_manifest
        manifest = execution_manifest('fixed_prior_task_simulation')
        science = manifest['scientific']
        science['config']['methods'] = ['B08:gd', 'B01']
        science['selectors'] = {'baseline': ['B08:gd'], 'joint': ['B01']}
        # Zero norm cap deliberately does not restrict the depth-one GD step.
        science['config']['budget']['initialization_norm_cap'] = 0.
        manifest['scientific_sha256'] = content_hash(science)
        manifest['protocol_sha256'] = content_hash({'fixture': science})
        with tempfile.TemporaryDirectory() as tmp, patch.object(confirmation, '_live_identity'):
            root = Path(tmp)
            result = confirmation.run_confirmation(manifest, root/'run')
            self.assertNotEqual(result.code, 'IMPLEMENTATION_INVALID')
            protocol = root/'protocol.json'
            protocol.write_text(json.dumps(manifest))
            entry = {'input_id': 0, 'label': 'B08 fixture', 'kind': 'confirmation',
                     'run_path': result.evidence['run_path'], 'protocol_path': str(protocol),
                     'fixture': True, 'evidence_level': 'fixture', 'descriptor_path': str(root/'input.json')}
            (root/'input.json').write_text(json.dumps(entry))
            records = []
            comparison, _ = reporting._confirmation(entry, records.append)
            rows = [r for r in records if r['kind']=='task_metrics' and r['method_id']=='baseline']
            self.assertTrue(rows)
            self.assertEqual({r['depth'] for r in rows}, {1})
            meta = rows[0]['execution_metadata']
            self.assertEqual(meta['scope'], 'outside_main_class_operational_control')
            self.assertFalse(meta['fixed_depth_eligible'])
            self.assertTrue(meta['reference_cap_fit']['fits_norm_cap'])
            self.assertTrue(meta['reference_cap_fit']['fits_parameter_cap'])
            self.assertFalse(meta['reference_cap_fit']['applies_as_eligibility'])
            self.assertEqual(meta['reference_budget']['depth'], 2)
            self.assertEqual(comparison['role_execution_metadata']['baseline'], [meta])
            self.assertEqual(comparison['architecture']['executed_roles']['baseline'], [meta])
            self.assertIn('outside-class', comparison['architecture']['class'])


class ExactExportHandoff(unittest.TestCase):
    def export(self, config, branch='F'):
        run = subprocess.run([sys.executable, '-m', 'prepared_learning', 'export-exact',
                              '--config', str(config), '--branch', branch], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr+run.stdout)
        return json.loads(run.stdout)

    def test_public_exact_export_admits_both_branches_and_replays(self):
        from prepared_learning._replay import worker
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root = Path(tmp).resolve()
            config = Path('configs/exploration/frontier.json').resolve()
            descriptors = []
            for branch in ('F', 'Z'):
                raw = self.export(config, branch)
                self.assertEqual(raw['certificate_level'], 'exact_rational')
                self.assertTrue(raw['verification']['verified'])
                self.assertIn('rational_qp', raw)
                path = root/(branch+'.json')
                path.write_text(json.dumps(raw))
                descriptor = root/(branch+'-input.json')
                descriptor.write_text(json.dumps({'kind': 'exact', 'raw_path': str(path), 'fixture': True,
                    'command': ['export-exact', '--config', str(config), '--branch', branch],
                    'assets': [str(config), str(Path('tests/fixtures/exact_cases.json').resolve())]}))
                descriptors.append(descriptor)
            manifest = reporting.build_report_inputs(descriptors)
            for index, branch in enumerate(('F', 'Z')):
                raw = json.loads((root/(branch+'.json')).read_text())
                manifest['claims'].append({'claim_id': branch, 'statement': 'exact_rational_fixture_kkt_verified',
                    'evidence_level': 'rigorous_certificate', 'input_ids': [index], **raw['contract']})
            ledger = reporting._claims(manifest, {'comparisons': [], 'benchmarks': [], 'statuses': []})
            self.assertEqual(len(ledger), 2)
            # Only committed-source admission is substituted in this precommit fixture.
            # The command, rational verifier, asset inventory and comparator are real.
            with patch('prepared_learning._replay._validate_source'):
                worker({'manifest': manifest, 'output': str(root/'replay')})
            self.assertTrue(all(r['status']=='matched' for r in json.loads((root/'replay/replay-worker.json').read_text())))
            original = json.loads((root/'F.json').read_text())
            for field in ('rational_qp', 'contract', 'configuration'):
                changed = copy.deepcopy(original)
                if field=='rational_qp':
                    # A changed constant still has valid KKT: verification alone is insufficient.
                    changed[field]['constant'] = [2, 1]
                elif field=='contract':
                    changed[field]['architecture']['depth'] = 3
                else:
                    changed[field]['n'] = 5
                (root/'F.json').write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    reporting._claims(manifest, {'comparisons': [], 'benchmarks': [], 'statuses': []})
            (root/'F.json').write_text(json.dumps(original))
            for missing in (config, Path('tests/fixtures/exact_cases.json').resolve()):
                changed = copy.deepcopy(manifest)
                changed['inputs'][0]['assets'].remove(str(missing))
                with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'inventory'):
                    reporting._claims(changed, {'comparisons': [], 'benchmarks': [], 'statuses': []})

    def test_export_rejects_configuration_and_fixture_mutation(self):
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root = Path(tmp)
            config = json.loads(Path('configs/exploration/frontier.json').read_text())
            config['n'] = 5
            path = root/'wrong.json'
            path.write_text(json.dumps(config))
            run = subprocess.run([sys.executable, '-m', 'prepared_learning', 'export-exact',
                                  '--config', str(path), '--branch', 'F'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertIn('configuration', json.loads(run.stdout)['error'])
            config['n'] = 4
            fixture = json.loads(Path('tests/fixtures/exact_cases.json').read_text())
            fixture['primary_richer_k_3_5']['n'] = 5
            altered = root/'fixture.json'
            altered.write_text(json.dumps(fixture))
            config['exact_fixture_path'] = str(altered)
            path.write_text(json.dumps(config))
            run = subprocess.run([sys.executable, '-m', 'prepared_learning', 'export-exact',
                                  '--config', str(path), '--branch', 'F'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertIn('fixture', json.loads(run.stdout)['error'])

    def test_float_relabeling_cannot_pass_exact_claim_or_replay_admission(self):
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root=Path(tmp).resolve()
            config=Path('configs/exploration/frontier.json').resolve()
            raw=self.export(config)
            raw['rational_qp']['constant'][0]=float(raw['rational_qp']['constant'][0])
            path=root/'raw.json'
            path.write_text(json.dumps(raw))
            descriptor=root/'input.json'
            descriptor.write_text(json.dumps({'kind':'exact','raw_path':str(path),'fixture':True,
                'command':['export-exact','--config',str(config),'--branch','F'],
                'assets':[str(config),str(Path('tests/fixtures/exact_cases.json').resolve())]}))
            manifest=reporting.build_report_inputs([descriptor])
            manifest['claims']=[{'claim_id':'float-relabel','statement':'exact_rational_fixture_kkt_verified',
                'evidence_level':'rigorous_certificate','input_ids':[0],**raw['contract']}]
            with self.assertRaisesRegex(ValueError,'integer|exact'):
                reporting._claims(manifest,{'comparisons':[],'benchmarks':[],'statuses':[]})
            with self.assertRaisesRegex(ValueError,'integer|exact'):
                reporting._verify_inputs(manifest,replay=True)


class FullArtifactReplay(unittest.TestCase):
    def producer(self, root, kind, *, failed=False, output_override=False):
        from prepared_learning.robustness import make_manifest
        from prepared_learning.specs import load_config
        if kind=='robustness':
            config = make_manifest(load_config(Path('configs/exploration/learned_prior.json')),
                                   ['student3', 'extreme_scale', 'resume_error'])
            input_flag, filename = '--manifest', 'robustness.json'
        else:
            config = {'schema_version': 1, 'spectrum': [.6, .4], 'n': 3, 'sigma2': 1.,
                      'depth': 1 if failed else 2, 'initialization_norm_cap': .2,
                      'trainable_parameter_cap': 6,
                      'grid': {'widths': [1, 2], 'norm_caps': [0., .2],
                               'risk_tolerance': 1., 'budget_tolerance': 1., 'output': str(root/'original')}}
            input_flag, filename = '--config', 'frontier.json'
        path = root/'config.json'
        path.write_text(json.dumps(config))
        command = [kind, input_flag, str(path)]
        if kind=='robustness' or output_override:
            command += ['--output', str(root/'original')]
        process = root/'process.json'
        run = subprocess.run([sys.executable, '-m', 'prepared_learning', *command,
                              '--process-record', str(process)], capture_output=True, text=True)
        self.assertTrue(process.is_file(), run.stderr+run.stdout)
        self.assertEqual(json.loads(process.read_text()), {'command': command, 'returncode': run.returncode,
                                                         'stdout': run.stdout, 'stderr': run.stderr})
        stdout = json.loads(run.stdout)
        actual = (Path(stdout['evidence']['report']) if kind=='robustness'
                  else Path(stdout['attempt_path'])/'frontier.json')
        descriptor = root/'input.json'
        descriptor.write_text(json.dumps({'kind': 'diagnostic', 'raw_path': str(actual), 'fixture': True,
            'command': command, 'assets': [str(path)], 'process_path': str(process),
            'output_artifact': {'filename': filename}}))
        return reporting.build_report_inputs([descriptor]), actual

    def test_real_robustness_and_grid_full_records_replay_without_overwrite(self):
        from prepared_learning._replay import worker
        for kind, failed in (('robustness', False), ('frontier', False), ('frontier', True)):
            with self.subTest(kind=kind, failed=failed), tempfile.TemporaryDirectory(dir='runs') as tmp:
                root = Path(tmp).resolve()
                manifest, actual = self.producer(root, kind, failed=failed)
                original = actual.read_bytes()
                expected = json.loads(original)
                if kind=='robustness':
                    self.assertFalse(expected['confirmation_performed'])
                    self.assertEqual(len(expected['arms']), 3)
                    self.assertEqual(len(expected['arms'][1]['outcomes']), 10)
                    self.assertEqual(expected['arms'][2]['failed_rows_retained'], 1)
                elif failed:
                    self.assertTrue(any(r.get('status')=='failed' for r in expected['records']))
                else:
                    self.assertTrue(all('Z' in r and 'F' in r for r in expected['records'] if r['kind']=='point'))
                with patch('prepared_learning._replay._validate_source'):
                    worker({'manifest': manifest, 'output': str(root/'replay')})
                self.assertEqual(actual.read_bytes(), original)
                receipt = json.loads((root/'replay/replay-worker.json').read_text())[0]
                self.assertEqual(receipt['status'], 'matched')
                fresh = Path(receipt['artifact_path'])
                self.assertTrue(fresh.is_relative_to(root/'replay'))
                reporting.compare_scientific(reporting._scientific(expected),
                    reporting._scientific(json.loads(fresh.read_text())), rtol=1e-10, atol=1e-12)
                self.assertEqual(receipt['returncode'], 1 if failed else 0)
                self.assertTrue((root/'replay/0/process.json').is_file())

    def test_artifact_selector_rejects_escape_missing_stale_and_changed_records(self):
        from prepared_learning._replay import worker
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root = Path(tmp).resolve()
            manifest, original = self.producer(root, 'frontier')
            original_bytes = original.read_bytes()
            for selector in ('../frontier.json', '/tmp/frontier.json', 'absent.json'):
                bad = copy.deepcopy(manifest)
                bad['inputs'][0]['output_artifact']['filename'] = selector
                from prepared_learning.records import content_hash
                bad['input_order_sha256'] = content_hash(bad['inputs'])
                with self.subTest(selector=selector), patch('prepared_learning._replay._validate_source'):
                    with self.assertRaises(ValueError):
                        worker({'manifest': bad, 'output': str(root/('bad-'+str(len(selector))))})
            # A successful summary with no newly written file must not reuse original raw.
            entry = manifest['inputs'][0]
            recorded = json.loads(Path(entry['process_path']).read_text())
            stale = subprocess.CompletedProcess([], recorded['returncode'], recorded['stdout'], recorded['stderr'])
            real_run = subprocess.run
            def no_write(command, **kwargs):
                return stale if command[1:3]==['-m', 'prepared_learning'] else real_run(command, **kwargs)
            with patch('prepared_learning._replay._validate_source'), patch('prepared_learning._replay.subprocess.run', side_effect=no_write):
                with self.assertRaisesRegex(ValueError, 'fresh|artifact|output'):
                    worker({'manifest': manifest, 'output': str(root/'stale')})
            # Inventoried scientific records cannot be altered after capture.
            changed = json.loads(original_bytes)
            changed['records'][0]['F']['objective'] += .1
            original.write_text(json.dumps(changed))
            with patch('prepared_learning._replay._validate_source'):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    worker({'manifest': manifest, 'output': str(root/'changed')})
            original.write_bytes(original_bytes)

    def test_fresh_artifact_missing_changed_or_symlinked_is_rejected(self):
        from prepared_learning._replay import worker
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root = Path(tmp).resolve()
            manifest, original = self.producer(root, 'frontier')
            original_bytes = original.read_bytes()
            real_run = subprocess.run
            for mutation in ('missing', 'changed', 'symlink', 'exit_status'):
                def run_then_change(command, **kwargs):
                    run = real_run(command, **kwargs)
                    if command[1:3]==['-m', 'prepared_learning']:
                        fresh = Path(json.loads(run.stdout)['attempt_path'])/'frontier.json'
                        if mutation=='missing':
                            fresh.unlink()
                        elif mutation=='symlink':
                            fresh.unlink()
                            fresh.symlink_to(original)
                        elif mutation=='changed':
                            raw = json.loads(fresh.read_text())
                            raw['records'][0]['F']['objective'] += .1
                            fresh.write_text(json.dumps(raw))
                        else:
                            run.returncode = 1
                    return run
                with self.subTest(mutation=mutation), patch('prepared_learning._replay._validate_source'), \
                        patch('prepared_learning._replay.subprocess.run', side_effect=run_then_change):
                    with self.assertRaises(ValueError):
                        worker({'manifest': manifest, 'output': str(root/mutation)})
                self.assertEqual(original.read_bytes(), original_bytes)

    def test_process_receipt_is_immutable_and_inventory_bound(self):
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root=Path(tmp).resolve()
            manifest,original=self.producer(root,'frontier')
            entry=manifest['inputs'][0]
            process=Path(entry['process_path'])
            before=process.read_bytes()
            attempts=list((root/'original').iterdir())
            run=subprocess.run([sys.executable,'-m','prepared_learning',*entry['command'],
                                '--process-record',str(process)],capture_output=True,text=True)
            self.assertEqual(run.returncode,1)
            self.assertEqual(process.read_bytes(),before)
            self.assertEqual(list((root/'original').iterdir()),attempts)
            process.write_text('{}')
            with self.assertRaisesRegex(ValueError,'inventory'):
                reporting._verify_inputs(manifest,replay=True)

    def test_file_producer_without_selector_is_refused_before_execution(self):
        from prepared_learning._replay import worker
        from prepared_learning.records import content_hash
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root=Path(tmp).resolve()
            manifest,_=self.producer(root,'frontier')
            del manifest['inputs'][0]['output_artifact']
            manifest['input_order_sha256']=content_hash(manifest['inputs'])
            attempts=list((root/'original').iterdir())
            with patch('prepared_learning._replay._validate_source'):
                with self.assertRaisesRegex(ValueError,'output_artifact'):
                    worker({'manifest':manifest,'output':str(root/'replay')})
            self.assertEqual(list((root/'original').iterdir()),attempts)

    def test_noncanonical_frontier_commands_reject_before_producer_dispatch(self):
        from prepared_learning._replay import worker
        from prepared_learning.records import content_hash
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root=Path(tmp).resolve()
            manifest,_=self.producer(root,'frontier')
            absolute=str(root/'config.json')
            relative=str((root/'config.json').relative_to(Path.cwd()))
            point='configs/exploration/frontier.json'
            variants=(
                ['frontier','--config='+absolute],
                ['frontier','--config='+relative],
                ['frontier','--conf',absolute],
                ['frontier','--conf='+relative],
                ['frontier','--config',point,'--config',absolute],
                ['frontier','--config',point,'--conf',relative],
                ['frontier','--config',point,'--config='+absolute],
                ['frontier','--certify','--config='+absolute],
                ['frontier','--certify','--config',relative],
                ['frontier','--config',point,'--cert'],
                ['frontier','--config',point,'--certify','--certify'],
                ['frontier','--output',str(root/'original'),'--config',relative],
            )
            before={str(p.relative_to(root/'original')):(p.read_bytes() if p.is_file() else None)
                    for p in (root/'original').rglob('*')}
            real_run=subprocess.run
            dispatched=[]
            def intercept(command,**kwargs):
                if command[1:3]==['-m','prepared_learning']:
                    dispatched.append(command)
                    raise ValueError('producer reached forbidden boundary')
                return real_run(command,**kwargs)
            for index,command in enumerate(variants):
                changed=copy.deepcopy(manifest)
                del changed['inputs'][0]['output_artifact']
                changed['inputs'][0]['command']=command
                changed['input_order_sha256']=content_hash(changed['inputs'])
                with self.subTest(command=command), patch('prepared_learning._replay._validate_source'), \
                        patch('prepared_learning._replay.subprocess.run',side_effect=intercept):
                    with self.assertRaisesRegex(ValueError,'canonical'):
                        worker({'manifest':changed,'output':str(root/('rejected-'+str(index)))})
                    self.assertEqual(dispatched,[])
                    self.assertEqual(before,{str(p.relative_to(root/'original')):(p.read_bytes() if p.is_file() else None)
                        for p in (root/'original').rglob('*')})

    def test_canonical_point_frontier_and_declared_output_override_still_replay(self):
        from prepared_learning._replay import worker
        for certify in (False,True):
            with self.subTest(certify=certify), tempfile.TemporaryDirectory(dir='runs') as tmp:
                root=Path(tmp).resolve()
                config=Path('configs/exploration/frontier.json')
                command=['frontier','--config',str(config)]+(['--certify'] if certify else [])
                run=subprocess.run([sys.executable,'-m','prepared_learning',*command],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr+run.stdout)
                raw=root/'raw.json';raw.write_text(run.stdout)
                descriptor=root/'input.json'
                descriptor.write_text(json.dumps({'kind':'diagnostic','fixture':True,'raw_path':str(raw),
                    'command':command,'assets':[str(config.resolve()),str(Path('tests/fixtures/exact_cases.json').resolve())]}))
                manifest=reporting.build_report_inputs([descriptor])
                with patch('prepared_learning._replay._validate_source'):
                    worker({'manifest':manifest,'output':str(root/'replay')})
                self.assertEqual(json.loads((root/'replay/replay-worker.json').read_text())[0]['status'],'matched')
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root=Path(tmp).resolve()
            manifest,original=self.producer(root,'frontier',output_override=True)
            before=original.read_bytes()
            with patch('prepared_learning._replay._validate_source'):
                worker({'manifest':manifest,'output':str(root/'replay')})
            self.assertEqual(original.read_bytes(),before)
            self.assertEqual(json.loads((root/'replay/replay-worker.json').read_text())[0]['status'],'matched')

