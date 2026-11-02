"""Same-machine isolated source/environment recovery and scientific re-execution.

The coordinator never changes the primary checkout. It restores the recorded
revision in a local clone, verifies every locked wheel, builds a fresh venv and
runs this worker from that clone. Protected execution uses the existing strict
live-identity guard; this module supplies no bypass or fixture promotion.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from .records import GateDecision, GateCode, content_hash
from .reporting import (_read, _write, _sha, _scientific, compare_scientific,
                        compare_run_stores, _verify_inputs, ReportingResourceLimit,
                        _validate_output_artifact, _artifact_summary, _guard_paths,
                        _object_bytes, _benchmark, _load_worker_index, _load_worker_ref,
                        _validate_benchmark_worker, _worker_key, _publish_worker,
                        _publish_worker_index, _RecordStream, REPORT_SCHEMA_VERSION)


def _run(command, *, cwd, env=None, log=None):
    result=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,check=False)
    if log is not None:
        _write(log,{'command':command,'cwd':str(cwd),'returncode':result.returncode,
                    'stdout':result.stdout,'stderr':result.stderr})
    if result.returncode==75:
        try: limited=json.loads(result.stdout)
        except ValueError: limited={}
        if limited.get('gate')=='LOCAL_RESOURCE_LIMIT': raise ReportingResourceLimit(limited['evidence'])
    if result.returncode: raise ValueError(f'replay operation failed ({result.returncode}): {command[:4]}; {result.stderr[-1500:]}')
    return result.stdout


def _run_file(command, *, cwd, env, stdout_path, stderr_path, log, package=None):
    """Keep complete process evidence on disk, including failed worker output."""
    stdout_path=Path(stdout_path); stderr_path=Path(stderr_path)
    stdout_path.parent.mkdir(parents=True,exist_ok=True); stderr_path.parent.mkdir(parents=True,exist_ok=True)
    with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
        result=subprocess.run(command,cwd=cwd,env=env,stdout=stdout,stderr=stderr,check=False)
    def reference(path):
        return {'path':os.path.relpath(path,Path(package) if package is not None else Path(log).parent),'sha256':_sha(path),'size_bytes':path.stat().st_size}
    _write(log,{'command':command,'cwd':str(cwd),'returncode':result.returncode,
                'stdout':reference(stdout_path),'stderr':reference(stderr_path)})
    if result.returncode==75:
        _guard_paths((stdout_path,),phase='replay_resource_gate_before_parse')
        try: limited=_read(stdout_path)
        except ValueError: limited={}
        if limited.get('gate')=='LOCAL_RESOURCE_LIMIT': raise ReportingResourceLimit(limited['evidence'])
    if result.returncode:
        raise ValueError(f'replay operation failed ({result.returncode}): {command[:4]}; complete stdout/stderr retained at {log}')
    return stdout_path


def _benchmark_scientific(worker):
    """The complete historical projection, with ONE cost alias map per worker."""
    resource=worker['resource']
    fields=('method_id','initialization_norm','trainable_parameters','runtime_stored_scalars','runtime_serialized_bytes',
            'selected_asset','selected_execution_hash','execution_hash','selection','risk','equality','equality_status','diagnostics')
    return {'resource':_scientific({k:resource[k] for k in fields}),
            'schedule':[{k:r[k] for k in ('method_id','block','timing_stratum')} for r in worker['rows']]}


def _replay_benchmark(entry, destination, root, replay, config, *, retained_bytes=lambda:0):
    """Validate/stage original workers, then compare exactly one full pair at a time."""
    from .benchmark import THREAD_ENV, _campaign_configuration
    summary=_benchmark(entry,output=destination,retained_bytes=retained_bytes)
    if summary['status']!='completed': raise ValueError('benchmark replay requires complete original workers')
    old_index_ref=summary['worker_records_ref']
    index=_load_worker_index(old_index_ref,destination,retained_bytes=retained_bytes()+_object_bytes(summary))
    old_refs={(r['worker_key']['block'],r['worker_key']['method_id']):r for r in index['records']}
    # Compact report risk vectors/rows are unnecessary for scientific replay.
    del index,summary
    frozen=config['frozen']; expected={(block,mid) for block,order in enumerate(frozen['orders']) for mid in order}
    if set(old_refs)!=expected: raise ValueError('benchmark original worker key set changed')
    requests=[]; requested=set(); new_refs=[]
    def reserve():
        return retained_bytes()+sum(_object_bytes(v) for v in (config,old_refs,requests,requested,new_refs,expected))
    for path in sorted((Path(entry['run_path'])/'requests').glob('*.json')):
        if path.name.endswith('.failure.json'): continue
        _guard_paths((path,),phase='benchmark_replay_request_before_parse',object_bytes=reserve())
        record=_read(path); key=(record['block'],record['method']['method_id'])
        if key in requested: raise ValueError('duplicate benchmark replay request')
        if key not in expected: raise ValueError('unexpected benchmark replay request key')
        block,mid=key; order=frozen['orders'][block]
        if (record['method_order']!=order or type(record['position'])!=int or
                not 0<=record['position']<len(order) or order[record['position']]!=mid or
                record['settings'].get('fixed_repetitions')!=frozen['repetitions'][mid]):
            raise ValueError('benchmark replay request schedule changed')
        if _campaign_configuration(record['inputs']['manifest'])!=_campaign_configuration(config):
            raise ValueError('benchmark replay request scientific inputs changed')
        requested.add(key); requests.append((path,key)); del record
    if requested!=expected: raise ValueError('benchmark replay request count/key mismatch')
    directory=Path(f"benchmark-input-{entry['input_id']}")/'remeasured'
    with _RecordStream(destination/'remeasured-workers.json') as aggregate:
        for ordinal,(path,key) in enumerate(requests):
            _guard_paths((path,),phase='benchmark_replay_request_execution',object_bytes=reserve())
            record=_read(path); env=dict(os.environ)
            env.update({name:str(record['settings']['threads']) for name in THREAD_ENV})
            process=destination/'processes'/f'{ordinal:08d}'
            stdout=process.with_suffix('.stdout.json'); stderr=process.with_suffix('.stderr.txt')
            _run_file([sys.executable,'-m','prepared_learning._benchmark_worker',str(path)],
                      cwd=root,env=env,stdout_path=stdout,stderr_path=stderr,log=process.with_suffix('.process.json'),package=destination)
            del record,env
            reference=old_refs[key]; old_path=destination/reference['path']
            # Two full workers plus both complete projections, using the same
            # semantic factor and all retained metadata. Never cohort-sized.
            _guard_paths((old_path,stdout),phase='benchmark_replay_pair_before_parse',copies=2,object_bytes=reserve())
            old=_load_worker_ref(reference,destination,retained_bytes=reserve())
            new=_read(stdout)
            if _worker_key(new)!=reference['worker_key']: raise ValueError('benchmark replay result/request key mismatch')
            _validate_benchmark_worker(new,frozen)
            old_science=_benchmark_scientific(old); new_science=_benchmark_scientific(new)
            compare_scientific(old_science,new_science,rtol=replay['rtol'],atol=replay['atol'])
            new_reference=_publish_worker(new,destination,directory,ordinal,source_path=stdout,
                                          source_sha256=_sha(stdout),source_pointer='')
            new_refs.append(new_reference); aggregate.write(new)
            del old,new,old_science,new_science,reference,new_reference
    new_index_ref=_publish_worker_index(new_refs,destination,directory)
    return {'report_schema_version':REPORT_SCHEMA_VERSION,'input_id':entry['input_id'],'kind':'benchmark','status':'matched','workers':len(new_refs),
            'package_path':destination.name,
            'worker_records_ref':old_index_ref,'remeasured_worker_records_ref':new_index_ref,
            'timing_policy':'remeasured_not_equal','timing_stratum':'technical_replay'}


def _git_bytes(root, revision, path):
    result=subprocess.run(['git','show',revision+':'+path],cwd=root,capture_output=True,check=False)
    if result.returncode: raise ValueError('recorded revision lacks required replay source: '+path)
    return result.stdout


def _validate_source(root, source):
    revision=source.get('git_commit','')
    if not re.fullmatch('[0-9a-f]{40}',revision): raise ValueError('invalid recorded revision identity')
    lock=_git_bytes(root,revision,'requirements.lock.txt')
    if hashlib.sha256(lock).hexdigest()!=source['dependency_lock_sha256']: raise ValueError('recorded dependency lock identity changed')
    names=_run(['git','ls-tree','-r','--name-only',revision,'src/prepared_learning'],cwd=root).splitlines()
    hashes={name:hashlib.sha256(_git_bytes(root,revision,name)).hexdigest() for name in names if name.endswith('.py') and Path(name).parent.as_posix()=='src/prepared_learning'}
    if content_hash(hashes)!=source['code_sha256']: raise ValueError('recorded source identity differs from committed revision')
    if 'src/prepared_learning/_replay.py' not in names: raise ValueError('recorded revision predates clean replay capability')
    return revision


def clean_replay(manifest, output):
    replay=manifest['replay']; root=Path(replay['source_root']).resolve()
    rtol,atol=replay['rtol'],replay['atol']
    if not 0<=rtol<=1e-8 or not 0<=atol<=1e-10: raise ValueError('undeclared or overly broad arithmetic tolerance')
    if replay.get('timing_policy')!='remeasured_not_equal': raise ValueError('timing must be remeasured')
    source=replay['source']; revision=_validate_source(root,source)
    executable=[e for e in manifest['inputs'] if e['kind'] not in ('not_run','source','forecast')]
    for entry in executable:
        for field in ('git_commit','code_sha256','dependency_lock_sha256'):
            if entry['source'][field]!=source[field]: raise ValueError('mixed recorded revision/lock identity; replay separately per executed revision')
        if entry['kind'] not in ('confirmation','benchmark') and not entry.get('command'):
            raise ValueError('executed diagnostic lacks recorded replay command')
    if not executable: raise ValueError('no scientific execution available for replay')
    wheel_manifest=json.loads(_git_bytes(root,revision,'wheel_manifest.json'))
    wheelhouse=Path(replay['wheelhouse']).resolve()
    for name,digest in wheel_manifest.items():
        if Path(name).name!=name or _sha(wheelhouse/name)!=digest: raise ValueError('wheel hash mismatch: '+name)
    output=output.resolve(); output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    # Persist logs/evidence, but discard the clone and its environment on exit.
    with tempfile.TemporaryDirectory(prefix='clean-source-',dir=output) as temporary:
        workspace=Path(temporary); clone=workspace/'source'
        _run(['git','clone','--no-hardlinks','--no-checkout',str(root),str(clone)],cwd=root,log=output/'clone.json')
        _run(['git','checkout','--detach',revision],cwd=clone,log=output/'checkout.json')
        # Each explicitly enumerated non-code input has identical bytes at its
        # recorded relative location. Tracked files must already match the commit.
        restored={}
        for entry in manifest['inputs']:
            for value in entry.get('assets',[]):
                original=Path(value).resolve()
                try: relative=original.relative_to(root)
                except ValueError: raise ValueError('replay asset must have a recorded repository-relative location')
                dest=clone/relative
                if dest.exists():
                    if _sha(dest)!=_sha(original): raise ValueError('asset differs from recorded revision: '+str(relative))
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(original,dest)
                restored[str(relative)]=_sha(dest)
        archive=clone/'docs/prepared_learning_research_bundle.zip'
        if archive.exists():
            from .environment import stage_source_bundle
            stage_source_bundle(archive,clone/'source_bundle',_read(clone/'source_manifest.json'))
        env=dict(os.environ)
        env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
        env['MPLCONFIGDIR']=str(workspace/'mpl'); env['MPLBACKEND']='Agg'; env['PIP_NO_CACHE_DIR']='1'
        python=Path(sys.executable).resolve()
        runtime=next((_read(e['protocol_path'])['evidence']['runtime_identity'] for e in executable if e['kind']=='confirmation'),None)
        if runtime:
            if _sha(python)!=runtime['python']['executable_sha256']: raise ValueError('same recorded Python executable required')
            for name,value in runtime['thread_environment'].items():
                if value is None: env.pop(name,None)
                else: env[name]=value
        _run([str(python),'-m','venv',str(clone/'.venv')],cwd=clone,env=env,log=output/'venv.json')
        interpreter=clone/'.venv/bin/python'
        # Only the verified wheel files are offered to pip, preventing selection
        # of an unmanifested alternative with the same version.
        verified=workspace/'wheels'; verified.mkdir()
        for name in wheel_manifest: os.link(wheelhouse/name,verified/name)
        _run([str(interpreter),'-m','pip','install','--no-index','--find-links',str(verified),'-r','requirements.lock.txt'],cwd=clone,env=env,log=output/'install.json')
        _run([str(interpreter),'-m','pip','install','--no-build-isolation','--no-deps','-e','.'],cwd=clone,env=env,log=output/'editable.json')
        _run([str(interpreter),'-m','pip','check'],cwd=clone,env=env,log=output/'pip-check.json')
        request={'manifest':manifest,'output':str(output/'executions')}
        request_path=clone/'runs/replay-request.json'; _write(request_path,request)
        _run([str(interpreter),'-m','prepared_learning._replay',str(request_path)],cwd=clone,env=env,log=output/'worker.json')
        worker=_read(output/'executions/replay-worker.json')
        # No mutation during a long run can replace the original authority.
        preflight=_verify_inputs(manifest,replay=True)
        receipt={'schema_version':1,'status':'matched','clean_environment':True,'source':source,
                 'fixture':any(e.get('fixture') for e in manifest['inputs']),
                 'input_order_sha256':manifest['input_order_sha256'],'wheel_hashes':wheel_manifest,
                 'resource_preflight':preflight,
                 'restored_assets':restored,'execution_packages_base':'executions','executions':worker,'rtol':rtol,'atol':atol,
                 'timing_policy':'remeasured_not_equal','elapsed_seconds':time.perf_counter()-start,
                 'verification_scope':'same machine, exact committed source and fresh locked editable environment; replay is not a new independent sample; fixtures cannot establish production G10'}
        _write(output/'replay.json',receipt)
    decision=GateDecision(GateCode.SUPPORTED_IN_SCOPE,{'receipt':str(output/'replay.json'),'sha256':_sha(output/'replay.json'),
        'status':'matched','input_order_sha256':manifest['input_order_sha256']},
        'Keyed scientific outputs matched in an isolated recorded-source environment; timings were independently remeasured.',
        'Attach this receipt to the report input manifest; do not pool replayed observations.')
    _write(output/'gate_decision.json',decision)
    return decision


def worker(request):
    manifest=request['manifest']; _verify_inputs(manifest,replay=True)
    output=Path(request['output']); output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]; replay=manifest['replay']; results=[]
    _validate_source(root,replay['source'])
    for entry in manifest['inputs']:
        index=entry['input_id']; destination=output/str(index); destination.mkdir()
        kind=entry['kind']
        if kind=='confirmation':
            from .confirmation import run_confirmation
            protocol=_read(entry['protocol_path'])
            # Original scientific bytes and protected keys are unchanged.
            protocol_path=root/'runs'/f'replay-protocol-{index}.json'; _write(protocol_path,protocol)
            decision=run_confirmation(protocol,destination,manifest_path=protocol_path,checkpoint_tasks=16)
            actual=Path(decision.evidence['run_path'])
            comparison=compare_run_stores(Path(entry['run_path']),actual,rtol=replay['rtol'],atol=replay['atol'])
            results.append({'input_id':index,'kind':kind,**comparison,'output':str(actual)})
        elif kind=='benchmark':
            from .benchmark import _benchmark_identity
            config=_read(Path(entry['run_path'])/'inputs.json')['inputs'].get('manifest',{})
            actual=_benchmark_identity(config)
            if config.get('phase')!='finalized': raise ValueError('clean benchmark replay requires finalized recorded schedule')
            if actual!=config['frozen']['identity']: raise ValueError('benchmark replay source/runtime identity changed')
            results.append(_replay_benchmark(entry,destination,root,replay,config,
                retained_bytes=lambda:_object_bytes(manifest)+_object_bytes(results)))
        elif kind in ('not_run','source','forecast'):
            results.append({'input_id':index,'kind':kind,'status':'retained_not_executed'})
        else:
            command=entry['command']
            if not isinstance(command,list) or not command or any(not isinstance(s,str) for s in command): raise ValueError('recorded CLI argument list required')
            if command[0]=='frontier' and 'output_artifact' not in entry:
                canonical=(len(command) in (3,4) and command[1]=='--config'
                           and command[2] and not command[2].startswith('-')
                           and (len(command)==3 or command[3]=='--certify'))
                if not canonical:
                    raise ValueError('frontier replay requires canonical --config PATH with optional trailing --certify')
            relocated=[]
            for argument in command:
                path=Path(argument)
                if path.is_absolute():
                    try: argument=str(root/path.relative_to(Path(replay['source_root'])))
                    except ValueError: raise ValueError('recorded command paths must be inside the source repository')
                relocated.append(argument)
            command=relocated
            artifact_root=None
            if 'output_artifact' not in entry and (command[0]=='robustness' or
                    (command[0]=='frontier' and 'grid' in _read(command[2]))):
                raise ValueError('file-producing diagnostics require output_artifact for full scientific replay')
            if 'output_artifact' in entry:
                expected_process,expected_summary=_validate_output_artifact(entry,replay['source_root'])
                artifact_root=destination/'generated'
                artifact_root.mkdir(exist_ok=False)
                # Both supported producers accept an operational output override.
                # Never change their inventoried configuration or reuse old output.
                command=command[:3]+['--output',str(artifact_root)]
            # A diagnostic may deliberately fail; retain its exit status and raw
            # JSON. It is not converted into a successful protected result.
            run=subprocess.run([sys.executable,'-m','prepared_learning',*command],cwd=root,capture_output=True,text=True)
            _write(destination/'process.json',{'command':command,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
            result={'input_id':index,'kind':kind,'status':'matched','returncode':run.returncode}
            if artifact_root is not None:
                process={'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
                artifact,summary=_artifact_summary(command[0],process)
                if not artifact.is_absolute(): artifact=root/artifact
                artifact=artifact.absolute()
                fresh_root=artifact_root.resolve()
                if not artifact.resolve().is_relative_to(fresh_root):
                    raise ValueError('output artifact escaped fresh replay directory')
                relative=artifact.relative_to(artifact_root.absolute())
                expected_parts=1 if command[0]=='robustness' else 2
                if (len(relative.parts)!=expected_parts or artifact.name!=entry['output_artifact']['filename']
                        or any(path.is_symlink() for path in (artifact,*artifact.parents))
                        or not artifact.is_file()):
                    raise ValueError('missing or unsafe fresh output artifact')
                if run.returncode!=expected_process['returncode']:
                    raise ValueError('diagnostic process returncode changed')
                compare_scientific(expected_summary,summary,rtol=replay['rtol'],atol=replay['atol'])
                actual=_read(artifact)
                result.update(artifact_path=str(artifact),artifact_sha256=_sha(artifact),
                              comparison_scope='complete scientific output artifact and process summary/status')
            else:
                actual=json.loads(run.stdout)
            expected=_read(entry['raw_path'])
            if kind=='exact':
                compare_scientific(expected,actual,rtol=0,atol=0)
                result['comparison_scope']='exact_rational_no_tolerance'
            else:
                compare_scientific(_scientific(expected),_scientific(actual),rtol=replay['rtol'],atol=replay['atol'])
            results.append(result)
    _write(output/'replay-worker.json',results)


if __name__=='__main__':
    try: worker(_read(sys.argv[1]))
    except ReportingResourceLimit as exc:
        print(json.dumps({'gate':'LOCAL_RESOURCE_LIMIT','evidence':exc.evidence},allow_nan=False))
        raise SystemExit(75)
