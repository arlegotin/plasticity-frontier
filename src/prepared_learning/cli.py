import argparse
import json
import tempfile
from pathlib import Path
from .environment import discover_environment, stage_source_bundle
from .records import GateCode, GateDecision, to_json_record

def command_doctor(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='attempt-', dir=output))
    try:
        result = discover_environment(output)
        archive = Path('docs/prepared_learning_research_bundle.zip'); snapshot = Path('source_bundle')
        source_status = 'optional source archive absent'
        if not archive.exists() and snapshot.exists():
            raise ValueError('cannot verify existing snapshot without its source archive')
        if archive.exists():
            frozen = json.loads(Path('source_manifest.json').read_text())
            manifest = stage_source_bundle(archive, snapshot, frozen)
            (output/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            source_status = 'validated against frozen manifest'
        valid = result['status'] == 'pass'
        gate = GateDecision(GateCode.SUPPORTED_IN_SCOPE if valid else GateCode.IMPLEMENTATION_INVALID,
            {'status': 'pass' if valid else 'failed', 'environment': 'environment.json', 'hardware': 'hardware.json', 'source': source_status},
            'Required local numerical operations passed.' if valid else 'Required local numerical operations failed.',
            'Proceed to WP01.' if valid else 'Repair the local environment.')
        (output/'gate_decision.json').write_text(json.dumps(to_json_record(gate), indent=2, allow_nan=False)+'\n')
        result['attempt_path'] = str(output)
        print(json.dumps(result, allow_nan=False))
        return 0 if valid else 1
    except Exception as exc:
        result = {'status':'failed', 'error':str(exc), 'attempt_path': str(output)}
        (output/'failure.json').write_text(json.dumps(result)+'\n')
        gate = GateDecision(GateCode.IMPLEMENTATION_INVALID, {'status': 'failed', 'failure': 'failure.json'}, str(exc), 'Repair and revalidate the source and environment.')
        (output/'gate_decision.json').write_text(json.dumps(to_json_record(gate), indent=2)+'\n')
        print(json.dumps(result)); return 1

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(required=True)
    doctor = sub.add_parser('doctor'); doctor.add_argument('--output', required=True); doctor.set_defaults(handler=command_doctor)
    theory = sub.add_parser('check-theory'); theory.add_argument('--config', required=True); theory.set_defaults(handler=command_check_theory)
    geometry = sub.add_parser('check-geometry'); geometry.add_argument('--suite', choices=['adversarial'], required=True); geometry.set_defaults(handler=command_check_geometry)
    reproduce = sub.add_parser('reproduce'); reproduce.add_argument('--suite', choices=['source_and_baselines'], required=True); reproduce.add_argument('--run-source', action='store_true'); reproduce.add_argument('--full-matrix', action='store_true'); reproduce.add_argument('--output', default='runs/source_reproduction'); reproduce.set_defaults(handler=command_reproduce)
    frontier = sub.add_parser('frontier'); frontier.add_argument('--config', required=True); frontier.add_argument('--certify', action='store_true'); frontier.set_defaults(handler=command_frontier)
    learned = sub.add_parser('fit-priors'); learned.add_argument('--config', required=True); learned.set_defaults(handler=command_fit_priors)
    pilot = sub.add_parser('pilot'); pilot.add_argument('--config', required=True); pilot.add_argument('--output', default='runs/pilots'); pilot.set_defaults(handler=command_pilot)
    freeze = sub.add_parser('freeze'); freeze.add_argument('--protocol', required=True); freeze.add_argument('--output'); freeze.set_defaults(handler=command_freeze)
    confirm = sub.add_parser('confirm'); confirm.add_argument('--manifest', required=True); confirm.add_argument('--output', default='runs/confirmation'); confirm.add_argument('--checkpoint-tasks',type=int,default=16); confirm.set_defaults(handler=command_confirm)
    analyze = sub.add_parser('analyze'); analyze.add_argument('--manifest', required=True); analyze.add_argument('--run'); analyze.add_argument('--output', default='runs/confirmation'); analyze.set_defaults(handler=command_analyze)
    robustness = sub.add_parser('robustness'); robustness.add_argument('--manifest', required=True); robustness.add_argument('--output', default='runs/robustness'); robustness.set_defaults(handler=command_robustness)
    benchmark = sub.add_parser('benchmark'); benchmark.add_argument('--manifest', required=True); benchmark.set_defaults(handler=command_benchmark)
    report = sub.add_parser('report'); report.add_argument('--manifest', required=True); report.add_argument('--output', default='reports'); report.add_argument('--inputs', nargs='+'); report.set_defaults(handler=command_report)
    replay = sub.add_parser('replay'); replay.add_argument('--manifest', required=True); replay.add_argument('--output', default='runs/clean_replay'); replay.set_defaults(handler=command_replay)
    args = parser.parse_args(argv); return args.handler(args)


def command_report(args: argparse.Namespace) -> int:
    from .reporting import build_report_inputs, write_report
    try:
        path=Path(args.manifest)
        if args.inputs:
            if path.exists(): raise ValueError('refusing to overwrite ordered report manifest')
            manifest=build_report_inputs([Path(p) for p in args.inputs])
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        else: manifest=json.loads(path.read_text())
        decision=write_report(manifest,Path(args.output)); print(json.dumps(to_json_record(decision),allow_nan=False))
        return 1 if decision.code==GateCode.IMPLEMENTATION_INVALID else 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'failed','gate':'IMPLEMENTATION_INVALID','error':str(exc)})); return 1


def command_replay(args: argparse.Namespace) -> int:
    from .reporting import replay_run
    try:
        manifest=json.loads(Path(args.manifest).read_text())
        decision=replay_run(manifest,Path(args.output)); print(json.dumps(to_json_record(decision),allow_nan=False))
        return 0 if decision.code==GateCode.SUPPORTED_IN_SCOPE else 1
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'failed','gate':'IMPLEMENTATION_INVALID','error':str(exc)})); return 1


def command_check_theory(args: argparse.Namespace) -> int:
    from checks.exact_reference import load_exact_cases, check_exact_cases, frac
    from fractions import Fraction
    from decimal import Decimal
    from .specs import load_config, validate_config
    try:
        config_path = Path(args.config)
        validate_config(load_config(config_path), confirmation=False)
        raw = json.loads(config_path.read_text(), parse_float=Decimal)
        path = Path(raw['exact_fixture_path'])
        cases = load_exact_cases(path)
        primary = cases['primary_richer_k_3_5']
        budget, prep, workload = raw['budget'], raw['preparation'], raw['workload']
        if (budget['depth'] != 2 or budget['trainable_parameter_cap'] != 21
                or Fraction(budget['initialization_norm_cap']) != frac(primary['budget'])
                or prep['target_context_size'] != primary['n']
                or prep['input_dimension'] != len(primary['spectrum'])
                or [Fraction(x) for x in workload['spectrum']] != list(map(frac, primary['spectrum']))
                or Fraction(workload['sigma2']) != frac(primary['sigma2'])):
            raise ValueError('configuration does not match the exact primary fixture')
        checks = check_exact_cases(cases)
        valid = all(item['status'] == 'pass' for item in checks)
        result = {'status': 'pass' if valid else 'failed', 'gate': 'SUPPORTED_IN_SCOPE' if valid else 'IMPLEMENTATION_INVALID', 'fixture': str(path), 'proof_obligations': 'protocol/proof_obligations.md', 'checks': checks, 'qualification': 'Finite checks establish fixture agreement, not a proof of the theorem.'}
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        result = {'status': 'failed', 'gate': 'IMPLEMENTATION_INVALID', 'error': str(exc)}
        valid = False
    print(json.dumps(result, allow_nan=False))
    return 0 if valid else 1


def command_check_geometry(args: argparse.Namespace) -> int:
    import numpy as np
    from .geometry import construct_canonical, diagnose_geometry, effective_coefficient, simultaneous_step
    from .metrics import CompactMetric, dense_metric, relative_norm
    from .records import SupportMetadata
    from checks.independent_reference import complex_step_jacobian
    try:
        rng = np.random.default_rng(731)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        records = []; derivative_count = 0; maximum = 0.0
        cases = [(0., [0.,0.,0.,0.], None), (1., [0.,0.,0.,0.], None), (0., [0.,0.,1.,2.], None), (1., [0.,0.,1.,2.], None), (1e-14, [0.,1e-14,0.,0.], None), (0., [1.,2.,3.,4.], None)]
        cases += [(scale, [0.,scale,0.,0.], scale) for scale in (1e-200, 1e200)]
        for floor, increments, scale in cases:
            m = CompactMetric(floor, q, np.array(increments), SupportMetadata('analytic', 0, increments.count(0.), tuple(i for i,x in enumerate(increments) if x)))
            for depth in ([4] if scale is not None else range(2, 9)):
                realization = construct_canonical(m, depth)
                diagnostic = diagnose_geometry(m, realization)
                jac = complex_step_jacobian(realization.layers)
                derivative_count += jac.shape[1]
                target = dense_metric(m)
                derivative_error = jac@jac.T-target
                relative, status = relative_norm(derivative_error, target)
                if status == 'undefined_zero_reference': raise ValueError('nonzero derivative error for zero target')
                z = np.array([.1, -.2, .3, -.4])
                update_error = effective_coefficient(simultaneous_step(realization.layers, -z, 1.))-target@z
                update_relative, status = relative_norm(update_error, target)
                if status == 'undefined_zero_reference': raise ValueError('nonzero update error for zero target')
                maximum = max(maximum, relative, update_relative)
                records.append({'scale':scale, 'depth':depth, 'diagnostics':to_json_record(diagnostic), 'derivative_relative_error':relative, 'update_scaled_error':update_relative})
        layers = (np.array([[1.],[0.]]), np.array([[1.,1.],[0.,1.]]), np.array([[0.,1.]]))
        coefficient = float(effective_coefficient(simultaneous_step(layers, np.array([-1.]), 1.))[0])
        valid = maximum < 1e-10 and coefficient == 4.0
        result = {'status':'pass' if valid else 'failed', 'gate':'SUPPORTED_IN_SCOPE' if valid else 'IMPLEMENTATION_INVALID', 'max_relative_error':maximum, 'target_accuracy':1e-10, 'derivative_count':derivative_count, 'records':records, 'counterexample':{'coefficient':coefficient, 'linear_prediction':3.0}}
    except (ValueError, OverflowError, FloatingPointError) as exc:
        valid = False; result = {'status':'failed', 'gate':'IMPLEMENTATION_INVALID', 'error':str(exc)}
    print(json.dumps(result, allow_nan=False))
    return 0 if valid else 1


def command_reproduce(args: argparse.Namespace) -> int:
    import shutil
    import subprocess
    import sys
    import numpy as np
    from .baselines import source_phase, solve_unrestricted_gamma, ridge_coefficient, posterior_coefficient
    from .methods import make_method
    from .records import BudgetSpec, PreparationSpec, PriorAsset, Context
    try:
        cfg=json.loads(Path('configs/exploration/source.json').read_text())
        d,r,n,sigma2=cfg['d'],cfg['r'],cfg['n'],cfg['sigma2']
        s=np.r_[np.full(r,1/r),np.zeros(d-r)]; q=(n+1)/n*s+(1+sigma2)/n
        rows=[]
        for k in cfg['budgets']:
            a,b,risk=source_phase(d,r,n,sigma2,k)
            g=solve_unrestricted_gamma(s,n,sigma2,k)
            qp=1-2*s@g+q@(g*g)
            rows.append({'budget':k,'a':a,'b':b,'analytic_risk':risk,'spectral_qp_risk':float(qp),'risk_residual':float(qp-risk)})
        prior=PriorAsset(np.diag(s),np.eye(d),s,sigma2,'oracle',{},'source-oracle')
        task=PreparationSpec(n,d,'gaussian','1/(2n)',1.)
        budget=BudgetSpec(2,.4,21,1.)
        context=Context(np.arange(n*d,dtype=float).reshape(n,d)/20,np.arange(n,dtype=float)/3,'fixture')
        pairs=[]
        for target in ('B01','B02'):
            method=make_method(target,{}); prepared=method.prepare(prior,budget,task)
            direct=make_method('B05:'+target,{'metric':prepared.metric}); dp=direct.prepare(prior,budget,task)
            residual=float(np.linalg.norm(method.adapt(prepared,context).coefficient-direct.adapt(dp,context).coefficient))
            pairs.append({'target_method_id':target,'direct_method_id':dp.method_id,'coefficient_residual':residual})
        tiny=Context(np.eye(2),np.array([1.,2.]),'regression-fixture')
        tiny_prior=PriorAsset(np.eye(2)/2,np.eye(2),np.array([.5,.5]),1.,'oracle',{},'tiny')
        expected=np.array([1/3,2/3])
        baseline_residuals={'ridge':float(np.linalg.norm(ridge_coefficient(tiny,2.)-expected)), 'posterior':float(np.linalg.norm(posterior_coefficient(tiny,tiny_prior)-expected))}
        valid=all(abs(row['risk_residual'])<1e-10 for row in rows) and all(x['coefficient_residual']<1e-10 for x in pairs) and max(baseline_residuals.values())<1e-10
        output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
        attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=output))
        scripts=[]
        if args.run_source:
            source=Path('source_bundle/prepared_learning_research')
            for name in ('verify_plasticity.py','phase_transition.py','audit_geometry.py'):
                original=source/name
                if not original.exists(): raise ValueError('source script missing: '+str(original))
                script=attempt/name if name=='audit_geometry.py' else original
                if name=='audit_geometry.py': shutil.copy2(original,script)
                command=[sys.executable,str(script)]
                if name!='audit_geometry.py': command += ['--output',str(attempt/(name+'.json'))]
                if name=='phase_transition.py' and args.full_matrix: command += ['--full-matrix']
                run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                (attempt/(name+'.stdout')).write_text(run.stdout); (attempt/(name+'.stderr')).write_text(run.stderr)
                scripts.append({'script':name,'returncode':run.returncode}); valid &= run.returncode==0
        elif args.full_matrix: raise ValueError('--full-matrix requires --run-source')
        result={'status':'pass' if valid else 'failed','phase_points':rows,'direct_equalities':pairs,'baseline_residuals':baseline_residuals,'source_scripts':scripts,'monte_carlo':{'tasks':200000,'uncertainty':'task-level standard errors in phase_transition.py.json'} if args.run_source else {'status':'not run'},'attempt_path':str(attempt)}
        (attempt/'reproduction.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,allow_nan=False)); return 0 if valid else 1
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)})); return 1


def command_frontier(args: argparse.Namespace) -> int:
    import numpy as np
    from .records import PriorAsset, BudgetSpec, PreparationSpec
    from .metrics import metric_resources
    from .frontier import solve_frontier
    try:
        cfg=json.loads(Path(args.config).read_text())
        if cfg['schema_version']!=1: raise ValueError('unsupported schema version')
        s=np.asarray(cfg['spectrum'],dtype=float); d=len(s)
        prior=PriorAsset(np.diag(s),np.eye(d),s,cfg['sigma2'],'oracle',{},'exploration-oracle')
        budget=BudgetSpec(cfg['depth'],cfg['initialization_norm_cap'],cfg['trainable_parameter_cap'],1.)
        task=PreparationSpec(cfg['n'],d,'gaussian','1/(2n)',1.)
        if 'grid' in cfg:
            import csv
            from .frontier import evaluate_frontier_grid
            grid=cfg['grid']
            rows=evaluate_frontier_grid(prior,budget,task,grid['widths'],grid['norm_caps'],grid['risk_tolerance'],grid['budget_tolerance'])
            output=Path(grid.get('output','runs/frontier')); output.mkdir(parents=True,exist_ok=True)
            attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=output))
            metadata={'widths':grid['widths'],'norm_caps':grid['norm_caps'],'risk_tolerance':grid['risk_tolerance'],'budget_tolerance':grid['budget_tolerance'],'certificate_level':'float64_diagnostic','qualification':'sampled interpolation diagnostics are not rigorous enclosures'}
            (attempt/'frontier.json').write_text(json.dumps({'metadata':metadata,'records':rows},indent=2,allow_nan=False)+'\n')
            with (attempt/'frontier.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['width','norm_cap','point_status','error','branch','objective','branch_status','selection_status','selected_branches','trainable_parameters','initialization_norm','diagnostics'])
                writer.writeheader()
                for row in rows:
                    if row['kind']!='point': continue
                    if row['F'] is None:
                        writer.writerow({'width':row['width'],'norm_cap':row['norm_cap'],'point_status':'failed','error':row['error'],'selection_status':row['selection_status']}); continue
                    for branch in ('F','Z'):
                        item=row[branch]; res=item['actual_resources']
                        writer.writerow({'width':row['width'],'norm_cap':row['norm_cap'],'point_status':row['status'],'error':row.get('error',''),'branch':branch,'objective':item['objective'],'branch_status':item['status'],'selection_status':row['selection_status'],'selected_branches':json.dumps(row['selected_branches']),'trainable_parameters':res['trainable_parameters'],'initialization_norm':res['initialization_norm'],'diagnostics':json.dumps(item['diagnostics'],allow_nan=False)})
            failed=sum(r['kind']=='point' and r['status']=='failed' for r in rows)
            print(json.dumps({'status':'incomplete' if failed else 'pass','attempt_path':str(attempt),'points':sum(r['kind']=='point' for r in rows),'failed_points':failed,'unresolved_intervals':sum(r['kind']=='interval' for r in rows),'certificate_level':'float64_diagnostic','qualification':'sampled interpolation diagnostics are not rigorous enclosures'})); return 1 if failed else 0
        frontier=solve_frontier(prior,budget,task)
        result=to_json_record(frontier)
        for branch in ('F','Z'):
            result[branch]['actual_resources']=to_json_record(metric_resources(getattr(frontier,branch).metric,budget.depth))
        result.update({'status':'pass' if all(b.status=='completed' for b in (frontier.F,frontier.Z)) else 'failed','caps':to_json_record(budget)})
        result['certificate_level']='float64_diagnostic'
        if args.certify:
            from fractions import Fraction
            from decimal import Decimal
            from checks.exact_reference import load_exact_cases, frac, primary_rational_qp, verify_rational_kkt
            from .quadratic import build_oracle_qp
            raw=json.loads(Path(args.config).read_text(),parse_float=Decimal)
            try:
                fixture=load_exact_cases(Path(raw['exact_fixture_path']))[raw['exact_fixture_id']]
                matches=(raw['exact_fixture_id']=='primary_richer_k_3_5' and raw['depth']==2 and raw['trainable_parameter_cap']==21 and raw['n']==fixture['n'] and Fraction(raw['initialization_norm_cap'])==frac(fixture['budget']) and Fraction(raw['sigma2'])==frac(fixture['sigma2']) and [Fraction(x) for x in raw['spectrum']]==list(map(frac,fixture['spectrum'])))
                if not matches: raise ValueError('configuration differs from rational fixture')
                certs={}
                for branch in ('F','Z'):
                    rational=primary_rational_qp(fixture,branch)
                    numerical=build_oracle_qp(s,task.target_context_size,prior.noise_variance,budget.initialization_norm_cap,3,branch)
                    for name in ('constant','linear','hessian','A','offset'):
                        exact=rational[name]
                        def as_float(value):
                            if isinstance(value,list) and len(value)==2 and all(type(v)==int for v in value): return float(frac(value))
                            return [as_float(v) for v in value]
                        if not np.array_equal(np.asarray(as_float(exact)),np.asarray(getattr(numerical,name))): raise ValueError('rational and float QP coefficients differ')
                    certs[branch]=verify_rational_kkt(rational,(3,))
                result['exact_certificates']=certs
                result['certificate_level']='exact_rational' if all(c['verified'] for c in certs.values()) else 'float64_diagnostic'
            except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError) as exc:
                result['exact_certificate_status']={'status':'unavailable','reason':str(exc)}
        print(json.dumps(result,allow_nan=False)); return 0 if result['status']=='pass' else 1
    except (ValueError,TypeError,KeyError,OSError,OverflowError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)})); return 1


def command_fit_priors(args: argparse.Namespace) -> int:
    from .specs import load_config, validate_config
    from .records import TaskKey, content_hash
    from .tasks import generate_source_task
    from .methods import make_method
    from .selection import public_method_config, select_method
    try:
        cfg=load_config(Path(args.config)); validate_config(cfg,confirmation=False)
        sample=cfg.sample_plan; root=cfg.keys['exploration']
        if type(root)!=int: raise ValueError('exploration seed must be integer')
        T=sample.source_training_tasks; V=sample.source_validation_tasks
        if not T or not V: raise ValueError('source training and validation counts required')
        training_keys=[TaskKey(root,1,0,i,0) for i in range(T)]; validation_keys=[TaskKey(root,2,0,i,0) for i in range(V)]
        public=public_method_config(cfg)
        public['source_access_manifest']={'training_keys':[content_hash({'key':key}) for key in training_keys], 'validation_keys':[content_hash({'key':key}) for key in validation_keys]}
        train=[generate_source_task(cfg,key) for key in training_keys]
        val=[generate_source_task(cfg,key) for key in validation_keys]
        candidates=[make_method(m,{}) for m in cfg.methods if m in ('B01','B02','B03','B04','B11:F','B11:Z')]
        selection=select_method(candidates,train,val,{**public,'selection_class':'mechanism_all'})
        from .records import PriorAsset, SelectionResult
        import numpy as np
        def subset(ids):
            rows=tuple(row for row in selection.candidate_records if row['method_id'] in ids)
            valid=[r for r in rows if r.get('status')=='completed']
            if not valid: raise ValueError('no successful eligible candidates')
            row=min(valid,key=lambda r:r['validation_loss']); asset=dict(row['asset'])
            for key in ('covariance','basis','eigenvalues'):
                if asset[key] is not None: asset[key]=np.asarray(asset[key],dtype=float)
            return SelectionResult(row['candidate_id'],rows,PriorAsset(**asset))
        baseline=subset({'B01','B02','B11:Z'}); joint=subset({'B03','B04','B11:F','B11:Z'})
        result={'status':'pass','analysis':'exploratory','public_method_config':public,'tuning_resolution':selection.candidate_records[0]['tuning_resolution'],'baseline':to_json_record(baseline),'joint':to_json_record(joint),'source_training_tasks':T,'source_validation_tasks':V,'unique_labels':sum(len(t.A.y)+len(t.B.y) for t in train+val),'training_observation_hash':content_hash(train),'validation_observation_hash':content_hash(val),'truth_separation':'methods receive public config and observed source blocks only'}
        output=Path('runs/learned_prior'); output.mkdir(parents=True,exist_ok=True); attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=output)); (attempt/'fit_priors.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':'pass','attempt_path':str(attempt),'unique_labels':result['unique_labels'],'tuning_resolution':result['tuning_resolution']})); return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)})); return 1


def command_pilot(args: argparse.Namespace) -> int:
    from .specs import load_config
    from .pilot import run_pilot
    try:
        result=run_pilot(load_config(Path(args.config)),Path(args.output))
        print(json.dumps({key:result[key] for key in ('status','run_path','gate')},allow_nan=False))
        return 0 if result['status']=='completed' else 1
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)})); return 1


def command_freeze(args: argparse.Namespace) -> int:
    from .protocol import freeze_protocol, validate_frozen_protocol, checkout_identity
    try:
        path=Path(args.protocol); record=json.loads(path.read_text())
        if record.get('status')=='frozen':
            validate_frozen_protocol(record)
            raise ValueError('protocol already frozen; immutable manifests cannot be overwritten')
        draft=record['draft']; evidence=json.loads((path.parent/record['evidence_path']).read_text())
        scientific=draft['scientific']
        current,dirty=checkout_identity(scientific['selectors'],scientific['config']['numerics']['fitting_candidates'])
        if dirty or current!=scientific['versions']: raise ValueError('freeze requires the clean committed revision and dependencies used for the pilot')
        output=Path(args.output) if args.output else path.with_name(path.stem+'.frozen.json')
        result=freeze_protocol(draft,evidence,output)
        print(json.dumps({'status':'frozen','path':str(output),'protocol_sha256':result['protocol_sha256'],
            'next_action':'execute the fixed protected schedule using this immutable manifest'})); return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)})); return 1


def command_confirm(args: argparse.Namespace) -> int:
    from .confirmation import run_confirmation
    try:
        result=run_confirmation(json.loads(Path(args.manifest).read_text()),Path(args.output),manifest_path=Path(args.manifest),checkpoint_tasks=args.checkpoint_tasks)
        print(json.dumps(to_json_record(result),allow_nan=False))
        return 1 if result.code in ('IMPLEMENTATION_INVALID','LOCAL_RESOURCE_LIMIT') else 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(json.dumps({'status':'failed','code':'IMPLEMENTATION_INVALID','error':str(exc)})); return 1


def command_analyze(args: argparse.Namespace) -> int:
    from .confirmation import analyze_run, run_record
    try:
        manifest=json.loads(Path(args.manifest).read_text())
        path=Path(args.run) if args.run else Path(args.output)/run_record(manifest)['run_id']
        result=analyze_run(path,manifest,manifest_path=Path(args.manifest))
        print(json.dumps(result,allow_nan=False)); return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(json.dumps({'status':'failed','code':'IMPLEMENTATION_INVALID','error':str(exc)})); return 1


def command_robustness(args: argparse.Namespace) -> int:
    from .robustness import run_robustness
    try:
        manifest = json.loads(Path(args.manifest).read_text())
        decision = run_robustness(manifest, Path(args.output))
        if 'natural_inputs' in manifest:
            from .natural_inputs import optional_result
            (Path(args.output)/'natural_inputs.json').write_text(json.dumps(optional_result(manifest['natural_inputs']), indent=2, allow_nan=False)+'\n')
        print(json.dumps(to_json_record(decision), allow_nan=False))
        return 1 if decision.code == GateCode.IMPLEMENTATION_INVALID else 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)}, allow_nan=False)); return 1


def command_benchmark(args: argparse.Namespace) -> int:
    import subprocess
    from .benchmark import benchmark_methods
    from .methods import make_method
    from ._benchmark_worker import generated_config
    try:
        manifest=json.loads(Path(args.manifest).read_text())
        if manifest.get('schema_version') != 1 or manifest.get('phase') not in ('exploratory','finalized'):
            raise ValueError('benchmark phase must be exploratory or finalized')
        generated_config(manifest)
        methods=[make_method(mid,manifest.get('method_options',{}).get(mid,{})) for mid in manifest['methods']]
        result=benchmark_methods(methods,[],dict(manifest,workflow='generated',_manifest_path=args.manifest),Path(manifest.get('output','runs/benchmark')))
        if 'natural_inputs' in manifest:
            from .natural_inputs import optional_result
            (Path(manifest.get('output','runs/benchmark'))/'natural_inputs.json').write_text(json.dumps(optional_result(manifest['natural_inputs']), indent=2, allow_nan=False)+'\n')
        print(json.dumps({'status':result['status'],'output':str(manifest.get('output','runs/benchmark'))}))
        return 0
    except (ValueError, KeyError, OSError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)})); return 1
