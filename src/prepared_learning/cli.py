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
    args = parser.parse_args(argv); return args.handler(args)


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
