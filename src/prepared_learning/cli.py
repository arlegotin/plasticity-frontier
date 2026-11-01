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
