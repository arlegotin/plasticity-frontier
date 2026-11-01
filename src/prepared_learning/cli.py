import argparse
import json
from pathlib import Path
from .environment import discover_environment, stage_source_bundle
from .records import GateCode, GateDecision, to_json_record

def command_doctor(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = discover_environment(output)
        archive = Path('docs/prepared_learning_research_bundle.zip'); snapshot = Path('source_bundle')
        source_status = 'optional source archive absent'
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
        print(json.dumps(result, allow_nan=False))
        return 0 if valid else 1
    except Exception as exc:
        result = {'status':'failed', 'error':str(exc)}
        (output/'failure.json').write_text(json.dumps(result)+'\n')
        gate = GateDecision(GateCode.IMPLEMENTATION_INVALID, {'status': 'failed', 'failure': 'failure.json'}, str(exc), 'Repair and revalidate the source and environment.')
        (output/'gate_decision.json').write_text(json.dumps(to_json_record(gate), indent=2)+'\n')
        print(json.dumps(result)); return 1

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(required=True)
    doctor = sub.add_parser('doctor'); doctor.add_argument('--output', required=True); doctor.set_defaults(handler=command_doctor)
    args = parser.parse_args(argv); return args.handler(args)
