import argparse
import json
from pathlib import Path
from .environment import discover_environment, stage_source_bundle

def command_doctor(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        result = discover_environment(output)
        archive = Path('docs/prepared_learning_research_bundle.zip'); snapshot = Path('source_bundle')
        if archive.exists() and not snapshot.exists():
            manifest = stage_source_bundle(archive, snapshot)
            (output/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(result, allow_nan=False))
        return 0 if result['status'] == 'pass' else 1
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        result = {'status':'failed', 'error':str(exc)}
        (output/'failure.json').write_text(json.dumps(result)+'\n'); print(json.dumps(result)); return 1

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(required=True)
    doctor = sub.add_parser('doctor'); doctor.add_argument('--output', required=True); doctor.set_defaults(handler=command_doctor)
    args = parser.parse_args(argv); return args.handler(args)
