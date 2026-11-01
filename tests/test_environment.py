import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from prepared_learning.environment import discover_environment, stage_source_bundle

ROOT = Path(__file__).resolve().parents[1]
class EnvironmentTests(unittest.TestCase):
    def test_source_identity_and_safe_extraction(self):
        archive = ROOT/'docs/prepared_learning_research_bundle.zip'
        before = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertEqual(before, 'cc94445ed1a15b852ee0accd32daca3de028c1c3b6e936d8d32cfd21273d6b9f')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); result = stage_source_bundle(archive, tmp/'out')
            self.assertEqual(result['members']['prepared_learning_research/research_note.pdf'], 'd21569825103249a352bba5e69f951bcc163dbc748b33df9124ff637920de4f9')
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), before)
            for kind in ('traversal', 'symlink'):
                bad = tmp/(kind+'.zip')
                with zipfile.ZipFile(bad, 'w') as z:
                    if kind == 'traversal': z.writestr('../escape', 'bad')
                    else:
                        info = zipfile.ZipInfo('prepared_learning_research/link'); info.create_system = 3; info.external_attr = 0o120777 << 16; z.writestr(info, '../escape')
                with self.assertRaises(ValueError): stage_source_bundle(bad, tmp/(kind+'out'))
                self.assertFalse((tmp/'escape').exists())
    def test_cpu_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = discover_environment(Path(tmp))
            self.assertEqual(result['status'], 'pass')
            self.assertLess(result['operations']['eigh_residual'], 1e-12)
            self.assertLess(result['operations']['solve_residual'], 1e-12)
            np.testing.assert_allclose(result['operations']['slsqp_solution'], [1, 2], atol=1e-7)
    def test_redacts_hardware_identifiers(self):
        from prepared_learning.environment import filter_hardware
        data = {'Model Name': 'Mac', 'Chip': 'M3', 'Serial Number (system)': 'secret', 'Hardware UUID': 'secret2'}
        self.assertEqual(filter_hardware(data), {'Model Name': 'Mac', 'Chip': 'M3'})
    def test_snapshot_reuse_rejects_tampered_missing_and_extra_files(self):
        archive = ROOT/'docs/prepared_learning_research_bundle.zip'
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'out'; manifest = stage_source_bundle(archive, dest)
            self.assertEqual(stage_source_bundle(archive, dest, manifest), manifest)
            altered = dict(manifest); altered['archive_sha256'] = '0'*64
            with self.assertRaises(ValueError): stage_source_bundle(archive, dest, altered)
            empty = Path(tmp)/'empty'; empty.mkdir()
            with self.assertRaises(ValueError): stage_source_bundle(archive, empty, manifest)
            member = dest/'prepared_learning_research/README.txt'
            member.parent.chmod(0o755); original = member.read_bytes(); member.chmod(0o644); member.write_bytes(b'tampered')
            with self.assertRaises(ValueError): stage_source_bundle(archive, dest, manifest)
            member.write_bytes(original); member.unlink()
            with self.assertRaises(ValueError): stage_source_bundle(archive, dest, manifest)
            member.write_bytes(original); extra = member.parent/'extra'; extra.write_bytes(b'extra')
            with self.assertRaises(ValueError): stage_source_bundle(archive, dest, manifest)
    def test_doctor_rejects_changed_snapshot_and_records_gate(self):
        import contextlib
        import io
        import json
        import os
        import shutil
        from prepared_learning.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'docs').mkdir()
            shutil.copyfile(ROOT/'docs/prepared_learning_research_bundle.zip', root/'docs/prepared_learning_research_bundle.zip')
            shutil.copyfile(ROOT/'source_manifest.json', root/'source_manifest.json')
            previous = Path.cwd()
            try:
                os.chdir(root)
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(main(['doctor', '--output', 'run']), 0)
                attempt = Path(json.loads(stream.getvalue())['attempt_path'])
                decision = json.loads((attempt/'gate_decision.json').read_text())
                self.assertEqual(decision['code'], 'SUPPORTED_IN_SCOPE')
                member = root/'source_bundle/prepared_learning_research/README.txt'; member.chmod(0o644); member.write_text('changed')
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(main(['doctor', '--output', 'run']), 1)
                attempt = Path(json.loads(stream.getvalue())['attempt_path'])
                decision = json.loads((attempt/'gate_decision.json').read_text())
                self.assertEqual(decision['code'], 'IMPLEMENTATION_INVALID')
                (root/'docs/prepared_learning_research_bundle.zip').unlink()
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(['doctor', '--output', 'run']), 1)
            finally:
                os.chdir(previous)
    def test_doctor_preserves_failure_and_success_attempt_evidence(self):
        import contextlib
        import io
        import json
        import os
        import shutil
        from prepared_learning.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'docs').mkdir()
            archive = root/'docs/prepared_learning_research_bundle.zip'
            shutil.copyfile(ROOT/'docs/prepared_learning_research_bundle.zip', archive)
            shutil.copyfile(ROOT/'source_manifest.json', root/'source_manifest.json')
            previous = Path.cwd()
            def run():
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream): status = main(['doctor', '--output', 'run'])
                return status, Path(json.loads(stream.getvalue())['attempt_path'])
            def evidence(path):
                return {p.name: (p.read_bytes(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in path.iterdir() if p.is_file()}
            try:
                os.chdir(root)
                status, initial = run(); self.assertEqual(status, 0)
                initial_evidence = evidence(initial)
                member = root/'source_bundle/prepared_learning_research/README.txt'
                original = member.read_bytes(); member.chmod(0o644); member.write_bytes(b'first corruption')
                status, first = run(); self.assertEqual(status, 1)
                first_evidence = evidence(first)
                member.write_bytes(b'second corruption')
                status, second = run(); self.assertEqual(status, 1)
                second_evidence = evidence(second)
                member.write_bytes(original); member.chmod(0o444)
                status, success = run(); self.assertEqual(status, 0)
                self.assertEqual(len({initial, first, second, success}), 4)
                self.assertEqual(evidence(initial), initial_evidence)
                self.assertEqual(evidence(first), first_evidence)
                self.assertEqual(evidence(second), second_evidence)
                self.assertEqual(json.loads((success/'gate_decision.json').read_text())['code'], 'SUPPORTED_IN_SCOPE')
            finally:
                os.chdir(previous)
