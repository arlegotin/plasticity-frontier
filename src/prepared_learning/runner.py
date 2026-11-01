"""Durable immutable blocks of paired evidence; summaries are read from raw rows."""
import fcntl
from contextlib import contextmanager
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .records import to_json_record, RAW_RECORD_FIELDS

IDENTITY_FIELDS = ('git_commit', 'dirty_diff_sha256', 'scientific_config_sha256', 'protocol_sha256', 'dependency_lock_sha256', 'prior_asset_sha256', 'data_manifest_sha256')
RUN_FIELDS = IDENTITY_FIELDS + ('schema_version', 'run_id', 'phase', 'operational_config_sha256', 'analysis_mode', 'hardware_record', 'information_access', 'timing_boundaries', 'status', 'pairing_spec')


def _bytes(value):
    return json.dumps(to_json_record(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _syncdir(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _write(path, data):
    with open(path, 'xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def _name(key):
    if not isinstance(key, str) or not key: raise ValueError('nonempty block key required')
    return hashlib.sha256(key.encode()).hexdigest()


class RunStore:
    def __init__(self, path): self.path = Path(path)

    @classmethod
    def create(cls, root, run):
        if any(k not in run for k in RUN_FIELDS) or run['schema_version'] != 1: raise ValueError('incomplete run record')
        for key in ('scientific_config_sha256', 'operational_config_sha256', 'protocol_sha256', 'dependency_lock_sha256'):
            if not isinstance(run[key], str) or not re.fullmatch(r'[0-9a-f]{64}', run[key]): raise ValueError('invalid hash')
        for key in ('dirty_diff_sha256', 'data_manifest_sha256', 'prior_asset_sha256'):
            if run[key] is not None and (not isinstance(run[key], str) or not re.fullmatch(r'[0-9a-f]{64}', run[key])): raise ValueError('invalid optional hash')
        if not isinstance(run['git_commit'], str) or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', run['git_commit']): raise ValueError('invalid git revision')
        if run['status'] not in ('running', 'completed', 'interrupted', 'invalid', 'failed') or run['information_access'] not in ('oracle', 'estimated', 'none') or run['phase'] not in ('exploration', 'confirmation') or run['analysis_mode'] not in ('deterministic_frontier', 'conditional_risk_over_preparations', 'fixed_prior_task_simulation', 'nested_task_simulation'): raise ValueError('invalid run metadata')
        specs = run['pairing_spec']
        if not isinstance(specs, dict): raise ValueError('pairing spec required')
        for kind, spec in specs.items():
            expected = {'task_metrics': ['outer_replication', 'task_id'], 'preparation_risks': ['outer_replication']}
            if kind not in expected or spec.get('unit_fields') != expected[kind] or not isinstance(spec.get('method_ids'), list) or len(spec['method_ids']) < 2 or len(set(spec['method_ids'])) != len(spec['method_ids']) or any(not isinstance(x, str) or not x for x in spec['method_ids']): raise ValueError('invalid pairing spec')
        if not isinstance(run['run_id'], str) or not run['run_id'] or Path(run['run_id']).name != run['run_id'] or run['run_id'] in ('.', '..'): raise ValueError('invalid run id')
        path = Path(root) / run['run_id']; path.mkdir(parents=True, exist_ok=False)
        (path / 'blocks').mkdir(); (path / 'journal').mkdir(); _write(path / 'run.json', _bytes(run)); _write(path / 'run.sha256', _hash(path / 'run.json').encode()); _syncdir(path)
        return cls(path)

    @classmethod
    def open(cls, path, expected_identity):
        path = Path(path)
        if _hash(path / 'run.json') != (path / 'run.sha256').read_text(): raise ValueError('run metadata corrupt')
        run = json.loads((path / 'run.json').read_text())
        if any(k not in expected_identity or expected_identity[k] != run.get(k) for k in IDENTITY_FIELDS): raise ValueError('run identity mismatch')
        if expected_identity.get('operational_config_sha256', run['operational_config_sha256']) != run['operational_config_sha256'] and any(k not in run or k not in expected_identity for k in ('thread_policy', 'blas_policy')):
            raise ValueError('operational changes require declared thread policy')
        for key in ('thread_policy', 'blas_policy'):
            if key in expected_identity and expected_identity[key] != run.get(key):
                raise ValueError('thread or BLAS changes require verified equivalence evidence; unsupported')
        obj = cls(path)
        with obj._lock():
            obj._reconcile()
            operational = expected_identity.get('operational_config_sha256')
            if operational is not None and operational != run['operational_config_sha256']:
                record = {k: expected_identity.get(k) for k in ('operational_config_sha256', 'thread_policy', 'blas_policy')}
                with open(path / 'operational_changes.jsonl', 'ab') as f:
                    f.write(_bytes(record)+b'\n'); f.flush(); os.fsync(f.fileno())
                _syncdir(path)
        return obj

    @contextmanager
    def _lock(self):
        with open(self.path / '.lock', 'a+b') as lock:
            try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc: raise ValueError('run store is busy') from exc
            try: yield
            finally: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _verify(self, block):
        try:
            marker = json.loads((block / 'commit.json').read_text())
            if _name(marker['key']) != block.name: raise ValueError('block key mismatch')
            manifest = json.loads((block / 'manifest.json').read_text())
            if _hash(block / 'manifest.json') != marker['manifest_sha256']: raise ValueError('manifest hash mismatch')
            for name, digest in manifest['files'].items():
                if Path(name).is_absolute() or '..' in Path(name).parts or _hash(block / name) != digest: raise ValueError('block hash mismatch')
            rows = json.loads((block / 'records.json').read_text())
            if not isinstance(rows, list) or len(rows) != manifest['count']: raise ValueError('block count mismatch')
            return marker['key']
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc: raise ValueError('corrupt committed block') from exc

    def _reconcile(self):
        if _hash(self.path / 'run.json') != (self.path / 'run.sha256').read_text(): raise ValueError('run metadata corrupt')
        blocks = self.path / 'blocks'; journal = self.path / 'journal'
        for entry in journal.iterdir():
            if entry.name.startswith('.pending-'):
                entry.unlink(); continue
            block = blocks / entry.name
            if not block.is_dir() or not (block / 'commit.json').is_file():
                raise ValueError('committed block or marker missing')
            if _hash(block / 'commit.json') != entry.read_text(): raise ValueError('commit marker mismatch')
            self._verify(block)
        for block in blocks.iterdir():
            if not block.is_dir(): raise ValueError('unexpected block entry')
            if not (journal / block.name).exists(): shutil.rmtree(block)
        _syncdir(blocks); _syncdir(journal)

    def _validate_rows(self, records):
        run = json.loads((self.path / 'run.json').read_text())
        specs = run.get('pairing_spec', {})
        grouped = {}
        shared_keys = {}
        for row in records:
            kind = row.get('kind')
            if kind not in RAW_RECORD_FIELDS or not RAW_RECORD_FIELDS[kind].issubset(row): raise ValueError('incomplete raw record schema')
            core = {
                'task_metrics': ('run_id', 'outer_replication', 'task_id', 'task_key', 'method_id', 'information_access', 'n', 'd', 'depth', 'norm_cap', 'parameter_cap', 'eta', 'status'),
                'preparation_risks': ('run_id', 'outer_replication', 'source_key', 'method_id', 'selection_procedure_hash', 'metric_hash', 'analysis_mode', 'source_training_tasks', 'source_validation_tasks', 'total_source_labels', 'paired_contrast_id', 'risk_evaluator_version', 'true_S_evaluator_hash', 'noise_truth_evaluator_hash', 'status'),
                'geometry': ('target_hash', 'realized_hash', 'spectral_scale', 'support_policy', 'minimum_eigenvalue', 'initial_norm', 'actual_trainable_count', 'stored_scalars', 'serialized_bytes', 'null_residual', 'kernel_absolute_error', 'kernel_relative_error_status', 'direct_step_error', 'numerical_repairs'),
                'certificates': ('configuration_key', 'branch', 'solver_status', 'solver_version', 'solver_options', 'certificate_level', 'roundoff_treatment'),
                'timing_blocks': ('method_order', 'cold_warm', 'block_size', 'generation_boundary', 'preparation_boundary', 'adaptation_boundary', 'prediction_boundary', 'serialization_boundary', 'repetitions', 'wall_time_ns', 'peak_rss', 'peak_rss_unit', 'power_thermal_condition', 'timing_stratum'),
                'decisions': ('timestamp', 'work_package', 'observation', 'evidence_paths', 'evidence_hashes', 'decision', 'rationale', 'affected_claim_ids', 'next_action', 'methodological_amendment_required', 'fresh_confirmation_required'),
            }
            if any(row[k] is None for k in core[kind]): raise ValueError('missing core evidence')
            if kind == 'task_metrics' and (any(not isinstance(row[k], int) or isinstance(row[k], bool) or row[k] < 1 for k in ('n', 'd', 'depth')) or not isinstance(row['parameter_cap'], int) or isinstance(row['parameter_cap'], bool) or row['parameter_cap'] < 0 or any(not isinstance(row[k], (int, float)) or not np.isfinite(row[k]) or row[k] < 0 for k in ('norm_cap', 'eta')) or row['information_access'] not in ('oracle', 'estimated', 'none') or (row['status'] == 'completed' and (not isinstance(row['loss_parameter_mse'], (int, float)) or not np.isfinite(row['loss_parameter_mse']) or row['loss_parameter_mse'] < 0))): raise ValueError('invalid task measurements')
            if kind == 'task_metrics' and (not isinstance(row['task_key'], str) or not row['task_key'] or not isinstance(row['method_id'], str) or not row['method_id'] or row['status'] not in ('completed', 'failed', 'invalid')): raise ValueError('invalid task record')
            if kind == 'preparation_risks' and (not row['source_key'] or not row['method_id'] or row['status'] not in ('completed', 'failed', 'invalid')): raise ValueError('invalid preparation record')
            if row.get('run_id', run['run_id']) != run['run_id']: raise ValueError('row run identity mismatch')
            if row.get('status') in ('failed', 'invalid') and not row.get('failure_reason'): raise ValueError('failure reason required')
            if kind in ('task_metrics', 'preparation_risks') and kind not in specs: raise ValueError('missing pairing declaration')
            if kind in specs:
                spec = specs[kind]; unit = tuple(row[f] for f in spec['unit_fields'])
                if any(x is None for x in unit): raise ValueError('missing unit key')
                key = (kind, _bytes(unit)); roles = grouped.setdefault(key, set()); role = row['method_id']
                shared = row['task_key'] if kind == 'task_metrics' else row['source_key']
                if key in shared_keys and shared_keys[key] != shared: raise ValueError('paired data keys differ')
                shared_keys[key] = shared
                if role in roles or role not in spec['method_ids']: raise ValueError('duplicate or undeclared method')
                roles.add(role)
        for (kind, unit), roles in grouped.items():
            if roles != set(specs[kind]['method_ids']): raise ValueError('incomplete paired unit')
        return set(grouped)

    def committed_keys(self):
        with self._lock():
            self._reconcile()
            return {self._verify(p) for p in (self.path / 'blocks').iterdir()}

    def read_records(self, kind):
        with self._lock():
            self._reconcile()
            rows = []
            for block in sorted((self.path / 'blocks').iterdir()):
                self._verify(block)
                rows.extend(r for r in json.loads((block / 'records.json').read_text()) if r.get('kind') == kind)
            return rows

    def commit_block(self, block_key, records, assets):
        with self._lock():
            self._reconcile()
            self._commit_block(block_key, list(records), assets)

    def _commit_block(self, block_key, records, assets):
        units = self._validate_rows(records)
        for old in (self.path / 'blocks').iterdir():
            existing = json.loads((old / 'records.json').read_text())
            if units & self._validate_rows(existing): raise ValueError('paired unit already committed')
        blocks = self.path / 'blocks'; name = _name(block_key); final = blocks / name
        if final.exists(): raise ValueError('block already exists')
        temp = Path(tempfile.mkdtemp(prefix='.tmp-', dir=blocks))
        try:
            _write(temp / 'records.json', _bytes(list(records)))
            (temp / 'assets').mkdir()
            for asset_name, array in assets.items():
                if not isinstance(asset_name, str) or not asset_name or Path(asset_name).name != asset_name or asset_name in ('.', '..'): raise ValueError('invalid asset name')
                arr = np.asarray(array)
                if arr.dtype.hasobject or not np.all(np.isfinite(arr)): raise ValueError('invalid asset array')
                with open(temp / 'assets' / (asset_name + '.npy'), 'xb') as f:
                    np.save(f, arr, allow_pickle=False); f.flush(); os.fsync(f.fileno())
            _syncdir(temp / 'assets')
            files = {str(p.relative_to(temp)): _hash(p) for p in temp.rglob('*') if p.is_file()}
            _write(temp / 'manifest.json', _bytes({'count': len(records), 'files': files}))
            for path, digest in files.items():
                if _hash(temp / path) != digest: raise ValueError('precommit verification failed')
            _syncdir(temp); os.rename(temp, final); _syncdir(blocks)
            _write(final / 'commit.pending', _bytes({'key': block_key, 'manifest_sha256': _hash(final / 'manifest.json')}))
            os.rename(final / 'commit.pending', final / 'commit.json'); _syncdir(final)
            journal = self.path / 'journal'; pending = journal / ('.pending-' + name)
            _write(pending, _hash(final / 'commit.json').encode()); os.rename(pending, journal / name); _syncdir(journal)
        finally:
            if temp.exists(): shutil.rmtree(temp)
