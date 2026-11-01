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
from .records import to_json_record, validate_run_record, validate_raw_record, validate_unit_id, JsonRecord, Array
from collections.abc import Sequence, Mapping

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
    def create(cls, root: Path, run: JsonRecord) -> "RunStore":
        run={**run,'lifecycle_version':1}
        validate_run_record(run)
        if not isinstance(run['run_id'], str) or not run['run_id'] or Path(run['run_id']).name != run['run_id'] or run['run_id'] in ('.', '..'): raise ValueError('invalid run id')
        path = Path(root) / run['run_id']; path.mkdir(parents=True, exist_ok=False)
        (path / 'blocks').mkdir(); (path / 'journal').mkdir(); _write(path / 'run.json', _bytes(run)); _write(path / 'run.sha256', _hash(path / 'run.json').encode());
        _write(path / 'initial_run.json', _bytes({'run':run,'sha256':_hash(path / 'run.json')})); _syncdir(path)
        return cls(path)

    @classmethod
    def open(cls, path: Path, expected_identity: JsonRecord) -> "RunStore":
        path = Path(path)
        obj = cls(path)
        with obj._lock():
            obj._recover_lifecycle()
            if _hash(path/'run.json')!=(path/'run.sha256').read_text(): raise ValueError('run metadata corrupt')
            run = json.loads((path / 'run.json').read_text())
            if any(k not in expected_identity or expected_identity[k] != run.get(k) for k in IDENTITY_FIELDS): raise ValueError('run identity mismatch')
            if expected_identity.get('operational_config_sha256', run['operational_config_sha256']) != run['operational_config_sha256'] and any(k not in run or k not in expected_identity for k in ('thread_policy', 'blas_policy')):
                raise ValueError('operational changes require declared thread policy')
            for key in ('thread_policy', 'blas_policy'):
                if key in expected_identity and expected_identity[key] != run.get(key):
                    raise ValueError('thread or BLAS changes require verified equivalence evidence; unsupported')
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

    def _replace_metadata(self, name, data):
        fd,pending=tempfile.mkstemp(prefix='.lifecycle-',dir=self.path)
        try:
            with os.fdopen(fd,'wb') as file:
                file.write(data); file.flush(); os.fsync(file.fileno())
            os.replace(pending,self.path/name); _syncdir(self.path)
        finally:
            if Path(pending).exists(): Path(pending).unlink()

    def _recover_lifecycle(self):
        """Replay only a verified terminal receipt, never repair arbitrary metadata."""
        initial_path=self.path/'initial_run.json'; terminal_path=self.path/'terminal.json'
        if not initial_path.exists():
            if terminal_path.exists() or json.loads((self.path/'run.json').read_text()).get('lifecycle_version') is not None:
                raise ValueError('lifecycle initial identity is missing')
            return  # Read-only compatibility with pre-lifecycle stores.
        try:
            initial=json.loads(initial_path.read_text()); initial_bytes=_bytes(initial['run'])
            if hashlib.sha256(initial_bytes).hexdigest()!=initial['sha256']: raise ValueError('initial metadata corrupt')
            validate_run_record(initial['run'])
            current=(self.path/'run.json').read_bytes(); checksum=(self.path/'run.sha256').read_text()
            if not terminal_path.exists():
                if current!=initial_bytes or checksum!=initial['sha256']: raise ValueError('run metadata changed without lifecycle receipt')
                return
            receipt=json.loads(terminal_path.read_text()); payload=receipt['payload']
            if hashlib.sha256(_bytes(payload)).hexdigest()!=receipt['sha256'] or payload['initial_sha256']!=initial['sha256']: raise ValueError('terminal receipt corrupt')
            terminal=payload['run']; validate_run_record(terminal)
            if terminal['status'] not in ('completed','incomplete','failed','invalid','interrupted') or {**terminal,'status':initial['run']['status']}!=initial['run']: raise ValueError('terminal receipt changes scientific metadata')
            actual={p.name:_hash(p) for p in (self.path/'journal').iterdir() if not p.name.startswith('.pending-')}
            if actual!=payload['block_receipts']: raise ValueError('terminal block inventory changed')
            terminal_bytes=_bytes(terminal); terminal_hash=hashlib.sha256(terminal_bytes).hexdigest()
            if current not in (initial_bytes,terminal_bytes) or checksum not in (initial['sha256'],terminal_hash): raise ValueError('run metadata corrupt')
        except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
            raise ValueError('corrupt lifecycle metadata') from exc
        if current!=terminal_bytes: self._replace_metadata('run.json',terminal_bytes)
        if checksum!=terminal_hash: self._replace_metadata('run.sha256',terminal_hash.encode())

    def finalize(self, status: str) -> None:
        """Durably seal blocks and publish terminal status under the store lock.

        The receipt is committed first; interruption while replacing the two
        metadata files is recovered from their verified initial/final versions.
        """
        if status not in ('completed','incomplete','failed','invalid','interrupted'): raise ValueError('invalid terminal status')
        with self._lock():
            self._reconcile()
            run=json.loads((self.path/'run.json').read_text())
            if (self.path/'terminal.json').exists():
                if run['status']!=status: raise ValueError('terminal status is immutable')
                return
            if not (self.path/'initial_run.json').exists(): raise ValueError('legacy store has no lifecycle identity; retain unchanged')
            payload={'initial_sha256':_hash(self.path/'run.json'),'run':{**run,'status':status},
                     'block_receipts':{p.name:_hash(p) for p in (self.path/'journal').iterdir()}}
            receipt={'payload':payload,'sha256':hashlib.sha256(_bytes(payload)).hexdigest()}
            self._replace_metadata('terminal.json',_bytes(receipt))
            self._recover_lifecycle()

    def _verify(self, block, marker=None):
        try:
            if marker is None: marker = json.loads((block / 'commit.json').read_text())
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
        self._recover_lifecycle()
        if _hash(self.path / 'run.json') != (self.path / 'run.sha256').read_text(): raise ValueError('run metadata corrupt')
        blocks = self.path / 'blocks'; journal = self.path / 'journal'
        for entry in journal.iterdir():
            if entry.name.startswith('.pending-'):
                entry.unlink(); continue
            block = blocks / entry.name
            if not block.is_dir(): raise ValueError('committed block missing')
            try: marker = json.loads(entry.read_text())
            except (OSError, json.JSONDecodeError) as exc: raise ValueError('corrupt journal receipt') from exc
            self._verify(block, marker)
            marker_path = block / 'commit.json'
            if marker_path.exists():
                if marker_path.read_bytes() != _bytes(marker): raise ValueError('commit marker mismatch')
            else:
                pending = block / 'commit.pending'
                if pending.exists(): pending.unlink()
                _write(pending, _bytes(marker)); os.rename(pending, marker_path); _syncdir(block)
        for block in blocks.iterdir():
            if not block.is_dir(): raise ValueError('unexpected block entry')
            if not (journal / block.name).exists():
                if (block / 'commit.json').exists(): raise ValueError('finalized block missing journal entry')
                shutil.rmtree(block)
        _syncdir(blocks); _syncdir(journal)

    def _validate_rows(self, records):
        run = json.loads((self.path / 'run.json').read_text())
        specs = run.get('pairing_spec', {})
        grouped = {}
        shared_keys = {}
        grouped_rows = {}
        for row in records:
            validate_raw_record(row, run)
            kind = row['kind']
            if kind in ('task_metrics', 'preparation_risks') and kind not in specs: raise ValueError('missing pairing declaration')
            if kind in specs:
                spec = specs[kind]; unit = tuple(row[f] for f in spec['unit_fields'])
                for value in unit: validate_unit_id(value)
                key = (kind, _bytes(unit)); roles = grouped.setdefault(key, set()); role = row['method_id']
                shared = row['task_key'] if kind == 'task_metrics' else row['source_key']
                if key in shared_keys and shared_keys[key] != shared: raise ValueError('paired data keys differ')
                shared_keys[key] = shared
                if role in roles or role not in spec['method_ids']: raise ValueError('duplicate or undeclared method')
                roles.add(role)
                grouped_rows.setdefault(key, []).append(row)
        for (kind, unit), roles in grouped.items():
            if roles != set(specs[kind]['method_ids']): raise ValueError('incomplete paired unit')
            if kind == 'preparation_risks':
                paired = grouped_rows[(kind, unit)]
                complete = all(row['status'] == 'completed' for row in paired)
                if any((row['paired_risk_difference'] is None) == complete for row in paired):
                    raise ValueError('paired contrast availability disagrees with paired statuses')
        return set(grouped)

    def committed_keys(self) -> set[str]:
        with self._lock():
            self._reconcile()
            return {self._verify(p) for p in (self.path / 'blocks').iterdir()}

    def read_records(self, kind: str) -> list[JsonRecord]:
        with self._lock():
            self._reconcile()
            rows = []
            for block in sorted((self.path / 'blocks').iterdir()):
                self._verify(block)
                rows.extend(r for r in json.loads((block / 'records.json').read_text()) if r.get('kind') == kind)
            return rows

    def commit_block(self, block_key: str, records: Sequence[JsonRecord], assets: Mapping[str, Array]) -> None:
        with self._lock():
            self._reconcile()
            if (self.path/'terminal.json').exists(): raise ValueError('terminal run cannot accept new blocks')
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
            marker = {'key': block_key, 'manifest_sha256': _hash(final / 'manifest.json')}
            journal = self.path / 'journal'; pending = journal / ('.pending-' + name)
            _write(pending, _bytes(marker)); os.rename(pending, journal / name); _syncdir(journal)
            _write(final / 'commit.pending', _bytes(marker))
            os.rename(final / 'commit.pending', final / 'commit.json'); _syncdir(final)
        finally:
            if temp.exists(): shutil.rmtree(temp)
