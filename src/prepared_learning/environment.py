"""Local environment diagnostics and safe source snapshotting."""
import hashlib
import json
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np
import scipy
from scipy.optimize import minimize

HARDWARE_ALLOWLIST = {'Model Name', 'Model Identifier', 'Model Number', 'Chip', 'Total Number of Cores', 'Memory', 'System Firmware Version', 'OS Loader Version'}
def filter_hardware(values): return {k: v for k, v in values.items() if k in HARDWARE_ALLOWLIST}
def _command(*args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
        return {'status': p.returncode, 'output': p.stdout.strip(), 'error': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc: return {'status': None, 'output': '', 'error': str(exc)}
def stage_source_bundle(archive: Path, output: Path, expected_manifest=None):
    """Validate archive and either create or verify the complete snapshot."""
    archive = Path(archive); output = Path(output)
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist(); names = set(); contents = {}
        for info in infos:
            name = info.filename; path = PurePosixPath(name)
            mode = info.external_attr >> 16
            if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'prepared_learning_research' or '\\' in name or stat.S_ISLNK(mode) or (mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)) or name in names:
                raise ValueError(f'unsafe archive member: {name}')
            names.add(name)
            if not info.is_dir(): contents[name] = z.read(info)
        manifest = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                    'members': {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}
        if expected_manifest is not None and manifest != expected_manifest:
            raise ValueError('archive differs from frozen source manifest')
        if output.is_symlink(): raise ValueError('snapshot destination is a symlink')
        if output.exists():
            actual = {}; directories = set()
            for path in output.rglob('*'):
                relative = path.relative_to(output).as_posix()
                if path.is_symlink(): raise ValueError('symlink in snapshot')
                if path.is_file(): actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                elif path.is_dir(): directories.add(relative)
                else: raise ValueError('special file in snapshot')
            expected_dirs = {str(parent) for name in contents for parent in PurePosixPath(name).parents if str(parent) != '.'}
            if actual != manifest['members'] or directories != expected_dirs:
                raise ValueError('snapshot differs from frozen source manifest')
            return manifest
        output.mkdir(parents=True, exist_ok=True)
        for name, data in contents.items():
            dest = output.joinpath(*PurePosixPath(name).parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data); dest.chmod(0o444)
        for path in sorted(output.rglob('*'), reverse=True):
            if path.is_dir(): path.chmod(0o555)
    return manifest

def discover_environment(output: Path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    commands = {'architecture': ('uname','-m'), 'os': ('sw_vers',), 'memory_bytes': ('sysctl','-n','hw.memsize'), 'logical_cpu': ('sysctl','-n','hw.logicalcpu'), 'physical_cpu': ('sysctl','-n','hw.physicalcpu'), 'disk': ('df','-h','.'), 'memory_pressure': ('vm_stat',), 'power': ('pmset','-g','batt'), 'power_mode': ('pmset','-g','custom')}
    hardware = {key: _command(*cmd) for key, cmd in commands.items()}
    # Avoid querying hardware inventory, which includes serial numbers and UUIDs.
    hardware['model'] = {}
    hardware['model_status'] = 'unknown'
    hardware['workload_state'] = 'unknown'
    hardware['power_state_interpretation'] = 'unknown'
    (output/'hardware.json').write_text(json.dumps(hardware, indent=2, allow_nan=False)+'\n')
    ops = {}; errors = []
    try:
        a = np.array([[4., 1.], [1., 3.]], dtype=np.float64); vals, vecs = np.linalg.eigh(a)
        ops['eigh_residual'] = float(np.linalg.norm((vecs * vals) @ vecs.T - a))
        ops['solve_residual'] = float(np.linalg.norm(a @ np.linalg.solve(a, [1.,2.]) - [1.,2.]))
        q, r = np.linalg.qr(a); u, sv, vh = np.linalg.svd(a)
        ops['qr_residual'] = float(np.linalg.norm(q@r-a)); ops['svd_residual'] = float(np.linalg.norm((u*sv)@vh-a))
        c = np.array([[1+2j]], dtype=np.complex128) @ np.array([[3-1j]], dtype=np.complex128); ops['complex128'] = bool(c.dtype == np.complex128 and np.allclose(c, [[5+5j]]))
        opt = minimize(lambda x: float(np.sum((x-[1,2])**2)), np.array([0.,0.]), jac=lambda x: 2*(x-[1,2]), method='SLSQP', options={'ftol': 1e-12})
        ops['slsqp_solution'] = opt.x.tolist(); ops['slsqp_success'] = bool(opt.success and np.linalg.norm(opt.x-[1,2]) < 1e-7)
        if any(ops[k] >= 1e-12 for k in ('eigh_residual','solve_residual','qr_residual','svd_residual')) or not ops['complex128'] or not ops['slsqp_success']: errors.append('required operation failed')
    except Exception as exc: errors.append(f'{type(exc).__name__}: {exc}')
    result = {'status': 'failed' if errors else 'pass', 'errors': errors, 'python': sys.version, 'executable': sys.executable, 'machine': platform.machine(), 'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'operations': ops}
    (output/'environment.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
