"""Optional fixed-pool, semi-synthetic stress test; no acquisition is implicit."""
import csv
import hashlib
import io
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np

DATASET_URL = 'https://archive.ics.uci.edu/dataset/186/wine+quality'
LICENSE_URL = 'https://creativecommons.org/licenses/by/4.0/legalcode'
FEATURE_COLUMNS = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                   'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                   'pH', 'sulphates', 'alcohol']


def _verified_bytes(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('a lowercase SHA256 is required')
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('asset or license hash mismatch')
    return data


def load_natural_asset(manifest):
    if manifest.get('dataset_url') != DATASET_URL or manifest.get('license_url') != LICENSE_URL:
        raise ValueError('official dataset and license routes required')
    if not isinstance(manifest.get('source_url'), str): raise ValueError('official source URL required')
    source = urlsplit(manifest['source_url'])
    if (manifest.get('operator_attested_official_origin') is not True or manifest.get('license_id') != 'CC-BY-4.0'
            or source.scheme != 'https' or source.netloc != 'archive.ics.uci.edu'
            or not source.path or source.query or source.fragment):
        raise ValueError('operator-attested official source and CC BY 4.0 provenance required')
    if manifest.get('columns') != FEATURE_COLUMNS or manifest.get('expected_rows') != 4898:
        raise ValueError('white wine requires the eleven feature columns and 4898 rows')
    raw = _verified_bytes(manifest['file_path'], manifest['file_sha256'])
    if not _verified_bytes(manifest['license_path'], manifest['license_sha256']):
        raise ValueError('empty license snapshot')
    try:
        reader = csv.reader(io.StringIO(raw.decode('utf-8-sig')), delimiter=';')
        if next(reader) != FEATURE_COLUMNS + ['quality']:
            raise ValueError('unexpected table columns')
        rows = list(reader)
        if len(rows) != 4898 or any(len(row) != 12 for row in rows):
            raise ValueError('unexpected table dimensions')
        X = np.array([[float(x) for x in row[:11]] for row in rows], dtype=np.float64)
    except (UnicodeError, StopIteration) as exc:
        raise ValueError('invalid table') from exc
    if not np.isfinite(X).all(): raise ValueError('nonfinite covariates')
    return X


def split_duplicate_groups(X, seed):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all(): raise ValueError('finite nonempty matrix required')
    _, inverse = np.unique(X, axis=0, return_inverse=True)
    order = np.random.default_rng(seed).permutation(inverse.max()+1)
    counts = np.floor(np.array([.6,.2,.2])*len(order)).astype(int)
    for i in np.argsort(-(np.array([.6,.2,.2])*len(order)-counts), kind='stable')[:len(order)-sum(counts)]: counts[i] += 1
    cuts = np.cumsum(counts)
    return {name:X[np.isin(inverse, groups)].copy() for name, groups in zip(
        ('development','validation','protected'), np.split(order, cuts[:2]))}


def fit_development_transform(development, whiten):
    X = np.asarray(development, dtype=np.float64)
    if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all() or not isinstance(whiten, bool):
        raise ValueError('finite development matrix and whitening flag required')
    mean = X.mean(axis=0); scale = X.std(axis=0)
    if np.any(scale == 0): raise ValueError('zero-variance direction requires declared amendment')
    standardized = (X-mean)/scale
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    tolerance = np.finfo(float).eps * max(X.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    if rank != X.shape[1]: raise ValueError('rank reduction requires declared amendment')
    matrix = (vt.T * (np.sqrt(len(X))/singular)) @ vt if whiten else np.eye(X.shape[1])
    return {'mean':mean.tolist(), 'scale':scale.tolist(), 'matrix':matrix.tolist(), 'rank':rank,
            'whiten':whiten, 'coordinate_note':'development-fitted coordinates; whitening changes norm interpretation'}


def sample_conditional_task(pool, context_size, query_size, sigma2, seed):
    pool = np.asarray(pool, dtype=np.float64)
    if pool.ndim != 2 or not np.isfinite(pool).all() or not len(pool) or type(context_size) is not int or type(query_size) is not int or min(context_size,query_size)<1 or not np.isfinite(sigma2) or sigma2 < 0:
        raise ValueError('invalid task sampling parameters')
    rng = np.random.default_rng(seed)
    if context_size + query_size <= len(pool):
        indices = rng.choice(len(pool), context_size+query_size, replace=False)
        context, query = indices[:context_size], indices[context_size:]
    else:
        context = rng.choice(len(pool), context_size, replace=context_size>len(pool))
        remaining = np.setdiff1d(np.arange(len(pool)), context)
        query = rng.choice(remaining if len(remaining) else np.arange(len(pool)), query_size,
                           replace=query_size > (len(remaining) if len(remaining) else len(pool)))
    theta = rng.normal(size=pool.shape[1])/np.sqrt(pool.shape[1])
    cn = rng.normal(size=context_size)*np.sqrt(sigma2); qn = rng.normal(size=query_size)*np.sqrt(sigma2)
    return {'context_indices':context.tolist(), 'query_indices':query.tolist(), 'theta':theta,
            'context_noise':cn, 'query_noise':qn, 'context_y':pool[context]@theta+cn,
            'query_y':pool[query]@theta+qn,
            'inference':'semi-synthetic task distribution conditional on the fixed input pool'}


def optional_result(config):
    if not isinstance(config, dict):
        return {'status':'not_run', 'reason':'optional configuration must be an object'}
    if config.get('status') != 'run':
        return {'status':'not_run', 'reason':config.get('reason') or 'No unresolved input-law question justifies acquisition.'}
    return {'status':'not_run', 'reason':'Official acquisition, license, and byte receipt must be reviewed before natural-data execution is implemented.'}


def run_optional(config, core):
    result = core()
    return {'core':result, 'natural':optional_result(config)}
