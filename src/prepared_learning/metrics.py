"""Compact PSD metrics and their canonical resource requirements."""
from dataclasses import dataclass
import math
import numpy as np
from .records import ImmutableArrays, SupportMetadata, Resources

@dataclass(frozen=True)
class CompactMetric(ImmutableArrays):
    floor: float
    basis: np.ndarray
    increments: np.ndarray
    support_metadata: SupportMetadata

    def __post_init__(self):
        super().__post_init__()
        b, u = self.basis, self.increments
        if b.ndim != 2 or b.shape[0] < 1 or u.ndim != 1 or b.shape[1] != len(u) or len(u) > b.shape[0]:
            raise ValueError('invalid compact metric shapes')
        if not np.isfinite(self.floor) or self.floor < 0 or not np.all(np.isfinite(b)) or not np.all(np.isfinite(u)) or np.any(u < 0):
            raise ValueError('metric values must be finite and nonnegative')
        if not np.allclose(b.T @ b, np.eye(len(u)), atol=1e-12, rtol=1e-12):
            raise ValueError('metric basis must be orthonormal')


def apply_metric(metric, z):
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (metric.basis.shape[0],) or not np.all(np.isfinite(z)):
        raise ValueError('invalid vector')
    return metric.floor*z + metric.basis @ (metric.increments*(metric.basis.T@z))


def dense_metric(metric):
    return metric.floor*np.eye(metric.basis.shape[0]) + (metric.basis*metric.increments)@metric.basis.T


def canonical_spectrum(metric):
    """Normalize full-basis representations without losing small positive gaps."""
    d = metric.basis.shape[0]
    inc = metric.increments
    shift = float(np.min(inc)) if len(inc) == d else 0.0
    gaps = inc - shift
    floor = float(metric.floor + shift)
    if not np.isfinite(floor): raise ValueError('metric spectrum not representable')
    return floor, metric.basis, gaps


def metric_resources(metric, depth):
    if not isinstance(depth, int) or depth < 2: raise ValueError('depth must be at least two')
    floor, _, gaps = canonical_spectrum(metric)
    gamma = math.fsum([floor, *map(float, gaps)])
    d = metric.basis.shape[0]
    if gamma == 0: count = d + depth - 1; energy = 0.0
    else:
        width = int(np.count_nonzero(gaps)) + int(floor > 0)
        count = (d+1)*width + depth-2
        energy = (depth-1)*math.exp(math.log(gamma)/(depth-1))
    return Resources(energy, count, count, count*8)


def metric_from_dense(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.shape[0] or not np.all(np.isfinite(a)):
        raise ValueError('dense metric must be finite and square')
    scale = float(np.linalg.norm(a, 2))
    tol = 64*np.finfo(float).eps*len(a)*scale
    skew = float(np.linalg.norm(a-a.T, 'fro'))
    if skew > tol: raise ValueError('metric is asymmetric')
    sym = (a+a.T)*0.5
    vals, vecs = np.linalg.eigh(sym)
    if vals[0] < -tol: raise ValueError('metric is indefinite')
    repairs = []
    if skew: repairs.append({'kind':'symmetrization', 'magnitude':skew/2})
    if np.any(vals < 0): repairs.append({'kind':'psd_projection', 'mass':float(-sum(vals[vals<0]))})
    vals = np.maximum(vals, 0)
    # No positive eigenvalue is discarded; algebraic rank remains approximate.
    meta = SupportMetadata('approximate', int(np.count_nonzero(vals == 0)), int(np.count_nonzero(vals == vals[0])), tuple(np.flatnonzero(vals)), tuple(repairs), ())
    return CompactMetric(0., vecs, vals, meta)
