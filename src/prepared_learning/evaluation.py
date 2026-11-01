"""Evaluator-only truth and analytic risk for fixed compact maps."""
from dataclasses import dataclass
import numpy as np
from .records import ImmutableArrays, Array, PopulationTruth
from .metrics import CompactMetric

@dataclass(frozen=True)
class EvaluationTruth(ImmutableArrays):
    theta: Array
    second_moment: Array
    sigma2: float
    task_key: object


def coefficient_loss(coefficient: Array, truth: EvaluationTruth) -> float:
    w = np.asarray(coefficient, dtype=np.float64)
    if w.shape != truth.theta.shape or w.ndim != 1 or not np.all(np.isfinite(w)):
        raise ValueError('invalid coefficient')
    diff = w - truth.theta
    return float(diff @ diff)


def conditional_risk(metric: CompactMetric, truth: PopulationTruth) -> float:
    S = np.asarray(truth.second_moment, dtype=np.float64)
    d, _ = metric.basis.shape
    n = truth.target_context_size
    if truth.input_law != "gaussian_isotropic": raise ValueError("conditional risk requires isotropic Gaussian inputs")
    if S.shape != (d, d) or not np.all(np.isfinite(S)) or not np.allclose(S, S.T, rtol=1e-12, atol=1e-12):
        raise ValueError('invalid population second moment')
    if not np.isclose(np.trace(S), 1., atol=1e-12, rtol=0) or np.linalg.eigvalsh(S).min() < -1e-12:
        raise ValueError('population must be PSD and trace one')
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or not np.isfinite(truth.sigma2) or truth.sigma2 < 0:
        raise ValueError('invalid population noise or context size')
    b, u, U = metric.floor, metric.increments, metric.basis
    projected = np.einsum('ij,ij->j', U, S @ U)
    trS = float(np.trace(S))
    trGS = b*trS + float(u @ projected)
    trG2S = b*b*trS + float((2*b*u + u*u) @ projected)
    trG2 = d*b*b + float(np.sum(2*b*u + u*u))
    return float(1 - 2*trGS + ((n+1)/n)*trG2S + ((1+truth.sigma2)/n)*trG2)
