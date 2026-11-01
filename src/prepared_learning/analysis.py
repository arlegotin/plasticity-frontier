"""Inference on independent paired units, with explicit moment assumptions."""
import numpy as np
from scipy import stats
from .records import GateDecision, GateCode, validate_unit_id


def paired_interval(differences, alpha, procedure, seed):
    x = np.asarray(differences, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)) or not 0 < alpha < 1:
        raise ValueError('finite nonempty independent differences and alpha in (0, 1) required')
    mean = float(np.mean(x)); n = len(x)
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else None
    result = {'n': n, 'mean': mean, 'standard_error': se, 'alpha': alpha,
              'procedure': procedure, 'interval': None, 'status': 'ok',
              'coverage': 'approximate', 'seed': seed}
    if procedure == 'descriptive_only':
        result.update(status='unsupported_or_unverified_moments', coverage=None)
    elif procedure == 'student':
        if n < 2: raise ValueError('student interval requires at least two independent units')
        h = float(stats.t.ppf(1-alpha/2, n-1)*se)
        result['interval'] = [mean-h, mean+h]
    elif procedure == 'outer_bootstrap':
        if n < 2: raise ValueError('bootstrap requires at least two independent units')
        rng = np.random.default_rng(seed)
        means = np.mean(x[rng.integers(0, n, size=(10000, n))], axis=1)
        result['interval'] = np.quantile(means, [alpha/2, 1-alpha/2]).tolist()
        result['bootstrap_replicates'] = 10000
    else: raise ValueError('unknown interval procedure')
    return result


def nested_replication_means(rows):
    groups = {}
    for row in rows:
        if row.get('status', 'completed') != 'completed': raise ValueError('failed or incomplete paired population')
        value = float(row['difference'])
        if not np.isfinite(value): raise ValueError('nonfinite difference')
        groups.setdefault(validate_unit_id(row['outer_replication']), []).append(value)
    if not groups: raise ValueError('no replications')
    return np.array([np.mean(groups[k]) for k in sorted(groups)], dtype=float)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('p values must be finite and in [0, 1]')
    order = np.argsort(p, kind='stable'); out = np.empty_like(p)
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out


def classify_gate(evidence):
    explicit = evidence.get('code')
    if explicit is not None:
        code = GateCode(explicit)
    elif evidence.get('status') in ('failed', 'invalid', 'incomplete'):
        code = GateCode.IMPLEMENTATION_INVALID
    else:
        interval = evidence.get('interval')
        if interval is None: code = GateCode.INSENSITIVE_OR_UNSUITABLE
        elif len(interval) != 2 or not np.all(np.isfinite(interval)) or interval[0] > interval[1]: raise ValueError('invalid interval')
        elif interval[0] > 0: code = GateCode.SUPPORTED_IN_SCOPE
        elif interval[1] < 0: code = GateCode.NO_COMPARATIVE_ADVANTAGE
        else: code = GateCode.INSENSITIVE_OR_UNSUITABLE
    return GateDecision(code.value, evidence, code.value.replace('_', ' ').lower(), 'retain evidence and follow the declared protocol')


def nested_variance_components(rows):
    groups = {}
    for row in rows:
        if row.get('status', 'completed') != 'completed': raise ValueError('incomplete population')
        x = float(row['difference'])
        if not np.isfinite(x): raise ValueError('nonfinite difference')
        groups.setdefault(validate_unit_id(row['outer_replication']), []).append(x)
    sizes = {len(v) for v in groups.values()}
    if len(groups) < 2 or len(sizes) != 1 or next(iter(sizes), 0) < 2:
        raise ValueError('balanced nested design with at least two outer and inner units required')
    n = next(iter(sizes)); j = len(groups)
    within = float(np.mean([np.var(v, ddof=1) for v in groups.values()]))
    outer_variance = float(np.var([np.mean(v) for v in groups.values()], ddof=1))
    between = max(0., outer_variance - within / n)
    return {'outer_replicates': j, 'tasks_per_replicate': n, 'within_variance': within,
            'between_variance': between, 'between_estimate_truncated_at_zero': outer_variance < within/n,
            'mean_variance': (between + within/n)/j}


def holm_family(p_values, family):
    family = list(family)
    if not family or len(set(family)) != len(family) or set(family) != set(p_values):
        raise ValueError('p-values must exactly match the declared family')
    return dict(zip(family, holm_adjust([p_values[k] for k in family]).tolist()))


def paired_differences(rows, unit_fields, baseline_id, joint_id, value_field):
    """Return baseline minus joint for every unit; reject any missing/failed role."""
    if not unit_fields or baseline_id == joint_id: raise ValueError('distinct roles and unit fields required')
    groups = {}
    for row in rows:
        role = row['method_id']
        if role not in (baseline_id, joint_id): continue
        unit = tuple(validate_unit_id(row[k]) for k in unit_fields)
        if any(x is None for x in unit): raise ValueError('missing unit')
        if row.get('status') != 'completed': raise ValueError('incomplete paired comparison')
        value = row.get(value_field)
        if value is None or not np.isfinite(value): raise ValueError('missing paired value')
        group = groups.setdefault(unit, {})
        if role in group: raise ValueError('duplicate role')
        group[role] = value
    if not groups or any(set(v) != {baseline_id, joint_id} for v in groups.values()): raise ValueError('incomplete paired comparison')
    return np.asarray([v[baseline_id]-v[joint_id] for _, v in sorted(groups.items())], dtype=float)
