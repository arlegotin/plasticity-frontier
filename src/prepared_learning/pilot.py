"""Exploratory precision planning; no protected observations are consumed here."""
from fractions import Fraction
import math
from statistics import NormalDist
import numpy as np
from scipy.stats import chi2

INITIAL_HALF_WIDTH = Fraction(1266883, 39703750) / 4


def plan_sample_sizes(pilot, target_half_width, alpha):
    """Plan from independent pilot variability, conservatively accounting for its uncertainty.

    Returns an explicit extension decision when variance cannot be estimated. The
    estimated precision is a planning approximation, not a significance threshold.
    """
    h, alpha = float(target_half_width), float(alpha)
    if not math.isfinite(h) or h <= 0 or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('invalid precision or alpha')
    mode = pilot.get('analysis_mode')
    z = NormalDist().inv_cdf(1-alpha/2)
    result = {'analysis_mode': mode, 'target_half_width': h, 'alpha': alpha,
              'coverage': 'approximate', 'decision': 'extend_pilot',
              'outer_replicates': None, 'tasks_per_replicate': None}
    if mode == 'conditional_risk_over_preparations':
        for name in ('source_training_tasks','source_validation_tasks','source_block_size'):
            value = pilot.get(name)
            if type(value) is not int or value <= 0: raise ValueError('missing source counts')
            result[name] = value
        blocks = pilot.get('blocks')
        if not isinstance(blocks, list) or len(blocks) < 2:
            result['reason'] = 'at least two independent pilot blocks required'; return result
        values = []
        variances = []
        for block in blocks:
            data = np.asarray(block.get('differences', []), dtype=float)
            if data.ndim != 1 or len(data) < 3 or not np.all(np.isfinite(data)):
                result['reason'] = 'each block needs at least three finite differences'; return result
            values.extend(data.tolist()); variances.append(float(np.var(data, ddof=1)))
        result['block_variances'] = variances
        if min(variances) <= 0 or max(variances)/min(variances) > 4:
            result['reason'] = 'block variances are zero or unstable'; return result
        n = len(values); pooled = float(np.var(values, ddof=1))
        upper = (n-1)*pooled / float(chi2.ppf(.05, n-1))
        result.update({'variance_estimate': pooled, 'variance_upper_planning': upper,
                       'outer_replicates': max(2, math.ceil(z*z*upper/(h*h))),
                       'decision': 'provisional', 'reason': 'variance ratio heuristic passed; independence, tails and calibration still require review'})
    elif mode == 'nested_task_simulation':
        try: between, within, a, b = (float(pilot[k]) for k in ('variance_between','variance_within','preparation_cost','task_cost'))
        except (KeyError, TypeError, ValueError):
            result['reason'] = 'measured variance components and costs required'; return result
        if not pilot.get('variance_stable') or not all(math.isfinite(v) and v > 0 for v in (between,within,a,b)):
            result['reason'] = 'positive stable variance components and measured costs required'; return result
        n = max(1, round(math.sqrt(within*a/(between*b))))
        result.update({'tasks_per_replicate': n, 'outer_replicates': max(2, math.ceil(z*z*(between+within/n)/(h*h))), 'decision':'provisional', 'reason':'nested allocation requires independent variance and cost verification'})
    else:
        raise ValueError('unsupported pilot analysis mode')
    return result
