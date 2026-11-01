"""Rational development oracles independent of production numerical solvers."""
import json
from fractions import Fraction as F
from pathlib import Path
from prepared_learning.records import JsonRecord


def frac(pair):
    if not isinstance(pair, list) or len(pair) != 2 or any(type(x) is not int for x in pair):
        raise ValueError('exact values require integer numerator/denominator pairs')
    return F(*pair)


def load_exact_cases(path: Path) -> dict[str, JsonRecord]:
    data = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(data, dict) or set(data) != {'phase', 'primary_richer_k_3_5'}:
        raise ValueError('invalid exact case collection')
    return data


def phase_results(case):
    d, r, n = case['d'], case['r'], case['n']
    c, rho = F(n + 1, n), (1 + frac(case['sigma2'])) / n
    alpha, beta = c + r*rho, (d-r)*rho
    k0 = F(r-1) / (alpha*(r-1) + beta*r)
    k1 = F(r) / alpha
    D = alpha*(r-1)**2 + beta*r*r
    risks = []
    for pair in case['budgets']:
        k = frac(pair)
        if k <= k0: a = b = k
        elif k < k1:
            a = ((r-1)**2 + beta*r*k)/D
            b = (r-1)*(r-alpha*k)/D
        else: a, b = 1/alpha, F(0)
        risks.append(1-2*a+alpha*a*a+beta*b*b)
    return (k0, k1), risks


def scalar_optimum(k, c, rho):
    return min(k, 1/(c+rho))


def zero_parameter_count(d, depth):
    if d < 1 or depth < 2: raise ValueError('positive dimension and depth >= 2 required')
    return d + depth - 1


def primary_results(case):
    s = list(map(frac, case['spectrum']))
    n = case['n']; c = F(n+1, n); rho = (1+frac(case['sigma2']))/n
    q = [c*x+rho for x in s]
    out = {}
    for branch in ('F', 'Z'):
        x = list(map(frac, case[branch]['gains']))
        gains = x[:2] + [x[2]]*(len(s)-2) if branch == 'F' else x + [F(0)]*(len(s)-3)
        out[branch] = 1 + sum(-2*si*gi + qi*gi*gi for si, qi, gi in zip(s, q, gains))
    return out


def _primary_kkt(case, branch):
    s = list(map(frac, case['spectrum'])); n = case['n']
    c = F(n+1, n); rho = (1+frac(case['sigma2']))/n
    q = [c*si+rho for si in s]; k = frac(case['budget'])
    x = list(map(frac, case[branch]['gains'])); v = frac(case[branch]['multiplier'])
    if branch == 'F':
        ss = s[:2]+[sum(s[2:])]; qq = q[:2]+[sum(q[2:])]
        a = [F(1), F(1), F(-1)]
        feasible = x[0] >= x[2] >= 0 and x[1] >= x[2]
    else:
        ss, qq, a = s[:3], q[:3], [F(1)]*3
        feasible = all(y >= 0 for y in x)
    slack = k-sum(ai*xi for ai, xi in zip(a, x))
    stationarity = all(2*qi*xi-2*si+v*ai == 0 for si, qi, xi, ai in zip(ss, qq, x, a))
    dual = 1-v*k-sum((-2*si+v*ai)**2/(4*qi) for si, qi, ai in zip(ss, qq, a))
    risk = primary_results(case)[branch]
    return feasible and slack >= 0 and v >= 0 and v*slack == 0 and stationarity and dual == risk and risk == frac(case[branch]['risk'])


def counterexample_coefficient(h: F) -> F:
    # W1=(1,0), W2=((1,1),(0,1)), W3=(0,1); all updates use old weights.
    w1 = [F(1), h]; w2 = [[F(1), F(1)], [h, F(1)]]; w3 = [h, F(1)]
    return sum(w3[i]*w2[i][j]*w1[j] for i in range(2) for j in range(2))


def check_exact_cases(cases: dict[str, JsonRecord]) -> list[JsonRecord]:
    phase = cases['phase']; primary = cases['primary_richer_k_3_5']
    thresholds, risks = phase_results(phase); results = primary_results(primary)
    iso = scalar_optimum(frac(primary['budget']), F(primary['n']+1, primary['n']), (1+frac(primary['sigma2']))/primary['n']*len(primary['spectrum']))
    s = list(map(frac, primary['spectrum'])); c = F(primary['n']+1, primary['n']); rho = (1+frac(primary['sigma2']))/primary['n']
    iso_risk = 1-2*iso*sum(s)+sum((c*si+rho)*iso*iso for si in s)
    checks = [
        ('phase', thresholds == tuple(map(frac, phase['thresholds'])) and risks == list(map(frac, phase['risks'])), 'blueprint §4.6'),
        ('primary_F_kkt', _primary_kkt(primary, 'F'), 'blueprint §§4.4, 4.6'),
        ('primary_Z_kkt', _primary_kkt(primary, 'Z'), 'blueprint §§4.4, 4.6'),
        ('gap_and_isotropy', results['Z']-results['F'] == frac(primary['gap']) and iso_risk == frac(primary['isotropic_risk']), 'blueprint §4.6'),
        ('noncanonical_update', counterexample_coefficient(F(1)) == 4 and counterexample_coefficient(F(2)) == 10, 'blueprint §4.7'),
        ('zero_and_scalar', zero_parameter_count(6, 3) == 8 and scalar_optimum(F(3, 5), F(5, 4), F(1, 2)) == F(4, 7), 'blueprint §§4.1, 4.6')]
    return [{'check': name, 'status': 'pass' if ok else 'failed', 'evidence': evidence} for name, ok, evidence in checks]
