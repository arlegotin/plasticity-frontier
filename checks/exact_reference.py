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


def _pair(x):
    return [x.numerator, x.denominator]


def primary_rational_qp(case, branch):
    """Construct fixture coefficients directly from rational source data."""
    s=list(map(frac,case['spectrum'])); n=case['n']; k=frac(case['budget'])
    q=[F(n+1,n)*v+(1+frac(case['sigma2']))/n for v in s]
    if branch=='F':
        ss=s[:2]+[sum(s[2:])]; qq=q[:2]+[sum(q[2:])]
        a=[[1,0,-1],[0,1,-1],[0,0,1],[-1,-1,1]]
    elif branch=='Z':
        ss=s[:3]; qq=q[:3]; a=[[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]]
    else: raise ValueError('unknown branch')
    return {'constant':_pair(F(1)), 'linear':[_pair(-2*v) for v in ss],
            'hessian':[[_pair(qq[i] if i==j else F(0)) for j in range(3)] for i in range(3)],
            'A':[[_pair(F(v)) for v in row] for row in a], 'offset':[_pair(F(0))]*3+[_pair(k)]}


def verify_rational_kkt(qp: JsonRecord, active_set: tuple[int, ...]) -> JsonRecord:
    """Solve and verify a rational KKT system; fractions are integer pairs."""
    c=frac(qp['constant']); l=list(map(frac,qp['linear'])); H=[list(map(frac,row)) for row in qp['hessian']]
    A=[list(map(frac,row)) for row in qp['A']]; o=list(map(frac,qp['offset'])); n=len(l); m=len(A)
    if len(H)!=n or any(len(row)!=n for row in H) or any(len(row)!=n for row in A) or len(o)!=m or len(set(active_set))!=len(active_set) or any(type(i)!=int or i<0 or i>=m for i in active_set): raise ValueError('invalid rational QP dimensions or active set')
    # Sylvester criterion, using exact elimination, proves convexity strictly.
    M=[row[:] for row in H]; positive=True
    if any(H[i][j]!=H[j][i] for i in range(n) for j in range(n)): positive=False
    for i in range(n):
        if M[i][i]<=0: positive=False; break
        for j in range(i+1,n):
            t=M[j][i]/M[i][i]
            for z in range(i+1,n): M[j][z]-=t*M[i][z]
    K=[[2*H[i][j] for j in range(n)]+[-A[a][i] for a in active_set]+[-l[i]] for i in range(n)]
    K += [[*A[a], *([F(0)]*len(active_set)), -o[a]] for a in active_set]
    size=len(K)
    for i in range(size):
        pivot=next((j for j in range(i,size) if K[j][i]),None)
        if pivot is None: return {'verified':False,'certificate_level':'unresolved','reason':'singular KKT system'}
        K[i],K[pivot]=K[pivot],K[i]; t=K[i][i]; K[i]=[v/t for v in K[i]]
        for j in range(size):
            if j!=i:
                t=K[j][i]; K[j]=[v-t*w for v,w in zip(K[j],K[i])]
    x=[K[i][-1] for i in range(n)]; mu=[F(0)]*m
    for j,a in enumerate(active_set): mu[a]=K[n+j][-1]
    slack=[sum(a*b for a,b in zip(row,x))+off for row,off in zip(A,o)]
    station=[l[i]+2*sum(H[i][j]*x[j] for j in range(n))-sum(A[j][i]*mu[j] for j in range(m)) for i in range(n)]
    primal=c+sum(a*b for a,b in zip(l,x))+sum(x[i]*H[i][j]*x[j] for i in range(n) for j in range(n))
    # Under stationarity, minimizing the Lagrangian gives this dual value.
    dual=primal-sum(a*b for a,b in zip(mu,slack))
    ok=positive and all(v>=0 for v in slack+mu) and all(v==0 for v in station) and all(a*b==0 for a,b in zip(mu,slack)) and primal==dual
    return {'verified':ok,'certificate_level':'exact_rational' if ok else 'unresolved','primal':_pair(primal),'dual':_pair(dual),'x':list(map(_pair,x)),'slack':list(map(_pair,slack)),'multipliers':list(map(_pair,mu)),'stationarity':list(map(_pair,station)),'positive_definite':positive}
