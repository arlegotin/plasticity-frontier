"""Independent oracle spectral and regression controls."""
import numpy as np
from scipy.optimize import brentq, minimize
from .metrics import CompactMetric
from .records import SupportMetadata


def source_phase(d, r, n, sigma2, k):
    if not (d >= r >= 1 and n > 0 and sigma2 >= 0 and k >= 0): raise ValueError('invalid phase parameters')
    c, rho = (n+1)/n, (1+sigma2)/n
    alpha, beta = c+r*rho, (d-r)*rho
    if r == d:
        a = b = min(k, 1/alpha)
    else:
        k0 = (r-1)/(alpha*(r-1)+beta*r)
        k1 = r/alpha
        if k <= k0: a = b = k
        elif k < k1:
            D = alpha*(r-1)**2+beta*r*r
            a, b = ((r-1)**2+beta*r*k)/D, (r-1)*(r-alpha*k)/D
        else: a, b = 1/alpha, 0.
    return float(a), float(b), float(1-2*a+alpha*a*a+beta*b*b)


def _spectral(s, n, sigma2, k):
    s = np.asarray(s, dtype=float)
    if s.ndim != 1 or not len(s) or not np.all(np.isfinite(s)) or np.any(s<0) or n<=0 or sigma2<0 or k<0 or not np.isfinite(k+sigma2): raise ValueError('invalid spectral inputs')
    return s, (n+1)/n*s+(1+sigma2)/n


def solve_rank_gains(s, n, sigma2, k, width):
    s,q = _spectral(s,n,sigma2,k)
    if not isinstance(width, int) or width<0: raise ValueError('invalid width')
    idx = np.argsort(-s, kind='stable')[:min(width,len(s))]
    out = np.zeros_like(s)
    if not len(idx) or k == 0: return out
    ss, qq = s[idx], q[idx]
    g = ss/qq
    if sum(g)>k:
        root = brentq(lambda t: np.maximum(0,(ss-t)/qq).sum()-k, 0, float(ss.max()), xtol=5e-324, rtol=1e-14)
        g = np.maximum(0,(ss-root)/qq)
    out[idx]=g
    return out


def solve_unrestricted_gamma(s, n, sigma2, k):
    s,q = _spectral(s,n,sigma2,k); d=len(s)
    if k == 0: return np.zeros(d)
    # Independent all-gain formulation, including an explicit minimum t.
    x0=np.r_[np.full(d,min(k,1/(1+(1+sigma2)*d/(n+1)))), min(k,1/(1+(1+sigma2)*d/(n+1)))]
    def fun(x): return np.dot(q,x[:d]**2)-2*np.dot(s,x[:d])
    def jac(x): return np.r_[2*q*x[:d]-2*s,0.]
    A=np.zeros((d+1,d+1)); A[:d,:d]=np.eye(d); A[:d,-1]=-1; A[-1,:d]=-1; A[-1,-1]=d-1
    result=minimize(fun,x0,jac=jac,method='SLSQP',bounds=[(0,None)]*(d+1),constraints={'type':'ineq','fun':lambda x: A@x+np.r_[np.zeros(d),k], 'jac':lambda x:A},options={'ftol':1e-14,'maxiter':2000})
    if not result.success or np.min(A@result.x+np.r_[np.zeros(d),k]) < -1e-9: raise RuntimeError('spectral QP failed: '+result.message)
    return result.x[:d]


def _solve(a,b,penalty):
    if penalty == 0: return np.linalg.lstsq(a,b,rcond=None)[0]
    if a.shape[1] <= a.shape[0]: return np.linalg.solve(a.T@a+penalty*np.eye(a.shape[1]),a.T@b)
    return a.T@np.linalg.solve(a@a.T+penalty*np.eye(a.shape[0]),b)


def ridge_coefficient(context, penalty, basis=None):
    X=np.asarray(context.X,dtype=float); y=np.asarray(context.y,dtype=float)
    if X.ndim!=2 or y.shape!=(len(X),) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or not np.isfinite(penalty) or penalty<0: raise ValueError('invalid ridge input')
    if basis is None: return _solve(X,y,penalty)
    U=np.asarray(basis,dtype=float)
    if U.ndim!=2 or U.shape[0]!=X.shape[1] or not np.allclose(U.T@U,np.eye(U.shape[1]),atol=1e-10): raise ValueError('invalid basis')
    return U@_solve(X@U,y,penalty)


def posterior_coefficient(context, prior):
    S=np.asarray(prior.covariance,dtype=float); X=np.asarray(context.X,dtype=float); y=np.asarray(context.y,dtype=float)
    if X.ndim!=2 or y.shape!=(len(X),) or S.shape!=(X.shape[1],X.shape[1]) or prior.noise_variance is None or prior.noise_variance<0 or not np.all(np.isfinite(S)) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or not np.allclose(S,S.T,atol=1e-12): raise ValueError('invalid posterior input')
    vals,U=np.linalg.eigh((S+S.T)/2)
    if vals.min() < -1e-12: raise ValueError('indefinite covariance')
    idx=vals>0; C=U[:,idx]*np.sqrt(vals[idx]); A=X@C
    return C@_solve(A,y,prior.noise_variance)


def metric_from_gains(basis,gains):
    gains=np.asarray(gains,dtype=float); basis=np.asarray(basis,dtype=float)
    return CompactMetric(0.,basis,gains,SupportMetadata('analytic',int(np.count_nonzero(gains==0)),int(np.count_nonzero(gains==gains.min())) if len(gains) else 0,tuple(np.flatnonzero(gains))))


def unrestricted_fixed_metric(prior, task):
    s=np.asarray(prior.eigenvalues,dtype=float); n=task.target_context_size
    _,q=_spectral(s,n,prior.noise_variance,0.)
    return metric_from_gains(prior.basis,s/q)
