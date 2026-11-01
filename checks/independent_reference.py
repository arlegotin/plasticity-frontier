"""Complex-step differentiation of the literal product, independent of geometry."""
import numpy as np


def complex_step_jacobian(layers, step=1e-25):
    if not np.isfinite(step) or step <= 0: raise ValueError('step must be positive and finite')
    arrays = [np.asarray(a, dtype=np.complex128).copy() for a in layers]
    columns = []
    for array in arrays:
        for index in np.ndindex(array.shape):
            array[index] += 1j*step
            product = arrays[0]
            for layer in arrays[1:]: product = layer @ product
            columns.append(product.reshape(-1).imag/step)
            array[index] -= 1j*step
    return np.stack(columns, axis=1)


def best_feasible_candidate(candidates):
    return min((c for c in candidates if c['feasible']), key=lambda c:c['risk'], default=None)


def enumerate_support_frontier(s, n, sigma2, k, width):
    """Enumerate small diagonal support programs, plus seeded rotated candidates.

    This is finite numerical evidence, not a global proof over rotations.
    """
    from itertools import combinations
    from scipy.optimize import minimize
    s=np.asarray(s,dtype=float); d=len(s)
    if d<1 or d>6 or s.ndim!=1 or np.any(s<0) or not np.isclose(s.sum(),1) or n<1 or sigma2<0 or k<0 or not 1<=width<=d: raise ValueError('invalid tiny frontier inputs')
    S=np.diag(s); q=(n+1)/n*S+(1+sigma2)/n*np.eye(d)
    rng=np.random.default_rng(731); results={}; all_candidates=[]
    for branch,count in [('F',min(width-1,d-1)),('Z',min(width,d))]:
        branch_candidates=[]
        for subset in combinations(range(d),count):
            basis=np.eye(d)[:,subset]
            for rotation in range(4):
                if rotation:
                    basis=np.linalg.qr(rng.normal(size=(d,d)))[0][:,:count]
                def matrix(x):
                    return ((x[-1]*np.eye(d)) if branch=='F' else np.zeros((d,d))) + (basis*x[:count])@basis.T
                def objective(x):
                    H=matrix(x)
                    return float(1-2*np.trace(S@H)+np.trace(q@H@H))
                def cost(x): return float(np.sum(x[:count])+(x[-1] if branch=='F' else 0))
                size=count+(branch=='F')
                if size:
                    fit=minimize(objective,np.zeros(size),method='SLSQP',bounds=[(0,None)]*size,constraints=[{'type':'ineq','fun':lambda x:k-cost(x)}],options={'ftol':1e-13,'maxiter':1000})
                    feasible=bool(fit.success and np.min(fit.x)>=-1e-9 and cost(fit.x)<=k+1e-9)
                    risk=objective(fit.x)
                else: feasible=True; risk=1.
                candidate={'branch':branch,'support':list(subset),'rotation':rotation,'risk':risk,'feasible':feasible}
                all_candidates.append(candidate)
                branch_candidates.append(candidate)
        results[branch]=best_feasible_candidate(branch_candidates)
    results['candidates']=all_candidates
    results['qualification']='finite numerical search; no global rotation proof'
    return results
