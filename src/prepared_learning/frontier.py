"""Joint norm and parameter frontier for the aligned oracle."""
import math
import numpy as np
from dataclasses import replace
from .baselines import solve_rank_gains
from .metrics import CompactMetric, metric_resources
from .quadratic import build_oracle_qp, solve_qp, diagnose_qp
from .records import BranchResult, FrontierResult, SupportMetadata


def solve_frontier(prior,budget,task):
    d=task.input_dimension; L=budget.depth
    if L<2 or budget.learning_rate!=1 or task.target_context_size<1 or d<1 or not np.isfinite(budget.initialization_norm_cap) or budget.initialization_norm_cap<0: raise ValueError('invalid budget or task')
    m=min(d,(budget.trainable_parameter_cap-(L-2))//(d+1))
    if m<1: raise ValueError('infeasible architecture')
    B=budget.initialization_norm_cap
    try: k=0. if B==0 else math.exp((L-1)*(math.log(B)-math.log(L-1)))
    except OverflowError: raise ValueError('budget power not representable') from None
    if B>0 and k==0: raise ValueError('budget power not representable')
    s=np.asarray(prior.eigenvalues,dtype=float); basis=np.asarray(prior.basis,dtype=float)
    if s.shape!=(d,) or basis.shape!=(d,d) or not np.allclose(basis.T@basis,np.eye(d),atol=1e-12): raise ValueError('invalid oracle basis')
    branches=[]
    for branch in ('F','Z'):
        qp=build_oracle_qp(s,task.target_context_size,prior.noise_variance,k,m,branch)
        try:
            x,diag=solve_qp(qp,min(k,task.target_context_size/(task.target_context_size+1)))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            x=np.zeros(len(qp.linear)); diag=diagnose_qp(qp,x)
            diag=replace(diag,kkt={**diag.kkt,'solver_success':False,'solver_message':str(exc)})
        if branch=='F':
            floor=float(x[-1]); inc=x[:-1]-floor; vectors=basis[:,:len(inc)]
        else:
            gains=np.r_[x,np.zeros(d-len(x))]; floor=float(gains.min()); inc=gains-floor; vectors=basis
            independent=solve_rank_gains(s,task.target_context_size,prior.noise_variance,k,m)
            diag=replace(diag,kkt={**diag.kkt,'rank_oracle_max_error':float(np.max(np.abs(gains-independent)))})
        meta=SupportMetadata('analytic',int(floor==0)*(d-int(np.count_nonzero(inc))),int(d-np.count_nonzero(inc)),tuple(np.flatnonzero(inc)))
        metric=CompactMetric(floor,vectors,inc,meta)
        status='completed' if diag.kkt['solver_success'] and diag.feasibility['minimum_slack']>=-1e-10*max(min(k,1.),1e-300) and diag.kkt['stationarity_inf']<1e-9 and diag.kkt['complementarity_inf']<1e-9 else 'failed'
        branches.append(BranchResult(branch,x,metric,diag.dual['primal'],diag,status))
    F,Z=branches; gap=abs(F.objective-Z.objective)
    errors=max(abs(F.diagnostics.dual['gap']),abs(Z.diagnostics.dual['gap']),1e-14)
    if F.status!='completed' or Z.status!='completed' or gap<=10*errors:
        selected=('F','Z'); status='unresolved'
    else:
        selected=('F',) if F.objective<Z.objective else ('Z',); status='selected'
    return FrontierResult(F,Z,selected,status)
