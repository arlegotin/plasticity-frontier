"""Oracle quadratic programs in original units."""
import numpy as np
from scipy.optimize import minimize, nnls
from .records import QPData, NumericDiagnostics


def build_oracle_qp(s, n, sigma2, k, width, branch):
    s=np.asarray(s,dtype=float)
    if s.ndim!=1 or not len(s) or not np.all(np.isfinite(s)) or np.any(s<0) or np.any(np.diff(s)>0) or not np.isclose(s.sum(),1.,atol=1e-12) or n<1 or not np.isfinite(sigma2) or sigma2<0 or not np.isfinite(k) or k<0 or not isinstance(width,int) or width<1 or branch not in ('F','Z'):
        raise ValueError('invalid oracle QP inputs')
    d=len(s); q=(n+1)/n*s+(1+sigma2)/n
    if branch=='F':
        j=min(width-1,d-1); linear=-2*np.r_[s[:j],s[j:].sum()]; diag=np.r_[q[:j],q[j:].sum()]
        A=np.zeros((j+2,j+1)); A[:j,:j]=np.eye(j); A[:j,-1]=-1; A[j,-1]=1; A[-1,:j]=-1; A[-1,-1]=j-1
    else:
        p=min(width,d); linear=-2*s[:p]; diag=q[:p]; A=np.vstack((np.eye(p),-np.ones(p)))
    offset=np.zeros(len(A)); offset[-1]=k
    return QPData(1.,linear,np.diag(diag),A,offset)


def diagnose_qp(qp,x):
    x=np.asarray(x,dtype=float); slack=qp.A@x+qp.offset; grad=qp.linear+2*qp.hessian@x
    scale=max(float(np.max(np.abs(x),initial=0)),np.finfo(float).tiny)
    active=np.flatnonzero(slack <= 1e-9*scale)
    multipliers=np.zeros(len(slack))
    if len(active): multipliers[active]=nnls(qp.A[active].T,grad,maxiter=10000)[0]
    residual=grad-qp.A.T@multipliers
    primal=float(qp.constant+qp.linear@x+x@qp.hessian@x)
    diag=np.diag(qp.hessian)
    dual=float(qp.constant-multipliers@qp.offset-.25*np.sum((qp.linear-qp.A.T@multipliers)**2/diag))
    return NumericDiagnostics({'slacks':slack,'minimum_slack':float(slack.min())}, {'stationarity_inf':float(np.max(np.abs(residual),initial=0)), 'complementarity_inf':float(np.max(np.abs(multipliers*slack),initial=0)), 'active_set':active.tolist()}, {'multipliers':multipliers,'primal':primal,'dual':dual,'gap':primal-dual,'minimum_multiplier':float(multipliers.min())}, {'certificate_level':'float64_diagnostic','treatment':'ordinary float64; no rigorous enclosure'})


def solve_qp(qp, variable_scale):
    if not np.isfinite(variable_scale) or variable_scale<0: raise ValueError('invalid variable scale')
    if variable_scale==0:
        x=np.zeros(len(qp.linear)); result=None; repair=0.
    else:
        scale=variable_scale; magnitude=max(float(np.max(np.abs(qp.linear))*scale),float(np.max(np.abs(qp.hessian))*scale*scale))
        if magnitude==0 or not np.isfinite(magnitude): raise ValueError('objective scaling not representable')
        A=qp.A; offset=qp.offset/scale
        result=minimize(lambda y:(qp.linear@(scale*y)+(scale*y)@qp.hessian@(scale*y))/magnitude, np.zeros(len(qp.linear)), jac=lambda y:scale*(qp.linear+2*qp.hessian@(scale*y))/magnitude, method='SLSQP',constraints={'type':'ineq','fun':lambda y:A@y+offset,'jac':lambda y:A},options={'ftol':1e-14,'maxiter':2000})
        y=np.asarray(result.x,dtype=float)
        if y.shape!=qp.linear.shape or not np.all(np.isfinite(y)):
            raise ValueError('nonfinite or invalid solver iterate: '+str(result.message))
        # Test the diagonal unconstrained minimizer directly before treating any
        # near-zero slack as active. This retains very weak positive directions.
        unconstrained=-qp.linear/(2*np.diag(qp.hessian))/scale
        polish_valid=True
        if np.all(np.isfinite(unconstrained)) and np.all(A@unconstrained+offset>=0):
            y=unconstrained
        else:
            active=np.flatnonzero(A@y+offset < 1e-8)
            H=2*qp.hessian*scale*scale/magnitude; l=qp.linear*scale/magnitude; C=A[active]
            K=np.block([[H,-C.T],[C,np.zeros((len(active),len(active)))]])
            try:
                solution=np.linalg.solve(K,np.r_[-l,-offset[active]])
                candidate=solution[:len(y)]; multipliers=solution[len(y):]
                polish_valid=(np.all(np.isfinite(solution)) and np.all(multipliers>=0)
                              and np.min(A@candidate+offset)>=-1e-12)
                if polish_valid: y=candidate
            except np.linalg.LinAlgError:
                polish_valid=False
        x=scale*y; before=x.copy()
        # Constraints encode either F or Z; remove rounding-scale violations.
        if len(qp.A)==len(x)+1 and np.all(qp.A[:-1]==np.eye(len(x))):
            x=np.maximum(x,0); total=x.sum()
        else:
            x[-1]=max(0.,x[-1]); x[:-1]=np.maximum(x[:-1],x[-1]); total=x[:-1].sum()-(len(x)-2)*x[-1]
        if total>qp.offset[-1]: x*=qp.offset[-1]/total
        repair=float(np.max(np.abs(x-before),initial=0))
    diagnostic=diagnose_qp(qp,x)
    from dataclasses import replace
    diagnostic=replace(diagnostic,kkt={**diagnostic.kkt,'solver_success':True if result is None else bool(result.success and polish_valid),'solver_message':'zero budget' if result is None else str(result.message),'options':{'ftol':1e-14,'maxiter':2000},'variable_scale':variable_scale,'repair_inf':repair,'active_set_polish_valid':True if result is None else bool(polish_valid)})
    return x,diagnostic
