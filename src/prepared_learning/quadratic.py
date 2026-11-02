"""Oracle quadratic programs in original units."""
from dataclasses import replace
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
    """Return a candidate and original-unit diagnostics, including admission.

    Polish is optional. A nonzero-budget solve succeeds only when the native
    solve succeeds and the final repaired point passes the frontier KKT gate.
    The independent Z rank comparison remains the frontier's responsibility.
    """
    if not np.isfinite(variable_scale) or variable_scale<0: raise ValueError('invalid variable scale')
    active=np.array([],dtype=int); multipliers=np.array([]); candidate=None
    rejection_reasons=[]; reconstructed=np.array([],dtype=int)
    face_repair=0.; feasibility_repair=0.; polish_valid=True
    origin='zero_budget'; native_iterate=None
    if variable_scale==0:
        x=np.zeros(len(qp.linear)); result=None; repair=0.
    else:
        scale=variable_scale; magnitude=max(float(np.max(np.abs(qp.linear))*scale),float(np.max(np.abs(qp.hessian))*scale*scale))
        if magnitude==0 or not np.isfinite(magnitude): raise ValueError('objective scaling not representable')
        A=qp.A; offset=qp.offset/scale
        result=minimize(lambda y:(qp.linear@(scale*y)+(scale*y)@qp.hessian@(scale*y))/magnitude, np.zeros(len(qp.linear)), jac=lambda y:scale*(qp.linear+2*qp.hessian@(scale*y))/magnitude, method='SLSQP',constraints={'type':'ineq','fun':lambda y:A@y+offset,'jac':lambda y:A},options={'ftol':1e-14,'maxiter':2000})
        native_rejections=[]
        try:
            y=np.asarray(result.x,dtype=float)
        except (TypeError,ValueError):
            y=None; native_rejections.append('malformed_iterate')
        if y is not None:
            if y.shape!=qp.linear.shape: native_rejections.append('malformed_shape')
            if not np.all(np.isfinite(y)): native_rejections.append('nonfinite_iterate')
        if native_rejections:
            # This zero point is only a finite diagnostic substitute. Never
            # admit it, even if native success or its own KKT checks pass.
            x=np.zeros(len(qp.linear)); diagnostic=diagnose_qp(qp,x)
            diagnostic=replace(diagnostic,kkt={**diagnostic.kkt,
                'solver_success':False,'solver_message':str(result.message),
                'native_solver_success':bool(result.success),'native_solver_status':int(result.status),
                'native_scaled_iterate':_json_safe_iterate(result.x),
                'native_iterate_shape':None if y is None else list(y.shape),
                'native_iterate_rejection_reasons':native_rejections,
                'candidate_origin':'invalid_native_zero_substitute',
                'final_quality_passed':qp_quality_passes(diagnostic,qp.offset[-1]),
                'options':{'ftol':1e-14,'maxiter':2000},'variable_scale':variable_scale,
                'repair_inf':None,'feasibility_repair_inf':0.,
                'active_set_polish_valid':False,'active_set_polish_rejection_reasons':[],
                'active_set_polish_constraints':[],'active_set_polish_candidate_scaled':None,
                'active_set_polish_multipliers_scaled':[],
                'active_face_reconstruction_indices':[],'active_face_reconstruction_size':0,
                'active_face_reconstruction_inf':0.})
            return x,diagnostic
        native_iterate=y.copy(); origin='native'
        # Test the diagonal unconstrained minimizer directly before treating any
        # near-zero slack as active. This retains very weak positive directions.
        unconstrained=-qp.linear/(2*np.diag(qp.hessian))/scale
        if np.all(np.isfinite(unconstrained)) and np.all(A@unconstrained+offset>=0):
            y=unconstrained; origin='unconstrained'
        else:
            active=np.flatnonzero(A@y+offset < 1e-8)
            H=2*qp.hessian*scale*scale/magnitude; l=qp.linear*scale/magnitude; C=A[active]
            K=np.block([[H,-C.T],[C,np.zeros((len(active),len(active)))]])
            try:
                solution=np.linalg.solve(K,np.r_[-l,-offset[active]])
                candidate=solution[:len(y)]; multipliers=solution[len(y):]
                if not np.all(np.isfinite(solution)): rejection_reasons.append('nonfinite_solution')
                if np.any(multipliers<0): rejection_reasons.append('negative_multiplier')
                if np.min(A@candidate+offset)<-1e-12: rejection_reasons.append('infeasible_candidate')
                polish_valid=not rejection_reasons
                if polish_valid: y=candidate; origin='active_set_polish'
            except np.linalg.LinAlgError:
                polish_valid=False; rejection_reasons.append('linear_solve_failure')
        x=scale*y; before=x.copy()
        # Constraints encode either F or Z; remove rounding-scale violations.
        if len(qp.A)==len(x)+1 and np.all(qp.A[:-1]==np.eye(len(x))):
            # Positive production-polish lower-bound multipliers force zeros
            # by complementarity. Do not infer support from a gain cutoff or
            # the independent rank oracle. Zero multipliers do not select a face.
            if origin=='active_set_polish':
                reconstructed=active[(active<len(x)) & (multipliers>0)]
                x[reconstructed]=0.
                face_repair=float(np.max(np.abs(x-before),initial=0))
            after_face=x.copy()
            x=np.maximum(x,0); total=x.sum()
        else:
            after_face=x.copy()
            x[-1]=max(0.,x[-1]); x[:-1]=np.maximum(x[:-1],x[-1]); total=x[:-1].sum()-(len(x)-2)*x[-1]
        if total>qp.offset[-1]: x*=qp.offset[-1]/total
        feasibility_repair=float(np.max(np.abs(x-after_face),initial=0))
        repair=float(np.max(np.abs(x-before),initial=0))
    diagnostic=diagnose_qp(qp,x)
    quality=qp_quality_passes(diagnostic,qp.offset[-1])
    diagnostic=replace(diagnostic,kkt={**diagnostic.kkt,
        'solver_success':bool(quality and (result is None or result.success)),
        'solver_message':'zero budget' if result is None else str(result.message),
        'native_solver_success':None if result is None else bool(result.success),
        'native_solver_status':None if result is None else int(result.status),
        'native_scaled_iterate':native_iterate,
        'candidate_origin':origin,'final_quality_passed':quality,
        'options':{'ftol':1e-14,'maxiter':2000},'variable_scale':variable_scale,
        # Total change from the admitted candidate, including face reconstruction.
        'repair_inf':repair,'feasibility_repair_inf':feasibility_repair,
        'active_set_polish_valid':bool(polish_valid),
        'active_set_polish_rejection_reasons':rejection_reasons,
        'active_set_polish_constraints':active.tolist(),
        # Rejected nonfinite entries are explicit nulls, so failure diagnostics
        # remain JSON-serializable without concealing the rejection reason.
        'active_set_polish_candidate_scaled':None if candidate is None else [float(v) if np.isfinite(v) else None for v in candidate],
        'active_set_polish_multipliers_scaled':[float(v) if np.isfinite(v) else None for v in multipliers],
        'active_face_reconstruction_indices':reconstructed.tolist(),
        'active_face_reconstruction_size':len(reconstructed),
        'active_face_reconstruction_inf':face_repair})
    return x,diagnostic


def _json_safe_iterate(value):
    """Retain invalid native values and shape, with nonfinite numbers as null."""
    if isinstance(value,np.ndarray): return _json_safe_iterate(value.tolist())
    if isinstance(value,np.generic): return _json_safe_iterate(value.item())
    if isinstance(value,(list,tuple)): return [_json_safe_iterate(v) for v in value]
    if isinstance(value,float): return value if np.isfinite(value) else None
    if value is None or isinstance(value,(str,int,bool)): return value
    return repr(value)


def qp_quality_passes(diagnostic,k):
    """The shared final original-unit gate; ordinary float64 diagnostics only."""
    return bool(diagnostic.feasibility['minimum_slack']>=-1e-10*max(min(k,1.),1e-300)
                and diagnostic.dual['minimum_multiplier']>=0
                and np.all(np.isfinite(diagnostic.dual['multipliers']))
                and 0<=diagnostic.kkt['stationarity_inf']<1e-9
                and 0<=diagnostic.kkt['complementarity_inf']<1e-9)
