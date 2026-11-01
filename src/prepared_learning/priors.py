"""Finite-source assets and constrained empirical query-loss fitting."""
import numpy as np
from scipy.optimize import minimize, nnls
from .records import PriorAsset, QPData, NumericDiagnostics, BranchResult, SupportMetadata, content_hash
from .metrics import CompactMetric


def _blocks(tasks, d=None):
    if not tasks: raise ValueError('source tasks required')
    seen=set(); block_keys=set()
    for t in tasks:
        if t.source_key in seen: raise ValueError('duplicate source task')
        seen.add(t.source_key)
        for c in (t.A,t.B):
            if c.task_key in block_keys: raise ValueError('duplicate observation block')
            block_keys.add(c.task_key)
            X=np.asarray(c.X); y=np.asarray(c.y)
            if X.ndim!=2 or len(X)==0 or y.shape!=(len(X),) or (d is not None and X.shape[1]!=d) or not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError('invalid observed block')
            yield c


def _asset(cov, noise, metadata, basis=None, eigenvalues=None):
    cov=(np.asarray(cov)+np.asarray(cov).T)/2
    if basis is None:
        vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]
    else:
        vecs=np.asarray(basis,dtype=float)
        if vecs.shape!=cov.shape or not np.allclose(vecs.T@vecs,np.eye(len(cov)),atol=1e-10): raise ValueError('invalid prior basis')
        diagonal=vecs.T@cov@vecs
        if not np.allclose(diagonal,np.diag(np.diag(diagonal)),atol=1e-10): raise ValueError('basis does not diagonalize covariance')
        vals=np.diag(diagonal).copy() if eigenvalues is None else np.asarray(eigenvalues,dtype=float).copy()
        if vals.shape!=(len(cov),) or not np.allclose(diagonal,np.diag(vals),atol=1e-10): raise ValueError('invalid prior eigenvalues')
    if vals.min() < -1e-10: raise ValueError('prior covariance is not PSD')
    vals=np.maximum(vals,0.)
    if eigenvalues is None or not np.isclose(vals.sum(),1.,atol=1e-12,rtol=0): vals/=vals.sum()
    cov=(vecs*vals)@vecs.T
    payload={'covariance':cov,'basis':vecs,'eigenvalues':vals,'noise_variance':noise,'information_access':'estimated','fit_metadata':metadata}
    return PriorAsset(cov,vecs,vals,noise,'estimated',metadata,content_hash(payload))


def estimate_cross_moment(tasks,d):
    blocks=list(_blocks(tasks,d)); raw=np.zeros((d,d))
    for a,b in zip(blocks[::2],blocks[1::2]):
        za=a.X.T@a.y/len(a.y); zb=b.X.T@b.y/len(b.y)
        raw+=(np.outer(za,zb)+np.outer(zb,za))/2
    raw/=len(tasks); vals,vecs=np.linalg.eigh(raw); positive=np.maximum(vals,0); mass=float(positive.sum())
    cov=(vecs*positive)@vecs.T/mass if mass>0 else np.eye(d)/d
    metadata={'estimator':'split_cross_moment','raw':raw,'removed_negative_mass':float(-np.minimum(vals,0).sum()),'projection_frobenius':float(np.linalg.norm(np.minimum(vals,0))),'fallback':None if mass>0 else 'isotropic','source_hash':content_hash(tasks),'tasks':len(tasks)}
    return _asset(cov,None,metadata)


def estimate_moment_directions(tasks,rank):
    blocks=list(_blocks(tasks)); d=blocks[0].X.shape[1]
    if not isinstance(rank,int) or not 1<=rank<=d: raise ValueError('invalid rank')
    moment=np.zeros((d,d)); count=0
    for c in blocks:
        if c.X.shape[1]!=d: raise ValueError('dimension mismatch')
        moment+=(c.X.T*(c.y*c.y))@c.X; count+=len(c.y)
    vals,vecs=np.linalg.eigh(moment/count); order=np.argsort(vals)[::-1]; vecs=vecs[:,order]
    cov=vecs[:,:rank]@vecs[:,:rank].T/rank
    return _asset(cov,None,{'estimator':'moment_directions','basis_provenance':'y_squared_xx','rank':rank,'moment_eigenvalues':vals[order], 'variance_interpretation':'equal-variance surrogate on selected directions','source_hash':content_hash(tasks)},basis=vecs,eigenvalues=np.r_[np.full(rank,1/rank),np.zeros(d-rank)])


def shrink_prior(prior,gamma):
    if prior.information_access=='oracle' or not np.isfinite(gamma) or not 0<=gamma<=1: raise ValueError('invalid prior or shrinkage')
    d=len(prior.covariance); return _asset((1-gamma)*prior.covariance+gamma*np.eye(d)/d,prior.noise_variance,{**prior.fit_metadata,'gamma':float(gamma),'parent_hash':prior.asset_hash},basis=prior.basis,eigenvalues=(1-gamma)*prior.eigenvalues+gamma/d)


def estimate_noise(tasks):
    blocks=list(_blocks(tasks)); means=np.array([np.mean(np.r_[a.y*a.y,b.y*b.y]) for a,b in zip(blocks[::2],blocks[1::2])])
    return {'variance':float(max(0.,means.mean()-1)), 'task_standard_error':float(means.std(ddof=1)/np.sqrt(len(means))) if len(means)>1 else None,'uncertainty_unit':'task','tasks':len(means)}


def _limits(budget,d):
    if budget.depth<2 or budget.learning_rate!=1: raise ValueError('invalid budget')
    width=min(d,(budget.trainable_parameter_cap-(budget.depth-2))//(d+1)); k=(budget.initialization_norm_cap/(budget.depth-1))**(budget.depth-1)
    if width<1 or not np.isfinite(k) or k<0: raise ValueError('infeasible budget')
    return width,k


def build_empirical_qp(tasks,basis,budget,branch,regularization):
    basis=np.asarray(basis,dtype=float); d=basis.shape[0]
    if basis.shape!=(d,d) or not np.allclose(basis.T@basis,np.eye(d),atol=1e-10) or branch not in ('F','Z') or not np.isfinite(regularization) or regularization<0: raise ValueError('invalid empirical inputs')
    width,k=_limits(budget,d); p=min(width-1,d-1) if branch=='F' else min(width,d)
    blocks=list(_blocks(tasks,d)); h=np.zeros((p+(branch=='F'),)*2); l=np.zeros(h.shape[0]); const=0.; count=0
    for a,b in zip(blocks[::2],blocks[1::2]):
        for context,query in ((a,b),(b,a)):
            z=context.X.T@context.y/len(context.y); V=basis[:,:p]; design=(query.X@V)*(V.T@z)
            if branch=='F': design=np.column_stack((design,query.X@z-design.sum(axis=1)))
            h+=design.T@design; l-=2*design.T@query.y; const+=query.y@query.y; count+=len(query.y)
    h=h/count+regularization*np.eye(len(l)); l/=count; const/=count
    if branch=='Z': A=np.vstack((np.eye(p),-np.ones(p)))
    else:
        A=np.zeros((p+2,p+1)); A[:p,:p]=np.eye(p); A[:p,-1]=-1; A[p,-1]=1; A[-1,:p]=-1; A[-1,-1]=p-1
    offset=np.zeros(len(A)); offset[-1]=k
    return QPData(float(const),l,h,A,offset)


def fit_empirical_gains(qp,budget,basis,branch):
    basis=np.asarray(basis); d=len(basis); width,k=_limits(budget,d); size=len(qp.linear)
    if branch not in ('F','Z'): raise ValueError('invalid branch')
    if k:
        magnitude=max(float(np.max(np.abs(qp.linear),initial=0)*k),float(np.max(np.abs(qp.hessian),initial=0)*k*k),1e-300)
        res=minimize(lambda y:(qp.linear@(k*y)+(k*y)@qp.hessian@(k*y))/magnitude,np.zeros(size),jac=lambda y:k*(qp.linear+2*qp.hessian@(k*y))/magnitude,method='SLSQP',constraints={'type':'ineq','fun':lambda y:qp.A@y+qp.offset/k,'jac':lambda y:qp.A},options={'ftol':1e-12,'maxiter':2000})
        x=k*res.x
    else: res=None; x=np.zeros(size)
    if branch=='Z': x=np.maximum(x,0); total=x.sum()
    else: x[-1]=max(0,x[-1]); x[:-1]=np.maximum(x[:-1],x[-1]); total=x[:-1].sum()-(size-2)*x[-1]
    if total>k: x*=k/total
    # Polish the active KKT system by least squares, valid also for singular H.
    slack=qp.A@x+qp.offset; active=np.flatnonzero(slack<=1e-7*max(k,1e-12)); C=qp.A[active]
    K=np.block([[2*qp.hessian,-C.T],[C,np.zeros((len(active),len(active)))]])
    rhs=np.r_[-qp.linear,-qp.offset[active]]
    solution=np.linalg.lstsq(K,rhs,rcond=None)[0]; candidate=solution[:size]; v=solution[size:]
    old=qp.linear@x+x@qp.hessian@x
    if np.all(np.isfinite(solution)) and np.min(qp.A@candidate+qp.offset)>=-1e-12*max(k,1.) and np.min(v,initial=0)>=-1e-10 and qp.linear@candidate+candidate@qp.hessian@candidate<=old+1e-12*max(1.,abs(old)):
        x=candidate
        if branch=='Z': x=np.maximum(x,0.); total=x.sum()
        else: x[-1]=max(0.,x[-1]); x[:-1]=np.maximum(x[:-1],x[-1]); total=x[:-1].sum()-(size-2)*x[-1]
        if total>k: x*=k/total
    slack=qp.A@x+qp.offset; grad=qp.linear+2*qp.hessian@x; active=np.flatnonzero(slack<=1e-8*max(k,1e-12)); multipliers=np.zeros(len(slack))
    if len(active): multipliers[active]=nnls(qp.A[active].T,grad,maxiter=10000)[0]
    residual=grad-qp.A.T@multipliers; objective=float(qp.constant+qp.linear@x+x@qp.hessian@x)
    from .certificates import empirical_residual_bound
    lower=empirical_residual_bound(qp,x,multipliers,k)
    bound=max(0.,objective-lower)
    tolerance=1e-7*max(1.,abs(objective),abs(qp.constant))
    diag=NumericDiagnostics({'slacks':slack,'minimum_slack':float(slack.min())},{'stationarity_inf':float(np.max(np.abs(residual),initial=0)),'complementarity_inf':float(np.max(np.abs(multipliers*slack),initial=0)),'acceptance_tolerance':tolerance,'solver_success':res is None or bool(res.success),'solver_message':'zero budget' if res is None else str(res.message)}, {'primal':objective,'residual_lower_bound':lower,'gap_bound':bound,'multipliers':multipliers},{'certificate_level':'float64_diagnostic','treatment':'bounded-domain residual; no inverse or rigorous enclosure'})
    if branch=='F': floor=float(x[-1]); inc=x[:-1]-floor; vectors=basis[:,:len(inc)]
    else:
        gains=np.r_[x,np.zeros(d-len(x))]; floor=float(gains.min()); inc=gains-floor; vectors=basis
    meta=SupportMetadata('analytic',int(floor==0)*(d-int(np.count_nonzero(inc))),int(d-np.count_nonzero(inc)),tuple(np.flatnonzero(inc)))
    metric=CompactMetric(floor,vectors,inc,meta)
    status='completed' if (res is None or res.success) and slack.min()>=-1e-9*max(k,1.) and bound<=tolerance else 'failed'
    return BranchResult(branch,x,metric,objective,diag,status)
