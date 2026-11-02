"""Joint norm and parameter frontier for the aligned oracle."""
import math
import numpy as np
from dataclasses import replace
from .baselines import solve_rank_gains
from .metrics import CompactMetric, metric_resources
from .quadratic import build_oracle_qp, solve_qp, diagnose_qp, qp_quality_passes
from .records import BranchResult, FrontierResult, SupportMetadata, QPData, to_json_record


def _rank_one_ineligibility(spectrum, k, width):
    if len(spectrum)<2: return 'scalar_dimension'
    if width<2: return 'width_one'
    if k==0: return 'zero_budget'
    if spectrum[0]<=0 or np.any(spectrum[1:]!=0): return 'not_exact_rank_one'
    return None


def _canonical_rank_one_target(qp: QPData, spectrum: np.ndarray, k: float,
                               width: int, branch: str) -> np.ndarray | None:
    """Proven scalar optimizer, only for just-built rank-one oracle QPs.

    Strictly positive zero-direction curvature makes every secondary gain and
    the F floor exactly zero. No numerical cutoff or independent solution is
    involved. Width one and scalar F have different geometry.
    """
    if _rank_one_ineligibility(spectrum,k,width) is not None: return None
    size=min(width,len(spectrum))
    if (branch not in ('F','Z') or qp.linear.shape!=(size,)
            or qp.hessian.shape!=(size,size) or not np.all(np.isfinite(spectrum))
            or not np.isfinite(k) or k<=0):
        raise ValueError('rank-one target geometry not representable')
    curvature=np.diag(qp.hessian)
    if (not np.all(np.isfinite(qp.hessian)) or np.any(curvature<=0)
            or np.any(qp.hessian!=np.diag(curvature)) or curvature[0]<spectrum[0]):
        raise ValueError('rank-one target curvature not representable')
    # Direct division avoids overflow in 2*q0 or in k*q0 and avoids forming
    # an unrepresentable reciprocal. A positive ratio must never round to zero.
    ratio=float(spectrum[0])/float(curvature[0])
    if not np.isfinite(ratio) or ratio<=0:
        raise ValueError('rank-one scalar ratio not representable')
    target=np.zeros(size); target[0]=min(k,ratio)
    if not np.isfinite(target[0]) or target[0]<=0:
        raise ValueError('rank-one target not representable')
    return target


def _rank_comparison(diag, x, independent):
    gains=np.r_[x,np.zeros(len(independent)-len(x))]
    discrepancy=np.abs(gains-independent)
    denominators=np.maximum(np.abs(gains),np.abs(independent))
    relative=np.divide(discrepancy,denominators,out=np.zeros_like(discrepancy),where=denominators>0)
    return replace(diag,kkt={**diag.kkt,
        'rank_oracle_max_error':float(np.max(discrepancy)),
        'rank_oracle_max_relative_error':float(np.max(relative))})


def _diagnostic_snapshot(diag):
    """Detached JSON-native history; invalid attempted numbers remain null."""
    def finite_json(value):
        if isinstance(value,dict): return {key:finite_json(v) for key,v in value.items()}
        if isinstance(value,list): return [finite_json(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value): return None
        return value
    return finite_json(to_json_record(diag))


def _canonicalize_rank_one(qp, spectrum, k, width, branch, x, diag, independent):
    reason=_rank_one_ineligibility(spectrum,k,width)
    record={'status':'ineligible','reason':reason,'pre_variables':None,
            'pre_diagnostics':None,'target_variables':None,'corrected_indices':(),
            'attempted_correction_inf':None,'correction_inf':0.,'post_diagnostics':None}
    origin=diag.kkt.get('candidate_origin')
    if reason is None:
        # The original solver and independent gates are prerequisites, never
        # repaired by the structural proof or by the target's fresh diagnosis.
        if (not diag.kkt.get('native_solver_success') or not diag.kkt.get('solver_success')
                or not np.all(np.isfinite(x)) or not qp_quality_passes(diag,k)):
            record.update(status='skipped',reason='pre_admission_failed')
            diag=replace(diag,kkt={**diag.kkt,'solver_success':False})
        elif branch=='Z' and not diag.kkt['rank_oracle_max_relative_error']<=1e-6:
            record.update(status='skipped',reason='pre_independent_comparison_failed')
            diag=replace(diag,kkt={**diag.kkt,'solver_success':False})
        else:
            record.update(status='rejected',pre_variables=tuple(float(v) for v in x),
                          pre_diagnostics=_diagnostic_snapshot(diag))
            try:
                target=_canonical_rank_one_target(qp,spectrum,k,width,branch)
                if target is None: raise ValueError('rank-one target unavailable')
                record.update(target_variables=tuple(float(v) for v in target),
                              corrected_indices=tuple(int(i) for i in np.flatnonzero(target!=x)),
                              attempted_correction_inf=float(np.max(np.abs(target-x),initial=0)))
                post=diagnose_qp(qp,target)
                if branch=='Z': post=_rank_comparison(post,target,independent)
                record['post_diagnostics']=_diagnostic_snapshot(post)
                # The quality gate does not inspect objective finiteness. Keep
                # attempted arithmetic failures explicit and strict-JSON safe.
                values=[post.feasibility['minimum_slack'],post.dual['primal'],
                        post.dual['dual'],post.dual['gap']]
                if not np.all(np.isfinite(values)) or not qp_quality_passes(post,k):
                    raise ValueError('post_quality_failed')
                if branch=='Z' and not post.kkt['rank_oracle_max_relative_error']<=1e-6:
                    raise ValueError('post_independent_comparison_failed')
                admitted=np.asarray(diag.kkt['admitted_candidate_original_units'],dtype=float)
                if admitted.shape!=target.shape or not np.all(np.isfinite(admitted)):
                    raise ValueError('original admitted candidate unavailable')
                total_repair=float(np.max(np.abs(target-admitted),initial=0))
                diag=replace(post,kkt={**diag.kkt,**post.kkt,'solver_success':True,
                                      'final_quality_passed':True,'repair_inf':total_repair})
                x=target; origin='rank_one_canonical'
                record.update(status='applied',reason='exact_rank_one_scalar_optimum',
                              correction_inf=record['attempted_correction_inf'])
            except (ValueError,RuntimeError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as exc:
                record['reason']=str(exc)
                diag=replace(diag,kkt={**diag.kkt,'solver_success':False})
    diag=replace(diag,kkt={**diag.kkt,'final_point_origin':origin,
                          'rank_one_canonicalization':record})
    return x,diag


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
        independent=None
        if branch=='Z':
            independent=solve_rank_gains(s,task.target_context_size,prior.noise_variance,k,m)
            diag=_rank_comparison(diag,x,independent)
        x,diag=_canonicalize_rank_one(qp,s,k,m,branch,x,diag,independent)
        if branch=='F':
            floor=float(x[-1]); inc=x[:-1]-floor; vectors=basis[:,:len(inc)]
        else:
            gains=np.r_[x,np.zeros(d-len(x))]; floor=float(gains.min()); inc=gains-floor; vectors=basis
        meta=SupportMetadata('analytic',int(floor==0)*(d-int(np.count_nonzero(inc))),int(d-np.count_nonzero(inc)),tuple(np.flatnonzero(inc)))
        metric=CompactMetric(floor,vectors,inc,meta)
        status='completed' if diag.kkt['solver_success'] and (branch!='Z' or diag.kkt['rank_oracle_max_relative_error']<=1e-6) and qp_quality_passes(diag,k) else 'failed'
        branches.append(BranchResult(branch,x,metric,diag.dual['primal'],diag,status))
    F,Z=branches; gap=abs(F.objective-Z.objective)
    errors=max(abs(F.diagnostics.dual['gap']),abs(Z.diagnostics.dual['gap']),1e-14)
    if F.status!='completed' or Z.status!='completed' or gap<=10*errors:
        selected=('F','Z'); status='unresolved'
    else:
        selected=('F',) if F.objective<Z.objective else ('Z',); status='selected'
    return FrontierResult(F,Z,selected,status)


def evaluate_frontier_grid(prior, budget, task, widths, norm_caps, risk_tolerance, budget_tolerance):
    """Adaptive sampled diagnostics; interpolation errors are not enclosures."""
    from .records import to_json_record
    if not widths or len(norm_caps)<2 or not np.isfinite(risk_tolerance) or not np.isfinite(budget_tolerance) or risk_tolerance<=0 or budget_tolerance<=0: raise ValueError('invalid grid')
    caps=sorted(set(float(x) for x in norm_caps))
    if len(caps)<2 or not np.all(np.isfinite(caps)) or caps[0]<0: raise ValueError('invalid grid caps')
    if any(type(w)!=int or w<1 or w>task.input_dimension for w in widths) or len(set(widths))!=len(widths): raise ValueError('invalid grid widths')
    rows=[]
    for width in widths:
        cache={}; unresolved=[]
        def solve(cap):
            if cap not in cache:
                b=replace(budget,initialization_norm_cap=cap,trainable_parameter_cap=(task.input_dimension+1)*width+budget.depth-2)
                try:
                    f=solve_frontier(prior,b,task); rec=to_json_record(f)
                    for name in ('F','Z'):
                        rec[name]['actual_resources']=to_json_record(metric_resources(getattr(f,name).metric,b.depth))
                    rec['status']='completed' if all(getattr(f,name).status=='completed' for name in ('F','Z')) else 'failed'
                    if rec['status']=='failed': rec['error']='one or more branch solvers failed'
                except (ValueError, OverflowError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    rec={'status':'failed','error':str(exc),'selection_status':'unresolved','selected_branches':[],'F':None,'Z':None}
                rec.update({'kind':'point','width':width,'norm_cap':cap,'certificate_level':'float64_diagnostic'})
                cache[cap]=rec
            return cache[cap]
        def refine(a,b,level=0):
            left,right=solve(a),solve(b)
            if left['status']=='failed' or right['status']=='failed':
                unresolved.append({'kind':'interval','width':width,'lower':a,'upper':b,'reason':'failed endpoint'}); return
            mid=a+(b-a)/2
            if mid==a or mid==b:
                unresolved.append({'kind':'interval','width':width,'lower':a,'upper':b,'reason':'floating-point resolution'}); return
            center=solve(mid)
            if center['status']=='failed':
                unresolved.append({'kind':'interval','width':width,'lower':a,'upper':b,'reason':'failed midpoint'}); return
            invalid=any(p['selection_status']=='unresolved' for p in (left,center,right))
            curve=max(abs(center[t]['objective']-(left[t]['objective']+right[t]['objective'])/2) for t in ('F','Z'))
            switch=left['selected_branches']!=right['selected_branches'] or left['selected_branches']!=center['selected_branches']
            if (curve>risk_tolerance or switch or invalid) and b-a>budget_tolerance and level<40:
                refine(a,mid,level+1); refine(mid,b,level+1)
            elif invalid or (curve>risk_tolerance and b-a<=budget_tolerance) or level>=40:
                unresolved.append({'kind':'interval','width':width,'lower':a,'upper':b,'reason':'unresolved numerical selection' if invalid else 'refinement limit','midpoint_error':curve})
        for a,b in zip(caps,caps[1:]): refine(a,b)
        rows.extend(cache[c] for c in sorted(cache)); rows.extend(unresolved)
    return rows
