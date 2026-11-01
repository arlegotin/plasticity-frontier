"""Local, source-only branch envelopes; ordinary float64 diagnostics, not enclosures.

The sensitivity and solver-error arguments use full physical gains. Validation
uses the original compressed QP coordinates: ||dx|| <= ||dg|| (F repeats its
floor d-j times). Thus the *unweighted* compressed gradient norm is a safe,
possibly loose supporting-hyperplane coefficient. No validation Hessian is
inverted or assumed numerically PSD: losses and gradients use residuals of the
actual fixed prediction design, a convex sum of squares in exact arithmetic.
"""
import math
import time
import numpy as np
from .metrics import scaled_norm
from .quadratic import build_oracle_qp, diagnose_qp
from .records import content_hash, to_json_record

_EPS = np.finfo(float).eps


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    array = np.asarray(value, dtype=float)
    _require(np.all(np.isfinite(array)), 'nonfinite arithmetic')
    return array


def _pad(length, scale):
    """Declared scale/reduction-aware engineering allowance, NOT directed rounding."""
    value = 64*_EPS*max(1, length)*float(scale)
    _require(math.isfinite(value) and (scale == 0 or value >= np.finfo(float).tiny),
             'arithmetic padding not representable')
    return value


def _validation_data(validation, basis, width, branch):
    d = len(basis); top = min(width-1, d-1) if branch == 'F' else min(width, d)
    V = basis[:, :top]; designs = []; targets = []; magnitudes = []
    operations = d + top + len(validation)
    for task in validation:
        for context, query in ((task.A, task.B), (task.B, task.A)):
            X, y, Q, target = map(_finite, (context.X, context.y, query.X, query.y))
            _require(X.ndim == Q.ndim == 2 and X.shape[1] == Q.shape[1] == d
                     and y.shape == (len(X),) and target.shape == (len(Q),)
                     and len(X) > 0 and len(Q) > 0, 'invalid validation arrays')
            z = X.T@y/len(y)
            design = (Q@V)*(V.T@z)
            # Absolute products retain cancellation scales, including the literal
            # complement I - U_top U_top^T; replacing it by omitted basis columns
            # would change the floating prediction map.
            az = np.abs(X).T@np.abs(y)/len(y)
            magnitude = (np.abs(Q)@np.abs(V))*(np.abs(V).T@az)
            if branch == 'F':
                design = np.column_stack((design, Q@z-design.sum(axis=1)))
                magnitude = np.column_stack((magnitude, np.abs(Q)@az+magnitude.sum(axis=1)))
            weight = math.sqrt(2*len(validation)*len(Q))
            designs.append(design/weight); targets.append(target/weight); magnitudes.append(magnitude/weight)
            operations += len(X)+len(Q)+d+top
    D, y = _finite(np.vstack(designs)), _finite(np.concatenate(targets))
    return D, y, scaled_norm(np.vstack(magnitudes)), operations


def _validation_design(validation, basis, width, branch):
    """Weighted design whose squared residual is mean two-direction query MSE."""
    D, y, _, _ = _validation_data(validation, basis, width, branch)
    return D, y


def _family(left, right, incumbent, validation, config):
    method = left['method_id']
    _require(method in ('B02', 'B03', 'B04'), 'unsupported method')
    prep, budget = config['preparation'], config['budget']
    d, n, L = prep['input_dimension'], prep['target_context_size'], budget['depth']
    _require(type(d) is type(n) is type(L) is int and d > 0 and n > 0 and L >= 2,
             'unsupported dimensions or depth')
    _require(prep['input_law'] == 'gaussian_isotropic' and prep['loss_normalization'] == 'squared_error_over_2n'
             and prep['task_second_moment_trace'] == 1 and budget['learning_rate'] == 1,
             'unsupported normalization')
    _require(method != 'B02' or L == 2, 'B02 depth mismatch')
    cap = budget['initialization_norm_cap']
    _require(math.isfinite(cap) and cap > 0, 'nonpositive or nonfinite budget')
    # Match each family's existing power evaluation; Method.prepare's initial
    # pow value is unused when it delegates B03/B04 gains to solve_frontier.
    k = ((cap/(L-1))**(L-1) if method == 'B02' else
         math.exp((L-1)*(math.log(cap)-math.log(L-1))))
    _require(1e-100 <= k <= 1e100, 'unsupported budget power or arithmetic scale')
    width = min(d, (budget['trainable_parameter_cap']-(L-2))//(d+1))
    _require(type(width) is int and width >= 1, 'invalid width')
    branches = ('Z',) if method == 'B02' else ('F',) if method == 'B03' else ('F','Z')
    _require('Z' not in branches or width < d, 'full-width Z is unsupported')
    a, b = float(left['gamma']), float(right['gamma'])
    _require(0 <= a < b <= 1 and a < (a+b)/2 < b and 1-a >= 1e-8,
             'unrepresentable interval or base reconstruction')
    _require(len(validation) >= 2, 'independent validation uncertainty unavailable')
    validation_hash = content_hash(validation)
    U = _finite(left['asset']['basis']); s = _finite(left['asset']['eigenvalues'])
    _require(U.shape == (d,d) and s.shape == (d,), 'invalid prior dimensions')
    orth_error = scaled_norm(U.T@U-np.eye(d))
    _require(orth_error <= _pad(d, 1.), 'basis is not numerically orthonormal')
    noise = float(left['asset']['noise_variance'])
    _require(math.isfinite(noise) and noise >= 0, 'invalid noise')
    base = (s-a/d)/(1-a)
    reconstruction_pad = _pad(d, max(1., scaled_norm(s)))/(1-a)
    _require(np.min(base) >= -reconstruction_pad and abs(base.sum()-1) <= d*reconstruction_pad,
             'invalid affine base spectrum')
    witness = left['asset']['fit_metadata']
    _require(bool(witness['parent_hash']) and bool(witness['source_hash']), 'missing prior witness')
    ses = []
    for row in (left,right):
        asset, state = row['asset'], row['prepared_state']
        _require(row['status'] == 'completed' and row['method_id'] == method and state['method_id'] == method,
                 'failed or changed method')
        _require(row['options'] == state['options'] == {} and row['tuning_kind'] == 'default'
                 and row['tuning_value'] is None and row['rank'] is None, 'unsupported fitted options')
        _require(state['budget'] == budget and state['preparation'] == prep, 'configuration drift')
        _require(state['prior'] == asset and row['asset_hash'] == asset['asset_hash']
                 and content_hash({key:value for key,value in asset.items() if key != 'asset_hash'}) == asset['asset_hash']
                 and content_hash(state) == row['prepared_hash'], 'inconsistent preparation witness')
        meta = asset['fit_metadata']
        _require(asset['information_access'] == 'estimated' and asset['noise_variance'] == noise
                 and asset['basis'] == left['asset']['basis'] and meta['gamma'] == row['gamma']
                 and {key:value for key,value in meta.items() if key != 'gamma'} == {key:value for key,value in witness.items() if key != 'gamma'},
                 'basis, noise or parent witness drift')
        spectrum = _finite(asset['eigenvalues'])
        expected = (1-row['gamma'])*base+row['gamma']/d
        _require(spectrum.shape == (d,) and np.all(spectrum >= 0) and np.all(np.diff(spectrum) <= 0)
                 and scaled_norm(spectrum-expected) <= d*reconstruction_pad
                 and scaled_norm(_finite(asset['covariance'])-(U*spectrum)@U.T) <= _pad(d,1.),
                 'inconsistent affine spectrum')
        _require(row['source_hash'] == witness['source_hash'] and row['validation_hash'] == validation_hash,
                 'observation witness drift')
        losses = _finite(row['validation_task_losses'])
        _require(losses.shape == (len(validation),) and np.all(losses >= 0), 'invalid task losses')
        loss, se = float(row['validation_loss']), row['validation_loss_se']
        _require(se is not None and math.isfinite(se) and se > 0 and math.isfinite(loss)
                 and abs(loss-float(losses.mean())) <= _pad(len(losses),max(1.,abs(loss)))
                 and abs(se-float(losses.std(ddof=1)/math.sqrt(len(losses)))) <= _pad(len(losses),se),
                 'missing, nonpositive or inconsistent uncertainty')
        ses.append(se)
    _require(any(incumbent['candidate_id'] == row['candidate_id'] and incumbent['prepared_hash'] == row['prepared_hash']
                 and incumbent['validation_loss'] == row['validation_loss'] for row in (left,right)),
             'incumbent must be an eligible interval endpoint')
    policy = config.get('information_policy', {})
    _require(policy.get('prior') != 'oracle' and (policy.get('noise_access','estimated') != 'known'
             or policy['permitted_noise_variance'] == noise), 'noise policy mismatch')
    _require(all(len(block.y) == n for task in validation for block in (task.A,task.B)), 'context normalization drift')
    c, rho = (n+1)/n, (1+noise)/n
    rho_lower = rho-_pad(3,rho)
    _require(rho_lower > 0, 'curvature not representable')
    derivative = scaled_norm(1/d-base)+math.sqrt(d)*reconstruction_pad
    K = max(1., abs(c*k-1))*derivative/rho_lower
    K += _pad(d+8,max(K,1.))
    return dict(method=method,d=d,n=n,depth=L,k=k,width=width,branches=branches,a=a,b=b,U=U,
                noise=noise,c=c,rho=rho_lower,K=K,tau=min(ses),base=base,
                reconstruction_pad=reconstruction_pad,orth_error=orth_error,
                parent_hash=witness['parent_hash'],basis_hash=content_hash(U),validation_hash=validation_hash)


def _solver_model(family, branch):
    """Uniform original-unit acceptance model, including multiplier scales.

    A strict feasible witness w has all slacks >= k/(2p) (Z), or
    k/(2j+2) (F: a_i=2b, b=k/(2j+2)). Nonnegative multipliers satisfy
    mu^T slack(w) <= (||grad||+||r||)||w-x|| + M epsC. This bounds
    their sum, so the arithmetic allowance does not assume that future
    multipliers have the same scale as those observed at the endpoints.
    """
    d,k,c,rho = (family[key] for key in ('d','k','c','rho'))
    j = min(family['width']-1,d-1)
    p = j+1 if branch == 'F' else min(family['width'],d)
    M = p+1
    epsF = 1e-10*max(min(k,1.),1e-300)
    projection = math.sqrt(d+3*j)+math.sqrt(d)*(2*j+1) if branch == 'F' else math.sqrt(p)+p+1
    # Slack feasibility implies these coordinate bounds even before repair.
    xmax = k+(2*j+2)*epsF if branch == 'F' else k+(p+1)*epsF
    qmax = c+(1+family['noise'])/family['n']
    gradmax = 2+2*d*qmax*xmax
    slater = k/(2*p)
    mu_bound = 4*(p*(gradmax+1e-9)*(xmax+k)+M*1e-9)/slater
    slack_scale = (2*p+1)*xmax+k
    rounding = {'stationarity': _pad(2*p+M+4, 2+2*d*qmax*xmax+M*mu_bound),
                'feasibility': _pad(p+2, slack_scale),
                'complementarity': _pad(p+4, mu_bound*slack_scale)}
    epsS,epsC,epsF_upper = 1e-9+rounding['stationarity'],1e-9+rounding['complementarity'],epsF+rounding['feasibility']
    G = 2*(1+(c+(1+family['noise'])/family['n'])*math.sqrt(d)*k)
    G += _pad(d+8,G)
    return {'p':p,'M':M,'stationarity_threshold':1e-9,'complementarity_threshold':1e-9,
            'feasibility_threshold':epsF,'rounding':rounding,'epsS':epsS,'epsC':epsC,'epsF':epsF_upper,
            'projection_factor':projection,'gradient_bound':G,'multiplier_sum_bound':mu_bound,
            'slater_minimum_slack':slater,'coordinate_bound':xmax,'rho_lower':rho}


def _solver_error(model, epsS, epsC, epsF):
    p,M,rho = model['p'],model['M'],model['rho_lower']
    e = model['projection_factor']*epsF
    # hypot avoids needlessly squaring a large/small stationarity bound.
    term = 8*rho*(M*epsC+model['gradient_bound']*e)
    _require(math.isfinite(term) and term > 0, 'solver-error arithmetic not representable')
    root = (math.sqrt(p)*epsS+math.hypot(math.sqrt(p)*epsS, math.sqrt(term)))/(4*rho)
    return root+_pad(p+M+12,root)


def _metric_record(x, U, branch):
    if branch == 'F':
        return float(x[-1]), U[:,:len(x)-1], x[:-1]-x[-1]
    return 0., U[:,:len(x)], x


def _check_metric(record, x, U, branch):
    floor,V,inc = _metric_record(x,U,branch)
    rU,ri = _finite(record['basis']),_finite(record['increments'])
    # Stored Z may use the full basis with an exactly zero trailing increment.
    _require(rU.ndim == 2 and ri.ndim == 1
             and record['floor'] == floor and rU.shape[0] == len(U) and rU.shape[1] == len(ri)
             and len(ri) >= len(inc) and np.array_equal(rU,U[:,:len(ri)])
             and np.array_equal(ri[:len(inc)],inc) and np.all(ri[len(inc):] == 0),
             'gain-to-metric witness mismatch')


def _probe(row, branch, family, data, model):
    started = time.perf_counter_ns()
    state = row['prepared_state']; U=family['U']; p=model['p']
    qp = build_oracle_qp(row['asset']['eigenvalues'],family['n'],family['noise'],family['k'],family['width'],branch)
    if family['method'] == 'B02':
        metric=state['metric']; gains=_finite(metric['increments'])
        _require(gains.shape == (family['d'],), 'invalid B02 gain array')
        x=gains[:p]
        _check_metric(metric,x,U,branch)
        diag=to_json_record(diagnose_qp(qp,x)); source='actual_b02_metric'
    else:
        result=state['solver_diagnostics'][branch]
        _require(result['status'] == 'completed' and result['branch'] == branch
                 and result['diagnostics']['kkt']['solver_success'] is True, 'branch solver did not succeed')
        if branch == 'Z':
            _require(0 <= result['diagnostics']['kkt']['rank_oracle_max_relative_error'] <= 1e-6,
                     'rank oracle disagreement')
        x=_finite(result['variables']); diag=result['diagnostics']; source='retained_frontier'
        _require(x.shape == (p,), 'invalid branch variables')
        _check_metric(result['metric'],x,U,branch)
    primal=float(qp.constant+qp.linear@x+x@qp.hessian@x)
    _finite([diag['dual'][key] for key in ('primal','dual','gap')])
    objective_pad=_pad(p+4,max(1.,abs(primal),float(np.abs(qp.linear)@np.abs(x)),float(np.abs(x)@np.abs(qp.hessian)@np.abs(x))))
    _require(abs(diag['dual']['primal']-primal) <= objective_pad, 'inconsistent QP objective witness')
    if family['method'] != 'B02':
        _require(math.isfinite(result['objective']) and abs(result['objective']-primal) <= objective_pad,
                 'inconsistent retained branch objective')
    mu=_finite(diag['dual']['multipliers'])
    _require(mu.shape == (model['M'],) and np.all(mu >= 0) and mu.sum() <= model['multiplier_sum_bound'],
             'invalid nonnegative multipliers')
    slack=qp.A@x+qp.offset; residual=qp.linear+2*qp.hessian@x-qp.A.T@mu
    stationarity=float(np.max(np.abs(residual))); complementarity=float(np.max(np.abs(mu*slack)))
    feasibility=max(0.,-float(slack.min()))
    # Check the actual original-unit equations as well as retained successful
    # diagnostics. A completed label never exempts these prerequisites.
    _require(stationarity < 1e-9 and complementarity < 1e-9 and feasibility <= model['feasibility_threshold']
             and 0 <= diag['kkt']['stationarity_inf'] < 1e-9 and 0 <= diag['kkt']['complementarity_inf'] < 1e-9
             and diag['feasibility']['minimum_slack'] >= -model['feasibility_threshold'], 'invalid solver acceptance residuals')
    pads=model['rounding']
    anchor_error=_solver_error(model,stationarity+pads['stationarity'],complementarity+pads['complementarity'],feasibility+pads['feasibility'])
    uniform_error=_solver_error(model,model['epsS'],model['epsC'],model['epsF'])
    D,y,absolute_design,operations=data
    residual=D@x-y; loss=float(residual@residual); gradient=2*D.T@residual
    task_count=len(row['validation_task_losses'])
    task_losses=task_count*np.sum(residual.reshape(task_count,-1)**2,axis=1)
    task_discrepancy=float(np.max(np.abs(task_losses-row['validation_task_losses'])))
    grad_norm=scaled_norm(gradient)
    # Covers design products, residual sums, gradient arithmetic, and the
    # canonical network-to-metric implementation discrepancy, at both anchors
    # and any accepted future output. Endpoint literal losses are also checked.
    length=operations+family['depth']*family['d']+p
    amplitude=absolute_design*scaled_norm(x)+scaled_norm(y)
    uniform_amplitude=absolute_design*math.sqrt(p)*model['coordinate_bound']+scaled_norm(y)
    loss_pad=_pad(length,max(1.,amplitude**2))
    gradient_pad=_pad(length,max(1.,2*absolute_design*amplitude))
    map_pad=_pad(length,max(1.,uniform_amplitude**2))
    distance=family['b']-family['a']; distance_pad=_pad(3,max(abs(family['a']),abs(family['b']),distance))
    optimizer_distance=family['K']*(distance+distance_pad)+anchor_error+uniform_error
    _require(optimizer_distance > 0 and math.isfinite(optimizer_distance), 'unrepresentable optimizer distance')
    lower=loss-loss_pad-(grad_norm+gradient_pad)*optimizer_distance-map_pad
    bound_pad=_pad(length,max(1.,abs(lower),loss,(grad_norm+gradient_pad)*optimizer_distance))
    lower-=bound_pad
    return {'probe_id':f"{row['candidate_id']}:branch_envelope:{branch}",'anchor_candidate_id':row['candidate_id'],
            'branch':branch,'gamma':row['gamma'],'eligible':False,'solution_source':source,
            'loss':loss,'gradient_norm':grad_norm,'loss_padding':loss_pad,'gradient_padding':gradient_pad,
            'uniform_map_loss_padding':map_pad,'bound_padding':bound_pad,'distance_padding':distance_pad,
            'anchor_solver_error':anchor_error,'uniform_solver_error':uniform_error,
            'optimizer_distance_upper':optimizer_distance,'lower_bound':lower,
            'original_unit_residuals':{'stationarity':stationarity,'complementarity':complementarity,'feasibility':feasibility},
            'multipliers':mu.tolist(),'task_loss_discrepancy':task_discrepancy,
            'validation_arithmetic':{'absolute_design_norm':absolute_design,'reduction_length':length},
            'elapsed_ns':time.perf_counter_ns()-started}


def _selected_branch(row, family):
    if family['method'] == 'B02': return 'Z'
    if family['method'] == 'B03': return 'F'
    frontier=row['prepared_state']['solver_diagnostics']
    _require(frontier['selection_status'] == 'selected' and len(frontier['selected_branches']) == 1,
             'unresolved prior branch comparison')
    selected=frontier['selected_branches'][0]
    _finite([frontier[name]['objective'] for name in ('F','Z')])
    gaps=_finite([frontier[name]['diagnostics']['dual']['gap'] for name in ('F','Z')])
    _require(abs(frontier['F']['objective']-frontier['Z']['objective']) > 10*max(1e-14,*np.abs(gaps)),
             'prior branch comparison within solver uncertainty')
    _require(selected in ('F','Z') and frontier[selected]['objective'] < frontier['Z' if selected == 'F' else 'F']['objective'],
             'inconsistent prior-selected branch')
    return selected


def branch_interval_check(left: dict, right: dict, incumbent: dict, validation: tuple, config: dict) -> dict:
    """Bound possible improvement locally, conditional on accepted future solves.

    Probes are noneligible, reuse observed tasks, and never mutate input rows.
    Unsupported prerequisites leave the caller's ordinary refinement intact.
    """
    started=time.perf_counter_ns()
    out={'status':'unsupported','certificate_level':'float64_diagnostic',
         'scope':'local_neighbor_interval_conditional_on_successful_solves',
         'interval':None,'method_id':left.get('method_id'),'incumbent_candidate_id':incumbent.get('candidate_id'),
         'objective_tolerance':None,'improvement_bound':None,'branches':{},'probes':[],
         'additional_labels':0,'additional_independent_tasks':0}
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise',under='raise'):
            f=_family(left,right,incumbent,validation,config)
            out.update(interval=[f['a'],f['b']],objective_tolerance=f['tau'],
                witnesses={'parent_hash':f['parent_hash'],'basis_hash':f['basis_hash'],'validation_hash':f['validation_hash'],
                           'noise_variance':f['noise'],'width':f['width'],'budget_hash':content_hash(config['budget']),
                           'preparation_hash':content_hash(config['preparation'])},
                error_model={'name':'uniform_original_unit_residual_complementarity_feasibility_v1',
                    'qualification':'ordinary float64 scale-aware padding; not a rigorous enclosure',
                    'future_output_contract':'conditional on nonnegative multipliers and the recorded original-unit acceptance thresholds; solver success throughout the interval is not asserted',
                    'padding_factor':64*_EPS,'sensitivity_full_gain':f['K'],'rho_lower':f['rho'],
                    'gain_cap':f['k'],'base_reconstruction_padding':f['reconstruction_pad'],
                    'basis_orthogonality_error':f['orth_error'],
                    'validation_norm':'raw compressed gradient; ||dx|| <= ||dg_full||',
                    'validation_normalization':'mean of per-task A-to-B and B-to-A query MSE'})
            selected={row['candidate_id']:_selected_branch(row,f) for row in (left,right)}
            for branch in f['branches']:
                data=_validation_data(validation,f['U'],f['width'],branch)
                model=_solver_model(f,branch)
                probes=[_probe(row,branch,f,data,model) for row in (left,right)]
                out['probes'].extend(probes)
                out['branches'][branch]={'lower_bound':max(p['lower_bound'] for p in probes),'solver_model':model}
                for row,probe in zip((left,right),probes):
                    if selected[row['candidate_id']] != branch: continue
                    if f['method'] != 'B02':
                        _require(row['prepared_state']['metric'] == row['prepared_state']['solver_diagnostics'][branch]['metric'],
                                 'selected metric witness mismatch')
                    mismatch=abs(probe['loss']-row['validation_loss'])
                    _require(mismatch <= probe['loss_padding']+probe['uniform_map_loss_padding'],
                             'literal selected loss disagrees with prediction quadratic')
                    _require(probe['task_loss_discrepancy'] <= len(validation)*(probe['loss_padding']+probe['uniform_map_loss_padding']),
                             'literal selected task losses disagree with prediction quadratic')
                    probe['selected_loss_discrepancy']=mismatch
            lower=min(item['lower_bound'] for item in out['branches'].values())
            incumbent_probe=next(p for p in out['probes'] if p['anchor_candidate_id'] == incumbent['candidate_id']
                                 and p['branch'] == selected[incumbent['candidate_id']])
            incumbent_padding=incumbent_probe['loss_padding']+incumbent_probe['uniform_map_loss_padding']
            improvement=max(0.,incumbent['validation_loss']+incumbent_padding-lower)
            comparison_padding=_pad(len(validation)+8,max(1.,abs(lower),abs(incumbent['validation_loss']),f['tau'],improvement))
            out.update(lower_bound=lower,incumbent_loss_padding=incumbent_padding,improvement_bound=improvement,
                       comparison_padding=comparison_padding,numerical_margin=f['tau']-improvement-comparison_padding,
                       status='resolved' if improvement+comparison_padding < f['tau'] else 'unresolved')
            # No NaN/Inf may escape even on nonpassing arithmetic.
            content_hash(out)
    except (KeyError,TypeError,ValueError,OverflowError,FloatingPointError,RuntimeError,np.linalg.LinAlgError) as exc:
        # Preserve only JSON-finite partial diagnostics on a refused route.
        out['status']='unsupported'; out['reason']=str(exc)
        try: content_hash(out)
        except (ValueError,TypeError):
            out={key:value for key,value in out.items() if key in ('status','certificate_level','scope','method_id','incumbent_candidate_id','reason','additional_labels','additional_independent_tasks')}
            out.update(interval=None,objective_tolerance=None,improvement_bound=None,branches={},probes=[])
    out['elapsed_ns']=time.perf_counter_ns()-started
    return to_json_record(out)
