"""Source-only preparation and validation selection."""
from .records import SelectionResult, to_json_record, content_hash

class CandidateFitError(ValueError):
    def __init__(self, records):
        super().__init__('all method candidates failed')
        self.records=tuple(records)

_ALLOWED={'budget','preparation','fitting_candidates','information_policy','source_access_manifest','selection_class','tuning_resolution_policy'}

TUNING_RESOLUTION_POLICY='min_marginal_loss_se_branch_envelope_v2'

def tuning_resolution_policy():
    """Fixed, source-versioned convention; not an optional tuning parameter."""
    return {'version':TUNING_RESOLUTION_POLICY,'loss_units':'mean_query_squared_error',
        'uncertainty_unit':'independent_source_validation_task',
        'objective_resolution':'absolute_mean_gap <= min(left_marginal_se, right_marginal_se)',
        'branch_envelope':'supported_local_possible_improvement_below_unchanged_tau_float64_diagnostic',
        'ranking':'minimum_validation_mean_with_paired_difference_diagnostics',
        'shrinkage_probe':'include_endpoints_and_an_interior_candidate_before_local_checks',
        'scope':'finite_local_objective_resolution_not_confidence_or_global_optimality',
        'common_task_components':'may_change_stopping_without_changing_paired_rankings',
        'validation_reuse':'adaptive_same_task_reuse_not_simultaneous_confidence'}

def public_method_config(config):
    return {'budget':to_json_record(config.budget),'preparation':to_json_record(config.preparation),'fitting_candidates':to_json_record(config.numerics.get('fitting_candidates',{})),'information_policy':{key:to_json_record(config.access[key]) for key in ('noise_access','permitted_noise_variance','selection','prior') if key in config.access},'source_access_manifest':to_json_record(config.access.get('source_access_manifest')),'tuning_resolution_policy':tuning_resolution_policy()}

def _authorize(train, validation, config):
    if not isinstance(config,dict) or set(config)-_ALLOWED: raise ValueError('nonpublic method configuration')
    if config.get('tuning_resolution_policy',tuning_resolution_policy())!=tuning_resolution_policy(): raise ValueError('unsupported tuning resolution policy')
    settings=config.get('fitting_candidates',{})
    if not isinstance(settings,dict) or set(settings)-{'gammas','regularization','ranks','ridge_multipliers','scalar_fractions','gd_steps','max_refinement_rounds'}: raise ValueError('nonpublic fitting candidates')
    manifest=config.get('source_access_manifest')
    if not isinstance(manifest,dict) or set(manifest)!={'training_keys','validation_keys'}: raise ValueError('source access manifest required')
    groups=[]
    for role,tasks in (('training_keys',train),('validation_keys',validation)):
        allowed=manifest[role]
        if not isinstance(allowed,(list,tuple)) or not allowed or any(not isinstance(x,str) or not x for x in allowed) or len(set(allowed))!=len(allowed): raise ValueError('invalid source manifest')
        actual=[t.source_key for t in tasks]
        if len(actual)!=len(set(actual)) or set(actual)!=set(allowed): raise ValueError('source key unauthorized or role mismatch')
        groups.append(set(allowed))
    if groups[0]&groups[1]: raise ValueError('overlapping source manifest')
    blocks=[block.task_key for task in (*train,*validation) for block in (task.A,task.B)]
    if len(blocks)!=len(set(blocks)): raise ValueError('duplicate observation block')


def _validate_preparation(config):
    from .records import PreparationSpec
    try: prep=PreparationSpec(**config['preparation'])
    except (KeyError,TypeError): raise ValueError('invalid preparation contract') from None
    if (type(prep.input_dimension) is not int or prep.input_dimension<1 or type(prep.target_context_size) is not int or prep.target_context_size<1 or prep.input_law!='gaussian_isotropic' or prep.loss_normalization!='squared_error_over_2n' or prep.task_second_moment_trace!=1):
        raise ValueError('unsupported learned preparation contract')
    return prep


def _validation_loss_se(row, task_count):
    import numpy as np
    try:
        losses=np.asarray(row.get('validation_task_losses'),dtype=float)
        if task_count<2 or losses.shape!=(task_count,) or not np.all(np.isfinite(losses)) or not np.isfinite(row['validation_loss']): return None
        with np.errstate(over='ignore',invalid='ignore'):
            se=float(losses.std(ddof=1)/np.sqrt(task_count))
        return se if np.isfinite(se) else None
    except (KeyError,TypeError,ValueError):
        return None


def _objective_check(left, right, dimension, inward_probe):
    """Compare objective levels; paired ranking precision remains a diagnostic."""
    import numpy as np
    left_se=left['validation_loss_se']; right_se=right['validation_loss_se']
    gap=abs(float(left['validation_loss'])-float(right['validation_loss']))
    gap=gap if np.isfinite(gap) else None
    tolerance=paired_se=None
    if left_se is not None and right_se is not None:
        tolerance=min(left_se,right_se)
        with np.errstate(over='ignore',invalid='ignore'):
            diff=np.asarray(left['validation_task_losses'])-np.asarray(right['validation_task_losses'])
            paired_se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
        if not np.isfinite(paired_se): paired_se=None
    decision='uncertainty_unavailable' if gap is None or tolerance is None or paired_se is None else ('refine_objective_gap' if gap>tolerance else 'objective_resolved')
    coordinate='gamma' if dimension=='shrinkage' else 'tuning_value'
    return {'dimension':dimension,'left_candidate_id':left['candidate_id'],'right_candidate_id':right['candidate_id'],
        'interval':[left[coordinate],right[coordinate]],'left_marginal_se':left_se,'right_marginal_se':right_se,
        'absolute_mean_gap':gap,'paired_difference_se':paired_se,'objective_tolerance':tolerance,
        'inward_probe':inward_probe,'decision':decision}


def select_method(candidates,train,validation,config):
    _authorize(train,validation,config)
    if not candidates or not train or not validation: raise ValueError('candidates and disjoint source splits required')
    if {t.source_key for t in train}&{t.source_key for t in validation}: raise ValueError('overlapping source splits')
    selection_class=config.get('selection_class','operational')
    eligibility={'mechanism_baseline':{'B01','B02','B11:Z'},'mechanism_joint':{'B03','B04','B11:F','B11:Z'},'mechanism_all':{'B01','B02','B03','B04','B11:F','B11:Z'}}
    if selection_class not in (*eligibility,'operational'): raise ValueError('unknown selection class')
    if selection_class in eligibility and any(m.method_id not in eligibility[selection_class] for m in candidates): raise ValueError('ineligible mechanism candidate')
    _validate_preparation(config)
    import time
    import numpy as np
    started=time.perf_counter_ns(); settings=dict(config.get('fitting_candidates',{})); gammas=sorted(set([0.,1.]+list(settings.get('gammas',[0.,.5,1.])))); multipliers=sorted(set(settings.get('ridge_multipliers',[0.,.001,.01,.1,1.,10.,100.,1000.])))
    max_rounds=int(settings.get('max_refinement_rounds',5))
    if max_rounds<0 or max_rounds>12: raise ValueError('invalid refinement limit')
    total_evaluations=0; resolution='unresolved'; fitted=[]; rows=[]; ledger=[]
    for iteration in range(max_rounds+1):
        opts={**settings,'gammas':gammas,'ridge_multipliers':multipliers}; trial_config={**config,'fitting_candidates':opts}
        rows=[]; fitted=[]
        for method in candidates:
            if method.method_id.split(':')[0] in ('B05','B06','B07') or method.method_id in ('B10:posterior',): raise ValueError('oracle or control candidate forbidden')
            method_start=time.perf_counter_ns()
            try:
                result=method.fit_meta(train,validation,trial_config)
                attempts=[]; selected_attempt=None
                for record in result.candidate_records:
                    item=dict(record); logical=item.get('candidate_id',method.method_id)
                    item['logical_candidate_id']=logical; item['refinement_iteration']=iteration
                    item['candidate_id']=f'{logical}:attempt={total_evaluations+len(rows)+len(attempts)}'
                    if logical==result.selected_id and item.get('status')=='completed': selected_attempt=item['candidate_id']
                    attempts.append(item)
                rows.extend(attempts); fitted.append(SelectionResult(selected_attempt,tuple(attempts),result.asset))
            except CandidateFitError as exc:
                for record in exc.records:
                    item=dict(record); logical=item.get('candidate_id',method.method_id); item['logical_candidate_id']=logical; item['refinement_iteration']=iteration; item['candidate_id']=f'{logical}:attempt={total_evaluations+len(rows)}'; rows.append(item)
            except (ValueError,RuntimeError) as exc:
                rows.append({'candidate_id':f'{method.method_id}:attempt={total_evaluations+len(rows)}','logical_candidate_id':method.method_id,'refinement_iteration':iteration,'method_id':method.method_id,'status':'failed','error':str(exc),'elapsed_ns':time.perf_counter_ns()-method_start})
        total_evaluations+=len(rows); ledger.extend(rows)
        if not fitted: return SelectionResult('',tuple(ledger),None)
        groups={}
        for row in rows:
            row['tuning_resolution_policy']=TUNING_RESOLUTION_POLICY
            if row.get('status')!='completed': continue
            row['validation_loss_se']=_validation_loss_se(row,len(validation))
            asset=row['asset']; key=(row['method_id'],asset['fit_metadata']['parent_hash'],row['tuning_kind'],row['tuning_value'],row.get('rank'))
            groups.setdefault(key,[]).append(row)
        refinements=set(); uncertainty_missing=len(validation)<2 or any(r.get('status')=='completed' and r['validation_loss_se'] is None for r in rows)
        for group in groups.values():
            group=sorted(group,key=lambda r:r['gamma']); best=min(range(len(group)),key=lambda i:group[i]['validation_loss'])
            # Shrinkage has true closed-domain endpoints, unlike a finite ridge grid.
            # The closed domain forbids outward extension, not inward refinement.
            # Probe the interior and retain endpoint-neighbor objective checks.
            interior=next((r for r in group if 0<r['gamma']<1),None)
            if interior is None:
                check=_objective_check(group[0],group[-1],'shrinkage','required_midpoint')
                check.update(decision='refine_initial_interior_probe',proposed_coordinate=.5)
                group[best].setdefault('tuning_checks',[]).append(check)
                refinements.add(.5); continue
            for i in (best-1,best):
                if i<0 or i+1>=len(group): continue
                left,right=group[i:i+2]
                check=_objective_check(left,right,'shrinkage','interior_candidate_present')
                check['inward_probe_candidate_id']=interior['candidate_id']
                group[best].setdefault('tuning_checks',[]).append(check)
                if check['decision']=='uncertainty_unavailable': uncertainty_missing=True
                if check['decision']=='refine_objective_gap':
                    from .tuning_bounds import branch_interval_check
                    envelope=branch_interval_check(left,right,group[best],validation,trial_config)
                    check['branch_envelope']=envelope
                    if envelope['status']=='resolved':
                        check['decision']='objective_resolved_by_branch_envelope'
                        continue
                    midpoint=(left['gamma']+right['gamma'])/2
                    check.update(proposed_coordinate=midpoint,representable_progress=left['gamma']<midpoint<right['gamma'])
                    refinements.add(midpoint)
        # Extend finite ridge boundary minima, shared across all methods.
        extend=set(); positives=[x for x in multipliers if x>0]
        ridge_rows=[r for r in rows if r.get('status')=='completed' and r.get('tuning_kind') in ('penalty','zero')]
        ridge_groups={}
        for row in ridge_rows:
            # Penalty and its zero-estimator limit vary within a fixed preparation.
            fixed_options={k:v for k,v in row.get('options',{}).items() if k not in ('penalty','zero_estimator')}
            key=(row['method_id'],row['asset_hash'],row.get('rank'),content_hash(fixed_options))
            ridge_groups.setdefault(key,[]).append(row)
        for group in ridge_groups.values():
            best=min(group,key=lambda r:r['validation_loss'])
            if best['tuning_kind']=='penalty' and positives:
                scale=best['ridge_scale']; multiplier=best['tuning_value']/scale if scale else 0.
                boundary=[]
                if np.isclose(multiplier,positives[0],rtol=1e-12,atol=0): boundary.append(positives[0]/10)
                if np.isclose(multiplier,positives[-1],rtol=1e-12,atol=0): boundary.append(positives[-1]*10)
                extend.update(boundary)
                best['ridge_boundary_check']={'best_multiplier':multiplier,'positive_grid_bounds':[positives[0],positives[-1]],
                    'extension_multipliers':boundary,'decision':'extend_finite_boundary' if boundary else 'finite_boundaries_checked'}
                same=sorted((r for r in group if r['tuning_kind']=='penalty'),key=lambda r:r['tuning_value'])
                index=next((i for i,r in enumerate(same) if r['candidate_id']==best['candidate_id']),None)
                if index is not None:
                    for i in (index-1,index):
                        if i<0 or i+1>=len(same): continue
                        left,right=same[i:i+2]
                        check=_objective_check(left,right,'ridge_penalty','not_applicable')
                        best.setdefault('tuning_checks',[]).append(check)
                        if check['decision']=='uncertainty_unavailable': uncertainty_missing=True
                        if check['decision']=='refine_objective_gap':
                            lo=left['tuning_value']/scale; hi=right['tuning_value']/scale
                            midpoint=np.sqrt(lo*hi) if lo>0 else hi/2
                            check.update(proposed_coordinate=float(midpoint*scale),representable_progress=lo<midpoint<hi)
                            extend.add(midpoint)
            elif best['tuning_kind']=='zero':
                best['ridge_boundary_check']={'decision':'zero_estimator_limit_selected','extension_multipliers':[]}
        if not refinements and not extend:
            resolution='uncertainty_unavailable' if uncertainty_missing else 'resolved'; break
        if iteration==max_rounds: break
        gammas=sorted(set(gammas)|refinements); multipliers=sorted(set(multipliers)|extend)
    elapsed=time.perf_counter_ns()-started
    rows=[{**r,'tuning_resolution':resolution,'refinement_rounds':iteration,'total_candidate_evaluations':total_evaluations,'total_tuning_elapsed_ns':elapsed} for r in ledger]
    best=min(fitted,key=lambda x:min(r['validation_loss'] for r in x.candidate_records if r.get('status')=='completed'))
    return SelectionResult(best.selected_id,tuple(rows),best.asset)



def fit_method(method,train,validation,config):
    import time
    import numpy as np
    from .records import BudgetSpec, PreparationSpec, content_hash
    from .priors import estimate_cross_moment, estimate_moment_directions, shrink_prior, estimate_noise, build_empirical_qp, fit_empirical_gains, _asset
    from .baselines import metric_from_gains
    _authorize(train,validation,config)
    _validate_preparation(config)
    if set(method.options)-{'penalty','rank','step_size','variant'}: raise ValueError('unpermitted pre-fitted method asset')
    if not train or not validation or {t.source_key for t in train}&{t.source_key for t in validation}: raise ValueError('invalid source split')
    budget=BudgetSpec(**config['budget']); prep=PreparationSpec(**config['preparation']); settings=config.get('fitting_candidates',{}); policy=config.get('information_policy',{})
    if set(policy)-{'noise_access','permitted_noise_variance','selection','prior'}: raise ValueError('nonpublic information policy')
    if policy.get('prior')=='oracle': raise ValueError('oracle prior forbidden in learned fitting')
    if 'permitted_noise_variance' in policy and policy.get('noise_access')!='known': raise ValueError('undeclared known-noise asset')
    noise_start=time.perf_counter_ns(); noise_fit_id=f'{method.method_id}:noise:{noise_start}'; noise=estimate_noise(train); noise_elapsed=time.perf_counter_ns()-noise_start
    if policy.get('noise_access','estimated')=='known':
        if 'permitted_noise_variance' not in policy: raise ValueError('known noise asset required')
        variance=float(policy['permitted_noise_variance']); noise_access='known'
    elif policy.get('noise_access','estimated')=='estimated': variance=noise['variance']; noise_access='estimated'
    else: raise ValueError('invalid noise policy')
    if not np.isfinite(variance) or variance<0: raise ValueError('invalid noise')
    bases=[]
    for estimator,argument in [(estimate_cross_moment,prep.input_dimension)]+[(estimate_moment_directions,int(rank)) for rank in settings.get('ranks',[1,prep.input_dimension])]:
        asset_start=time.perf_counter_ns(); asset=estimator(train,argument); bases.append((asset,time.perf_counter_ns()-asset_start,f'{method.method_id}:base:{asset_start}'))
    gammas=sorted(set([0.,1.]+list(settings.get('gammas',[0.,.5,1.])))); regs=sorted(set([0.]+list(settings.get('regularization',[0.,.01])))); family=method.method_id.split(':')[0]
    rows=[]; prepared=[]
    training_labels=sum(len(t.A.y)+len(t.B.y) for t in train); validation_labels=sum(len(t.A.y)+len(t.B.y) for t in validation); labels=training_labels+validation_labels
    for base,base_elapsed,base_fit_id in bases:
      for gamma in gammas:
        asset_start=time.perf_counter_ns(); asset_fit_id=f'{method.method_id}:{asset_start}'
        prior=shrink_prior(base,float(gamma)); prior=_asset(prior.covariance,variance,{**prior.fit_metadata,'noise_access':noise_access,'noise_estimate':noise},basis=prior.basis,eigenvalues=prior.eigenvalues)
        shrink_fit_elapsed=time.perf_counter_ns()-asset_start; asset_fit_elapsed=base_elapsed+shrink_fit_elapsed
        if family=='B11': choices=[('reg',r) for r in regs]
        elif family=='B01':
            k=(budget.initialization_norm_cap/(budget.depth-1))**(budget.depth-1)
            choices=[('scalar',float(v)*k) for v in settings.get('scalar_fractions',[0.,.125,.25,.5,.75,1.])]
        elif method.method_id in ('B08:ridge','B09:ridge'):
            scale=float(np.mean([np.sum(c.X*c.X)/prep.input_dimension for t in train for c in (t.A,t.B)]))
            choices=[('penalty',float(v)*scale) for v in settings.get('ridge_multipliers',[0.,.001,.01,.1,1.,10.,100.,1000.])]+[('zero',None)]
        elif method.method_id=='B08:gd': choices=[('step',float(v)) for v in settings.get('gd_steps',[0.,.25,.5,1.,2.])]
        else: choices=[('default',None)]
        ranks=settings.get('ranks',[1,prep.input_dimension]) if family=='B09' else [None]
        for kind,value_option,rank in ((kind,value_option,rank) for kind,value_option in choices for rank in ranks):
            reg=value_option if kind=='reg' else None
            candidate_start=time.perf_counter_ns(); options=dict(method.options); name=f'{method.method_id}:base={base.asset_hash}:gamma={gamma}:{kind}={value_option}:rank={rank}'
            fit=None; state=None
            try:
                if kind=='scalar': options['fitted_metric']=metric_from_gains(np.eye(prep.input_dimension),np.full(prep.input_dimension,value_option))
                if rank is not None: options['rank']=int(rank)
                if kind=='step': options['step_size']=value_option
                if kind=='penalty': options['penalty']=value_option
                if kind=='zero': options['zero_estimator']=True
                if family=='B11':
                    branch=method.method_id.split(':')[1]; qp=build_empirical_qp(train,prior.basis,budget,branch,float(reg)); fit=fit_empirical_gains(qp,budget,prior.basis,branch)
                    if fit.status!='completed': raise RuntimeError('empirical QP unresolved')
                    options['fitted_metric']=fit.metric
                trial=type(method)(method.method_id,options); state=trial.prepare(prior,budget,prep); losses=[]
                if state.initial_layers is not None and (state.resources.initialization_norm>budget.initialization_norm_cap+1e-9*max(1.,budget.initialization_norm_cap) or state.resources.trainable_parameters>budget.trainable_parameter_cap): raise ValueError('prepared network exceeds budget')
                for task in validation:
                    task_losses=[]
                    for context,query in ((task.A,task.B),(task.B,task.A)):
                        adapted=trial.adapt(state,context); loss=np.mean((trial.predict(adapted,query.X)-query.y)**2); task_losses.append(float(loss))
                    losses.append(float(np.mean(task_losses)))
                value=float(np.mean(losses)); row={'candidate_id':name,'method_id':method.method_id,'status':'completed','validation_loss':value,'validation_task_losses':losses,'asset_hash':prior.asset_hash,'asset_fit_id':asset_fit_id,'asset_fit_elapsed_ns':asset_fit_elapsed,'base_fit_id':base_fit_id,'base_fit_elapsed_ns':base_elapsed,'shrink_fit_id':asset_fit_id,'shrink_fit_elapsed_ns':shrink_fit_elapsed,'noise_fit_id':noise_fit_id,'noise_fit_elapsed_ns':noise_elapsed,'asset':to_json_record(prior),'options':to_json_record(options),'gamma':float(gamma),'regularization':reg,'tuning_kind':kind,'tuning_value':value_option,'rank':rank,'ridge_scale':scale if family in ('B08','B09') and kind in ('penalty','zero') else None,'unique_labels':labels,'training_labels':training_labels,'validation_labels':validation_labels,'training_tasks':len(train),'validation_tasks':len(validation),'source_hash':content_hash(train),'validation_hash':content_hash(validation),'prepared_state':to_json_record(state),'prepared_hash':content_hash(state),'elapsed_ns':time.perf_counter_ns()-candidate_start}
                if family=='B11': row['fit']=to_json_record(fit)
                rows.append(row); prepared.append((value,trial,prior,name))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                from .execution import failure_evidence
                evidence=failure_evidence(exc,state,name)
                rows.append({'candidate_id':name,'method_id':method.method_id,'status':'failed',**evidence,'asset_hash':prior.asset_hash,'asset_fit_id':asset_fit_id,'asset_fit_elapsed_ns':asset_fit_elapsed,'base_fit_id':base_fit_id,'base_fit_elapsed_ns':base_elapsed,'shrink_fit_id':asset_fit_id,'shrink_fit_elapsed_ns':shrink_fit_elapsed,'noise_fit_id':noise_fit_id,'noise_fit_elapsed_ns':noise_elapsed,'asset':to_json_record(prior),'options':to_json_record(options),'regularization':reg,'unique_labels':labels,'training_labels':training_labels,'validation_labels':validation_labels,'fit':to_json_record(fit) if fit is not None else None,'elapsed_ns':time.perf_counter_ns()-candidate_start})
    if not prepared: raise CandidateFitError(rows)
    _,best,prior,name=min(prepared,key=lambda x:x[0])
    return SelectionResult(name,tuple(rows),prior)
