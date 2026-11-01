"""Oracle method lifecycle; learned fitting is supplied by the prior-fitting stage."""
import numpy as np
from .baselines import metric_from_gains, solve_rank_gains, solve_unrestricted_gamma, ridge_coefficient, posterior_coefficient, unrestricted_fixed_metric
from .geometry import construct_canonical, construct_head_zero, effective_coefficient, simultaneous_step
from .metrics import apply_metric, metric_resources
from .records import AdaptedState, PreparedState, Resources, to_json_record

class PreparationError(RuntimeError):
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.preparation_diagnostics=diagnostics


def execution_metadata(prepared):
    """Separate actual architecture from numerical fit to the comparison budget."""
    depth = 1 if prepared.method_id=='B08:gd' else (
        len(prepared.initial_layers) if prepared.initial_layers is not None else None)
    fixed_depth = depth==prepared.budget.depth
    resources = prepared.resources
    fits_norm = (resources.initialization_norm<=prepared.budget.initialization_norm_cap
                 if resources.initialization_norm is not None else None)
    fits_count = (resources.trainable_parameters<=prepared.budget.trainable_parameter_cap
                  if resources.trainable_parameters is not None else None)
    return {'method_id':prepared.method_id, 'actual_depth':depth,
            'scope':'main_fixed_depth_class' if fixed_depth else 'outside_main_class_operational_control',
            'fixed_depth_eligible':fixed_depth,
            'reference_budget':to_json_record(prepared.budget),
            'reference_cap_fit':{'fits_norm_cap':fits_norm, 'fits_parameter_cap':fits_count,
                                 'applies_as_eligibility':fixed_depth}}


class Method:
    def __init__(self, method_id, options): self.method_id, self.options = method_id, dict(options)
    def fit_meta(self, train_tasks, validation_tasks, config):
        from .selection import fit_method
        return fit_method(self, train_tasks, validation_tasks, config)
    def prepare(self, prior, budget, task):
        d=task.input_dimension; n=task.target_context_size; mid=self.method_id; family=mid.split(':')[0]
        if budget.depth<2 or budget.learning_rate!=1 or n<1 or d<1: raise ValueError('invalid preparation')
        k=(budget.initialization_norm_cap/(budget.depth-1))**(budget.depth-1)
        width=min(d,(budget.trainable_parameter_cap-(budget.depth-2))//(d+1))
        if family in ('B00','B01','B02','B03','B04') and width<1: raise ValueError('infeasible architecture')
        s=np.asarray(prior.eigenvalues) if prior.eigenvalues is not None else None
        basis=prior.basis; solver_diagnostics=None
        if 'fitted_metric' in self.options: metric=self.options['fitted_metric']
        elif family=='B00': metric=metric_from_gains(np.eye(d),np.zeros(d))
        elif family=='B01': metric=metric_from_gains(np.eye(d),np.full(d,min(k,1/((n+1)/n+d*(1+prior.noise_variance)/n))))
        elif family=='B02':
            if budget.depth!=2: raise ValueError('B02 requires depth two')
            metric=metric_from_gains(basis,solve_rank_gains(s,n,prior.noise_variance,k,width))
        elif family in ('B03','B04'):
            from .frontier import solve_frontier
            frontier=solve_frontier(prior,budget,task)
            solver_diagnostics=to_json_record(frontier)
            if family=='B03':
                if frontier.F.status!='completed': raise PreparationError('F solver unresolved',solver_diagnostics)
                metric=frontier.F.metric
            else:
                if frontier.F.status!='completed' or frontier.Z.status!='completed': raise PreparationError('frontier solver unresolved',solver_diagnostics)
                if frontier.selection_status=='unresolved':
                    if frontier.F.objective != frontier.Z.objective: raise PreparationError('frontier comparison unresolved',solver_diagnostics)
                    # Exact float ties retain both; resource use is an implementation tie-break.
                    from .metrics import metric_resources
                    chosen=min((frontier.F,frontier.Z),key=lambda b:metric_resources(b.metric,budget.depth).trainable_parameters)
                else: chosen=frontier.F if frontier.selected_branches==('F',) else frontier.Z
                metric=chosen.metric
        elif family=='B05':
            metric=self.options.get('metric')
            if metric is None: raise ValueError('B05 requires shared metric')
        elif family=='B06': metric=metric_from_gains(basis,solve_unrestricted_gamma(s,n,prior.noise_variance,k))
        elif family=='B07': metric=unrestricted_fixed_metric(prior,task)
        else: metric=None
        realization=None
        if family=='B11' and mid=='B11:Z' and budget.depth==2: realization=construct_head_zero(metric)
        elif family in ('B00','B01','B03','B04','B11'): realization=construct_canonical(metric,budget.depth)
        elif family=='B02': realization=construct_head_zero(metric)
        resources=realization.resources if realization else Resources(None,None,0,0)
        if metric is not None and realization is None:
            count=1+metric.basis.size+metric.increments.size
            resources=Resources(None,None,count,8*count)
        if mid=='B08:gd': resources=Resources(0.,d,d,8*d)
        return PreparedState(metric, realization.layers if realization else None,budget,task,resources,mid,prior,self.options,solver_diagnostics)
    def adapt(self, prepared, context):
        X=np.asarray(context.X); y=np.asarray(context.y); p=prepared.preparation
        if X.shape!=(p.target_context_size,p.input_dimension) or y.shape!=(p.target_context_size,) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)): raise ValueError('context does not match preparation')
        family=prepared.method_id.split(':')[0]; variant=prepared.method_id.partition(':')[2]
        layers=None
        if prepared.options.get('zero_estimator',False): coefficient=np.zeros(p.input_dimension)
        elif prepared.initial_layers is not None:
            layers=simultaneous_step(prepared.initial_layers,-X.T@y/len(X),prepared.budget.learning_rate); coefficient=effective_coefficient(layers)
        elif prepared.metric is not None: coefficient=apply_metric(prepared.metric,X.T@y/len(X))
        elif family=='B08' and variant=='gd': coefficient=float(prepared.options.get('step_size',1.))*X.T@y/len(X)
        elif family in ('B08','B09'):
            basis=prepared.prior.basis[:,:int(prepared.options.get('rank',p.input_dimension))] if family=='B09' else None
            penalty=0. if variant=='least_squares' else float(prepared.options.get('penalty',0.))
            coefficient=ridge_coefficient(context,penalty,basis)
        elif family=='B10': coefficient=posterior_coefficient(context,prepared.prior)
        else: raise ValueError('unsupported method')
        return AdaptedState(coefficient,layers,'completed',None)
    def predict(self, adapted, X_query): return np.asarray(X_query)@adapted.coefficient
    def diagnostics(self, prepared, adapted):
        info={'method_id':prepared.method_id,'information_access':prepared.prior.information_access,'resource_class':'network' if prepared.initial_layers is not None or prepared.method_id=='B08:gd' else ('compact_direct' if prepared.metric is not None else 'software_estimator')}
        info['execution_metadata']=execution_metadata(prepared)
        if prepared.metric is not None and prepared.initial_layers is None:
            reference=metric_resources(prepared.metric,prepared.budget.depth)
            fits_norm=reference.initialization_norm <= prepared.budget.initialization_norm_cap
            fits_parameters=reference.trainable_parameters <= prepared.budget.trainable_parameter_cap
            info['canonical_reference']={'resources':to_json_record(reference),'fits_norm_cap':fits_norm,'fits_parameter_cap':fits_parameters,'fits_caps':fits_norm and fits_parameters}
        if prepared.method_id.startswith('B09:'): info['basis_provenance']=prepared.prior.fit_metadata.get('basis_provenance','oracle covariance directions')
        if prepared.method_id.startswith('B10:'):
            eigenvalues=np.linalg.eigvalsh(prepared.prior.covariance)
            rank=int(np.linalg.matrix_rank(prepared.prior.covariance))
            condition=float(eigenvalues[-1]/eigenvalues[-rank]) if rank else None
            info.update({'posterior_reference':'Gaussian prior and noise', 'covariance_rank':rank, 'covariance_condition':condition, 'added_ridge':0.})
        return info


def make_method(method_id, options):
    options=dict(options); family,sep,variant=method_id.partition(':')
    from collections.abc import Mapping
    if isinstance(options.get('fitted_metric'),Mapping):
        from .metrics import CompactMetric
        from .records import SupportMetadata
        record=options['fitted_metric']; meta=record['support_metadata']
        options['fitted_metric']=CompactMetric(record['floor'],np.asarray(record['basis'],dtype=float),np.asarray(record['increments'],dtype=float),SupportMetadata(**meta))
    if family=='B05':
        target=variant or options.get('target_method_id')
        if not target: raise ValueError('B05 requires target_method_id')
        if options.get('target_method_id',target)!=target: raise ValueError('conflicting target method ID')
        method_id='B05:'+target
    elif family=='B11':
        if variant not in ('F','Z'): raise ValueError('B11 requires F or Z variant')
        method_id=family+':'+variant
    elif family in ('B08','B09','B10'):
        variant=variant or options.get('variant','')
        allowed={'B08':('gd','ridge'),'B09':('least_squares','ridge'),'B10':('posterior','empirical_bayes')}
        if variant not in allowed[family]: raise ValueError('explicit valid variant required')
        method_id=family+':'+variant
    elif sep or family not in ('B00','B01','B02','B03','B04','B06','B07'): raise ValueError('unknown method')
    return Method(method_id,options)
