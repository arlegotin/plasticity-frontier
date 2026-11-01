"""Shared immutable records; mathematical behavior belongs in later modules."""
from dataclasses import dataclass, fields, is_dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
from enum import Enum
import hashlib
import json
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Layers = tuple[Array, ...]
JsonRecord = dict[str, object]

def to_json_record(value):
    """Return detached JSON-compatible values for immutable records and assets."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_json_record(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: to_json_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [to_json_record(item) for item in value]
    if isinstance(value, np.ndarray): return to_json_record(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Enum): return value.value
    return value

def content_hash(value: JsonRecord) -> str:
    return hashlib.sha256(json.dumps(to_json_record(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _freeze(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject: raise ValueError('object arrays cannot be frozen')
        # An immutable bytes buffer also prevents callers from re-enabling writes.
        copied = np.ascontiguousarray(value)
        return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)
    if isinstance(value, Mapping): return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)): return tuple(_freeze(v) for v in value)
    return value

class ImmutableArrays:
    def __post_init__(self):
        for field in fields(self): object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))

class Status(str, Enum):
    RUNNING = 'running'; COMPLETED = 'completed'; INCOMPLETE = 'incomplete'; INTERRUPTED = 'interrupted'; INVALID = 'invalid'; FAILED = 'failed'

@dataclass(frozen=True)
class BudgetSpec(ImmutableArrays):
    depth: int; initialization_norm_cap: float; trainable_parameter_cap: int; learning_rate: float
@dataclass(frozen=True)
class PreparationSpec(ImmutableArrays):
    target_context_size: int; input_dimension: int; input_law: str; loss_normalization: str; task_second_moment_trace: float

    def __post_init__(self):
        if self.input_law == "isotropic_gaussian":
            object.__setattr__(self, "input_law", "gaussian_isotropic")
        super().__post_init__()
@dataclass(frozen=True)
class SamplePlan(ImmutableArrays):
    analysis_mode: str; outer_replicates: int | None; tasks_per_replicate: int | None; source_training_tasks: int | None; source_validation_tasks: int | None; source_block_size: int | None
@dataclass(frozen=True)
class RunConfig(ImmutableArrays):
    schema_version: int; workload: dict; budget: BudgetSpec; preparation: PreparationSpec; numerics: dict; access: dict; methods: tuple; keys: dict; sample_plan: SamplePlan; confirmation: dict
@dataclass(frozen=True)
class SupportMetadata(ImmutableArrays):
    policy: str; exact_zero_multiplicity: int; minimum_multiplicity: int; retained_indices: tuple; projections: tuple = (); truncations: tuple = ()
@dataclass(frozen=True)
class Resources(ImmutableArrays):
    initialization_norm: float | None; trainable_parameters: int | None; stored_scalars: int; serialized_bytes: int
@dataclass(frozen=True)
class NetworkRealization(ImmutableArrays):
    layers: Layers; resources: Resources; support_metadata: SupportMetadata
@dataclass(frozen=True)
class GeometryDiagnostics(ImmutableArrays):
    residuals: JsonRecord; repairs: tuple
@dataclass(frozen=True)
class QPData(ImmutableArrays):
    constant: float; linear: Array; hessian: Array; A: Array; offset: Array
@dataclass(frozen=True)
class NumericDiagnostics(ImmutableArrays):
    feasibility: object; kkt: object; dual: object; roundoff: object
@dataclass(frozen=True)
class BranchResult(ImmutableArrays):
    branch: str; variables: object; metric: object; objective: float; diagnostics: NumericDiagnostics; status: str
@dataclass(frozen=True)
class FrontierResult(ImmutableArrays):
    F: BranchResult; Z: BranchResult; selected_branches: tuple; selection_status: str
@dataclass(frozen=True)
class Context(ImmutableArrays):
    X: Array; y: Array; task_key: str
@dataclass(frozen=True)
class SourceTask(ImmutableArrays):
    A: Context; B: Context; source_key: str
@dataclass(frozen=True)
class PriorAsset(ImmutableArrays):
    covariance: Array | None; basis: Array | None; eigenvalues: Array | None; noise_variance: float | None; information_access: str; fit_metadata: JsonRecord; asset_hash: str
@dataclass(frozen=True)
class PreparedState(ImmutableArrays):
    metric: object; initial_layers: Layers | None; budget: BudgetSpec; preparation: PreparationSpec; resources: Resources; method_id: str; prior: PriorAsset | None = None; options: JsonRecord | None = None; solver_diagnostics: JsonRecord | None = None
@dataclass(frozen=True)
class AdaptedState(ImmutableArrays):
    coefficient: Array; layers: Layers | None; status: str; timing: object
@dataclass(frozen=True)
class SelectionResult(ImmutableArrays):
    selected_id: str; candidate_records: tuple; asset: PriorAsset | None
class GateCode(str, Enum):
    IMPLEMENTATION_INVALID = 'IMPLEMENTATION_INVALID'
    MATHEMATICAL_COUNTEREXAMPLE = 'MATHEMATICAL_COUNTEREXAMPLE'
    INSENSITIVE_OR_UNSUITABLE = 'INSENSITIVE_OR_UNSUITABLE'
    LOCAL_RESOURCE_LIMIT = 'LOCAL_RESOURCE_LIMIT'
    NO_COMPARATIVE_ADVANTAGE = 'NO_COMPARATIVE_ADVANTAGE'
    SUPPORTED_IN_SCOPE = 'SUPPORTED_IN_SCOPE'

@dataclass(frozen=True)
class GateDecision(ImmutableArrays):
    code: str; evidence: object; rationale: str; next_action: str

@dataclass(frozen=True)
class TaskKey(ImmutableArrays):
    root_seed: int; phase_id: int; outer_replication: int; task_id: int; stream_id: int
    def __post_init__(self):
        super().__post_init__()
        if any(not isinstance(getattr(self, f.name), int) or isinstance(getattr(self, f.name), bool) or getattr(self, f.name) < 0 for f in fields(self)):
            raise ValueError('key fields must be nonnegative integers')

@dataclass(frozen=True)
class PopulationTruth(ImmutableArrays):
    second_moment: Array; sigma2: float; population_hash: str
    target_context_size: int = field(kw_only=True)
    input_law: str = field(default="gaussian_isotropic", kw_only=True)

# Required raw evidence fields. Units are encoded in field names where applicable;
# null measurements require an accompanying status/reason in the producing record.
RAW_RECORD_FIELDS = {
    'execution_task': frozenset('outer_replication result status failure_reason'.split()),
    'replication_completion': frozenset('outer_replication result status failure_reason'.split()),
    'execution': frozenset('outer_replication result status failure_reason'.split()),
    'task_metrics': frozenset('run_id outer_replication task_id task_key method_id information_access n d depth norm_cap parameter_cap eta branch loss_parameter_mse loss_noiseless_prediction_mse adaptation_ns prediction_ns status failure_reason'.split()),
    'preparation_risks': frozenset('run_id outer_replication source_key method_id selection_procedure_hash prior_asset_hash metric_hash analysis_mode source_training_tasks source_validation_tasks total_source_labels conditional_risk paired_contrast_id paired_risk_difference risk_evaluator_version true_S_evaluator_hash noise_truth_evaluator_hash numerical_error_assessment status failure_reason'.split()),
    'geometry': frozenset('target_hash realized_hash spectral_scale support_policy minimum_eigenvalue gamma initial_norm post_norm actual_trainable_count stored_scalars serialized_bytes null_residual kernel_absolute_error kernel_relative_error kernel_relative_error_status direct_step_error numerical_repairs'.split()),
    'certificates': frozenset('configuration_key branch objective primal_upper dual_lower primal_slack dual_feasibility stationarity_residual complementarity_residual solver_status solver_version solver_options repair_size certificate_level roundoff_treatment roundoff_enclosures claimed_risk_precision rational_numerators rational_denominators'.split()),
    'timing_blocks': frozenset('method_order cold_warm block_size generation_boundary preparation_boundary adaptation_boundary prediction_boundary serialization_boundary repetitions wall_time_ns peak_rss peak_rss_unit power_thermal_condition gpu_metadata timing_stratum'.split()),
    'decisions': frozenset('timestamp work_package observation evidence_paths evidence_hashes decision rationale affected_claim_ids next_action methodological_amendment_required fresh_confirmation_required'.split()),
}


def validate_unit_id(value: object) -> int:
    """Validate the canonical integer representation used in keys and inference."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError('unit IDs must be nonnegative integers')
    return value


def _count(value, name, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f'{name} must be a finite number')


def _text(value, name):
    if not isinstance(value, str) or not value.strip(): raise ValueError(f'{name} must be nonempty text')


def _sha(value, name):
    import re
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value): raise ValueError(f'{name} must be a SHA-256 hash')


def validate_run_record(run: JsonRecord) -> None:
    """Validate the persisted run identity and its declared measurement policy."""
    required = ('schema_version', 'run_id', 'phase', 'git_commit', 'dirty_diff_sha256', 'scientific_config_sha256', 'operational_config_sha256', 'analysis_mode', 'protocol_sha256', 'dependency_lock_sha256', 'hardware_record', 'data_manifest_sha256', 'prior_asset_sha256', 'information_access', 'timing_boundaries', 'status', 'pairing_spec')
    if not isinstance(run, dict) or any(k not in run for k in required) or run['schema_version'] != 1 or isinstance(run['schema_version'], bool): raise ValueError('incomplete run schema')
    import re
    if not isinstance(run['git_commit'], str) or not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', run['git_commit']): raise ValueError('invalid revision')
    for name in ('scientific_config_sha256', 'operational_config_sha256', 'protocol_sha256', 'dependency_lock_sha256'): _sha(run[name], name)
    for name in ('dirty_diff_sha256', 'data_manifest_sha256', 'prior_asset_sha256'):
        if run[name] is not None: _sha(run[name], name)
    for name in ('run_id', 'hardware_record'): _text(run[name], name)
    if not isinstance(run['timing_boundaries'], dict) or not run['timing_boundaries']: raise ValueError('timing boundaries required')
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k,v in run['timing_boundaries'].items()): raise ValueError('invalid timing boundaries')
    if run['phase'] not in ('exploration', 'confirmation') or run['status'] not in ('running', 'completed', 'incomplete', 'interrupted', 'invalid', 'failed') or run['information_access'] not in ('oracle', 'estimated', 'none') or run['analysis_mode'] not in ('deterministic_frontier', 'conditional_risk_over_preparations', 'fixed_prior_task_simulation', 'nested_task_simulation'): raise ValueError('invalid run enum')
    specs = run['pairing_spec']; expected = {'task_metrics':['outer_replication','task_id'], 'preparation_risks':['outer_replication']}
    if not isinstance(specs, dict): raise ValueError('invalid pairing spec')
    for kind, spec in specs.items():
        if kind not in expected or not isinstance(spec, dict) or spec.get('unit_fields') != expected[kind]: raise ValueError('invalid pairing unit')
        methods = spec.get('method_ids')
        if not isinstance(methods, list) or len(methods) < 2 or any(not isinstance(x, str) or not x for x in methods) or len(set(methods)) != len(methods): raise ValueError('invalid pairing methods')



def _fractions(numerators, denominators):
    if isinstance(numerators, dict) and isinstance(denominators, dict):
        if not numerators or numerators.keys() != denominators.keys(): raise ValueError('fraction keys differ')
        for key in numerators:
            _text(key, 'fraction key'); _fractions(numerators[key], denominators[key])
    elif isinstance(numerators, list) and isinstance(denominators, list):
        if not numerators or len(numerators) != len(denominators): raise ValueError('fraction shapes differ')
        for n, d in zip(numerators, denominators): _fractions(n, d)
    else:
        if not isinstance(numerators, int) or isinstance(numerators, bool): raise ValueError('invalid numerator')
        _count(denominators, 'denominator', 1)


def _enclosures(value):
    if isinstance(value, dict):
        if not value: raise ValueError('empty enclosures')
        for key, item in value.items():
            _text(key, 'enclosure key'); _enclosures(item)
    elif isinstance(value, list):
        if len(value) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
            for x in value: _number(x, 'enclosure endpoint')
            if value[0] > value[1]: raise ValueError('reversed enclosure')
        else:
            if not value: raise ValueError('empty enclosures')
            for item in value: _enclosures(item)
    else: raise ValueError('invalid enclosure')

def validate_raw_record(row: JsonRecord, run: JsonRecord) -> None:
    """Validate raw evidence without inventing values for unavailable measurements."""
    if not isinstance(row, dict): raise ValueError('record must be a mapping')
    kind = row.get('kind')
    if kind not in RAW_RECORD_FIELDS or not RAW_RECORD_FIELDS[kind].issubset(row): raise ValueError('incomplete raw record schema')
    to_json_record(row)  # JSON serialization is checked at the store boundary.
    status = row.get('status')
    if status is not None and status not in ('completed', 'failed', 'invalid'): raise ValueError('invalid record status')
    if status in ('failed', 'invalid'): _text(row.get('failure_reason'), 'failure_reason')
    if row.get('run_id', run['run_id']) != run['run_id']: raise ValueError('row run identity mismatch')
    for name in ('outer_replication', 'task_id'):
        if name in row: validate_unit_id(row[name])
    reasons = row.get('null_reasons', {})
    if not isinstance(reasons, dict): raise ValueError('null_reasons must be a mapping')
    for field in RAW_RECORD_FIELDS[kind]:
        if row[field] is None and field != 'failure_reason':
            if status in ('failed', 'invalid'): continue
            if not isinstance(reasons.get(field), str) or not reasons[field].strip(): raise ValueError(f'{field} requires null reason')
    if kind in ('task_metrics', 'preparation_risks'):
        for name in ('method_id', 'status'): _text(row[name], name)
    if kind in ('execution','execution_task','replication_completion'):
        if not isinstance(row['result'],dict) or row['result'].get('status')!=status:
            raise ValueError('invalid execution result')
        return
    if kind == 'task_metrics':
        _text(row['task_key'], 'task_key')
        for name in ('n','d','depth'): _count(row[name], name, 1)
        _count(row['parameter_cap'], 'parameter_cap')
        for name in ('norm_cap','eta'): _number(row[name], name, 0)
        if row['information_access'] not in ('oracle','estimated','none'): raise ValueError('invalid information access')
        if status == 'completed' or row['loss_parameter_mse'] is not None: _number(row['loss_parameter_mse'], 'loss_parameter_mse', 0)
        if row['loss_noiseless_prediction_mse'] is not None: _number(row['loss_noiseless_prediction_mse'], 'loss_noiseless_prediction_mse', 0)
        for name in ('adaptation_ns', 'prediction_ns'):
            if row[name] is not None: _count(row[name], name)
        if row['branch'] is not None and row['branch'] not in ('F', 'Z'): raise ValueError('invalid branch')
    elif kind == 'preparation_risks':
        for name in ('source_key','paired_contrast_id','risk_evaluator_version'): _text(row[name], name)
        if row['analysis_mode'] not in ('conditional_risk_over_preparations', 'nested_task_simulation'): raise ValueError('invalid preparation mode')
        if row['analysis_mode'] != run['analysis_mode']: raise ValueError('analysis mode mismatch')
        for name in ('source_training_tasks','source_validation_tasks','total_source_labels'): _count(row[name], name)
        for name in ('selection_procedure_hash','true_S_evaluator_hash','noise_truth_evaluator_hash'): _sha(row[name], name)
        for name in ('prior_asset_hash','metric_hash'):
            if row[name] is not None: _sha(row[name], name)
        if status == 'completed': _sha(row['metric_hash'], 'metric_hash')
        if status == 'completed' or row['conditional_risk'] is not None: _number(row['conditional_risk'], 'conditional_risk', 0)
        if row['paired_risk_difference'] is not None: _number(row['paired_risk_difference'], 'paired_risk_difference')
        if (status == 'completed' or row['numerical_error_assessment'] is not None) and not isinstance(row['numerical_error_assessment'], dict): raise ValueError('numerical error assessment required')
    elif kind == 'geometry':
        for name in ('target_hash','realized_hash'): _sha(row[name], name)
        if row['support_policy'] not in ('analytic','approximate','exact'): raise ValueError('invalid support policy')
        for name in ('actual_trainable_count','stored_scalars','serialized_bytes'): _count(row[name], name)
        for name in ('spectral_scale','minimum_eigenvalue','gamma','initial_norm','post_norm','null_residual','kernel_absolute_error','kernel_relative_error','direct_step_error'):
            if row[name] is not None: _number(row[name], name)
        _text(row['kernel_relative_error_status'], 'kernel_relative_error_status')
        if not isinstance(row['numerical_repairs'], (list, dict)): raise ValueError('invalid repairs')
        if row['kernel_relative_error'] is None and row['kernel_relative_error_status'] in ('ok', 'defined'): raise ValueError('undefined relative error requires status')
    elif kind == 'certificates':
        for name in ('configuration_key','solver_status','solver_version','certificate_level','roundoff_treatment'): _text(row[name], name)
        if row['branch'] not in ('F','Z'): raise ValueError('invalid branch')
        if row['certificate_level'] not in ('float64_diagnostic','exact_rational','rigorous_enclosure'): raise ValueError('invalid certificate level')
        if not isinstance(row['solver_options'], dict): raise ValueError('invalid solver options')
        for name in ('roundoff_enclosures', 'rational_numerators', 'rational_denominators'):
            if row[name] is not None and not isinstance(row[name], (dict, list)): raise ValueError(f'invalid {name}')
        if row['certificate_level'] == 'exact_rational' and (row['rational_numerators'] is None or row['rational_denominators'] is None): raise ValueError('exact rational certificate needs fractions')
        if row['certificate_level'] == 'rigorous_enclosure' and row['roundoff_enclosures'] is None: raise ValueError('enclosure certificate needs enclosures')
        if (row['rational_numerators'] is None) != (row['rational_denominators'] is None): raise ValueError('both fraction payloads required')
        if row['rational_numerators'] is not None: _fractions(row['rational_numerators'], row['rational_denominators'])
        if row['roundoff_enclosures'] is not None: _enclosures(row['roundoff_enclosures'])
        for name in ('objective','primal_upper','dual_lower','primal_slack','dual_feasibility','stationarity_residual','complementarity_residual','repair_size','claimed_risk_precision'):
            if row[name] is not None: _number(row[name], name)
    elif kind == 'timing_blocks':
        if row['cold_warm'] not in ('cold','warm') or row['peak_rss_unit'] not in ('bytes','KiB'): raise ValueError('invalid timing units or state')
        for name in ('block_size','repetitions'): _count(row[name], name, 1)
        for name in ('wall_time_ns','peak_rss'): _count(row[name], name)
        if not isinstance(row['method_order'], list) or not row['method_order'] or any(not isinstance(x, str) or not x for x in row['method_order']): raise ValueError('method order required')
        for name in ('generation_boundary','preparation_boundary','adaptation_boundary','prediction_boundary','serialization_boundary','power_thermal_condition','timing_stratum'): _text(row[name], name)
        if row['gpu_metadata'] is not None and not isinstance(row['gpu_metadata'], dict): raise ValueError('invalid GPU metadata')
    else:
        for name in ('timestamp','work_package','observation','decision','rationale','next_action'): _text(row[name], name)
        for name in ('evidence_paths','evidence_hashes','affected_claim_ids'):
            if not isinstance(row[name], list): raise ValueError('invalid decision evidence')
        for digest in row['evidence_hashes']: _sha(digest, 'evidence_hash')
        if any(not isinstance(x, str) or not x for name in ('evidence_paths','affected_claim_ids') for x in row[name]): raise ValueError('invalid evidence identifier')
        for name in ('methodological_amendment_required','fresh_confirmation_required'):
            if not isinstance(row[name], bool): raise ValueError('invalid decision flag')
