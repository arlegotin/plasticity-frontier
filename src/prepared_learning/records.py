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
    RUNNING = 'running'; COMPLETED = 'completed'; INTERRUPTED = 'interrupted'; INVALID = 'invalid'; FAILED = 'failed'

@dataclass(frozen=True)
class BudgetSpec(ImmutableArrays):
    depth: int; initialization_norm_cap: float; trainable_parameter_cap: int; learning_rate: float
@dataclass(frozen=True)
class PreparationSpec(ImmutableArrays):
    target_context_size: int; input_dimension: int; input_law: str; loss_normalization: str; task_second_moment_trace: float
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
    metric: object; initial_layers: Layers | None; budget: BudgetSpec; preparation: PreparationSpec; resources: Resources; method_id: str; prior: PriorAsset | None = None; options: JsonRecord | None = None
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
    'task_metrics': frozenset('run_id outer_replication task_id task_key method_id information_access n d depth norm_cap parameter_cap eta branch loss_parameter_mse loss_noiseless_prediction_mse adaptation_ns prediction_ns status failure_reason'.split()),
    'preparation_risks': frozenset('run_id outer_replication source_key method_id selection_procedure_hash prior_asset_hash metric_hash analysis_mode source_training_tasks source_validation_tasks total_source_labels conditional_risk paired_contrast_id paired_risk_difference risk_evaluator_version true_S_evaluator_hash noise_truth_evaluator_hash numerical_error_assessment status failure_reason'.split()),
    'geometry': frozenset('target_hash realized_hash spectral_scale support_policy minimum_eigenvalue gamma initial_norm post_norm actual_trainable_count stored_scalars serialized_bytes null_residual kernel_absolute_error kernel_relative_error kernel_relative_error_status direct_step_error numerical_repairs'.split()),
    'certificates': frozenset('configuration_key branch objective primal_upper dual_lower primal_slack dual_feasibility stationarity_residual complementarity_residual solver_status solver_version solver_options repair_size certificate_level roundoff_treatment roundoff_enclosures claimed_risk_precision rational_numerators rational_denominators'.split()),
    'timing_blocks': frozenset('method_order cold_warm block_size generation_boundary preparation_boundary adaptation_boundary prediction_boundary serialization_boundary repetitions wall_time_ns peak_rss peak_rss_unit power_thermal_condition gpu_metadata timing_stratum'.split()),
    'decisions': frozenset('timestamp work_package observation evidence_paths evidence_hashes decision rationale affected_claim_ids next_action methodological_amendment_required fresh_confirmation_required'.split()),
}
