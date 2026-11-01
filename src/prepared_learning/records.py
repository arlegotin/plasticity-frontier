"""Shared immutable records; mathematical behavior belongs in later modules."""
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Layers = tuple[Array, ...]
JsonRecord = dict[str, object]

def content_hash(value: JsonRecord) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _freeze(value):
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True); result.setflags(write=False); return result
    if isinstance(value, tuple): return tuple(_freeze(v) for v in value)
    return value

class ImmutableArrays:
    def __post_init__(self):
        for field in fields(self): object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))

class Status(str, Enum):
    RUNNING = 'running'; COMPLETED = 'completed'; INTERRUPTED = 'interrupted'; INVALID = 'invalid'; FAILED = 'failed'

@dataclass(frozen=True)
class BudgetSpec:
    depth: int; initialization_norm_cap: float; trainable_parameter_cap: int; learning_rate: float
@dataclass(frozen=True)
class PreparationSpec:
    target_context_size: int; input_dimension: int; input_law: str; loss_normalization: str; task_second_moment_trace: float
@dataclass(frozen=True)
class SamplePlan:
    analysis_mode: str; outer_replicates: int | None; tasks_per_replicate: int | None; source_training_tasks: int | None; source_validation_tasks: int | None; source_block_size: int | None
@dataclass(frozen=True)
class RunConfig:
    schema_version: int; workload: dict; budget: BudgetSpec; preparation: PreparationSpec; numerics: dict; access: dict; methods: tuple; keys: dict; sample_plan: SamplePlan; confirmation: dict
@dataclass(frozen=True)
class SupportMetadata:
    policy: str; exact_zero_multiplicity: int; minimum_multiplicity: int; retained_indices: tuple; projections: tuple = (); truncations: tuple = ()
@dataclass(frozen=True)
class Resources:
    initialization_norm: float; trainable_parameters: int; stored_scalars: int; serialized_bytes: int
@dataclass(frozen=True)
class NetworkRealization(ImmutableArrays):
    layers: Layers; resources: Resources; support_metadata: SupportMetadata
@dataclass(frozen=True)
class GeometryDiagnostics:
    residuals: JsonRecord; repairs: tuple
@dataclass(frozen=True)
class QPData(ImmutableArrays):
    constant: float; linear: Array; hessian: Array; A: Array; offset: Array
@dataclass(frozen=True)
class NumericDiagnostics:
    feasibility: object; kkt: object; dual: object; roundoff: object
@dataclass(frozen=True)
class BranchResult:
    branch: str; variables: object; metric: object; objective: float; diagnostics: NumericDiagnostics; status: str
@dataclass(frozen=True)
class FrontierResult:
    F: BranchResult; Z: BranchResult; selected_branches: tuple; selection_status: str
@dataclass(frozen=True)
class Context(ImmutableArrays):
    X: Array; y: Array; task_key: str
@dataclass(frozen=True)
class SourceTask(ImmutableArrays):
    A: Array; B: Array; source_key: str
@dataclass(frozen=True)
class PriorAsset(ImmutableArrays):
    covariance: Array | None; basis: Array | None; eigenvalues: Array | None; noise_variance: float | None; information_access: str; fit_metadata: JsonRecord; asset_hash: str
@dataclass(frozen=True)
class PreparedState(ImmutableArrays):
    metric: object; initial_layers: Layers | None; budget: BudgetSpec; preparation: PreparationSpec; resources: Resources; method_id: str
@dataclass(frozen=True)
class AdaptedState(ImmutableArrays):
    coefficient: Array; layers: Layers | None; status: str; timing: object
@dataclass(frozen=True)
class SelectionResult:
    selected_id: str; candidate_records: tuple; asset: PriorAsset | None
class GateCode(str, Enum):
    IMPLEMENTATION_INVALID = 'IMPLEMENTATION_INVALID'
    MATHEMATICAL_COUNTEREXAMPLE = 'MATHEMATICAL_COUNTEREXAMPLE'
    INSENSITIVE_OR_UNSUITABLE = 'INSENSITIVE_OR_UNSUITABLE'
    LOCAL_RESOURCE_LIMIT = 'LOCAL_RESOURCE_LIMIT'
    NO_COMPARATIVE_ADVANTAGE = 'NO_COMPARATIVE_ADVANTAGE'
    SUPPORTED_IN_SCOPE = 'SUPPORTED_IN_SCOPE'

@dataclass(frozen=True)
class GateDecision:
    code: str; evidence: object; rationale: str; next_action: str
