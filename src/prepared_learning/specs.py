"""Strict loading and validation of scientific configuration."""
import json
import math
from pathlib import Path
from .records import BudgetSpec, PreparationSpec, SamplePlan, RunConfig

def load_config(path: Path) -> RunConfig:
    def bad(value): raise ValueError(f'nonstandard JSON constant: {value}')
    try:
        data = json.loads(path.read_text(), parse_constant=bad)
        return RunConfig(schema_version=data['schema_version'], workload=data['workload'], budget=BudgetSpec(**data['budget']), preparation=PreparationSpec(**data['preparation']), numerics=data['numerics'], access=data['access'], methods=tuple(data['methods']), keys=data['keys'], sample_plan=SamplePlan(**data['sample_plan']), confirmation=data.get('confirmation', {}))
    except (KeyError, TypeError, json.JSONDecodeError) as exc: raise ValueError(f'invalid config: {exc}') from exc

def validate_config(config: RunConfig, confirmation: bool) -> None:
    b, p, s = config.budget, config.preparation, config.sample_plan
    def positive_int(x): return isinstance(x, int) and not isinstance(x, bool) and x > 0
    def finite_nonnegative(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
    if config.schema_version != 1 or not positive_int(b.depth) or b.depth < 2 or b.learning_rate != 1.0: raise ValueError('invalid main architecture')
    if not finite_nonnegative(b.initialization_norm_cap) or not isinstance(b.trainable_parameter_cap, int) or isinstance(b.trainable_parameter_cap, bool) or b.trainable_parameter_cap < 0: raise ValueError('invalid caps')
    if not positive_int(p.target_context_size) or not positive_int(p.input_dimension) or b.trainable_parameter_cap < p.input_dimension + b.depth - 1: raise ValueError('infeasible dimensions or parameter cap')
    if p.input_law != 'gaussian_isotropic' or p.loss_normalization != 'squared_error_over_2n' or p.task_second_moment_trace != 1: raise ValueError('invalid core preparation')
    seed=config.workload.get('population_seed')
    if seed is not None and (type(seed) is not int or seed<0): raise ValueError('invalid population seed')
    spectrum = config.workload.get('spectrum'); sigma2 = config.workload.get('sigma2')
    if not isinstance(spectrum, (list, tuple)) or len(spectrum) != p.input_dimension or not all(finite_nonnegative(x) for x in spectrum) or not math.isclose(sum(spectrum), 1.0, abs_tol=1e-12, rel_tol=0): raise ValueError('invalid spectrum')
    if not finite_nonnegative(sigma2) or config.numerics.get('dtype') != 'float64': raise ValueError('invalid numeric contract')
    if config.access.get('selection') == 'learned_only' and (config.access.get('prior') == 'oracle' or 'oracle' in config.methods): raise ValueError('oracle asset in learned-only selection')
    if s.analysis_mode not in ('deterministic_frontier', 'conditional_risk_over_preparations', 'nested_task_simulation', 'fixed_prior_task_simulation'): raise ValueError('unknown analysis mode')
    for value in (s.outer_replicates, s.tasks_per_replicate, s.source_training_tasks, s.source_validation_tasks, s.source_block_size):
        if value is not None and not positive_int(value): raise ValueError('invalid sample count')
    if confirmation:
        import re
        if not config.confirmation.get('primary_contrast') or not re.fullmatch(r'[0-9a-f]{64}', str(config.confirmation.get('environment_lock_sha256', ''))): raise ValueError('missing confirmation identity')
        if s.analysis_mode in ('conditional_risk_over_preparations', 'nested_task_simulation') and s.outer_replicates is None: raise ValueError('missing outer count')
        if s.analysis_mode in ('nested_task_simulation', 'fixed_prior_task_simulation') and s.tasks_per_replicate is None: raise ValueError('missing inner count')
        if s.analysis_mode in ('conditional_risk_over_preparations', 'nested_task_simulation') and any(x is None for x in (s.source_training_tasks, s.source_validation_tasks, s.source_block_size)): raise ValueError('missing source count')
