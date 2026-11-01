"""Deterministic, isolated Gaussian task and observation streams."""
import numpy as np
from .records import TaskKey, PopulationTruth, Context, SourceTask, content_hash
from .evaluation import EvaluationTruth

# These fixed domain tags are part of the replay contract. User stream_id remains
# an independent namespace, not a mutable RNG counter.
TASK, CONTEXT, NOISE, SOURCE_A_CONTEXT, SOURCE_A_NOISE, SOURCE_B_CONTEXT, SOURCE_B_NOISE, ROTATION = range(1, 9)
RNG_SPEC = {'bit_generator': 'PCG64', 'numpy_version': np.__version__, 'seed_sequence': 'numpy.random.SeedSequence', 'rotation_qr_sign': 'R diagonal nonnegative; zero uses +1', 'stream_mapping_version': 1}


def rng_for(key: TaskKey) -> np.random.Generator:
    if not isinstance(key, TaskKey): raise TypeError('expected TaskKey')
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(key.root_seed, spawn_key=(key.phase_id, key.outer_replication, key.task_id, key.stream_id))))


def _rng(key, tag):
    # Nesting the domain under the user stream prevents overlaps across streams.
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(key.root_seed, spawn_key=(key.phase_id, key.outer_replication, key.task_id, key.stream_id, tag))))


def population_for(config, key):
    """Return evaluator truth and replay artifacts; never pass either to a method."""
    p = config.preparation
    s = np.asarray(config.workload['spectrum'], dtype=np.float64)
    sigma2 = config.workload['sigma2']
    if p.input_law != 'isotropic_gaussian' or s.shape != (p.input_dimension,) or not np.all(np.isfinite(s)) or np.any(s < 0) or not np.isclose(s.sum(), 1, atol=1e-12, rtol=0) or not np.isfinite(sigma2) or sigma2 < 0:
        raise ValueError('invalid Gaussian population')
    # Population orientation is fixed for every task and phase of a root seed.
    rotation_key = TaskKey(key.root_seed, 0, 0, 0, 0)
    q, r = np.linalg.qr(_rng(rotation_key, ROTATION).normal(size=(len(s), len(s))))
    signs = np.where(np.diag(r) < 0, -1., 1.)
    q = q * signs
    S = (q*s) @ q.T
    artifact = {'rotation': q, 'spectrum': s, 'second_moment': S, 'rng': RNG_SPEC, 'rotation_key': rotation_key}
    phash = content_hash({'second_moment': S, 'sigma2': sigma2})
    artifact['artifact_hash'] = content_hash(artifact)
    return PopulationTruth(S, float(sigma2), phash, target_context_size=p.target_context_size), artifact


def _theta(config, key, artifact):
    s, q = artifact['spectrum'], artifact['rotation']
    return q @ (np.sqrt(s) * _rng(key, TASK).normal(size=len(s)))


def _block(key, theta, n, sigma2, xtag, etag):
    X = _rng(key, xtag).normal(size=(n, len(theta)))
    y = X @ theta + np.sqrt(sigma2) * _rng(key, etag).normal(size=n)
    return Context(X, y, content_hash({'key': key, 'context_stream': xtag, 'noise_stream': etag}))


def generate_task(config, key: TaskKey):
    population, artifact = population_for(config, key)
    theta = _theta(config, key, artifact)
    context = _block(key, theta, config.preparation.target_context_size, population.sigma2, CONTEXT, NOISE)
    return context, EvaluationTruth(theta, population.second_moment, population.sigma2, key)


def generate_source_task(config, key: TaskKey):
    population, artifact = population_for(config, key)
    theta = _theta(config, key, artifact)
    n = config.sample_plan.source_block_size
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0: raise ValueError('source block size required')
    A = _block(key, theta, n, population.sigma2, SOURCE_A_CONTEXT, SOURCE_A_NOISE)
    B = _block(key, theta, n, population.sigma2, SOURCE_B_CONTEXT, SOURCE_B_NOISE)
    return SourceTask(A, B, content_hash({'key': key}))
