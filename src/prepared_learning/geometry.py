"""Zero-output network compilation and literal simultaneous SGD."""
import math
import numpy as np
from .metrics import canonical_spectrum, dense_metric, metric_resources
from .records import NetworkRealization, Resources, GeometryDiagnostics


def _resources(layers):
    count = sum(a.size for a in layers)
    return Resources(float(sum(np.sum(a*a) for a in layers)), count, count, sum(a.nbytes for a in layers))


def construct_canonical(metric, depth):
    resources = metric_resources(metric, depth)
    floor, basis, gaps = canonical_spectrum(metric)
    gamma = math.fsum([floor, *map(float, gaps)])
    d = basis.shape[0]
    if gamma == 0:
        layers = [np.zeros((1,d))] + [np.zeros((1,1)) for _ in range(depth-1)]
    else:
        log_gamma = math.log(gamma)
        log_e = log_gamma/(depth-1)
        log_t = (depth-2)*log_e
        positive = np.flatnonzero(gaps > 0)
        # Relative spectral information below the float64 dynamic range cannot be verified.
        if any(math.log(float(x))-log_gamma < math.log(np.nextafter(0.,1.)) for x in [floor, *gaps] if x > 0):
            raise ValueError('spectrum not representable at float64 relative scale')
        def weight(x):
            try: value = math.exp((math.log(float(x))-log_t)/2)
            except OverflowError: raise ValueError('weight not representable') from None
            if not math.isfinite(value) or value == 0: raise ValueError('weight not representable')
            return value
        W = np.zeros((len(positive)+int(floor>0),d))
        for j,i in enumerate(positive): W[j] = weight(gaps[i])*basis[:,i]
        a = np.zeros((1,len(W)))
        if floor > 0: a[0,-1] = weight(floor)
        suffix = math.exp(log_e/2)
        if not math.isfinite(suffix) or suffix == 0: raise ValueError('weight not representable')
        layers = [W,a] + [np.array([[suffix]]) for _ in range(depth-2)]
    return NetworkRealization(tuple(layers), _resources(layers), metric.support_metadata)


def construct_head_zero(metric):
    floor, basis, gaps = canonical_spectrum(metric)
    d = basis.shape[0]
    if floor > 0:
        # A complete QR supplies only the missing orthogonal complement; retain
        # the supplied eigenvectors to avoid rotating near-degenerate directions.
        complement = np.linalg.qr(basis, mode='complete')[0][:, basis.shape[1]:]
        vectors = np.concatenate((basis, complement), axis=1)
        vals = np.concatenate((floor + gaps, np.full(complement.shape[1], floor)))
        W = np.sqrt(vals)[:,None]*vectors.T
    else:
        indices = np.flatnonzero(gaps > 0)
        W = np.sqrt(gaps[indices])[:,None]*basis[:,indices].T
    if len(W) == 0: W = np.zeros((1,d))
    layers = (W, np.zeros((1,len(W))))
    return NetworkRealization(layers, _resources(layers), metric.support_metadata)


def effective_coefficient(layers):
    product = layers[0]
    for layer in layers[1:]: product = layer @ product
    return product.reshape(-1)


def parameter_gradients(layers, g):
    g = np.asarray(g, dtype=float)
    if g.shape != (layers[0].shape[1],) or not np.all(np.isfinite(g)): raise ValueError('invalid gradient')
    gradients = []
    for i, layer in enumerate(layers):
        right = np.eye(layers[0].shape[1])
        for prev in layers[:i]: right = prev @ right
        left = np.eye(1)
        for later in reversed(layers[i+1:]): left = left @ later
        gradients.append(left.T @ (g @ right.T)[None,:])
    return tuple(gradients)


def tangent_matrix(layers):
    d = layers[0].shape[1]
    result = np.zeros((d, d))
    right = np.eye(d)
    for i, layer in enumerate(layers):
        left = np.eye(1)
        for later in reversed(layers[i+1:]): left = left @ later
        result += float(np.sum(left*left)) * (right.T @ right)
        right = layer @ right
    return result


def simultaneous_step(layers, g, eta):
    if not np.isfinite(eta): raise ValueError('invalid learning rate')
    gradients = parameter_gradients(layers, g)
    return tuple(np.array(a-eta*grad, copy=True) for a,grad in zip(layers, gradients))


def diagnose_geometry(target, realization):
    expected = dense_metric(target); actual = tangent_matrix(realization.layers)
    delta = actual-expected
    abs_f = float(np.linalg.norm(delta, 'fro')); ref = float(np.linalg.norm(expected, 'fro'))
    abs_op = float(np.linalg.norm(delta, 2)); op_ref = float(np.linalg.norm(expected,2))
    def rel(a,b): return (a/b, 'defined') if b else ((0.0,'exact_zero') if not a else (None,'undefined_zero_reference'))
    rf, sf = rel(abs_f,ref); ro, so = rel(abs_op,op_ref)
    coefficient = effective_coefficient(realization.layers)
    product_norm = math.prod(float(np.linalg.norm(a)) for a in realization.layers)
    null, sn = rel(float(np.linalg.norm(coefficient)),product_norm)
    residuals = {'absolute_frobenius':abs_f, 'relative_frobenius':rf, 'relative_frobenius_status':sf, 'absolute_operator':abs_op, 'relative_operator':ro, 'relative_operator_status':so, 'null_output':float(np.linalg.norm(coefficient)), 'normalized_null_output':null, 'normalized_null_status':sn, 'initialization_norm':realization.resources.initialization_norm, 'trainable_parameters':realization.resources.trainable_parameters}
    return GeometryDiagnostics(residuals, target.support_metadata.projections + target.support_metadata.truncations)
