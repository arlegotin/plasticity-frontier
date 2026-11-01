"""Complex-step differentiation of the literal product, independent of geometry."""
import numpy as np


def complex_step_jacobian(layers, step=1e-25):
    if not np.isfinite(step) or step <= 0: raise ValueError('step must be positive and finite')
    arrays = [np.asarray(a, dtype=np.complex128).copy() for a in layers]
    columns = []
    for array in arrays:
        for index in np.ndindex(array.shape):
            array[index] += 1j*step
            product = arrays[0]
            for layer in arrays[1:]: product = layer @ product
            columns.append(product.reshape(-1).imag/step)
            array[index] -= 1j*step
    return np.stack(columns, axis=1)
