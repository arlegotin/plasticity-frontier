"""Floating-point diagnostics are not rigorous certificates."""
from .quadratic import diagnose_qp

__all__ = ['diagnose_qp']


def empirical_residual_bound(qp, x, multipliers, k):
    """Supporting-plane lower bound on the branch box [0,k]^d; float diagnostic."""
    import numpy as np
    x=np.asarray(x,dtype=float); v=np.asarray(multipliers,dtype=float)
    if not np.isfinite(k) or k<0 or x.shape!=qp.linear.shape or v.shape!=qp.offset.shape or np.any(v<0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
        raise ValueError('invalid bound inputs')
    if np.linalg.eigvalsh(qp.hessian).min() < -1e-12 or not np.allclose(qp.hessian,qp.hessian.T): raise ValueError('Hessian must be PSD')
    if np.any(x<0) or np.any(x>k) or np.any(qp.A@x+qp.offset < -1e-12): raise ValueError('x must be feasible')
    # The branch domain must be contained in the box [0,k]^d.
    grad=qp.linear+2*qp.hessian@x-qp.A.T@v
    value=qp.constant+qp.linear@x+x@qp.hessian@x-v@(qp.A@x+qp.offset)
    return float(value-k*np.sum(np.abs(grad)))
