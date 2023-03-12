"""
Entropy, KL Divergence, etc.
Functions in this module operate only on pre-computed probability distributions.
"""


import numpy as np
from probabilities import raise_invalid_proba


def kldiv(p, q, axis=None):
    """
    Kullback–Leibler Divergence between distributions p and q.
    - if axis = -1, computes the KLDiv between all rows of p and of q:
        d = kldiv(p, q, axis=-1) returns a 2d array of size (len(p), len(q)), such that d[i, j] = KLDiv(p[i]|q[j]).
    - if axis >= 0 or is None, computes the KLDiv(p|q) along given axis. e.g., for axis=0, d[i] = KLDiv(p[i]|q[i]).
    """

    if axis == -1:
        kl = np.zeros((len(p), len(q)), float)
        for i in range(len(p)):
            kl[i, :] = kldiv(*np.broadcast_arrays(p[i], q), axis=1)
        assert not np.any(np.isnan(kl))
        return kl

    assert p.shape == q.shape
    raise_invalid_proba(p, axis=axis)
    raise_invalid_proba(q, axis=axis)

    mask = (p > 0) & (q > 0)
    assert np.all(q[mask] > 0)
    lg = np.zeros_like(p)
    lg[mask] = np.log(p[mask] / q[mask])
    return np.sum(p * lg, axis=axis)


def jsdiv(p, q, axis=None):
    """
    Jensen–Shannon divergence. Same API as kldiv().
    """
    return .5 * (kldiv(p, q, axis=axis) + kldiv(q, p, axis=axis))


def entropy(p, axis=None):
    """
    Entropy of distribution p along given axis.
    """
    raise_invalid_proba(p, axis=axis)
    log_p = np.zeros_like(p)
    log_p[p > 0] = np.log(p[p > 0])
    return -np.sum(p * log_p, axis=axis)


def mi(pxy):
    """
    Mutual information of joint distribution, pxy[i,j] = P(X=xi,Y=yj)
    """
    px_py = pxy.sum(axis=1, keepdims=True) @ pxy.sum(axis=0, keepdims=True)
    return kldiv(pxy.flatten(), px_py.flatten())
