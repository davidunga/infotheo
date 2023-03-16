import numpy as np
from itertools import product


class InvalidProbability(Exception):
    pass


def raise_invalid_proba(p, axis=None, tol=1e-12, msg_prefix=""):
    if msg_prefix:
        msg_prefix += ": "
    if np.any(np.isnan(p)):
        raise InvalidProbability(msg_prefix + "NaNs")
    if not np.all(p >= 0):
        raise InvalidProbability(msg_prefix + "Negative values", p.min())
    s = p.sum(axis=axis)
    if np.any(np.abs(1 - s) > tol):
        raise InvalidProbability(msg_prefix + f"Does not sum to 1 along axis={axis}:" + str(s))


def normalize_proba(x: np.ndarray):
    return x / x.sum(axis=np.ndim(x)-1, keepdims=True)


def calc_proba(xs, bins):
    """
    Normalized joint counts
    Args:
        xs: np array, or list of np arrays (for joint probability)
        bins: bin specification for each x in xs., either as int (number of bins), or as array of bin edges
    Returns:
        p: np array with size (len(xs[0]), len(xs[1]), ..). p[i,j,k..] is the normalized count of
            co-occurrence of (xs[0][i], xs[1][j], xs[2][k], ..)
        bins: dict of x and y bins
    """

    p, bins = np.histogramdd(xs, bins)
    p /= len(xs[0])
    raise_invalid_proba(p)
    return p, bins


def get_marginals(p):
    return [p.sum(axis=tuple(np.nonzero(np.arange(p.ndim) != a)[0])) for a in range(p.ndim)]
