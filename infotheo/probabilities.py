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


def make_joint_variable(xs, bins, norm=False):
    """
    Construct variable Z ~ (X1, X2, ..)
    Args:
        xs: list of 1d np arrays, all of the same length, xs = [x1, x2, ..]
        bins: list, bin spec for each x in xs
        norm: if True, normalizes the values of the joint variable to be between 0 and 1. If False (default),
            values are between 0 and (len(bins[0]) * len(bins[1]) * ..) - 1.
    Returns:
        an array, same size as x1
    """
    multi_index = []
    for j in range(len(xs)):
        if isinstance(bins[j], int):
            bins[j] = np.linspace(np.min(xs[j]) - 1e-12, np.max(xs[j]) + 1e-12, bins[j] + 1)
        multi_index.append(np.digitize(xs[j], bins[j]) - 1)
    dims = [len(b) - 1 for b in bins]
    z = np.ravel_multi_index(multi_index, dims=dims)
    if norm:
        z = z.astype(float) / (np.prod(dims) - 1)
    return z
