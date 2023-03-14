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


def calc_distributions(xs, ys, xbins, ybins):
    """
    pairwise-joint and marginal distributions.
    Args:
        xs: 2d ndarray, where each column corresponds to an X variable
        ys: 2d ndarray, where each column corresponds to a Y variable
        xbins: x bin edges, or number of bins
        ybins: y bin edges, or number of bins
    Returns:
        pxys: dict of 2d arrays. pxys[(i, j)] is an array representation of P(xs[:, i] | ys[:, j])
        pxs: list of 1d arrays, pxs[i] is the marginal distribution P(xs[:, i])
        pys: list of 1d arrays, pys[j] is the marginal distribution P(ys[:, j])
        bins: dict of x and y bins
    """

    if xs.ndim == 1: xs = xs[:, None]
    if ys.ndim == 1: ys = ys[:, None]

    assert xs.ndim == ys.ndim == 2

    if isinstance(xbins, int):  xbins = np.linspace(xs.min() - 1e-16, xs.max() + 1e-16, xbins + 1)
    if isinstance(ybins, int):  ybins = np.linspace(ys.min() - 1e-16, ys.max() + 1e-16, ybins + 1)

    pxys = {(i, j): np.histogram2d(x, y, [xbins, ybins])[0] / len(x)
            for (i, x), (j, y) in product(enumerate(xs.T), enumerate(ys.T))}

    pxs = [pxys[(i, 0)].sum(axis=1) for i in range(xs.shape[1])]
    pys = [pxys[(0, j)].sum(axis=0) for j in range(ys.shape[1])]

    for px in pxs:
        raise_invalid_proba(px)
    for py in pys:
        raise_invalid_proba(py)
    for pxy in pxys.values():
        raise_invalid_proba(pxy)

    bins = {'x': xbins, 'y': ybins}
    return pxys, pxs, pys, bins

