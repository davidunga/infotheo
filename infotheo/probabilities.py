import numpy as np


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
    Joint and marginal distributions.
    Given lists of X and Y values, computes the pairwise joint distribution of every x~X,y~Y pairs.
    Args:
        xs: an array of X values, or a list of such arrays
        ys: Y values, given in the same format as xs
        xbins: x bin edges, or number of bins
        ybins: y bin edges, or number of bins
    Returns:
        if xs, ys were given as a list of nparrays:
            pxys: dict of 2d arrays. pxys[(i,j)] is an array representation of P(xs[i] | ys[j])
            pxs: list of 1d arrays, pxs[i] is the marginal distribution P(xs[i])
            pys: list of 1d arrays, pys[i] is the marginal distribution P(ys[i])
        if xs, ys are nparrays:
            pxys: 2d array of P(xs | ys)
            pxs: 1d array the marginal distribution P(xs)
            pys: 1d array the marginal distribution P(ys)
    """

    assert isinstance(xs, list) == isinstance(ys, list)
    multi_input = isinstance(xs, list)
    if not multi_input:
        xs, ys = [xs], [ys]

    pxys = {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            joint_counts, xbins, ybins = np.histogram2d(x, y, [xbins, ybins])
            pxys[(i, j)] = joint_counts / joint_counts.sum()

    pxs = [pxys[(i, 0)].sum(axis=1) for i in range(len(xs))]
    pys = [pxys[(0, j)].sum(axis=0) for j in range(len(ys))]

    for px in pxs:
        raise_invalid_proba(px)
    for py in pys:
        raise_invalid_proba(py)
    for pxy in pxys.values():
        raise_invalid_proba(pxy)

    if multi_input:
        return pxys, pxs, pys
    else:
        assert len(pxys) == len(pxs) == len(pys) == 1
        return pxys[(0, 0)], pxs[0], pys[0]
