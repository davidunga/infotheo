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


def calc_distributions(samples, domains):
    assert len(samples) == len(domains)
    assert len(set([len(s) for s in samples])) == 1

    marginals = []
    for i in range(len(domains)):
        if not hasattr(domains[i], '__len__'):
            domains[i] = np.arange(domains[i])
            domains[i][0] -= 1e-16
            domains[i][-1] += 1e-16
        marginals.append(np.histogram(samples[i], domains[i])[0] / len(samples[i]))

    joint = np.histogramdd(np.array(samples).T, domains)[0] / len(samples[0])

    return joint, *tuple(marginals)
