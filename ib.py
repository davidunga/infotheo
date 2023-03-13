"""
Information Bottleneck
"""

import measures
import numpy as np
from probabilities import normalize_proba, raise_invalid_proba
from tools import random_scope


def aib(pxys, betas=None, normalize_betas=False):
    """
    Agglomerative Information Bottleneck with side information.
    Args:
        pxys: either a 2d array such that pxy[i,j] = P(x=i,y=j), or a list of such arrays
        betas: either a scalar (if pxys is not a list), or a list such that betas[i] corresponds to pxys[i].
        normalize_betas: rescale each beta to normalize mutual information cost elements.
            i.e., given the cost: L = MI(X;T) - beta_1 * MI(X;Y1) - beta_2 * MI(X;Y2) ..
            Scales each beta to normalize its corresponding MI term: beta_i <- beta_i * H(X) / H(Y_i)
    Returns:
        clusters: list of size = |X|, clusters[k][i] is the cluster label of X=xi for nClusters=k
        costs: costs[k] is the cost of splitting k-1 clusters into k clusters
    """

    if betas is None:
        betas = 1.0

    if not isinstance(pxys, list):
        pxys = [pxys.copy()]
        betas = [betas]
    else:
        pxys = [pxy.copy() for pxy in pxys]

    if np.max([measures.jsdiv(pxys[0].sum(axis=1), pxy.sum(axis=1)) for pxy in pxys]) > np.finfo(float).eps:
        raise ValueError("At least two of the P(X,Y)'s have different marginal distribution P(x)")

    if normalize_betas:
        betas = _normalize_betas(pxys, betas)

    def _cluster_dist(ci, cj):
        px_i = pxys[0][ci].sum()
        px_j = pxys[0][cj].sum()
        if np.isnan(px_i) or np.isnan(px_j):
            return np.Inf
        cost = .0
        for pxy, beta in zip(pxys, betas):
            cost += beta * measures.jsdiv(pxy[ci] / px_i, pxy[cj] / px_j)
        return (px_i + px_j) * cost

    n = len(pxys[0])
    d = np.zeros((n, n), float) + np.Inf
    for i in range(n - 1):
        for j in range(i + 1, n):
            d[i, j] = _cluster_dist(i, j)

    clusters = [np.arange(n)]
    costs = [0]
    for itr in range(1, n):
        i, j = np.unravel_index(np.argmin(d), d.shape)
        assert np.isfinite(d[i, j])
        assert i < j
        costs.append(d[i, j])
        for pxy in pxys:
            pxy[i] += pxy[j]
            pxy[j] = np.nan
        d[i, i + 1:] = [_cluster_dist(i, k) for k in range(i + 1, n)]
        d[j, :] = np.Inf
        d[:, j] = np.Inf
        clusters.append(clusters[-1].copy())
        clusters[-1][clusters[-1] == j] = i
        assert len(set(clusters[-1])) == (n - itr)

    clusters = clusters[::-1]
    costs = costs[::-1]

    return clusters, costs


# ----------------------------------------


@random_scope
def iib(pxys, betas, Nt, max_itrs=1000, converge_eps=1e-12, rand_seed=None,
        normalize_betas=False, beta_search=True, init_noise=.1):
    """
    Iterative Information Bottleneck with side information.
    Given distributions: P(X), P(X,Y1) <, P(X,Y2), P(X,Y3), ..> and corresponding betas: b1, <, b2, b3, ..>,
    computes soft clustering P(T|X) which minimizes:
        L = MI(X;T) - b1*MI(T;Y1) - b2*MI(T;Y2), ..
    Args:
        pxys: either a 2d array such that pxy[i,j] = P(x=i,y=j), or a list of such arrays
        betas: either a scalar (if pxys is not a list), or a list such that betas[i] corresponds to pxys[i]
        Nt: number of clusters (domain size of T)
        max_itrs: max number of iterations
        converge_eps: early stop if max{JSDiv[P(T|X)_current,P(T|X)_prev]} < converge_eps
        rand_seed: random seed
        normalize_betas: rescale each beta to normalize mutual information cost elements.
            i.e., given the cost: L = MI(X;T) - beta_1 * MI(X;Y1) - beta_2 * MI(X;Y2) ..
            Scales each beta to normalize its corresponding MI term: beta_i <- beta_i * H(X) / H(Y_i)
        beta_search: binary search for a global betas scaling factor. seeks a factor such that
            betas <- factor * betas yield "healthy" optimization behaviour.
        init_noise: P(T|X) is initialized to: 1/Nt + UniformRandom * init_noise
    Returns:

    """

    # ---- !Keep at top ----
    if beta_search:
        return _iib_beta_binary_search(**locals())
    # ----------------------

    if rand_seed:
        np.random.seed(rand_seed)

    if not isinstance(pxys, list):
        betas = [betas]
        pxys = [pxys]

    def _calc_ys_given_x(p_t_given_x, pt):
        return [normalize_proba(((pxy.T @ p_t_given_x) / pt).T) for pxy in pxys]

    def _calc_cost(p_ys_given_t, p_t_given_x, pt):
        cost = measures.mi(p_t_given_x * px[:, None])
        for beta, p_y_given_t in zip(betas, p_ys_given_t):
            cost -= beta * measures.mi(p_y_given_t * pt[:, None])
        return cost

    px = pxys[0].sum(axis=1)

    if len(px) < Nt:
        raise ValueError(f"Number of clusters ({Nt}) exceeds X domain size ({len(px)})")

    if np.max([measures.jsdiv(px, pxy.sum(axis=1)) for pxy in pxys]) > np.finfo(float).eps:
        raise ValueError("At least two of the P(X,Y)'s have different marginal distribution P(x)")

    raise_invalid_proba(px, msg_prefix="Px")
    for i in range(len(pxys)):
        raise_invalid_proba(pxys[i], msg_prefix=f"Pxy[{i}]")

    if normalize_betas:
        betas = _normalize_betas(pxys, betas)

    # init P(t|x) as uniform + small perturbation:
    p_t_given_x = normalize_proba(.5 + 2 * init_noise * (np.random.random((len(px), Nt)) - .5))
    pt = normalize_proba(p_t_given_x.T @ px)

    p_ys_given_t = _calc_ys_given_x(p_t_given_x, pt)
    p_ys_given_x = [normalize_proba(pxy / px[:, None]) for pxy in pxys]

    optim = {"cost": [], "js": [], "converged": False}
    for itr in range(max_itrs):
        p_t_given_x_prev = p_t_given_x

        klsum = .0
        for i in range(len(betas)):
            klsum += -betas[i] * measures.kldiv(p_ys_given_x[i], p_ys_given_t[i], axis=-1)
        p_t_given_x = normalize_proba(pt[None, :] * np.exp(klsum))

        p_ys_given_t = _calc_ys_given_x(p_t_given_x, pt)
        pt = normalize_proba(p_t_given_x.T @ px)

        optim["cost"].append(_calc_cost(p_ys_given_t, p_t_given_x, pt))
        optim["js"].append(measures.jsdiv(p_t_given_x, p_t_given_x_prev, axis=1).max())
        if optim["js"][-1] < converge_eps:
            optim["converged"] = True
            break

    optim["itrs"] = len(optim["js"])

    probas = {
        "p_t_given_x": p_t_given_x,
        "pt": pt,
        "p_ys_given_t": p_ys_given_t
    }

    return probas, optim


def _iib_beta_binary_search(pxys, betas, **kwargs):

    kwargs['beta_search'] = False

    if not hasattr(betas, '__len__'):
        betas = [betas]
        pxys = [pxys]
    betas = np.array(betas)

    probas, optim = None, None

    min_scale, max_scale = .1, 50

    while max_scale - min_scale > 2:  # (keep scale at least a unit apart from search bounds)

        scale = (min_scale + max_scale) / 2

        probas_, optim_ = iib(pxys, scale * betas, **kwargs)
        js = np.array(optim_['js'])

        if np.any((js[1:-1] > js[2:]) & (js[1:-1] > js[:-2])):
            # js overcame a local maxima, and converged

            probas, optim = probas_, optim_
            optim['beta_scale'] = scale

            # try finding something closer to 1..
            if scale > 1:
                max_scale = scale
            else:
                min_scale = scale

        elif abs(js[-2] - js[-1]) > 1e-5:
            max_scale = scale  # js did not converge- classify as overshoot
        else:
            min_scale = scale  # js converged- classify as undershoot

    return probas, optim


def _normalize_betas(pxys, betas):
    Hx = measures.entropy(pxys.sum(axis=1))
    for i in range(len(betas)):
        Hy = measures.entropy(pxys[i].sum(axis=0))
        betas[i] *= Hx / Hy
    return betas
