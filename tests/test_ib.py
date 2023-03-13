import numpy as np
import matplotlib.pyplot as plt
from tools import random_scope, clusters2proba
import ib


@random_scope
def test_iib(ib_kind="a"):
    """
    Synthetic example similar to the one in Chechik & Tishby 2002, section 6.1.
    Args:
        ib_kind: "a" = Agglomerative, "i" = Iterative
    """
    pxy1, pxy2 = _prep_test_distributions()

    if ib_kind == "a":
        c, d = ib.aib(pxy1, 1)
        c_si, d_si = ib.aib([pxy1, pxy2], [1, -1])
        p_t_given_x = clusters2proba(c[1], pxy1.shape)
        p_t_given_x_si = clusters2proba(c_si[1], pxy1.shape)
    elif ib_kind == "i":
        pr, _ = ib.iib(pxy1, 1, Nt=2)
        pr_si, _ = ib.iib([pxy1, pxy2], [1, -1], Nt=2)
        p_t_given_x = pr['p_t_given_x']
        p_t_given_x_si = pr_si['p_t_given_x']
    else:
        raise ValueError("Unknown IB kind")

    _, axs = plt.subplots(1, 4, figsize=(12, 5))

    plt.sca(axs[0])
    plt.imshow(pxy1, cmap='gray')
    plt.title("P(X,Y1)")
    plt.xlabel("Y")
    plt.ylabel("X")

    plt.sca(axs[1])
    plt.imshow(pxy2, cmap='gray')
    plt.title("P(X,Y2)")
    plt.xlabel("Y")

    plt.sca(axs[2])
    plt.imshow(p_t_given_x, cmap='gray')
    plt.title("P(T|X) No Side Info")
    plt.xlabel("T")

    plt.sca(axs[3])
    plt.imshow(p_t_given_x_si, cmap='gray')
    plt.title("P(T|X) With Side Info")
    plt.xlabel("T")

    plt.suptitle(ib_kind + "IB with Side Information")
    plt.show()


@random_scope
def _prep_test_distributions(Nx=25, Ny=20, rand_seed=0, weak=.5, noise=.05):
    np.random.seed(rand_seed)

    r = 1

    def add_line_(m, a, j, istart, istop):
        m[istart:istop, j * r: j * r + r] = a

    x, y = np.mgrid[:Nx, :Ny]

    pxy2 = np.zeros_like(x, float)
    add_line_(pxy2, a=1, j=1, istart=0, istop=int(Nx / 2))
    add_line_(pxy2, a=1, j=2, istart=int(Nx / 2), istop=Nx)

    pxy1 = pxy2.copy()
    add_line_(pxy1, a=weak, j=5, istart=0, istop=int(Nx / 3))
    add_line_(pxy1, a=weak, j=4, istart=int(Nx / 3), istop=int(2 * Nx / 3))
    add_line_(pxy1, a=weak, j=5, istart=int(2 * Nx / 3), istop=Nx)

    pxy1 += noise * np.random.random(pxy1.shape)
    pxy2 += noise * np.random.random(pxy2.shape)

    pxy2 /= pxy2.sum(axis=1, keepdims=True)
    pxy2 *= pxy1.sum(axis=1, keepdims=True)

    pxy1 /= pxy1.sum()
    pxy2 /= pxy2.sum()

    return pxy1, pxy2


if __name__ == "__main__":
    test_iib("a")
