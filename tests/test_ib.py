import numpy as np
import matplotlib.pyplot as plt
from tools import random_scope
import ib


@random_scope
def test_iib():
    """
    Example test similar to the one in: Chechik et al 2002.
    """

    np.random.seed(0)

    def add_line_(m, a, r, j, istart, istop):
        m[istart:istop, j-r:j+r] = a

    Nx = 25
    Ny = 15
    x, y = np.mgrid[:Nx, :Ny]

    r = 2

    pxy1 = np.zeros_like(x, float)
    add_line_(pxy1, a=1, j=r, r=r, istart=0, istop=int(Nx / 2))
    add_line_(pxy1, a=1, j=2*r, r=r, istart=int(Nx / 2), istop=Nx)

    add_line_(pxy1, a=.5, j=5*r, r=r, istart=0, istop=int(Nx / 3))
    add_line_(pxy1, a=.5, j=4*r, r=r, istart=int(Nx / 3), istop=int(2 * Nx / 3))
    add_line_(pxy1, a=.5, j=5 * r, r=r, istart=int(2 * Nx / 3), istop=Nx)

    pxy2 = np.zeros_like(x, float)
    add_line_(pxy2, a=1, j=r, r=r, istart=0, istop=int(Nx / 2))
    add_line_(pxy2, a=1, j=2*r, r=r, istart=int(Nx / 2), istop=Nx)

    pxy1 += .05 * np.random.random(pxy1.shape)
    pxy2 += .05 * np.random.random(pxy2.shape)

    pxy2 /= pxy2.sum(axis=1, keepdims=True)
    pxy2 *= pxy1.sum(axis=1, keepdims=True)

    pxy1 /= pxy1.sum()
    pxy2 /= pxy2.sum()

    c1, d1 = ib.aib([pxy1, pxy2], [1, 0])
    c2, d2 = ib.aib([pxy1, pxy2], [1, -1])

    _, axs = plt.subplots(1, 4, figsize=(12, 5))

    plt.sca(axs[0])
    plt.imshow(pxy1)
    plt.title("PXY1")
    plt.xlabel("Y")
    plt.ylabel("X")

    plt.sca(axs[1])
    plt.imshow(pxy2)
    plt.title("PXY2")

    plt.sca(axs[2])
    plt.plot(c1[1], np.arange(len(c1)), '*')
    plt.title("No Side Info")
    plt.xlabel("T")
    plt.ylabel("X")

    plt.sca(axs[3])
    plt.plot(c2[1], np.arange(len(c2)), '*')
    plt.title("With Side Info")

    plt.show()



test_iib()