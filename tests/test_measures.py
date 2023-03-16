import numpy as np
from infotheo import measures
import matplotlib.pyplot as plt
from infotheo.tools import random_scope
from infotheo.probabilities import calc_proba, get_marginals


def show_coin_entropy():
    h = []
    ps = np.linspace(0, 1, 100)
    for p in ps:
        px = np.array([p, 1 - p])
        h.append(measures.entropy(px))
    plt.plot(ps, h)
    plt.title("Biased Coin Entropy")
    plt.ylabel("Entropy")
    plt.xlabel("P[heads]")
    plt.show()


@random_scope
def test_entropy_and_mi():
    np.random.seed(0)

    gt_Hx = 2
    gt_Hy = 1
    gt_Hxy = 2

    xbins = 4
    ybins = 2

    x = np.random.randint(4, size=10_000)
    y = np.mod(x, 2)

    assert len(set(x)) == xbins
    assert len(set(y)) == ybins

    pxy, _ = calc_proba([x, y], [xbins, ybins])
    px, py = get_marginals(pxy)

    Hx = measures.entropy(px)
    Hy = measures.entropy(py)
    Hxy = measures.entropy(pxy)

    if not np.all(np.isclose([gt_Hx, gt_Hy, gt_Hxy], [Hx, Hy, Hxy], rtol=1e-3)):
        print("Entropy test Failed.")
    else:
        print("Entropy test Passed.")

    if not np.isclose(measures.mi(pxy), Hx + Hy - Hxy):
        print("MI test Failed.")
    else:
        print("MI test Passed.")


@random_scope
def show_noise_mi():
    d = 5
    x = np.clip(np.random.standard_normal(10_000), -d, d)
    noise = np.clip(np.random.standard_normal(10_000), -d, d)
    sigs = np.linspace(0, 5, 100)
    bins = np.linspace(-d, d, 11)

    mi, Hx, Hy = [], [], []
    for sig in sigs:
        y = np.clip(x + sig * noise, -d, d)
        pxy, _ = calc_proba([x, y], [bins, bins])
        px, py = get_marginals(pxy)

        Hx.append(measures.entropy(px))
        Hy.append(measures.entropy(py))
        mi.append(measures.mi(pxy))

    plt.plot(sigs, mi, label="MI")
    plt.plot(sigs, Hx, label="Hx")
    plt.plot(sigs, Hy, label="Hy")
    plt.xlabel("Noise")
    plt.ylabel("Bits")
    plt.title("MI(X;Y) Where Y = X + Noise")
    plt.legend()
    plt.show()


test_entropy_and_mi()
