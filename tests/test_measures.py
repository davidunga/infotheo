import numpy as np
from infotheo import measures
import matplotlib.pyplot as plt
from infotheo.tools import random_scope
from infotheo.probabilities import calc_distributions


def test_entropy():
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
def test_mi():
    d = 5
    x = np.clip(np.random.standard_normal(10_000), -d, d)
    noise = np.clip(np.random.standard_normal(10_000), -d, d)
    sigs = np.linspace(0, 5, 100)

    mi, Hx, Hy = [], [], []
    for sig in sigs:
        y = np.clip(x + sig * noise, -d, d)
        pxy, px, py = calc_distributions(x, y, np.linspace(-d, d, 10), np.linspace(-d, d, 10))

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


test_mi()
