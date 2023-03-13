import numpy as np


def random_scope(fnc):
    """
    Decorator for isolating numpy random state
    """
    def inner(*args, **kwargs):
        rand_state = np.random.get_state()
        ret = fnc(*args, **kwargs)
        np.random.set_state(rand_state)
        return ret
    return inner


def clusters2proba(c, shape):
    """
    Make P(T|X) from hard clustering labels
    Args:
        c: 1d array, c[i] is the label of x=i
        shape: (Nx, Ny)
    """
    p_t_given_x = np.zeros(shape, float)
    for i, j in enumerate(c):
        p_t_given_x[i, j] = 1
    return p_t_given_x
