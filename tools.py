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
