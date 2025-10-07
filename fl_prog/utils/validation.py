from functools import wraps

import numpy as np


def check_rng(func):
    @wraps(func)
    def _check_rng(*args, **kwargs):
        kwargs["rng"] = kwargs.get("rng", None)
        if kwargs["rng"] is None:
            kwargs["rng"] = np.random.default_rng()
        return func(*args, **kwargs)

    return _check_rng
