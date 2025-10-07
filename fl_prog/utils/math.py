import numpy as np
from typing import Iterable


def multivariate_logistic(
    x: Iterable[float],
    k_values: Iterable[float],
    x0_values: Iterable[float],
) -> np.ndarray:
    """
    Calculate D-dimensional vectors of values between 0 and 1 using logistic functions.

    Parameters
    ----------
    x : Iterable[float]
        1D array-like of timepoints, shape (N,).
    k_values : Iterable[float]
        1D array-like of steepness parameter for each dimension, shape (D,).
    x0_values : Iterable[float]
        1D array-like of midpoint parameter for each dimension, shape (D,).

    Returns
    -------
    np.ndarray of shape (N, D)

    """
    x = np.asarray(x)
    k_values = np.asarray(k_values)
    x0_values = np.asarray(x0_values)

    # reshape x for broadcasting if needed
    if x.ndim == 1:
        x = x[:, np.newaxis]

    if len(k_values) != len(x0_values):
        raise ValueError(
            "k_values and t0_values must have the same length, got "
            f"{len(k_values)} and {len(x0_values)} respectively"
        )

    return 1 / (1 + np.exp(-k_values * (x - x0_values)))
