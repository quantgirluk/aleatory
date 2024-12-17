from numbers import Number

import numpy as np
from scipy.stats import ncx2
from aleatory.stats import ncx
import math


def get_times(end, n):
    """Generate a linspace from 0 to end for n increments."""
    return np.linspace(0, end, n)


def check_positive_integer(n, name=""):
    """Ensure that the number is a positive integer."""
    if not isinstance(n, int):
        raise TypeError(f"{name} must be an integer.")
    if n <= 0:
        raise ValueError(f"{name} must be positive.")


def check_numeric(value, name=""):
    """Ensure that the value is numeric."""
    if not isinstance(value, Number):
        raise TypeError(f"{name} value must be a number.")


def check_positive_number(value, name=""):
    """Ensure that the value is a positive number."""
    check_numeric(value, name)
    if value <= 0:
        raise ValueError(f"{name} value must be positive.")


def check_increments(times):
    increments = np.diff(times)
    if np.any([t < 0 for t in times]):
        raise ValueError("Times must be non-negative.")
    if np.any([t <= 0 for t in increments]):
        raise ValueError("Times must be strictly increasing.")
    return increments


def times_to_increments(times):
    """Ensure a positive, monotonically increasing sequence."""
    return check_increments(times)


def sample_besselq_global(T, initial, dim, n):
    t_size = T / n
    path = np.zeros(n)
    path[0] = initial
    x = initial
    for t in range(n - 1):
        sample = ncx2(df=dim, nc=x / t_size, scale=t_size).rvs(1)[0]
        path[t + 1] = sample
        x = sample

    return path


def sample_bessel_global(T, initial, dim, n):
    t_size = T / n
    tss = math.sqrt(t_size)
    path = np.zeros(n)
    path[0] = initial
    x = initial
    for t in range(n - 1):
        sample = ncx(df=dim, nc=x / tss, scale=tss).rvs(1)[0]
        path[t + 1] = sample
        x = sample

    return path
