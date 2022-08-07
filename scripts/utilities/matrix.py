import numpy as np
from numpy import ndarray

from functools import cache


def skew(vec: ndarray) -> ndarray:
    """Get the skew matrix from a vector
    Args:
        vec (ndarray[3]): vector
    Returns:
        S (ndarray[3,3]): skew matrix
    """
    S = np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

    return S


@cache
def block_3x3(i: int, j: int):
    """Generate 3x3 slices, useful for setting blocks in larger matrices
    ...
    Args:
        i (int): row
        j (int): column
    Returns:
        slice: The 3x3 slice at the specified location
    """
    return slice(i * 3, (i + 1) * 3), slice(j * 3, (j + 1) * 3)
