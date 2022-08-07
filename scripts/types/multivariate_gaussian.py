import numpy as np
from numpy import linalg as nla, ndarray
from dataclasses import dataclass
from functools import cached_property

from config import DEBUG


def isPSD(arr: np.ndarray) -> bool:
    """if this fail you can try the more robust update step:
    Before:
        P_upd =(np.eye(*P.shape) - W @ H) @ P
    After:
        I_WH = np.eye(*P.shape) - W @ H
        P_upd = (I_WH @ P @ I_WH.T+ W @ R @ W.T)
    """
    return np.allclose(arr, arr.T, atol=1e-6) and np.all(np.linalg.eigvals(arr) >= 0)


@dataclass
class MultiVariateGaussian:
    mean: "ndarray[:]"
    cov: "ndarray[:,:]"
    ts: float

    @cached_property
    def ndim(self) -> int:
        return self.mean.shape[0]

    @cached_property
    def scaling(self) -> float:
        scaling = (2 * np.pi) ** (-self.ndim / 2) * nla.det(self.cov) ** (-1 / 2)
        return scaling

    def mahalanobis_distance_sq(self, x: np.ndarray) -> float:
        """Calculate the mahalanobis distance between self and x."""
        # this method could be vectorized for efficient calls
        error = x - self.mean
        mahalanobis_distance = error.T @ nla.solve(self.cov, error)
        return mahalanobis_distance

    def pdf(self, x):
        density = self.scaling * np.exp(-self.mahalanobis_distance_sq(x) / 2)
        return density

    def marginalize(self, idxs):
        return MultiVariateGaussian(
            self.mean[idxs], self.cov[np.ix_(idxs, idxs)], self.ts
        )

    def __iter__(self):  # in order to use tuple unpacking
        return iter((self.mean, self.cov))
