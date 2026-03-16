"""
Welford's online algorithm for running mean and standard deviation.

Replaces RunningStat from learning_from_human_preferences.envs.utils.
"""

import numpy as np


class RunningStat:
    """
    Track running mean and std via Welford's online algorithm.

    Args:
        shape: int or tuple — shape of each sample (e.g. n_preds)
    """

    def __init__(self, shape=()):
        if isinstance(shape, int):
            shape = (shape,)
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x: np.ndarray) -> None:
        """Update statistics with one new sample."""
        x = np.asarray(x)
        assert x.shape == self._M.shape, (
            f"Shape mismatch: expected {self._M.shape}, got {x.shape}"
        )

        self._n += 1

        if self._n == 1:
            self._M[...] = x
            return

        oldM = self._M.copy()
        self._M[...] = oldM + (x - oldM) / self._n
        self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> np.ndarray:
        return self._M

    @property
    def var(self) -> np.ndarray:
        if self._n >= 2:
            return self._S / (self._n - 1)
        return np.square(self._M)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape
