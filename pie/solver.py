from typing import Tuple

import numpy as np


class Solver(object):
  """Numpy-based Jacobi method solver implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.N = 0

  def partition(self, mask: np.ndarray) -> np.ndarray:
    return np.cumsum((mask > 0).reshape(-1)).reshape(mask.shape)

  def reset(self, N: int, A: np.ndarray, X: np.ndarray, B: np.ndarray) -> None:
    """(4 - A)X = B"""
    self.N = N
    self.A = A
    self.B = B
    self.X = X

  def single_step(self) -> None:
    """X = (B + AX) / 4"""
    X = (
      self.B + self.X[self.A[:, 0]] + self.X[self.A[:, 1]] +
      self.X[self.A[:, 2]] + self.X[self.A[:, 3]]
    ) / 4.0
    X[0] = 0
    self.err = np.abs(self.X - X).sum(axis=0)
    self.X = X

  def step(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(iteration):
      self.single_step()
    return self.X, self.err
