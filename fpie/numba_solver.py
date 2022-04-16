from typing import Tuple

import numpy as np
from numba import njit


@njit(fastmath=True)
def equ_iter(N: int, A: np.ndarray, B: np.ndarray, X: np.ndarray) -> np.ndarray:
  return (B + X[A[:, 0]] + X[A[:, 1]] + X[A[:, 2]] + X[A[:, 3]]) / 4.0


@njit(fastmath=True)
def grid_iter(grad: np.ndarray, tgt: np.ndarray) -> np.ndarray:
  result = grad.copy()
  result[1:] += tgt[:-1]
  result[:-1] += tgt[1:]
  result[:, 1:] += tgt[:, :-1]
  result[:, :-1] += tgt[:, 1:]
  return result


class EquSolver(object):
  """Numba-based Jacobi method equation solver implementation."""

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
    self.X = equ_iter(N, A, B, X)

  def sync(self) -> None:
    pass

  def step(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(iteration):
      # X = (B + AX) / 4
      self.X = equ_iter(self.N, self.A, self.B, self.X)
    tmp = self.B + self.X[self.A[:, 0]] + self.X[self.A[:, 1]] + \
      self.X[self.A[:, 2]] + self.X[self.A[:, 3]] - 4.0 * self.X
    err = np.abs(tmp).sum(axis=0)
    x = self.X.copy()
    x[x < 0] = 0
    x[x > 255] = 255
    return x, err


class GridSolver(object):
  """Numba-based Jacobi method grid solver implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.N = 0

  def reset(
    self, N: int, mask: np.ndarray, tgt: np.ndarray, grad: np.ndarray
  ) -> None:
    self.N = N
    self.mask = mask
    self.bool_mask = mask.astype(bool)
    tmp = grid_iter(grad, tgt)
    tgt[self.bool_mask] = tmp[self.bool_mask] / 4.0
    self.tgt = tgt
    self.grad = grad

  def sync(self) -> None:
    pass

  def step(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(iteration):
      tgt = grid_iter(self.grad, self.tgt)
      self.tgt[self.bool_mask] = tgt[self.bool_mask] / 4.0

    tmp = 4 * self.tgt - self.grad
    tmp[1:] -= self.tgt[:-1]
    tmp[:-1] -= self.tgt[1:]
    tmp[:, 1:] -= self.tgt[:, :-1]
    tmp[:, :-1] -= self.tgt[:, 1:]

    err = np.abs(tmp[self.bool_mask]).sum(axis=0)

    tgt = self.tgt.copy()
    tgt[tgt < 0] = 0
    tgt[tgt > 255] = 255
    return tgt, err
