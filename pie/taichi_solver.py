from typing import Tuple

import numpy as np
import taichi as ti


@ti.data_oriented
class EquSolver(object):
  """Taichi-based Jacobi method equation solver implementation."""

  def __init__(self, backend: str, n_cpu: int, block_size: int) -> None:
    super().__init__()
    self.parallelize = n_cpu
    self.block_dim = block_size
    ti.init(arch=getattr(ti, backend.split("-")[-1]))
    self.N = 0
    self.tA = ti.field(ti.i32)
    self.tB = ti.field(ti.f32)
    self.tX = ti.field(ti.f32)

  def partition(self, mask: np.ndarray) -> np.ndarray:
    return np.cumsum((mask > 0).reshape(-1)).reshape(mask.shape)

  def reset(self, N: int, A: np.ndarray, X: np.ndarray, B: np.ndarray) -> None:
    """(4 - A)X = B"""
    self.N = N
    self.A = A
    self.B = B
    self.X = X
    ti.root.dense(ti.ij, A.shape).place(self.tA)
    ti.root.dense(ti.ij, B.shape).place(self.tB)
    ti.root.dense(ti.ij, X.shape).place(self.tX)
    self.tA.from_numpy(A)
    self.tB.from_numpy(B)
    self.tX.from_numpy(X)

  def sync(self) -> None:
    pass

  @ti.kernel
  def iter_kernel(self) -> int:
    ti.loop_config(parallelize=self.parallelize, block_dim=self.block_dim)
    for i in ti.static(range(1, self.tX.shape[0])):
      # X = (B + AX) / 4
      self.tX[i] = (self.tB[i] + self.tX[self.A[i]].sum(axis=0)) / 4.0
    return 0

  def step(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(iteration):
      self.iter_kernel()
    self.X = self.tX.to_numpy()
    tmp = 4.0 * self.X - self.B - (
      self.X[self.A[:, 0]] + self.X[self.A[:, 1]] + self.X[self.A[:, 2]] +
      self.X[self.A[:, 3]]
    )
    err = np.abs(tmp).sum(axis=0)
    x = self.X.copy()
    x[x < 0] = 0
    x[x > 255] = 255
    return x, err


class GridSolver(object):
  """Taichi-based Jacobi method grid solver implementation."""

  def __init__(
    self, grid_x: int, grid_y: int, backend: str, n_cpu: int, block_size: int
  ) -> None:
    super().__init__()
    self.N = 0

  def reset(
    self, N: int, mask: np.ndarray, tgt: np.ndarray, grad: np.ndarray
  ) -> None:
    """(4 - A)X = B"""
    self.N = N
    self.mask = mask
    self.bool_mask = mask.astype(bool)
    self.tgt = tgt
    self.grad = grad

  def sync(self) -> None:
    pass

  def step(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(iteration):
      # X = (grad + AX) / 4
      tgt = self.grad.copy()
      tgt[1:] += self.tgt[:-1]
      tgt[:-1] += self.tgt[1:]
      tgt[:, 1:] += self.tgt[:, :-1]
      tgt[:, :-1] += self.tgt[:, 1:]
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
