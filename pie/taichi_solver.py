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
    self.terr = ti.field(ti.f32)
    self.tmp = ti.field(ti.f32)
    ti.root.dense(ti.i, 3).place(self.terr)

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
    ti.root.dense(ti.ij, X.shape).place(self.tmp)
    self.tA.from_numpy(A)
    self.tB.from_numpy(B)
    self.tX.from_numpy(X)
    self.tmp.from_numpy(X)

  def sync(self) -> None:
    pass

  @ti.kernel
  def iter_kernel(self) -> int:
    ti.loop_config(parallelize=self.parallelize, block_dim=self.block_dim)
    for i in range(1, self.N):
      # X = (B + AX) / 4
      i0, i1 = self.tA[i, 0], self.tA[i, 1]
      i2, i3 = self.tA[i, 2], self.tA[i, 3]
      self.tX[i, 0] = (
        self.tB[i, 0] + self.tX[i0, 0] + self.tX[i1, 0] + self.tX[i2, 0] +
        self.tX[i3, 0]
      ) / 4.0
      self.tX[i, 1] = (
        self.tB[i, 1] + self.tX[i0, 1] + self.tX[i1, 1] + self.tX[i2, 1] +
        self.tX[i3, 1]
      ) / 4.0
      self.tX[i, 2] = (
        self.tB[i, 2] + self.tX[i0, 2] + self.tX[i1, 2] + self.tX[i2, 2] +
        self.tX[i3, 2]
      ) / 4.0
    return 0

  def step(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(iteration):
      self.iter_kernel()
    x = self.tX.to_numpy()
    err = np.abs(
      self.B + x[self.A[:, 0]] + x[self.A[:, 1]] + x[self.A[:, 2]] +
      x[self.A[:, 3]] - x * 4.0
    ).sum(axis=0)
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
    self.grid_x = grid_x
    self.grid_y = grid_y
    self.parallelize = n_cpu
    self.block_dim = block_size
    ti.init(arch=getattr(ti, backend.split("-")[-1]))
    self.tmask = ti.field(ti.i32)
    self.ttgt = ti.field(ti.f32)
    self.tgrad = ti.field(ti.f32)

  def reset(
    self, N: int, mask: np.ndarray, tgt: np.ndarray, grad: np.ndarray
  ) -> None:
    """(4 - A)X = B"""
    self.N = N
    self.mask = mask
    self.bool_mask = mask.astype(bool)
    self.tgt = tgt
    self.grad = grad
    # ti.root.dense(ti.ij, A.shape).place(self.tA)
    # ti.root.dense(ti.ij, B.shape).place(self.tB)
    # ti.root.dense(ti.ij, X.shape).place(self.tX)
    # ti.root.dense(ti.ij, X.shape).place(self.tmp)

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
