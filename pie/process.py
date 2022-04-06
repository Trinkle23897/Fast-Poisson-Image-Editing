from typing import Tuple

import numpy as np

from pie.solver import Solver

try:
  import pie_core_openmp
except:
  pie_core_openmp = None


class Processor(object):
  """PIE Processor"""

  def __init__(self, backend: str):
    super().__init__()
    self.backend = backend
    self.core: Optional[Any] = None
    if backend == "numpy":
      self.core = Solver()
    elif backend == "openmp":
      self.core = pie_core_openmp.Solver()
    assert self.core is not None, f"Backend {backend} is invalid."

  def mask2index(
    self, mask: np.ndarray
  ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    x, y = np.nonzero(mask)
    max_id = x.shape[0] + 1
    index = np.zeros((max_id, 3))
    ids = self.core.partition(mask)
    ids[mask == 0] = 0  # reserve id=0 for constant
    index = ids[x, y].argsort()
    return ids, max_id, x[index], y[index]

  def reset(
    self,
    src: np.ndarray,
    mask: np.ndarray,
    tgt: np.ndarray,
    mask_on_src: Tuple[int, int],
    mask_on_tgt: Tuple[int, int],
  ) -> int:
    # check validity
    # assert 0 <= mask_on_src[0] and 0 <= mask_on_src[1]
    # assert mask_on_src[0] + mask.shape[0] <= src.shape[0]
    # assert mask_on_src[1] + mask.shape[1] <= src.shape[1]
    # assert mask_on_tgt[0] + mask.shape[0] <= tgt.shape[0]
    # assert mask_on_tgt[1] + mask.shape[1] <= tgt.shape[1]

    if len(mask.shape) == 3:
      mask = mask.mean(-1)
    mask = (mask >= 128).astype(np.int32)

    # zero-out edge
    mask[0] = 0
    mask[-1] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0

    src_grad = src * 4.0
    src_grad[:-1] -= src[1:]
    src_grad[1:] -= src[:-1]
    src_grad[:, :-1] -= src[:, 1:]
    src_grad[:, 1:] -= src[:, :-1]

    ids, max_id, index_x, index_y = self.mask2index(mask)
    A = np.zeros((max_id, 4), np.int32)
    X = np.zeros((max_id, 3), np.float32)
    B = np.zeros((max_id, 3), np.float32)

    X[1:] = tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1]]
    # four-way
    A[1:, 0] = ids[index_x - 1, index_y]
    A[1:, 1] = ids[index_x + 1, index_y]
    A[1:, 2] = ids[index_x, index_y - 1]
    A[1:, 3] = ids[index_x, index_y + 1]
    B[1:] = src_grad[index_x + mask_on_src[0], index_y + mask_on_src[1]]
    m = (mask[index_x - 1, index_y] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0] - 1, index_y + mask_on_tgt[1]]
    m = (mask[index_x + 1, index_y] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0] + 1, index_y + mask_on_tgt[1]]
    m = (mask[index_x, index_y - 1] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1] - 1]
    m = (mask[index_x, index_y + 1] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1] + 1]

    self.tgt = tgt.copy()
    self.tgt_index = (index_x + mask_on_tgt[0], index_y + mask_on_tgt[1])
    self.core.reset(max_id, A, X, B)
    return max_id

  def step(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
    x, err = self.core.step(iteration)
    x[x < 0] = 0
    x[x > 255] = 255
    self.tgt[self.tgt_index] = x[1:]
    return self.tgt, err
